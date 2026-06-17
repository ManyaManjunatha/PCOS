from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
from PIL import Image

try:
    from torchvision import models, transforms
except ImportError as exc:  # pragma: no cover - depends on local ML environment.
    models = None
    transforms = None
    TORCHVISION_IMPORT_ERROR = exc
else:
    TORCHVISION_IMPORT_ERROR = None


CLASS_NAMES = ("Cyst", "Papules", "Pustules", "normal_skin")
SEVERITY_WEIGHTS = {
    "cyst": 1.0,
    "pustules": 0.8,
    "papules": 0.6,
    "normal_skin": 0.0,
}


@dataclass(frozen=True)
class AcnePrediction:
    acne_class: str
    acne_confidence: float
    acne_severity: str
    acne_severity_score: float
    class_probabilities: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "acne_class": self.acne_class,
            "acne_confidence": self.acne_confidence,
            "acne_severity": self.acne_severity,
            "acne_severity_score": self.acne_severity_score,
            "class_probabilities": self.class_probabilities,
        }


class AcneModel:
    """ConvNeXt-Tiny acne classifier with normalized acne severity output."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.class_names = CLASS_NAMES
        self.model: Optional[nn.Module] = None
        self.model_path: Optional[str] = None
        if transforms is None:
            self.transform = None
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        if model_path is not None:
            self.load(model_path)

    def _create_model(self) -> nn.Module:
        if models is None:
            raise ImportError(
                "torchvision is required for AcneModel. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from TORCHVISION_IMPORT_ERROR
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.features[6:].parameters():
            parameter.requires_grad = True
        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Dropout(p=0.4),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, len(self.class_names)),
        )
        return model.to(self.device)

    def load(self, path: str) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Acne model checkpoint not found: {checkpoint_path}")

        model = self._create_model()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        self.model_path = str(checkpoint_path)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save acne model before loading or training it.")
        torch.save(self.model.state_dict(), path)
        self.model_path = path

    def predict(self, image_path: str) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Acne model is not loaded. Provide acne_model_path or call load().")
        if self.transform is None:
            raise ImportError(
                "torchvision is required for image preprocessing. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from TORCHVISION_IMPORT_ERROR

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu()

        class_probabilities = {
            class_name: float(probabilities[index].item())
            for index, class_name in enumerate(self.class_names)
        }
        predicted_index = int(probabilities.argmax().item())
        acne_class = self.class_names[predicted_index]
        acne_confidence = class_probabilities[acne_class]
        acne_severity_score = self._compute_normalized_severity(class_probabilities)
        acne_severity = self._severity_label(acne_severity_score)

        return AcnePrediction(
            acne_class=acne_class,
            acne_confidence=acne_confidence,
            acne_severity=acne_severity,
            acne_severity_score=acne_severity_score,
            class_probabilities=class_probabilities,
        ).to_dict()

    def _compute_normalized_severity(self, class_probabilities: Mapping[str, float]) -> float:
        score = 0.0
        for class_name, probability in class_probabilities.items():
            score += probability * SEVERITY_WEIGHTS[class_name.lower()]
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _severity_label(score: float) -> str:
        if score < 0.15:
            return "None/minimal"
        if score < 0.45:
            return "Mild"
        if score < 0.75:
            return "Moderate"
        return "Severe"

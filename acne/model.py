import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Dict, Any, List, Optional
from collections import Counter


SEVERITY_WEIGHTS = {
    'papules': 0.6,
    'pustules': 0.8,
    'cyst': 1.0,
}

CLASS_NAMES = ['Cyst', 'Papules', 'Pustules', 'normal_skin']


class AcneModel(nn.Module):
    """Acne classification model using ConvNeXt-Tiny."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        super().__init__()
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model = None
        self.model_path = model_path
        self.class_names = CLASS_NAMES
        self.num_classes = len(CLASS_NAMES)

        if model_path:
            self.load(model_path)

    def _create_model(self) -> nn.Module:
        from torchvision import models

        model = models.convnext_tiny(weights='IMAGENET1K_V1')

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze later layers
        for param in model.features[6:].parameters():
            param.requires_grad = True

        # Replace classifier head
        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Dropout(p=0.4),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.num_classes)
        )

        return model.to(self.device)

    def load(self, path: str) -> None:
        self.model = self._create_model()
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        self.model_path = path

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("No model to save. Create or load first.")
        torch.save(self.model.state_dict(), path)
        self.model_path = path

    def _compute_severity_score(self, probs: torch.Tensor) -> tuple:
        acne_score = 0.0
        acne_sum = 0.0

        for i, cls in enumerate(self.class_names):
            key = cls.lower().replace(' ', '_')
            if key == 'normal_skin':
                continue

            p = probs[i].item()
            weight = SEVERITY_WEIGHTS.get(key, 0.5)
            acne_score += p * weight
            acne_sum += p

        if acne_sum > 0:
            score = acne_score / acne_sum
        else:
            score = 0.0

        return score, acne_sum

    def _get_severity_label(self, score: float, acne_sum: float, max_acne_prob: float) -> tuple:
        if acne_sum < 0.03 and max_acne_prob < 0.03:
            return 0, "No acne"

        if score < 0.5:
            return 1, "Mild acne"
        elif score < 0.8:
            return 2, "Moderate acne"
        else:
            return 3, "Severe acne"

    def predict(self, image_path: str) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        from .transforms import get_val_transforms

        img = Image.open(image_path).convert('RGB')
        transform = get_val_transforms()
        tensor = transform(img).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0].cpu()

        pred_idx = probs.argmax().item()
        pred_class = self.class_names[pred_idx]
        confidence = probs[pred_idx].item()

        # Compute severity
        score, acne_sum = self._compute_severity_score(probs)

        max_acne_prob = max([
            probs[i].item()
            for i, cls in enumerate(self.class_names)
            if cls.lower() != "normal_skin"
        ])

        severity_code, severity_label = self._get_severity_label(score, acne_sum, max_acne_prob)

        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'all_probs': dict(zip(self.class_names, probs.tolist())),
            'severity_score': score,
            'acne_probability_sum': acne_sum,
            'severity_code': severity_code,
            'severity_label': severity_label
        }

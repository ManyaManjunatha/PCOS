import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .pcos import PCOSModel
from .acne import AcneModel


class Diagnose:
    """Unified diagnostic system for PCOS and Acne detection."""

    def __init__(
        self,
        pcos_model_path: Optional[str] = None,
        acne_model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the unified diagnostic system.

        Args:
            pcos_model_path: Path to PCOS model (.pkl file). Required for predict_pcos().
            acne_model_path: Path to Acne model (.pth file). Required for predict_acne().
            device: Device for acne model inference ('cuda', 'cpu', or None for auto).
        """
        self._pcos_model = None
        self._acne_model = None
        self._pcos_path = pcos_model_path
        self._acne_path = acne_model_path
        self._device = device

        if pcos_model_path:
            self._pcos_model = PCOSModel(pcos_model_path)

        if acne_model_path:
            self._acne_model = AcneModel(acne_model_path, device=device)

    def load_pcos_model(self, path: str) -> None:
        """Load PCOS model from path."""
        if self._pcos_model is None:
            self._pcos_model = PCOSModel()
        self._pcos_model.load(path)
        self._pcos_path = path

    def load_acne_model(self, path: str) -> None:
        """Load Acne model from path."""
        if self._acne_model is None:
            self._acne_model = AcneModel(device=self._device)
        self._acne_model.load(path)
        self._acne_path = path

    def predict_pcos(
        self,
        data: Union[str, pd.DataFrame],
        return_proba: bool = False
    ) -> Dict[str, Any]:
        """
        Predict PCOS from patient data.

        Args:
            data: CSV file path or pandas DataFrame with patient data.
            return_proba: If True, return prediction probabilities.

        Returns:
            Dictionary with 'prediction', 'probability' (if requested), and 'input_data'.
        """
        if self._pcos_model is None:
            raise ValueError(
                "PCOS model not loaded. Provide pcos_model_path in __init__ "
                "or call load_pcos_model()."
            )

        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        prediction = self._pcos_model.predict(df)[0]

        result = {
            'pcos_detected': bool(prediction),
            'pcos_label': 'YES' if prediction == 1 else 'NO',
            'input_data': df
        }

        if return_proba:
            proba = self._pcos_model.predict_proba(df)[0]
            result['probability'] = {
                'NO': float(proba[0]),
                'YES': float(proba[1])
            }

        return result

    def predict_acne(self, image_path: str) -> Dict[str, Any]:
        """
        Predict acne type and severity from image.

        Args:
            image_path: Path to skin image.

        Returns:
            Dictionary with predicted_class, confidence, severity_label, and all_probs.
        """
        if self._acne_model is None:
            raise ValueError(
                "Acne model not loaded. Provide acne_model_path in __init__ "
                "or call load_acne_model()."
            )

        return self._acne_model.predict(image_path)

    def diagnose(self, patient_data: Union[str, pd.DataFrame], image_path: str) -> Dict[str, Any]:
        """
        Run both PCOS and Acne diagnostics.

        Args:
            patient_data: CSV file path or DataFrame for PCOS prediction.
            image_path: Path to skin image for acne prediction.

        Returns:
            Dictionary with both 'pcos' and 'acne' results.
        """
        pcos_result = self.predict_pcos(patient_data, return_proba=True)
        acne_result = self.predict_acne(image_path)

        return {
            'pcos': pcos_result,
            'acne': acne_result
        }

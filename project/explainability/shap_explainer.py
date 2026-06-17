from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


class ShapExplainer:
    """Per-prediction SHAP explanations for the XGBoost fusion model."""

    def __init__(self, model: Any, feature_columns: List[str]) -> None:
        self.model = model
        self.feature_columns = feature_columns
        self._explainer = None

    def explain(self, features: pd.DataFrame) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("A trained fusion model is required for SHAP explanations.")
        if features.empty:
            raise ValueError("features must contain at least one row.")
        missing = [column for column in self.feature_columns if column not in features.columns]
        if missing:
            raise ValueError(f"Missing SHAP feature columns: {missing}")

        values = self._shap_values(features[self.feature_columns])
        first_row = np.asarray(values[0], dtype=float)
        contributions = {
            feature: float(contribution)
            for feature, contribution in zip(self.feature_columns, first_row)
        }
        return dict(
            sorted(
                contributions.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        )

    def _shap_values(self, features: pd.DataFrame) -> np.ndarray:
        import shap

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        values = self._explainer.shap_values(features)
        if isinstance(values, list):
            values = values[-1]
        return np.asarray(values)

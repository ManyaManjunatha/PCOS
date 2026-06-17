from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class ClinicalShapExplainer:
    """SHAP explainability for the best clinical PCOS model pipeline."""

    def __init__(self, pipeline: Pipeline, max_background_rows: int = 250) -> None:
        self.pipeline = pipeline
        self.max_background_rows = max_background_rows

    def compute(self, x_reference: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
        import shap

        preprocessor = self.pipeline.named_steps["preprocessor"]
        estimator = self.pipeline.named_steps["model"]

        sample = x_reference.sample(
            n=min(len(x_reference), self.max_background_rows),
            random_state=42,
        )
        transformed_array = preprocessor.transform(sample)
        feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
        transformed = pd.DataFrame(transformed_array, columns=feature_names)

        if estimator.__class__.__name__ in {"RandomForestClassifier", "XGBClassifier"}:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
        else:
            background = shap.sample(transformed, min(len(transformed), 100), random_state=42)
            explainer = shap.LinearExplainer(estimator, background)
            shap_values = explainer.shap_values(transformed)

        return np.asarray(shap_values), transformed, feature_names

    @staticmethod
    def feature_importance(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, -1]
        mean_abs = np.abs(values).mean(axis=0)
        return pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    @staticmethod
    def plot_summary(
        shap_values: np.ndarray,
        transformed_features: pd.DataFrame,
        feature_names: List[str],
        output_path: Path,
    ) -> None:
        import matplotlib.pyplot as plt
        import shap

        output_path.parent.mkdir(parents=True, exist_ok=True)
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, -1]
        shap.summary_plot(
            values,
            transformed_features[feature_names],
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

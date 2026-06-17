from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from project.preprocessing.feature_engineering import FeatureEngineer, MODEL_FEATURE_COLUMNS

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - exercised only in incomplete environments.
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


RISK_THRESHOLDS = {
    "low_max": 30,
    "moderate_max": 70,
}


class FusionModel:
    """XGBoost fusion model for PCOS risk screening."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model: Optional[Any] = None
        self.feature_engineer = FeatureEngineer()
        self.feature_columns = MODEL_FEATURE_COLUMNS
        self.model_path: Optional[str] = None
        if model_path is not None:
            self.load(model_path)

    def _create_classifier(self, **overrides: Any) -> Any:
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is required for FusionModel. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from XGBOOST_IMPORT_ERROR

        params = {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": 1,
        }
        params.update(overrides)
        return XGBClassifier(**params)

    def fit(self, records: Iterable[Mapping[str, Any]], labels: Iterable[int], **model_params: Any) -> Dict[str, Any]:
        x = self.feature_engineer.transform_records(records)
        y = pd.Series(labels, dtype=int)
        if len(x) != len(y):
            raise ValueError("records and labels must have the same length.")
        if y.nunique() != 2:
            raise ValueError("Fusion model training requires both negative and positive PCOS labels.")

        self.model = self._create_classifier(**model_params)
        self.model.fit(x, y)
        return {"n_samples": int(len(x)), "n_features": int(x.shape[1]), "feature_columns": self.feature_columns}

    def fit_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str = "pcos_label",
        **model_params: Any,
    ) -> Dict[str, Any]:
        if target_column not in df.columns:
            raise ValueError(f"Target column `{target_column}` was not found.")
        records = df.drop(columns=[target_column]).to_dict(orient="records")
        labels = df[target_column].map(self._coerce_label)
        return self.fit(records, labels, **model_params)

    def predict_patient(
        self,
        *,
        acne_severity_score: float,
        age: float,
        bmi: float,
        ethnicity: str,
        menstrual_irregularity: Any,
        hirsutism_score: float,
        hair_loss_score: float,
    ) -> Dict[str, Any]:
        features = self.feature_engineer.transform_patient(
            acne_severity_score=acne_severity_score,
            age=age,
            bmi=bmi,
            ethnicity=ethnicity,
            menstrual_irregularity=menstrual_irregularity,
            hirsutism_score=hirsutism_score,
            hair_loss_score=hair_loss_score,
        )
        probability = self.predict_probability(features)[0]
        risk_score = self.probability_to_score(probability)
        return {
            "pcos_probability": probability,
            "pcos_risk_score": risk_score,
            "risk_category": self.risk_category(risk_score),
            "model_features": features.iloc[0].to_dict(),
        }

    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Fusion model is not loaded or trained.")
        self._validate_feature_frame(features)
        probabilities = self.model.predict_proba(features[self.feature_columns])
        return probabilities[:, 1].astype(float)

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str = "pcos_label",
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Fusion model is not loaded or trained.")
        if target_column not in df.columns:
            raise ValueError(f"Target column `{target_column}` was not found.")

        labels = df[target_column].map(self._coerce_label).astype(int)
        features = self.feature_engineer.transform_records(df.drop(columns=[target_column]).to_dict(orient="records"))
        probabilities = self.predict_probability(features)
        predictions = (probabilities >= 0.5).astype(int)
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        matrix = confusion_matrix(labels, predictions)

        metrics = {
            "confusion_matrix": matrix.tolist(),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1_score": float(f1_score(labels, predictions, zero_division=0)),
            "auc": float(auc(fpr, tpr)),
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            },
        }
        if output_dir is not None:
            self._write_evaluation_plots(matrix, fpr, tpr, metrics["auc"], Path(output_dir))
        return metrics

    def train_test_evaluate(
        self,
        df: pd.DataFrame,
        target_column: str = "pcos_label",
        test_size: float = 0.2,
        random_state: int = 42,
        output_dir: Optional[str] = None,
        **model_params: Any,
    ) -> Dict[str, Any]:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_column].map(self._coerce_label),
        )
        training_summary = self.fit_dataframe(train_df, target_column=target_column, **model_params)
        evaluation = self.evaluate_dataframe(test_df, target_column=target_column, output_dir=output_dir)
        return {"training": training_summary, "evaluation": evaluation}

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save fusion model before training it.")
        artifact = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "risk_thresholds": RISK_THRESHOLDS,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
        self.model_path = path

    def load(self, path: str) -> None:
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Fusion model artifact not found: {artifact_path}")
        artifact = joblib.load(artifact_path)
        if isinstance(artifact, dict) and "model" in artifact:
            self.model = artifact["model"]
            self.feature_columns = artifact.get("feature_columns", MODEL_FEATURE_COLUMNS)
        else:
            self.model = artifact
            self.feature_columns = MODEL_FEATURE_COLUMNS
        self.model_path = str(artifact_path)

    @staticmethod
    def probability_to_score(probability: float) -> int:
        return int(round(max(0.0, min(1.0, float(probability))) * 100))

    @staticmethod
    def risk_category(risk_score: int) -> str:
        if risk_score <= RISK_THRESHOLDS["low_max"]:
            return "Low"
        if risk_score <= RISK_THRESHOLDS["moderate_max"]:
            return "Moderate"
        return "High"

    @staticmethod
    def recommendation(risk_category: str) -> str:
        recommendations = {
            "Low": "Low screening risk. Continue routine self-monitoring and seek clinical care if symptoms change.",
            "Moderate": "Moderate screening risk. Consider discussing symptoms with a qualified clinician.",
            "High": "High screening risk. Arrange clinical evaluation for confirmatory assessment and management.",
        }
        return recommendations[risk_category]

    @staticmethod
    def _coerce_label(value: Any) -> int:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"yes", "pcos", "positive", "1", "true"}:
                return 1
            if normalized in {"no", "non_pcos", "negative", "0", "false"}:
                return 0
        return int(value)

    def _validate_feature_frame(self, features: pd.DataFrame) -> None:
        missing = [column for column in self.feature_columns if column not in features.columns]
        if missing:
            raise ValueError(f"Missing engineered model features: {missing}")

    @staticmethod
    def _write_evaluation_plots(
        matrix: np.ndarray,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_value: float,
        output_dir: Path,
    ) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=200)
        plt.close()

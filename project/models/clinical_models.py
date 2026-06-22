from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from project.preprocessing.clinical_preprocessor import ClinicalPCOSPreprocessor

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


@dataclass(frozen=True)
class TrainingResult:
    best_model_name: str
    cross_validation: pd.DataFrame
    holdout_metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    roc_curve: Dict[str, List[float]]
    feature_importance: pd.DataFrame


class ClinicalPCOSModelTrainer:
    """Trains and compares clinical PCOS classifiers with 5-fold CV."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.preprocessor = ClinicalPCOSPreprocessor()

    def build_model_candidates(self, features: pd.DataFrame) -> Mapping[str, Pipeline]:
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is required to train the requested XGBoost model. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from XGBOOST_IMPORT_ERROR

        return OrderedDict(
            {
                "Logistic Regression": Pipeline(
                    [
                        ("preprocessor", self.preprocessor.build_preprocessor(features, scale_numeric=True)),
                        (
                            "model",
                            LogisticRegression(
                                max_iter=2000,
                                class_weight="balanced",
                                solver="lbfgs",
                                random_state=self.random_state,
                            ),
                        ),
                    ]
                ),
                "Random Forest": Pipeline(
                    [
                        ("preprocessor", self.preprocessor.build_preprocessor(features, scale_numeric=False)),
                        (
                            "model",
                            RandomForestClassifier(
                                n_estimators=400,
                                max_depth=None,
                                min_samples_leaf=2,
                                class_weight="balanced",
                                random_state=self.random_state,
                                n_jobs=1,
                            ),
                        ),
                    ]
                ),
                "XGBoost": Pipeline(
                    [
                        ("preprocessor", self.preprocessor.build_preprocessor(features, scale_numeric=False)),
                        (
                            "model",
                            XGBClassifier(
                                n_estimators=300,
                                max_depth=3,
                                learning_rate=0.05,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                objective="binary:logistic",
                                eval_metric="logloss",
                                random_state=self.random_state,
                                n_jobs=1,
                            ),
                        ),
                    ]
                ),
            }
        )

    def train_compare(
        self,
        dataset_path: str,
        *,
        model_output_path: str = "artifacts/pcos_model.joblib",
        figures_dir: str = "outputs/figures",
        reports_dir: str = "outputs/reports",
    ) -> TrainingResult:
        df = self.preprocessor.load_excel(dataset_path)
        bundle = self.preprocessor.split_features_target(df)
        x, y = bundle.features, bundle.target

        candidates = self.build_model_candidates(x)
        cv_results = self.cross_validate_models(candidates, x, y)
        best_model_name = self.select_best_model(cv_results)
        best_pipeline = candidates[best_model_name]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            stratify=y,
            random_state=self.random_state,
        )
        best_pipeline.fit(x_train, y_train)
        holdout_metrics, matrix, roc_data = self.evaluate_holdout(best_pipeline, x_test, y_test)

        figures_path = Path(figures_dir)
        reports_path = Path(reports_dir)
        figures_path.mkdir(parents=True, exist_ok=True)
        reports_path.mkdir(parents=True, exist_ok=True)

        self.plot_confusion_matrix(matrix, figures_path / "confusion_matrix.png")
        self.plot_roc_curve(roc_data, holdout_metrics["roc_auc"], figures_path / "roc_curve.png")

        feature_importance = self.compute_feature_importance(best_pipeline, x_train, figures_path)
        feature_importance.to_csv(reports_path / "feature_importance.csv", index=False)
        cv_results.to_csv(reports_path / "model_comparison_cv.csv", index=False)
        pd.DataFrame([holdout_metrics]).to_csv(reports_path / "holdout_metrics.csv", index=False)

        final_pipeline = candidates[best_model_name]
        final_pipeline.fit(x, y)
        self.save_model(final_pipeline, model_output_path, best_model_name, cv_results)

        return TrainingResult(
            best_model_name=best_model_name,
            cross_validation=cv_results,
            holdout_metrics=holdout_metrics,
            confusion_matrix=matrix.tolist(),
            roc_curve=roc_data,
            feature_importance=feature_importance,
        )

    def cross_validate_models(
        self,
        candidates: Mapping[str, Pipeline],
        x: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        rows = []
        for model_name, pipeline in candidates.items():
            scores = cross_validate(
                pipeline,
                x,
                y,
                scoring=SCORING,
                cv=cv,
                n_jobs=1,
                error_score="raise",
            )
            row = {"model": model_name}
            for metric in SCORING:
                values = scores[f"test_{metric}"]
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values))
            rows.append(row)
        return pd.DataFrame(rows).sort_values(
            by=["roc_auc_mean", "f1_mean", "accuracy_mean"],
            ascending=False,
        ).reset_index(drop=True)

    @staticmethod
    def select_best_model(cv_results: pd.DataFrame) -> str:
        return str(cv_results.iloc[0]["model"])

    @staticmethod
    def evaluate_holdout(
        pipeline: Pipeline,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> tuple[Dict[str, float], np.ndarray, Dict[str, List[float]]]:
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        fpr, tpr, thresholds = roc_curve(y_test, probabilities)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
        return metrics, confusion_matrix(y_test, predictions), {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

    def compute_feature_importance(
        self,
        pipeline: Pipeline,
        x_reference: pd.DataFrame,
        figures_dir: Path,
    ) -> pd.DataFrame:
        from project.explainability.shap_explainer import ClinicalShapExplainer

        explainer = ClinicalShapExplainer(pipeline)
        shap_values, transformed, feature_names = explainer.compute(x_reference)
        explainer.plot_summary(shap_values, transformed, feature_names, figures_dir / "shap_summary.png")
        ranking = explainer.feature_importance(shap_values, feature_names)
        return ranking

    @staticmethod
    def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted PCOS label")
        plt.ylabel("True PCOS label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def plot_roc_curve(roc_data: Dict[str, List[float]], roc_auc: float, output_path: Path) -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 4))
        plt.plot(roc_data["fpr"], roc_data["tpr"], label=f"ROC-AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def save_model(
        pipeline: Pipeline,
        output_path: str,
        best_model_name: str,
        cv_results: pd.DataFrame,
    ) -> None:
        artifact = {
            "model": pipeline,
            "best_model_name": best_model_name,
            "cv_results": cv_results,
            "positive_label": "YES",
            "negative_label": "NO",
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, output_path)


class ClinicalPCOSPredictor:
    """Loads the saved best clinical model for prediction."""

    def __init__(self, model_path: str = "artifacts/pcos_model.joblib") -> None:
        artifact = joblib.load(model_path)
        self.model: Pipeline = artifact["model"]
        self.best_model_name: str = artifact["best_model_name"]
        self.preprocessor = ClinicalPCOSPreprocessor()

    def predict(self, records: pd.DataFrame) -> pd.DataFrame:
        cleaned_records = self.preprocessor.clean_column_names(records)
        probabilities = self.model.predict_proba(cleaned_records)[:, 1]
        labels = (probabilities >= 0.5).astype(int)
        return pd.DataFrame(
            {
                "pcos_probability": probabilities,
                "pcos_prediction": labels,
                "pcos_label": np.where(labels == 1, "YES", "NO"),
            }
        )

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from typing import Dict, Any, Optional, Tuple
from .preprocessor import PCOSPreprocessor


class PCOSModel:
    """PCOS detection model using XGBoost."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.preprocessor = PCOSPreprocessor()

        if model_path:
            self.load(model_path)

    def _create_model(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        self.model_path = path

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("No model to save. Train or load first.")
        joblib.dump(self.model, path)
        self.model_path = path

    def train(self, df: pd.DataFrame, compare_models: bool = False) -> Dict[str, float]:
        df = self.preprocessor.preprocess(df, training=True)

        X = df.drop(columns=["PCOS"])
        y = df["PCOS"]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        if compare_models:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost": self._create_model()
            }

            scores = {}
            for name, m in models.items():
                cv_scores = cross_val_score(m, X, y, cv=cv)
                scores[name] = cv_scores.mean()

            self.model = self._create_model()
            return scores

        self.model = self._create_model()
        return {}

    def fit(self, df: pd.DataFrame) -> None:
        df = self.preprocessor.preprocess(df, training=True)

        X = df.drop(columns=["PCOS"])
        y = df["PCOS"]

        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")

        df = self.preprocessor.preprocess(df, training=False)
        X = df.drop(columns=["PCOS"]) if "PCOS" in df.columns else df
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")

        df = self.preprocessor.preprocess(df, training=False)
        X = df.drop(columns=["PCOS"]) if "PCOS" in df.columns else df
        return self.model.predict_proba(X)

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

from project.models.clinical_models import ClinicalPCOSPredictor


def predict_pcos(
    records: Iterable[Dict[str, object]] | pd.DataFrame,
    model_path: str = "artifacts/pcos_model.joblib",
) -> pd.DataFrame:
    """Predict PCOS risk from clinical and phenotypic records."""

    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    predictor = ClinicalPCOSPredictor(model_path=model_path)
    return predictor.predict(frame)

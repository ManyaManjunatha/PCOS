from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd


CLINICAL_INPUT_COLUMNS = [
    "age",
    "bmi",
    "ethnicity",
    "menstrual_irregularity",
    "hirsutism_score",
    "hair_loss_score",
]

MODEL_FEATURE_COLUMNS = [
    "acne_severity_score",
    "age",
    "bmi",
    "ethnicity_encoded",
    "menstrual_irregularity",
    "hirsutism_score",
    "hair_loss_score",
    "age_group_18_25",
    "age_group_26_30",
    "bmi_underweight",
    "bmi_normal",
    "bmi_overweight",
    "bmi_obese",
]

ETHNICITY_MAP = {
    "south_asian": 1,
    "east_asian": 2,
    "middle_eastern": 3,
    "black": 4,
    "white": 5,
    "hispanic": 6,
    "latina": 6,
    "latino": 6,
    "mixed": 7,
    "other": 0,
    "unknown": 0,
}


@dataclass(frozen=True)
class PatientFeatures:
    acne_severity_score: float
    age: float
    bmi: float
    ethnicity: str
    menstrual_irregularity: Any
    hirsutism_score: float
    hair_loss_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureEngineer:
    """Builds the fusion-model feature matrix from acne and questionnaire inputs."""

    def transform_patient(
        self,
        *,
        acne_severity_score: float,
        age: float,
        bmi: float,
        ethnicity: str,
        menstrual_irregularity: Any,
        hirsutism_score: float,
        hair_loss_score: float,
    ) -> pd.DataFrame:
        patient = PatientFeatures(
            acne_severity_score=acne_severity_score,
            age=age,
            bmi=bmi,
            ethnicity=ethnicity,
            menstrual_irregularity=menstrual_irregularity,
            hirsutism_score=hirsutism_score,
            hair_loss_score=hair_loss_score,
        )
        return self.transform_records([patient.to_dict()])

    def transform_records(self, records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
        raw = pd.DataFrame(records)
        if raw.empty:
            raise ValueError("At least one patient record is required.")

        missing_columns = [
            column
            for column in ["acne_severity_score", *CLINICAL_INPUT_COLUMNS]
            if column not in raw.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")

        engineered = pd.DataFrame(index=raw.index)
        engineered["acne_severity_score"] = raw["acne_severity_score"].map(
            lambda value: self._bounded_float(value, "acne_severity_score", 0.0, 1.0)
        )
        engineered["age"] = raw["age"].map(lambda value: self._bounded_float(value, "age", 12.0, 60.0))
        engineered["bmi"] = raw["bmi"].map(lambda value: self._bounded_float(value, "bmi", 10.0, 80.0))
        engineered["ethnicity_encoded"] = raw["ethnicity"].map(self.encode_ethnicity)
        engineered["menstrual_irregularity"] = raw["menstrual_irregularity"].map(
            self.encode_menstrual_irregularity
        )
        engineered["hirsutism_score"] = raw["hirsutism_score"].map(
            lambda value: self._normalize_score(value, "hirsutism_score", 36.0)
        )
        engineered["hair_loss_score"] = raw["hair_loss_score"].map(
            lambda value: self._normalize_score(value, "hair_loss_score", 3.0)
        )

        engineered["age_group_18_25"] = engineered["age"].map(lambda value: float(18 <= value <= 25))
        engineered["age_group_26_30"] = engineered["age"].map(lambda value: float(26 <= value <= 30))

        bmi_categories = engineered["bmi"].map(self.bmi_category)
        engineered["bmi_underweight"] = (bmi_categories == "Underweight").astype(float)
        engineered["bmi_normal"] = (bmi_categories == "Normal").astype(float)
        engineered["bmi_overweight"] = (bmi_categories == "Overweight").astype(float)
        engineered["bmi_obese"] = (bmi_categories == "Obese").astype(float)

        return engineered[MODEL_FEATURE_COLUMNS].astype(float)

    @staticmethod
    def age_group(age: float) -> str:
        numeric_age = FeatureEngineer._bounded_float(age, "age", 12.0, 60.0)
        if 18 <= numeric_age <= 25:
            return "18-25"
        if 26 <= numeric_age <= 30:
            return "26-30"
        return "outside_target_range"

    @staticmethod
    def bmi_category(bmi: float) -> str:
        numeric_bmi = FeatureEngineer._bounded_float(bmi, "bmi", 10.0, 80.0)
        if numeric_bmi < 18.5:
            return "Underweight"
        if numeric_bmi < 25:
            return "Normal"
        if numeric_bmi < 30:
            return "Overweight"
        return "Obese"

    @staticmethod
    def encode_ethnicity(value: Any) -> int:
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        return ETHNICITY_MAP.get(normalized, ETHNICITY_MAP["other"])

    @staticmethod
    def encode_menstrual_irregularity(value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
            return int(float(value) >= 0.5)
        normalized = str(value).strip().lower()
        yes_values = {"yes", "y", "true", "1", "irregular", "irregular_cycle", "irregular cycles"}
        no_values = {"no", "n", "false", "0", "regular", "regular_cycle", "regular cycles"}
        if normalized in yes_values:
            return 1
        if normalized in no_values:
            return 0
        raise ValueError("menstrual_irregularity must be a boolean, 0/1, yes/no, or regular/irregular value.")

    @staticmethod
    def _normalize_score(value: Any, name: str, max_score: float) -> float:
        numeric = FeatureEngineer._bounded_float(value, name, 0.0, max_score)
        return float(numeric / max_score)

    @staticmethod
    def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
        if value is None or pd.isna(value):
            raise ValueError(f"{name} is required.")
        numeric = float(value)
        if not minimum <= numeric <= maximum:
            raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}.")
        return numeric

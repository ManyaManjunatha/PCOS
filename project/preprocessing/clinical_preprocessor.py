from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "PCOS"
ID_COLUMNS = ["PatientID"]
YES_NO_COLUMNS = ["HypoA", "Cycle", TARGET_COLUMN]
DERIVED_CYCLE_COLUMNS = ["cycle_mean", "cycle_std", "cycle_variance", "cycle_range"]


@dataclass(frozen=True)
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series


class ClinicalPCOSPreprocessor:
    """Prepares clinical and phenotypic PCOS features from the Excel dataset."""

    def load_excel(self, path: str, sheet_name: str | int = 0) -> pd.DataFrame:
        df = pd.read_excel(path, sheet_name=sheet_name)
        return self.clean_column_names(df)

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned.columns = cleaned.columns.astype(str).str.strip()
        return cleaned

    def split_features_target(self, df: pd.DataFrame) -> DatasetBundle:
        cleaned = self.clean_column_names(df)
        if TARGET_COLUMN not in cleaned.columns:
            raise ValueError(f"Required target column `{TARGET_COLUMN}` was not found.")

        y = cleaned[TARGET_COLUMN].map(self._encode_binary_label)
        if y.isna().any():
            bad_values = sorted(cleaned.loc[y.isna(), TARGET_COLUMN].astype(str).unique())
            raise ValueError(f"Unsupported PCOS target values: {bad_values}")

        drop_columns = [column for column in [TARGET_COLUMN, *ID_COLUMNS] if column in cleaned.columns]
        x = cleaned.drop(columns=drop_columns)
        return DatasetBundle(features=x, target=y.astype(int))

    def build_preprocessor(self, features: pd.DataFrame, *, scale_numeric: bool) -> ColumnTransformer:
        numeric_steps: List[Tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        numeric_pipeline = Pipeline(numeric_steps)
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, make_column_selector(dtype_include="number")),
                ("categorical", categorical_pipeline, make_column_selector(dtype_exclude="number")),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def feature_groups(self, features: pd.DataFrame) -> Tuple[List[str], List[str]]:
        categorical_columns = [
            column
            for column in features.columns
            if features[column].dtype == "object" or str(features[column].dtype).startswith("category")
        ]
        numeric_columns = [column for column in features.columns if column not in categorical_columns]
        return categorical_columns, numeric_columns

    def feature_names_after_transform(self, fitted_preprocessor: ColumnTransformer) -> List[str]:
        return [str(name) for name in fitted_preprocessor.get_feature_names_out()]

    @staticmethod
    def _encode_binary_label(value: object) -> int | None:
        if pd.isna(value):
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"yes", "y", "true", "1", "positive", "pcos"}:
                return 1
            if normalized in {"no", "n", "false", "0", "negative", "non-pcos", "non_pcos"}:
                return 0
        if value in {1, True}:
            return 1
        if value in {0, False}:
            return 0
        return None


def required_feature_columns(df: pd.DataFrame) -> Iterable[str]:
    return [
        column
        for column in df.columns
        if column not in {TARGET_COLUMN, *ID_COLUMNS} and column not in DERIVED_CYCLE_COLUMNS
    ]

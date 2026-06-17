import pandas as pd
import numpy as np
from typing import Dict, Any


class PCOSPreprocessor:
    """Handles data cleaning and preprocessing for PCOS model."""

    def __init__(self, noise_injection: bool = False, seed: int = 42):
        self.noise_injection = noise_injection
        self.seed = seed
        np.random.seed(seed)

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip()
        return df

    def convert_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df["PCOS"] = df["PCOS"].map({"YES": 1, "NO": 0})
        return df

    def convert_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        if "HypoA" in df.columns:
            df["HypoA"] = df["HypoA"].map({"Yes": 1, "No": 0})
        if "Cycle" in df.columns:
            df["Cycle"] = df["Cycle"].map({"YES": 1, "NO": 0})
        return df

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = ["PatientID"]
        leakage_cols = ["cycle_mean", "cycle_std", "cycle_variance", "cycle_range"]

        for col in drop_cols + leakage_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        return df

    def inject_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.noise_injection:
            return df

        # Cycle noise
        cycle_cols = ["Month_t-6","Month_t-5","Month_t-4","Month_t-3","Month_t-2","Month_t-1"]
        for col in cycle_cols:
            if col in df.columns:
                df[col] += np.random.normal(0, 2, size=len(df))

        # Label noise
        flip_prob = 0.05
        valid_mask = df["PCOS"].notna()
        flip_mask = (np.random.rand(len(df)) < flip_prob) & valid_mask
        df.loc[flip_mask, "PCOS"] = 1 - df.loc[flip_mask, "PCOS"]

        # Feature noise
        feature_cols = [
            "UpperLip","Chin","Chest","UpperAbdomen","LowerAbdomen",
            "UpperArm","Thigh","UpperBack","LowerBack",
            "Ludwig","GAGS_score"
        ]
        for col in feature_cols:
            if col in df.columns:
                df[col] += np.random.normal(0, 0.5, size=len(df))
                df[col] = df[col].clip(lower=0)

        # Missing values
        missing_prob = 0.05
        for col in df.columns:
            if col != "PCOS":
                df.loc[np.random.rand(len(df)) < missing_prob, col] = np.nan

        return df

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(df.mean(numeric_only=True))

    def shuffle(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def preprocess(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        df = self.clean_column_names(df)
        df = self.convert_target(df)
        df = self.convert_categorical(df)
        df = self.drop_columns(df)

        if training:
            df = self.shuffle(df)
            df = self.inject_noise(df)

        df = self.handle_missing(df)
        return df

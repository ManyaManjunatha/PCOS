from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from project.models.fusion_model import FusionModel


def train_from_csv(
    dataset_path: str,
    model_output_path: str,
    *,
    target_column: str = "pcos_label",
    metrics_output_dir: str = "outputs/fusion_metrics",
    test_size: float = 0.2,
) -> Dict[str, Any]:
    df = pd.read_csv(dataset_path)
    model = FusionModel()
    summary = model.train_test_evaluate(
        df,
        target_column=target_column,
        test_size=test_size,
        output_dir=metrics_output_dir,
    )
    model.fit_dataframe(df, target_column=target_column)
    model.save(model_output_path)

    Path(metrics_output_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(metrics_output_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGBoost PCOS fusion model.")
    parser.add_argument("--data", required=True, help="CSV containing MVP features and PCOS labels.")
    parser.add_argument("--target", default="pcos_label", help="Binary target column name.")
    parser.add_argument("--model-out", default="artifacts/fusion_model.joblib", help="Model artifact path.")
    parser.add_argument("--metrics-out", default="outputs/fusion_metrics", help="Metrics/plot output directory.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction for evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_from_csv(
        dataset_path=args.data,
        model_output_path=args.model_out,
        target_column=args.target,
        metrics_output_dir=args.metrics_out,
        test_size=args.test_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from project.models.clinical_models import ClinicalPCOSModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train explainable clinical PCOS prediction models.")
    parser.add_argument("--data", default="data/pcos_final.xlsx", help="Path to pcos_final.xlsx.")
    parser.add_argument("--model-out", default="artifacts/pcos_model.joblib", help="Best model output path.")
    parser.add_argument("--figures-out", default="outputs/figures", help="Directory for generated plots.")
    parser.add_argument("--reports-out", default="outputs/reports", help="Directory for metric CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = ClinicalPCOSModelTrainer()
    result = trainer.train_compare(
        args.data,
        model_output_path=args.model_out,
        figures_dir=args.figures_out,
        reports_dir=args.reports_out,
    )
    print(
        json.dumps(
            {
                "best_model_name": result.best_model_name,
                "holdout_metrics": result.holdout_metrics,
                "confusion_matrix": result.confusion_matrix,
                "top_features": result.feature_importance.head(10).to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

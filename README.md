# Explainable PCOS Prediction System

This repository trains and evaluates an explainable PCOS prediction system using the clinical and phenotypic feature dataset `data/pcos_final.xlsx`.

The project is tabular-only. It does **not** use acne images, acne checkpoints, fusion image models, or multimodal image pipelines.

## Dataset

The sole dataset is:

```text
data/pcos_final.xlsx
```

The workbook contains 3,000 records with a balanced `PCOS` target:

- `YES`: 1,500
- `NO`: 1,500

Input features include mFG region scores, total mFG score, Ludwig score, GAGS score, hypoandrogenism indicator, menstrual-cycle history, and cycle regularity.

## Project Structure

```text
project/
├── api/
│   └── predict.py
├── explainability/
│   └── shap_explainer.py
├── models/
│   └── clinical_models.py
├── preprocessing/
│   └── clinical_preprocessor.py
└── training/
    └── train_clinical_models.py

data/
└── pcos_final.xlsx

artifacts/
└── pcos_model.joblib

outputs/
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── shap_summary.png
└── reports/
    ├── feature_importance.csv
    ├── holdout_metrics.csv
    └── model_comparison_cv.csv
```

## Models Trained

The training pipeline compares:

- Logistic Regression
- Random Forest
- XGBoost

All models are evaluated with 5-fold stratified cross-validation using:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

The best model is selected by mean cross-validation ROC-AUC, with F1 and accuracy as tie-breakers.

## Installation

```bash
pip install -r requirements.txt
```

## Train and Evaluate

```bash
python -m project.training.train_clinical_models \
  --data data/pcos_final.xlsx \
  --model-out artifacts/pcos_model.joblib \
  --figures-out outputs/figures \
  --reports-out outputs/reports
```

This generates:

- `artifacts/pcos_model.joblib`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/roc_curve.png`
- `outputs/figures/shap_summary.png`
- `outputs/reports/model_comparison_cv.csv`
- `outputs/reports/holdout_metrics.csv`
- `outputs/reports/feature_importance.csv`

## Prediction API

```python
import pandas as pd
from project.api.predict import predict_pcos

records = pd.read_excel("data/pcos_final.xlsx").drop(columns=["PCOS", "PatientID"]).head(5)
predictions = predict_pcos(records, model_path="artifacts/pcos_model.joblib")

print(predictions)
```

Output columns:

- `pcos_probability`
- `pcos_prediction`
- `pcos_label`

## Explainability

The best model is explained with SHAP. The training pipeline produces:

- A SHAP summary plot: `outputs/figures/shap_summary.png`
- A ranked feature importance table: `outputs/reports/feature_importance.csv`

SHAP values explain model behavior, not biological causality.

## Clinical Framing

This system is a research prototype for explainable PCOS prediction from clinical and phenotypic variables. It is not a standalone medical diagnostic system and should not be used to replace clinician assessment, laboratory tests, imaging, or formal diagnostic criteria.

## Removed Dependencies

The current project has no dependency on:

- `acne_model.pth`
- `fusion_model.joblib`
- `fusion_training.csv`
- image-based acne modules
- multimodal fusion image pipelines

## Tests

```bash
pytest
```

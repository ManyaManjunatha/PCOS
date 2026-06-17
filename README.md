# AI-Assisted PCOS Risk Screening Tool

Research MVP for:

**A Multi-Modal AI Framework for Non-Invasive PCOS Risk Screening Using Dermatological Phenotypes and Clinical Features**

This repository implements a realistic research prototype for PCOS risk screening. It is **not** a PCOS diagnostic system and does not replace clinician evaluation, laboratory testing, ultrasound, or formal diagnostic criteria.

## Repository Audit Report

### Previous Architecture

The repository previously contained a flat unified interface:

- `diagnosis.py`: combined PCOS and acne prediction under a diagnostic API.
- `acne/`: ConvNeXt-Tiny acne classifier and image transforms.
- `pcos/`: XGBoost-style tabular model using legacy dataset columns.
- `models/pcos_model.pkl`: serialized legacy PCOS model artifact.
- `README.md`, `USAGE.md`, `test_unified.py`: documentation and tests for the old interface.

### Missing Components

- No fusion model combining acne severity with clinical questionnaire inputs.
- No SHAP explanation layer.
- No training CLI for the target MVP schema.
- No research evaluation outputs for confusion matrix, ROC, precision, recall, F1, and AUC.
- No clear distinction between screening output and clinical diagnosis.

### Dead Code Removed

- Legacy `diagnosis.py`.
- Legacy `pcos/` preprocessing/model code tied to unsupported columns.
- Legacy `acne/` package after moving the ConvNeXt acne wrapper into `project/models/acne_model.py`.
- Outdated `USAGE.md` and `test_unified.py`.

### Incorrect Assumptions Fixed

- The old PCOS flow expected body-region hirsutism columns and Ludwig/GAGS-style fields, which does not match the target MVP.
- The old documentation described a diagnostic system. The refactor describes a risk screening tool.
- The old combined output presented acne and PCOS as separate predictions instead of using acne severity as one fusion feature.

### Technical Debt Addressed

- Added explicit feature validation and bounded numeric ranges.
- Added reproducible model save/load artifacts.
- Added deterministic public API output shape.
- Added focused tests for feature engineering, risk categories, and report generation.

## Refactored Architecture

```text
project/
├── models/
│   ├── acne_model.py
│   └── fusion_model.py
├── preprocessing/
│   └── feature_engineering.py
├── explainability/
│   └── shap_explainer.py
├── api/
│   └── diagnosis.py
├── training/
│   └── train_fusion.py
├── tests/
└── README.md
```

## MVP Pipeline

```text
Acne image
  -> ConvNeXt acne model
  -> acne_class, acne_confidence, acne_severity_score

Clinical questionnaire
  -> age, BMI, ethnicity, menstrual regularity, hirsutism score, hair loss score
  -> feature engineering

Acne severity + clinical features
  -> XGBoost fusion model
  -> PCOS probability and 0-100 risk score
  -> SHAP ranked feature contributions
```

## Inputs

The public diagnosis function accepts:

- Acne image path
- Age
- BMI
- Ethnicity
- Menstrual irregularity
- Self-reported hirsutism score
- Self-reported hair loss score

Hirsutism and hair-loss values are questionnaire-based. The system does not infer mFG or Ludwig scores from images.

## Output

```python
{
    "pcos_risk_score": 0,
    "pcos_probability": 0.0,
    "risk_category": "Low",
    "acne_class": "normal_skin",
    "acne_confidence": 0.95,
    "acne_severity": "None/minimal",
    "acne_severity_score": 0.02,
    "feature_importance": {
        "menstrual_irregularity": 0.31,
        "bmi": 0.12
    },
    "recommendation": "Low screening risk. Continue routine self-monitoring and seek clinical care if symptoms change.",
    "screening_inputs": {
        "age": 24,
        "bmi": 22.8,
        "ethnicity": "South Asian",
        "menstrual_irregularity": True,
        "hirsutism_score": 8,
        "hair_loss_score": 1
    }
}
```

Risk categories:

- `0-30`: Low
- `31-70`: Moderate
- `71-100`: High

## Installation

```bash
pip install -r requirements.txt
```

The fusion model requires `xgboost`. SHAP explanations require `shap`.

## Usage

```python
from project.api.diagnosis import diagnose_patient

report = diagnose_patient(
    image_path="patient_acne_image.jpg",
    age=24,
    bmi=27.4,
    ethnicity="South Asian",
    menstrual_irregularity=True,
    hirsutism_score=14,
    hair_loss_score=2,
    acne_model_path="artifacts/acne_model.pth",
    fusion_model_path="artifacts/fusion_model.joblib",
)

print(report)
```

## Training Data Schema

The fusion training CSV must contain:

```text
acne_severity_score
age
bmi
ethnicity
menstrual_irregularity
hirsutism_score
hair_loss_score
pcos_label
```

`pcos_label` may be `0/1`, `yes/no`, `positive/negative`, or equivalent values handled by `FusionModel`.

## Training Instructions

```bash
python -m project.training.train_fusion \
  --data data/fusion_training.csv \
  --target pcos_label \
  --model-out artifacts/fusion_model.joblib \
  --metrics-out outputs/fusion_metrics
```

Training outputs:

- `artifacts/fusion_model.joblib`
- `outputs/fusion_metrics/metrics.json`
- `outputs/fusion_metrics/confusion_matrix.png`
- `outputs/fusion_metrics/roc_curve.png`

## Evaluation Outputs

`FusionModel.evaluate_dataframe()` returns:

- Confusion matrix
- Precision
- Recall
- F1 score
- AUC
- ROC curve arrays

## SHAP Integration

`project/explainability/shap_explainer.py` uses `shap.TreeExplainer` for the trained XGBoost fusion model. Each prediction returns ranked feature contributions sorted by absolute SHAP magnitude.

## Research-Paper Methodology Draft

This study implements a non-invasive, AI-assisted PCOS risk screening prototype combining dermatological phenotype information with structured self-reported clinical features. Acne imagery is processed using a ConvNeXt-Tiny convolutional neural network pretrained on ImageNet and adapted for four acne-related classes: cyst, papules, pustules, and normal skin. The resulting class probabilities are transformed into a normalized acne severity score using severity weights assigned to acne phenotypes.

Structured patient features include age, BMI, ethnicity, menstrual regularity, self-reported hirsutism score, and self-reported hair-loss score. Feature engineering maps age into target reproductive-age groups, categorizes BMI using standard thresholds, encodes ethnicity and menstrual irregularity, and normalizes questionnaire scores. The normalized acne severity feature is concatenated with the engineered clinical features and passed to an XGBoost binary classifier to estimate PCOS risk probability.

The predicted probability is converted to a 0-100 risk score and categorized as low, moderate, or high risk. Model evaluation uses a held-out test set and reports confusion matrix, precision, recall, F1 score, AUC, and ROC curve. SHAP TreeExplainer is used to generate ranked per-prediction feature contributions, supporting transparent review of the dominant factors influencing each screening result.

## Assumptions and Limitations

- This is a research MVP for risk screening, not clinical diagnosis.
- The system assumes a trained acne checkpoint and a trained fusion model artifact are available.
- The legacy `models/pcos_model.pkl` was trained for the old schema and should not be used as the new fusion model.
- Hirsutism and hair loss are questionnaire inputs, not automated image-derived scores.
- Acne severity is only one dermatological phenotype and may be affected by lighting, skin tone representation, image quality, and dataset bias.
- Ethnicity encoding is simplified for MVP feasibility and must be validated carefully before any real-world use.
- Risk categories are operational thresholds for research screening and require clinical validation.
- SHAP explains model behavior, not biological causality.

## Tests

```bash
pytest
```

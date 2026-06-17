# Unified Diagnosis System

A combined diagnostic system for PCOS (Polycystic Ovary Syndrome) detection and acne severity classification.

## Overview

This system provides a unified interface for two medical diagnostic models:

| Model | Type | Architecture | Input |
|-------|------|--------------|-------|
| **PCOS Detection** | Binary classification | XGBoost | Tabular patient data (CSV) |
| **Acne Classification** | Multi-class + severity | ConvNeXt-Tiny | Skin images |

## Installation

```bash
cd unified_diagnosis
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- For PCOS: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`
- For Acne: `torch`, `torchvision`, `pillow`

## Quick Start

```python
from diagnosis import Diagnose

# Initialize with both models
diagnoser = Diagnose(
    pcos_model_path="models/pcos_model.pkl",
    acne_model_path="models/acne_model.pth"
)

# PCOS prediction
pcos_result = diagnoser.predict_pcos("patient_data.csv")
print(f"PCOS detected: {pcos_result['pcos_detected']}")

# Acne prediction
acne_result = diagnoser.predict_acne("skin_image.jpg")
print(f"Predicted: {acne_result['predicted_class']}")
print(f"Severity: {acne_result['severity_label']}")

# Combined diagnosis
full_result = diagnoser.diagnose("patient_data.csv", "skin_image.jpg")
```

## API Reference

### `Diagnose` Class

#### Initialization

```python
Diagnose(
    pcos_model_path: Optional[str] = None,
    acne_model_path: Optional[str] = None,
    device: Optional[str] = None
)
```

| Parameter | Description |
|-----------|-------------|
| `pcos_model_path` | Path to `.pkl` file. Required for `predict_pcos()`. |
| `acne_model_path` | Path to `.pth` checkpoint. Required for `predict_acne()`. |
| `device` | Device for acne inference (`'cuda'`, `'cpu'`, or `None` for auto). |

#### Methods

##### `predict_pcos(data, return_proba=False)`

Predict PCOS from patient tabular data.

```python
result = diagnoser.predict_pcos("patient.csv", return_proba=True)
```

**Returns:**
```python
{
    'pcos_detected': True,
    'pcos_label': 'YES',
    'probability': {'NO': 0.15, 'YES': 0.85},  # if return_proba=True
    'input_data': <DataFrame>
}
```

##### `predict_acne(image_path)`

Predict acne type and severity from an image.

```python
result = diagnoser.predict_acne("face.jpg")
```

**Returns:**
```python
{
    'predicted_class': 'Pustules',
    'confidence': 0.87,
    'all_probs': {
        'Cyst': 0.05,
        'Papules': 0.08,
        'Pustules': 0.87,
        'normal_skin': 0.00
    },
    'severity_score': 0.72,
    'severity_label': 'Moderate acne',
    'severity_code': 2  # 0=None, 1=Mild, 2=Moderate, 3=Severe
}
```

##### `diagnose(patient_data, image_path)`

Run both diagnostics in a single call.

```python
result = diagnoser.diagnose("patient.csv", "face.jpg")
# Returns: {'pcos': {...}, 'acne': {...}}
```

##### `load_pcos_model(path)` / `load_acne_model(path)`

Dynamically load models after initialization.

## Model Details

### PCOS Detection

**Architecture:** XGBoost classifier (150 estimators, max_depth=4, lr=0.1)

**Expected Input Columns:**
- Cycle history: `Month_t-6` through `Month_t-1`
- Hirsutism features: `UpperLip`, `Chin`, `Chest`, `UpperAbdomen`, `LowerAbdomen`, `UpperArm`, `Thigh`, `UpperBack`, `LowerBack`
- Hormonal markers: `Ludwig`, `GAGS_score`
- Binary flags: `Cycle` (YES/NO), `HypoA` (Yes/No)

**Preprocessing:**
- Automatic column name cleaning
- Categorical encoding (YES/NO → 1/0)
- Missing value imputation (mean)
- Data leakage column removal

### Acne Classification

**Architecture:** ConvNeXt-Tiny (ImageNet pretrained)
- Frozen backbone: layers 0-5
- Trainable: layers 6+ and custom classifier head

**Classes:** `Cyst`, `Papules`, `Pustules`, `normal_skin`

**Severity Scoring:**
| Class | Weight |
|-------|--------|
| Cyst | 1.0 |
| Pustules | 0.8 |
| Papules | 0.6 |

**Severity Levels:**
| Code | Label | Score Range |
|------|-------|-------------|
| 0 | No acne | acne_sum < 0.03 |
| 1 | Mild acne | score < 0.5 |
| 2 | Moderate acne | 0.5 ≤ score < 0.8 |
| 3 | Severe acne | score ≥ 0.8 |

**Image Requirements:**
- Format: Any PIL-compatible format (JPG, PNG, etc.)
- Automatically resized to 224x224
- Normalized to ImageNet statistics

## Project Structure

```
unified_diagnosis/
├── diagnosis.py          # Main unified interface
├── pcos/
│   ├── model.py          # PCOSModel class
│   └── preprocessor.py   # Data cleaning pipeline
├── acne/
│   ├── model.py          # AcneModel class (ConvNeXt)
│   └── transforms.py     # Image transforms
├── models/               # Model artifacts
│   ├── pcos_model.pkl
│   └── acne_model.pth
├── requirements.txt
├── test_unified.py       # Test script
└── README.md
```

## Testing

```bash
cd unified_diagnosis
python test_unified.py
```

## Training Your Own Models

### PCOS Model

```python
from pcos.model import PCOSModel

model = PCOSModel()
model.train(df, compare_models=True)  # Compare LR, RF, XGBoost
model.fit(df)  # Train on full data
model.save("models/pcos_model.pkl")
```

### Acne Model

See `muffi_model.py` in the parent directory for the full training pipeline.

## License

MIT

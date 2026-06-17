# Usage Examples

## Basic Usage

### PCOS-Only Diagnosis

```python
from diagnosis import Diagnose

# Load only the PCOS model
diagnoser = Diagnose(pcos_model_path="models/pcos_model.pkl")

# Predict from CSV file
result = diagnoser.predict_pcos("patient_data.csv")
print(f"PCOS: {result['pcos_label']}")

# Predict from DataFrame
import pandas as pd
df = pd.read_csv("patient_data.csv")
result = diagnoser.predict_pcos(df)

# Get probabilities
result = diagnoser.predict_pcos(df, return_proba=True)
print(f"Probability of PCOS: {result['probability']['YES']:.2%}")
```

### Acne-Only Diagnosis

```python
from diagnosis import Diagnose

# Load only the acne model
diagnoser = Diagnose(acne_model_path="models/acne_model.pth")

# Predict from image
result = diagnoser.predict_acne("patient_photo.jpg")

print(f"Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Severity: {result['severity_label']}")

# Get all class probabilities
for cls, prob in result['all_probs'].items():
    print(f"  {cls}: {prob:.2%}")
```

### Combined Diagnosis

```python
from diagnosis import Diagnose

diagnoser = Diagnose(
    pcos_model_path="models/pcos_model.pkl",
    acne_model_path="models/acne_model.pth"
)

# Single call for full diagnosis
result = diagnoser.diagnose("patient_data.csv", "skin_image.jpg")

print("=== PCOS Results ===")
print(f"Detected: {result['pcos']['pcos_label']}")
print(f"Confidence: {result['pcos']['probability']['YES']:.2%}")

print("\n=== Acne Results ===")
print(f"Type: {result['acne']['predicted_class']}")
print(f"Severity: {result['acne']['severity_label']}")
```

## Advanced Usage

### Dynamic Model Loading

```python
from diagnosis import Diagnose

# Start without any models
diagnoser = Diagnose()

# Load models on demand
diagnoser.load_pcos_model("models/pcos_model.pkl")
diagnoser.load_acne_model("models/acne_model.pth")

# Now use as normal
result = diagnoser.diagnose("data.csv", "image.jpg")
```

### GPU Acceleration

```python
from diagnosis import Diagnose

# Force GPU usage for acne model
diagnoser = Diagnose(
    acne_model_path="models/acne_model.pth",
    device="cuda"  # or "cpu"
)
```

### Batch PCOS Predictions

```python
from diagnosis import Diagnose
import pandas as pd

diagnoser = Diagnose(pcos_model_path="models/pcos_model.pkl")

# Load multiple patients
df = pd.read_csv("multiple_patients.csv")

# Get predictions for all rows
predictions = diagnoser._pcos_model.predict(df)

# Add to dataframe
df['prediction'] = predictions
df['pcos_flag'] = df['prediction'].map({0: 'NO', 1: 'YES'})
```

### Custom Severity Thresholds

```python
from acne.model import AcneModel

model = AcneModel(model_path="models/acne_model.pth")

# Get raw prediction
result = model.predict("image.jpg")

# Apply custom severity logic
score = result['severity_score']
if score > 0.9:
    custom_label = "Critical - refer to specialist"
elif score > 0.6:
    custom_label = "High priority"
else:
    custom_label = "Standard care"
```

## Error Handling

```python
from diagnosis import Diagnose

diagnoser = Diagnose()  # No models loaded

try:
    diagnoser.predict_pcos("data.csv")
except ValueError as e:
    print(f"Error: {e}")
    # "PCOS model not loaded. Provide pcos_model_path..."

# Check if models are loaded before use
if diagnoser._pcos_model is None:
    diagnoser.load_pcos_model("models/pcos_model.pkl")
```

## Integration Example: Web API

```python
from fastapi import FastAPI, File, UploadFile, Form
from diagnosis import Diagnose

app = FastAPI()
diagnoser = Diagnose(
    pcos_model_path="models/pcos_model.pkl",
    acne_model_path="models/acne_model.pth"
)

@app.post("/diagnose/")
async def diagnose(
    data_file: UploadFile,
    image: UploadFile = File(...)
):
    # Save uploads temporarily
    with open("temp_data.csv", "wb") as f:
        f.write(await data_file.read())
    with open("temp_image.jpg", "wb") as f:
        f.write(await image.read())
    
    # Run diagnosis
    result = diagnoser.diagnose("temp_data.csv", "temp_image.jpg")
    
    # Clean up
    import os
    os.remove("temp_data.csv")
    os.remove("temp_image.jpg")
    
    return result
```

## Interpreting Results

### PCOS Output

| Field | Description |
|-------|-------------|
| `pcos_detected` | Boolean: True if PCOS predicted |
| `pcos_label` | Human-readable: "YES" or "NO" |
| `probability` | Dict with confidence for each class |
| `input_data` | The preprocessed DataFrame used |

### Acne Output

| Field | Description |
|-------|-------------|
| `predicted_class` | Most likely class (Cyst/Papules/Pustules/normal_skin) |
| `confidence` | Probability of predicted class |
| `all_probs` | Full probability distribution |
| `severity_score` | Weighted score (0-1) based on acne type |
| `severity_label` | Human-readable severity |
| `severity_code` | Numeric severity (0-3) |

### Severity Code Reference

```
0 = No acne (acne probability < 3%)
1 = Mild acne (severity score < 0.5)
2 = Moderate acne (severity score 0.5-0.8)
3 = Severe acne (severity score >= 0.8)
```

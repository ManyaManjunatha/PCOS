"""Test script for unified diagnosis system."""

import sys
import pandas as pd

# Test PCOS model directly
print("Testing PCOS model...")
from pcos.model import PCOSModel

model = PCOSModel("models/pcos_model.pkl")
df = pd.read_csv("../pcos_detection/pcos_final.csv")
prediction = model.predict(df.head(1))
print(f"PCOS prediction (first row): {prediction[0]}")
print("✅ PCOS model working!")

# Test Diagnose class
print("\nTesting Diagnose class...")
sys.path.insert(0, '.')
from diagnosis import Diagnose

diagnoser = Diagnose(pcos_model_path="models/pcos_model.pkl")
result = diagnoser.predict_pcos("../pcos_detection/pcos_final.csv", return_proba=True)
print(f"PCOS detected: {result['pcos_detected']}")
print(f"Probability: {result['probability']}")
print("✅ Diagnose class working!")

# Note: Acne model requires .pth file which needs to be saved from training
print("\nNote: Acne model requires .pth checkpoint file from training")

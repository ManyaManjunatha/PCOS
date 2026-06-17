import pandas as pd

from project.models.clinical_models import ClinicalPCOSModelTrainer


def test_model_candidate_names_are_expected():
    features = pd.DataFrame(
        {
            "UpperLip": [0, 1, 2, 3],
            "Ludwig": [0, 1, 2, 3],
            "HypoA": ["No", "Yes", "No", "Yes"],
            "Cycle": ["NO", "YES", "NO", "YES"],
        }
    )

    candidates = ClinicalPCOSModelTrainer().build_model_candidates(features)

    assert list(candidates.keys()) == ["Logistic Regression", "Random Forest", "XGBoost"]

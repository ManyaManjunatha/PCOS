import pandas as pd

from project.preprocessing.clinical_preprocessor import ClinicalPCOSPreprocessor


def test_clinical_preprocessor_splits_features_and_target():
    df = pd.DataFrame(
        {
            "PatientID": [1, 2],
            "UpperLip": [2, 0],
            "HypoA": ["Yes", "No"],
            "Cycle": ["YES", "NO"],
            "PCOS": ["YES", "NO"],
        }
    )

    bundle = ClinicalPCOSPreprocessor().split_features_target(df)

    assert "PatientID" not in bundle.features.columns
    assert "PCOS" not in bundle.features.columns
    assert bundle.target.tolist() == [1, 0]


def test_clinical_preprocessor_builds_feature_names():
    df = pd.DataFrame(
        {
            "UpperLip": [2, 0],
            "HypoA": ["Yes", "No"],
            "Cycle": ["YES", "NO"],
        }
    )
    preprocessor = ClinicalPCOSPreprocessor().build_preprocessor(df, scale_numeric=True)

    transformed = preprocessor.fit_transform(df)
    names = preprocessor.get_feature_names_out()

    assert transformed.shape[0] == 2
    assert "UpperLip" in names
    assert any(name.startswith("HypoA_") for name in names)


def test_clinical_preprocessor_strips_column_whitespace():
    df = pd.DataFrame({"mfg result ": [3], "PCOS": ["YES"]})

    cleaned = ClinicalPCOSPreprocessor().clean_column_names(df)

    assert "mfg result" in cleaned.columns
    assert "mfg result " not in cleaned.columns

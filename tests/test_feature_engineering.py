import pytest

from project.preprocessing.feature_engineering import FeatureEngineer, MODEL_FEATURE_COLUMNS


def test_feature_engineering_encodes_questionnaire_inputs():
    engineer = FeatureEngineer()

    features = engineer.transform_patient(
        acne_severity_score=0.72,
        age=24,
        bmi=31.2,
        ethnicity="South Asian",
        menstrual_irregularity="irregular",
        hirsutism_score=18,
        hair_loss_score=2,
    )

    row = features.iloc[0].to_dict()
    assert list(features.columns) == MODEL_FEATURE_COLUMNS
    assert row["age_group_18_25"] == 1.0
    assert row["age_group_26_30"] == 0.0
    assert row["bmi_obese"] == 1.0
    assert row["ethnicity_encoded"] == 1.0
    assert row["menstrual_irregularity"] == 1.0
    assert row["hirsutism_score"] == pytest.approx(0.5)
    assert row["hair_loss_score"] == pytest.approx(2 / 3)


def test_feature_engineering_rejects_out_of_range_scores():
    engineer = FeatureEngineer()

    with pytest.raises(ValueError, match="hirsutism_score"):
        engineer.transform_patient(
            acne_severity_score=0.5,
            age=27,
            bmi=24,
            ethnicity="other",
            menstrual_irregularity=False,
            hirsutism_score=40,
            hair_loss_score=1,
        )

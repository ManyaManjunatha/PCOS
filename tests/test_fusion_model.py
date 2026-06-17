from project.models.fusion_model import FusionModel


def test_risk_score_categories_and_recommendations():
    model = FusionModel()

    assert model.probability_to_score(0.304) == 30
    assert model.risk_category(30) == "Low"
    assert model.risk_category(31) == "Moderate"
    assert model.risk_category(70) == "Moderate"
    assert model.risk_category(71) == "High"
    assert "clinical evaluation" in model.recommendation("High")

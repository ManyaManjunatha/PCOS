from project.api.diagnosis import DiagnosisPipeline


class FakeAcneModel:
    def predict(self, image_path):
        return {
            "acne_class": "Pustules",
            "acne_confidence": 0.88,
            "acne_severity": "Moderate",
            "acne_severity_score": 0.68,
            "class_probabilities": {
                "Cyst": 0.1,
                "Papules": 0.1,
                "Pustules": 0.78,
                "normal_skin": 0.02,
            },
        }


class FakeFusionModel:
    model = object()
    feature_columns = []

    def predict_patient(self, **kwargs):
        return {
            "pcos_probability": 0.73,
            "pcos_risk_score": 73,
            "risk_category": "High",
            "model_features": kwargs,
        }

    def recommendation(self, risk_category):
        return "High screening risk. Arrange clinical evaluation for confirmatory assessment and management."


def test_diagnosis_pipeline_returns_research_mvp_report():
    pipeline = DiagnosisPipeline(
        acne_model=FakeAcneModel(),
        fusion_model=FakeFusionModel(),
        enable_shap=False,
    )

    report = pipeline.diagnose_patient(
        image_path="sample.jpg",
        age=23,
        bmi=29,
        ethnicity="South Asian",
        menstrual_irregularity=True,
        hirsutism_score=12,
        hair_loss_score=1,
    )

    assert report["pcos_risk_score"] == 73
    assert report["risk_category"] == "High"
    assert report["acne_severity"] == "Moderate"
    assert report["feature_importance"] == {}
    assert report["screening_inputs"]["hirsutism_score"] == 12

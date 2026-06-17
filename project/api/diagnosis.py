from __future__ import annotations

from typing import Any, Dict, Optional

from project.explainability.shap_explainer import ShapExplainer
from project.models.acne_model import AcneModel
from project.models.fusion_model import FusionModel


class DiagnosisPipeline:
    """End-to-end AI-assisted PCOS risk screening pipeline."""

    def __init__(
        self,
        *,
        acne_model_path: Optional[str] = None,
        fusion_model_path: Optional[str] = None,
        device: Optional[str] = None,
        acne_model: Optional[Any] = None,
        fusion_model: Optional[FusionModel] = None,
        enable_shap: bool = True,
    ) -> None:
        self.acne_model = acne_model if acne_model is not None else AcneModel(acne_model_path, device=device)
        self.fusion_model = fusion_model if fusion_model is not None else FusionModel(fusion_model_path)
        self.enable_shap = enable_shap

    def diagnose_patient(
        self,
        *,
        image_path: str,
        age: float,
        bmi: float,
        ethnicity: str,
        menstrual_irregularity: Any,
        hirsutism_score: float,
        hair_loss_score: float,
    ) -> Dict[str, Any]:
        acne_result = self.acne_model.predict(image_path)
        fusion_result = self.fusion_model.predict_patient(
            acne_severity_score=acne_result["acne_severity_score"],
            age=age,
            bmi=bmi,
            ethnicity=ethnicity,
            menstrual_irregularity=menstrual_irregularity,
            hirsutism_score=hirsutism_score,
            hair_loss_score=hair_loss_score,
        )

        feature_importance: Dict[str, float] = {}
        if self.enable_shap:
            explainer = ShapExplainer(self.fusion_model.model, self.fusion_model.feature_columns)
            feature_frame = self.fusion_model.feature_engineer.transform_patient(
                acne_severity_score=acne_result["acne_severity_score"],
                age=age,
                bmi=bmi,
                ethnicity=ethnicity,
                menstrual_irregularity=menstrual_irregularity,
                hirsutism_score=hirsutism_score,
                hair_loss_score=hair_loss_score,
            )
            feature_importance = explainer.explain(feature_frame)

        risk_category = fusion_result["risk_category"]
        return {
            "pcos_risk_score": fusion_result["pcos_risk_score"],
            "pcos_probability": fusion_result["pcos_probability"],
            "risk_category": risk_category,
            "acne_class": acne_result["acne_class"],
            "acne_confidence": acne_result["acne_confidence"],
            "acne_severity": acne_result["acne_severity"],
            "acne_severity_score": acne_result["acne_severity_score"],
            "feature_importance": feature_importance,
            "recommendation": self.fusion_model.recommendation(risk_category),
            "screening_inputs": {
                "age": age,
                "bmi": bmi,
                "ethnicity": ethnicity,
                "menstrual_irregularity": menstrual_irregularity,
                "hirsutism_score": hirsutism_score,
                "hair_loss_score": hair_loss_score,
            },
        }


def diagnose_patient(
    image_path: str,
    age: float,
    bmi: float,
    ethnicity: str,
    menstrual_irregularity: Any,
    hirsutism_score: float,
    hair_loss_score: float,
    *,
    acne_model_path: Optional[str] = None,
    fusion_model_path: Optional[str] = None,
    device: Optional[str] = None,
    enable_shap: bool = True,
) -> Dict[str, Any]:
    pipeline = DiagnosisPipeline(
        acne_model_path=acne_model_path,
        fusion_model_path=fusion_model_path,
        device=device,
        enable_shap=enable_shap,
    )
    return pipeline.diagnose_patient(
        image_path=image_path,
        age=age,
        bmi=bmi,
        ethnicity=ethnicity,
        menstrual_irregularity=menstrual_irregularity,
        hirsutism_score=hirsutism_score,
        hair_loss_score=hair_loss_score,
    )

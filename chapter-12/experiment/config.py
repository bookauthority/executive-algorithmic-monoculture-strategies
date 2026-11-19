from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# Lightweight in-code equivalents of the YAML configuration files


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    model_type: str
    owner: str


@dataclass(frozen=True)
class WorkloadConfig:
    name: str
    description: str
    models: List[ModelConfig]


def get_workload_configs() -> List[WorkloadConfig]:
    """Return statically defined workload configurations.

    These mirror, in simplified form, the workloads from the specification.
    """

    return [
        WorkloadConfig(
            name="Credit_Risk_Portfolio_Monitoring",
            description=(
                "Monitor multiple credit risk scoring models used in different "
                "regions or product lines."
            ),
            models=[
                ModelConfig(
                    model_id="CRD_A",
                    model_type="Gradient boosted trees",
                    owner="Retail_Consumer_Credit",
                ),
                ModelConfig(
                    model_id="CRD_B",
                    model_type="Vendor SaaS credit model",
                    owner="SME_Lending",
                ),
                ModelConfig(
                    model_id="CRD_C",
                    model_type="Logistic regression benchmark",
                    owner="Risk_Methodology_Team",
                ),
            ],
        ),
        WorkloadConfig(
            name="Customer_Support_and_Routing_Monitoring",
            description=(
                "Monitor models that classify and route customer support tickets."
            ),
            models=[
                ModelConfig(
                    model_id="SUP_INTENT_A",
                    model_type="Fine-tuned transformer classifier",
                    owner="Customer_Support_AI",
                ),
                ModelConfig(
                    model_id="SUP_ROUTING_B",
                    model_type="Rules+ML hybrid router",
                    owner="Contact_Center_Operations",
                ),
            ],
        ),
        WorkloadConfig(
            name="Recommendation_and_Personalization_Monitoring",
            description=(
                "Monitor recommendation and ranking models across channels "
                "(web, app, email)."
            ),
            models=[
                ModelConfig(
                    model_id="REC_WEB",
                    model_type="Matrix factorization + boosting",
                    owner="Digital_Product_Team",
                ),
                ModelConfig(
                    model_id="REC_APP",
                    model_type="Deep learning recommender",
                    owner="Mobile_App_Team",
                ),
                ModelConfig(
                    model_id="REC_EMAIL",
                    model_type="Rules-based with ML ranking",
                    owner="CRM_Marketing",
                ),
            ],
        ),
    ]


def get_indicator_thresholds() -> Dict[str, Dict[str, float]]:
    """Return simplified indicator thresholds.

    These act as a compact in-code stand-in for config/alert_thresholds.yaml.
    Values are chosen to keep metrics in a reasonable numeric range.
    """

    return {
        "Input_Data_Drift_Magnitude": {"green": 0.0, "amber": 0.08, "red": 0.15},
        "Prediction_Distribution_Drift": {
            "green": 0.0,
            "amber": 0.08,
            "red": 0.2,
        },
        "Cross_Model_Output_Correlation": {
            # Higher correlation suggests emerging monoculture.
            "green": 0.0,
            "amber": 0.7,
            "red": 0.9,
        },
        "Exposure_Diversity_Index": {
            # Lower values suggest narrowing exposure.
            "green": 0.7,
            "amber": 0.5,
            "red": 0.3,
        },
        "Feedback_Loop_Intensity_Index": {
            "green": 0.0,
            "amber": 0.4,
            "red": 0.7,
        },
    }

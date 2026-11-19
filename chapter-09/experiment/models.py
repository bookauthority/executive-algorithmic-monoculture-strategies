from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


# Simple vector type alias for clarity
Vector = List[float]


@dataclass
class SyntheticModel:
    """A tiny deterministic synthetic model.

    This is *not* a real ML model; it's a simple function taking a feature
    vector and returning a binary prediction and an internal latency estimate.

    The internal latency estimate is an artificial cost representing relative
    complexity. The runner later combines this with small measured timing
    noise, but the majority is deterministic and scenario-dependent.
    """

    model_id: str
    vendor: str
    architecture: str
    base_threshold: float
    complexity: float  # used to scale simulated latency

    def predict(self, x: Vector, shock_bias: float = 0.0) -> Tuple[int, float]:
        """Return (prediction, simulated_latency_ms).

        The decision rule is a weighted sum over features with a simple hash-
        based weight scheme, plus an optional bias due to scenario shock.
        """

        if not x:
            score = 0.0
        else:
            # Deterministic pseudo-weights based on indices.
            s = 0.0
            for i, v in enumerate(x):
                w = ((i * 31) % 17 + 1) / 19.0
                s += w * v
            score = s / len(x)

        effective_threshold = self.base_threshold + shock_bias
        pred = 1 if score >= effective_threshold else 0

        # Deterministic latency component in ms
        simulated_latency_ms = 0.01 + self.complexity * 0.02 + abs(shock_bias) * 0.005
        return pred, simulated_latency_ms


@dataclass
class Portfolio:
    """Portfolio of models and simple routing.

    Baseline and treatment differ in how they route and combine models.

    For simplicity, each portfolio exposes a single `predict` method that:
    - Returns a binary prediction.
    - Uses internal routing / fallback logic.
    - Aggregates simulated latencies from the models it calls.
    """

    portfolio_id: str
    models: Dict[str, SyntheticModel]
    routing_strategy: Callable[[str, Dict[str, SyntheticModel], Vector, float], Tuple[int, float]]

    def predict(self, workload_id: str, x: Vector, shock_bias: float) -> Tuple[int, float]:
        return self.routing_strategy(workload_id, self.models, x, shock_bias)


# --- Routing strategies ---


def baseline_routing(
    workload_id: str, models: Dict[str, SyntheticModel], x: Vector, shock_bias: float
) -> Tuple[int, float]:
    """Baseline routing: monoculture single primary model across workloads.

    - Uses a single `baseline_primary_model` for every workload.
    - Shock bias is applied directly, representing lack of diversification.
    """

    primary = models["baseline_primary_model"]
    pred, latency = primary.predict(x, shock_bias=shock_bias)
    return pred, latency


def treatment_routing(
    workload_id: str, models: Dict[str, SyntheticModel], x: Vector, shock_bias: float
) -> Tuple[int, float]:
    """Treatment routing: diversified portfolio with fallbacks.

    - Different core models per workload where appropriate.
    - Under strong negative shock, may fall back to a rule-engine model.
    - A shadow model can influence decision in some workloads.
    """

    total_latency = 0.0

    # Choose primary based on workload
    if workload_id == "wl-credit-risk-batch":
        primary = models["treatment_core_model_A"]
    elif workload_id == "wl-fraud-streaming":
        primary = models["treatment_core_model_B"]
    elif workload_id == "wl-churn-batch":
        # share model A but with different bias handling
        primary = models["treatment_core_model_A"]
    else:  # wl-eligibility-api and any others
        primary = models["treatment_core_model_A"]

    # Apply a moderated shock bias: diversification cushions impact.
    moderated_bias = shock_bias * 0.6
    pred_primary, lat_primary = primary.predict(x, shock_bias=moderated_bias)
    total_latency += lat_primary

    # Shadow model used for eligibility and credit workloads; may adjust decision.
    if workload_id in ("wl-credit-risk-batch", "wl-eligibility-api"):
        shadow = models.get("treatment_shadow_model_C")
        if shadow is not None:
            pred_shadow, lat_shadow = shadow.predict(x, shock_bias=moderated_bias * 0.3)
            total_latency += lat_shadow
            # Simple disagreement handling: if they disagree, take conservative path.
            if pred_shadow != pred_primary:
                # Conservative: classify as risk-positive (1)
                pred_primary = 1

    # Fallback rule engine under severe shock (|shock_bias| > threshold)
    if abs(shock_bias) > 0.25:
        rule_engine = models.get("treatment_fallback_rule_engine")
        if rule_engine is not None:
            pred_fb, lat_fb = rule_engine.predict(x, shock_bias=0.0)
            total_latency += lat_fb
            # Blend: if fallback says high risk (1), override
            if pred_fb == 1:
                pred_primary = 1

    return pred_primary, total_latency


# --- Factory functions to build portfolios ---


def build_baseline_portfolio() -> Portfolio:
    models = {
        "baseline_primary_model": SyntheticModel(
            model_id="baseline_primary_model",
            vendor="vendor_A_cloud_platform",
            architecture="gradient_boosted_trees",
            base_threshold=0.5,
            complexity=1.0,
        )
    }
    return Portfolio(
        portfolio_id="baseline",
        models=models,
        routing_strategy=baseline_routing,
    )


def build_treatment_portfolio() -> Portfolio:
    models = {
        "treatment_core_model_A": SyntheticModel(
            model_id="treatment_core_model_A",
            vendor="vendor_A_cloud_platform",
            architecture="gradient_boosted_trees",
            base_threshold=0.48,
            complexity=1.2,
        ),
        "treatment_core_model_B": SyntheticModel(
            model_id="treatment_core_model_B",
            vendor="vendor_B_managed_ml",
            architecture="deep_neural_network",
            base_threshold=0.52,
            complexity=1.5,
        ),
        "treatment_fallback_rule_engine": SyntheticModel(
            model_id="treatment_fallback_rule_engine",
            vendor="on_prem_rules_engine",
            architecture="rule_based",
            base_threshold=0.55,
            complexity=0.6,
        ),
        "treatment_shadow_model_C": SyntheticModel(
            model_id="treatment_shadow_model_C",
            vendor="vendor_C_custom_stack",
            architecture="logistic_regression",
            base_threshold=0.5,
            complexity=0.9,
        ),
    }
    return Portfolio(
        portfolio_id="treatment",
        models=models,
        routing_strategy=treatment_routing,
    )

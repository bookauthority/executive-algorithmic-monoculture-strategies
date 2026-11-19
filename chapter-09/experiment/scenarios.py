from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


WorkloadID = str
ScenarioID = str


@dataclass(frozen=True)
class WorkloadConfig:
    workload_id: WorkloadID
    size: int
    description: str


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: ScenarioID
    shock_type: str
    shock_intensity: float  # e.g., 0.0 = none, 0.3 = moderate


# Define workloads (smaller sizes to ensure fast execution in sandbox)
WORKLOADS: Dict[WorkloadID, WorkloadConfig] = {
    "wl-credit-risk-batch": WorkloadConfig(
        workload_id="wl-credit-risk-batch",
        size=200,
        description="Batch credit risk scoring synthetic workload.",
    ),
    "wl-fraud-streaming": WorkloadConfig(
        workload_id="wl-fraud-streaming",
        size=200,
        description="Streaming fraud detection synthetic workload.",
    ),
    "wl-churn-batch": WorkloadConfig(
        workload_id="wl-churn-batch",
        size=200,
        description="Batch customer churn synthetic workload.",
    ),
    "wl-eligibility-api": WorkloadConfig(
        workload_id="wl-eligibility-api",
        size=200,
        description="Online eligibility decision synthetic workload.",
    ),
}


# Define a small set of scenarios for each workload
SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig(
        scenario_id="regulatory_shift",
        shock_type="regulatory_shift",
        shock_intensity=0.2,
    ),
    ScenarioConfig(
        scenario_id="market_shift",
        shock_type="market_distribution_shift",
        shock_intensity=0.15,
    ),
    ScenarioConfig(
        scenario_id="vendor_outage",
        shock_type="vendor_region_outage",
        shock_intensity=0.3,
    ),
]


def scenario_bias_for_portfolio(
    portfolio_id: str,
    scenario: ScenarioConfig,
    workload_id: WorkloadID,
) -> float:
    """Return a signed bias to the model threshold under a given scenario.

    Positive bias makes it harder to classify as 1, negative makes it easier.

    We make baseline more sensitive to shocks than treatment by design, but the
    actual impact on metrics still depends on data and model thresholds.
    """

    base = scenario.shock_intensity

    if scenario.shock_type == "regulatory_shift":
        # Stricter thresholds.
        direction = +1.0
    elif scenario.shock_type == "market_distribution_shift":
        # Could go either way; use positive for some workloads.
        direction = +1.0 if workload_id in ("wl-credit-risk-batch", "wl-churn-batch") else -0.5
    elif scenario.shock_type == "vendor_region_outage":
        # Outage / latency shock effectively reduces quality slightly.
        direction = -0.5
    else:
        direction = 0.0

    # Baseline more exposed: multiply by 1.0 vs 0.7 for treatment
    scale = 1.0 if portfolio_id == "baseline" else 0.7

    return base * direction * scale


def generate_synthetic_data(
    seed: int,
    workload: WorkloadConfig,
) -> Tuple[List[List[float]], List[int]]:
    """Generate deterministic synthetic features and labels for a workload.

    Labels are correlated with a simple underlying risk score so that accuracy
    has meaningful variation under shocks.
    """

    rng = random.Random(seed)
    features: List[List[float]] = []
    labels: List[int] = []

    for i in range(workload.size):
        # Feature dimensionality varies slightly by workload
        if workload.workload_id == "wl-fraud-streaming":
            dim = 8
        elif workload.workload_id == "wl-eligibility-api":
            dim = 6
        else:
            dim = 10

        x = [rng.random() for _ in range(dim)]

        # Underlying risk score: weighted sum
        risk = 0.0
        for j, v in enumerate(x):
            w = ((j * 13) % 11 + 1) / 17.0
            risk += w * v
        risk /= dim

        # Different base label thresholds per workload
        if workload.workload_id == "wl-credit-risk-batch":
            base_thr = 0.52
        elif workload.workload_id == "wl-fraud-streaming":
            base_thr = 0.6
        elif workload.workload_id == "wl-churn-batch":
            base_thr = 0.5
        else:  # eligibility
            base_thr = 0.55

        y = 1 if risk >= base_thr else 0

        features.append(x)
        labels.append(y)

    return features, labels

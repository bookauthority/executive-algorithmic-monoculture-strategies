from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import WorkloadConfig, get_indicator_thresholds, get_workload_configs
from .monitoring import (
    compute_cross_model_correlation,
    compute_drift_metrics,
    compute_exposure_diversity,
    feedback_loop_intensity,
)


@dataclass
class SimulationConfig:
    num_time_steps: int
    num_samples_per_step: int
    num_feature_bins: int
    num_score_bins: int
    num_exposure_bins: int


DEFAULT_SIM_CONFIG = SimulationConfig(
    num_time_steps=12,
    num_samples_per_step=200,
    num_feature_bins=5,
    num_score_bins=5,
    num_exposure_bins=5,
)


def _seed_for(workload_name: str, variant: str) -> int:
    return abs(hash((workload_name, variant))) % (2**31)


def _simulate_baseline_workload(
    workload: WorkloadConfig, sim: SimulationConfig
) -> Dict[str, float]:
    """Simulate a static, snapshot-like assessment for a workload.

    Returns a dict of averaged numeric metrics for this workload under the
    baseline (static) variant.
    """

    rng = random.Random(_seed_for(workload.name, "baseline"))

    # Single time step snapshot.
    num_models = len(workload.models)
    baseline_feature_dist = [rng.randint(50, 100) for _ in range(sim.num_feature_bins)]
    baseline_score_dist = [rng.randint(50, 100) for _ in range(sim.num_score_bins)]

    # Current snapshot counts with mild random perturbation.
    current_feature_dist = [max(1, c + rng.randint(-10, 10)) for c in baseline_feature_dist]
    current_score_dist = [max(1, c + rng.randint(-10, 10)) for c in baseline_score_dist]

    js_input, psi_input = compute_drift_metrics(baseline_feature_dist, current_feature_dist)
    js_scores, psi_scores = compute_drift_metrics(baseline_score_dist, current_score_dist)

    # Model scores per model for correlation (weak convergence in baseline).
    model_scores: List[List[float]] = []
    for _ in range(num_models):
        scores = [rng.uniform(0.0, 1.0) for _ in range(sim.num_samples_per_step)]
        model_scores.append(scores)

    avg_corr = compute_cross_model_correlation(model_scores)

    # Diversity: simulate balanced exposure.
    exposure_counts = [rng.randint(50, 100) for _ in range(sim.num_exposure_bins)]
    diversity_index = compute_exposure_diversity(exposure_counts)

    # Feedback intensity: very modest in baseline.
    past_decisions = []
    future_presence = []
    for _ in range(sim.num_samples_per_step):
        d = 1 if rng.random() < 0.5 else 0
        # Future presence slightly higher given positive decision.
        if d == 1:
            f = 1 if rng.random() < 0.55 else 0
        else:
            f = 1 if rng.random() < 0.5 else 0
        past_decisions.append(d)
        future_presence.append(f)

    feedback_index = feedback_loop_intensity(past_decisions, future_presence)

    # Aggregate metrics meaningful at workload level.
    metrics = {
        "avg_input_drift": js_input,
        "avg_prediction_drift": js_scores,
        "avg_cross_model_corr": avg_corr,
        "avg_diversity_index": diversity_index,
        "avg_feedback_intensity_index": feedback_index,
    }

    return metrics


def _simulate_treatment_workload(
    workload: WorkloadConfig, sim: SimulationConfig
) -> Dict[str, float]:
    """Simulate longitudinal monitoring for a workload.

    Returns aggregated numeric metrics across time steps.
    """

    rng = random.Random(_seed_for(workload.name, "treatment"))

    num_models = len(workload.models)

    # Base distributions for time step 0.
    base_feature_dist = [rng.randint(50, 100) for _ in range(sim.num_feature_bins)]
    base_score_dist = [rng.randint(50, 100) for _ in range(sim.num_score_bins)]

    # Accumulators across time.
    input_drifts: List[float] = []
    prediction_drifts: List[float] = []
    cross_model_corrs: List[float] = []
    diversity_values: List[float] = []
    feedback_values: List[float] = []

    for t in range(sim.num_time_steps):
        # Gradually drift feature and score distributions over time.
        drift_scale = 1 + (t / max(1, sim.num_time_steps - 1)) * 0.6  # up to +60%
        feature_counts = [max(1, int(c * drift_scale + rng.randint(-5, 5))) for c in base_feature_dist]
        score_counts = [max(1, int(c * (1 + t * 0.03) + rng.randint(-5, 5))) for c in base_score_dist]

        js_input, psi_input = compute_drift_metrics(base_feature_dist, feature_counts)
        js_scores, psi_scores = compute_drift_metrics(base_score_dist, score_counts)

        input_drifts.append(js_input)
        prediction_drifts.append(js_scores)

        # Simulate increasing cross-model convergence: later steps correlate more.
        model_scores: List[List[float]] = []
        # Shared global trend that gets stronger over time.
        shared_component = [rng.uniform(0.0, 1.0) for _ in range(sim.num_samples_per_step)]
        for _ in range(num_models):
            scores = []
            for sc in shared_component:
                noise = rng.uniform(0.0, 1.0)
                alpha = 0.2 + 0.6 * (t / max(1, sim.num_time_steps - 1))  # from 0.2 to 0.8
                val = alpha * sc + (1 - alpha) * noise
                scores.append(val)
            model_scores.append(scores)

        avg_corr = compute_cross_model_correlation(model_scores)
        cross_model_corrs.append(avg_corr)

        # Exposure diversity: slowly decreases to simulate monoculture.
        exposure_counts = []
        for b in range(sim.num_exposure_bins):
            # Make the first bin more dominant over time.
            base = 100 - 5 * b
            if b == 0:
                base += int(20 * (t / max(1, sim.num_time_steps - 1)))
            count = max(1, base + rng.randint(-10, 10))
            exposure_counts.append(count)
        diversity_index = compute_exposure_diversity(exposure_counts)
        diversity_values.append(diversity_index)

        # Feedback intensity: strengthen over time (e.g., more self-reinforcing).
        past_decisions = []
        future_presence = []
        for _ in range(sim.num_samples_per_step):
            d = 1 if rng.random() < 0.5 else 0
            base_future = 0.5
            # Increase gap between treated and control over time.
            gap = 0.05 + 0.25 * (t / max(1, sim.num_time_steps - 1))
            if d == 1:
                f_prob = min(1.0, base_future + gap)
            else:
                f_prob = max(0.0, base_future - 0.05)
            f = 1 if rng.random() < f_prob else 0
            past_decisions.append(d)
            future_presence.append(f)

        feedback_index = feedback_loop_intensity(past_decisions, future_presence)
        feedback_values.append(feedback_index)

    # Aggregate across time with simple averages.
    def _avg(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    metrics = {
        "avg_input_drift": _avg(input_drifts),
        "avg_prediction_drift": _avg(prediction_drifts),
        "avg_cross_model_corr": _avg(cross_model_corrs),
        "avg_diversity_index": _avg(diversity_values),
        "avg_feedback_intensity_index": _avg(feedback_values),
    }

    return metrics


def run_experiment() -> Tuple[List[Dict[str, object]], str]:
    """Execute baseline and treatment variants for all workloads.

    Returns a tuple of (runs, notes) where `runs` is a list of dictionaries
    conforming to the evidence schema's `runs` entries, and `notes` is a
    human-readable summary string.
    """

    workloads = get_workload_configs()
    sim_config = DEFAULT_SIM_CONFIG

    all_runs: List[Dict[str, object]] = []
    notes_payload: Dict[str, Dict[str, Dict[str, float]]] = {}

    for workload in workloads:
        # Baseline simulation
        baseline_metrics = _simulate_baseline_workload(workload, sim_config)
        baseline_run = {
            "variant": "baseline",
            "workload_id": workload.name,
            "run_id": f"baseline-{workload.name}-0",
            "success": True,
            "metrics": baseline_metrics,
        }
        all_runs.append(baseline_run)

        # Treatment simulation
        treatment_metrics = _simulate_treatment_workload(workload, sim_config)
        treatment_run = {
            "variant": "treatment",
            "workload_id": workload.name,
            "run_id": f"treatment-{workload.name}-0",
            "success": True,
            "metrics": treatment_metrics,
        }
        all_runs.append(treatment_run)

        notes_payload[workload.name] = {
            "baseline": baseline_metrics,
            "treatment": treatment_metrics,
        }

    notes = (
        "Deterministic synthetic simulations were executed for baseline (static) "
        "and treatment (longitudinal) monitoring variants across all workloads. "
        "Metrics are simplified numeric proxies for drift, convergence, "
        "diversity, and feedback intensity. Raw aggregated metrics per workload "
        "and variant: "
        + json.dumps(notes_payload, sort_keys=True)
    )

    return all_runs, notes

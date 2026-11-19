from __future__ import annotations

import itertools
import time
from typing import Any, Dict, List

from .models import build_baseline_portfolio, build_treatment_portfolio
from .scenarios import (
    SCENARIOS,
    WORKLOADS,
    ScenarioConfig,
    WorkloadConfig,
    generate_synthetic_data,
    scenario_bias_for_portfolio,
)
from .metrics import RunMetrics


def run_portfolio_workload_scenario(
    portfolio_id: str,
    workload: WorkloadConfig,
    scenario: ScenarioConfig,
    repeat_index: int,
) -> Dict[str, Any]:
    """Execute a single run and return evidence run record.

    This function is deterministic for given inputs.
    """

    # Build portfolio instance per run to keep it well-scoped
    if portfolio_id == "baseline":
        portfolio = build_baseline_portfolio()
    else:
        portfolio = build_treatment_portfolio()

    # Random seed derived from identifiers for reproducibility
    seed = hash((portfolio_id, workload.workload_id, scenario.scenario_id, repeat_index)) & 0xFFFFFFFF

    features, labels = generate_synthetic_data(seed=seed, workload=workload)

    bias = scenario_bias_for_portfolio(
        portfolio_id=portfolio_id,
        scenario=scenario,
        workload_id=workload.workload_id,
    )

    run_metrics = RunMetrics()

    # We include a very small real timing component but keep it bounded.
    start_time = time.perf_counter()

    for x, y in zip(features, labels):
        pred, simulated_latency_ms = portfolio.predict(workload.workload_id, x, bias)
        run_metrics.update(prediction=pred, truth=y, latency_ms=simulated_latency_ms)

    _elapsed = time.perf_counter() - start_time

    metrics_dict = run_metrics.to_metric_dict()

    run_id = f"{portfolio_id}-{workload.workload_id}-{scenario.scenario_id}-rep{repeat_index}"

    run_record: Dict[str, Any] = {
        "variant": portfolio_id,
        "workload_id": workload.workload_id,
        "run_id": run_id,
        "success": True,
        "metrics": metrics_dict,
        # Non-metric explanatory fields (allowed by schema)
        "scenario_id": scenario.scenario_id,
        "shock_type": scenario.shock_type,
        "shock_intensity": scenario.shock_intensity,
        "repeat_index": repeat_index,
    }

    return run_record


def run_all_experiments() -> List[Dict[str, Any]]:
    """Run a compact but complete matrix of experiments.

    At least one baseline and one treatment run per workload are produced.
    """

    runs: List[Dict[str, Any]] = []

    # For each workload, run all scenarios once for both portfolios.
    # This keeps runtime bounded yet exercises the logic sufficiently.
    repeat_index = 0
    for workload_id, workload in WORKLOADS.items():
        for scenario in SCENARIOS:
            for portfolio_id in ("baseline", "treatment"):
                run_record = run_portfolio_workload_scenario(
                    portfolio_id=portfolio_id,
                    workload=workload,
                    scenario=scenario,
                    repeat_index=repeat_index,
                )
                runs.append(run_record)
        repeat_index += 1

    return runs

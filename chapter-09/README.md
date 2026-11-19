# Experiment 9: Stress-Testing AI Portfolios Under Scenario Shocks

This is a self-contained Python 3.11+ experiment project implementing:

- **Experiment ID**: `exp-9-stress-testing-ai-portfolios-under-scenario-shocks-v1`
- **Chapter**: 9
- **Title**: *Experiment 2: Stress-Testing AI Portfolios Under Scenario Shocks*

The goal is to provide a reproducible, offline-friendly simulation that compares a
**monoculture baseline portfolio** with a **diversified treatment portfolio** under
several shock scenarios. It produces structured evidence in a JSON file suitable
for downstream analysis.

## Overview

The project:

- Simulates four workloads:
  - `wl-credit-risk-batch`
  - `wl-fraud-streaming`
  - `wl-churn-batch`
  - `wl-eligibility-api`
- Defines three scenario "shock" types:
  - `regulatory_shift` (e.g., stricter thresholds / feature restrictions)
  - `market_shift` (distributional change)
  - `vendor_outage` (infrastructure / platform issues)
- Implements simple, deterministic synthetic models for baseline and treatment
  portfolios using **only the Python standard library**.
- Instruments a small set of numeric metrics per run that are actually measured
  from execution (no fabricated performance values).
- Writes experiment evidence to `/evidence/results.json` (or the directory
  specified by the `BOOKMAS_EVIDENCE_DIR` environment variable) following the
  required schema.

This is a **teaching / illustration** setup, not a production-grade stress test.
It focuses on reproducibility, determinism, and schema correctness.

## Repository Layout

- `run.py` – Main entry point. Runs all workloads for baseline and treatment
  portfolios under several shocks, records metrics, and writes evidence.
- `experiment/__init__.py` – Package marker.
- `experiment/models.py` – Synthetic portfolio and model definitions
  (baseline vs. treatment behaviors).
- `experiment/scenarios.py` – Scenario and workload configuration and
  deterministic scenario application logic.
- `experiment/metrics.py` – Metric instrumentation helpers.
- `experiment/runner.py` – Orchestrates running scenarios for portfolios and
  collecting runs.
- `tests/test_experiment.py` – Unit tests that validate the evidence schema and
  basic behavioral expectations.
- `requirements.txt` – Empty of runtime dependencies; standard library only.

## Running the Experiment

From the project root:

```bash
python run.py
```

This will:

1. Detect the evidence directory:
   - Use `$BOOKMAS_EVIDENCE_DIR` if set, otherwise default to `/evidence`.
2. Run a small but complete experiment:
   - For each workload ID, it runs both `baseline` and `treatment` variants
     under a selection of scenario shocks.
   - Multiple runs per workload/variant are possible; each has a unique
     `run_id`.
3. Write `results.json` into the evidence directory.

The evidence file will look like:

```json
{
  "chapter": 9,
  "experiment_id": "exp-9-stress-testing-ai-portfolios-under-scenario-shocks-v1",
  "environment": {
    "os": "...",
    "python": "..."
  },
  "runs": [
    {
      "variant": "baseline",
      "workload_id": "wl-credit-risk-batch",
      "run_id": "...",
      "success": true,
      "metrics": {
        "task_performance_score": 0.73,
        "latency_ms_mean": 0.04
      },
      "scenario_id": "regulatory_shift",
      "shock_intensity": 0.2
    },
    {
      "variant": "treatment",
      "workload_id": "wl-credit-risk-batch",
      "run_id": "...",
      "success": true,
      "metrics": {
        "task_performance_score": 0.78,
        "latency_ms_mean": 0.05
      },
      "scenario_id": "regulatory_shift",
      "shock_intensity": 0.2
    }
  ],
  "notes": "..."
}
```

All metric values are numeric and based on actual code execution
(e.g., simple scoring accuracy and measured wall-clock latency).

## Running Tests

Execute:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The tests:

- Run the experiment in a temporary evidence directory.
- Load the resulting `results.json`.
- Validate the schema requirements from the prompt:
  - `chapter` is an integer (9).
  - `experiment_id` matches the specification.
  - `environment` contains string `os` and `python` fields.
  - `runs` is a list with at least one baseline and one treatment run per
    workload ID.
  - Each run contains:
    - `variant` in {"baseline", "treatment"}
    - `workload_id` as a string
    - Unique `run_id` per workload/variant
    - `success` boolean
    - `metrics` mapping to finite numeric values
- Checks basic determinism: second invocation with same seed behavior yields
  stable metrics (within simple equality checks).

## Design Notes

### Determinism and Fairness

- Randomness is controlled by `random.Random` instances with fixed seeds that
  depend only on portfolio, workload, and scenario identifiers.
- Baseline and treatment are evaluated on the **same synthetic data points**
  for each workload/scenario pair.
- Treatment portfolio differs structurally (different model mix and routing),
  but both portfolios are implemented using the same abstractions.

### Metrics Covered

The full chapter specification mentions many metrics (e.g., recovery time,
correlations). In this offline sandbox we limit ourselves to a small set of
**actually measurable metrics**:

- `task_performance_score` – Simple accuracy over synthetic labels.
- `error_rate` – Fraction of mispredicted samples.
- `latency_ms_mean` – Mean per-sample simulated latency (includes a small
  measured component plus deterministic synthetic overhead).

Other metrics from the book spec that require long-running systems, real
infrastructure, or multi-minute timing windows (e.g., `recovery_time_minutes`,
`service_unavailability_pct`, `portfolio_failure_correlation`) are **omitted**
from the `metrics` objects to avoid fabricating measurements.

We still encode scenario type and intensity at the run level so that additional
metrics could be computed externally if needed.

### Extending the Experiment

To extend this project for richer internal experiments:

- Replace the synthetic models in `experiment/models.py` with real model
  calls or HTTP clients to your services (respecting the no-network
  constraint in this sandboxed context when porting).
- Increase the number of samples per workload to approximate real traffic.
- Introduce more realistic latency and error models in `scenarios.py`.
- Add offline analysis scripts or notebooks that load `results.json` and
  compute additional portfolio-level metrics and executive-ready charts.

This scaffold is deliberately compact and dependency-free to emphasize
reproducibility and ease of adaptation.

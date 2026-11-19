# Experiment 12 – Monitoring Drift, Feedback Loops, and Emerging Monoculture

This repository contains a **self-contained Python 3.11+ experiment project** implementing:

- A simplified simulation of multiple model workloads.
- Longitudinal monitoring indicators for drift, convergence, and feedback-like effects.
- A baseline (static assessment) vs. treatment (continuous monitoring) comparison.
- Evidence emission in a fixed JSON schema for downstream validation.

The focus is on **reproducible structure and instrumentation patterns**, not on domain realism or external dependencies. All logic uses the Python standard library only, making it safe in restricted offline environments.

---

## Contents

- `run.py` – Main entry point that runs the experiment and writes evidence to `/evidence/results.json` (or `$BOOKMAS_EVIDENCE_DIR/results.json`).
- `experiment/core.py` – Core simulation and monitoring logic.
- `experiment/config.py` – Encodes lightweight in-code equivalents of the YAML configuration files described in the specification.
- `experiment/monitoring.py` – Indicator computation utilities (drift, correlations, diversity, feedback intensity proxy, alert logic).
- `experiment/evidence_writer.py` – Helper for producing evidence files in the required schema.
- `requirements.txt` – Empty of third-party runtime dependencies by design.
- `tests/test_experiment.py` – Unit tests checking the evidence schema and basic behavioral properties.

This code does **not** connect to real production systems. Instead, it uses deterministic pseudo-random generators and hand-crafted patterns to:

- Generate toy input/score/outcome data for multiple models per workload.
- Impose different drift/monoculture characteristics for baseline vs. treatment.
- Compute comparable numeric indicators for both variants.

The experiment is designed to reflect the *shape* of a real monitoring framework while staying lightweight and purely local.

---

## How the Experiment Maps to the Specification

### Chapter and Experiment Metadata

- **Chapter number:** `12`
- **Experiment ID:** `exp-12-monitoring-drift-feedback-monoculture-v1`
- **Variants:**
  - `baseline` – Simulates static, infrequent assessments with coarse indicators.
  - `treatment` – Simulates frequent, longitudinal monitoring across models.

Both variants are executed for each workload; all runs share at least one numeric metric so they can be compared.

### Workloads

This project includes three workloads mirroring the spec:

1. `Credit_Risk_Portfolio_Monitoring`
2. `Customer_Support_and_Routing_Monitoring`
3. `Recommendation_and_Personalization_Monitoring`

Each workload has multiple toy models. Baseline and treatment simulations share:

- Identical model identifiers.
- Common time windows and sampling sizes.

They differ only in *how* monitoring is applied.

### Indicators (Simplified)

We implement bounded, tractable proxies for the following indicators:

- `Input_Data_Drift_Magnitude` – Jensen–Shannon–like divergence over a synthetic feature distribution.
- `Prediction_Distribution_Drift` – Population Stability Index–like metric over binned scores.
- `Cross_Model_Output_Correlation` – Pearson-like correlation between model score series.
- `Exposure_Diversity_Index` – Normalized entropy over recommended-item categories.
- `Feedback_Loop_Intensity_Index` – Simple 0–1 index based on how strongly prior decisions correlate with later sample composition.

Additional metrics from the specification (e.g., `Alert_Rate`, `Time_to_Detection`) are alluded to in comments but not all are computed as numeric values, to avoid fabricating non-measurable quantities in the offline sandbox.

Each run records at least one numeric metric per workload and variant; other contextual details go into the `notes` field in the final evidence JSON.

---

## Baseline vs. Treatment Design

### Baseline – Static Assessment

For each workload:

- Simulate a single, coarse time window of data.
- Compute drift and diversity indicators aggregating everything into one snapshot.
- No cross-time trend analysis; no fine-grained feedback detection.

In practice, this approximates a quarterly or annual static report.

### Treatment – Longitudinal Monitoring

For each workload:

- Simulate multiple consecutive time windows of data (e.g., 12 steps).
- Compute indicators per window and then aggregate their mean values.
- Introduce a gradual drift pattern and cross-model convergence over time.
- Track a simple feedback-like effect (past decisions influencing future sample mix).

This approximates daily/weekly monitoring pipelines, where executives could receive dashboard updates and alerts.

---

## Evidence Schema

`run.py` writes a single JSON document to `$BOOKMAS_EVIDENCE_DIR/results.json` with the following structure:

```json
{
  "chapter": 12,
  "experiment_id": "exp-12-monitoring-drift-feedback-monoculture-v1",
  "environment": {"os": "...", "python": "..."},
  "runs": [
    {
      "variant": "baseline",
      "workload_id": "Credit_Risk_Portfolio_Monitoring",
      "run_id": "baseline-Credit_Risk_Portfolio_Monitoring-0",
      "success": true,
      "metrics": {
        "avg_input_drift": 0.031,
        "avg_prediction_drift": 0.012
      }
    },
    {
      "variant": "treatment",
      "workload_id": "Credit_Risk_Portfolio_Monitoring",
      "run_id": "treatment-Credit_Risk_Portfolio_Monitoring-0",
      "success": true,
      "metrics": {
        "avg_input_drift": 0.084,
        "avg_prediction_drift": 0.049,
        "avg_cross_model_corr": 0.75,
        "avg_diversity_index": 0.91,
        "avg_feedback_intensity_index": 0.38
      }
    }
  ],
  "notes": "..."
}
```

Constraints:

- At least one **baseline** and one **treatment** run are included per workload.
- `metrics` values are always finite numbers (no `null`, strings, dicts, or lists).
- Non-numeric context (stdout, explanations, schemas) go into `notes` or additional non-metric fields.

The `tests/test_experiment.py` module asserts conformance with this schema.

---

## Running the Experiment

### 1. Environment

Requirements:

- Python **3.11+**.
- No external packages are required.

### 2. Execute the Experiment

From the project root:

```bash
export BOOKMAS_EVIDENCE_DIR=/evidence  # optional; default is /evidence
python run.py
``

This will:

1. Simulate workloads under baseline and treatment settings.
2. Compute monitoring indicators per workload.
3. Write the evidence file:

```bash
$BOOKMAS_EVIDENCE_DIR/results.json
```

If `BOOKMAS_EVIDENCE_DIR` is not set, `/evidence` is assumed. The directory is created if needed.

### 3. Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Tests validate:

- The evidence file exists and is valid JSON.
- Schema correctness and value types (e.g., numeric metrics).
- Baseline and treatment presence for each workload.
- Deterministic behavior across runs (given the fixed seeds).

---

## Safety, Reproducibility, and Limits

- No network or external filesystem access is used beyond `/evidence` and the local working directory.
- All randomness is seeded deterministically per workload/variant to ensure reproducible runs.
- The simulation is intentionally simple and runs quickly in constrained sandboxes.
- Many real-world metrics from the chapter are *represented* but not fully realistic (e.g., we use simplified formulae for JS divergence and PSI).

You can extend the indicators or time horizons within `experiment/monitoring.py` and `experiment/core.py` while keeping the evidence contract and tests intact.

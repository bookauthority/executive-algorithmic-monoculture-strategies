# Experiment 1: Measuring Model Diversity and Correlated Errors in Your AI Stack

This repository contains a **self-contained, offline Python 3.11+ experiment** implementing:

> **Experiment 1: Measuring Model Diversity and Correlated Errors in Your AI Stack**

It is designed to be a minimal, reproducible template for the specification:

- **experiment_id:** `exp-06-model-diversity-corr-errors-v1`
- **chapter:** `6`
- **Goal:** provide a reproducible framework to quantify diversity and correlated failures across a portfolio of models on a shared task.

This implementation uses **synthetic tabular classification data** and three simple models implemented directly in Python’s standard library (no external ML packages). It focuses on deterministically computing:

- Per-model accuracy
- Pairwise prediction agreement rates
- Pairwise error correlation (Phi coefficient)
- Very simple portfolio diversity summary

Both **baseline** and **treatment** variants are evaluated for each workload, and results are written to `/evidence/results.json` using the exact runtime contract provided.

Because the execution environment is offline and restricted, this project:

- Uses **only the Python standard library** (no network or external dependencies).
- Implements very simple “models” (threshold-based rules and a hand-coded linear model) to stand in for real models in an enterprise AI stack.
- Focuses on illustrating the *experimental pattern* and evidence format, not on model sophistication.

---

## Contents

- `run.py` – main entry point; runs the experiment and writes `/evidence/results.json`.
- `experiment/__init__.py` – package marker.
- `experiment/models.py` – simple deterministic model implementations and wrappers.
- `experiment/data.py` – deterministic synthetic dataset generation and utilities.
- `experiment/metrics.py` – computation of accuracy, agreement, error-correlation, and simple diversity indicators.
- `experiment/runner.py` – orchestrates baseline and treatment variants for each workload and prepares evidence structure.
- `tests/test_experiment.py` – unit tests that validate the evidence schema and core behaviors.
- `requirements.txt` – intentionally empty of runtime dependencies (standard library only).

---

## Installation & Environment

Requirements:

- Python **3.11+**
- No external Python packages are required.

Set up a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install requirements (no-op except ensuring a clean env):

```bash
pip install -r requirements.txt
```

---

## Running the Experiment

From the project root:

```bash
python run.py
```

This will:

1. Generate a synthetic classification dataset representing a single decision domain.
2. Define three deterministic models (`M1`, `M2`, `M3`) acting on the same features.
3. Run **two workloads**:
   - `WL1-core-decision-domain` – main cross-model evaluation.
   - `WL2-segmented-analysis` – simple segment-based analysis (segments derived from a feature threshold).
4. For each workload, run both:
   - **Baseline** variant – evaluates models individually; computes per-model accuracy.
   - **Treatment** variant – evaluates cross-model agreement and error correlation while still logging accuracy.
5. Write evidence to:

   ```text
   /evidence/results.json
   ```

   unless overridden by the `BOOKMAS_EVIDENCE_DIR` environment variable.

The evidence file will follow this schema (plus some optional extra fields):

```json
{
  "chapter": 6,
  "experiment_id": "exp-06-model-diversity-corr-errors-v1",
  "environment": {"os": "...", "python": "..."},
  "runs": [
    {
      "variant": "baseline",
      "workload_id": "WL1-core-decision-domain",
      "run_id": "...",
      "success": true,
      "metrics": {
        "M1_accuracy": 0.85,
        "M2_accuracy": 0.83,
        "M3_accuracy": 0.81
      }
    },
    {
      "variant": "treatment",
      "workload_id": "WL1-core-decision-domain",
      "run_id": "...",
      "success": true,
      "metrics": {
        "M1_accuracy": 0.85,
        "M2_accuracy": 0.83,
        "M3_accuracy": 0.81,
        "M1_M2_agreement": 0.9,
        "M1_M2_error_correlation": 0.6,
        "portfolio_diversity_index": 0.2
      }
    },
    {
      "variant": "baseline",
      "workload_id": "WL2-segmented-analysis",
      "run_id": "...",
      "success": true,
      "metrics": {"M1_accuracy_segment_low": 0.8, "...": 0.0}
    },
    {
      "variant": "treatment",
      "workload_id": "WL2-segmented-analysis",
      "run_id": "...",
      "success": true,
      "metrics": {"M1_M2_agreement_segment_low": 0.9, "...": 0.0}
    }
  ],
  "notes": "Textual summary and limitations."
}
```

Actual metric names and values will be deterministic but depend on the generated dataset and models.

All metrics are **numeric JSON scalars** (no lists, dicts, strings, or nulls).

---

## Workloads Implemented

This template maps the specification’s workloads to concrete, offline-friendly implementations.

### WL1-core-decision-domain

- Synthetic binary classification dataset with features `x1`, `x2`, `x3`.
- Three models:
  - `M1`: Primary model (threshold on linear score).
  - `M2`: Slightly different weightings/threshold, sharing similar structure to `M1`.
  - `M3`: More diverse rule set (nonlinear-ish with an interaction and different thresholds).
- Baseline:
  - Compute accuracy per model on the full dataset.
- Treatment:
  - Compute accuracy per model.
  - Compute pairwise prediction agreement for each model pair.
  - Compute pairwise error correlation (Phi coefficient) for each pair.
  - Compute a simple portfolio diversity index: average pairwise disagreement.

### WL2-segmented-analysis

- Uses the same dataset and models as WL1.
- Defines a simple segmentation based on one feature (e.g., `x1` median split into `segment_low` and `segment_high`).
- Baseline:
  - Compute per-segment accuracy per model.
- Treatment:
  - Compute per-segment accuracy per model.
  - Compute per-segment pairwise agreement and error correlation.
  - Compute per-segment diversity index.

WL3 (shifted/stress eval) is **not included** to keep runtime and code size bounded; this is documented as a limitation in `notes`.

---

## Re-running & Reproducibility

- Synthetic data generation is **fully deterministic** via fixed random seeds.
- Model definitions are pure Python functions with no external randomness.
- The evidence file is overwritten on each run; if you need multiple versions, copy it elsewhere before re-running.

To re-run:

```bash
python run.py
```

To run tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## Extending This Template

To adapt this experiment to a real organizational context:

1. **Replace the synthetic data generator** in `experiment/data.py` with adapters that load your real evaluation datasets.
2. **Replace model implementations** in `experiment/models.py` with wrappers around your production models or vendor APIs, while keeping the shared interface:
   - Input: feature vectors or records.
   - Output: predicted label and, optionally, probability.
3. **Extend metrics** in `experiment/metrics.py` to include additional portfolio-level diversity metrics, clustering, and segment-level indicators as specified in the chapter.
4. Preserve the evidence schema in `run.py` so that governance and leadership teams can easily compare runs across time and domains.

---

## Limitations of This Template

- Models are simple heuristics, not real ML systems.
- No external data sources or APIs are used; everything runs locally and deterministically.
- Advanced visualizations (heatmaps, clustering diagrams) are not generated; this template focuses on the **numerical metrics and evidence contract**.

These limitations are intentional to ensure that the project is safe, bounded, and fully executable in a restricted offline sandbox, while still demonstrating how to structure a robust experiment and evidence record.
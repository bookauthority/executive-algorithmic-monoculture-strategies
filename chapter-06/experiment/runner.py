from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Tuple

from .data import Dataset, generate_synthetic_dataset, segment_indices_by_feature
from .metrics import (
    accuracy,
    disagreement_rate_on_errors,
    error_vector,
    pairwise_agreement,
    phi_correlation,
    portfolio_diversity_index_from_predictions,
    subset_indices,
)
from .models import SimpleModel, get_portfolio_models


def _run_wl1_baseline(dataset: Dataset, models: List[SimpleModel]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for model in models:
        preds = model.predict_batch(dataset.X)
        acc = accuracy(dataset.y, preds)
        metrics[f"{model.model_id}_accuracy"] = float(acc)
    return metrics


def _run_wl1_treatment(dataset: Dataset, models: List[SimpleModel]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    preds_by_model: Dict[str, List[int]] = {}

    for model in models:
        preds = model.predict_batch(dataset.X)
        preds_by_model[model.model_id] = preds
        acc = accuracy(dataset.y, preds)
        metrics[f"{model.model_id}_accuracy"] = float(acc)

    # Pairwise agreement and error correlations.
    model_ids = list(preds_by_model.keys())
    for i in range(len(model_ids)):
        for j in range(i + 1, len(model_ids)):
            m1 = model_ids[i]
            m2 = model_ids[j]
            p1 = preds_by_model[m1]
            p2 = preds_by_model[m2]
            agr = pairwise_agreement(p1, p2)
            e1 = error_vector(dataset.y, p1)
            e2 = error_vector(dataset.y, p2)
            corr = phi_correlation(e1, e2)
            dr = disagreement_rate_on_errors(dataset.y, p1, p2)
            key_prefix = f"{m1}_{m2}"
            metrics[f"{key_prefix}_agreement"] = float(agr)
            metrics[f"{key_prefix}_error_correlation"] = float(corr)
            metrics[f"{key_prefix}_disagreement_rate_on_errors"] = float(dr)

    # Portfolio diversity index.
    diversity = portfolio_diversity_index_from_predictions(preds_by_model)
    metrics["portfolio_diversity_index"] = float(diversity)
    return metrics


def _run_wl2_baseline(dataset: Dataset, models: List[SimpleModel]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    low_idx, high_idx = segment_indices_by_feature(dataset, feature_index=0)

    for model in models:
        preds = model.predict_batch(dataset.X)
        preds_low = subset_indices(preds, low_idx)
        preds_high = subset_indices(preds, high_idx)
        y_low = subset_indices(dataset.y, low_idx)
        y_high = subset_indices(dataset.y, high_idx)

        acc_low = accuracy(y_low, preds_low)
        acc_high = accuracy(y_high, preds_high)
        metrics[f"{model.model_id}_accuracy_segment_low"] = float(acc_low)
        metrics[f"{model.model_id}_accuracy_segment_high"] = float(acc_high)

    return metrics


def _run_wl2_treatment(dataset: Dataset, models: List[SimpleModel]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    low_idx, high_idx = segment_indices_by_feature(dataset, feature_index=0)

    preds_by_model_full: Dict[str, List[int]] = {}
    for model in models:
        preds = model.predict_batch(dataset.X)
        preds_by_model_full[model.model_id] = preds

        preds_low = subset_indices(preds, low_idx)
        preds_high = subset_indices(preds, high_idx)
        y_low = subset_indices(dataset.y, low_idx)
        y_high = subset_indices(dataset.y, high_idx)

        acc_low = accuracy(y_low, preds_low)
        acc_high = accuracy(y_high, preds_high)
        metrics[f"{model.model_id}_accuracy_segment_low"] = float(acc_low)
        metrics[f"{model.model_id}_accuracy_segment_high"] = float(acc_high)

    model_ids = list(preds_by_model_full.keys())
    # Pairwise metrics per segment.
    for segment_name, indices in ("segment_low", low_idx), ("segment_high", high_idx):
        y_seg = subset_indices(dataset.y, indices)
        preds_seg_by_model: Dict[str, List[int]] = {}
        for mid in model_ids:
            preds_seg_by_model[mid] = subset_indices(preds_by_model_full[mid], indices)

        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                m1 = model_ids[i]
                m2 = model_ids[j]
                p1 = preds_seg_by_model[m1]
                p2 = preds_seg_by_model[m2]
                agr = pairwise_agreement(p1, p2)
                e1 = error_vector(y_seg, p1)
                e2 = error_vector(y_seg, p2)
                corr = phi_correlation(e1, e2)
                dr = disagreement_rate_on_errors(y_seg, p1, p2)
                prefix = f"{m1}_{m2}_{segment_name}"
                metrics[f"{prefix}_agreement"] = float(agr)
                metrics[f"{prefix}_error_correlation"] = float(corr)
                metrics[f"{prefix}_disagreement_rate_on_errors"] = float(dr)

        # Segment-level diversity.
        diversity_seg = portfolio_diversity_index_from_predictions(preds_seg_by_model)
        metrics[f"portfolio_diversity_index_{segment_name}"] = float(diversity_seg)

    return metrics


def _make_run_record(variant: str, workload_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
    # UUID4 ensures uniqueness even if tests run multiple times.
    run_id = str(uuid.uuid4())
    return {
        "variant": variant,
        "workload_id": workload_id,
        "run_id": run_id,
        "success": True,
        "metrics": metrics,
    }


def run_all_workloads() -> Tuple[List[Dict[str, Any]], str]:
    """Run baseline and treatment for each workload and return run records.

    Returns (runs, notes).
    """

    models = get_portfolio_models()
    dataset = generate_synthetic_dataset(n=5000)

    runs: List[Dict[str, Any]] = []
    notes_parts: List[str] = []

    # WL1: core decision domain.
    wl1_id = "WL1-core-decision-domain"

    wl1_baseline_metrics = _run_wl1_baseline(dataset, models)
    runs.append(_make_run_record("baseline", wl1_id, wl1_baseline_metrics))

    wl1_treatment_metrics = _run_wl1_treatment(dataset, models)
    runs.append(_make_run_record("treatment", wl1_id, wl1_treatment_metrics))

    notes_parts.append(
        "WL1 computed per-model accuracy and cross-model agreement/error correlation on a synthetic core dataset."
    )

    # WL2: segmented analysis.
    wl2_id = "WL2-segmented-analysis"

    wl2_baseline_metrics = _run_wl2_baseline(dataset, models)
    runs.append(_make_run_record("baseline", wl2_id, wl2_baseline_metrics))

    wl2_treatment_metrics = _run_wl2_treatment(dataset, models)
    runs.append(_make_run_record("treatment", wl2_id, wl2_treatment_metrics))

    notes_parts.append(
        "WL2 computed per-segment accuracy, agreement, error correlation, and diversity based on a median split segment definition."
    )

    # Explicitly document omitted shifted/stress evaluation.
    notes_parts.append(
        "WL3 (shifted/stress dataset) is not implemented in this offline template; no metrics under distribution shift are reported."
    )

    overall_notes = " ".join(notes_parts)
    return runs, overall_notes

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    assert len(y_true) == len(y_pred)
    if not y_true:
        return 0.0
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
    return correct / float(len(y_true))


def pairwise_agreement(pred_a: List[int], pred_b: List[int]) -> float:
    assert len(pred_a) == len(pred_b)
    if not pred_a:
        return 0.0
    same = 0
    for a, b in zip(pred_a, pred_b):
        if a == b:
            same += 1
    return same / float(len(pred_a))


def error_vector(y_true: List[int], y_pred: List[int]) -> List[int]:
    assert len(y_true) == len(y_pred)
    return [1 if yt != yp else 0 for yt, yp in zip(y_true, y_pred)]


def phi_correlation(e1: List[int], e2: List[int]) -> float:
    """Compute Phi coefficient (equivalent to Pearson for binary vars).

    Values are in [-1, 1]. If variance is zero for either vector, returns 0.0
    (no defined correlation in that edge case).
    """

    assert len(e1) == len(e2)
    if not e1:
        return 0.0

    n11 = n10 = n01 = n00 = 0
    for a, b in zip(e1, e2):
        if a == 1 and b == 1:
            n11 += 1
        elif a == 1 and b == 0:
            n10 += 1
        elif a == 0 and b == 1:
            n01 += 1
        else:
            n00 += 1

    n1_ = n10 + n11
    n0_ = n00 + n01
    n_1 = n01 + n11
    n_0 = n00 + n10
    n = n11 + n10 + n01 + n00

    denom = float(n1_ * n0_ * n_1 * n_0)
    if denom == 0.0:
        return 0.0
    num = float(n11 * n00 - n10 * n01)
    return num / (denom ** 0.5)


def disagreement_rate_on_errors(y_true: List[int], pred_a: List[int], pred_b: List[int]) -> float:
    """Proportion of samples where at least one model is wrong and predictions differ."""

    assert len(y_true) == len(pred_a) == len(pred_b)
    if not y_true:
        return 0.0

    count_error_disagree = 0
    count_error_any = 0
    for yt, pa, pb in zip(y_true, pred_a, pred_b):
        err_a = yt != pa
        err_b = yt != pb
        if err_a or err_b:
            count_error_any += 1
            if pa != pb:
                count_error_disagree += 1
    if count_error_any == 0:
        return 0.0
    return count_error_disagree / float(count_error_any)


def portfolio_diversity_index_from_predictions(predictions: Dict[str, List[int]]) -> float:
    """Simple diversity index: average pairwise disagreement rate across models.

    disagreement = 1 - agreement.
    """

    model_ids = list(predictions.keys())
    if len(model_ids) < 2:
        return 0.0

    total_disagreement = 0.0
    count = 0
    for i in range(len(model_ids)):
        for j in range(i + 1, len(model_ids)):
            a = predictions[model_ids[i]]
            b = predictions[model_ids[j]]
            agr = pairwise_agreement(a, b)
            total_disagreement += (1.0 - agr)
            count += 1
    if count == 0:
        return 0.0
    return total_disagreement / float(count)


def subset_indices(values: List[int], indices: Iterable[int]) -> List[int]:
    return [values[i] for i in indices]

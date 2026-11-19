from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


# --------- Utility functions ---------


def _safe_probabilities(counts: Sequence[float]) -> List[float]:
    total = float(sum(counts))
    if total <= 0.0:
        n = len(counts)
        if n == 0:
            return []
        return [1.0 / n for _ in counts]
    return [max(0.0, c) / total for c in counts]


def jensen_shannon_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Compute a bounded Jensen–Shannon divergence between two discrete dists.

    The result is in [0, 1] when using log base 2.
    This is a simplified implementation suitable for monitoring-style drift.
    """

    if len(p) != len(q):
        raise ValueError("p and q must have the same length")

    p_probs = _safe_probabilities(p)
    q_probs = _safe_probabilities(q)
    m = [(pi + qi) / 2.0 for pi, qi in zip(p_probs, q_probs)]

    def _kl_divergence(a: Sequence[float], b: Sequence[float]) -> float:
        s = 0.0
        for ai, bi in zip(a, b):
            if ai <= 0.0 or bi <= 0.0:
                continue
            s += ai * math.log(ai / bi, 2.0)
        return s

    js = 0.5 * _kl_divergence(p_probs, m) + 0.5 * _kl_divergence(q_probs, m)
    # Guard against tiny negative values from floating-point rounding.
    if js < 0.0 and js > -1e-12:
        js = 0.0
    return float(js)


def population_stability_index(
    expected: Sequence[float], actual: Sequence[float]
) -> float:
    """Simplified Population Stability Index.

    PSI = sum((a - e) * ln(a / e)) over bins.
    The inputs are assumed to be counts per bin and are normalized internally.
    Result is non-negative; zero implies identical distributions.
    """

    if len(expected) != len(actual):
        raise ValueError("expected and actual must have the same length")

    e_probs = _safe_probabilities(expected)
    a_probs = _safe_probabilities(actual)

    psi = 0.0
    for e, a in zip(e_probs, a_probs):
        if e <= 0.0 or a <= 0.0:
            continue
        psi += (a - e) * math.log(a / e)

    if psi < 0.0 and psi > -1e-12:
        psi = 0.0
    return float(psi)


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute a simple Pearson correlation coefficient.

    Returns a value in [-1, 1]. For constant vectors, returns 0.0.
    """

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n == 0:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = 0.0
    var_x = 0.0
    var_y = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mean_x
        dy = yi - mean_y
        num += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    if var_x <= 0.0 or var_y <= 0.0:
        return 0.0
    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return 0.0
    r = num / denom
    # Clip to [-1, 1] for numerical safety
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
    return float(r)


def normalized_entropy(counts: Sequence[float]) -> float:
    """Compute entropy normalized to [0, 1].

    A value close to 1 indicates high diversity, 0 indicates a degenerate
    distribution (all mass in a single bin).
    """

    probs = _safe_probabilities(counts)
    n = len(probs)
    if n <= 1:
        return 0.0

    entropy = 0.0
    for p in probs:
        if p <= 0.0:
            continue
        entropy -= p * math.log(p, 2.0)

    max_entropy = math.log(n, 2.0)
    if max_entropy <= 0.0:
        return 0.0
    value = entropy / max_entropy
    if value < 0.0 and value > -1e-12:
        value = 0.0
    if value > 1.0 and value < 1.0 + 1e-12:
        value = 1.0
    return float(value)


def feedback_loop_intensity(
    past_decisions: Sequence[int], future_presence: Sequence[int]
) -> float:
    """Compute a simple 0–1 feedback loop intensity index.

    We model feedback as how strongly a binary decision (e.g., approve vs.
    decline, routed vs. not) predicts whether an entity appears in future
    samples. This is proxied by a normalized difference in conditional
    future-presence rates.

    intensity = max(0, min(1, (rate_if_decision - rate_if_not) / 1.0))

    where rates are in [0, 1]. If denominators are zero, returns 0.0.
    """

    if len(past_decisions) != len(future_presence):
        raise ValueError("Inputs must have the same length")

    treated_present = 0
    treated_total = 0
    control_present = 0
    control_total = 0

    for d, f in zip(past_decisions, future_presence):
        if d:
            treated_total += 1
            if f:
                treated_present += 1
        else:
            control_total += 1
            if f:
                control_present += 1

    if treated_total == 0 or control_total == 0:
        return 0.0

    rate_treated = treated_present / treated_total
    rate_control = control_present / control_total
    diff = rate_treated - rate_control
    if diff < 0.0:
        diff = 0.0
    if diff > 1.0:
        diff = 1.0
    return float(diff)


# --------- Aggregation helpers used by the experiment core ---------


def compute_drift_metrics(
    baseline_counts: Sequence[float], current_counts: Sequence[float]
) -> Tuple[float, float]:
    """Return (js_divergence, psi) for input or prediction distributions."""

    js = jensen_shannon_divergence(baseline_counts, current_counts)
    psi = population_stability_index(baseline_counts, current_counts)
    return js, psi


def compute_cross_model_correlation(
    model_scores: Sequence[Sequence[float]],
) -> float:
    """Compute average pairwise Pearson correlation across models.

    If fewer than two models are present, returns 0.0.
    """

    num_models = len(model_scores)
    if num_models < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(num_models):
        for j in range(i + 1, num_models):
            r = pearson_correlation(model_scores[i], model_scores[j])
            total += r
            count += 1
    if count == 0:
        return 0.0
    return float(total / count)


def compute_exposure_diversity(exposure_counts: Sequence[float]) -> float:
    """Wrapper around normalized_entropy for readability."""

    return normalized_entropy(exposure_counts)

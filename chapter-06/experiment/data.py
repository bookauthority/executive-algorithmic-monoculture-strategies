from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


RANDOM_SEED = 12345


@dataclass(frozen=True)
class Dataset:
    X: List[List[float]]  # features
    y: List[int]  # binary labels 0/1

    def size(self) -> int:
        return len(self.X)


def _latent_score(x1: float, x2: float, x3: float) -> float:
    """Hidden ground-truth scoring function.

    Combines linear and nonlinear elements so that true decision boundary is
    not exactly aligned with any single simple model.
    """

    base = 1.2 * x1 + 0.8 * x2 + 0.5 * x3
    interaction = 0.7 * x1 * x3
    curved = 0.5 * (x2 - 0.5) ** 2
    return base + interaction - curved


def generate_synthetic_dataset(n: int = 5000) -> Dataset:
    """Generate a deterministic synthetic dataset.

    Features x1, x2, x3 are uniform in [0, 1]. Labels are derived from a
    latent score with a deterministic threshold, plus a tiny, deterministic
    perturbation based on index to avoid perfect separability.
    """

    rnd = random.Random(RANDOM_SEED)
    X: List[List[float]] = []
    y: List[int] = []
    for i in range(n):
        x1 = rnd.random()
        x2 = rnd.random()
        x3 = rnd.random()
        score = _latent_score(x1, x2, x3)
        # Deterministic perturbation: alternate sign by index parity.
        noise = 0.05 * math.sin(i * 0.01)
        label = 1 if score + noise >= 1.5 else 0
        X.append([x1, x2, x3])
        y.append(label)
    return Dataset(X=X, y=y)


def segment_indices_by_feature(dataset: Dataset, feature_index: int = 0) -> Tuple[List[int], List[int]]:
    """Split dataset indices into two segments based on a feature median.

    Returns (indices_low, indices_high) where:
    - indices_low: indices with feature <= median
    - indices_high: indices with feature > median
    """

    values = [row[feature_index] for row in dataset.X]
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0 if len(sorted_vals) % 2 == 0 else sorted_vals[mid]

    low: List[int] = []
    high: List[int] = []
    for i, v in enumerate(values):
        if v <= median:
            low.append(i)
        else:
            high.append(i)
    return low, high

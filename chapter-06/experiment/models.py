from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class SimpleModel:
    """A simple deterministic classification model.

    Models operate on fixed-length feature vectors [x1, x2, x3] and return
    binary predictions {0, 1}. These stand in for real-world models in an
    enterprise stack.

    The models are intentionally simple but slightly different so that
    agreement and error correlations are non-trivial.
    """

    model_id: str
    weights: Sequence[float]
    bias: float
    threshold: float
    nonlinear: bool = False

    def predict_batch(self, X: List[List[float]]) -> List[int]:
        preds: List[int] = []
        for row in X:
            x1, x2, x3 = row
            if self.nonlinear:
                # A simple nonlinear-ish rule: include an interaction and
                # piecewise component to induce some diverse behavior.
                score = (
                    self.weights[0] * x1
                    + self.weights[1] * x2
                    + self.weights[2] * x3
                    + 0.3 * x1 * x2
                )
                if x3 > 0.7:
                    score += 0.5
                score += self.bias
            else:
                score = (
                    self.weights[0] * x1
                    + self.weights[1] * x2
                    + self.weights[2] * x3
                    + self.bias
                )
            pred = 1 if score >= self.threshold else 0
            preds.append(pred)
        return preds


def get_portfolio_models() -> List[SimpleModel]:
    """Return a list of models M1, M2, M3.

    - M1 and M2 share similar linear structure but differ in weights/thresholds.
    - M3 uses a nonlinear flag to behave differently, adding diversity.
    """

    m1 = SimpleModel(
        model_id="M1", weights=[0.8, 0.6, 0.4], bias=-0.3, threshold=0.5, nonlinear=False
    )
    m2 = SimpleModel(
        model_id="M2", weights=[0.7, 0.7, 0.3], bias=-0.1, threshold=0.6, nonlinear=False
    )
    m3 = SimpleModel(
        model_id="M3", weights=[0.5, 0.2, 0.9], bias=-0.2, threshold=0.7, nonlinear=True
    )
    return [m1, m2, m3]

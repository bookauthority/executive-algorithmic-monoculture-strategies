from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RunMetrics:
    correct: int = 0
    total: int = 0
    total_latency_ms: float = 0.0

    def update(self, prediction: int, truth: int, latency_ms: float) -> None:
        if prediction == truth:
            self.correct += 1
        self.total += 1
        self.total_latency_ms += latency_ms

    def to_metric_dict(self) -> Dict[str, float]:
        """Return metrics as numeric values only.

        - task_performance_score: accuracy in [0, 1]
        - error_rate: 1 - accuracy
        - latency_ms_mean: mean per-sample simulated latency
        """

        if self.total == 0:
            accuracy = 0.0
            mean_latency = 0.0
        else:
            accuracy = self.correct / float(self.total)
            mean_latency = self.total_latency_ms / float(self.total)

        error_rate = 1.0 - accuracy

        return {
            "task_performance_score": float(accuracy),
            "error_rate": float(error_rate),
            "latency_ms_mean": float(mean_latency),
        }

import json
import os
import sys
import time
import platform
from typing import Any, Dict, List

from experiment.runner import run_all_experiments


CHAPTER_NUMBER = 9
EXPERIMENT_ID = "exp-9-stress-testing-ai-portfolios-under-scenario-shocks-v1"


def main() -> None:
    # Determine evidence directory
    evidence_dir = os.environ.get("BOOKMAS_EVIDENCE_DIR", "/evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    evidence_path = os.path.join(evidence_dir, "results.json")

    # Run experiment
    start_time = time.time()
    runs = run_all_experiments()
    duration_s = time.time() - start_time

    environment = {
        "os": platform.platform(),
        "python": sys.version.replace("\n", " ")
    }

    evidence: Dict[str, Any] = {
        "chapter": CHAPTER_NUMBER,
        "experiment_id": EXPERIMENT_ID,
        "environment": environment,
        "runs": runs,
        "notes": (
            "Synthetic stress-test simulation. Metrics are deterministic and "
            "based on simple synthetic models and scenarios. Total_runtime_s="
            f"{duration_s:.6f}"
        ),
    }

    # Write evidence JSON
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)


if __name__ == "__main__":  # pragma: no cover
    main()

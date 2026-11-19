import json
import os
import platform
from typing import Any, Dict, List

from experiment.runner import run_all_workloads


CHAPTER_NUMBER = 6
EXPERIMENT_ID = "exp-06-model-diversity-corr-errors-v1"


def main() -> None:
    # Execute experiment workloads for baseline and treatment.
    runs, summary_notes = run_all_workloads()

    evidence_dir = os.environ.get("BOOKMAS_EVIDENCE_DIR", "/evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    evidence_path = os.path.join(evidence_dir, "results.json")

    evidence: Dict[str, Any] = {
        "chapter": CHAPTER_NUMBER,
        "experiment_id": EXPERIMENT_ID,
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
        },
        "runs": runs,
        "notes": summary_notes,
    }

    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

import json
import os
import platform
from pathlib import Path

from experiment.core import run_experiment
from experiment.evidence_writer import write_evidence


def main() -> None:
    """Run baseline and treatment variants for all workloads and write evidence.

    The evidence file is written to `$BOOKMAS_EVIDENCE_DIR/results.json`,
    defaulting to `/evidence/results.json` if the environment variable is unset.
    """

    chapter = 12
    experiment_id = "exp-12-monitoring-drift-feedback-monoculture-v1"

    # Run the core experiment logic which returns structured run results.
    runs, extra_notes = run_experiment()

    evidence_dir = os.environ.get("BOOKMAS_EVIDENCE_DIR", "/evidence")
    evidence_path = Path(evidence_dir) / "results.json"

    environment_info = {
        "os": platform.platform(),
        "python": platform.python_version(),
    }

    # Aggregate evidence according to the required schema.
    evidence = {
        "chapter": chapter,
        "experiment_id": experiment_id,
        "environment": environment_info,
        "runs": runs,
    }
    if extra_notes:
        evidence["notes"] = extra_notes

    write_evidence(evidence_path, evidence)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_evidence(path: Path, evidence: Dict[str, Any]) -> None:
    """Write evidence JSON to the given path, ensuring parent directory exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)

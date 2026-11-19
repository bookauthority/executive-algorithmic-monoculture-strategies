import json
import os
import unittest
from pathlib import Path

import run  # noqa: F401  (imported to ensure module is loadable)
from experiment.core import run_experiment
from experiment.config import get_workload_configs
from experiment.evidence_writer import write_evidence


class TestExperimentEvidenceSchema(unittest.TestCase):
    def setUp(self) -> None:
        # Use a temporary directory under /tmp for test isolation.
        self.tmp_dir = Path("/tmp/bookmas_exp12_tests")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_path = self.tmp_dir / "results.json"

    def _generate_evidence(self) -> None:
        # Directly use the experiment core and writer, mirroring run.py behavior.
        chapter = 12
        experiment_id = "exp-12-monitoring-drift-feedback-monoculture-v1"

        runs, notes = run_experiment()
        evidence = {
            "chapter": chapter,
            "experiment_id": experiment_id,
            "environment": {"os": "test-os", "python": "0.0.0"},
            "runs": runs,
        }
        if notes:
            evidence["notes"] = notes

        write_evidence(self.evidence_path, evidence)

    def test_evidence_file_schema_and_metrics(self) -> None:
        self._generate_evidence()
        self.assertTrue(self.evidence_path.exists(), "Evidence file was not created")

        with self.evidence_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Top-level structure
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("chapter"), 12)
        self.assertEqual(
            data.get("experiment_id"),
            "exp-12-monitoring-drift-feedback-monoculture-v1",
        )
        self.assertIn("environment", data)
        self.assertIn("runs", data)

        env = data["environment"]
        self.assertIsInstance(env, dict)
        self.assertIn("os", env)
        self.assertIn("python", env)

        runs = data["runs"]
        self.assertIsInstance(runs, list)
        self.assertGreater(len(runs), 0)

        # Workload coverage and variant presence
        workloads = {w.name for w in get_workload_configs()}
        seen = {w: set() for w in workloads}

        for run_entry in runs:
            self.assertIsInstance(run_entry, dict)
            variant = run_entry.get("variant")
            self.assertIn(variant, {"baseline", "treatment"})

            workload_id = run_entry.get("workload_id")
            self.assertIn(workload_id, workloads)

            run_id = run_entry.get("run_id")
            self.assertIsInstance(run_id, str)
            self.assertTrue(run_id)

            success = run_entry.get("success")
            self.assertIsInstance(success, bool)

            metrics = run_entry.get("metrics")
            self.assertIsInstance(metrics, dict)
            self.assertGreater(len(metrics), 0)

            # Every metric value must be a finite JSON number (int or float).
            for key, value in metrics.items():
                self.assertIsInstance(
                    value,
                    (int, float),
                    msg=f"Metric {key} has non-numeric value {value!r}",
                )

            seen[workload_id].add(variant)

        # Ensure each workload has both baseline and treatment variants.
        for w, variants in seen.items():
            self.assertIn(
                "baseline",
                variants,
                msg=f"Workload {w} missing baseline variant in evidence",
            )
            self.assertIn(
                "treatment",
                variants,
                msg=f"Workload {w} missing treatment variant in evidence",
            )

    def test_run_experiment_deterministic_and_numeric(self) -> None:
        # Call run_experiment twice and expect deterministic metrics
        runs1, notes1 = run_experiment()
        runs2, notes2 = run_experiment()

        self.assertEqual(len(runs1), len(runs2))
        for r1, r2 in zip(runs1, runs2):
            self.assertEqual(r1["variant"], r2["variant"])
            self.assertEqual(r1["workload_id"], r2["workload_id"])
            self.assertEqual(r1["run_id"], r2["run_id"])
            self.assertEqual(r1["success"], r2["success"])
            self.assertEqual(r1["metrics"], r2["metrics"])

            for key, value in r1["metrics"].items():
                self.assertIsInstance(
                    value,
                    (int, float),
                    msg=f"Metric {key} has non-numeric value {value!r}",
                )

        # Notes should be deterministic as well.
        self.assertEqual(notes1, notes2)


if __name__ == "__main__":
    unittest.main()

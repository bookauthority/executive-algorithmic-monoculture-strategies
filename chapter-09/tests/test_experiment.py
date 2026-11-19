import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List

import run as experiment_run


class TestExperimentEvidenceSchema(unittest.TestCase):
    def setUp(self) -> None:
        # Isolate evidence directory for each test
        self.tmpdir = tempfile.mkdtemp(prefix="evidence_")
        os.environ["BOOKMAS_EVIDENCE_DIR"] = self.tmpdir

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.environ.pop("BOOKMAS_EVIDENCE_DIR", None)

    def _load_results(self) -> Dict[str, Any]:
        path = os.path.join(self.tmpdir, "results.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_run_produces_valid_schema(self) -> None:
        # Execute the experiment
        experiment_run.main()

        # Load results
        data = self._load_results()

        # Top-level checks
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("chapter"), 9)
        self.assertEqual(
            data.get("experiment_id"),
            "exp-9-stress-testing-ai-portfolios-under-scenario-shocks-v1",
        )

        env = data.get("environment")
        self.assertIsInstance(env, dict)
        self.assertIsInstance(env.get("os"), str)
        self.assertIsInstance(env.get("python"), str)

        runs = data.get("runs")
        self.assertIsInstance(runs, list)
        self.assertGreater(len(runs), 0)

        # Track presence of baseline and treatment per workload
        workloads_seen: Dict[str, Dict[str, bool]] = {}

        for run in runs:
            self.assertIsInstance(run, dict)

            variant = run.get("variant")
            self.assertIn(variant, ("baseline", "treatment"))

            workload_id = run.get("workload_id")
            self.assertIsInstance(workload_id, str)

            run_id = run.get("run_id")
            self.assertIsInstance(run_id, str)

            success = run.get("success")
            self.assertIsInstance(success, bool)

            metrics = run.get("metrics")
            self.assertIsInstance(metrics, dict)
            self.assertGreater(len(metrics), 0)

            for m_name, m_value in metrics.items():
                self.assertIsInstance(m_name, str)
                # Must be a number (int/float) and finite
                self.assertIsInstance(m_value, (int, float))
                self.assertTrue(m_value == m_value)  # not NaN

            # Record presence of variant for this workload
            info = workloads_seen.setdefault(workload_id, {"baseline": False, "treatment": False})
            info[variant] = True

        # Ensure at least one baseline and one treatment run per workload
        for workload_id, flags in workloads_seen.items():
            self.assertTrue(
                flags["baseline"], msg=f"No baseline run for workload {workload_id}"
            )
            self.assertTrue(
                flags["treatment"], msg=f"No treatment run for workload {workload_id}"
            )

    def test_deterministic_metrics_for_same_seed(self) -> None:
        # First run
        experiment_run.main()
        data1 = self._load_results()

        # Second run in same environment directory should overwrite file
        experiment_run.main()
        data2 = self._load_results()

        # Compare runs by (variant, workload_id, scenario_id, repeat_index)
        def index_runs(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
            index: Dict[str, Dict[str, Any]] = {}
            for run in data["runs"]:
                key = (
                    f"{run.get('variant')}|{run.get('workload_id')}|"
                    f"{run.get('scenario_id')}|{run.get('repeat_index')}"
                )
                index[key] = run
            return index

        idx1 = index_runs(data1)
        idx2 = index_runs(data2)

        self.assertEqual(set(idx1.keys()), set(idx2.keys()))

        for key in idx1:
            m1 = idx1[key]["metrics"]
            m2 = idx2[key]["metrics"]
            self.assertEqual(m1, m2, msg=f"Metrics differ between runs for key {key}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

import json
import os
import unittest


class TestExperimentEvidenceSchema(unittest.TestCase):
    """Validate that running the experiment produces evidence with the required schema.

    These tests use only the Python standard library and do not assume any
    external dependencies. They execute run.py indirectly by importing and
    calling its main entry function so that we can control the environment
    directory used for evidence output.
    """

    def setUp(self) -> None:
        # Ensure a clean evidence directory under /tmp for tests.
        self.test_evidence_dir = "/tmp/test_evidence"
        if os.path.exists(self.test_evidence_dir):
            # Remove existing files but keep directory.
            for name in os.listdir(self.test_evidence_dir):
                path = os.path.join(self.test_evidence_dir, name)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except OSError:
                    pass
        else:
            os.makedirs(self.test_evidence_dir, exist_ok=True)
        os.environ["BOOKMAS_EVIDENCE_DIR"] = self.test_evidence_dir

    def _load_results(self):
        results_path = os.path.join(self.test_evidence_dir, "results.json")
        self.assertTrue(os.path.exists(results_path), "results.json must be created")
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_run_produces_valid_evidence(self):
        # Import and run the main experiment entrypoint.
        import run as experiment_run

        experiment_run.main()

        data = self._load_results()

        # Top-level keys.
        self.assertIn("chapter", data)
        self.assertIn("experiment_id", data)
        self.assertIn("environment", data)
        self.assertIn("runs", data)

        self.assertIsInstance(data["chapter"], int)
        self.assertEqual(data["chapter"], 6)

        self.assertIsInstance(data["experiment_id"], str)
        self.assertEqual(data["experiment_id"], "exp-06-model-diversity-corr-errors-v1")

        env = data["environment"]
        self.assertIsInstance(env, dict)
        self.assertIn("os", env)
        self.assertIn("python", env)
        self.assertIsInstance(env["os"], str)
        self.assertIsInstance(env["python"], str)

        runs = data["runs"]
        self.assertIsInstance(runs, list)
        self.assertGreaterEqual(len(runs), 4, "At least four runs (two workloads x baseline/treatment) expected")

        # Track presence of baseline and treatment per workload.
        variants_by_workload = {}

        for run in runs:
            self.assertIn("variant", run)
            self.assertIn("workload_id", run)
            self.assertIn("run_id", run)
            self.assertIn("success", run)
            self.assertIn("metrics", run)

            variant = run["variant"]
            workload_id = run["workload_id"]

            self.assertIn(variant, ("baseline", "treatment"))
            self.assertIsInstance(workload_id, str)
            self.assertIsInstance(run["run_id"], str)
            self.assertIsInstance(run["success"], bool)

            metrics = run["metrics"]
            self.assertIsInstance(metrics, dict)
            # Every metric value must be a finite JSON number (int or float).
            self.assertGreater(len(metrics), 0, "Each run must report at least one metric")
            for key, value in metrics.items():
                self.assertIsInstance(key, str)
                # Must be int or float, not bool (bool is subclass of int, so check explicitly).
                self.assertFalse(isinstance(value, bool), "Metric values must not be booleans")
                self.assertIsInstance(value, (int, float))

            variants_by_workload.setdefault(workload_id, set()).add(variant)

        # Ensure at least one baseline and one treatment run per workload.
        for workload_id, variants in variants_by_workload.items():
            self.assertIn("baseline", variants, f"Workload {workload_id} must have a baseline run")
            self.assertIn("treatment", variants, f"Workload {workload_id} must have a treatment run")

    def test_metrics_non_empty_and_consistent(self):
        import run as experiment_run

        experiment_run.main()

        data = self._load_results()
        runs = data["runs"]

        # Sanity: ensure some expected metric names appear and are numeric.
        metric_keys = set()
        for run in runs:
            metric_keys.update(run["metrics"].keys())

        # Expect some core metrics for WL1.
        self.assertTrue(any(k.startswith("M1_accuracy") for k in metric_keys))
        self.assertTrue(any("agreement" in k for k in metric_keys), "Agreement metrics should be present in treatment runs")
        self.assertTrue(any("portfolio_diversity_index" in k for k in metric_keys))


if __name__ == "__main__":
    unittest.main()

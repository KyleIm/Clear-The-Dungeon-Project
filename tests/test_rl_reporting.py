import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rl.reporting import EpisodeStats, ExperimentStore, Telemetry, generate_report


class TestRLReporting(unittest.TestCase):
    def test_episode_aggregation(self):
        stats = EpisodeStats()
        for floor, outcome in [(1, "death"), (5, "deadlock"), (100, "clear"), (5, "timeout")]:
            stats.add({"floor": floor, "outcome": outcome, "target_success": floor >= 5,
                       "reward": 0.5, "actions": 4, "deadlock_mimic_counter": outcome == "timeout"})
        summary = stats.summary()
        self.assertEqual(summary["episodes"], 4)
        self.assertEqual(summary["median_floor"], 5)
        self.assertEqual(summary["clear_rate"], 0.25)
        self.assertEqual(summary["deadlock_rate"], 0.25)
        self.assertEqual(summary["confirmed_deadlock_rate"], 0.5)
        self.assertEqual(summary["timeout_rate"], 0.25)
        self.assertEqual(summary["target_success_rate"], 0.75)
        self.assertEqual(summary["survival"][5], 0.75)
        self.assertEqual(summary["survival"][6], 0.25)

    def test_store_and_report(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("Matplotlib unavailable")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "test.sqlite3"
            with ExperimentStore(database, batch_size=2) as store:
                for step in [4, 8, 12]:
                    store.episode({"step": step, "stage": 1, "split": "train", "floor": 2,
                                   "outcome": "death", "target_success": True, "reward": 1, "actions": 4})
                    store.update({"step": step, "stage": 1, "episodes": step // 4, "elapsed": step / 2,
                                  "throughput": 2.0, "policy_loss": 0.1, "value_loss": 0.2,
                                  "phase_steps": {"combat": 4}, "phase_entropy": {"combat": 0.6},
                                  "telemetry": {"cpu_percent": 32.0}})
                store.discard_after(8)
                stats = EpisodeStats()
                stats.add({"floor": 2, "outcome": "death", "target_success": True, "reward": 1, "actions": 4})
                for policy in ["ppo", "random"]:
                    store.evaluation(stats.summary() | {"step": 8, "stage": 1, "policy": policy})
                store.log_event({"step": 8, "stage": 1, "event": "checkpoint"})
                marks = store.watermarks()
                store.log_event({"step": 8, "stage": 1, "event": "rollback this"})
                store.restore_watermarks(marks)
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0], 2)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)
            output = root / "report"
            report = generate_report(database, output)
            self.assertIn("Uniform Random Policy", report.read_text(encoding="utf-8"))
            for name in ("learning_curves", "floor_survival", "head_statistics", "hardware_metrics"):
                self.assertGreater((output / (name + ".png")).stat().st_size, 200)

    def test_telemetry_shape(self):
        # No subprocess is needed to validate metric availability and stale-value semantics.
        from unittest.mock import patch

        with patch("rl.reporting.shutil.which", return_value=None):
            telemetry = Telemetry(min_interval=60)
        row = telemetry.sample()
        self.assertIn("cpu_percent", row)
        self.assertIn("ram_used_mib", row)
        self.assertIsNone(row["gpu_percent"])
        self.assertEqual(row, telemetry.sample())


if __name__ == "__main__":
    unittest.main()

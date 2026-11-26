from contextlib import closing
from pathlib import Path
import tempfile
import unittest

from experiments.random_baseline import experiment_seeds
from experiments.random_baseline_large import batch_jobs, connect, run, summarize


class LargeBaselineTests(unittest.TestCase):
    def test_seeds_are_independent_of_batches_and_unique(self):
        jobs = batch_jobs(100, 0, 100, 314159)
        self.assertEqual(jobs, batch_jobs(100, 0, 37, 314159) + batch_jobs(100, 1, 37, 314159)
                         + batch_jobs(100, 2, 37, 314159))
        self.assertEqual(len({seed for row in jobs for seed in row[1:]}), 200)
        self.assertNotEqual(jobs[0], experiment_seeds(1, 314159)[0])

    def test_pause_resume_preserves_completed_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "run"
            settings = {"games": 12, "workers": 2, "batch_size": 4, "master_seed": 314159,
                        "action_cap": 10000, "timeout_seconds": 30}
            self.assertFalse(run(target, settings, max_batches=1))
            with closing(connect(target)) as db:
                before = dict(db.execute("select game_id, runtime_s from episodes"))
                self.assertEqual(len(before), 4)
            self.assertTrue(run(target, settings))
            with closing(connect(target)) as db:
                self.assertEqual(db.execute("pragma integrity_check").fetchone()[0], "ok")
                after = dict(db.execute("select game_id, runtime_s from episodes"))
                self.assertEqual(len(after), 12)
                self.assertEqual({key: after[key] for key in before}, before)
                self.assertEqual(len(summarize(db)["floors"]), 100)
                self.assertEqual(db.execute("select sum(wins+losses+unresolved)=sum(encounters) from episode_monsters").fetchone()[0], 1)
            with self.assertRaises(ValueError):
                run(target, settings | {"master_seed": 2})

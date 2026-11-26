import sqlite3
import unittest
from unittest.mock import patch

from experiments.random_baseline import episode, experiment_seeds, save_results, tables
from simulator.game import Game


class RandomBaselineTests(unittest.TestCase):
    def test_seed_plan_is_unique_and_repeatable_for_both_streams(self):
        seeds = experiment_seeds(1000, 314159)
        self.assertEqual(seeds, experiment_seeds(1000, 314159))
        self.assertEqual(len({value for game in seeds for value in game[1:]}), 2000)

    def test_replay_is_deterministic_except_runtime(self):
        game = Game()
        a = episode(game, (1, 9, 8), 5000, 30)
        b = episode(game, (1, 9, 8), 5000, 30)
        self.assertEqual(a[:-2], b[:-2])
        self.assertEqual(a[-1], b[-1])

    def test_action_cap_is_a_timeout_and_keeps_partial_progress(self):
        result = episode(Game(), (2, 9, 8), 1, 30)
        self.assertEqual(result.outcome, "timeout")
        self.assertEqual(result.actions, 1)
        self.assertEqual(result.floor, 1)
        self.assertEqual(result.detail, "action_limit")

    def test_wall_limit_keeps_a_timeout_out_of_normal_defeats(self):
        with patch("experiments.random_baseline.perf_counter", side_effect=[0, 30, 30]):
            result = episode(Game(), (2, 9, 8), 100, 30)
        self.assertEqual(result.outcome, "timeout")
        self.assertEqual(result.detail, "wall_limit")
        self.assertEqual(result.actions, 0)

    def test_engine_exception_is_an_error_with_attempted_action(self):
        game = Game()
        with patch.object(game, "step", side_effect=ValueError("test only")):
            result = episode(game, (3, 9, 8), 5000, 30)
        self.assertEqual(result.outcome, "error")
        self.assertEqual(result.actions, 1)
        self.assertIn("ValueError", result.detail)

    def test_sqlite_keeps_censored_episodes_out_of_death_counts(self):
        records = [episode(Game(), (1, 9, 8), 5000, 30), episode(Game(), (2, 10, 11), 1, 30)]
        with sqlite3.connect(":memory:") as db:
            save_results(db, {"games": 2}, records)
            summary = tables(db)
        self.assertEqual(len(summary["floor_survival"]), 100)
        self.assertEqual(sum(row["reached"] for row in summary["floor_distribution"]), 2)
        self.assertEqual(sum(row["deaths"] for row in summary["death_monsters"]),
                         sum(row.outcome == "game_over" for row in records))

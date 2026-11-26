import importlib.util
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                         ('gymnasium', 'torch', 'matplotlib')), 'RL dependencies required')
class RunnerTests(unittest.TestCase):
    def test_training_validation_resume_and_report(self):
        from rl.runner import validate

        result = validate()
        self.assertEqual(result['steps'], 8)
        self.assertEqual(result['stage'], 2)
        self.assertTrue(result['report'])

    def test_split_seeds(self):
        from rl.runner import seed_for

        seeds = {seed_for(7, split, index) for split in ('game', 'validation', 'test')
                 for index in range(10)}
        self.assertEqual(len(seeds), 30)

    def test_both_policy_modes_reuse_game_seeds_and_stochastic_draws(self):
        from contextlib import closing
        import json
        from pathlib import Path
        import sqlite3
        from tempfile import TemporaryDirectory
        import torch

        from rl.environment import DungeonEnv
        from rl.reporting import ExperimentStore
        from rl.runner import SHOP_CONFIG, evaluate, model_for, read_config

        cfg = read_config(SHOP_CONFIG)
        cfg['workers'] = 1
        cfg['evaluation'].update(validation_episodes=2, action_limit=4)
        cfg['ppo']['hidden_dims'] = [8]
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = model_for(DungeonEnv(cfg['curriculum'][0], shop_policy='strategic'), cfg, 'cpu')
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'paired.sqlite3'
            with ExperimentStore(path) as store:
                for step in (10, 20):
                    evaluate(cfg, model, 1, store, split='validation', step=step, elapsed=0,
                             train_episodes=0, device='cpu', policies=('ppo_deterministic', 'ppo_stochastic'))
            with closing(sqlite3.connect(path)) as connection:
                def outcomes(step, policy):
                    rows = connection.execute('SELECT payload FROM episodes WHERE step=? AND policy=?', (step, policy))
                    return {row['seed']: (row['floor'], row['actions'], row['outcome'], row['last_actions'])
                            for (raw,) in rows if (row := json.loads(raw))}

                sampling = outcomes(10, 'ppo_stochastic')
                self.assertEqual(len(sampling), 2)
                self.assertEqual(sampling, outcomes(20, 'ppo_stochastic'))
                self.assertEqual(sampling.keys(), outcomes(10, 'ppo_deterministic').keys())

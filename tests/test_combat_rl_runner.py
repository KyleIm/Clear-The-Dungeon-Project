from contextlib import closing
from copy import deepcopy
import importlib.util
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch', 'matplotlib', 'numpy')),
                     'Combat/Reward training dependencies required')
class RunnerTests(unittest.TestCase):
    def test_cached_critic_only_updates_pending_transitions(self):
        from combat_rl.ppo import Transition
        from combat_rl.runner import fill_bootstraps

        rollout = [Transition(None, 0, None, 0, 0.0, 0.0, 1.0, 0.0, done, slot)
                   for done, slot in ((False, 0), (True, 1), (False, 2))]
        pending = {0: 0, 2: 2}
        fill_bootstraps(rollout, pending, {1: 100.0, 2: 7.0, 0: 5.0})
        self.assertFalse(pending)
        self.assertEqual([step.next_value for step in rollout], [5.0, 0.0, 7.0])

    def test_cpu_runtime_grouping_and_critic_reuse_preserve_update(self):
        import torch

        from combat_rl.runner import DEFAULT, config, load, train

        cfg = config(DEFAULT)
        cfg.update(episodes=4, workers=2, device='cpu', action_cap=3, rollout_rounds=3,
                   eval_every_episodes=20, eval_episodes=1, cpu_threads=1, hidden_dims=[8])
        cfg['ppo'].update(epochs=1, minibatch=8)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = train(cfg, root / 'baseline', benchmark=True)
            optimized = dict(cfg, runtime={'processes': 1, 'reuse_critic': True})
            second = train(optimized, root / 'optimized', benchmark=True)
            for key in ('episodes', 'steps', 'decisions', 'updates'):
                self.assertEqual(first[key], second[key])
            reference = load(root / 'baseline' / 'latest.pt')['model']
            result = load(root / 'optimized' / 'latest.pt')['model']
            for key in reference:
                torch.testing.assert_close(reference[key], result[key], rtol=1e-6, atol=1e-7)

    def test_tiny_parallel_run_resume_and_paired_evaluation(self):
        from combat_rl.runner import DEFAULT, config, load, train

        cfg = config(DEFAULT)
        cfg.update(episodes=3, workers=2, device='cpu', action_cap=2, rollout_rounds=1,
                   checkpoint_updates=1, eval_every_episodes=3, eval_episodes=2, cpu_threads=1, hidden_dims=[8])
        cfg['ppo'].update(epochs=1, minibatch=4)
        cfg['runtime'] = {'processes': 1, 'reuse_critic': True}
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.assertTrue(train(cfg, directory).is_file())
            first = load(directory / 'latest.pt')
            self.assertEqual(first['state']['episodes'], 3)
            extended = deepcopy(cfg)
            extended['episodes'] = 4
            self.assertTrue(train(extended, directory, resume=True).is_file())
            last = load(directory / 'latest.pt')
            self.assertEqual(last['state']['episodes'], 4)
            self.assertEqual(last['state']['steps'], 8)
            self.assertGreater(last['state']['updates'], first['state']['updates'])
            with closing(sqlite3.connect(directory / 'training.sqlite3')) as db:
                self.assertEqual(dict(db.execute('SELECT split, COUNT(*) FROM episodes GROUP BY split')),
                                 {'train': 4, 'eval': 4, 'random': 4})

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
    def test_tiny_parallel_run_resume_and_paired_evaluation(self):
        from combat_rl.runner import DEFAULT, config, load, train

        cfg = config(DEFAULT)
        cfg.update(episodes=3, workers=2, device='cpu', action_cap=2, rollout_rounds=1,
                   checkpoint_updates=1, eval_every_episodes=3, eval_episodes=2, cpu_threads=1, hidden_dims=[8])
        cfg['ppo'].update(epochs=1, minibatch=4)
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

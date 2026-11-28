from contextlib import closing
import importlib.util
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch','numpy','matplotlib')),
                     'Alternating learner dependencies required')
class AlternatingRunnerTests(unittest.TestCase):
    def test_exact_schedule(self):
        from alternating_rl.runner import segment

        schedule = [segment(i * 25_000, 25_000) for i in range(40)]
        self.assertEqual(sum(target == 'combat_reward' for _, _, target in schedule), 20)
        self.assertEqual(schedule[-1], (39, 20, 'shop'))
        self.assertEqual(segment(24_999, 25_000), (0, 1, 'combat_reward'))
        self.assertEqual(segment(25_000, 25_000), (1, 1, 'shop'))

    def test_resume_active_slots_and_freeze_combat_in_shop_segment(self):
        import torch

        from combat_rl.ppo import ActorCritic
        from combat_rl.runner import DEFAULT as COMBAT_CONFIG, config
        from alternating_rl.runner import DEFAULT, Experiment, read_config, train

        cfg = read_config(DEFAULT)
        cfg.update(episodes=4, segment_episodes=2, eval_episodes=1, workers=2, processes=1,
                   device='cpu', action_cap=3, rollout_rounds=1, checkpoint_rounds=1)
        initial = ActorCritic([128, 128])
        optimizer = torch.optim.Adam(initial.parameters())
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, run = root / 'pretrained.pt', root / 'experiment'
            torch.save(dict(version=1, state={'episodes': 1_000_000}, config=config(COMBAT_CONFIG),
                            model=initial.state_dict(), optimizer=optimizer.state_dict()), source)
            original_save = Experiment.save

            def interrupt(self, pool, archive=False):
                original_save(self, pool, archive)
                if self.state['rounds'] == 1 and not archive:
                    raise KeyboardInterrupt

            with patch.object(Experiment, 'save', interrupt), self.assertRaises(KeyboardInterrupt):
                train(cfg, run, source)
            pending = torch.load(run / 'latest.pt', weights_only=False)
            self.assertEqual(len(pending['workers']), 2)
            self.assertEqual(pending['state']['episodes'], 0)
            self.assertTrue(train(cfg, run, root / 'unused.pt', resume=True).is_file())
            first, final = [torch.load(run / f'episode-{step:07d}.pt', weights_only=False) for step in (2, 4)]
            self.assertEqual(first['state']['version'], final['state']['version'])
            self.assertEqual(final['schedule'], {'segment': 2, 'cycle': 1, 'target': 'complete'})
            for key in first['model']:
                torch.testing.assert_close(first['model'][key], final['model'][key], rtol=0, atol=0)
            with closing(sqlite3.connect(run / 'training.sqlite3')) as db:
                self.assertEqual(dict(db.execute('SELECT split,count(*) FROM episodes GROUP BY split')),
                                 {'train':4, 'eval':3})

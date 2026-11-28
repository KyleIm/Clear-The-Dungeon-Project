import sqlite3
import unittest
from random import Random

import numpy as np
import torch

from alternating_rl.dqn import ShopDQN, bellman_targets, update_dqn
from alternating_rl.replay import Replay


def observation(head, mask):
    return (np.array([1, 2], dtype=np.float32), head, np.asarray(mask, dtype=np.bool_))


def example(cycle=1, version=1, *, terminal=False, reward=0):
    return dict(state=observation(0, [True, False, True]), action=2,
                next_state=None if terminal else observation(1, [False, True, True]),
                reward=reward, terminal=terminal, cycle=cycle, version=version)


class DQNTests(unittest.TestCase):
    def models(self):
        online = ShopDQN(2, (3, 3, 2), (4,))
        target = ShopDQN(2, (3, 3, 2), (4,))
        with torch.no_grad():
            for model in (online, target):
                for parameter in model.parameters():
                    parameter.zero_()
            online.heads[1].bias.copy_(torch.tensor([99.0, 0.7, 0.5]))
            target.heads[1].bias.copy_(torch.tensor([99.0, 0.3, 0.8]))
        return online, target

    def test_cross_head_double_target_and_terminal(self):
        online, target = self.models()
        rows = [example(), example(terminal=True, reward=1), example(terminal=True, reward=0)]
        # Online picks legal action 1 in the next head. The target's own best is action 2.
        torch.testing.assert_close(bellman_targets(online, target, rows), torch.tensor([0.3, 1.0, 0.0]))
        metrics = update_dqn(online, target, torch.optim.Adam(online.parameters()), rows)
        self.assertEqual(metrics['dqn_terminal'], 2)
        self.assertGreater(metrics['dqn_loss'], 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in target.parameters()))

    def test_act_masks_greedy_and_exploration(self):
        online, _ = self.models()
        obs = observation(1, [False, True, True])
        rng = Random(9)
        state = rng.getstate()
        greedy, = online.act([obs], rng, 0.0)
        self.assertEqual((greedy[0], greedy[2]), (1, False))
        self.assertEqual(rng.getstate(), state)
        sampled = online.act([obs] * 30, rng, 1.0)
        self.assertEqual({action[0] for action in sampled}, {1, 2})
        self.assertTrue(all(action[2] for action in sampled))


class ReplayTests(unittest.TestCase):
    def test_scope_mask_round_trip_and_rollback(self):
        with sqlite3.connect(':memory:') as db:
            replay = Replay(db, capacity=4, previous_cycles=1)
            replay.append(example(cycle=1, version=1))
            replay.append(example(cycle=2, version=3, terminal=True, reward=1))
            replay.append(example(cycle=3, version=4))
            mark = replay.mark()
            db.commit()
            replay.append(example(cycle=3, version=5))
            replay.restore(mark)
            rows = replay.sample(10, Random(3), current_cycle=3, current_version=4)
            self.assertEqual({(row['cycle'], row['version']) for row in rows}, {(2, 3), (3, 4)})
            self.assertEqual(replay.mark(), mark)
            for row in rows:
                np.testing.assert_array_equal(row['state'][2], [True, False, True])
                if not row['terminal']:
                    np.testing.assert_array_equal(row['next_state'][2], [False, True, True])

    def test_capacity_and_pruning_leave_newer_uncommitted_rows(self):
        with sqlite3.connect(':memory:') as db:
            replay = Replay(db, capacity=3, previous_cycles=1)
            for version in range(1, 8):
                replay.append(example(cycle=2, version=version))
            mark = replay.mark()
            replay.append(example(cycle=2, version=8))
            replay.prune(mark)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM shop_replay').fetchone()[0], 4)
            self.assertEqual({row['version'] for row in replay.sample(9, Random(2), 2, 8)}, {6, 7, 8})
            replay.restore(mark)
            self.assertEqual({row['version'] for row in replay.sample(9, Random(2), 2, 7)}, {5, 6, 7})


if __name__ == '__main__':
    unittest.main()

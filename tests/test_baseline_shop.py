import random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from baseline_shop.configurations import configurations, replay
from baseline_shop.environment import Run, compare_visit, frozen_policy
from baseline_shop.model import ConfigurationValue, fit
from baseline_shop.storage import Store
from combat_rl.ppo import ActorCritic
from combat_rl.runner import batch
import torch


class BaselineShopTests(unittest.TestCase):
    def shop(self):
        run = Run(17, 42, 10000, mode='paired')
        game = run.game
        game.floor = run.floor = 10
        game.phase = 'shop'
        game.gold = 1500
        run.state = game.get_state()
        return run

    def test_candidates_replay_and_do_not_touch_entry_or_rng(self):
        run = self.shop()
        original = run.game.get_state()
        rng = run.game.rng.getstate()
        for head in range(3):
            candidates = configurations(run.game, head, random.Random(42), 8, 'Party', None)
            self.assertEqual(candidates[0].actions, ())
            self.assertTrue(any(c.actions for c in candidates))
            for c in candidates:
                game = replay(run.game, c.actions)
                self.assertEqual(game.rng.getstate(), rng)
                self.assertEqual(c.features.shape, (1370,))
                if head != 0:
                    self.assertEqual(game.gold, original['gold'])
                if head != 1:
                    self.assertEqual(game.player.base_attack, original['player']['base_attack'])
                if head != 2:
                    for name in original['player']['current_skills']:
                        if name != 'lock':
                            self.assertIn(name, game.player.current_skills)
            self.assertEqual(run.game.get_state(), original)
            self.assertEqual(run.game.rng.getstate(), rng)

    def test_unchanged_pairs_match_and_do_not_move_live_game(self):
        run = self.shop()
        def policy(observations):
            return np.ones((len(observations), 7))
        before = run.game.get_state()
        result = compare_visit(run, ((), (), (), ()), policy, budget=8)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(r == result[0] for r in result))
        self.assertEqual(run.game.get_state(), before)

    def test_reward_source_changes_only_on_reward_phase(self):
        run = self.shop()
        run.game.phase = 'reward'
        run.state = run.game.get_state()
        legal = run._decision_actions()
        # This test inspects the selection boundary without executing a synthetic reward state.
        run.explore_reward = 1.0
        selected, source = run.select_action(legal[0][0])
        self.assertIn(selected, dict(legal))
        self.assertEqual(source, 'random')
        run.game.phase = 'combat'
        run.state = run.game.get_state()
        chosen = run._decision_actions()[0][0]
        self.assertEqual(run.select_action(chosen), (chosen, 'ppo'))

    def test_counterfactual_inference_matches_masked_ppo(self):
        model = ActorCritic([128, 128])
        run = Run(17, 42, 10000, mode='paired')
        observation = run.observation()
        run.game._offer_reward()
        run.state = run.game.get_state()
        cases = [observation, run.observation()]
        weights = {name: value.numpy() for name, value in model.state_dict().items()}
        with torch.inference_mode():
            expected = model(*model._inputs(batch(cases)))[0].probs.numpy()
        np.testing.assert_allclose(frozen_policy(weights)(cases), expected, rtol=1e-5, atol=1e-7)

    def test_gold_fit_uses_balanced_legal_examples_and_resume_discards_tail(self):
        run = self.shop()
        feature = configurations(run.game, 0, random.Random(42), 8, 'Party', None)[1].features
        with TemporaryDirectory() as folder, Store(Path(folder) / 'test.sqlite3') as store:
            store.identity('sample', False)
            for label in (-1, 1, None, 0):
                store.comparison(1, 7, 0, feature, dict(label=label, noop=label == 0))
            examples = store.sample(7, 0, 12, 8, random.Random(17))
            self.assertEqual(sorted(y for _, y in examples), [-1]*6+[1]*6)
            self.assertEqual(store.sample(8, 0, 12, 8, random.Random(17)), [])
            snapshot = store.marks()
            store.comparison(2, 7, 0, feature, dict(label=0, noop=False))
            store.restore(snapshot)
            self.assertEqual(store.marks(), snapshot)
        model = ConfigurationValue()
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        before = model.heads[0].weight.detach().clone()
        self.assertGreater(fit(model, opt, 0, examples), 0)
        self.assertFalse(torch.equal(before, model.heads[0].weight))


if __name__ == '__main__':
    unittest.main()

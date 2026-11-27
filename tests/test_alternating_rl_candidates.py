import importlib.util
from copy import deepcopy
from random import Random
import unittest


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy required')
class CandidateTests(unittest.TestCase):
    def shop(self):
        from simulator.game import Game

        game = Game()
        game.reset(11)
        game.floor = 10
        game.gold = 10_000
        game._open_shop()
        game.player.learned_skills.extend(name for name in ('recovery', 'defend', 'heal')
                                         if name not in game.player.learned_skills)
        return game

    def test_671_schema_keeps_exhausted_skill_and_clears_locked_slots(self):
        from alternating_rl.candidates import FEATURE_COUNT, encode
        from combat_rl.environment import scale

        game = self.shop()
        skill = game.player.current_skills[0]
        game.player.memorized_skills[skill] = 0
        features = encode(game, previous_area='Forest', event_choice='A')
        self.assertEqual((FEATURE_COUNT, features.size), (671, 671))
        self.assertEqual(features.dtype.name, 'float32')
        self.assertEqual((features[26], features[27]), (1, 0))
        self.assertAlmostEqual(features[100], scale(game.player.skill_costs[skill], 10))
        self.assertFalse(any(features[26 + 4 * 75:26 + 6 * 75]))
        names = sorted(game.data.skills, key=lambda name: game.data.skills[name]['id'])
        for reserved in ('strike', 'sacrifice'):
            index = names.index(reserved)
            self.assertEqual(features[476 + index], 0)
            self.assertEqual(features[564 + index], 0)
        self.assertEqual(tuple(features[-13:-2]), tuple(name == 'Forest' for name in game.data.regions))
        self.assertEqual(tuple(features[-2:]), (1, 0))

    def test_candidates_replay_legally_and_do_not_touch_live_rng(self):
        from alternating_rl.candidates import encode, generate, summarize

        game = self.shop()
        before = game.get_state()
        engine_rng, reroll_rng = game.rng.getstate(), game.reroll_rng.getstate()
        kwargs = dict(previous_area='Forest', event_choice=None)
        plans = generate(game, 24, Random(12), max_edits=12, **kwargs)
        self.assertEqual(plans[0].actions, (('leave_shop',),))
        self.assertGreater(len(plans), 6)
        self.assertLessEqual(len(plans), 24)
        self.assertIn('buy', {action[0] for plan in plans for action in plan.actions})
        self.assertIn('reduce_skill_cost', {action[0] for plan in plans for action in plan.actions})
        self.assertEqual(game.get_state(), before)
        self.assertEqual(game.rng.getstate(), engine_rng)
        self.assertEqual(game.reroll_rng.getstate(), reroll_rng)
        self.assertEqual(len({plan.features.tobytes() for plan in plans}), len(plans))
        for plan in plans:
            trial = deepcopy(game, {id(game.data): game.data})
            self.assertLessEqual(len(plan.actions), 3 * 12 + 1)
            for action in plan.actions[:-1]:
                trial.step(action)
            self.assertEqual(summarize(trial, **kwargs), plan.exit_state)
            self.assertEqual(encode(trial, **kwargs).tobytes(), plan.features.tobytes())
            self.assertEqual(trial.rng.getstate(), engine_rng)
            trial.step(plan.actions[-1])
            self.assertEqual(trial.floor, 11)

    def test_zero_edits_require_no_spending_or_allocation(self):
        from alternating_rl.candidates import generate

        game = self.shop()
        game.player.available_points = 20
        plan, = generate(game, 10, Random(0), max_edits=0)
        self.assertEqual(plan.actions, (('leave_shop',),))
        self.assertEqual(plan.exit_state['unused_stat_points'], 20)
        self.assertEqual(plan.exit_state['gold'], 10_000)

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy required for the RL encoder')
class CombatRLEnvironmentTests(unittest.TestCase):
    def test_features_masks_and_deactivated_blocks(self):
        import numpy as np

        from combat_rl.environment import Run

        run = Run(24, 32, 100)
        features, phase, mask = run.observation()
        self.assertEqual((features.shape, features.dtype, phase), ((583,), np.float32, 0))
        self.assertEqual(mask.tolist(), [True, True, True, True, False, False, False])
        self.assertEqual(features[509:582].sum(), 0)
        self.assertEqual(features[497:508].sum(), 1)
        for skill in run.game.player.memorized_skills:
            run.game.player.memorized_skills[skill] = 0
        run.state = run.game.get_state()
        self.assertEqual(run.observation()[2].tolist(), [False, False, False, False, False, False, True])
        run.game._offer_reward()
        run.game.reward_skill = 'sacrifice'
        run.state = run.game.get_state()
        features, phase, mask = run.observation()
        self.assertEqual(phase, 1)
        self.assertEqual(mask.tolist(), [True, True, True, True, True, False, False])
        self.assertFalse(features[26:52].any())
        self.assertEqual(features[581], 1)  # Category 99 is the 65th distinct category.

    def test_uniform_random_reproduces_baseline(self):
        from combat_rl.environment import Run
        from experiments.random_baseline import episode
        from simulator.game import Game

        for seed in (0, 12, 24):
            historical = episode(Game(), (0, seed, 991), 50, float('inf'))
            run = Run(seed, 991, 50)
            while not run.done:
                run.step(run.random_action())
            actual = run.result()
            for field in ('floor', 'actions', 'outcome', 'detail'):
                self.assertEqual(actual[field], getattr(historical, field))
            self.assertEqual(actual['reward'], actual['floor'] - 1)

    def test_public_shop_replay_matches_original_choices_and_exit(self):
        import json

        from combat_rl.environment import Run
        from experiments.random_baseline import episode
        from experiments.shop_exit_replay import ShopExits
        from simulator.game import Game

        class TraceGame(Game):
            def step(self, action):
                if self.phase == 'shop':
                    shop_actions.append({'floor': self.floor, 'action': list(action)})
                return super().step(action)

        # Game 323671 in docs/random_replay_1000_verification.json.
        pair = (2538817474026044487, 3909549678122254972)
        shop_actions = []
        observer = ShopExits()
        historical = episode(TraceGame(), (323671, *pair), 100, float('inf'), observer)
        self.assertEqual((historical.floor, historical.actions, historical.outcome), (12, 83, 'game_over'))
        run = Run(*pair, 100)
        while not run.done:
            run.step(run.random_action())
        result = run.result()
        self.assertEqual((result['floor'], result['actions'], result['outcome']), (12, 83, 'game_over'))
        self.assertEqual(result['shop_actions'], shop_actions)
        self.assertEqual(len(result['shop_exits']), 1)
        actual, expected = result['shop_exits'][0], observer.rows[0]
        for field in ('unused_stat_points', 'max_mana', 'used_mana', 'remaining_mana', 'entry_gold', 'exit_gold'):
            self.assertEqual(actual[field], expected[field])
        self.assertEqual([[slot['skill'], slot['uses'], slot['mana_cost']] for slot in actual['slots']],
                         json.loads(expected['slots']))

    def test_shop_bridge_records_the_exit_and_includes_its_floor_reward(self):
        from random import Random

        from combat_rl.environment import Run

        run = Run(24, 32, 100)
        run.game.floor = run.floor = 10
        run.game._offer_reward()
        run.state = run.game.get_state()
        # Seed 5 chooses the fifth entry (Leave) at the shop main menu.
        run.policy_rng = Random(5)
        self.assertEqual(run.step(0), 1)
        self.assertEqual(run.actions, 2)
        self.assertEqual(run.shop_actions, [{'floor': 10, 'action': ['leave_shop']}])
        self.assertEqual(len(run.shop_exits), 1)
        exit_record = run.shop_exits[0]
        self.assertEqual(exit_record['used_mana'] + exit_record['remaining_mana'], exit_record['max_mana'])
        self.assertEqual(run.reward_actions, {'Gold': 1})

    def test_store_event_choice_and_respect_primitive_action_cap(self):
        from random import Random
        from unittest.mock import patch

        from combat_rl.environment import Run

        run = Run(24, 32, 2)
        run.game.floor = run.floor = 35
        run.state = run.game.get_state()

        def defeat_monster(game, *_):
            game.monster.hp = 0

        with patch('simulator.game.combat.resolve_turn', side_effect=defeat_monster):
            self.assertEqual(run.step(0), 0)
        chosen = Random(32).choice([['ella', 'pickpocket'], ['ella', 'help']])
        result = run.result()
        self.assertEqual(result['event_actions'], [{'floor': 35, 'action': chosen}])
        self.assertEqual(result['actions'], 2)
        self.assertEqual(result['timeout_phase'], 'reward')
        self.assertEqual(result['detail'], 'action_limit')

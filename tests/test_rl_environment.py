"""Small deterministic checks for the RL adapter around the unmodified game."""
from copy import deepcopy
import importlib.util
import random
import unittest


@unittest.skipUnless(importlib.util.find_spec("gymnasium"), "Install the RL requirements")
class DungeonEnvironmentTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from rl.environment import DungeonEnv

        self.np = np
        self.make_env = DungeonEnv
        self.stage = {"target": "next_floor", "weights": {"target": 2.0, "floor": 0.5}}

    def test_actions_match_original_order_and_transitions(self):
        from simulator.game import Game

        env = self.make_env(self.stage)
        raw = Game()
        state = raw.reset(42)
        obs, _ = env.reset(seed=42)
        policy = random.Random(981)
        for _ in range(24):
            self.assertTrue(env.observation_space.contains(obs))
            legal = raw.get_legal_actions()
            ids = env.legal_ids_in_engine_order()
            self.assertEqual(set(ids), set(self.np.flatnonzero(obs["action_mask"])))
            self.assertEqual(len(ids), len(legal))
            index = policy.randrange(len(legal))
            state = raw.step(legal[index])
            obs, _, terminated, truncated, _ = env.step(ids[index])
            self.assertEqual(env.game.get_state(), state)
            self.assertEqual(terminated, raw.is_terminal())
            self.assertFalse(truncated)
            if terminated:
                break

    def test_hidden_state_never_changes_features_or_mask(self):
        env = self.make_env(self.stage)
        obs, _ = env.reset(seed=42)
        env.game.remaining_areas.reverse()
        env.game.ella_skills[:] = ["sacrifice"]
        env.game.reward_candidates[:] = ["counter"]
        enemy = env.game.monster
        enemy.monster_id = -1
        enemy.selected_action = enemy.ram_action = "counter"
        enemy.available_skills[:] = ["counter"]
        enemy.imitation_backup = {"hidden": True}
        enemy.base_attack += 123
        env.game.rng.random()
        changed = env.observe(env.game.get_state())
        self.np.testing.assert_array_equal(changed["features"], obs["features"])
        self.np.testing.assert_array_equal(changed["action_mask"], obs["action_mask"])

    def test_substitute_hides_body_hp_and_status(self):
        env = self.make_env(self.stage)
        env.reset(seed=31)
        env.game.monster_substitute.hp = 7
        visible = env.observe(env.game.get_state())
        env.game.monster.hp = 3
        env.game.monster.apply_status_effect("poison")
        hidden = env.observe(env.game.get_state())
        self.np.testing.assert_array_equal(hidden["features"], visible["features"])

    def test_shop_masks_match_the_engine_and_ignore_hidden_bookkeeping(self):
        env = self.make_env(self.stage)
        env.reset(seed=31)
        game = env.game
        game.phase = "shop"
        game.shop_menu = "merchandise"
        game.gold = 5000
        game.player.type = game.player.base_type = "Beast"
        game.player.EVA = game.player.base_EVA = 1
        env._state = game.get_state()
        merchandise = env.legal_ids_in_engine_order()
        self.assertIn(env.action_catalog.index(("reroll",)), merchandise)
        self.assertIn(env.action_catalog.index(("buy", "blue_key")), merchandise)
        before = env.observe(env._state)
        game.player.base_EVA = 0
        game.player.base_type = "Undead"
        after = env.observe(game.get_state())
        self.np.testing.assert_array_equal(before["action_mask"], after["action_mask"])
        self.np.testing.assert_array_equal(before["features"], after["features"])
        for menu in ("skills", "costs", "stats"):
            game.shop_menu = menu
            env._state = game.get_state()
            obs = env.observe(env._state)
            ids = env.legal_ids_in_engine_order()
            self.assertEqual(set(ids), set(self.np.flatnonzero(obs["action_mask"])))

    def test_milestone_reward_keeps_episode_active(self):
        env = self.make_env(self.stage)
        env.reset(seed=4)
        env.game.phase = "reward"
        env.game.reward_skill = None
        env._state = env.game.get_state()
        obs = env.observe(env._state)
        action = env.action_catalog.index(("reward", "Gold"))
        self.assertTrue(obs["action_mask"][action])
        _, reward, terminal, truncated, info = env.step(action)
        self.assertEqual(reward, 2.5)
        self.assertFalse(terminal or truncated)
        self.assertTrue(info["target_success"])

    def test_reward_exchange_does_not_expose_candidate_pool(self):
        env = self.make_env(self.stage)
        env.reset(seed=4)
        env.game.phase = "reward"
        env.game.reward_skill = None
        before = env.observe(env.game.get_state())
        env.game.reward_candidates = ["thunder", "counter"]
        after = env.observe(env.game.get_state())
        self.np.testing.assert_array_equal(before["features"], after["features"])
        self.np.testing.assert_array_equal(before["action_mask"], after["action_mask"])
        self.assertTrue(after["action_mask"][env.action_catalog.index(("reward", "Exchange"))])

    def test_deadlock_cut_is_optional_and_not_a_normal_death(self):
        for early, steps, reason in ((True, 1, "deadlock"), (False, 2, "timeout")):
            with self.subTest(early_deadlock=early):
                env = self.make_env(self.stage, action_limit=2, early_deadlock=early)
                env.reset(seed=17)
                env.game.monster = env.game.monster_roster[21]
                player = env.game.player
                player.current_skills = ["counter"] + ["lock"] * 5
                player.memorized_skills = {"counter": 3}
                env._state = env.game.get_state()
                obs = env.observe(env._state)
                self.assertEqual(self.np.flatnonzero(obs["action_mask"]).size, 1)
                action = int(self.np.flatnonzero(obs["action_mask"])[0])
                before = deepcopy(player.memorized_skills)
                for step in range(steps):
                    obs, _, terminated, truncated, info = env.step(action)
                    self.assertFalse(terminated)
                    self.assertEqual(truncated, step + 1 == steps)
                self.assertEqual(info["outcome"], reason)
                self.assertTrue(info["deadlock_mimic_counter"])
                self.assertEqual(player.memorized_skills, before)
                self.assertEqual(self.np.flatnonzero(obs["action_mask"]).size, 1)

import importlib.util
import unittest

if importlib.util.find_spec('gymnasium'):
    from rl.environment import DungeonEnv


@unittest.skipUnless(importlib.util.find_spec('gymnasium'), 'RL dependencies required')
class StrategicShopTests(unittest.TestCase):
    def shop(self, *, epsilon=0.0001, limit=100):
        env = DungeonEnv({'target': 'clear', 'shop_repeat_epsilon': epsilon},
                         shop_policy='strategic', action_limit=limit)
        env.game._open_shop()
        env._state = env.game.get_state()
        return env

    def act(self, env, descriptor):
        return env.step(env.action_catalog.index(descriptor))

    def test_repetition_and_parent_visit_counts(self):
        env = self.shop()
        for i in range(2):
            _, reward, _, _, info = self.act(env, ('shop_strategy', 'skills'))
            self.assertAlmostEqual(reward, -0.0001 * i)
            self.assertEqual(info['shop_selection']['repeat'], bool(i))
            _, reward, _, _, info = self.act(env, ('draft_keep',))
            self.assertEqual(reward, 0)
        _, reward, _, _, info = self.act(env, ('shop_strategy', 'leave'))
        self.assertEqual(reward, 0)
        self.assertEqual(info['shop_visit']['strategic_actions'], 3)
        self.assertEqual(info['shop_visit']['engine_actions'], 1)
        self.assertEqual(info['shop_visit']['penalty'], -0.0001)
        self.assertTrue(info['shop_visit']['left'])
        env.game._open_shop()
        env._state = env.game.get_state()
        _, reward, _, _, info = self.act(env, ('shop_strategy', 'skills'))
        self.assertEqual(reward, 0)

    def test_skill_order_can_swap_with_spare_and_charges_primitive_edits(self):
        env = self.shop()
        p = env.game.player
        original = p.current_skills.copy()
        spare = next(skill for skill in env.catalog_skills if skill not in p.learned_skills)
        p.learned_skills.append(spare)
        env._state = env.game.get_state()
        self.act(env, ('shop_strategy', 'skills'))
        pair = [original[1], original[0], *original[2:4]]
        for skill in pair:
            self.act(env, ('draft_skill', skill))
        for count in (1, 0, 0, 0):
            _, reward, _, _, info = self.act(env, ('draft_use', count))
            self.assertEqual(reward, 0)
        self.assertEqual(p.current_skills[:4], pair)
        self.assertEqual(p.memorized_skills[pair[0]], 1)
        self.assertGreaterEqual(info['engine_steps'], 5)
        self.assertEqual(info['shop_change']['after']['mana_used'], p.skill_costs[pair[0]])

    def test_stat_draft_keeps_points_and_timeout_closes_visit(self):
        env = self.shop()
        p = env.game.player
        total = p.base_attack + p.base_defense + p.base_hp + p.base_speed + p.available_points
        self.act(env, ('shop_strategy', 'stats'))
        _, _, _, _, info = self.act(env, ('draft_stat_max',))
        for _ in range(3):
            _, _, _, _, info = self.act(env, ('draft_stat_max',))
        self.assertEqual(info['shop_change']['after']['stats']['attack'], total - 3)
        self.assertEqual(p.available_points, 0)

        capped = self.shop(epsilon=0, limit=1)
        _, reward, _, truncated, info = self.act(capped, ('shop_strategy', 'stats'))
        self.assertTrue(truncated)
        self.assertEqual(reward, 0)
        self.assertFalse(info['shop_visit']['left'])
        self.assertEqual(info['shop_visit']['outcome'], 'timeout')

    def test_each_paid_feature_has_its_own_first_free_selection(self):
        env = self.shop()
        p = env.game.player
        env.game.gold = 10_000
        env._state = env.game.get_state()
        initial_features = env.observe(env._state)['features']
        _, reward, _, _, _ = self.act(env, ('shop_strategy', 'skills'))
        self.assertEqual(reward, 0)
        _, reward, _, _, _ = self.act(env, ('draft_keep',))
        self.assertEqual(reward, 0)
        self.assertFalse((initial_features == env.observe(env._state)['features']).all())

        for domain, first, second in (
            ('costs', ('reduce_skill_cost', p.current_skills[0]),
             ('reduce_skill_cost', p.current_skills[1])),
            ('merchandise', ('buy', 'blue_mana_potion'), ('buy', 'elemental_mana_potion')),
        ):
            for repeat, action in enumerate((first, second)):
                _, reward, _, _, info = self.act(env, ('shop_strategy', domain))
                self.assertEqual(reward, -0.0001 * repeat)
                self.assertEqual(info['engine_steps'], 0)
                _, reward, _, _, info = self.act(env, action)
                self.assertEqual(reward, 0)
                self.assertEqual(info['engine_steps'], 3)
                self.assertGreater(info['shop_change']['gold_spent'], 0)

    def test_zero_penalty_preserves_repetition_counters(self):
        env = self.shop(epsilon=0)
        for i in range(2):
            _, reward, _, _, info = self.act(env, ('shop_strategy', 'stats'))
            self.assertEqual(reward, 0)
            self.assertEqual(info['shop_selection']['repeat'], bool(i))
            self.act(env, ('draft_keep',))
        _, _, _, _, info = self.act(env, ('shop_strategy', 'leave'))
        self.assertEqual(info['shop_visit']['repeated'], 1)
        self.assertEqual(info['shop_visit']['penalty'], 0)

    def test_exact_stat_configuration_with_digits(self):
        env = self.shop()
        p = env.game.player
        targeted = [77, 90, 110, 59]
        self.act(env, ('shop_strategy', 'stats'))
        for wanted in targeted:
            digits = str(wanted - 1).zfill(len(str(env.strategic_shop.points_left)))
            for digit in digits:
                self.act(env, ('draft_stat_digit', int(digit)))
        self.assertEqual([p.base_attack, p.base_defense, p.base_hp, p.base_speed], targeted)
        self.assertEqual(p.available_points, 400 - sum(targeted))

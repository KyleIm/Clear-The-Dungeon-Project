import unittest
from random import Random

import numpy as np

from alternating_rl.shop import BUDGETS, FEATURE_COUNT, FINISH, GOLD, HEAD_SIZES, SKILLS, STATS, Shop
from simulator.game import Game


class OrderedShopTests(unittest.TestCase):
    def game(self):
        game = Game()
        game.reset(17)
        game.floor = 10
        game._open_shop()
        return game

    def test_catalog_and_exact_mask_with_split_gold_menus(self):
        game = self.game()
        game.gold = 9000
        game.player.base_EVA = 8
        game.player.type, game.player.base_type = 'Elemental', 'Beast'
        adapter = Shop(game, 'Forest', None)
        self.assertEqual(BUDGETS, (25, 100, 125))
        self.assertEqual(HEAD_SIZES, (98, 18, 541))
        features, stage, mask = adapter.observation()
        self.assertEqual(features.shape, (FEATURE_COUNT,))
        self.assertEqual(FEATURE_COUNT, 688)
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(stage, GOLD)
        np.testing.assert_array_equal(features[671:675], [1, 0, 0, 1])
        gold = set()
        for menu in ('costs', 'merchandise'):
            game.shop_menu = menu
            gold.update(set(game.get_legal_actions()) - {('shop_back',)})
        game.shop_menu = 'main'
        self.assertEqual(set(adapter.legal_engine_actions().values()), gold | {None})
        self.assertEqual(set(np.flatnonzero(mask)), set(adapter.legal_engine_actions()))
        self.assertIn(('buy', 'blue_key'), gold)
        self.assertIn(('buy', 'blue_mana_potion'), gold)
        self.assertIn(('change_type', 'Undead'), gold)
        self.assertNotIn(('change_type', 'Beast'), gold)
        self.assertIn(('reroll',), gold)

    def test_reroll_reads_no_future_rng_and_uses_actual_roll(self):
        game = self.game()
        game.gold = 9000
        game.player.base_EVA = game.player.EVA = 15
        adapter = Shop(game, 'Forest', None)
        original = game.reroll_rng.getstate()
        check = Random()
        check.setstate(original)
        low, high = sorted((check.randint(0, 15), check.randint(0, 15)))
        adapter.observation()
        self.assertEqual(game.reroll_rng.getstate(), original)
        info = adapter.step(adapter.action_id(('reroll',)), game.step)
        self.assertEqual(game.reroll_rng.getstate(), check.getstate())
        self.assertEqual((game.player.EVA, game.player.ACC, game.player.DR), (low, high-low, 15-high))
        self.assertEqual(info['reroll']['cost'], 100)
        self.assertEqual(info['reroll']['next_cost'], 200)
        self.assertEqual(info['reroll']['before']['EVA'], 15)
        self.assertEqual(info['reroll']['after']['base_DR'], 15-high)
        self.assertEqual(game.gold, 8900)

    def test_finish_skips_all_optional_costs_and_edits(self):
        game = self.game()
        game.gold = 9000
        adapter = Shop(game, 'Forest', None)
        commands = []

        def advance(move):
            commands.append(move)
            game.step(move)

        for head in (GOLD, STATS, SKILLS):
            self.assertEqual(adapter.head, head)
            info = adapter.step(adapter.action_id(FINISH), advance)
            self.assertTrue(info['stage_ended'])
            self.assertEqual(info['end_reason'], 'early')
            self.assertEqual(info['modifications'], 0)
        self.assertTrue(adapter.done)
        self.assertEqual(commands, [('leave_shop',)])
        self.assertEqual(game.gold, 9000)
        self.assertEqual(game.floor, 11)

    def test_stat_edit_updates_base_until_original_back_action(self):
        game = self.game()
        adapter = Shop(game, 'Forest', None)
        adapter.step(adapter.action_id(FINISH), game.step)
        self.assertEqual(adapter.head, STATS)
        original_attack = game.player.attack
        adapter.step(adapter.action_id(('reset_redistribution',)), game.step)
        self.assertEqual(game.player.base_attack, 1)
        self.assertIsNone(adapter.primitive_id(('adjust_stat', 'attack', -1)))
        for _ in range(20):
            adapter.step(adapter.action_id(('adjust_stat', 'attack', 10)), game.step)
        self.assertEqual(game.player.base_attack, 201)
        self.assertEqual(game.player.attack, original_attack)
        self.assertGreater(game.player.available_points, 0)
        adapter.step(adapter.action_id(FINISH), game.step)
        self.assertEqual(game.player.attack, 201)
        self.assertEqual(game.player.hp, game.player.max_hp)
        self.assertEqual(adapter.head, SKILLS)

    def test_caps_and_slot_mapping_use_real_primitives(self):
        game = self.game()
        game.gold = 9000
        new = next(name for name in game.data.skills if name not in game.player.learned_skills
                   and name not in {'strike', 'sacrifice'})
        game.player.learned_skills.append(new)
        adapter = Shop(game, 'Forest', None, budgets=(1, 1, 2))
        i = adapter.action_id(('buy', 'blue_mana_potion'))
        self.assertEqual(adapter.step(i, game.step)['end_reason'], 'cap')
        self.assertEqual(adapter.head, STATS)
        info = adapter.step(adapter.action_id(('reset_redistribution',)), game.step)
        self.assertEqual(info['modifications'], 1)
        self.assertEqual(info['end_reason'], 'cap')
        self.assertEqual(adapter.head, SKILLS)
        self.assertIsNone(adapter.primitive_id(('memorize', 'lock')))
        old = game.player.current_skills[0]
        replace = adapter.step(adapter.action_id(('replace_skill', 0, new)), game.step)
        self.assertEqual(replace['engine_action'], ('replace_skill', old, new))
        self.assertEqual(game.player.current_skills[0], new)
        self.assertEqual(game.player.memorized_skills[new], 0)
        self.assertFalse(adapter.observation()[2][adapter.action_id(('forget', 0))])
        exit_info = adapter.step(adapter.action_id(('memorize', 0)), game.step)
        self.assertEqual(exit_info['end_reason'], 'cap')
        self.assertEqual(game.player.saved_skills[new], 1)
        self.assertTrue(adapter.done)

    def test_interrupted_callback_does_not_create_a_false_exit(self):
        game = self.game()
        adapter = Shop(game, 'Forest', None)
        adapter.step(adapter.action_id(FINISH), game.step)
        adapter.step(adapter.action_id(FINISH), game.step)
        info = adapter.step(adapter.action_id(FINISH), lambda _: False)
        self.assertTrue(info['interrupted'])
        self.assertFalse(adapter.done)
        self.assertEqual(game.phase, 'shop')


if __name__ == '__main__':
    unittest.main()

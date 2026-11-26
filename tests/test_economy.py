import random
import unittest

from simulator import rewards, shop
from simulator.game import Game


class EconomyTests(unittest.TestCase):
    def setUp(self):
        self.game = Game()
        self.game.reset(3)
        self.player = self.game.player

    def test_reward_draw_filters_learned_skills(self):
        learned = self.player.learned_skills[0]
        self.game.monster.available_skills = [learned, "bear hand"]
        rewards.prepare(self.game)
        self.assertEqual(self.game.reward_candidates, ["bear hand"])
        self.assertEqual(self.game.reward_skill, "bear hand")
        rewards.apply(self.game, ("reward", "Gold"))
        self.assertEqual(self.game.gold, 1000)
        self.assertEqual(self.player.learned_skills[-1], "bear hand")

    def test_exchange_can_redraw_the_same_offer(self):
        self.game.monster.available_skills = ["bear hand"]
        rewards.prepare(self.game)
        rewards.apply(self.game, ("reward", "Exchange"))
        self.assertEqual(self.player.learned_skills.count("bear hand"), 1)
        self.assertEqual(self.game.gold, 0)

    def test_exchange_only_teaches_the_redrawn_skill(self):
        self.game.rng = random.Random(4)  # Draw indices zero, then one.
        self.game.monster.available_skills = ["bear hand", "tiger claw"]
        rewards.prepare(self.game)
        self.assertEqual(self.game.reward_skill, "bear hand")
        rewards.apply(self.game, ("reward", "Exchange"))
        self.assertNotIn("bear hand", self.player.learned_skills)
        self.assertEqual(self.player.learned_skills[-1], "tiger claw")

    def test_exchange_without_an_offer_keeps_the_deck(self):
        self.game.reward_skill = "bear hand"
        self.game.monster.available_skills = self.player.learned_skills[:1]
        before = self.player.learned_skills.copy()
        rewards.prepare(self.game)
        self.assertIsNone(self.game.reward_skill)
        self.assertIn(("reward", "Exchange"), rewards.legal_actions(self.game))
        rewards.apply(self.game, ("reward", "Exchange"))
        self.assertEqual(self.player.learned_skills, before)

    def test_reward_heal_and_stat_reset_have_distinct_effects(self):
        self.player.attack = 7
        self.player.hp = 4
        rewards.apply(self.game, ("reward", "Potion"))
        self.assertEqual((self.player.hp, self.player.attack), (200, 7))
        self.player.hp = 4
        rewards.apply(self.game, ("reward", "Initialize"))
        self.assertEqual((self.player.hp, self.player.attack), (4, 100))

    def test_evasion_reward_checks_base_and_current_caps_independently(self):
        self.game.rng = random.Random(1)  # First of the three reward stats.
        self.player.base_EVA = 80
        self.player.EVA = 78
        rewards.apply(self.game, ("reward", "Stat"))
        self.assertEqual((self.player.base_EVA, self.player.EVA), (80, 79))
        self.game.rng.seed(1)
        self.player.base_EVA = 78
        self.player.EVA = 82
        rewards.apply(self.game, ("reward", "Stat"))
        self.assertEqual((self.player.base_EVA, self.player.EVA), (79, 82))

    def test_shop_restores_and_saves_an_independent_copy_of_uses(self):
        skill = self.player.current_skills[0]
        self.player.memorized_skills[skill] = 1
        self.player.hp = 2
        self.player.attack = 4
        shop.enter(self.game)
        self.assertEqual((self.player.hp, self.player.attack), (200, 100))
        self.assertEqual(self.player.memorized_skills[skill], 20)
        shop.apply(self.game, ("shop_menu", "skills"))
        shop.apply(self.game, ("forget", skill))
        shop.apply(self.game, ("shop_back",))
        self.game.reroll_cost = 400
        shop.leave(self.game)
        self.player.memorized_skills[skill] = 0
        self.assertEqual(self.player.saved_skills[skill], 19)
        self.assertEqual(self.game.reroll_cost, 100)

    def test_skill_replacement_releases_the_old_allocation(self):
        old = self.player.current_skills[0]
        self.player.learned_skills.append("bear hand")
        shop.enter(self.game)
        shop.apply(self.game, ("shop_menu", "skills"))
        replace = ("replace_skill", old, "bear hand")
        self.assertIn(replace, shop.legal_actions(self.game))
        self.assertNotIn(("replace_skill", "lock", "bear hand"), shop.legal_actions(self.game))
        shop.apply(self.game, replace)
        self.assertNotIn(old, self.player.memorized_skills)
        self.assertEqual(self.player.current_skills[0], "bear hand")
        self.assertEqual(self.player.memorized_skills["bear hand"], 0)
        self.assertIn(("memorize", "bear hand"), shop.legal_actions(self.game))
        shop.apply(self.game, ("memorize", "bear hand"))
        self.assertEqual(self.player.memorized_skills["bear hand"], 1)

    def test_mana_budget_and_reduced_cost_unlock_one_more_use(self):
        skill = self.player.current_skills[0]
        shop.enter(self.game)
        spent = sum(self.player.skill_costs[name] * uses for name, uses in self.player.memorized_skills.items())
        self.player.max_mana = spent
        self.game.gold = 2000
        shop.apply(self.game, ("shop_menu", "skills"))
        self.assertNotIn(("memorize", skill), shop.legal_actions(self.game))
        shop.apply(self.game, ("shop_back",))
        shop.apply(self.game, ("shop_menu", "costs"))
        reduce = ("reduce_skill_cost", skill)
        self.assertIn(reduce, shop.legal_actions(self.game))
        shop.apply(self.game, reduce)
        self.assertEqual(self.player.skill_costs[skill], 2)
        self.assertEqual(self.game.gold, 1000)
        self.assertNotIn(reduce, shop.legal_actions(self.game))
        shop.apply(self.game, ("shop_back",))
        shop.apply(self.game, ("shop_menu", "skills"))
        self.assertIn(("memorize", skill), shop.legal_actions(self.game))

    def test_redistribution_preserves_points_and_recovers_when_confirmed(self):
        shop.enter(self.game)
        shop.apply(self.game, ("shop_menu", "stats"))
        self.assertNotIn(("leave_shop",), shop.legal_actions(self.game))
        shop.apply(self.game, ("reset_redistribution",))
        self.assertEqual(self.player.available_points, 396)
        self.assertNotIn(("adjust_stat", "hp", -1), shop.legal_actions(self.game))
        shop.apply(self.game, ("adjust_stat", "hp", 10))
        shop.apply(self.game, ("adjust_stat", "hp", 1))
        self.assertEqual((self.player.base_hp, self.player.hp), (12, 200))
        self.assertEqual(self.player.available_points, 385)
        shop.apply(self.game, ("shop_back",))
        self.assertEqual((self.player.attack, self.player.hp), (1, 24))
        self.assertIn(("leave_shop",), shop.legal_actions(self.game))

    def test_keys_open_in_order_with_first_available_learned_skill(self):
        self.player.learned_skills.extend(["bear hand", "tiger claw"])
        self.game.gold = 3000
        shop.enter(self.game)
        shop.apply(self.game, ("shop_menu", "merchandise"))
        self.assertNotIn(("buy", "red_key"), shop.legal_actions(self.game))
        shop.apply(self.game, ("buy", "blue_key"))
        self.assertEqual(self.player.current_skills[-2:], ["bear hand", "lock"])
        self.assertNotIn(("buy", "blue_key"), shop.legal_actions(self.game))
        shop.apply(self.game, ("buy", "red_key"))
        self.assertEqual(self.player.current_skills[-2:], ["bear hand", "tiger claw"])
        self.assertEqual(self.game.gold, 0)
        self.assertEqual(self.player.memorized_skills["tiger claw"], 0)

    def test_key_attempt_with_no_unequipped_skill_spends_gold(self):
        self.game.gold = 2000
        shop.enter(self.game)
        shop.apply(self.game, ("shop_menu", "merchandise"))
        shop.apply(self.game, ("buy", "blue_key"))
        self.assertEqual(self.game.gold, 1000)
        self.assertEqual(self.player.current_skills[-2:], ["lock", "lock"])
        self.assertIn(("buy", "blue_key"), shop.legal_actions(self.game))
        self.assertNotIn(("buy", "red_key"), shop.legal_actions(self.game))

    def test_mana_potions_require_the_prior_tier(self):
        self.game.gold = 6000
        shop.enter(self.game)
        shop.apply(self.game, ("shop_menu", "merchandise"))
        for item, mana, gold in (
            ("blue_mana_potion", 180, 5000),
            ("elemental_mana_potion", 190, 3000),
            ("undead_mana_potion", 200, 0),
        ):
            with self.subTest(potion=item):
                actions = shop.legal_actions(self.game)
                self.assertEqual([action[1] for action in actions if action[0] == "buy" and "potion" in action[1]], [item])
                shop.apply(self.game, ("buy", item))
                self.assertEqual((self.player.max_mana, self.game.gold), (mana, gold))

    def test_type_purchase_checks_both_current_and_base_type(self):
        self.game.gold = 500
        shop.enter(self.game)
        shop.apply(self.game, ("shop_menu", "merchandise"))
        self.assertFalse(any(action[0] == "change_type" for action in shop.legal_actions(self.game)))
        self.player.base_type = "Undead"
        self.player.type = "Elemental"
        self.assertNotIn(("change_type", "Undead"), shop.legal_actions(self.game))
        self.assertNotIn(("change_type", "Elemental"), shop.legal_actions(self.game))
        shop.apply(self.game, ("change_type", "Beast"))
        self.assertEqual((self.player.base_type, self.player.type, self.game.gold), ("Beast", "Beast", 0))

    def test_reroll_uses_a_separate_random_stream_and_doubles_price(self):
        self.game.gold = 1000
        shop.enter(self.game)
        shop.apply(self.game, ("shop_menu", "merchandise"))
        self.assertNotIn(("reroll",), shop.legal_actions(self.game))
        self.player.base_ACC = 100
        self.game.reroll_rng = random.Random(2)  # First cuts are 7 and 11 out of 100.
        rng_before = self.game.rng.getstate()
        shop.apply(self.game, ("reroll",))
        self.assertEqual((self.player.EVA, self.player.ACC, self.player.DR), (7, 4, 89))
        self.assertEqual((self.player.base_EVA, self.player.base_ACC, self.player.base_DR), (7, 4, 89))
        self.assertEqual((self.game.gold, self.game.reroll_cost), (900, 200))
        self.assertEqual(self.game.rng.getstate(), rng_before)

import random
import unittest

from simulator import combat
from simulator.game import Game


class Rolls(random.Random):
    def __init__(self, *rolls):
        super().__init__(7)
        self.rolls = iter(rolls)

    def randint(self, first, last):
        roll = next(self.rolls)
        assert first <= roll <= last
        return roll


class CombatTests(unittest.TestCase):
    def setUp(self):
        self.game = Game()
        self.game.reset(17)
        self.player = self.game.player
        self.monster = self.game.monster
        self.player.type = self.monster.type = "Normal"
        self.player.speed, self.monster.speed = 100, 50
        self.player.attack = self.monster.attack = 100
        self.player.defense = self.monster.defense = 100
        self.monster.hp = 150

    def turn(self, player_skill, enemy_skill="recovery", rolls=None):
        self.player.memorized_skills.setdefault(player_skill, 2)
        if rolls is not None:
            self.game.rng = Rolls(*rolls)
        return combat.resolve_turn(self.game, player_skill, enemy_skill)

    def test_matching_type_damage_and_use_consumption(self):
        self.turn("strike", rolls=[100])
        self.assertEqual(self.monster.hp, 102)
        self.assertEqual(self.player.memorized_skills["strike"], 1)

    def test_miss_still_spends_a_use(self):
        self.monster.EVA = 1
        self.turn("strike", rolls=[100])
        self.assertEqual(self.monster.hp, 150)
        self.assertEqual(self.player.memorized_skills["strike"], 1)

    def test_priority_attack_prevents_slower_combatant_from_acting(self):
        self.monster.speed = 200
        self.monster.hp = 1
        self.turn("fast claw", "strike", [1])
        self.assertEqual(self.monster.hp, 0)
        self.assertEqual(self.player.hp, 200)

    def test_second_hit_uses_surviving_substitute_defense_but_hits_body(self):
        self.player.attack = 1
        substitute = self.game.monster_substitute
        substitute.defense = 4
        substitute.hp = 20
        self.turn("double strike", rolls=[1, 1])
        self.assertEqual(substitute.hp, 10)
        self.assertEqual(self.monster.hp, 140)

    def test_second_hit_recalculates_after_dropping_substitute(self):
        substitute = self.game.monster_substitute
        substitute.hp = 20
        self.turn("double strike", rolls=[1, 1])
        self.assertEqual(substitute.hp, 0)
        self.assertEqual(self.monster.hp, 110)

    def test_hybrid_retains_first_damage_when_second_roll_misses(self):
        self.player.ACC = -220
        self.turn("hybrid attack", rolls=[1, 100])
        self.assertEqual(self.monster.hp, 125)

    def test_counter_uses_only_last_hit_damage(self):
        self.turn("counter", "double strike", [1, 1, 1])
        self.assertEqual(self.player.hp, 120)
        self.assertEqual(self.monster.hp, 90)
        self.assertEqual(self.player.ram_damage, 0)

    def test_counter_is_blocked_by_opposing_substitute(self):
        self.monster.substitute_hit_flag = 1
        self.game.monster_substitute.hp = 20
        self.turn("counter", "strike", [1, 1])
        self.assertEqual(self.monster.hp, 150)
        self.assertEqual(self.game.monster_substitute.hp, 20)

    def test_first_low_priority_action_is_skipped_without_charging(self):
        self.turn("counter", "mimic", [])
        self.assertEqual(self.player.memorized_skills["counter"], 2)
        self.assertEqual(self.monster.hp, 150)

    def test_mimic_rechecks_status_and_bypasses_copied_skill_charge(self):
        self.player.apply_status_effect("poison")
        self.turn("mimic", "gold rush", [1, 1, 100])
        self.assertEqual(self.player.hp, 100)  # Two poison ticks and the incoming hit.
        self.assertEqual(self.monster.hp, 100)
        self.assertEqual(self.game.gold, 200)
        self.assertEqual(self.player.memorized_skills["mimic"], 1)
        self.assertNotIn("gold rush", self.player.memorized_skills)

    def test_recovery_acts_while_asleep_and_preserves_source_list_skip(self):
        self.player.apply_status_effect("sleep")
        self.player.apply_status_effect("immunity")
        self.turn("recovery", rolls=[])
        self.assertEqual(self.player.status_conditions, [{"Name": "immunity", "chance_to_cure": 20}])

    def test_action_denial_stops_later_status_ticks(self):
        self.player.apply_status_effect("stun")
        self.player.apply_status_effect("poison")
        self.turn("strike", rolls=[49, 100, 100])
        self.assertEqual(self.player.hp, 200)
        self.assertEqual(self.player.memorized_skills["strike"], 2)
        self.assertEqual([e["chance_to_cure"] for e in self.player.status_conditions], [16.5, 16.5])

    def test_mark_overrides_a_normally_impossible_hit(self):
        self.monster.apply_status_effect("target")
        self.player.ACC = -80
        self.monster.EVA = 80
        self.turn("strike", rolls=[100])
        self.assertEqual(self.monster.hp, 102)

    def test_judgement_advantage_is_inverted_by_reverse(self):
        self.monster.type = "Undead"
        self.player.apply_status_effect("reverse")
        self.turn("judgement", rolls=[1])
        self.assertEqual(self.monster.hp, 117)

    def test_helmet_prevents_sacrifice_recoil_without_a_hit_roll(self):
        self.player.apply_status_effect("helmet")
        self.turn("sacrifice", rolls=[])
        self.assertEqual(self.monster.hp, 100)
        self.assertEqual(self.player.hp, 200)
        self.assertEqual(self.player.memorized_skills["sacrifice"], 2)

    def test_revive_intercepts_damage_in_the_same_turn(self):
        self.turn("revive", "strike", [1])
        self.assertEqual(self.player.hp, 180)
        self.assertEqual(self.game.player_substitute.hp, 0)

    def test_body_status_skill_blocked_by_substitute_still_consumes_a_roll(self):
        self.game.monster_substitute.hp = 20
        self.turn("paralyze", rolls=[1])
        with self.assertRaises(StopIteration):
            next(self.game.rng.rolls)
        self.assertFalse(self.monster.has_status_effect("stun"))

    def test_acid_stone_can_lower_reduction_without_damaging_hp(self):
        self.monster.DR = 65
        self.turn("acid stone", rolls=[1])
        self.assertEqual(self.monster.hp, 150)
        self.assertEqual(self.monster.DR, 60)

    def test_player_imitation_copies_and_restores_original_slots(self):
        self.monster.attack = 140
        self.player.current_skills[0] = "imitation"
        self.turn("imitation", rolls=[])
        self.assertEqual(self.player.attack, 140)
        self.assertTrue(self.player.has_status_effect("imitation"))
        self.assertEqual(set(self.player.memorized_skills.values()), {5})
        self.turn("recovery", rolls=[])
        self.assertEqual(self.player.attack, 100)
        self.assertEqual(self.player.current_skills[0], "imitation")
        self.assertEqual(self.player.memorized_skills["imitation"], 1)
        self.assertFalse(self.player.has_status_effect("imitation"))

    def test_every_registry_skill_resolves_through_combat(self):
        for name in self.game.data.skills:
            with self.subTest(name=name):
                self.setUp()
                enemy_skill = "strike" if name in {"counter", "mimic"} else "recovery"
                self.turn(name, enemy_skill, [1] * 5)
                self.assertGreaterEqual(self.player.hp, 0)
                self.assertGreaterEqual(self.monster.hp, 0)

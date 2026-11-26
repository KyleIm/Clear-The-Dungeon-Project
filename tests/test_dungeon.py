import random
import unittest

from simulator import dungeon, enemy_ai
from simulator.game import Game


class DungeonTests(unittest.TestCase):
    def setUp(self):
        self.game = Game()
        self.game.reset(71)

    def test_region_draw_removes_one_and_leaves_gate_and_lair_fixed(self):
        game = self.game
        game.remaining_areas = ["Forest", "Fort", "Graveyard"]
        game.rng.seed(71)
        dungeon.select_area(game)
        self.assertEqual(game.area, "Graveyard")
        self.assertEqual(game.remaining_areas, ["Forest", "Fort"])
        rng_state = game.rng.getstate()
        for floor, area in ((91, "Gate"), (100, "Lair")):
            game.floor = floor
            dungeon.select_area(game)
            self.assertEqual(game.area, area)
            self.assertEqual(game.remaining_areas, ["Forest", "Fort"])
            self.assertEqual(game.rng.getstate(), rng_state)

    def test_starting_decks_split_one_shuffled_attack_pool(self):
        game = self.game
        game.rng.seed(71)
        game.area = "Forest"
        dungeon.initial_skills(game)
        starter = ["skull dagger", "pierce", "counter", "terror", "lock", "lock"]
        self.assertEqual(game.player.current_skills, starter)
        self.assertEqual(game.player.learned_skills, starter)
        self.assertEqual(game.ella_skills, ["thunder", "soul strike", "fast claw", "weakness"])
        self.assertEqual(game.player.memorized_skills, dict(zip(starter, [20, 10, 10, 10])))
        game.player.memorized_skills["pierce"] -= 1
        self.assertEqual(game.player.saved_skills["pierce"], 10)

    def test_ordinary_encounters_keep_floor_one_rng_and_scale_later_floors(self):
        game = self.game
        game.area = "Forest"
        game.rng.seed(12)
        rng_state = game.rng.getstate()
        dungeon.spawn_encounter(game)
        self.assertEqual(game.monster.key, "rat")
        self.assertEqual(game.rng.getstate(), rng_state)

        game.floor = 2
        dungeon.spawn_encounter(game)
        self.assertEqual(game.monster.key, "rat")
        reference = random.Random(12)
        reference.choice(range(7))
        reference.randint(0, 0)
        reference.randint(0, 0)
        self.assertEqual(game.rng.getstate(), reference.getstate())

        game.floor = 12
        game.rng.seed(12)
        dungeon.spawn_encounter(game)
        self.assertEqual(game.monster.key, "golem")
        self.assertEqual((game.monster.ACC, game.monster.EVA, game.monster.DR),
                         (game.monster.base_ACC + 2, game.monster.base_EVA + 3, game.monster.base_DR))

        game.floor = 17
        dungeon.spawn_encounter(game)
        self.assertEqual(game.monster.key, "minocentaur")
        surplus = sum(getattr(game.monster, stat) - getattr(game.monster, f"base_{stat}")
                      for stat in ("ACC", "EVA", "DR"))
        self.assertEqual(surplus, 5)

    def test_gate_recovers_stage_boss_in_first_in_first_out_order(self):
        game = self.game
        game.gate_queue = []
        rng_state = game.rng.getstate()
        game.floor, game.area = 10, "Forest"
        dungeon.spawn_encounter(game)
        lionheart = game.monster
        lionheart.hp, lionheart.attack = 0, 1
        game.floor, game.area = 20, "Fort"
        dungeon.spawn_encounter(game)
        self.assertEqual(game.gate_queue, [43, 42])

        game.floor, game.area = 91, "Gate"
        dungeon.spawn_encounter(game)
        self.assertIs(game.monster, lionheart)
        self.assertEqual((lionheart.hp, lionheart.attack), (lionheart.max_hp, lionheart.base_attack))
        self.assertEqual(game.gate_queue, [42])
        game.floor = 92
        dungeon.spawn_encounter(game)
        self.assertEqual(game.monster.key, "grimreaper")
        self.assertEqual(game.gate_queue, [])
        self.assertEqual(game.rng.getstate(), rng_state)

    def test_ella_later_decks_read_area_path_and_current_player_type(self):
        game = self.game
        game.area, game.floor = "Temple", 35
        dungeon.spawn_encounter(game)
        self.assertEqual((game.monster.key, game.monster.type, game.monster.base_type),
                         ("ellasecond", "Beast", "Normal"))
        self.assertEqual(game.ella_skills, ["bear hand", "fast claw", "whisker needle", "double strike"])

        game.floor, game.player.type, game.player.base_hp = 75, "Elemental", 101
        for path, key, kind, opener in ((1, "ellaa", "Elemental", "tornado"),
                                        (2, "ellad", "Beast", "tiger claw")):
            with self.subTest(path=path):
                game.ella_path = path
                dungeon.spawn_encounter(game)
                self.assertEqual((game.monster.key, game.monster.type), (key, kind))
                self.assertEqual(game.ella_skills, [opener, "recovery", "despair", "plague"])

    def test_floor_99_ella_does_not_draw_a_gate_boss_or_advance_floor(self):
        game = self.game
        game.floor, game.area, game.ella_path, game.ella_encounter = 99, "Gate", 1, 1
        game.player.type, game.player.EVA = "Elemental", 11
        game.gate_queue = []
        dungeon.spawn_encounter(game)
        self.assertEqual((game.floor, game.monster.key, game.monster.type), (99, "ellaf", "Beast"))
        self.assertEqual(game.ella_skills, ["pierce", "recovery", "counter", "paralyze"])

    def test_first_ella_bonus_uses_base_fraction_and_does_not_cap_accuracy(self):
        game = self.game
        game.area, game.player.ACC, game.player.base_speed = "Temple", 60, 103
        old_speed = game.player.speed
        dungeon.apply_first_ella_bonus(game)
        self.assertEqual((game.player.ACC, game.player.speed), (110, old_speed + 25))
        self.assertEqual((game.player.base_ACC, game.player.base_speed), (0, 103))


class EnemyActionTests(unittest.TestCase):
    def setUp(self):
        self.game = Game()
        self.game.reset(8)

    def test_normal_deck_is_rebuilt_and_only_valid_starters_upgrade(self):
        game = self.game
        game.monster = game.monster_roster[3]
        for floor, opener in ((30, "skull dagger"), (31, "bone spear"), (61, "undead missile")):
            game.floor = floor
            selected = enemy_ai.select_action(game)
            self.assertEqual(game.monster.available_skills, [opener, "despair", "sticky", "fog"])
            self.assertEqual(game.monster.ram_action, selected)
            self.assertEqual(game.monster.selected_action, selected)
        self.assertEqual(game.data.monsters[3]["skills"][0], "skull dagger")

    def test_ella_uses_live_deck_and_preserves_misspelled_upgrade(self):
        game = self.game
        game.monster = game.monster_roster[54]
        game.floor = 75
        game.ella_skills = ["bone spear", "recovery", "counter", "plague"]
        enemy_ai.select_action(game)
        self.assertEqual(game.ella_skills[0], "bone spear")
        game.ella_skills[0] = "wave"
        enemy_ai.select_action(game)
        self.assertEqual(game.ella_skills[0], "tornado")

    def test_imitation_uses_copied_deck_without_upgrade(self):
        game = self.game
        game.floor = 61
        game.monster = game.monster_roster[21]
        game.monster.status_conditions = [{"Name": "imitation"}]
        game.monster.current_skills = ["wolf fang", "sticky", "curse", "terror"]
        enemy_ai.select_action(game)
        self.assertEqual(game.monster.available_skills, game.monster.current_skills)
        self.assertEqual(game.monster.available_skills[0], "wolf fang")

    def test_floor_35_uses_a_second_draw_after_overwritten_random_action(self):
        game = self.game
        game.floor, game.area = 35, "Forest"
        dungeon.spawn_encounter(game)
        for hp, accuracy, expected in ((200, 0, "soul strike"), (39, 0, "drain"),
                                        (200, -1, "drain")):
            with self.subTest(hp=hp, accuracy=accuracy):
                game.monster.hp, game.player.ACC = hp, accuracy
                game.rng.seed(4)
                self.assertEqual(enemy_ai.select_action(game), expected)

    def test_boss_conditional_decks_take_priority_over_random_choices(self):
        game = self.game
        tests = [
            (42, {"ACC": 30}, {}, "deathscythe"),
            (43, {}, {}, "heart of lion"),
            (45, {}, {"type": "Undead"}, "judgement"),
            (46, {}, {}, "veilguard"),
            (47, {"ACC": 20}, {}, "shadowstar"),
            (48, {}, {"DR": 11}, "acid stone"),
            (49, {}, {"defense": 99}, "curse"),
            (50, {"hp": 101, "status_conditions": [{"Name": "sleep"}]}, {}, "recovery"),
        ]
        for monster_id, enemy_stats, player_stats, expected in tests:
            with self.subTest(monster=monster_id):
                game.reset(8)
                game.rng.seed(12)
                game.monster = game.monster_roster[monster_id]
                for target, changes in ((game.player, player_stats), (game.monster, enemy_stats)):
                    for name, value in changes.items():
                        setattr(target, name, value)
                self.assertEqual(enemy_ai.select_action(game), expected)

    def test_late_ella_status_precedes_full_health_fast_opener(self):
        game = self.game
        game.floor, game.ella_path = 75, 1
        dungeon.spawn_encounter(game)
        self.assertEqual(enemy_ai.select_action(game), "paralyze")
        game.monster.status_conditions = [{"Name": "poison"}]
        self.assertEqual(enemy_ai.select_action(game), "recovery")


if __name__ == "__main__":
    unittest.main()

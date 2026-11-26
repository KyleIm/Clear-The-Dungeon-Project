from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import random
import unittest
from unittest.mock import patch

from simulator import __main__ as cli
from simulator.game import Game
from simulator.headless import play_headless


class GameTests(unittest.TestCase):
    def test_initial_stats_and_detached_observation(self):
        game = Game()
        state = game.reset(71)
        self.assertEqual(state["phase"], "combat")
        self.assertEqual(state["floor"], 1)
        self.assertEqual(state["player"]["hp"], 200)
        self.assertEqual(
            [state["player"][name] for name in ("attack", "defense", "speed", "base_hp")],
            [100] * 4,
        )
        self.assertEqual(list(game.player.memorized_skills.values()), [20, 10, 10, 10])
        self.assertEqual(game.player.current_skills[-2:], ["lock", "lock"])
        self.assertEqual(len(game.get_legal_actions()), 4)
        state["player"]["current_skills"].clear()
        self.assertEqual(len(game.player.current_skills), 6)
        json.dumps(game.get_state())

    def test_invalid_action_does_not_consume_randomness(self):
        game = Game()
        game.reset(123)
        before = game.rng.getstate(), game.reroll_rng.getstate(), game.get_state()
        with self.assertRaises(ValueError):
            game.step(("skill", "missing"))
        self.assertEqual(before, (game.rng.getstate(), game.reroll_rng.getstate(), game.get_state()))

    def test_reproducible_legal_action_trace(self):
        one, two = Game(), Game()
        self.assertEqual(one.reset(9), two.reset(9))
        policy = random.Random(901)
        for _ in range(100):
            actions = one.get_legal_actions()
            self.assertEqual(actions, two.get_legal_actions())
            if not actions:
                break
            action = policy.choice(actions)
            self.assertEqual(one.step(action), two.step(action))
        self.assertEqual(one.reset(9), two.reset(9))

    def test_json_action_list(self):
        a, b = Game(), Game()
        a.reset(23)
        b.reset(23)
        action = a.get_legal_actions()[0]
        self.assertEqual(a.step(action), b.step(list(action)))

    def test_cli_labels_skill_uses_without_changing_api(self):
        game = Game()
        game.reset(24)
        game.phase, game.shop_menu = "shop", "skills"
        skill = game.player.current_skills[0]
        game.player.memorized_skills[skill] -= 1
        self.assertIn(("memorize", skill), game.get_legal_actions())
        self.assertIn(("forget", skill), game.get_legal_actions())

        output = StringIO()
        with (
            patch.object(cli, "Game", return_value=game),
            patch.object(game, "reset", return_value=game.get_state()),
            patch("sys.argv", ["simulator"]),
            patch("builtins.input", return_value="q"),
            redirect_stdout(output),
        ):
            cli.main()
        self.assertIn(f"Increase skill uses {skill}", output.getvalue())
        self.assertIn(f"Decrease skill uses {skill}", output.getvalue())

    def test_death_beats_victory_on_simultaneous_knockout(self):
        game = Game()
        game.reset(1)

        def double_ko(engine, _player_skill, _enemy_skill):
            engine.player.hp = engine.monster.hp = 0

        with patch("simulator.game.combat.resolve_turn", side_effect=double_ko):
            game.step(game.get_legal_actions()[0])
        self.assertTrue(game.is_terminal())
        self.assertFalse(game.is_cleared())
        self.assertEqual(game.get_state()["phase"], "game_over")
        self.assertEqual(game.get_legal_actions(), [])

    def doppelganger_encounter(self):
        game = Game()
        game.reset(24)
        game.floor, game.area = 17, "Mansion"
        game.monster = game.monster_roster[8]
        game.player.base_speed = game.player.speed = 1
        return game

    def test_ko_after_imitation_uses_the_turns_original_reward_deck(self):
        game = self.doppelganger_encounter()
        game.monster.hp = 1
        game.step(("skill", game.player.current_skills[0]))
        self.assertTrue(game.monster.has_status_effect("imitation"))
        self.assertEqual(game.phase, "reward")
        self.assertEqual(game.reward_candidates, ["imitation"])
        self.assertEqual(game.reward_skill, "imitation")
        game.step(("reward", "Gold"))
        self.assertIn("imitation", game.player.learned_skills)

    def test_ko_after_copied_recovery_uses_the_turns_copied_reward_deck(self):
        game = self.doppelganger_encounter()
        player = game.player
        player.current_skills[1] = "recovery"
        player.memorized_skills["recovery"] = 2
        player.learned_skills.append("recovery")
        game.step(("skill", "recovery"))
        self.assertTrue(game.monster.has_status_effect("imitation"))
        active = [skill for skill in player.current_skills if skill != "lock"]
        for seed in range(100):
            if random.Random(seed).choice(active) == "recovery":
                game.rng.seed(seed)
                break
        else:
            self.fail("No seed selected the copied Recovery")
        game.monster.hp = 1
        game.step(("skill", player.current_skills[0]))
        self.assertEqual(game.monster.ram_action, "recovery")
        self.assertFalse(game.monster.has_status_effect("imitation"))
        self.assertEqual(game.phase, "reward")
        self.assertEqual(game.reward_candidates, [])
        self.assertIsNone(game.reward_skill)

    def test_complete_progression_both_ella_paths(self):
        for route in ("pickpocket", "help"):
            with self.subTest(route=route):
                game = Game()
                game.reset(40)
                encounters = []
                shops = []

                def force_enemy_ko(engine, _player_skill, _enemy_skill):
                    encounters.append((engine.floor, engine.monster.key))
                    engine.monster.hp = 0

                with patch("simulator.game.combat.resolve_turn", side_effect=force_enemy_ko):
                    while not game.is_terminal():
                        if game.phase == "combat":
                            action = game.get_legal_actions()[0]
                        elif game.phase == "event":
                            action = ("ella", route)
                        elif game.phase == "shop":
                            shops.append(game.floor)
                            action = ("leave_shop",)
                        else:
                            action = game.get_legal_actions()[0]
                        game.step(action)
                self.assertTrue(game.is_cleared())
                self.assertEqual(encounters[-1], (100, "dragon"))
                self.assertEqual([key for floor, key in encounters if floor <= 90 and floor % 10 == 0],
                                 [key for floor, key in encounters if 91 <= floor <= 99 and key != "ellaf"])
                self.assertEqual(sum(key == "ellaf" for _, key in encounters), int(route == "pickpocket"))
                self.assertEqual(shops, list(range(10, 91, 10)) + ([99] if route == "help" else []))
                self.assertEqual(len(encounters), 101 if route == "pickpocket" else 100)

    def test_headless_random_policy(self):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), patch("builtins.input", side_effect=AssertionError):
            for seed in range(5):
                rng = random.Random(seed + 99)
                result = play_headless(Game(), lambda _state, actions: rng.choice(actions), seed=seed)
                self.assertIn(result["phase"], {"cleared", "game_over"})
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()

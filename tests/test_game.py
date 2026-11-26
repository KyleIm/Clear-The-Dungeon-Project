from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import random
import unittest
from unittest.mock import patch

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

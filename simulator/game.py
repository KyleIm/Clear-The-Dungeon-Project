"""Coordinate source-ordered transitions and expose a UI-independent action API."""
from dataclasses import asdict
import random

from . import combat, dungeon, enemy_ai, rewards, shop
from .character import Character
from .data_loader import GameData

TERMINAL_PHASES = {"cleared", "game_over"}


def _character_state(character):
    state = asdict(character)
    state["purchases"] = sorted(character.purchases)
    state["max_hp"] = character.max_hp
    return state


class Game:
    """One dungeon run. Pass legal action tuples, or their JSON list equivalents, to step."""

    def __init__(self, data=None):
        self.data = GameData() if data is None else data
        self.reset()

    def reset(self, seed=None):
        config = self.data.config
        initial = config["player"]
        self.rng = random.Random(seed)
        self.reroll_rng = random.Random(None if seed is None else f"reroll:{seed!r}")
        self.floor = initial["starting_floor"]
        self.area_number = initial["starting_area"]
        self.area = ""
        self.remaining_areas = [row["id"] for row in self.data.areas["regions"] if row["id"] not in {"Gate", "Lair"}]
        self.gate_queue = []
        self.ella_path = initial["starting_ella_path"]
        self.ella_encounter = initial["starting_ella_encounter"]
        self.ella_skills = []
        self.monster_roster = {
            monster_id: Character.from_record(record)
            for monster_id, record in self.data.monsters.items()
        }
        self.player = Character.from_record(initial | {"key": "player", "name": "Rogue"})
        self.player.max_mana = config["mana"]["max"]
        self.player.available_points = initial["starting_unallocated_points"]
        self.player.skill_costs = {name: skill["mana_cost"] for name, skill in self.data.skills.items()}
        substitute = config["combat"]["substitute"]
        self.player_substitute = Character.from_record(
            substitute | {"key": "player_substitute", "name": "Rogue's Substitute"}
        )
        self.monster_substitute = Character.from_record(
            substitute | {"key": "monster_substitute", "name": "Substitute"}
        )
        self.gold = initial["starting_gold"]
        self.reroll_cost = config["shop"]["prices"]["reroll_initial"]
        self.shop_menu = "main"
        self.reward_candidates = []
        self.reward_skill = None
        self.phase = "combat"
        self.turn = 0
        self.wins = 0
        dungeon.select_area(self)
        dungeon.initial_skills(self)
        dungeon.spawn_encounter(self)
        return self.get_state()

    def get_state(self):
        """Return a detached snapshot suitable for serialization or policy input."""
        return {
            "phase": self.phase,
            "floor": self.floor,
            "area": self.area,
            "area_number": self.area_number,
            "turn": self.turn,
            "wins": self.wins,
            "gold": self.gold,
            "player": _character_state(self.player) | {"current_mana": self.player.current_mana},
            "monster": _character_state(self.monster),
            "player_substitute": _character_state(self.player_substitute),
            "monster_substitute": _character_state(self.monster_substitute),
            "reward_skill": self.reward_skill,
            "reward_candidates": list(self.reward_candidates),
            "shop_menu": self.shop_menu,
            "reroll_cost": self.reroll_cost,
            "remaining_areas": list(self.remaining_areas),
            "gate_queue": list(self.gate_queue),
            "ella_path": self.ella_path,
            "ella_encounter": self.ella_encounter,
            "ella_skills": list(self.ella_skills),
        }

    def is_terminal(self):
        return self.phase in TERMINAL_PHASES

    def is_cleared(self):
        return self.phase == "cleared"

    def get_legal_actions(self):
        if self.phase == "combat":
            equipped = [name for name in self.player.current_skills if name != "lock"]
            uses = self.player.memorized_skills
            if all(uses[name] == 0 for name in equipped):
                return [("skill", "sacrifice")]
            return [("skill", name) for name in equipped if uses[name] > 0]
        if self.phase == "event":
            return [("ella", "pickpocket"), ("ella", "help")]
        if self.phase == "reward":
            return rewards.legal_actions(self)
        if self.phase == "shop":
            return shop.legal_actions(self)
        return []

    def step(self, action):
        """Execute one legal action and return the resulting observation."""
        if not isinstance(action, (list, tuple)) or not action:
            raise ValueError(f"Illegal action in {self.phase}: {action!r}")
        action = tuple(action)
        if action not in self.get_legal_actions():
            raise ValueError(f"Illegal action in {self.phase}: {action!r}")
        if self.phase == "combat":
            self._combat_step(action[1])
        elif self.phase == "event":
            self.ella_path = 1 if action[1] == "pickpocket" else 2
            self.player.base_type = self.player.type = self.monster.type
            self._offer_reward()
        elif self.phase == "reward":
            rewards.apply(self, action)
            self.reward_skill = None
            self.reward_candidates = []
            self._advance_after_reward()
        elif action == ("leave_shop",):
            shop.leave(self)
            self.floor += 1
            self.area_number += 1
            dungeon.select_area(self)
            self._start_encounter()
        else:
            shop.apply(self, action)
        return self.get_state()

    def _combat_step(self, player_skill):
        self.player.selected_action = self.player.ram_action = player_skill
        enemy_skill = enemy_ai.select_action(self)
        combat.resolve_turn(self, player_skill, enemy_skill)
        self.turn += 1
        if self.player.hp <= 0:
            self.phase = "game_over"
        elif self.monster.hp <= 0:
            self.wins += 1
            if self.floor == self.data.config["run"]["terminal_floor"]:
                self.phase = "cleared"
            elif self.floor == 35:
                self.phase = "event"
            else:
                if self.floor == 5:
                    dungeon.apply_first_ella_bonus(self)
                self._offer_reward()

    def _offer_reward(self):
        rewards.prepare(self)
        self.phase = "reward"

    def _start_encounter(self):
        dungeon.spawn_encounter(self)
        self.turn = 0
        self.phase = "combat"

    def _open_shop(self):
        shop.enter(self)
        if self.floor != 99:
            self.area = "Shop"
        self.phase = "shop"

    def _advance_after_reward(self):
        if self.floor == 99:
            if self.ella_encounter:
                self.floor += 1
                dungeon.select_area(self)
                self._start_encounter()
            else:
                self.ella_encounter = 1
                if self.ella_path == 1:
                    self._start_encounter()
                else:
                    self._open_shop()
        elif self.floor % 10 == 0:
            self._open_shop()
        else:
            if self.floor <= 89:
                if self.floor in {4, 34, 74} or self.floor % 10 == 9:
                    combat.recover(self.monster)
                    combat.reset_stats(self.monster)
                else:
                    combat.reset_stats(self.monster)
                    combat.recover(self.monster)
            self.floor += 1
            self._start_encounter()

"""Gymnasium boundary around the headless game, with public observations only."""
from __future__ import annotations

from collections import defaultdict, deque
from functools import lru_cache
import math

import gymnasium as gym
import numpy as np

from simulator.data_loader import GameData
from simulator.game import Game


PHASES = ("combat", "reward", "shop", "event")
MAIN_STATS = ("attack", "defense", "hp", "speed")
EXTRA_STATS = ("EVA", "ACC", "DR")
COMBAT_STATS = ("attack", "defense", "speed", *EXTRA_STATS)
SHOP_MENUS = ("skills", "costs", "stats", "merchandise")
KEYS = ("blue_key", "red_key")
TYPES = ("Undead", "Elemental", "Beast")
LOCK = "lock"


@lru_cache(maxsize=1)
def _game_data():
    return GameData()


def _scaled(value, scale):
    return math.copysign(math.log1p(abs(value) / scale), value)


def _one_hot(value, vocabulary):
    return [float(value == entry) for entry in vocabulary]


class DungeonEnv(gym.Env):
    """One engine decision per step, with a fixed masked action catalog."""

    metadata = {"render_modes": []}

    def __init__(self, stage: dict, *, action_limit=10_000, early_deadlock=False):
        super().__init__()
        if type(action_limit) is not int or action_limit < 1:
            raise ValueError("action_limit must be a positive integer")
        self.stage = dict(stage)
        self.weights = dict(stage.get("weights", {}))
        if self.stage["target"] not in {"next_floor", "boss", "floor", "clear"}:
            raise ValueError("Unknown curriculum target")
        if self.stage["target"] in {"boss", "floor"}:
            floor = self.stage.get("floor")
            if type(floor) is not int or not 1 <= floor <= 100:
                raise ValueError("Curriculum target floor must be an integer between 1 and 100")
        if self.weights.keys() - {"clear", "death", "target", "floor"}:
            raise ValueError("Unknown reward weight")
        if not all(math.isfinite(weight) for weight in self.weights.values()):
            raise ValueError("Reward weights must be finite")
        self.action_limit = action_limit
        self.early_deadlock = early_deadlock
        self.data = _game_data()
        self.game = Game(self.data)
        self.shop = self.data.config["shop"]
        self.catalog_skills = tuple(sorted(set(self.data.skills) - {"sacrifice"}))
        self.skill_tokens = (None, LOCK, *self.catalog_skills)
        self.enemy_names = (*sorted({monster["name"] for monster in self.data.monsters.values()}), None)
        self.areas = (*sorted(self.data.regions), "Shop", None)
        self.statuses = tuple(sorted(self.data.config["status"]["registry_values"]))
        self.slots = tuple(range(self.data.config["slots"]["total"]))
        self.potions = tuple(self.shop["mana_potions_in_order"])
        self.items = KEYS + self.potions
        self.rewards = tuple(self.data.config["rewards"]["options"])
        self.action_catalog, self.action_phases = self._catalog()
        self._action_ids = {entry: index for index, entry in enumerate(self.action_catalog)}
        self._verbs = (None, *dict.fromkeys(entry[0] for entry in self.action_catalog))
        self.action_space = gym.spaces.Discrete(len(self.action_catalog))
        obs, _ = self.reset(seed=0)
        self.observation_space = gym.spaces.Dict({
            "features": gym.spaces.Box(-np.inf, np.inf, shape=obs["features"].shape, dtype=np.float32),
            "phase": gym.spaces.Discrete(len(PHASES)),
            "action_mask": gym.spaces.MultiBinary(len(self.action_catalog)),
        })

    def _catalog(self):
        combat = [("cast_slot", slot) for slot in self.slots] + [("skill", "sacrifice")]
        reward = [("reward", option) for option in self.rewards]
        shop = [("shop_menu", menu) for menu in SHOP_MENUS] + [("leave_shop",)]
        for verb in ("memorize", "forget"):
            shop.extend((verb, slot) for slot in self.slots)
        shop.extend(("replace_skill", slot, skill) for slot in self.slots for skill in self.catalog_skills)
        shop.extend(("reduce_skill_cost", skill) for skill in self.catalog_skills)
        for stat in MAIN_STATS:
            for step in self.shop["redistribution_steps"]:
                shop.extend((("adjust_stat", stat, step), ("adjust_stat", stat, -step)))
        shop.extend([("reset_redistribution",), ("reroll",)])
        shop.extend(("buy", item) for item in self.items)
        shop.extend(("change_type", kind) for kind in TYPES)
        shop.append(("shop_back",))
        event = [("ella", "pickpocket"), ("ella", "help")]
        blocks = combat, reward, shop, event
        return (tuple(action for block in blocks for action in block),
                tuple(phase for phase, block in enumerate(blocks) for _ in block))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(seed) if seed is not None else int(self.np_random.integers(2**63))
        self.action_space.seed(game_seed)
        self._state = self.game.reset(game_seed)
        self.actions = 0
        self.finished = False
        self.target_success = False
        self.previous_action = None
        self.previous_skill = None
        self.ella_choice = None
        self.start_floor = self._state["floor"]
        self.furthest_floor = self.start_floor
        self.last_phase = "combat"
        self.visited = set()
        self.last_hp = {}
        self.last_slots = None
        self.last_enemy_name = None
        self.known_player_max_hp = self.data.config["player"]["max_hp"]
        self._record_visible(self._state, new_encounter=True)
        return self.observe(self._state), self._info()

    def _record_visible(self, state, *, new_encounter=False):
        if new_encounter:
            self.last_hp.pop("monster", None)
            self.last_enemy_name = None
        phase = state["phase"]
        if phase == "combat" or (phase == "shop" and state["shop_menu"] == "skills"):
            player = state["player"]
            self.last_slots = tuple((skill, player["memorized_skills"].get(skill, 0))
                                    for skill in player["current_skills"])
        if phase == "shop" and state["shop_menu"] == "stats":
            self.known_player_max_hp = 2 * state["player"]["base_hp"]
        if phase in {"combat", "reward", "event"}:
            if phase == "combat":
                self.visited.add(state["area"])
            for side in ("player", "monster"):
                if state[f"{side}_substitute"]["hp"] <= 0:
                    visible_hp = 0 if side == "monster" and phase != "combat" else state[side]["hp"]
                    self.last_hp[side] = (visible_hp, self.actions)
            if state["monster_substitute"]["hp"] <= 0:
                self.last_enemy_name = state["monster"]["name"]

    def _shop_public(self, state):
        """Project skill-book and shop controls before applying their visible rules."""
        p = state["player"]
        learned = set(p["learned_skills"]) & set(self.catalog_skills)
        skills = tuple(p["current_skills"])
        uses = {skill: p["memorized_skills"].get(skill, 0) for skill in skills if skill != LOCK}
        costs = {skill: p["skill_costs"][skill] for skill in learned | uses.keys()}
        section = state["shop_menu"]
        return {
            "menu": section, "gold": state["gold"], "skills": skills, "uses": uses,
            "learned": learned, "costs": costs,
            "mana": p["max_mana"] - sum(costs[skill] * count for skill, count in uses.items()),
            "base_stats": {stat: p[f"base_{stat}"] for stat in MAIN_STATS} if section == "stats" else {},
            "points": p["available_points"] if section == "stats" else None,
            "reroll_price": state["reroll_cost"] if section == "merchandise" else None,
            "extra_stats_total": sum(p[stat] for stat in EXTRA_STATS),
            "type": p["type"], "purchases": set(p["purchases"]),
        }

    def _public_actions(self, state, shop_state=None):
        phase = state["phase"]
        if phase == "combat":
            p = state["player"]
            uses = [p["memorized_skills"][skill] for skill in p["current_skills"] if skill != LOCK]
            if all(count == 0 for count in uses):
                return [("skill", "sacrifice")]
            return [("cast_slot", slot) for slot, skill in enumerate(p["current_skills"])
                    if skill != LOCK and p["memorized_skills"][skill] > 0]
        if phase == "reward":
            return [("reward", option) for option in self.rewards]
        if phase == "event":
            return [("ella", "pickpocket"), ("ella", "help")]
        if phase != "shop":
            return []
        public = shop_state if shop_state is not None else self._shop_public(state)
        menu, gold = public["menu"], public["gold"]
        prices = self.shop["prices"]
        if menu == "main":
            return [("shop_menu", name) for name in SHOP_MENUS] + [("leave_shop",)]
        result = []
        if menu == "skills":
            unequipped = public["learned"] - set(public["skills"])
            for slot, skill in enumerate(public["skills"]):
                if skill == LOCK:
                    continue
                if public["mana"] >= public["costs"][skill]:
                    result.append(("memorize", slot))
                if public["uses"][skill] > 0:
                    result.append(("forget", slot))
                result.extend(("replace_skill", slot, replacement) for replacement in sorted(unequipped))
        elif menu == "costs" and gold >= prices["skill_cost_reduction"]:
            amount = self.shop["skill_cost_reduction_amount"]
            minimum = self.shop["minimum_reduced_skill_cost"]
            for skill in sorted(public["learned"]):
                cost = public["costs"][skill]
                if cost == self.data.skills[skill]["mana_cost"] and cost - amount >= minimum:
                    result.append(("reduce_skill_cost", skill))
        elif menu == "stats":
            for stat, value in public["base_stats"].items():
                for step in self.shop["redistribution_steps"]:
                    if public["points"] >= step:
                        result.append(("adjust_stat", stat, step))
                    if value - step >= self.shop["redistribution_minimum"]:
                        result.append(("adjust_stat", stat, -step))
            result.append(("reset_redistribution",))
        elif menu == "merchandise":
            if gold >= public["reroll_price"] and public["extra_stats_total"] >= self.shop["reroll_min_points"]:
                result.append(("reroll",))
            for group in (KEYS, self.potions):
                item = next((item for item in group if item not in public["purchases"]), None)
                if item and gold >= prices[item] and (item not in KEYS or LOCK in public["skills"]):
                    result.append(("buy", item))
            # Shop entry resets the current type and current additional stats to baseline.
            if public["type"] != "Normal" and gold >= prices["type_change"]:
                result.extend(("change_type", kind) for kind in TYPES if kind != public["type"])
        return [*result, ("shop_back",)]

    def _actor_features(self, state, side):
        actor = state[side]
        combat = state["phase"] == "combat"
        shop_player = state["phase"] == "shop" and side == "player"
        hp_bar = combat or (state["phase"] == "reward" and side == "player")
        substitute = state[f"{side}_substitute"]["hp"] if hp_bar else 0
        hp_visible = hp_bar and substitute <= 0
        if hp_visible:
            hp, age = actor["hp"], 0
            known = True
        else:
            hp, seen_step = self.last_hp.get(side, (0, self.actions))
            age = self.actions - seen_step
            known = side in self.last_hp
        result = [float(hp_visible), float(known), _scaled(hp, 100), _scaled(age, 10),
                  float(substitute > 0), _scaled(substitute, 20)]
        result.extend(_one_hot(actor["type"] if combat or shop_player else None, (None, "Normal", *TYPES)))
        for stat in COMBAT_STATS:
            visible = combat or (shop_player and stat in EXTRA_STATS)
            result.extend((float(visible), _scaled(actor[stat], 100) if visible else 0.0))
        status = {item["Name"] for item in actor["status_conditions"]} if hp_visible else set()
        result.extend(float(name in status) for name in self.statuses)
        return result

    def observe(self, state):
        """Encode only the fields allowed for the current decision screen."""
        phase = state["phase"]
        p = state["player"]
        shop = self._shop_public(state) if phase == "shop" else None
        stage = PHASES.index(phase if phase in PHASES else self.last_phase)
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        mask[[self._action_ids[action] for action in self._public_actions(state, shop)]] = 1
        features = [state["floor"] / 100, _scaled(state["gold"], 1000),
                    _scaled(state["turn"], 10) if phase == "combat" else 0,
                    _scaled(state["wins"], 10), _scaled(self.known_player_max_hp, 100)]
        features.extend(_one_hot(state["area"], self.areas))
        features.extend(float(area in self.visited) for area in self.areas)
        features.extend(_one_hot(shop["menu"] if shop else None, (None, "main", *SHOP_MENUS)))
        features.extend(_one_hot(self.ella_choice, (None, "pickpocket", "help")))
        features.extend(_one_hot(self.previous_action, self._verbs))
        features.extend(_one_hot(self.previous_skill, (*self.skill_tokens, "sacrifice")))
        name = self.last_enemy_name if phase in {"combat", "reward", "event"} else None
        if phase == "combat" and state["monster_substitute"]["hp"] <= 0:
            name = state["monster"]["name"]
        features.extend(_one_hot(name, self.enemy_names))
        features.extend(self._actor_features(state, "player"))
        features.extend(self._actor_features(state, "monster"))
        features.extend(_one_hot(state["reward_skill"] if phase == "reward" else None, self.skill_tokens))
        learned = set(p["learned_skills"])
        for skill in self.catalog_skills:
            owned = skill in learned
            features.extend((float(owned), _scaled(p["skill_costs"][skill], 10) if owned else 0.0))
        skills_menu = bool(shop and shop["menu"] == "skills")
        slots_visible = phase == "combat" or skills_menu
        slot_records = ([(skill, p["memorized_skills"].get(skill, 0)) for skill in p["current_skills"]]
                        if slots_visible else self.last_slots)
        features.extend((float(slots_visible), float(slot_records is not None)))
        for slot in self.slots:
            skill, uses = slot_records[slot] if slot_records is not None else (None, 0)
            features.extend(_one_hot(skill, self.skill_tokens))
            features.append(_scaled(uses, 10))
        known_potions = sum(item in p["purchases"] for item in self.potions)
        mana_config = self.data.config["mana"]
        max_mana = p["max_mana"] if skills_menu else mana_config["max"] + known_potions * mana_config["potion_increment"]
        features.extend((float(skills_menu), _scaled(shop["mana"], 100) if skills_menu else 0.0, _scaled(max_mana, 100)))
        stats_menu = bool(shop and shop["menu"] == "stats")
        features.extend((float(stats_menu), _scaled(shop["points"], 100) if stats_menu else 0.0))
        features.extend(_scaled(shop["base_stats"][key], 100) if stats_menu else 0.0 for key in MAIN_STATS)
        merch = bool(shop and shop["menu"] == "merchandise")
        features.extend((float(merch), _scaled(shop["reroll_price"], 100) if merch else 0.0))
        features.extend(float(item in p["purchases"]) for item in self.items)
        return {"features": np.asarray(features, dtype=np.float32), "phase": np.int64(stage), "action_mask": mask}

    def _decode(self, action):
        verb, *args = self.action_catalog[action]
        if verb in {"cast_slot", "memorize", "forget", "replace_skill"}:
            slot, *rest = args
            skill = self._state["player"]["current_skills"][slot]
            return ("skill" if verb == "cast_slot" else verb, skill, *rest)
        return (verb, *args)

    def legal_ids_in_engine_order(self):
        """Preserve seeded random-baseline order after verifying public legality."""
        ids = [self._action_ids[action] for action in self._public_actions(self._state)]
        by_tuple = defaultdict(deque)
        for action in ids:
            by_tuple[self._decode(action)].append(action)
        engine = self.game.get_legal_actions()
        ordered = []
        for action in engine:
            if not by_tuple[action]:
                raise RuntimeError("Public legal-action projection lacks an engine action")
            ordered.append(by_tuple[action].popleft())
        if len(ordered) != len(ids):
            raise RuntimeError("Public legal-action projection has an extra action")
        return ordered

    def _deadlocked(self, state):
        p, m = state["player"], state["monster"]
        return (state["phase"] == "combat" and m["monster_id"] == 21
                and m["available_skills"] == ["mimic"] and p["speed"] > m["speed"]
                and p["hp"] > 0 and m["hp"] > 0
                and not p["status_conditions"] and not m["status_conditions"]
                and state["player_substitute"]["hp"] == state["monster_substitute"]["hp"] == 0
                and len(self._public_actions(state)) == 1
                and {skill for skill in p["current_skills"] if skill != LOCK and p["memorized_skills"][skill] > 0}
                == {"counter"})

    def _target_met(self, before, after):
        match self.stage["target"]:
            case "next_floor":
                return after["floor"] > self.start_floor
            case "floor":
                return after["floor"] >= self.stage["floor"]
            case "boss":
                return before["floor"] == self.stage["floor"] and after["wins"] > before["wins"]
            case "clear":
                return after["phase"] == "cleared"

    def _info(self, outcome=None, deadlocked=False):
        return {"floor": self.furthest_floor, "target_success": self.target_success,
                "outcome": outcome, "deadlock_mimic_counter": deadlocked}

    def step(self, action):
        if self.finished:
            raise gym.error.ResetNeeded("Reset after a terminal or truncated episode")
        if not self.action_space.contains(action) or self.action_catalog[action] not in self._public_actions(self._state):
            raise ValueError(f"Action {action!r} is masked in the current phase")
        before = self._state
        decoded = self._decode(action)
        after = self._state = self.game.step(decoded)
        self.actions += 1
        self.previous_action = self.action_catalog[action][0]
        self.previous_skill = decoded[1] if decoded[0] in {"skill", "memorize", "forget", "reduce_skill_cost"} else None
        if decoded[0] == "ella":
            self.ella_choice = decoded[1]
        new_encounter = after["phase"] == "combat" and after["turn"] == 0
        self._record_visible(after, new_encounter=new_encounter)
        if after["phase"] in PHASES:
            self.last_phase = after["phase"]
        target_hit = not self.target_success and self._target_met(before, after)
        self.target_success |= target_hit
        floor_progress = max(0, after["floor"] - self.furthest_floor)
        self.furthest_floor += floor_progress
        terminal = self.game.is_terminal()
        capped = self.actions >= self.action_limit
        deadlocked = not terminal and (self.early_deadlock or capped) and self._deadlocked(after)
        outcome = None
        if terminal:
            outcome = "clear" if self.game.is_cleared() else "death"
        elif self.early_deadlock and deadlocked:
            outcome = "deadlock"
        elif capped:
            outcome = "timeout"
        truncated = outcome in {"deadlock", "timeout"}
        self.finished = terminal or truncated
        reward = (self.weights.get("target", 0.0) * int(target_hit)
                  + self.weights.get("floor", 0.0) * floor_progress
                  + self.weights.get("clear", 1.0) * int(outcome == "clear")
                  + self.weights.get("death", 0.0) * int(outcome == "death"))
        return self.observe(after), float(reward), terminal, truncated, self._info(outcome, deadlocked)

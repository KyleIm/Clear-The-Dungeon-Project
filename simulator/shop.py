from __future__ import annotations

from typing import TYPE_CHECKING

from . import combat

if TYPE_CHECKING:
    from .game import Game


_MAIN_STATS = ("attack", "defense", "hp", "speed")
_ADDITIONAL_STATS = ("EVA", "ACC", "DR")
_KEYS = ("blue_key", "red_key")
_TYPES = ("Undead", "Elemental", "Beast")


def enter(game: Game) -> None:
    player = game.player
    combat.recover(player)
    combat.reset_stats(player)
    player.memorized_skills = player.saved_skills.copy()
    game.shop_menu = "main"


def leave(game: Game) -> None:
    game.player.saved_skills = game.player.memorized_skills.copy()
    game.reroll_cost = game.data.config["shop"]["prices"]["reroll_initial"]


def _unequipped_skills(game: Game) -> list[str]:
    player = game.player
    locked = game.data.config["slots"]["locked_marker"]
    return [skill for skill in player.learned_skills if skill != locked and skill not in player.current_skills]


def legal_actions(game: Game) -> list[tuple]:
    if game.shop_menu == "main":
        return [("shop_menu", menu) for menu in ("skills", "costs", "stats", "merchandise")] + [("leave_shop",)]

    player = game.player
    config = game.data.config
    locked = config["slots"]["locked_marker"]
    shop = config["shop"]
    prices = shop["prices"]
    actions: list[tuple] = []
    if game.shop_menu == "skills":
        unequipped = _unequipped_skills(game)
        for skill in player.current_skills:
            if skill == locked:
                continue
            if player.current_mana >= player.skill_costs[skill]:
                actions.append(("memorize", skill))
            if player.memorized_skills[skill] > 0:
                actions.append(("forget", skill))
            actions.extend(("replace_skill", skill, replacement) for replacement in unequipped)
    elif game.shop_menu == "costs" and game.gold >= prices["skill_cost_reduction"]:
        amount = shop["skill_cost_reduction_amount"]
        minimum = shop["minimum_reduced_skill_cost"]
        for skill in player.learned_skills:
            if skill != locked:
                cost = player.skill_costs[skill]
                if cost == game.data.skills[skill]["mana_cost"] and cost - amount >= minimum:
                    actions.append(("reduce_skill_cost", skill))
    elif game.shop_menu == "stats":
        minimum = shop["redistribution_minimum"]
        for stat in _MAIN_STATS:
            value = getattr(player, f"base_{stat}")
            for step in shop["redistribution_steps"]:
                if player.available_points >= step:
                    actions.append(("adjust_stat", stat, step))
                if value - step >= minimum:
                    actions.append(("adjust_stat", stat, -step))
        actions.append(("reset_redistribution",))
    elif game.shop_menu == "merchandise":
        additional = player.base_EVA + player.base_ACC + player.base_DR
        if game.gold >= game.reroll_cost and additional >= shop["reroll_min_points"]:
            actions.append(("reroll",))
        purchase_groups = [_KEYS, shop["mana_potions_in_order"]]
        for items in purchase_groups:
            for item in items:
                if item in player.purchases:
                    continue
                if game.gold >= prices[item] and (item not in _KEYS or locked in player.current_skills):
                    actions.append(("buy", item))
                break
        if player.type != "Normal" and game.gold >= prices["type_change"]:
            actions.extend(("change_type", target) for target in _TYPES if target not in (player.type, player.base_type))
    return actions + [("shop_back",)]


def _buy(game: Game, item: str) -> None:
    player = game.player
    config = game.data.config
    game.gold -= config["shop"]["prices"][item]
    if item in _KEYS:
        candidates = _unequipped_skills(game)
        if not candidates:
            # The shop charges before looking for a skill to fill the slot.
            return
        skill = candidates[0]
        locked = config["slots"]["locked_marker"]
        player.current_skills[player.current_skills.index(locked)] = skill
        player.memorized_skills[skill] = 0
    else:
        player.max_mana += config["mana"]["potion_increment"]
    player.purchases.add(item)


def apply(game: Game, action: tuple) -> None:
    player = game.player
    config = game.data.config
    shop = config["shop"]
    verb = action[0]
    if verb == "shop_menu":
        game.shop_menu = action[1]
    elif verb == "shop_back":
        if game.shop_menu == "stats":
            combat.reset_stats(player)
            combat.recover(player)
        game.shop_menu = "main"
    elif verb == "memorize":
        player.memorized_skills[action[1]] += 1
    elif verb == "forget":
        player.memorized_skills[action[1]] -= 1
    elif verb == "replace_skill":
        _, old, new = action
        player.current_skills[player.current_skills.index(old)] = new
        locked = config["slots"]["locked_marker"]
        player.memorized_skills = {
            skill: 0 if skill == new else player.memorized_skills[skill]
            for skill in player.current_skills if skill != locked
        }
    elif verb == "reduce_skill_cost":
        player.skill_costs[action[1]] -= shop["skill_cost_reduction_amount"]
        game.gold -= shop["prices"]["skill_cost_reduction"]
    elif verb == "adjust_stat":
        _, stat, points = action
        field = f"base_{stat}"
        setattr(player, field, getattr(player, field) + points)
        player.available_points -= points
    elif verb == "reset_redistribution":
        minimum = shop["redistribution_minimum"]
        for stat in _MAIN_STATS:
            field = f"base_{stat}"
            player.available_points += getattr(player, field) - minimum
            setattr(player, field, minimum)
    elif verb == "reroll":
        total = sum(getattr(player, f"base_{stat}") for stat in _ADDITIONAL_STATS)
        low, high = sorted((game.reroll_rng.randint(0, total), game.reroll_rng.randint(0, total)))
        for stat, points in zip(_ADDITIONAL_STATS, (low, high - low, total - high)):
            setattr(player, f"base_{stat}", points)
            setattr(player, stat, points)
        game.gold -= game.reroll_cost
        game.reroll_cost *= shop["reroll_multiplier"]
    elif verb == "buy":
        _buy(game, action[1])
    elif verb == "change_type":
        player.base_type = player.type = action[1]
        game.gold -= shop["prices"]["type_change"]
    else:
        raise ValueError(f"Unknown shop action: {verb!r}")

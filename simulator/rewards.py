from __future__ import annotations

from typing import TYPE_CHECKING

from . import combat

if TYPE_CHECKING:
    from .game import Game


_ADDITIONAL_STATS = {"evade": "EVA", "accuracy": "ACC", "damage_reduction": "DR"}


def prepare(game: Game) -> None:
    """Offer one of the defeated enemy's last available skills."""
    game.reward_candidates = [
        skill for skill in game.monster.available_skills if skill not in game.player.learned_skills
    ]
    game.reward_skill = None
    if game.reward_candidates:
        game.reward_skill = game.rng.choice(game.reward_candidates)


def legal_actions(game: Game) -> list[tuple]:
    return [("reward", option) for option in game.data.config["rewards"]["options"]]


def apply(game: Game, action: tuple) -> None:
    player = game.player
    config = game.data.config
    reward = config["rewards"]
    _, option = action
    if option == "Gold":
        game.gold += reward["gold"]
    elif option == "Potion":
        combat.recover(player)
    elif option == "Initialize":
        combat.reset_stats(player)
    elif option == "Stat":
        stat = game.rng.choice(reward["stat_candidates"])
        attribute = _ADDITIONAL_STATS[stat]
        bound = {"evade": "evade_bounds", "accuracy": "accuracy_bounds"}.get(stat)
        for field in (f"base_{attribute}", attribute):
            value = getattr(player, field)
            if bound is None or value < config["combat"][bound][1]:
                setattr(player, field, value + reward["stat_increment"])
    elif option == "Exchange":
        if game.reward_candidates:
            game.reward_skill = game.rng.choice(game.reward_candidates)
    else:
        raise ValueError(f"Unknown reward: {option!r}")

    if game.reward_skill is not None:
        player.learned_skills.append(game.reward_skill)

"""Resolve initiative, action wrappers, and status timing for one combat turn."""
from .skills import cure_status_effects, resolve_skill, substitute_for


def recover(character):
    character.hp = character.max_hp


def reset_stats(character):
    cure_status_effects(character)
    for stat in ("attack", "defense", "speed", "ACC", "EVA", "DR", "type"):
        setattr(character, stat, getattr(character, f"base_{stat}"))


def _start_status_effects(game, actor, opponent):
    config = game.data.config["status"]
    values = config["registry_values"]
    for effect in actor.status_conditions:
        name = effect["Name"]
        if name == "stun" and actor.selected_action != "recovery":
            if game.rng.randint(1, 100) < config["stun_skip_threshold"]:
                return True
        elif name == "sleep" and actor.selected_action != "recovery":
            return True
        elif name in {"poison", "sprout"}:
            amount = min(actor.hp - config["minimum_hp_after_poison_or_sprout"], actor.max_hp // values[name])
            actor.hp -= amount
            if name == "sprout":
                opponent.hp = min(opponent.max_hp, opponent.hp + amount)
        elif name == "regeneration":
            actor.hp = min(actor.max_hp, actor.hp + actor.max_hp // values[name])
    return False


def _end_status_effects(game, actor):
    config = game.data.config["status"]
    steps = config["cure_counter_steps"]
    # The live list is intentional. A deletion advances past the following record.
    for effect in actor.status_conditions:
        name = effect["Name"]
        chance = effect["chance_to_cure"]
        if name in {"stun", "poison", "sprout", "sleep"}:
            cured = game.rng.randint(1, 100) <= chance
        elif name == "regeneration":
            cured = game.rng.randint(1, 100) == chance
            if not cured and chance == 0:
                effect["chance_to_cure"] = config["regeneration_cure_roll"]
        elif name in steps:
            cured = chance == (99 if name == "helmet" else 100)
        else:
            continue

        if cured:
            actor.status_conditions.remove(effect)
        elif name in steps:
            effect["chance_to_cure"] += steps[name]


def _execute_counter(game, actor, opponent):
    damage = min(opponent.hp, actor.ram_damage + actor.ram_damage // 2)
    if damage > 0:
        substitute = substitute_for(game, opponent)
        protected = opponent.substitute_hit_flag if actor is game.player else substitute.hp > 0
        if not protected:
            opponent.hp -= damage
        if actor is game.player and substitute.hp == 0:
            opponent.substitute_hit_flag = 0


def _execute_action(game, actor, opponent):
    if _start_status_effects(game, actor, opponent):
        return

    skill_name = actor.selected_action
    if actor is game.player and actor.mm_button == 0 and skill_name != "sacrifice":
        actor.memorized_skills[skill_name] -= 1

    result = resolve_skill(game, actor, opponent, skill_name)
    if result.kind == "mimic":
        if opponent.ram_action not in {"mimic", "counter"}:
            actor.selected_action = opponent.ram_action
            if actor is game.player:
                actor.mm_button = 1
            try:
                _execute_action(game, actor, opponent)
            finally:
                actor.mm_button = 0
    elif result.kind == "counter":
        _execute_counter(game, actor, opponent)
    elif result.kind == "damage":
        if actor is game.player and game.monster_substitute.hp == 0:
            opponent.substitute_hit_flag = 0
        opponent.wake_up()
        if result.second_hit and opponent.hp > 0:
            opponent.hp = max(0, opponent.hp - result.damage)
        opponent.ram_damage = result.damage
        if result.status is not None:
            opponent.apply_status_effect(result.status)
        if skill_name == "gold rush":
            game.gold += game.floor * 100


def _priority(game, skill):
    category = game.data.skills[skill]["category"]
    config = game.data.config["combat"]["priority"]
    if category in config["high_categories"]:
        return 1
    if category in config["low_categories"]:
        return -1
    return config["default"]


def resolve_turn(game, player_skill, enemy_skill):
    """Execute two selected skills in source order and apply the end-of-turn status phase."""
    player, monster = game.player, game.monster
    for actor, skill in ((player, player_skill), (monster, enemy_skill)):
        actor.selected_action = actor.ram_action = skill
        actor.priority_level = _priority(game, skill)

    player_key = player.priority_level, player.speed
    monster_key = monster.priority_level, monster.speed
    if player_key == monster_key:
        player_first = game.rng.choice((True, False))
    else:
        player_first = player_key > monster_key
    first, second = (player, monster) if player_first else (monster, player)

    if first.priority_level >= 0:
        _execute_action(game, first, second)
    if player.hp > 0 and monster.hp > 0:
        _execute_action(game, second, first)

    player.priority_level = monster.priority_level = 0
    _end_status_effects(game, player)
    _end_status_effects(game, monster)
    player.ram_damage = monster.ram_damage = 0
    return ("player", "monster") if player_first else ("monster", "player")

"""Interpret the skill registry and mutate the caster or actual target."""
from dataclasses import dataclass

BASE_BUFFS = {
    10: ("attack",), 11: ("defense",), 12: ("speed",),
    30: ("attack", "defense"), 31: ("attack",), 32: ("attack", "speed"),
}
BASE_DEBUFFS = {15: ("attack",), 16: ("defense",), 17: ("speed",), 20: ("attack", "defense")}
SELF_STATUSES = {24: "regeneration", 28: "aurora", 29: "immunity", 39: "helmet"}
TARGET_STATUSES = {21: "stun", 22: "poison", 23: "sprout", 25: "target", 26: "sleep", 27: "reverse"}
BODY_ONLY = BASE_DEBUFFS.keys() | TARGET_STATUSES.keys() | {18, 19}
FORMULA_DAMAGE = {0, 1, 2, 3, 4, 40, 42, 43, 44, 46, 48, 49, 51, 52, 53, 54, 55, 58, 59, 98}
FIXED_DAMAGE = {6, 45, 57, 60}
TYPES_BY_CASTER = {
    "Undead": ("Beast", "Elemental"),
    "Elemental": ("Beast", "Undead"),
    "Beast": ("Undead", "Elemental"),
    "Normal": ("Undead", "Elemental", "Beast"),
}
COPIED_STATS = (
    "base_attack", "attack", "base_defense", "defense", "base_ACC", "ACC",
    "base_EVA", "EVA", "base_DR", "DR", "type",
)


@dataclass(frozen=True)
class SkillEffect:
    kind: str = "effect"
    damage: int = 0
    second_hit: bool = False
    status: str | None = None


def substitute_for(game, character):
    return game.player_substitute if character is game.player else game.monster_substitute


def damage_output(caster, target, skill, rules):
    category, element = skill["category"], skill["type"]
    multiplier = 1.0
    if element and caster.type and target.type:
        if caster.type == element:
            multiplier *= rules["matching_type_multiplier"]
        advantage = rules["type_advantage"]
        favorable = [element, target.type] in advantage or (
            category == 54 and target.type == rules["judgement_extra_advantage"]
        )
        if favorable or [target.type, element] in advantage:
            if favorable != caster.has_status_effect("reverse"):
                multiplier *= rules["favorable_multiplier"]
            else:
                ratio = rules["unfavorable_multiplier"]
                multiplier *= ratio["numerator"] / ratio["denominator"]

    numerator = {51: target.attack, 52: caster.speed, 53: caster.defense}.get(category, caster.attack)
    divisor = target.defense
    if target.has_status_effect("aurora"):
        divisor *= rules["aurora_damage_divisor"]
    return max(0, int(skill["power"] * numerator / divisor * multiplier) - target.DR)


def _apply_status(character, name):
    if not character.has_status_effect(name) and not character.has_status_effect("immunity"):
        character.apply_status_effect(name)


def _change_base_stats(character, stats, power, ranks):
    for stat in stats:
        base = getattr(character, f"base_{stat}")
        value = getattr(character, stat) + ranks * base // power if ranks > 0 else max(
            1, getattr(character, stat) - base // power
        )
        setattr(character, stat, value)


def _heal(character, amount):
    character.hp = min(character.max_hp, character.hp + amount)


def _recoil(character, damage):
    if not character.has_status_effect("helmet"):
        character.hp = max(0, character.hp - damage // 2)


def _imitate(game, caster, opponent):
    if caster.imitation_backup is None:
        fields = COPIED_STATS + ("current_skills", "memorized_skills", "available_skills")
        caster.imitation_backup = {name: getattr(caster, name) for name in fields}
    for name in COPIED_STATS:
        setattr(caster, name, getattr(opponent, name))

    slots = game.data.config["slots"]
    locked = slots["locked_marker"]
    if caster is game.player:
        record = game.data.monsters[opponent.monster_id]
        copied = list(record["skills"] or game.ella_skills)
        caster.current_skills = copied + [locked] * (slots["total"] - len(copied))
        caster.memorized_skills = dict.fromkeys(copied, slots["imitation_uses_per_skill"])
    else:
        caster.current_skills = [name for name in opponent.current_skills if name != locked]
        caster.available_skills = caster.current_skills.copy()

    if not caster.has_status_effect("imitation"):
        caster.apply_status_effect("imitation")


def cure_status_effects(character):
    """Apply the immediate cure, preserving the source's in-place cure ordering."""
    if character.imitation_backup is not None:
        for name, value in character.imitation_backup.items():
            setattr(character, name, value)
        character.imitation_backup = None

    for effect in character.status_conditions:
        character.status_conditions.remove(effect)

    # A skipped imitation record must not leave the copied deck reported as active.
    character.status_conditions[:] = [effect for effect in character.status_conditions if effect["Name"] != "imitation"]


def _amount(caster, target, skill, rules):
    category, power = skill["category"], skill["power"]
    if category in FORMULA_DAMAGE:
        return damage_output(caster, target, skill, rules)
    if category in FIXED_DAMAGE:
        return power
    if category == 7:
        return target.hp // 2
    if category == 41:
        return target.hp - 1
    if category == 56:
        return max(0, power - target.DR)
    if category == 99:
        return caster.base_hp // 2
    raise ValueError(f"Unhandled skill category {category}")


def _resolve_damage(game, caster, opponent, target, skill, threshold):
    category = skill["category"]
    rules = game.data.config["combat"]
    damage = min(target.hp, _amount(caster, target, skill, rules))
    target.hp -= damage

    if category in {2, 57}:
        # Both rolls use the initial target's evasion. The action wrapper applies the second hit to the body.
        if game.rng.randint(1, 100) <= threshold and (category == 57 or opponent.hp > 0):
            next_target = target if target.hp > 0 else opponent
            damage = min(next_target.hp, _amount(caster, next_target, skill, rules))
            return SkillEffect("damage", damage, second_hit=True)
    elif category in {3, 45, 99}:
        _recoil(caster, damage)
    elif category in {4, 42}:
        _heal(caster, damage // 2 if category == 4 else damage)
    elif category == 40:
        _change_base_stats(caster, ("defense",), 4, 1)
    elif category == 43 and target is opponent:
        target.ACC = max(rules["accuracy_bounds"][0], target.ACC - 10)
    elif category == 44:
        caster.attack -= caster.base_attack // 4
    elif category == 48:
        _change_base_stats(caster, ("attack",), 4, 1)
    elif category == 49:
        caster.ACC = max(rules["accuracy_bounds"][0], caster.ACC - 10)
    elif category == 56:
        target.DR = max(0, target.DR - 5)
    elif category in {58, 59, 60} and target is opponent and not target.has_status_effect("immunity"):
        status = _on_hit_status(game, target, category)
        return SkillEffect("damage", damage, status=status)

    return SkillEffect("damage", damage)


def _on_hit_status(game, target, category):
    if category == 60:
        if target.hp > 0 and not any(target.has_status_effect(name) for name in ("stun", "poison")):
            return game.rng.choice(("stun", "poison"))
        return None

    name, rule = ("stun", "icebolt_stun_percent") if category == 58 else ("poison", "poison_tail_poison_percent")
    chance = game.data.config["status"][rule]
    # Icebolt and Poison Tail roll before testing for a knockout.
    if not target.has_status_effect(name) and game.rng.randint(1, 100) <= chance and target.hp > 0:
        return name
    return None


def _resolve_self_effect(game, caster, opponent, skill):
    """Return whether the skill activates without the standard target roll."""
    category, power = skill["category"], skill["power"]
    rules = game.data.config["combat"]
    if category in BASE_BUFFS:
        _change_base_stats(caster, BASE_BUFFS[category], power, 1)
        if category == 31:
            caster.ACC = min(rules["accuracy_bounds"][1], caster.ACC + 10)
    elif category == 13:
        caster.ACC = min(rules["accuracy_bounds"][1], caster.ACC + power)
    elif category == 14:
        caster.EVA = min(rules["evade_bounds"][1], caster.EVA + power)
    elif category in SELF_STATUSES:
        _apply_status(caster, SELF_STATUSES[category])
    elif category == 33:
        if game.rng.randint(1, 100) <= skill["accuracy"] + caster.ACC:
            _change_base_stats(caster, ("attack", "speed"), power, 1)
            caster.defense += caster.base_attack // power
    elif category == 34:
        stats = ["attack", "defense", "speed"]
        game.rng.shuffle(stats)
        _change_base_stats(caster, (stats[0],), 4, 3)
        _change_base_stats(caster, (stats[1],), 4, -1)
    elif category == 35:
        if caster.hp > caster.base_hp:
            caster.hp -= caster.base_hp
            caster.attack += caster.base_attack
    elif category == 36:
        caster.DR += 5
    elif category == 37:
        _change_base_stats(caster, ("defense",), power, -1)
        caster.hp = caster.max_hp
    elif category == 47:
        caster.type = game.rng.choice(TYPES_BY_CASTER.get(caster.type, TYPES_BY_CASTER["Normal"]))
    elif category == 61:
        _heal(caster, caster.max_hp // power)
    elif category == 62:
        cure_status_effects(caster)
    elif category == 96:
        _imitate(game, caster, opponent)
    elif category == 97:
        substitute = substitute_for(game, caster)
        config = rules["substitute"]
        if caster.hp > config["cost_hp"] and substitute.hp <= 0:
            caster.hp -= config["cost_hp"]
            substitute.hp = config["hp"]
            if caster is game.monster:
                caster.substitute_hit_flag = 1
    else:
        return False
    return True


def resolve_skill(game, caster, opponent, skill_name):
    """Resolve one already-charged action, returning any action-wrapper effects."""
    skill = game.data.skills[skill_name]
    category, power = skill["category"], skill["power"]
    if category == 50:
        return SkillEffect("mimic")
    if _resolve_self_effect(game, caster, opponent, skill):
        return SkillEffect()

    substitute = substitute_for(game, opponent)
    target = substitute if substitute.hp > 0 else opponent
    rules = game.data.config["combat"]
    threshold = skill["accuracy"] + caster.ACC - target.EVA
    if category in BODY_ONLY and target is substitute:
        threshold = 0
    elif target.has_status_effect("target"):
        threshold = rules["target_hit_threshold"]
    if category != 99 and game.rng.randint(1, 100) > threshold:
        if category == 46:
            if caster.has_status_effect("helmet"):
                return SkillEffect("damage")
            _recoil(caster, damage_output(caster, target, skill, rules))
        return SkillEffect()

    if category == 5:
        return SkillEffect("counter")
    if category == 98 and caster.hp > caster.base_hp:
        return SkillEffect()
    if category in BASE_DEBUFFS:
        _change_base_stats(target, BASE_DEBUFFS[category], power, -1)
    elif category == 18:
        target.ACC = max(rules["accuracy_bounds"][0], target.ACC - power)
    elif category == 19:
        target.EVA = max(rules["evade_bounds"][0], target.EVA - power)
    elif category in TARGET_STATUSES:
        _apply_status(target, TARGET_STATUSES[category])
        if category == 27:
            _apply_status(caster, TARGET_STATUSES[category])
    elif category == 38:
        stolen = min(target.attack - 1, target.base_attack // power)
        target.attack -= stolen
        _heal(caster, stolen)
    else:
        return _resolve_damage(game, caster, opponent, target, skill, threshold)
    return SkillEffect()

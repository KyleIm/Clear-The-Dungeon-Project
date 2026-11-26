"""Build the live enemy deck and choose its action for a turn."""


def _available_skills(game):
    monster = game.monster
    if monster.has_status_effect("imitation"):
        return monster.current_skills

    if monster.name == "Ella":
        skills = game.ella_skills
    else:
        skills = list(game.data.monsters[monster.monster_id]["skills"])
    for upgrade in game.data.config["enemy_skill_upgrades"]:
        if game.floor >= upgrade["minimum_floor"]:
            slot = upgrade["slot"]
            skills[slot] = upgrade["substitutions"].get(skills[slot], skills[slot])
            break
    return skills


def _ella_action(game, skills):
    monster, player, rng = game.monster, game.player, game.rng
    if game.floor < 75:
        # An early random choice on floor 35 is overwritten, but still advances the RNG.
        if game.floor == 35 and monster.hp >= 0.4 * monster.base_hp and player.ACC >= 0:
            rng.choice(skills)
        return rng.choice(skills)

    if any(monster.has_status_effect(name) for name in ("stun", "sleep", "poison")):
        return skills[1]
    if monster.speed > player.speed and monster.hp == monster.max_hp:
        return skills[3]
    if monster.hp > 1.6 * monster.base_hp and player.hp > player.base_hp:
        return skills[2]
    return skills[0]


def _choose_action(game, skills):
    monster, player, rng = game.monster, game.player, game.rng
    accuracy_margin = monster.ACC - player.EVA
    match monster.key:
        case "lionheart":
            if monster.hp > 1.6 * monster.base_hp:
                return skills[3]
            if player.attack < 30 and player.speed < 120 and monster.hp > monster.base_hp:
                return skills[2]
            return rng.choice(skills[:2])
        case "ogre":
            return rng.choice(skills if player.type == "Undead" else skills[:3])
        case "tarrasque":
            return skills[3] if player.DR > 10 else rng.choice(skills[:3])
        case "queen":
            return rng.choice(skills[:3]) if monster.has_status_effect("aurora") else skills[3]
        case "superminion":
            if accuracy_margin >= 20:
                return skills[3]
            if accuracy_margin < 0:
                return skills[2]
            return rng.choice(skills[:2])
        case "vampire":
            if 9 < player.defense < player.base_defense and monster.hp > monster.base_hp:
                return skills[2]
            if monster.hp < monster.base_hp:
                return skills[3]
            return rng.choice(skills[:2])
        case "grimreaper":
            if accuracy_margin >= 30 and player.hp > player.base_hp:
                return skills[3]
            if accuracy_margin < 0:
                return skills[2]
            return rng.choice(skills[:2])
        case "dragon":
            if monster.has_status_effect("stun") or monster.has_status_effect("sleep"):
                return skills[1]
            if monster.speed == monster.base_speed and 90 < player.speed < 112:
                return skills[2]
            if monster.hp < monster.base_hp:
                return skills[3]
            return skills[0]
        case _:
            if monster.name == "Ella":
                return _ella_action(game, skills)
            return rng.choice(skills)


def select_action(game):
    monster = game.monster
    monster.available_skills = _available_skills(game)
    action = _choose_action(game, monster.available_skills)
    monster.selected_action = monster.ram_action = action
    return action

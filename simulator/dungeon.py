"""Generate regions, encounters, and Ella's decks from the run state."""

from .combat import recover, reset_stats


def select_area(game):
    late_game = game.data.areas["late_game"]
    if game.floor == late_game["final_floor"]:
        game.area = "Lair"
    elif game.floor == late_game["gate_first_floor"]:
        game.area = "Gate"
    else:
        game.rng.shuffle(game.remaining_areas)
        game.area = game.remaining_areas.pop(0)


def _initial_skill_group(game):
    groups = game.data.config["initial_skills"]["area_groups"]
    return next(group for group in groups if game.area in group["areas"])


def initial_skills(game):
    config = game.data.config["initial_skills"]
    slots = game.data.config["slots"]
    group = _initial_skill_group(game)
    pools = config["attack_pools"]
    attacks = list(pools[game.rng.randint(0, len(pools) - 1)])
    ranks = list(config["rank_pool"])
    game.rng.shuffle(attacks)
    game.rng.shuffle(ranks)

    equipped = [group["player"], *attacks[:2], ranks[0]]
    locks = [slots["locked_marker"]] * (slots["total"] - len(equipped))
    player = game.player
    player.current_skills = equipped + locks
    player.learned_skills = player.current_skills.copy()
    player.memorized_skills = dict(zip(equipped, config["player_uses"]))
    player.saved_skills = player.memorized_skills.copy()
    game.ella_skills = [group["ella"], *attacks[2:4], ranks[1]]


def _set_ella_deck(game, monster_id):
    if game.floor == 5:
        return  # The first deck was drafted alongside the player's deck.

    player = game.player
    variants = game.data.monsters[monster_id]["skill_variants"]
    if game.floor == 35:
        group = _initial_skill_group(game)
        kind = {"skull dagger": "Undead", "wolf fang": "Beast", "thunder": "Elemental"}[group["player"]]
        variant = next(variant for variant in variants if variant["type"] == kind)
        deck = list(variant["skills"])
    else:
        if game.floor == 75:
            index = {"Beast": 0, "Elemental": 1}.get(player.type, 2)
        elif player.type == "Beast":
            index = 0
        elif player.type == "Elemental":
            index = 1 if player.EVA > 10 else 2
        else:
            index = 3
        variant = variants[index]
        skills = variant["skills"]
        deck = [*skills[:2], skills[2 if player.base_hp > 100 else 3],
                skills[4 if player.base_hp > player.base_attack else 5]]

    game.monster.type = variant["type"]
    game.ella_skills = deck


def spawn_encounter(game):
    floor = game.floor
    areas = game.data.areas
    late_game = areas["late_game"]
    ella = next((encounter for encounter in areas["ella_encounters"] if encounter["floor"] == floor), None)

    if floor == late_game["final_floor"]:
        game.monster = game.monster_roster[late_game["final_boss_id"]]
        recover(game.monster)
        reset_stats(game.monster)
        return

    if ella and (floor != late_game["gate_last_floor"] or (game.ella_path == 1 and game.ella_encounter == 1)):
        ids = ella["monster_ids"]
        monster_id = ids[1] if len(ids) == 2 and game.ella_path != 1 else ids[0]
        game.monster = game.monster_roster[monster_id]
        _set_ella_deck(game, monster_id)
        return

    if floor >= late_game["gate_first_floor"]:
        game.monster = game.monster_roster[game.gate_queue.pop(0)]
        reset_stats(game.monster)
        recover(game.monster)
        return

    rules = areas["floor_rules"]
    region = game.data.regions[game.area]
    chapter, remainder = divmod(floor, rules["modulus"])
    if remainder == rules["stage_boss_remainder"]:
        monster_id = region["boss_id"]
        game.gate_queue.append(monster_id)
        game.monster = game.monster_roster[monster_id]
        return

    if remainder == rules["entry_remainder"]:
        monster_id = region["entry_monster_id"]
    elif remainder == rules["midboss_remainder"]:
        monster_id = region["midboss_id"]
    else:
        pool = region["normal_monster_ids"]
        if floor < rules["reduced_pool_below_floor"]:
            pool = pool[:region["first_area_pool_size"]]
        monster_id = game.rng.choice(pool)

    monster = game.monster = game.monster_roster[monster_id]
    recover(monster)
    reset_stats(monster)
    if floor != game.data.config["player"]["starting_floor"]:
        total = chapter * rules["additional_stats"]["points_per_decade"]
        low, high = sorted((game.rng.randint(0, total), game.rng.randint(0, total)))
        monster.ACC += low
        monster.EVA += high - low
        monster.DR += total - high


def apply_first_ella_bonus(game):
    player = game.player
    match game.area:
        case "Forest" | "Highland":
            player.attack += 3 * player.base_attack // 4
        case "Fort":
            player.EVA += 50
            player.attack += player.base_attack // 4
        case "Graveyard":
            player.EVA += 50
            player.defense += player.base_defense // 4
        case "Mansion":
            player.attack += player.base_attack // 2
            player.defense += player.base_defense // 4
        case "Party" | "Spiritland":
            player.speed += player.base_speed // 4
            player.defense += player.base_defense // 2
        case "Ruin":
            player.defense += 3 * player.base_defense // 4
        case "Temple":
            player.ACC += 50
            player.speed += player.base_speed // 4

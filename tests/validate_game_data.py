"""Validate the Phase 1 static datasets. No third-party packages required."""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTES = ("attack", "defense", "hp", "speed", "evade", "accuracy", "damage_reduction")
TYPES = {"Normal", "Beast", "Undead", "Elemental"}
SOURCE_CHAR = re.compile(
    r'^default (\w+) =\s+Char\((?:_\()?["\']([^"\']+)["\']\)?, '
    r'base_attack=(\d+), base_defense=(\d+), base_hp=(\d+), base_speed=(\d+), '
    r'base_EVA=(\d+), base_ACC=(\d+), base_DR=(\d+), monster_index=(\d+), '
    r'monster_type=_\(["\'](\w+)["\']\)\)', re.M,
)


def require(condition, context):
    if not condition:
        raise ValueError(context)


def no_duplicate_keys(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def invalid_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def read_json(directory, name):
    return json.loads(
        (directory / f"{name}.json").read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicate_keys,
        parse_constant=invalid_constant,
    )


def integer(value):
    return type(value) is int


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def numeric_fields(mapping, names, context):
    for name in names:
        require(name in mapping and integer(mapping[name]), f"{context}.{name}: expected integer")


def stat_blocks(item, context):
    base, additional = item["base_stats"], item["additional_stats"]
    require(set(base) == set(ATTRIBUTES[:4]), f"{context}.base_stats: wrong keys")
    require(set(additional) == set(ATTRIBUTES[4:]), f"{context}.additional_stats: wrong keys")
    numeric_fields(base, base, context)
    numeric_fields(additional, additional, context)
    require(item["type"] in TYPES, f"{context}: unknown type")
    return tuple((base | additional)[key] for key in ATTRIBUTES)


def check_sequence_ids(rows, count, field):
    require(isinstance(rows, list) and len(rows) == count, f"{field}: expected {count} rows")
    ids = [row["id"] for row in rows]
    require(all(map(integer, ids)) and ids == list(range(1, count + 1)),
            f"{field}: missing, duplicate, or unordered ID")


def check_refs(refs, valid, label):
    require(isinstance(refs, list) and all(type(item) in (int, str) and item in valid for item in refs),
            f"{label}: invalid or unknown reference")


def check_rules(rules, label):
    require(isinstance(rules, list) and rules, f"{label}: expected ordered behavior rules")
    for rule in rules:
        require(nonempty(rule["condition"]), f"{label}: empty condition")
        require(nonempty(rule["selection"]), f"{label}: empty selection")


def fixed_integers(values, count, label):
    require(isinstance(values, list) and len(values) == count and all(map(integer, values)),
            f"{label}: expected {count} integers")


def validate_config(config, skill_names):
    player = config["player"]
    combat = config["combat"]
    numeric_fields(player, ("starting_unallocated_points", "starting_ella_path", "starting_ella_encounter"), "player")
    numeric_fields(combat, ("hp_multiplier", "target_hit_threshold"), "combat")
    require(player["max_hp"] == player["base_stats"]["hp"] * combat["hp_multiplier"], "wrong initial HP")
    for label in ("accuracy_bounds", "evade_bounds"):
        fixed_integers(combat[label], 2, label)
        require(combat[label][0] < combat[label][1], f"{label}: bounds out of order")
    for label in ("matching_type_multiplier", "favorable_multiplier", "aurora_damage_divisor"):
        require(type(combat[label]) in (int, float) and combat[label] > 0, f"{label}: expected positive number")
    numeric_fields(combat["unfavorable_multiplier"], ("numerator", "denominator"), "unfavorable multiplier")
    require(combat["unfavorable_multiplier"]["denominator"] > 0, "invalid type multiplier denominator")
    require(len(combat["type_advantage"]) == 3, "missing type matchup")
    for pair in combat["type_advantage"]:
        check_refs(pair, TYPES - {"Normal"}, "type matchup")
        require(len(pair) == 2 and pair[0] != pair[1], "invalid type matchup")
    require(combat["judgement_extra_advantage"] in TYPES, "unknown Judgement matchup")
    fixed_integers(combat["priority"]["high_categories"], 3, "high priority")
    fixed_integers(combat["priority"]["low_categories"], 2, "low priority")
    numeric_fields(combat["priority"], ("default",), "priority")
    substitute = combat["substitute"]
    stat_blocks(substitute, "substitute")
    numeric_fields(substitute, ("cost_hp", "hp"), "substitute")
    status = config["status"]
    numeric_fields(status, ("initial_cure_counter", "stun_skip_threshold", "regeneration_cure_roll",
                            "minimum_hp_after_poison_or_sprout", "icebolt_stun_percent",
                            "poison_tail_poison_percent"), "status")
    status_keys = {"stun", "poison", "sprout", "regeneration", "target", "sleep", "reverse",
                   "aurora", "immunity", "imitation", "helmet"}
    require(set(status["registry_values"]) == set(status["cure_conditions"]) == status_keys,
            "missing or unknown status key")
    require(all(map(nonempty, status["cure_conditions"].values())), "invalid cure rule")
    steps = status["cure_counter_steps"]
    require(set(steps) == status_keys - {"regeneration", "imitation"} and
            all(type(value) in (int, float) and value > 0 for value in steps.values()), "invalid cure steps")
    shop = config["shop"]
    prices = {"blue_key", "red_key", "blue_mana_potion", "elemental_mana_potion",
              "undead_mana_potion", "type_change", "skill_cost_reduction", "reroll_initial"}
    require(set(shop["prices"]) == prices, "missing or unknown shop price")
    numeric_fields(shop, ("reroll_multiplier", "reroll_min_points", "redistribution_minimum",
                          "skill_cost_reduction_amount", "minimum_reduced_skill_cost",
                          "skill_cost_reduction_limit_per_skill"), "shop")
    fixed_integers(shop["redistribution_steps"], 2, "redistribution steps")
    check_refs(shop["mana_potions_in_order"], prices, "mana potion order")
    check_refs(shop["reroll_allocation_order"], set(ATTRIBUTES[4:]), "reroll allocation")
    check_refs(config["rewards"]["stat_candidates"], set(ATTRIBUTES[4:]), "reward stats")
    for upgrade in config["enemy_skill_upgrades"]:
        numeric_fields(upgrade, ("minimum_floor", "slot"), "enemy skill upgrade")
        substitutions = upgrade["substitutions"]
        check_refs(list(substitutions) + list(substitutions.values()), skill_names, "enemy skill upgrade")
    rng = config["rng"]
    fixed_integers(rng["percentage_roll_inclusive"], 2, "RNG draw range")
    require(rng["explicit_seed_in_inspected_source"] is None, "unexpected explicit RNG seed")
    run = config["run"]
    numeric_fields(run, ("terminal_floor",), "run")
    require(type(run["rollback_enabled"]) is bool and nonempty(run["save_slot"]), "invalid run settings")


def validate(directory):
    skills, monsters, areas, config = [
        read_json(directory, key) for key in ("skills", "monsters", "areas", "game_config")
    ]
    check_sequence_ids(skills, 88, "skills")
    check_sequence_ids(monsters, 55, "monsters")
    skill_names = {item["name"] for item in skills}
    require(len(skill_names) == 88, "duplicate skill name")
    for item in skills:
        name = item["name"]
        require(nonempty(name) and item["type"] in TYPES, f"{name}: invalid name or type")
        numeric_fields(item, ("category", "power", "accuracy", "mana_cost", "source_line", "handler_line"), name)
        require(all(item[key] >= 0 for key in ("category", "power", "accuracy", "mana_cost")),
                f"{name}: negative registry value")
        require(nonempty(item["effect"]) and nonempty(item["condition"]), f"{name}: incomplete effect")
    monster_ids = {item["id"] for item in monsters}
    keys = {item["key"] for item in monsters}
    require(len(keys) == 55 and all(map(nonempty, keys)), "duplicate or empty monster key")
    for item in monsters:
        name = item["key"]
        require(nonempty(item["name"]), f"{name}: empty display name")
        stat_blocks(item, name)
        numeric_fields(item, ("source_line",), name)
        check_refs(item["skills"], skill_names, name)
        check_rules(item["action_rules"], name)
        variants = item.get("skill_variants", [])
        require(isinstance(variants, list), f"{name}: invalid skill variants")
        require(bool(variants) if item["id"] >= 51 else bool(item["skills"]) and not variants,
                f"{name}: missing or unexpected skill data")
        for slot in variants:
            require(nonempty(slot["condition"]), f"{name}: empty skill variant condition")
            require(slot["type"] in TYPES, f"{name}: invalid variant type")
            check_refs(slot["skills"], skill_names, name)
            require(slot["skills"], f"{name}: empty skill variant")

    regions = areas["regions"]
    require(isinstance(regions, list) and len(regions) == 11, "expected 11 regions")
    require(len({row["id"] for row in regions}) == len(regions), "duplicate region ID")
    for region in regions:
        region_id = region["id"]
        require(nonempty(region_id) and nonempty(region["selection_condition"]), f"{region_id}: invalid region")
        if region_id in ("Gate", "Lair"):
            require(nonempty(region["encounter_rule"]), f"{region_id}: missing encounter rule")
            continue
        numeric_fields(region, ("first_area_pool_size", "entry_monster_id", "midboss_id", "boss_id"), region_id)
        check_refs(region["normal_monster_ids"], monster_ids, region_id)
        check_refs([region[k] for k in ("entry_monster_id", "midboss_id", "boss_id")], monster_ids, region_id)
        require(0 < region["first_area_pool_size"] <= len(region["normal_monster_ids"]),
                f"{region_id}: invalid first-area pool size")
    late = areas["late_game"]
    numeric_fields(late, ("gate_first_floor", "gate_last_floor", "final_floor", "final_boss_id"), "late game")
    check_refs([late["final_boss_id"]], monster_ids, "final boss")
    floors = areas["floor_rules"]
    fixed_integers(floors["normal_floors"], 2, "normal floor range")
    numeric_fields(floors, ("modulus", "entry_remainder", "midboss_remainder", "stage_boss_remainder",
                           "reduced_pool_below_floor"), "floor rules")
    numeric_fields(floors["additional_stats"], ("points_per_decade",), "floor additional stats")
    require(len(areas["ella_encounters"]) == 4, "missing Ella encounter")
    for encounter in areas["ella_encounters"]:
        numeric_fields(encounter, ("floor",), "Ella encounter")
        check_refs(encounter["monster_ids"], monster_ids, "Ella encounter")
        require(nonempty(encounter["condition"]), "missing Ella encounter condition")

    stat_blocks(config["player"], "player")
    numeric_fields(config["player"], ("starting_gold", "max_hp", "starting_floor", "starting_area"), "player")
    numeric_fields(config["mana"], ("max", "current_at_construction", "potion_increment"), "mana")
    numeric_fields(config["slots"], ("total", "starting_unlocked", "imitation_uses_per_skill"), "slots")
    initial = config["initial_skills"]
    check_refs(initial["rank_pool"], skill_names, "initial rank pool")
    for pool in initial["attack_pools"]:
        check_refs(pool, skill_names, "initial attack pool")
    for group in initial["area_groups"]:
        check_refs([group["player"], group["ella"]], skill_names, "initial area group")
        check_refs(group["areas"], {row["id"] for row in regions}, "initial area group")
    fixed_integers(initial["player_uses"], config["slots"]["starting_unlocked"], "initial uses")
    validate_config(config, skill_names)
    numeric_fields(config["shop"]["prices"], config["shop"]["prices"], "shop prices")
    numeric_fields(config["rewards"], ("gold", "stat_increment", "gold_rush_per_floor"), "rewards")
    for name, value in config["status"]["registry_values"].items():
        require(type(value) in (int, float), f"{name}: invalid status constant")
    return skills, monsters, areas, config


def block(lines, start, end):
    return "\n".join(lines[start - 1:end])


def compare_source(script_path, datasets):
    skills, monsters, areas, config = datasets
    source = script_path.read_text(encoding="utf-8-sig")
    lines = source.splitlines()
    definitions = re.findall(
        r"^\s*(?:_\()?['\"]([^'\"]+)['\"]\)?\s*:\s*"
        r"\((\d+),\s*(\d+),\s*(\d+),\s*['\"](\w+)['\"],\s*(\d+),",
        block(lines, 106, 204), re.M,
    )
    extracted = [(name, int(power), int(category), int(accuracy), type_, int(cost))
                 for name, power, category, accuracy, type_, cost in definitions]
    actual = [tuple(row[k] for k in ("name", "power", "category", "accuracy", "type", "mana_cost"))
              for row in skills]
    require(extracted == actual, "skill registry differs from source")
    for row in skills:
        definition = lines[row["source_line"] - 1]
        require(re.search(rf"['\"]{re.escape(row['name'])}['\"]", definition),
                f"{row['name']}: wrong definition line")
        handler = lines[row["handler_line"] - 1]
        require(re.search(rf"skill_category\s*==\s*{row['category']}\s*:", handler),
                f"{row['name']}: wrong handler source line")
    characters = {int(match[-2]): match for match in SOURCE_CHAR.findall(source)
                  if match[0] not in {"player_substitute", "monster_substitute"}}
    require(set(characters) == set(range(56)), "unexpected source character registry")
    decks = dict(re.findall(r"_\('([^']+)'\)\s*:\s*(\([^\n]*\))", block(lines, 206, 257)))
    name_to_id = {row["name"]: row["id"] for row in monsters if row["id"] < 51}
    for row in monsters:
        key, name, *values, _, type_ = characters[row["id"]]
        require((key, name, tuple(map(int, values)), type_) ==
                (row["key"], row["name"], stat_blocks(row, key), row["type"]),
                f"{key}: differs from source")
        if name in decks:
            require(list(ast.literal_eval(decks[name])) == row["skills"], f"{key}: wrong skill order")
    bosses = re.findall(r"\$ monster_name = '([a-z]+)'", block(lines, 4165, 4182))
    require([name_to_id[name.capitalize()] for name in bosses] ==
            [row["boss_id"] for row in areas["regions"][:9]], "stage boss mapping differs from source")
    for row in areas["regions"][:9]:
        name = row["id"]
        pool = re.search(rf"^default {name}_table = (\[.*\])$", source, re.M).group(1)
        require([name_to_id[m] for m in ast.literal_eval(pool)] == row["normal_monster_ids"],
                f"{name}: pool differs from source")
        encounter = re.search(
            rf"selected_area == '{name}'\):\n\s+if\(level % 10 == 1\):"
            r"\n\s+\$ monster_name = '([^']+)'\n\s+elif\(level % 10 == 7\):"
            r"\n\s+\$ monster_name = '([^']+)'", block(lines, 2234, 2325),
        )
        if name == "Temple":
            encounter = re.search(
                r"else:\n\s+if\(level % 10 == 1\):\n\s+\$ monster_name = '([^']+)'"
                r"\n\s+elif\(level % 10 == 7\):\n\s+\$ monster_name = '([^']+)'",
                block(lines, 2311, 2325),
            )
        require(encounter is not None and
                [name_to_id[m] for m in encounter.groups()] == [row["entry_monster_id"], row["midboss_id"]],
                f"{name}: fixed encounters differ from source")
        require(f"{name}_table[0:{row['first_area_pool_size']}]" in source, f"{name}: wrong pool size")
    player = characters[0]
    require(tuple(map(int, player[2:-2])) == stat_blocks(config["player"], "player"),
            "player starting stats differ from source")
    status = dict((name, float(number)) for name, number in
                  re.findall(r"'(\w+)'\s*:\s*(-?\d+(?:\.\d+)?)", block(lines, 290, 302)))
    require(status == config["status"]["registry_values"], "status constants differ from source")
    for key, var in (("rank_pool", "rank_skill_table"), ("attack_pools", "bc_skill_table")):
        raw = re.search(rf"^\s*{var} = (\[.*\])", source, re.M).group(1)
        require(ast.literal_eval(raw) == config["initial_skills"][key], f"{key}: differs from source")


    start_uses = re.search(r"self\.memorized_skills = \{skill_a : (\d+), skill_b : (\d+), "
                           r"skill_c : (\d+), skill_d : (\d+)\}", block(lines, 1753, 1786))
    require(list(map(int, start_uses.groups())) == config["initial_skills"]["player_uses"],
            "starting uses differ from source")
    price_lines = {
        "blue_key": (1686, 1698), "red_key": (1699, 1714),
        "blue_mana_potion": (1715, 1723), "elemental_mana_potion": (1724, 1733),
        "undead_mana_potion": (1734, 1743), "type_change": (1744, 1752),
        "skill_cost_reduction": (1574, 1580),
    }
    for key, (start, end) in price_lines.items():
        price = re.search(r"player_gold -= (\d+)", block(lines, start, end))
        require(int(price.group(1)) == config["shop"]["prices"][key], f"{key}: price differs from source")
    for mapping, key, pattern, start, end in (
        (config["mana"], "max", r"self\.max_mana = (\d+)", 307, 360),
        (config["mana"], "current_at_construction", r"self\.current_mana = (\d+)", 307, 360),
        (config["mana"], "potion_increment", r"self\.max_mana \+= (\d+)", 1715, 1743),
        (config["combat"], "hp_multiplier", r"self\.hp = base_hp \* (\d+)", 307, 360),
        (config["status"], "icebolt_stun_percent", r"<= (\d+) and real_target\.hp > 0", 1415, 1428),
        (config["status"], "poison_tail_poison_percent", r"<= (\d+) and real_target\.hp > 0", 1430, 1443),
        (config["shop"], "reroll_multiplier", r"reroll_cost = reroll_cost\*(\d+)", 1661, 1684),
        (config["shop"]["prices"], "reroll_initial", r"reroll_cost = (\d+)", 99, 105),
        (config["rewards"], "gold_rush_per_floor", r"stage_gold = level \* (\d+)", 2804, 2818),
        (areas["floor_rules"]["additional_stats"], "points_per_decade",
         r"total_additional_stats = \(level//10\)\*(\d+)", 4216, 4242),
    ):
        value = re.search(pattern, block(lines, start, end))
        require(value is not None and int(next(v for v in value.groups() if v is not None)) == mapping[key],
                f"{key}: differs from source")
    screen_path = script_path.with_name("custom_screen.rpy")
    if screen_path.is_file():
        screen = screen_path.read_text(encoding="utf-8-sig")
        reward_gold = re.search(r"player_gold \+ (\d+)", screen).group(1)
        require(int(reward_gold) == config["rewards"]["gold"], "gold reward differs from screen source")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--source-script", type=Path, help="Optional local game/script.rpy from the pinned revision")
    args = parser.parse_args()
    try:
        datasets = validate(args.data_dir)
        if args.source_script:
            compare_source(args.source_script, datasets)
    except (OSError, ValueError, SyntaxError, KeyError, TypeError, AttributeError, IndexError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        return 1
    print("Validated 4 JSON files, 88 skills, 55 monsters, and 11 regions.")
    if args.source_script:
        print("Compared source registries, base stats, encounters, and selected configuration constants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

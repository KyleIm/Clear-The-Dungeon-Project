"""The inherited 671-node shop state schema and public build summaries."""

from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from combat_rl.environment import CHARACTER_NUMBERS, character_features, scale, skill_features


BASE_FEATURE_COUNT = 671
BASE_STATS = ('attack', 'defense', 'hp', 'speed')


@lru_cache(maxsize=8)
def _vocabulary(data):
    names = tuple(sorted(data.skills, key=lambda name: data.skills[name]['id']))
    categories = tuple(sorted({row['category'] for row in data.skills.values()}))
    regions = tuple(data.regions)
    if (len(names), len(categories), len(regions), data.config['slots']['total']) != (88, 65, 11, 6):
        raise ValueError('The shop encoder requires 88 skills, 65 categories, 11 regions and six slots')
    view = SimpleNamespace(data=data)
    slot_features = {name: skill_features(view, name, categories) for name in names}
    return names, regions, slot_features


def _region(game, previous_area):
    # The ordinary shop overwrites area with Shop. No next area has been drawn yet.
    return game.area if game.area in game.data.regions else previous_area


def _route(event_choice):
    if event_choice in (None, 'A', 'B'):
        return event_choice
    return {'pickpocket': 'A', 'help': 'B'}[event_choice]


def summarize(game, *, previous_area=None, event_choice=None):
    """Return detached, JSON-safe resource and character data for an entry or exit."""
    player = game.player
    locked = game.data.config['slots']['locked_marker']
    return {
        'floor': game.floor,
        'region': _region(game, previous_area),
        'event_route': _route(event_choice),
        'type': player.type,
        'conditions': sorted({effect['Name'] for effect in player.status_conditions}),
        'hp': player.hp,
        'max_hp': player.max_hp,
        'base_stats': {key: getattr(player, f'base_{key}') for key in BASE_STATS},
        'current_stats': {key: getattr(player, key) for key in ('attack', 'defense', 'speed', 'ACC', 'EVA', 'DR')},
        'gold': game.gold,
        'max_mana': player.max_mana,
        'used_mana': player.max_mana - player.current_mana,
        'remaining_mana': player.current_mana,
        'unused_stat_points': player.available_points,
        'purchases': sorted(player.purchases),
        'learned_skills': [name for name in dict.fromkeys(player.learned_skills) if name in game.data.skills],
        'slots': [dict(slot=index, skill=None if name == locked else name, locked=name == locked,
                       uses=player.memorized_skills.get(name, 0), mana_cost=player.skill_costs.get(name, 0))
                  for index, name in enumerate(player.current_skills, 1)],
        'unlocked_slots': sum(name != locked for name in player.current_skills),
    }


def encode(game, *, previous_area=None, event_choice=None):
    names, regions, slot_features = _vocabulary(game.data)
    player = game.player
    locked = game.data.config['slots']['locked_marker']
    actor = {key: getattr(player, key) for key in (*CHARACTER_NUMBERS, 'type', 'status_conditions')}
    features = character_features(actor)
    if len(player.current_skills) != 6:
        raise ValueError('Expected six equipment slots at the shop')
    for name in player.current_skills:
        if name == locked:
            features.extend([0.0] * 75)
            continue
        availability, *definition = slot_features[name]
        features.extend((availability, scale(player.memorized_skills.get(name, 0), 10),
                         *definition, scale(player.skill_costs[name], 10)))
    learned = set(player.learned_skills)
    features.extend(name in learned for name in names)
    features.extend(scale(player.skill_costs[name], 10) if name in learned else 0.0 for name in names)
    remaining = player.current_mana
    features.extend((scale(player.max_mana, 100), scale(player.max_mana - remaining, 100),
                     scale(remaining, 100), scale(player.available_points, 100),
                     scale(game.gold, 1000), game.floor / 100))
    region = _region(game, previous_area)
    features.extend(region == name for name in regions)
    route = _route(event_choice)
    features.extend((route == 'A', route == 'B'))
    return np.asarray(features, dtype=np.float32)

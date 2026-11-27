"""Encode shop states and construct bounded legal exit plans without running fights."""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from combat_rl.environment import CHARACTER_NUMBERS, character_features, scale, skill_features
from simulator import shop


FEATURE_COUNT = 671
MENU_NAMES = ('skills', 'costs', 'stats', 'merchandise')
BASE_STATS = ('attack', 'defense', 'hp', 'speed')


@dataclass(frozen=True)
class Candidate:
    actions: tuple[tuple, ...]
    features: np.ndarray
    exit_state: dict


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


def _legal_edits(scratch):
    return [action for action in shop.legal_actions(scratch) if action[0] not in {'shop_back', 'reroll'}]


def _plan(game, rng, budget, planner):
    # Only the shop-readable fields are present. The scratch object has no RNG or combat state.
    scratch = SimpleNamespace(data=game.data, player=deepcopy(game.player), floor=game.floor,
                              area=game.area, gold=game.gold, shop_menu='main', reroll_cost=game.reroll_cost)
    actions = []

    def apply(action):
        shop.apply(scratch, action)
        actions.append(action)

    while budget > 0:
        menus = {}
        for menu in MENU_NAMES:
            scratch.shop_menu = menu
            edits = _legal_edits(scratch)
            if edits:
                menus[menu] = edits
        scratch.shop_menu = 'main'
        if not menus:
            break
        if planner == 'uniform_action':
            menu = rng.choices(tuple(menus), weights=[len(actions) for actions in menus.values()], k=1)[0]
        else:
            menu = rng.choice(tuple(menus))
        edits = menus[menu]
        apply(('shop_menu', menu))
        chunk = rng.randint(1, budget)
        for edit_index in range(chunk):
            if edit_index:
                edits = _legal_edits(scratch)
            if not edits:
                break
            apply(rng.choice(edits))
            budget -= 1
        apply(('shop_back',))
    return scratch, (*actions, ('leave_shop',))


def generate(game, count, rng, *, previous_area=None, event_choice=None, max_edits=24, planner='uniform_menu'):
    """Return immediate leave plus up to count-1 distinct randomly edited exit plans."""
    if game.phase != 'shop' or game.shop_menu != 'main':
        raise ValueError('Candidate generation starts at the shop main menu')
    if type(count) is not int or count < 1 or type(max_edits) is not int or max_edits < 0:
        raise ValueError('Expected a positive candidate count and nonnegative edit budget')
    if planner not in ('uniform_menu', 'uniform_action'):
        raise ValueError('Unknown shop candidate planner')
    kwargs = dict(previous_area=previous_area, event_choice=event_choice)
    features = encode(game, **kwargs)
    candidates = [Candidate((('leave_shop',),), features, summarize(game, **kwargs))]
    unique = {features.tobytes()}
    if max_edits:
        for _ in range(count - 1):
            scratch, actions = _plan(game, rng, rng.randint(1, max_edits), planner)
            features = encode(scratch, **kwargs)
            signature = features.tobytes()
            if signature not in unique:
                unique.add(signature)
                candidates.append(Candidate(actions, features, summarize(scratch, **kwargs)))
    return candidates

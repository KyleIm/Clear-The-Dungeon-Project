"""Generate complete shop configurations through original deterministic shop actions."""

from copy import copy, deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from alternating_rl.features import BASE_STATS, encode
from alternating_rl.shop import ITEMS
from combat_rl.environment import TYPE_ORDER, scale
from simulator import shop as engine

HEADS = ('gold', 'stats', 'skills')
FEATURE_COUNT = 1370


@dataclass
class Candidate:
    actions: tuple
    features: np.ndarray


def clone_shop(game):
    shadow = copy(game)
    shadow.player = deepcopy(game.player)
    return shadow


def edit(game, action):
    if action == ('leave_shop',) or action == ('reroll',) or action not in engine.legal_actions(game):
        raise ValueError(f'Illegal deterministic shop edit: {action}')
    engine.apply(game, action)


def replay(game, actions):
    shadow = clone_shop(game)
    for action in actions:
        edit(shadow, action)
    if shadow.shop_menu != 'main':
        raise ValueError('A complete configuration must return to the main menu')
    return shadow


def features(game, area, route):
    p = game.player
    extra = [scale(p.base_hp, 100), p.hp / max(p.max_hp, 1),
             *(scale(getattr(p, 'base_' + k), 100) for k in ('ACC', 'EVA', 'DR')),
             *(p.base_type == kind for kind in TYPE_ORDER), *(item in p.purchases for item in ITEMS)]
    return np.concatenate((encode(game, previous_area=area, event_choice=route), np.asarray(extra, dtype=np.float32)))


def build(shadow, head, rng, ordinal):
    actions = []

    def append(action):
        edit(shadow, action)
        actions.append(action)

    p = shadow.player
    if head == 0:
        used = set()
        for _ in range(rng.randint(1, 3)):
            choices = []
            for menu in ('merchandise', 'costs'):
                view = SimpleNamespace(data=shadow.data, player=p, gold=shadow.gold,
                                       reroll_cost=shadow.reroll_cost, shop_menu=menu)
                choices.extend((menu, a) for a in engine.legal_actions(view)
                               if a not in used and a[0] in ('buy', 'reduce_skill_cost', 'change_type'))
            if ('change_type',) in used:
                choices = [(menu, a) for menu, a in choices if a[0] != 'change_type']
            if not choices:
                break
            menu, action = rng.choice(choices)
            if shadow.shop_menu != menu:
                if shadow.shop_menu != 'main':
                    append(('shop_back',))
                append(('shop_menu', menu))
            append(action)
            used.add(action)
            if action[0] == 'change_type':
                used.add(('change_type',))
    elif head == 1:
        append(('shop_menu', 'stats'))
        if ordinal % 2 == 0:
            append(('reset_redistribution',))
        total = p.available_points
        weights = [rng.expovariate(1) for _ in BASE_STATS]
        density = rng.choice((0.0, rng.random(), 1.0))
        target = int(total * density)
        # A random priority distributes rounding residue without privileging a stat.
        amounts = [int(target * w / sum(weights)) for w in weights]
        for i in rng.sample(range(4), target - sum(amounts)):
            amounts[i] += 1
        for stat, points in zip(BASE_STATS, amounts):
            for delta in (10, 1):
                for _ in range(points // delta):
                    append(('adjust_stat', stat, delta))
                points %= delta
    else:
        append(('shop_menu', 'skills'))
        equipped = [name for name in p.current_skills if name != 'lock']
        locked_out = set(equipped)
        alternatives = [name for name in dict.fromkeys(p.learned_skills) if name not in locked_out and name != 'lock']
        changed = rng.sample(equipped, rng.randint(0, min(len(equipped), len(alternatives))))
        for old, new in zip(changed, rng.sample(alternatives, len(changed))):
            append(('replace_skill', old, new))
        names = [name for name in p.current_skills if name != 'lock']
        reduced = rng.sample(names, rng.randrange(len(names) + 1))
        for name in reduced:
            for _ in range(rng.randint(0, p.memorized_skills[name])):
                append(('forget', name))
        remaining = [name for name in names if name not in reduced]
        rng.shuffle(remaining)
        for name in remaining:
            budget = p.current_mana // p.skill_costs[name]
            for _ in range(rng.randint(0, budget)):
                append(('memorize', name))
    if shadow.shop_menu != 'main':
        append(('shop_back',))
    return tuple(actions)


def configuration_features(entry, final, area, route):
    return np.concatenate((features(entry, area, route), features(final, area, route)))


def configurations(game, head, rng, count, area, route):
    entry = features(game, area, route)
    result = [Candidate((), np.concatenate((entry, entry)))]
    seen = {entry.tobytes()}
    for ordinal in range(3 * (count - 1)):
        if len(result) == count:
            break
        final = clone_shop(game)
        actions = build(final, head, rng, ordinal)
        encoded = features(final, area, route)
        key = encoded.tobytes()
        if key not in seen:
            result.append(Candidate(actions, np.concatenate((entry, encoded))))
            seen.add(key)
    return result


def allocation_probes(game, area, route):
    """Keep every prior choice fixed and make one legal allocation, where possible."""
    for head, menu, verb, field in ((1, 'stats', 'adjust_stat', 'stat_points'),
                                     (2, 'skills', 'memorize', 'mana')):
        edit_game = clone_shop(game)
        enter = ('shop_menu', menu)
        edit(edit_game, enter)
        candidates = [a for a in engine.legal_actions(edit_game) if a[0] == verb and
                      (head == 2 or a[2] == 1)]
        for action in candidates:
            final = replay(game, (enter, action, ('shop_back',)))
            before = game.player.available_points if head == 1 else game.player.current_mana
            after = final.player.available_points if head == 1 else final.player.current_mana
            yield head, field, action, before, after, configuration_features(game, final, area, route)

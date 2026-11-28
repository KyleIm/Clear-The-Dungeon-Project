"""Retrospectively verify forward-compatible uniform shop trajectories."""

from copy import copy, deepcopy
from random import Random

from .shop import Shop


MENU_STAGE = {'costs': 0, 'merchandise': 0, 'stats': 1, 'skills': 2}


def shadow_shop(game):
    """Clone the mutable resources shop actions can touch, leaving combat untouched."""
    shadow = copy(game)
    shadow.player = deepcopy(game.player)
    shadow.reroll_rng = Random()
    shadow.reroll_rng.setstate(game.reroll_rng.getstate())
    return shadow


def compatible_trace(shadow, actual, trace, region, event, budgets):
    """Return verified DQN transitions up to the unexecuted leave action, or None."""
    edits = [[], [], []]
    current = highest = -1
    entered_stats = False
    for action in trace:
        if action[0] == 'shop_menu':
            current = MENU_STAGE[action[1]]
            if current < highest:
                return None
            if current == 1:
                # Re-entering stats can apply a mid-stage refresh unavailable to the DQN.
                if entered_stats:
                    return None
                entered_stats = True
            highest = current
        elif action[0] == 'shop_back':
            current = -1
        elif current < 0:
            return None
        else:
            edits[current].append(action)
    if any(len(group) > limit for group, limit in zip(edits, budgets)):
        return None

    adapter = Shop(shadow, region, event, budgets)
    decisions = []
    leave_reached = False

    def advance(action):
        nonlocal leave_reached
        if action == ('leave_shop',):
            # The original run really chose leave. Compare before that RNG-consuming step.
            leave_reached = True
            return False
        if action not in shadow.get_legal_actions():
            raise ValueError('Blocked shadow shop action')
        shadow.step(action)
        return True

    positions = [0, 0, 0]
    try:
        while not leave_reached:
            observation = adapter.observation()
            _, head, mask = observation
            actions = edits[head]
            position = positions[head]
            if position < len(actions):
                legal = adapter.legal_engine_actions()
                matches = [i for i, action in legal.items() if action == actions[position] and mask[i]]
                if len(matches) != 1:
                    return None
                choice = matches[0]
                positions[head] += 1
            else:
                choice = len(mask) - 1
            adapter.step(choice, advance)
            decisions.append(dict(state=observation, action=choice,
                                  next_state=None if leave_reached else adapter.observation()))
    except (ValueError, RuntimeError):
        return None
    if positions != list(map(len, edits)):
        return None
    before, after = actual.get_state(), shadow.get_state()
    if any(before[key] != after[key] for key in ('player', 'gold', 'reroll_cost')):
        return None
    if actual.reroll_rng.getstate() != shadow.reroll_rng.getstate():
        return None
    return decisions

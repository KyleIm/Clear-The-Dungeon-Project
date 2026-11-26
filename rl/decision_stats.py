"""Aggregate policy decisions and workshop outcomes without storing turn traces."""

from collections import Counter, defaultdict
from copy import deepcopy


PHASES = ('combat', 'reward', 'shop', 'event')
REWARDS = ('Gold', 'Potion', 'Initialize', 'Stat', 'Exchange')
SHOP_FUNCTIONS = ('Skill Configuration', 'Skill Cost Reduction', 'Stat Redistribution',
                  'Merchandise Purchase', 'Leave Shop')
LEAVE = SHOP_FUNCTIONS[-1]
_SHOP_ALIASES = dict(zip(('skills', 'costs', 'stats', 'merchandise', 'leave'), SHOP_FUNCTIONS))
_MAX_CHANGE_EXAMPLES = 3


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _label(descriptor):
    if isinstance(descriptor, (list, tuple)):
        return _SHOP_ALIASES.get(str(descriptor[-1]), str(descriptor[-1]))
    return str(descriptor)


class DecisionStats:
    """One evaluation panel's summaries and a small amount of state per active slot."""

    def __init__(self):
        self.combat = defaultdict(Counter)
        self.reward_floors = defaultdict(Counter)
        self.reward_available = defaultdict(Counter)
        self.shop_functions = defaultdict(Counter)
        self.shop_visits = Counter()
        self.shop_penalty = 0.0
        self.shop_changes = defaultdict(_ShopChanges)
        self.leave_probabilities = Counter()
        self.episode_counts = Counter()
        self.slot_shop_seen = set()
        self.slot_events = defaultdict(set)
        self.slot_repeat = defaultdict(dict)
        self.event_choices = Counter()
        self.event_available = Counter()
        self.event_floors = defaultdict(Counter)
        self.event_outcomes = defaultdict(Counter)

    def record(self, phase, floor, legal_labels, choice, probabilities, *, slot=0):
        """Collect availability and actions from the pre-step policy decision."""
        if not isinstance(phase, str):
            phase = PHASES[int(phase)]
        selected = _label(legal_labels[choice])
        if phase == 'combat':
            available = defaultdict(float)
            for action_id, descriptor in legal_labels.items():
                available[_label(descriptor)] += (float(probabilities[action_id])
                                                  if probabilities is not None else 0.0)
            for skill, probability in available.items():
                row = self.combat[skill]
                row['available'] += 1
                if probabilities is not None:
                    row['probability_sum'] += probability
                    row['probability_count'] += 1
            self.combat[selected]['selected'] += 1
        elif phase == 'reward':
            floor = int(floor)
            self.reward_floors[floor][selected] += 1
            self.reward_available[floor].update({_label(descriptor) for descriptor in legal_labels.values()})
        elif phase == 'event':
            self.event_choices[selected] += 1
            self.event_available.update({_label(descriptor) for descriptor in legal_labels.values()})
            self.slot_events[slot].add(selected)
        elif phase == 'shop':
            self.slot_shop_seen.add(slot)
            if selected in SHOP_FUNCTIONS:
                self.shop_functions[selected]['selected'] += 1
                for action_id, descriptor in legal_labels.items():
                    function = _label(descriptor)
                    if function in SHOP_FUNCTIONS:
                        self.shop_functions[function]['available'] += 1
                    if function == LEAVE and probabilities is not None:
                        self.leave_probabilities['sum'] += float(probabilities[action_id])
                        self.leave_probabilities['count'] += 1

    def transition(self, info, *, slot=0):
        """Collect committed shop results and engine-step accounting."""
        selection = info.get('shop_selection')
        if selection:
            function = selection['function']
            if function != LEAVE:
                repeat = bool(selection['repeat'])
                self.shop_functions[function]['repeat' if repeat else 'first'] += 1
                self.slot_repeat[slot][function] = repeat
        change = info.get('shop_change')
        if change:
            function = change['function']
            repeat = self.slot_repeat.get(slot, {}).get(function, False)
            self.shop_changes[function].add(change, repeat)
        self.shop_penalty += float(info.get('shop_penalty', 0.0))
        visit = info.get('shop_visit')
        if visit:
            self.slot_shop_seen.add(slot)
            self.shop_visits['recorded'] += 1
            self.shop_visits['exits'] += bool(visit['left'])
            self.shop_visits['repeated'] += bool(visit['repeated'])
            self.shop_visits['timeouts'] += visit['outcome'] == 'timeout'
            self.shop_visits['strategic_actions'] += int(visit['strategic_actions'])
            self.shop_visits['engine_actions'] += int(visit['engine_actions'])
            self.slot_repeat.pop(slot, None)

    def episode(self, row, *, slot=0):
        """Attach downstream outcomes to each event option selected in this episode."""
        floor = int(row['floor'])
        self.episode_counts['total'] += 1
        self.episode_counts['reached_floor_10'] += floor >= 10
        self.episode_counts['shop_seen'] += slot in self.slot_shop_seen
        self.episode_counts['shop_timeouts'] += bool(row.get('shop_timeout'))
        for option in self.slot_events.get(slot, ()):
            self.event_floors[option][floor] += 1
            self.event_outcomes[option][row['outcome']] += 1

    def reset_slot(self, slot):
        self.slot_events.pop(slot, None)
        self.slot_repeat.pop(slot, None)
        self.slot_shop_seen.discard(slot)

    def summary(self):
        combat = {}
        for skill, row in sorted(self.combat.items()):
            available, selected = row['available'], row['selected']
            combat[skill] = {'available': available, 'selected': selected,
                             'selection_rate': _rate(selected, available),
                             'mean_probability': _rate(row['probability_sum'], row['probability_count'])}
        rewards = Counter()
        reward_available = Counter()
        for row in self.reward_floors.values():
            rewards.update(row)
        for row in self.reward_available.values():
            reward_available.update(row)
        reward_total = sum(rewards.values())
        strategic_total = sum(row['selected'] for row in self.shop_functions.values())
        functions = {}
        for function in SHOP_FUNCTIONS:
            row = self.shop_functions.get(function, Counter())
            functions[function] = {'available': row['available'], 'selected': row['selected'],
                                   'rate': _rate(row['selected'], strategic_total),
                                   'selection_rate': _rate(row['selected'], row['available']),
                                   'first': row['first'] if function != LEAVE else None,
                                   'repeat': row['repeat'] if function != LEAVE else None}
        visits = self.shop_visits
        total_episodes = self.episode_counts['total']
        event_total = sum(self.event_choices.values())
        events = {}
        for option, available in sorted(self.event_available.items()):
            selected = self.event_choices[option]
            floors = self.event_floors[option]
            total = sum(floors.values())
            reached = total
            survival = {}
            for floor in range(1, 101):
                survival[str(floor)] = _rate(reached, total)
                reached -= floors[floor]
            events[option] = {'available': available, 'selected': selected, 'rate': _rate(selected, event_total),
                              'selection_rate': _rate(selected, available), 'episodes': total,
                              'mean_floor': _rate(sum(floor * n for floor, n in floors.items()), total),
                              'max_floor': max(floors, default=None),
                              'clear_rate': _rate(self.event_outcomes[option]['clear'], total),
                              'outcomes': dict(self.event_outcomes[option]), 'survival': survival}
        return {
            'combat': combat,
            'reward': {'total': reward_total, 'selected': {option: rewards[option] for option in REWARDS},
                       'rates': {option: _rate(rewards[option], reward_total) for option in REWARDS},
                       'available': {option: reward_available[option] for option in REWARDS},
                       'selection_rates': {option: _rate(rewards[option], reward_available[option])
                                           for option in REWARDS},
                       'available_by_floor': {str(floor): {option: row[option] for option in REWARDS}
                                              for floor, row in sorted(self.reward_available.items())},
                       'by_floor': {str(floor): {option: row[option] for option in REWARDS}
                                    for floor, row in sorted(self.reward_floors.items())}},
            'shop': {'functions': functions, 'visits': visits['recorded'], 'exits': visits['exits'],
                     'repeated_visits': visits['repeated'], 'timeouts': visits['timeouts'],
                     'exit_rate': _rate(visits['exits'], visits['recorded']),
                     'repeat_rate': _rate(visits['repeated'], visits['recorded']),
                     'timeout_rate': _rate(visits['timeouts'], visits['recorded']),
                     'episode_shop_timeout_rate': _rate(max(visits['timeouts'], self.episode_counts['shop_timeouts']),
                                                        total_episodes),
                     'mean_strategic_actions': _rate(visits['strategic_actions'], visits['recorded']),
                     'mean_engine_actions': _rate(visits['engine_actions'], visits['recorded']),
                     'mean_leave_probability': _rate(self.leave_probabilities['sum'], self.leave_probabilities['count']),
                     'shop_reach_rate': _rate(self.episode_counts['shop_seen'], total_episodes),
                     'floor_10_reach_rate': _rate(self.episode_counts['reached_floor_10'], total_episodes),
                     'penalty': self.shop_penalty,
                     'changes': {function: stats.summary() for function, stats in self.shop_changes.items()}},
            'event': {'total': event_total, 'options': events},
        }


class _ShopChanges:
    """Bounded aggregate of full before/after builds and a few complete examples."""

    def __init__(self):
        self.counts = Counter()
        self.gold_spent = 0.0
        self.items = Counter()
        self.examples = []
        self.built = {'before': _BuildStats(), 'after': _BuildStats()}

    def add(self, change, repeat):
        before, after = change['before'], change['after']
        self.counts['commits'] += 1
        self.counts['changed'] += before != after
        self.counts['repeated_commits'] += repeat
        self.counts['repeated_changed'] += repeat and before != after
        self.gold_spent += float(change.get('gold_spent', 0))
        if change.get('item') is not None:
            self.items[str(change['item'])] += 1
        for direction, snapshot in (('before', before), ('after', after)):
            self.built[direction].add(snapshot)
        if len(self.examples) < _MAX_CHANGE_EXAMPLES:
            self.examples.append(deepcopy(change))

    def summary(self):
        return {**{key: self.counts[key] for key in ('commits', 'changed', 'repeated_commits', 'repeated_changed')},
                'gold_spent': self.gold_spent, 'items': dict(sorted(self.items.items())),
                'examples': self.examples, **{direction: data.summary() for direction, data in self.built.items()}}


class _BuildStats:
    def __init__(self):
        self.counts = Counter()
        self.sums = Counter()
        self.skills = defaultdict(Counter)

    def add(self, snapshot):
        scalars = {key: snapshot[key] for key in ('mana_used', 'mana_remaining', 'max_mana',
                                                 'unlocked_slots', 'available_points', 'gold') if key in snapshot}
        scalars.update(snapshot.get('stats', {}))
        self.counts.update(scalars.keys())
        self.sums.update(scalars)
        for slot in snapshot.get('slots', ()):
            if slot.get('skill') and not slot.get('locked'):
                row = self.skills[slot['skill']]
                row['equipped'] += 1
                row['uses'] += slot.get('uses', 0)
                row['mana_cost'] += slot.get('mana_cost', 0)

    def summary(self):
        return {'mean': {key: self.sums[key] / n for key, n in sorted(self.counts.items())},
                'skills': {skill: {'equipped': row['equipped'], 'mean_uses': row['uses'] / row['equipped'],
                                   'mean_mana_cost': row['mana_cost'] / row['equipped']}
                           for skill, row in sorted(self.skills.items())}}

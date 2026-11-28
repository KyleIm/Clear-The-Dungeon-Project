"""Live and paired-counterfactual episodes using the original simulator."""

from collections import Counter, defaultdict
from copy import deepcopy
from random import Random
from time import perf_counter

import numpy as np

from alternating_rl.environment import Run as ShopRun, attained_target
from alternating_rl.features import summarize
from combat_rl.environment import REWARD_CHOICES, Run as CombatRun
from .configurations import HEADS, configurations, replay


def hp_bucket(hp, maximum):
    ratio = hp / max(maximum, 1)
    return next((label for edge, label in ((.20, '0–20%'), (.40, '20–40%'), (.60, '40–60%'),
                                         (.80, '60–80%'), (.90, '80–90%')) if ratio < edge), '90–100%')


def next_shop_distance(state):
    floor = state['floor']
    if floor <= 90:
        return 10 - (floor - 1) % 10 - 1
    if state['ella_path'] != 1 and floor <= 99 and not state['ella_encounter']:
        return 99 - floor
    return 'no_shop'


class Run(ShopRun):
    def __init__(self, game_seed, policy_seed, action_cap, *, mode, version=0):
        super().__init__(game_seed, policy_seed, action_cap, 0, version, random_shop=True, collect_replay=False)
        self.mode = mode
        self.explore_reward = .20 if mode == 'paired' else 0
        self.reward_rng = Random(f'reward-exploration:{policy_seed}')
        self.candidate_rng = Random(f'configuration:{policy_seed}')
        self.reward_context = defaultdict(Counter)
        self.shop_work = Counter()
        self.candidates = []
        self.applied = {}
        self.probes = []

    def select_action(self, selected):
        if self.phase == 'reward' and self.explore_reward and self.reward_rng.random() < self.explore_reward:
            return self.reward_rng.choice(self._decision_actions())[0], 'random'
        return selected, 'ppo'

    def learned(self, selected, probabilities, version):
        self.version = version
        action, source = self.select_action(selected)
        if self.phase == 'reward':
            p = self.state['player']
            name = dict(self._decision_actions())[action][1]
            drift = [np.sign(p[k] - p['base_' + k]) for k in ('attack', 'defense', 'speed', 'ACC', 'EVA', 'DR')]
            changed = ('mixed' if min(drift) < 0 < max(drift) else 'below' if min(drift) < 0 else
                       'above' if max(drift) > 0 else 'same')
            for axis, value in (('overall', 'all'), ('hp', hp_bucket(p['hp'], p['max_hp'])),
                                ('stat_drift', changed), ('shop_distance', next_shop_distance(self.state))):
                counts = self.reward_context[axis, str(value), source]
                counts['choices'] += 1
                counts['chose:' + name] += 1
                for label, prob in zip(REWARD_CHOICES, probabilities):
                    counts['ppo_prob:' + label] += float(prob)
        return self.step(action)

    def context(self):
        return (self._entry or {}).get('region'), self.event_route

    def offer(self, head, count):
        start = perf_counter()
        self.candidates = configurations(self.game, head, self.candidate_rng, count, *self.context())
        self.shop_work[f'generate_{HEADS[head]}_seconds'] += perf_counter() - start
        self.shop_work[f'candidates_{HEADS[head]}'] += len(self.candidates)
        return [c.features for c in self.candidates]

    def apply_candidate(self, head, index, score):
        before = self.actions
        final = self.candidates[index]
        self.candidates = []
        for action in final.actions:
            if self.done:
                break
            self._advance(action)
        self.applied[HEADS[head]] = dict(changes=self.actions - before, score=score)
        return self.actions - before

    def leave(self):
        before, floor = self.actions, self.floor
        if not self.done:
            self._advance(('leave_shop',))
            self.shop_exits[-1]['configuration_choices'] = self.applied
            self.shop_exits[-1]['allocation_probes'] = self.probes
        self.applied, self.probes = {}, []
        return self._reward_since(floor), self.actions - before

    def result(self):
        return dict(super().result(), shop_mode=self.mode,
                    reward_context=[dict(axis=k[0], bucket=k[1], source=k[2], **v)
                                    for k, v in self.reward_context.items()],
                    shop_work=dict(self.shop_work))


class Branch(CombatRun):
    """A lightweight PPO observation view on a cloned trajectory."""

    def __init__(self, run, actions, key, budget):
        original = run.game
        # Clone the full mutable game including both RNG streams, sharing only static GameData.
        self.game = replay(deepcopy(original, {id(original.data): original.data}), actions)
        self.state = self.game.step(('leave_shop',))
        self.categories, self.regions = run.categories, run.regions
        self.action_cap = budget
        self.actions = 1
        self.floor = self.game.floor
        self.shop_floor = original.floor
        self.tapes = [Random(f'{key}:{label}') for label in ('combat', 'reward', 'event', 'mix', 'random_reward')]

    def complete(self):
        return self.done or attained_target(self.shop_floor, self.state)

    def advance(self, action):
        self.state = self.game.step(action)
        self.actions += 1
        self.floor = max(self.floor, self.game.floor)

    def result(self):
        right_censored = not (self.game.is_terminal() or attained_target(self.shop_floor, self.state))
        return self.floor, self.actions, right_censored


def frozen_policy(weights):
    """CPU-only inference for four paired branches, with the inherited PPO weights."""
    def layer(name, x):
        return np.einsum('bi,oi->bo', x, weights[name + '.weight']) + weights[name + '.bias']

    def predict(observations):
        x, phases, masks = map(np.stack, zip(*observations))
        for name in ('encoder.0', 'encoder.2'):
            x = np.tanh(layer(name, x))
        logits = layer('combat_head', x)
        logits[phases == 1, :5] = layer('reward_head', x[phases == 1])
        logits = np.where(masks, logits, -np.inf)
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        return probabilities / probabilities.sum(axis=1, keepdims=True)
    return predict


def compare_visit(run, plans, policy, budget):
    key = f'paired:{run.policy_seed}:{run.actions}'
    branches = [Branch(run, plan, key, budget) for plan in plans]
    while active := [branch for branch in branches if not branch.complete()]:
        for branch in active:
            if branch.state['phase'] == 'event':
                branch.advance(branch.tapes[2].choice(branch.game.get_legal_actions()))
        decisions = [branch for branch in active if not branch.complete() and branch.state['phase'] != 'event']
        if not decisions:
            continue
        observations = [b.observation() for b in decisions]
        for branch, (_, phase, mask), probability in zip(decisions, observations, policy(observations)):
            legal = np.flatnonzero(mask)
            draw = branch.tapes[phase].random()
            if phase == 1 and branch.tapes[3].random() < run.explore_reward:
                action = branch.tapes[4].choice(legal)
            else:
                weights = np.asarray(probability)[legal]
                cdf = np.cumsum(weights)
                action = legal[min(int(np.searchsorted(cdf, draw * cdf[-1], side='right')), len(legal) - 1)]
            branch.advance(dict(branch._decision_actions())[action])
    return [branch.result() for branch in branches]


def compare(run, weights, budget):
    start = perf_counter()
    options = [configurations(run.game, i, run.candidate_rng, 2, *run.context()) for i in range(3)]
    selected = [c[-1] for c in options]
    outcomes = compare_visit(run, [(), *(c.actions for c in selected)], frozen_policy(weights), budget)
    before, _, censored = outcomes[0]
    rows = []
    for head, (candidate, (floor, actions, timeout)) in enumerate(zip(selected, outcomes[1:])):
        delta = floor - before
        final = replay(run.game, candidate.actions)
        rows.append((head, candidate.features, dict(shop_floor=run.floor, baseline_floor=before, floor=floor,
                     delta=delta, label=None if censored or timeout else int(np.sign(delta)),
                     noop=not candidate.actions, branch_actions=actions, entry_gold=run.game.gold,
                     configuration=summarize(final, previous_area=run.context()[0], event_choice=run.event_route))))
    run.shop_work['paired_seconds'] += perf_counter() - start
    run.shop_work['paired_actions'] += sum(r[1] for r in outcomes)
    run.shop_work['paired_branches'] += 4
    return rows

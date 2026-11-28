"""Alternating experiment adapter over the unmodified dungeon simulator."""

from collections import Counter
from operator import index
from unittest.mock import patch

from combat_rl.environment import Run as CombatRun
from simulator import combat

from .bootstrap import compatible_trace, shadow_shop
from .shop import BUDGETS, Shop


def attained_target(shop_floor, state):
    """Return the first state that confirms the shop's survival target."""
    if shop_floor == 99:
        return state['phase'] == 'cleared'
    if shop_floor == 90:
        return state['floor'] >= 100
    return state['phase'] == 'shop' and state['floor'] == shop_floor + 10


def shop_state(state, area, action_count):
    p = state['player']
    stat_names = ('attack', 'defense', 'hp', 'speed')
    slots = [dict(slot=slot, skill=name, uses=p['memorized_skills'][name], mana_cost=p['skill_costs'][name])
             for slot, name in enumerate(p['current_skills'], 1) if name != 'lock']
    return dict(floor=state['floor'], region=area, action=action_count, gold=state['gold'],
                attack=p['base_attack'], defense=p['base_defense'], hp=p['base_hp'], speed=p['base_speed'],
                base_stats={name: p[f'base_{name}'] for name in (*stat_names, 'ACC', 'EVA', 'DR')},
                current_stats={name: p[name] for name in (*stat_names, 'ACC', 'EVA', 'DR')},
                max_hp=p['max_hp'], player_type=p['type'],
                statuses=[effect['Name'] for effect in p['status_conditions']],
                max_mana=p['max_mana'], used_mana=p['max_mana'] - p['current_mana'],
                remaining_mana=p['current_mana'], unused_stat_points=p['available_points'],
                slots=slots, unlocked_slots=len(slots),
                learned_skills=[name for name in p['learned_skills'] if name != 'lock'],
                skill_costs={name: p['skill_costs'][name] for name in p['learned_skills'] if name != 'lock'},
                purchases=p['purchases'], ella_path=state['ella_path'], ella_encounter=state['ella_encounter'])


class Run(CombatRun):
    """Pause at each shop and emit labels once the corresponding horizon resolves."""

    def __init__(self, game_seed, policy_seed, action_cap, cycle, version,
                 *, random_shop=False, budgets=BUDGETS, collect_uniform=False, collect_replay=True):
        super().__init__(game_seed, policy_seed, action_cap)
        self.cycle = cycle
        self.start_version = self.version = version
        self.random_shop = random_shop
        self.collect_uniform = collect_uniform
        self.collect_replay = collect_replay
        self.budgets = budgets
        self.event_route = None
        self.event_version = None
        self.shop_entries = []
        self.skill_actions = Counter()
        self.skill_casts = Counter()
        self._entry = None
        self._entry_action_index = 0
        self._shop = None
        self._uniform_shadow = None
        self._visit_decisions = []
        self._active_transition = None
        self._pending = []
        self._samples = []
        self._transitions = []

    @property
    def phase(self):
        return 'terminal' if self.done else self.state['phase']

    def _reward_since(self, floor):
        reward = float(self.floor - floor)
        self.reward += reward
        return reward

    def step(self, action_id):
        action_id = index(action_id)
        legal = dict(self._decision_actions())
        if action_id not in legal:
            raise ValueError(f'Action {action_id} is masked in this phase')
        action = legal[action_id]
        floor_before = self.floor
        if self.phase == 'combat':
            label = 'sacrifice' if action_id == 6 else f'slot_{action_id + 1}'
            self.combat_actions[label] += 1
            self.skill_actions[action[1]] += 1
            original = combat.resolve_skill

            def observe_cast(game, actor, target, name):
                if actor is game.player:
                    self.skill_casts[name] += 1
                return original(game, actor, target, name)

            # This observation-only probe forwards the unmodified simulator call, as in the random baseline.
            with patch.object(combat, 'resolve_skill', observe_cast):
                self._advance(action)
        else:
            self.reward_actions[action[1]] += 1
            self._advance(action)
        if self.phase == 'event':
            self._advance(self.policy_rng.choice(self.game.get_legal_actions()))
        return self._reward_since(floor_before)

    def shop_observation(self):
        if self.phase != 'shop' or self.random_shop:
            raise RuntimeError('There is no ordered shop decision')
        return self._shop.observation()

    def choose_shop(self, choice, q, explored):
        floor_before = self.floor
        transition = dict(state=self.shop_observation(), action=int(choice), source='dqn',
                          version=self.version, cycle=self.cycle, terminal=False, reward=0.0)
        self._active_transition = transition
        transition['q'] = q

        def advance(action):
            if self.done:
                return False
            self._advance(action)
            return True

        info = self._shop.step(choice, advance)
        self._visit_decisions.append(dict(info, q=q, explored=explored))
        self._active_transition = None
        if self.phase == 'shop' and self.collect_replay:
            transition['next_state'] = self.shop_observation()
            self._transitions.append(transition)
        return self._reward_since(floor_before)

    def choose_random_shop(self):
        if not self.random_shop or self.phase != 'shop':
            raise RuntimeError('There is no pending uniform-random shop')
        floor_before = self.floor
        while self.phase == 'shop':
            self._advance(self.policy_rng.choice(self.game.get_legal_actions()))
        return self._reward_since(floor_before)

    def _advance(self, action):
        before = self.state
        if before['phase'] == 'event':
            self.event_route = {'pickpocket': 'A', 'help': 'B'}[action[1]]
            self.event_version = self.version
        exiting = before['phase'] == 'shop' and action == ('leave_shop',)
        if exiting:
            region = self._entry['region']
            exit_state = shop_state(before, region, self.actions)
            transitions = []
            terminal = self._active_transition
            if self._uniform_shadow is not None:
                history = [tuple(row['action']) for row in self.shop_actions[self._entry_action_index:]]
                verified = compatible_trace(self._uniform_shadow, self.game, history, region,
                                            self.event_route, self.budgets)
                if verified is not None:
                    transitions = [dict(row, source='uniform', version=self.version, cycle=self.cycle,
                                        terminal=False, reward=0.0) for row in verified]
                    terminal = transitions.pop()
                    self._transitions.extend(transitions)
                exit_state['uniform_replay_compatible'] = verified is not None
                self._uniform_shadow = None
        super()._advance(action)
        entered_shop = before['phase'] != 'shop' and self.state['phase'] == 'shop'
        if entered_shop:
            self._entry = shop_state(self.state, before['area'], self.actions)
            self._entry.update(version=self.version, cycle=self.cycle)
            self.shop_entries.append(self._entry)
            self._entry_action_index = len(self.shop_actions)
            self._visit_decisions = []
            if self.random_shop:
                self._uniform_shadow = shadow_shop(self.game) if self.collect_uniform else None
            else:
                self._shop = Shop(self.game, self._entry['region'], self.event_route, self.budgets)
        if exiting:
            row = self.shop_exits[-1]
            q = terminal.get('q') if terminal else None
            row.update(exit_state, entry_gold=self._entry['gold'], exit_gold=before['gold'],
                       action_start=self._entry_action_index, action_stop=len(self.shop_actions),
                       exit_q=q, prediction=None if q is None else min(1.0, max(0.0, q)),
                       dqn_actions=self._visit_decisions, version=self.version, cycle=self.cycle)
            self._pending.append((terminal, row))
            self._entry = None
        # A successful transition can occur on the last allowed primitive action.
        reached_final_floor = before['floor'] < 100 and self.state['floor'] == 100
        if self._pending and (self.done or entered_shop or reached_final_floor):
            self._resolve_samples()

    def _resolve_samples(self):
        remaining = []
        for terminal, row in self._pending:
            if attained_target(row['floor'], self.state):
                target, reason = 1, 'milestone'
            elif not self.done:
                remaining.append((terminal, row))
                continue
            elif self.state['phase'] == 'game_over':
                target, reason = 0, 'game_over'
            else:
                target, reason = None, 'action_limit'
            row.update(target=target, target_reason=reason, resolution_version=self.version,
                       resolved_floor=self.floor, resolved_action=self.actions)
            self._samples.append(dict(target=target, shop_floor=row['floor'], exit_q=row['exit_q'],
                                      version=row['version'], cycle=row['cycle'], prediction=row['prediction'],
                                      game_seed=self.game_seed, policy_seed=self.policy_seed, reason=reason,
                                      resolution_version=self.version, exit_action=row['action'],
                                      resolved_action=self.actions, resolved_floor=self.floor))
            if terminal is not None and target is not None and self.collect_replay:
                self._transitions.append(dict(terminal, next_state=None, terminal=True,
                                              reward=float(target)))
        self._pending = remaining

    def drain_samples(self):
        rows, self._samples = self._samples, []
        return rows

    def drain_transitions(self):
        rows, self._transitions = self._transitions, []
        return rows

    def result(self):
        row = super().result()
        return dict(row, skill_actions=dict(self.skill_actions), skill_casts=dict(self.skill_casts),
                    cycle=self.cycle, start_version=self.start_version, version=self.version,
                    event_route=self.event_route, event_version=self.event_version,
                    effective_ella_path=self.state['ella_path'], shop_entries=self.shop_entries,
                    unfinished_shop=(None if self._entry is None else
                                     dict(floor=self._entry['floor'], dqn_actions=self._visit_decisions)),
                    special_shop_entered=any(entry['floor'] == 99 for entry in self.shop_entries),
                    special_shop_exited=any(exit_state['floor'] == 99 for exit_state in self.shop_exits))

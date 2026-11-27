"""Combat/reward-only training interface over the unmodified simulator."""

from collections import Counter
from functools import lru_cache
from math import copysign, log1p
from operator import index
from random import Random

import numpy as np

from simulator.combat import _priority
from simulator.data_loader import GameData
from simulator.game import Game


FEATURE_COUNT = 583
COMBAT, REWARD = 0, 1
REWARD_CHOICES = ('Gold', 'Potion', 'Initialize', 'Stat', 'Exchange')
TYPE_ORDER = ('Normal', 'Beast', 'Elemental', 'Undead')
STATUS_ORDER = ('stun', 'poison', 'sprout', 'regeneration', 'target', 'sleep', 'reverse',
                'aurora', 'immunity', 'imitation', 'helmet')
CHARACTER_NUMBERS = ('hp', 'max_hp', 'attack', 'defense', 'speed', 'base_attack',
                     'base_defense', 'base_speed', 'ACC', 'EVA', 'DR')


@lru_cache(maxsize=1)
def game_data():
    return GameData()


def scale(value, unit):
    return copysign(log1p(abs(value) / unit), value)


def character_features(character):
    conditions = {status['Name'] for status in character['status_conditions']}
    return [*(scale(character[key], 100) for key in CHARACTER_NUMBERS),
            *(character['type'] == name for name in TYPE_ORDER),
            *(status in conditions for status in STATUS_ORDER)]


def skill_features(game, name, categories):
    if name is None or name == 'lock':
        return [0.0] * 73
    skill = game.data.skills[name]
    return [1.0, *(skill['type'] == kind for kind in TYPE_ORDER),
            skill['power'] / 100, skill['accuracy'] / 100, _priority(game, name),
            *(skill['category'] == category for category in categories)]


class Run:
    """Expose only learned decisions, with original random shop/event actions between them."""

    def __init__(self, game_seed, policy_seed, action_cap):
        if action_cap < 1:
            raise ValueError('The engine-action limit must be positive')
        self.game_seed = game_seed
        self.policy_seed = policy_seed
        self.action_cap = action_cap
        self.game = Game(game_data())
        self.policy_rng = Random(policy_seed)
        self.state = self.game.reset(game_seed)
        self.categories = tuple(sorted({skill['category'] for skill in self.game.data.skills.values()}))
        self.regions = tuple(self.game.data.regions)
        if len(self.categories) != 65 or len(self.regions) != 11:
            raise ValueError('The 583-input schema requires 65 skill categories and 11 regions')
        self.actions = 0
        self.floor = self.state['floor']
        self.reward = 0.0
        self.combat_actions = Counter()
        self.reward_actions = Counter()
        self.shop_actions = []
        self.event_actions = []
        self.shop_exits = []
        self.shop_entry_gold = None

    @property
    def done(self):
        return self.game.is_terminal() or self.actions >= self.action_cap

    def _decision_actions(self):
        if self.done:
            raise RuntimeError('This episode is over')
        legal = self.game.get_legal_actions()
        if self.state['phase'] == 'combat':
            slots = [*(('skill', name) for name in self.state['player']['current_skills']), ('skill', 'sacrifice')]
        elif self.state['phase'] == 'reward':
            slots = [('reward', name) for name in REWARD_CHOICES]
        else:
            raise RuntimeError(f"Unexpected learned decision phase: {self.state['phase']}")
        # Preserve the engine's ordering separately for exact uniform-policy replay.
        return [(slots.index(action), action) for action in legal]

    def observation(self):
        legal = self._decision_actions()
        state = self.state
        phase = COMBAT if state['phase'] == 'combat' else REWARD
        player = state['player']
        features = character_features(player)
        features.extend(character_features(state['monster']) if phase == COMBAT else [0.0] * 26)
        for name in player['current_skills']:
            availability, *skill = skill_features(self.game, name, self.categories)
            features.extend([availability, scale(player['memorized_skills'].get(name, 0), 10), *skill])
        features.append(state['floor'] / 100)
        features.extend(state['area'] == region for region in self.regions)
        features.append(scale(state['gold'], 1000))
        features.extend(skill_features(self.game, state['reward_skill'] if phase == REWARD else None, self.categories))
        features.append(phase)
        mask = np.zeros(7, dtype=np.bool_)
        mask[[action_id for action_id, _ in legal]] = True
        if len(features) != FEATURE_COUNT:
            raise ValueError(f'Expected {FEATURE_COUNT} input features, got {len(features)}')
        return np.asarray(features, dtype=np.float32), phase, mask

    def random_action(self):
        """Replay the old policy when the caller chooses uniformly in all phases."""
        return self.policy_rng.choice(self._decision_actions())[0]

    def step(self, action_id):
        """Apply a learned action and return +1 per floor gained, including bridged moves."""
        action_id = index(action_id)
        legal = dict(self._decision_actions())
        if action_id not in legal:
            raise ValueError(f'Action {action_id} is masked in this phase')
        action = legal[action_id]
        floor_before = self.floor
        if self.state['phase'] == 'combat':
            label = 'sacrifice' if action_id == 6 else f'slot_{action_id + 1}'
            self.combat_actions[label] += 1
        else:
            self.reward_actions[action[1]] += 1
        self._advance(action)
        while not self.done and self.state['phase'] in {'shop', 'event'}:
            self._advance(self.policy_rng.choice(self.game.get_legal_actions()))
        reward = float(self.floor - floor_before)
        self.reward += reward
        return reward

    def _advance(self, action):
        before = self.state
        phase = before['phase']
        if phase in {'shop', 'event'}:
            log = self.shop_actions if phase == 'shop' else self.event_actions
            log.append({'floor': before['floor'], 'action': list(action)})
        self.state = self.game.step(action)
        self.actions += 1
        self.floor = max(self.floor, self.state['floor'])
        if phase != 'shop' and self.state['phase'] == 'shop':
            self.shop_entry_gold = self.state['gold']
        if phase == 'shop' and self.state['phase'] != 'shop':
            player = before['player']
            slots = [dict(skill=name, uses=player['memorized_skills'][name], mana_cost=player['skill_costs'][name])
                     for name in player['current_skills'] if name != 'lock']
            self.shop_exits.append(dict(floor=before['floor'], unused_stat_points=player['available_points'],
                                        max_mana=player['max_mana'], used_mana=player['max_mana'] - player['current_mana'],
                                        remaining_mana=player['current_mana'], entry_gold=self.shop_entry_gold,
                                        exit_gold=before['gold'], slots=slots))
            self.shop_entry_gold = None

    def result(self):
        if not self.done:
            raise RuntimeError('An unfinished episode has no final result')
        terminal = self.game.is_terminal()
        outcome = self.state['phase'] if terminal else 'timeout'
        monster = self.state['monster']
        return dict(game_seed=self.game_seed, policy_seed=self.policy_seed, floor=self.floor, actions=self.actions,
                    outcome=outcome, detail=None if terminal else 'action_limit', reward=self.reward,
                    combat_actions=dict(self.combat_actions), reward_actions=dict(self.reward_actions),
                    shop_actions=self.shop_actions, event_actions=self.event_actions, shop_exits=self.shop_exits,
                    death_monster=monster['name'] if outcome == 'game_over' else None,
                    death_monster_id=monster['monster_id'] if outcome == 'game_over' else None,
                    timeout_phase=self.state['phase'] if not terminal else None,
                    timeout_monster=monster['name'] if not terminal and self.state['phase'] == 'combat' else None)

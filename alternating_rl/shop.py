"""Ordered Shop DQN interface over the unchanged legal simulator actions."""

from functools import lru_cache
from operator import index
from types import SimpleNamespace

import numpy as np

from combat_rl.environment import TYPE_ORDER, scale
from simulator import shop as engine_shop
from .features import BASE_FEATURE_COUNT, BASE_STATS, _vocabulary, encode


GOLD, STATS, SKILLS = range(3)
HEAD_NAMES = ('gold', 'stats', 'skills')
HEAD_SIZES = (98, 18, 541)
BUDGETS = (25, 100, 125)
FEATURE_COUNT = BASE_FEATURE_COUNT + 3 + 1 + 1 + 5 + 3 + 4
ITEMS = ('blue_key', 'red_key', 'blue_mana_potion', 'elemental_mana_potion', 'undead_mana_potion')
EXTRA_STATS = ('ACC', 'EVA', 'DR')
FINISH = ('finish',)


@lru_cache(maxsize=8)
def _action_schema(data):
    names, _, _ = _vocabulary(data)
    types = ('Undead', 'Elemental', 'Beast')
    increments = tuple(data.config['shop']['redistribution_steps'])
    potions = tuple(data.config['shop']['mana_potions_in_order'])
    if increments != (1, 10) or ITEMS[2:] != potions:
        raise ValueError('Unexpected original shop increments or purchase catalog')
    gold = (*[('reduce_skill_cost', name) for name in names], *[('buy', item) for item in ITEMS],
            *[('change_type', kind) for kind in types], ('reroll',), FINISH)
    stats = (*[('adjust_stat', stat, delta) for stat in BASE_STATS
               for delta in (-10, -1, 1, 10)], ('reset_redistribution',), FINISH)
    skills = (*[('replace_skill', slot, name) for slot in range(6) for name in names],
              *[(verb, slot) for verb in ('memorize', 'forget') for slot in range(6)], FINISH)
    catalogs = gold, stats, skills
    if tuple(map(len, catalogs)) != HEAD_SIZES:
        raise ValueError('Unexpected Shop DQN action catalog')
    return catalogs, tuple({action: i for i, action in enumerate(catalog)} for catalog in catalogs)


def _additional(game):
    p = game.player
    return {key: getattr(p, key) for stat in EXTRA_STATS for key in (stat, f'base_{stat}')}


class Shop:
    """Map each high-level stage decision to genuine simulator shop actions."""

    def __init__(self, game, previous_area, event_choice, budgets=BUDGETS):
        self.game = game
        self.previous_area = previous_area
        self.event_choice = event_choice
        self.budgets = tuple(budgets)
        if len(self.budgets) != 3 or any(type(count) is not int or count < 1 for count in self.budgets):
            raise ValueError('Provide a positive modification cap for each head')
        if game.phase != 'shop' or game.shop_menu != 'main':
            raise ValueError('An ordered shop visit starts at the shop main menu')
        self.catalogs, self.ids = _action_schema(game.data)
        self.head = GOLD
        self.counts = [0, 0, 0]
        self.done = self.interrupted = False

    def _descriptor(self, primitive):
        verb, *args = primitive
        if self.head == SKILLS and verb in ('memorize', 'forget', 'replace_skill'):
            return verb, self.game.player.current_skills.index(args[0]), *args[1:]
        return tuple(primitive)

    def action_id(self, descriptor):
        """Look up a stable head-local action ID, regardless of current availability."""
        return self.ids[self.head][tuple(descriptor)]

    def primitive_id(self, primitive):
        """Look up a legal real modification in the current head, if one exists."""
        try:
            i = self.ids[self.head].get(self._descriptor(primitive))
        except ValueError:
            return None
        return i if i in self._legal() and tuple(primitive) != FINISH else None

    def legal_engine_actions(self):
        """Map legal IDs to original engine tuples, with None for the stage exit."""
        return {i: (primitive if menu is not None else None)
                for i, (menu, primitive) in self._legal().items()}

    def _legal(self):
        if self.done or self.interrupted:
            raise RuntimeError('Ordered shop visit has no further decisions')
        menus = ('costs', 'merchandise') if self.head == GOLD else (HEAD_NAMES[self.head],)
        allowed = {self.action_id(FINISH): (None, FINISH)}
        for menu in menus:
            # Shop legality reads this shallow facade. It does not run actions or access RNG.
            view = SimpleNamespace(data=self.game.data, player=self.game.player, gold=self.game.gold,
                                   reroll_cost=self.game.reroll_cost, shop_menu=menu)
            for move in engine_shop.legal_actions(view):
                if move != ('shop_back',):
                    allowed[self.action_id(self._descriptor(move))] = menu, move
        return allowed

    def observation(self):
        allowed = self._legal()
        p = self.game.player
        prefix = encode(self.game, previous_area=self.previous_area, event_choice=self.event_choice)
        suffix = np.asarray([
            *(self.head == stage for stage in range(3)),
            1 - self.counts[self.head] / self.budgets[self.head], scale(self.game.reroll_cost, 100),
            *(item in p.purchases for item in ITEMS),
            *(scale(getattr(p, f'base_{name}'), 100) for name in EXTRA_STATS),
            *(p.base_type == name for name in TYPE_ORDER),
        ], dtype=np.float32)
        features = np.concatenate((prefix, suffix))
        if features.shape != (FEATURE_COUNT,):
            raise ValueError('Unexpected Shop DQN feature shape')
        mask = np.zeros(HEAD_SIZES[self.head], dtype=np.bool_)
        mask[list(allowed)] = True
        return features, self.head, mask

    def step(self, action_id, advance):
        """Execute a decision. The callback returns False only if it cannot execute the primitive."""
        action_id = index(action_id)
        allowed = self._legal()
        if action_id not in allowed:
            raise ValueError(f'Action {action_id} is masked in Shop head {self.head}')
        head = self.head
        descriptor = self.catalogs[head][action_id]
        menu, primitive = allowed[action_id]
        info = dict(head=head, action=descriptor, engine_action=None, stage_ended=False,
                    end_reason=None, modifications=self.counts[head])

        def apply(move):
            if advance(move) is False:
                self.interrupted = True
                return False
            return True

        if menu is not None:
            if self.game.shop_menu != menu:
                if self.game.shop_menu != 'main' and not apply(('shop_back',)):
                    return dict(info, interrupted=True)
                if not apply(('shop_menu', menu)):
                    return dict(info, interrupted=True)
            if primitive == ('reroll',):
                before = _additional(self.game)
                price = self.game.reroll_cost
            if not apply(primitive):
                return dict(info, interrupted=True)
            info['engine_action'] = primitive
            self.counts[head] += 1
            info['modifications'] = self.counts[head]
            if primitive == ('reroll',):
                info['reroll'] = dict(before=before, after=_additional(self.game), cost=price,
                                      next_cost=self.game.reroll_cost)

        if descriptor == FINISH or self.counts[head] >= self.budgets[head]:
            if self.game.shop_menu != 'main' and not apply(('shop_back',)):
                return dict(info, interrupted=True)
            if head == SKILLS:
                if not apply(('leave_shop',)):
                    return dict(info, interrupted=True)
                self.done = True
            else:
                self.head += 1
            info['stage_ended'] = True
            info['end_reason'] = 'early' if descriptor == FINISH else 'cap'
        return dict(info, interrupted=False)

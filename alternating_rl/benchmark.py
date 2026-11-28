"""Measure ordered DQN shop decisions from one recorded floor-10 visit."""

from copy import deepcopy
import json
from random import Random
from statistics import median
from time import perf_counter

import torch

from simulator.game import Game
from .dqn import ShopDQN
from .shop import FEATURE_COUNT, HEAD_SIZES, Shop


def main():
    torch.manual_seed(314159)
    torch.set_num_threads(2)
    model = ShopDQN(FEATURE_COUNT, HEAD_SIZES, [128, 128]).eval()
    game = Game()
    state = game.reset(2538817474026044487)
    policy = Random(3909549678122254972)
    while state['phase'] != 'shop' and not game.is_terminal():
        previous_area = state['area']
        state = game.step(policy.choice(game.get_legal_actions()))
    assert state['phase'] == 'shop' and state['floor'] == 10
    observation = Shop(game, previous_area, None).observation()
    model.act([observation], Random(0), 0)
    encoding, scoring, visits, decisions = [], [], [], []
    for repeat in range(30):
        started = perf_counter()
        observation = Shop(game, previous_area, None).observation()
        encoded = perf_counter()
        model.act([observation], Random(repeat), 0)
        encoding.append(encoded - started)
        scoring.append(perf_counter() - encoded)
        shadow = deepcopy(game, {id(game.data): game.data})
        shop = Shop(shadow, previous_area, None)
        rng = Random(repeat)
        steps = 0
        started = perf_counter()
        while not shop.done:
            action, _, _ = model.act([shop.observation()], rng, .15)[0]
            shop.step(action, shadow.step)
            steps += 1
        visits.append(perf_counter() - started)
        decisions.append(steps)
    print(json.dumps(dict(initial_encoding_ms=median(encoding) * 1000, one_head_scoring_ms=median(scoring) * 1000,
                          complete_visit_ms=median(visits) * 1000, decisions_per_visit=median(decisions))))


if __name__ == '__main__':
    main()

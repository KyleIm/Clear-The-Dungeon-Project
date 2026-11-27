"""Measure candidate planning and batched scoring from one recorded floor-10 visit."""

import json
from random import Random
from statistics import median, mean
from time import perf_counter

import numpy as np
import torch

from simulator.game import Game
from .candidates import generate
from .runner import DEFAULT, read_config
from .shop_value import ShopValue


def main():
    cfg = read_config(DEFAULT)['shop']
    torch.set_num_threads(2)
    model = ShopValue(cfg['hidden_dims'])
    game = Game()
    state = game.reset(2538817474026044487)
    policy = Random(3909549678122254972)
    while state['phase'] != 'shop' and not game.is_terminal():
        previous_area = state['area']
        state = game.step(policy.choice(game.get_legal_actions()))
    assert state['phase'] == 'shop' and state['floor'] == 10
    for count in (8, 16, 24):
        planning, scoring, generated = [], [], []
        for repeat in range(30):
            started = perf_counter()
            plans = generate(game, count, Random(repeat), previous_area=previous_area, max_edits=cfg['max_edits'])
            planned = perf_counter()
            with torch.inference_mode():
                model(np.stack([plan.features for plan in plans])).numpy()
            planning.append(planned - started)
            scoring.append(perf_counter() - planned)
            generated.append(len(plans))
        print(json.dumps(dict(requested=count, average_candidates=mean(generated),
                              planning_ms=median(planning) * 1000, scoring_ms=median(scoring) * 1000)))


if __name__ == '__main__':
    main()

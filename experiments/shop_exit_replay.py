"""Replay historical shop-reaching episodes and capture each shop exit as CSV."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sqlite3

from experiments.random_baseline import episode
from simulator.game import Game


STATS = ('attack', 'defense', 'hp', 'speed')


class ShopExits:
    def start(self, state):
        self.entry_gold = None
        self.rows = []

    def step(self, before, after):
        if before['phase'] != 'shop' and after['phase'] == 'shop':
            self.entry_gold = after['gold']
        elif before['phase'] == 'shop' and after['phase'] != 'shop':
            assert self.entry_gold is not None
            player = before['player']
            slots = [(name, player['memorized_skills'][name], player['skill_costs'][name])
                     for name in player['current_skills'] if name != 'lock']
            used = sum(uses * cost for _, uses, cost in slots)
            assert used + player['current_mana'] == player['max_mana']
            self.rows.append(dict(shop_floor=before['floor'], next_floor=after['floor'],
                                  **{stat: player[f'base_{stat}'] for stat in STATS},
                                  unused_stat_points=player['available_points'], max_mana=player['max_mana'],
                                  used_mana=used, remaining_mana=player['current_mana'],
                                  slots=json.dumps(slots, separators=(',', ':')),
                                  zero_use_skills=sum(uses == 0 for _, uses, _ in slots),
                                  entry_gold=self.entry_gold, exit_gold=before['gold']))
            self.entry_gold = None


def replay_batch(job):
    records, action_cap, timeout = job
    game = Game()
    rows = []
    for game_id, game_seed, policy_seed, floor, actions, outcome, detail in records:
        tracker = ShopExits()
        result = episode(game, (game_id, game_seed, policy_seed), action_cap, timeout, observer=tracker)
        recorded = (floor, actions, outcome, detail)
        actual = (result.floor, result.actions, result.outcome, result.detail)
        if recorded != actual:
            raise ValueError(f'Game {game_id}: recorded {recorded!r}, replayed {actual!r}')
        if not tracker.rows:
            raise ValueError(f'Game {game_id} reached floor {floor} without a shop exit')
        rows.extend(dict(game_id=game_id, game_seed=game_seed, policy_seed=policy_seed,
                         final_floor=floor, actions=actions, outcome=outcome, **row) for row in tracker.rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        settings = json.loads(db.execute('SELECT config FROM run WHERE id=1').fetchone()[0])
        records = db.execute('''SELECT game_id, game_seed, policy_seed, floor, actions, outcome, detail
                                FROM episodes WHERE floor >= 11 ORDER BY game_id''').fetchall()
    jobs = [(records[start:start + 256], settings['action_cap'], settings['timeout_seconds'])
            for start in range(0, len(records), 256)]
    if not jobs:
        raise ValueError('No historical episodes reached floor 11')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix('.partial')
    exits = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool, pending.open('w', newline='', encoding='utf-8') as file:
        writer = None
        for rows in pool.map(replay_batch, jobs):
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            exits += len(rows)
    pending.replace(args.output)
    print(f'{len(records):,} complete matches, {exits:,} recorded shop exits', flush=True)


if __name__ == '__main__':
    main()

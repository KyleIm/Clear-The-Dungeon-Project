"""Checkpoint-aware shop samples and streaming reports for alternating learning."""

from collections import Counter, defaultdict
from contextlib import closing
import csv
import json
from math import isfinite, sqrt
from pathlib import Path
import sqlite3

import numpy as np

from combat_rl.report import (
    TABLES, EpisodeStats, Store as CombatStore, _baseline, _encode, _fmt, _progress_rows, _read_only,
)


SHOP_SCHEMA = '''
CREATE TABLE IF NOT EXISTS shop_samples (
    id INTEGER PRIMARY KEY, split TEXT NOT NULL, step INTEGER NOT NULL,
    shop_floor INTEGER NOT NULL, version INTEGER NOT NULL, cycle INTEGER NOT NULL,
    target INTEGER CHECK(target IN (0, 1)), prediction REAL CHECK(prediction >= 0 AND prediction <= 1),
    features BLOB NOT NULL, payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS shop_sample_training_window ON shop_samples(split, version DESC, id DESC)
    WHERE target IS NOT NULL;
'''
SHOP_TABLES = (*TABLES, 'shop_samples')
REWARDS = ('Potion', 'Initialize', 'Gold', 'Stat', 'Exchange')
ROOT = Path(__file__).resolve().parents[1]
HEADS = ('Gold Expenditure', 'Stat Redistribution', 'Skill Configuration')


class Store(CombatStore):
    """Commit episode telemetry and milestone labels on the same SQLite boundary."""

    def __enter__(self):
        super().__enter__()
        try:
            self.db.executescript(SHOP_SCHEMA)
        except BaseException:
            self.db.close()
            raise
        return self

    def sample(self, split, step, row):
        data = dict(row)
        values = data.pop('features', None)
        features = np.asarray(values, dtype='<f4') if values is not None else None
        if features is not None and (features.shape != (671,) or not np.isfinite(features).all()):
            raise ValueError('Legacy shop samples require 671 finite features')
        target = data.pop('target')
        if target is not None and target not in (0, 1):
            raise ValueError('Shop sample target must be binary or censored')
        prediction = data.pop('prediction', None)
        self._write('''INSERT INTO shop_samples(split, step, shop_floor, version, cycle,
                           target, prediction, features, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (split, step, int(data.pop('shop_floor')), int(data.pop('version')),
                     int(data.pop('cycle')), None if target is None else int(target),
                     None if prediction is None else float(prediction),
                     features.tobytes() if features is not None else b'', _encode(data)))

    def recent_samples(self, limit, current_version):
        if type(limit) is not int or limit < 1:
            raise ValueError('Shop replay window must be a positive integer')
        rows = self.db.execute('''SELECT step, shop_floor, version, cycle, target, prediction, features, payload
                                  FROM shop_samples
                                  WHERE split = 'train' AND target IS NOT NULL AND length(features) > 0 AND version <= ?
                                  ORDER BY version DESC, id DESC LIMIT ?''', (current_version, limit))
        return [dict(json.loads(payload), step=step, shop_floor=floor, version=version, cycle=cycle,
                     target=target, prediction=prediction,
                     features=np.frombuffer(features, dtype='<f4').copy())
                for step, floor, version, cycle, target, prediction, features, payload in rows]

    def marks(self):
        self.flush()
        return {table: self.db.execute(f'SELECT COALESCE(MAX(id),0) FROM {table}').fetchone()[0]
                for table in SHOP_TABLES}

    def restore(self, marks):
        for table in SHOP_TABLES:
            self.db.execute(f'DELETE FROM {table} WHERE id > ?', (marks[table],))
        self.flush()


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _percent(numerator, denominator):
    return _fmt(_ratio(numerator, denominator), '.2%')


def _wilson(successes, n):
    if not n:
        return '—'
    fraction = successes / n
    z = 1.95996398454
    scale = 1 + z * z / n
    center = (fraction + z * z / (2 * n)) / scale
    half = z * sqrt(fraction * (1 - fraction) / n + z * z / (4 * n * n)) / scale
    return f'{max(0, center - half):.2%}–{min(1, center + half):.2%}'


def _markdown_table(columns, rows):
    return ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |',
            *('| ' + ' | '.join(str(value).replace('|', '/') for value in row) + ' |' for row in rows), '']


def _history_baseline(directory, label):
    """Read the final held-out row from an existing published run report."""
    report = ROOT / 'docs' / directory / 'training_report.md'
    line = next((line for line in reversed(report.read_text(encoding='utf-8').splitlines())
                 if line.startswith('| Fixed-seed evaluation at ')), None) if report.exists() else None
    if line is None:
        return None
    fields = [value.strip() for value in line.strip('|').split('|')]
    if len(fields) < 7:
        return None
    return (label, *fields[1:7])


class ShopStats:
    """Track exact bounded marginals and outcome calibration per shop floor."""

    def __init__(self):
        self.counts = Counter()
        self.fields = defaultdict(Counter)
        self.skills = Counter()
        self.uses = Counter()
        self.loadouts = Counter()
        self.purchases = Counter()
        self.calibration = defaultdict(Counter)
        self.heads = defaultdict(Counter)
        self.head_actions = defaultdict(Counter)
        self.head_q = defaultdict(Counter)
        self.reroll = defaultdict(Counter)

    def add_snapshot(self, label, snapshot):
        if not isinstance(snapshot, dict):
            return
        for key in ('attack', 'defense', 'hp', 'speed', 'unused_stat_points', 'max_mana',
                    'used_mana', 'remaining_mana', 'gold', 'unlocked_slots'):
            value = snapshot.get(key, snapshot.get('stats', {}).get(key))
            if isinstance(value, (float, int)):
                self.fields[(label, key)][value] += 1
        slots = snapshot.get('slots', ())
        if label != 'exit':
            return
        allocations = [snapshot.get(stat) for stat in ('attack', 'defense', 'hp', 'speed')]
        if all(isinstance(value, (int, float)) for value in allocations):
            discretionary = [max(0, value - 1) for value in allocations]
            self.counts['stat_exits'] += 1
            if sum(discretionary) and max(discretionary) > sum(discretionary) / 2:
                self.counts['concentrated_stats'] += 1
        loadout = []
        stats = [snapshot.get(stat, snapshot.get('stats', {}).get(stat))
                 for stat in ('attack', 'defense', 'hp', 'speed')]
        if all(isinstance(stat, (int, float)) for stat in stats) and sum(stats) > 0:
            self.counts['known_stat_allocations'] += 1
            self.counts['concentrated_stat_allocations'] += max(stats) / sum(stats) >= .5
        for slot in slots:
            if isinstance(slot, dict):
                name = slot.get('name', slot.get('skill'))
                count = slot.get('uses', slot.get('memorize', 0))
            else:
                name, count = slot[:2]
            if name not in (None, 'lock'):
                self.skills[name] += 1
                self.uses[(name, count)] += 1
                loadout.append(name)
        self.loadouts[tuple(loadout)] += 1

    def add_visit(self, visit):
        self.counts['entries'] += 1
        before, after = visit.get('entry'), visit.get('exit')
        self.add_snapshot('entry', before)
        if after is None:
            return
        self.counts['exits'] += 1
        self.add_snapshot('exit', after)
        self.counts['explored'] += bool(visit.get('explored'))
        self.counts['candidate_visits'] += bool(visit.get('candidates'))
        self.counts['candidates'] += int(visit.get('candidates') or 0)
        self.add_dqn(visit.get('dqn_actions', ()))
        for action in visit.get('actions', ()):
            if isinstance(action, dict):
                action = action['action']
            if action[0] in ('buy', 'reduce_cost', 'reduce_skill_cost', 'change_type', 'reroll'):
                self.purchases[' '.join(map(str, action))] += 1
        if isinstance(before, dict) and isinstance(after, dict):
            first, last = before.get('unlocked_slots'), after.get('unlocked_slots')
            if isinstance(first, (int, float)) and isinstance(last, (int, float)):
                self.counts['keys'] += max(0, last - first)
            if 'gold' in before and 'gold' in after:
                self.fields[('exit', 'gold_spent')][before['gold'] - after['gold']] += 1

    def add_dqn(self, actions):
        previous = {}
        for row in actions:
            head = HEADS[row['head']]
            action = tuple(row['action'])
            name = ' '.join(map(str, action))
            self.head_actions[head][name] += 1
            counts = self.heads[head]
            counts['selections'] += 1
            counts['explored'] += bool(row.get('explored'))
            end = row.get('end_reason')
            if end in ('early', 'cap'):
                counts[end] += 1
                counts['finished_modifications'] += row['modifications']
            if action[0] != 'finish' and not row.get('interrupted'):
                counts['modifications'] += 1
                prior = previous.get(head)
                if prior is not None:
                    counts['adjacent_modifications'] += 1
                    counts['repeat'] += prior['action'] == row['action']
                    counts['inverse'] += _inverse(prior, row)
                previous[head] = row
            q = row.get('q', row.get('q_value'))
            if q is not None and isfinite(q):
                self.head_q[head][_q_band(q)] += 1
            reroll = row.get('reroll')
            if reroll:
                self.add_reroll(reroll)

    def add_reroll(self, reroll):
        self.reroll['Cost'][reroll['cost']] += 1
        for label in ('before', 'after'):
            snapshot = reroll[label]
            for key in ('ACC', 'EVA', 'DR'):
                for kind, field in (('base', f'base_{key}'), ('current', key)):
                    if field in snapshot:
                        self.reroll[f'{label} {kind} {key}'][snapshot[field]] += 1

    def add_target(self, target, prediction, raw_q=None):
        key = 'censored' if target is None else 'success' if target else 'failure'
        self.counts[key] += 1
        value = raw_q if raw_q is not None else prediction
        if target is None or value is None or not isfinite(value):
            return
        group = self.calibration[_q_band(value)]
        group.update(n=1, success=target, predictions=value, squared_error=(target - value) ** 2)


def _inverse(prior, row):
    first, second = tuple(prior['action']), tuple(row['action'])
    if first[0] == second[0] == 'adjust_stat':
        return first[1] == second[1] and first[2] == -second[2]
    if {first[0], second[0]} == {'memorize', 'forget'}:
        return first[1:] == second[1:]
    if first[0] == second[0] == 'replace_skill':
        first_engine = prior.get('engine_action')
        second_engine = row.get('engine_action')
        return (first[1] == second[1] and first_engine is not None and second_engine is not None
                and first_engine[1] == second[2] and second_engine[1] == first[2])
    return False


def _q_band(value):
    return -1 if value < 0 else 10 if value > 1 else min(9, int(value * 10))


def _q_band_label(band):
    if band == -1:
        return '<0'
    if band == 10:
        return '>1'
    return f'{band / 10:.1f}–{(band + 1) / 10:.1f}'


def _median(values):
    total = sum(values.values())
    if not total:
        return None
    cumulative = 0
    middle = [(total - 1) // 2, total // 2]
    selected = []
    for value, count in sorted(values.items()):
        selected.extend(value for rank in middle if cumulative <= rank < cumulative + count)
        cumulative += count
    return sum(selected) / 2


def _cohort_row(label, stats):
    result = stats.summary()
    return (label, f'{stats.episodes:,}', _fmt(result['mean_floor'], '.2f'),
            _fmt(result['median_floor'], 'g'), _fmt(result['max_floor'], 'd'),
            _fmt(result['clear_rate'], '.2%'), _fmt(result['timeout_rate'], '.2%'))


def _dqn_lines(stats):
    if not stats.heads:
        return []
    lines = ['#### Masked DQN behavior', '',
             'Early and cap shares use completed heads as the denominator. Counts include DQN visits that '
             'timed out within the shop. Repeated and inverse edits are adjacent modifications in the same '
             'head and visit.', '']
    rows = []
    for head, counts in sorted(stats.heads.items()):
        completed = counts['early'] + counts['cap']
        rows.append((head, counts['selections'], counts['modifications'], completed,
                     _fmt(_ratio(counts['finished_modifications'], completed), '.2f'),
                     _percent(counts['early'], completed), _percent(counts['cap'], completed),
                     _percent(counts['explored'], counts['selections']), counts['repeat'], counts['inverse']))
    lines.extend(_markdown_table(('Head', 'Selections', 'Applied edits', 'Completed', 'Mean edits per completion',
                                  'Early exits', 'Cap exits', 'Exploration', 'Repeated edits', 'Inverse edits'), rows))
    lines.extend(_markdown_table(('Head', 'Selected action', 'Count', 'Share within head'),
        [(head, action, n, _percent(n, sum(counts.values())))
         for head, counts in sorted(stats.head_actions.items()) for action, n in counts.most_common()]))
    lines.extend(_markdown_table(('Head', 'Selected raw Q-value band', 'Count'),
        [(head, _q_band_label(band), n) for head, counts in sorted(stats.head_q.items())
         for band, n in sorted(counts.items())]))
    if stats.reroll:
        rolls = stats.reroll['Cost']
        lines += [f'Rerolls recorded {sum(rolls.values()):,}. Total reroll cost '
                  f'{sum(value * count for value, count in rolls.items()):,} gold.', '']
        lines.extend(_markdown_table(('Reroll metric', 'Records', 'Min', 'Median', 'Mean', 'Max'),
            [(name, sum(counts.values()), min(counts), _fmt(_median(counts), '.2f'),
              _fmt(_ratio(sum(value * n for value, n in counts.items()), sum(counts.values())), '.2f'), max(counts))
             for name, counts in sorted(stats.reroll.items()) if counts]))
    return lines


def _read_run(db):
    cohorts = defaultdict(EpisodeStats)
    segments = defaultdict(EpisodeStats)
    branch = defaultdict(EpisodeStats)
    visits = defaultdict(ShopStats)
    skills = defaultdict(Counter)
    casts = defaultdict(Counter)
    branch_99 = Counter()
    branch_99_exit = Counter()
    shop_trend = defaultdict(Counter)
    steps = db.execute('SELECT DISTINCT split,step FROM episodes WHERE split != ? ORDER BY step', ('train',)).fetchall()
    latest = {}
    for split, step in steps:
        latest[split] = step
    for split, step, payload in db.execute('SELECT split,step,payload FROM episodes ORDER BY id'):
        row = json.loads(payload)
        tag = 'train' if split == 'train' else (split, step)
        cohorts[tag].add(row)
        if split == 'train':
            segments[(row.get('cycle', 0), row.get('segment', 0))].add(row)
        for visit in row.get('shop_exits', ()):
            totals = shop_trend['train' if split == 'train' else split,
                                row.get('segment', 0) if split == 'train' else step]
            totals['exits'] += 1
            totals['unused_stats'] += visit.get('unused_stat_points', 0) > 0
            totals['unused_mana'] += visit.get('remaining_mana', 0) > 0
            totals['mean_point_numerator'] += visit.get('unused_stat_points', 0)
            totals['mean_mana_numerator'] += visit.get('remaining_mana', 0)
        if split == 'train' or step == latest[split]:
            skills[split].update(row.get('skill_actions', {}))
            casts[split].update(row.get('skill_casts', {}))
            route = row.get('event_route')
            if route in ('A', 'B'):
                event_version = row.get('event_version')
                route_key = (split, route, row.get('cycle', 0),
                             row.get('version', 0) if event_version is None else event_version)
                branch[route_key].add(row)
                branch_99[route_key] += bool(row.get('special_shop_entered'))
                branch_99_exit[route_key] += bool(row.get('special_shop_exited'))
            exits_by_floor = defaultdict(list)
            for exit_state in row.get('shop_exits', ()):
                exits_by_floor[exit_state['floor']].append(exit_state)
            for entry in row.get('shop_entries', ()):
                floor = entry['floor']
                exit_state = exits_by_floor[floor].pop(0) if exits_by_floor[floor] else None
                visit = dict(exit_state or {}, entry=entry, exit=exit_state)
                if exit_state is not None and 'action_start' in exit_state:
                    visit['actions'] = row.get('shop_actions', ())[exit_state['action_start']:exit_state['action_stop']]
                visits[(split, floor)].add_visit(visit)
            unfinished = row.get('unfinished_shop')
            if unfinished:
                visits[(split, unfinished['floor'])].add_dqn(unfinished.get('dqn_actions', ()))
    for split, step, floor, target, prediction, payload in db.execute(
            'SELECT split,step,shop_floor,target,prediction,payload FROM shop_samples ORDER BY id'):
        if split == 'train' or step == latest.get(split):
            sample = json.loads(payload)
            visits[(split, floor)].add_target(target, prediction, sample.get('exit_q', sample.get('q_value')))
    return cohorts, segments, branch, branch_99, branch_99_exit, visits, skills, casts, latest, shop_trend


def _plot(output, cohorts, latest, updates, samples, shop_trend):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    depth = axes[0, 0]
    for split in latest:
        series = [(key[1], stats.summary()['mean_floor']) for key, stats in cohorts.items()
                  if isinstance(key, tuple) and key[0] == split and stats.episodes]
        if series:
            depth.plot(*zip(*sorted(series)), marker='.', label=split)
    losses = axes[0, 1]
    for key, label in (('policy_loss', 'PPO policy'), ('value_loss', 'PPO value'),
                       ('dqn_loss', 'Shop DQN'), ('shop_loss', 'Shop BCE'), ('bce_loss', 'Shop BCE')):
        series = [(step, row[key]) for step, row in updates if row.get(key) is not None]
        if series:
            losses.plot(*zip(*series), label=label)
    survival = axes[1, 0]
    for name, stats in [('Training', cohorts['train']), *((split, cohorts[(split, step)])
                         for split, step in latest.items())]:
        if stats.episodes:
            floors = range(1, max(stats.floors) + 2)
            survival.step(floors, [stats.reach(floor) for floor in floors], where='post', label=name)
    survival.yaxis.set_major_formatter(PercentFormatter(1))
    survival.xaxis.set_major_locator(MaxNLocator(integer=True))
    calibration = axes[1, 1]
    for (split, floor), stats in sorted(samples.items(), key=lambda item: str(item[0])):
        if split == 'train' or not stats.calibration:
            continue
        scores = [row for _, row in sorted(stats.calibration.items())]
        calibration.plot([row['predictions'] / row['n'] for row in scores],
                         [row['success'] / row['n'] for row in scores], marker='o', label=f'{split} F{floor}')
    calibration.plot([0, 1], [0, 1], linestyle='--', color='#aaaaaa', label='Ideal')
    for ax, title, x, y in [(depth, 'Checkpoint evaluation', 'Completed episodes', 'Mean furthest floor'),
                            (losses, 'Optimization', 'Completed episodes', 'Loss'),
                            (survival, 'Floor reach', 'Floor', 'Fraction reaching floor'),
                            (calibration, 'Shop calibration', 'Mean exit score', 'Observed success rate')]:
        ax.set(title=title, xlabel=x, ylabel=y)
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'progress.png', dpi=150)
    plt.close(fig)

    fig, (reward, resources) = plt.subplots(1, 2, figsize=(12, 4.5))
    for split in latest:
        panels = sorted((key[1], stats) for key, stats in cohorts.items()
                        if isinstance(key, tuple) and key[0] == split and sum(stats.rewards.values()))
        if panels:
            for choice in REWARDS:
                reward.plot([step for step, _ in panels],
                            [stats.rewards[choice] / sum(stats.rewards.values()) for _, stats in panels],
                            label=f'{split} {choice}')
    for split in latest:
        series = sorted((step, counts) for (kind, step), counts in shop_trend.items() if kind == split)
        for name, key in [('Unused stat points', 'unused_stats'), ('Unused mana', 'unused_mana')]:
            if series:
                resources.plot([step for step, _ in series],
                               [count[key] / count['exits'] for _, count in series], label=f'{split} {name}')
    for ax, title in [(reward, 'Reward selection by evaluation'), (resources, 'Unused resources at shop exit')]:
        ax.set(title=title, xlabel='Completed episodes', ylabel='Share')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'decision_trends.png', dpi=150)
    plt.close(fig)


def generate_report(db_path, output_dir, baseline_db=None):
    """Write English run results without retaining raw episode rows in memory."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with closing(_read_only(db_path)) as db:
        db.execute('BEGIN')
        cohorts, segments, branches, branch_99, branch_99_exit, visits, skills, casts, latest, shop_trend = _read_run(db)
        updates = _progress_rows(db)
        evaluation = [(step, policy, json.loads(payload)) for step, policy, payload in db.execute(
            'SELECT step, policy, payload FROM evaluations ORDER BY id')]
        version_balance = db.execute('''SELECT cycle,version,shop_floor,target,COUNT(*) FROM shop_samples
                                        WHERE split = 'train' GROUP BY cycle,version,shop_floor,target
                                        ORDER BY cycle,version,shop_floor,target''').fetchall()
        revised = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='shop_replay'").fetchone()
        revised = revised or any(stats.heads for stats in visits.values())
    _plot(output, cohorts, latest, updates, visits, shop_trend)
    model = 'Masked Shop DQN' if revised else 'Shop Value Network'
    lines = ['# Phase 3C alternating training results', '',
             f'Combat and Reward PPO alternates with a separate {model}. Timeouts before the shop '
             'milestone are censored and excluded from the binary target. Checkpoint panels use fixed independent '
             'seeds. Earlier runs provide context and may use a different policy or panel.', '',
             '![Progress, losses, floor reach and calibration](progress.png)', '',
             '![Reward and shop resource trends](decision_trends.png)', '', '## Floor and outcome summary', '']
    if any(stats.reroll for stats in visits.values()):
        with (output / 'reroll_distributions.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(('split', 'shop_floor', 'metric', 'value', 'count'))
            writer.writerows((split, floor, key, value, count)
                             for (split, floor), stats in sorted(visits.items())
                             for key, values in sorted(stats.reroll.items())
                             for value, count in sorted(values.items()))
    rows = [_cohort_row('All training', cohorts['train']),
            _cohort_row('Historical random 1M', _baseline(baseline_db))]
    for folder, label in [('combat_rl_100k', 'Historical PPO 100K evaluation'),
                          ('combat_rl_1m', 'Historical PPO 1M evaluation')]:
        result = _history_baseline(folder, label)
        if result:
            rows.append(result)
    rows.extend(_cohort_row(f'{key[0]} at {key[1]:,}', stats) for key, stats in cohorts.items()
                if isinstance(key, tuple))
    lines.extend(_markdown_table(('Cohort', 'Episodes', 'Mean floor', 'Median', 'Maximum', 'Clear rate', 'Timeout rate'), rows))
    latest_cohorts = [('Training', cohorts['train']), *((split, cohorts[(split, step)]) for split, step in latest.items())]
    maximum = max((max(stats.floors, default=1) for _, stats in latest_cohorts), default=1)
    lines.extend(_markdown_table(('Floor', *(name for name, _ in latest_cohorts)),
                                 [(f, *(_fmt(stats.reach(f), '.2%') for _, stats in latest_cohorts))
                                  for f in range(1, maximum + 1)]))
    lines += ['## Progress by training segment', '']
    lines.extend(_markdown_table(('Cohort', 'Episodes', 'Mean floor', 'Median', 'Maximum', 'Clear rate', 'Timeout rate'),
                                 [_cohort_row(f'Cycle {cycle} / segment {segment}', stats)
                                  for (cycle, segment), stats in sorted(segments.items())]))
    lines += ['## Reward selection by checkpoint', '']
    lines.extend(_markdown_table(('Panel', *REWARDS), [
        (f'{key[0]} at {key[1]:,}', *(_percent(stats.rewards[name], sum(stats.rewards.values())) for name in REWARDS))
        for key, stats in cohorts.items() if isinstance(key, tuple)]))
    lines += ['## Combat decisions and terminal causes', '']
    for key, stats in latest_cohorts:
        split = 'train' if key == 'Training' else key
        lines += [f'### {key}', '']
        for title, counts in (('Combat slots', stats.combat), ('Skill selections', skills[split]),
                              ('Resolved player skill casts', casts[split]),
                              ('Reward selections', stats.rewards), ('Death opponents', stats.deaths),
                              ('Timeout reasons', stats.timeouts)):
            lines += [f'#### {title}', '']
            lines.extend(_markdown_table(('Label', 'Count', 'Share'), [(name, f'{count:,}',
                            _percent(count, sum(counts.values()))) for name, count in counts.most_common()]))
    lines += ['## Shop exits and observed milestones', '']
    if revised:
        lines += ['Exit calibration uses the raw Q-value of the selected transition. A raw value can fall outside '
                  '[0, 1], and only resolved targets are compared with Q-values.', '']
    if any(stats.reroll for stats in visits.values()):
        lines += ['[Exact reroll cost and before/after stat histograms](reroll_distributions.csv)', '']
    lines.extend(_markdown_table(('Split', 'Segment or checkpoint', 'Exits', 'Unused stats', 'Unused mana',
                                  'Mean remaining points', 'Mean remaining mana'),
        [(split, step, row['exits'], _percent(row['unused_stats'], row['exits']),
          _percent(row['unused_mana'], row['exits']), _fmt(_ratio(row['mean_point_numerator'], row['exits']), '.2f'),
          _fmt(_ratio(row['mean_mana_numerator'], row['exits']), '.2f'))
         for (split, step), row in sorted(shop_trend.items())]))
    for (split, floor), stats in sorted(visits.items(), key=lambda item: str(item[0])):
        counts = stats.counts
        labeled = counts['success'] + counts['failure']
        lines += [f'### {split} · floor {floor}', '',
                  f"Entries {counts['entries']:,}. Exits {counts['exits']:,}. Success/failure/censored "
                  f"{counts['success']:,}/{counts['failure']:,}/{counts['censored']:,}. "
                  f"Observed success {_percent(counts['success'], labeled)}. New slots {counts['keys']:,}. "
                  f"At least 50% of the four base stats in one stat at "
                  f"{_percent(counts['concentrated_stat_allocations'], counts['known_stat_allocations'])} of exits.", '']
        if counts['candidate_visits']:
            lines += [f"Mean candidates {_fmt(_ratio(counts['candidates'], counts['candidate_visits']), '.2f')}. "
                      f"Candidate exploration {_percent(counts['explored'], counts['candidate_visits'])}.", '']
        lines += [f"Stat concentration at exit {_percent(counts['concentrated_stats'], counts['stat_exits'])}. "
                  'Concentration means one stat has more than half the allocated points above the four minima.', '']
        lines.extend(_markdown_table(('Snapshot', 'Field', 'Records', 'Min', 'Median', 'Mean', 'Max', 'Zero share'),
            [(where, key, sum(group.values()), min(group), _fmt(_median(group), '.2f'),
              _fmt(_ratio(sum(v * n for v, n in group.items()), sum(group.values())), '.2f'),
              max(group), _percent(group[0], sum(group.values()))) for (where, key), group in sorted(stats.fields.items())]))
        lines.extend(_markdown_table(('Purchase or investment', 'Count'), stats.purchases.most_common()))
        lines.extend(_markdown_table(('Most common ordered skill configurations', 'Exits'),
                                     [(', '.join(combo), n) for combo, n in stats.loadouts.most_common(15)]))
        lines.extend(_markdown_table(('Most frequent equipped skills', 'Slots', 'Zero-use slots', 'Mean uses'),
            [(name, n, stats.uses[name, 0], _fmt(_ratio(sum(k * c for (skill, k), c in stats.uses.items()
                                                         if skill == name), n), '.2f'))
             for name, n in stats.skills.most_common(20)]))
        lines.extend(_dqn_lines(stats))
        lines.extend(_markdown_table(('Exit score band', 'Labeled', 'Mean exit score', 'Observed success', 'Squared error'),
            [(_q_band_label(band), row['n'], _fmt(_ratio(row['predictions'], row['n']), '.4f'),
              _percent(row['success'], row['n']), _fmt(_ratio(row['squared_error'], row['n']), '.4f'))
             for band, row in sorted(stats.calibration.items())]))
    lines += ['### Shop sample balance by policy version', '']
    lines.extend(_markdown_table(('Cycle', 'Combat/Reward version', 'Shop floor', 'Target', 'Count'),
                                 [(c, v, f, 'Censored' if t is None else t, n) for c, v, f, t, n in version_balance]))
    lines += ['## Event branches', '', 'Branch labels report the selected choice. The recorded floor-99 shop entry '
              'follows the engine route. Wilson intervals use the episode count in each cohort.', '']
    lines.extend(_markdown_table(('Split', 'A/B', 'Cycle', 'Version', 'Episodes', 'Mean floor', 'Median', 'Maximum',
                                  'F99 shop entries', 'F99 shop exits', 'Clear rate', '95% clear interval'),
        [(*key, stats.episodes, _fmt(stats.summary()['mean_floor'], '.2f'),
          _fmt(stats.summary()['median_floor'], 'g'), max(stats.floors), branch_99[key], branch_99_exit[key],
          _percent(stats.outcomes['cleared'], stats.episodes), _wilson(stats.outcomes['cleared'], stats.episodes))
         for key, stats in sorted(branches.items())]))
    lines.extend(_markdown_table(('Split', 'A/B', 'Cycle', 'Version', 'Floor', 'Reached', 'Rate'),
        [(*key, floor, sum(n for f, n in stats.floors.items() if f >= floor), _fmt(stats.reach(floor), '.2%'))
         for key, stats in sorted(branches.items()) for floor in range(35, max(stats.floors) + 1)]))
    lines += ['## Computational performance', '',
              'Metrics below are the most recent saved trainer measurements. Full per-update and evaluation records '
              'are available in SQLite.', '']
    if updates:
        step, meta = updates[-1]
        lines.extend(_markdown_table(('Step', 'Metric', 'Value'), [(step, key, value)
            for key, value in sorted(meta.items()) if not isinstance(value, (list, dict))]))
        lines.extend(_markdown_table(('Resource metric', 'Value'), sorted(meta.get('resources', {}).items())))
        lines.extend(_markdown_table(('Profiled phase', 'Seconds', 'Calls'),
                                    [(name, _fmt(row['seconds'], '.3f'), row['calls'])
                                     for name, row in sorted(meta.get('phases', {}).items())]))
    dqn_keys = ('dqn_loss', 'dqn_mean_q', 'dqn_mean_target', 'dqn_batch', 'dqn_terminal', 'epsilon', 'replay_size',
                'replay_added', 'replay_reused', 'replay_rejected', 'dqn_updates')
    last_values = {}
    for step, row in reversed(updates):
        for key in dqn_keys:
            if key in row:
                last_values.setdefault(key, (step, key, row[key]))
        if len(last_values) == len(dqn_keys):
            break
    if last_values:
        lines += ['## Latest DQN learner and replay diagnostics', '']
        lines.extend(_markdown_table(('Step', 'Metric', 'Value'), last_values.values()))
    replay = next(((step, row['replay']) for step, row in reversed(updates) if 'replay' in row), None)
    if replay and replay[1]['count']:
        step, summary = replay
        lines += [f'Replay composition at checkpoint {step:,}.', '']
        lines.extend(_markdown_table(('Cycle', 'Head', 'Source', 'Transitions', 'Terminals', 'Successes'),
                                    [(row['cycle'], row['head'], row['source'], row['transitions'],
                                      row['terminals'], row['successes']) for row in summary['groups']]))
    lines.extend(_markdown_table(('Step', 'Evaluation', 'Seconds'), [(s, p, _fmt(row.get('seconds'), '.2f'))
                                                                     for s, p, row in evaluation]))
    path = output / 'training_report.md'
    path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
    return path

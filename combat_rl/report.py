"""Local SQLite telemetry and reports for the Combat and Reward learner."""

from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3


TABLES = ('episodes', 'updates', 'evaluations')
SCHEMA = '''
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY, split TEXT NOT NULL, step INTEGER NOT NULL,
    game_seed TEXT NOT NULL, policy_seed TEXT NOT NULL, floor INTEGER NOT NULL,
    actions INTEGER NOT NULL, outcome TEXT NOT NULL, detail TEXT, payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS episodes_split_step ON episodes(split, step);
CREATE TABLE IF NOT EXISTS updates (
    id INTEGER PRIMARY KEY, step INTEGER NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY, step INTEGER NOT NULL, policy TEXT NOT NULL, payload TEXT NOT NULL
);
'''


def _encode(row):
    return json.dumps(row, separators=(',', ':'), allow_nan=False)


class Store:
    """Store full episode telemetry in bounded transactions and support resume rollback."""

    def __init__(self, path, batch_size=256):
        self.path = Path(path)
        self.batch_size = batch_size
        self.pending = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        try:
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.executescript(SCHEMA)
        except BaseException:
            self.db.close()
            raise
        return self

    def __exit__(self, exc_type, _exc, _trace):
        try:
            if exc_type is None:
                self.flush()
            else:
                self.db.rollback()
        finally:
            self.db.close()

    def _write(self, sql, values):
        self.db.execute(sql, values)
        self.pending += 1
        if self.pending >= self.batch_size:
            self.flush()

    def episode(self, split, step, row):
        self._write('''INSERT INTO episodes(split,step,game_seed,policy_seed,floor,actions,outcome,detail,payload)
                       VALUES (?,?,?,?,?,?,?,?,?)''',
                    (split, step, str(row['game_seed']), str(row['policy_seed']), row['floor'],
                     row['actions'], row['outcome'], row.get('detail'), _encode(row)))

    def update(self, step, row):
        self._write('INSERT INTO updates(step,payload) VALUES (?,?)', (step, _encode(row)))

    def evaluation(self, step, policy, metrics):
        self._write('INSERT INTO evaluations(step,policy,payload) VALUES (?,?,?)', (step, policy, _encode(metrics)))

    def flush(self):
        self.db.commit()
        self.pending = 0

    def marks(self):
        self.flush()
        return {table: self.db.execute(f'SELECT COALESCE(MAX(id),0) FROM {table}').fetchone()[0]
                for table in TABLES}

    def restore(self, marks):
        for table in TABLES:
            self.db.execute(f'DELETE FROM {table} WHERE id > ?', (marks[table],))
        self.flush()


def _weighted_median(histogram):
    n = sum(histogram.values())
    ranks = [(n - 1) // 2, n // 2]
    selected = []
    before = 0
    for value, count in sorted(histogram.items()):
        selected.extend(value for rank in ranks if before <= rank < before + count)
        before += count
    return sum(selected) / 2 if selected else None


def _rate(count, total):
    return count / total if total else None


def _fmt(value, fmt):
    return format(value, fmt) if value is not None else '—'


class EpisodeStats:
    """Aggregate per-floor and action distributions without retaining episode rows."""

    def __init__(self):
        self.floors = Counter()
        self.outcomes = Counter()
        self.combat = Counter()
        self.rewards = Counter()
        self.shops = Counter()
        self.events = Counter()
        self.deaths = Counter()
        self.timeouts = Counter()
        self.resources = Counter()
        self.actions = 0

    @property
    def episodes(self):
        return sum(self.floors.values())

    def add(self, row):
        self.floors[row['floor']] += 1
        self.outcomes[row['outcome']] += 1
        self.actions += row['actions']
        self.combat.update(row.get('combat_actions', {}))
        self.rewards.update(row.get('reward_actions', {}))
        for key, counter in (('shop_actions', self.shops), ('event_actions', self.events)):
            counter.update(' '.join(map(str, action['action'])) for action in row.get(key, []))
        for shop in row.get('shop_exits', []):
            self.resources['visits'] += 1
            self.resources['unused_stat_points'] += shop['unused_stat_points']
            self.resources['remaining_mana'] += shop['remaining_mana']
            self.resources['any_unused_stats'] += shop['unused_stat_points'] > 0
            self.resources['any_unused_mana'] += shop['remaining_mana'] > 0
        if row['outcome'] == 'game_over':
            self.deaths[row.get('death_monster') or 'Unrecorded'] += 1
        if row['outcome'] == 'timeout':
            reason, phase = row.get('detail') or 'Unrecorded', row.get('timeout_phase') or 'Unrecorded'
            monster = row.get('timeout_monster')
            self.timeouts[' · '.join((reason, phase, monster) if monster else (reason, phase))] += 1

    def reach(self, floor):
        return _rate(sum(n for value, n in self.floors.items() if value >= floor), self.episodes)

    def summary(self):
        n = self.episodes
        return dict(episodes=n, mean_floor=_rate(sum(floor * count for floor, count in self.floors.items()), n),
                    median_floor=_weighted_median(self.floors), max_floor=max(self.floors, default=None),
                    clear_rate=_rate(self.outcomes['cleared'], n),
                    timeout_rate=_rate(self.outcomes['timeout'], n), mean_actions=_rate(self.actions, n))


def _read_only(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def _baseline(path):
    stats = EpisodeStats()
    if path is not None and Path(path).is_file():
        with closing(_read_only(path)) as db:
            for floor, outcome, count, actions in db.execute(
                    'SELECT floor,outcome,COUNT(*),SUM(actions) FROM episodes GROUP BY floor,outcome'):
                if floor is not None:
                    stats.floors[floor] += count
                stats.outcomes[outcome] += count
                stats.actions += actions
        return stats
    # Read the published floor histogram if the local SQLite baseline is unavailable.
    original = Path(__file__).resolve().parents[1] / 'docs' / 'baseline_random_1m_results.md'
    reach_counts = {}
    for line in original.read_text(encoding='utf-8').splitlines():
        fields = [part.strip() for part in line.split('|')]
        if len(fields) == 8 and fields[1].isascii() and fields[1].isdigit() and fields[3].endswith('%'):
            reach_counts[int(fields[1])] = int(fields[2].replace(',', ''))
    for floor, n in reach_counts.items():
        if n:
            stats.floors[floor] = n - reach_counts.get(floor + 1, 0)
    stats.outcomes.update(game_over=999_913, timeout=87)
    stats.actions = 18_130_000  # Printed baseline mean (rounded) × run size.
    return stats


def _count_table(title, counts, limit=None):
    if not counts:
        return []
    n = sum(counts.values())
    return [f'### {title}', '', '| Selection or reason | Count | Share of group |', '|---|---:|---:|',
            *(f'| {label.replace("|", " / ")} | {count:,} | {count / n:.2%} |'
              for label, count in counts.most_common(limit)), '']


def _progress_rows(db):
    total = db.execute('SELECT COUNT(*) FROM updates').fetchone()[0]
    stride = max(1, (total + 3999) // 4000)
    last = db.execute('SELECT MAX(id) FROM updates').fetchone()[0]
    return [(step, json.loads(payload)) for step, payload in db.execute(
        'SELECT step,payload FROM updates WHERE id % ? = 0 OR id = ? ORDER BY id', (stride, last))]


def generate_report(db_path, output_dir, baseline_db=None):
    """Write an English report, floor and loss plots, and paired-policy comparisons."""
    import matplotlib

    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with closing(_read_only(db_path)) as db:
        training = EpisodeStats()
        panels = {}
        for split, step, payload in db.execute('SELECT split,step,payload FROM episodes ORDER BY id'):
            if split == 'train':
                stats = training
            else:
                key = (split, step)
                if key not in panels:
                    panels[key] = EpisodeStats()
                stats = panels[key]
            stats.add(json.loads(payload))
        updates = _progress_rows(db)
        eval_timing = list(db.execute('SELECT step,policy,payload FROM evaluations ORDER BY id'))
    baseline = _baseline(baseline_db)
    latest_step = max((step for split, step in panels if split == 'eval'), default=None)
    latest_random_step = max((step for split, step in panels if split == 'random'), default=None)
    plot_cohorts = {'Training': training}
    for label, key in (('Fixed-seed evaluation', ('eval', latest_step)),
                       ('Paired uniform random', ('random', latest_random_step))):
        if key in panels:
            plot_cohorts[label] = panels[key]
    plot_cohorts['Historical uniform random'] = baseline

    upper = max(5, max((max(stats.floors, default=0) for stats in plot_cohorts.values()), default=0))
    floors = list(range(1, upper + 1))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3))
    for label, stats in plot_cohorts.items():
        n = stats.episodes
        if n:
            left.step(floors, [stats.reach(floor) for floor in floors], where='post', label=label)
            right.plot(floors, [stats.floors[floor] / n for floor in floors], marker='.', label=label)
    left.set(title='Floor survival', xlabel='Floor reached', ylabel='Share reaching at least this floor')
    right.set(title='Final floor distribution', xlabel='Furthest floor', ylabel='Share of episodes')
    for axis in (left, right):
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'floor_reach.png', dpi=150)
    plt.close(fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3))
    for split, label in (('eval', 'Fixed-seed evaluation'), ('random', 'Paired uniform random')):
        series = sorted((step, stats.summary()['mean_floor']) for (kind, step), stats in panels.items() if kind == split)
        if series:
            left.plot(*zip(*series), marker='o', label=label)
    for name in ('policy_loss', 'value_loss', 'entropy'):
        series = [(step, data[name]) for step, data in updates if data.get(name) is not None]
        if series:
            right.plot(*zip(*series), label=name.replace('_', ' '))
    left.set(xlabel='Completed training episodes', ylabel='Mean furthest floor', title='Checkpoint evaluations')
    right.set(xlabel='Completed training episodes', ylabel='Value', title='PPO update metrics')
    for axis in (left, right):
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.png', dpi=150)
    plt.close(fig)

    lines = ['# Combat and Reward reinforcement learning results', '',
             'The trained policy selects combat skills and victory rewards. Shop and Event decisions use uniform random '
             'sampling, so unallocated resources and depleted loadouts can remain a performance bottleneck. '
             'Training samples evolve with the policy, while checkpoint evaluation uses fixed separate seed panels.', '',
             'The historical 1,000,000-episode uniform random baseline is a descriptive benchmark. '
             'Paired uniform-random evaluation uses the same game seeds and action cap as the learned policy '
             'for its panel. Reaching a floor means reaching it at any point in an episode. '
             '[Original baseline report](https://github.com/KyleIm/Clear-The-Dungeon-Project/blob/main/docs/baseline_random_1m_results.md)', '',
             '## Outcomes and floor reach', '',
             '| Cohort | Episodes | Mean floor | Median floor | Max floor | Clear rate | Timeout rate | Mean actions |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels = {'eval': 'Fixed-seed evaluation', 'random': 'Paired uniform random'}
    cohorts = [('Training', training), ('Historical uniform random', baseline)]
    cohorts.extend((f'{labels.get(split, split)} at {step:,}', stats) for (split, step), stats in sorted(panels.items()))
    for label, stats in cohorts:
        summary = stats.summary()
        cells = [label, f'{stats.episodes:,}', _fmt(summary['mean_floor'], '.2f'),
                 _fmt(summary['median_floor'], 'g'), _fmt(summary['max_floor'], 'd'),
                 _fmt(summary['clear_rate'], '.4%'), _fmt(summary['timeout_rate'], '.4%'),
                 _fmt(summary['mean_actions'], '.2f')]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '![Floor reach and distribution](floor_reach.png)', '',
              '![Learning curves and losses](learning_curves.png)', '',
              '### Latest floor reach and distribution', '',
              'Each cell shows final episode count at that floor and the share reaching at least that floor.', '',
              '| Floor | ' + ' | '.join(plot_cohorts) + ' |',
              '|---:|' + '---:|' * len(plot_cohorts)]
    for floor in floors:
        lines.append(f'| {floor} | ' + ' | '.join(f'{stats.floors[floor]:,} · {_fmt(stats.reach(floor), ".4%")}'
                                                     for stats in plot_cohorts.values()) + ' |')
    lines.append('')
    if eval_timing:
        lines += ['### Evaluation metadata', '']
        for step, policy, raw in eval_timing:
            details = json.loads(raw)
            seconds = details.get('elapsed', details.get('seconds'))
            if seconds is not None:
                lines.append(f'- {policy} at {step:,} took {seconds:.2f} seconds.')
        lines.append('')
    for label, stats in plot_cohorts.items():
        if label == 'Historical uniform random' or not stats.episodes:
            continue
        n = stats.resources['visits']
        lines += [f'## {label} decisions', '',
                  f'Recorded shop exits: **{n:,}**. Mean unused stat points '
                  f'{_fmt(_rate(stats.resources["unused_stat_points"], n), ".2f")} and mean remaining mana '
                  f'{_fmt(_rate(stats.resources["remaining_mana"], n), ".2f")}. The shares of exits with any unused '
                  f'stat points or mana were {_fmt(_rate(stats.resources["any_unused_stats"], n), ".2%")} and '
                  f'{_fmt(_rate(stats.resources["any_unused_mana"], n), ".2%")} respectively.', '']
        for title, counts, limit in (('Combat actions', stats.combat, None), ('Reward actions', stats.rewards, None),
                                     ('Most frequent shop actions', stats.shops, 15), ('Event actions', stats.events, None),
                                     ('Most frequent defeat opponents', stats.deaths, 15), ('Timeout reasons', stats.timeouts, None)):
            lines.extend(_count_table(title, counts, limit))
    path = output / 'training_report.md'
    path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
    return path

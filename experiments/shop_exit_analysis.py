"""Summarize shop exit telemetry without pooling different shop floors."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, median

from experiments.shop_exit_replay import STATS


FEATURES = (*STATS, 'unused_stat_points', 'max_mana', 'used_mana', 'remaining_mana',
            'zero_use_skills', 'entry_gold', 'exit_gold', 'gold_spent')
NAMES = dict(attack='ATK', defense='DEF', hp='HP allocation', speed='SPD',
             unused_stat_points='Unused stat points', max_mana='Maximum mana', used_mana='Used mana',
             remaining_mana='Remaining mana', zero_use_skills='Zero-use skills', entry_gold='Entry gold',
             exit_gold='Exit gold', gold_spent='Gold spent')


def band(value, cuts):
    previous = None
    for cut in cuts:
        if value <= cut:
            if previous is None:
                return f'≤{cut}'
            return str(cut) if previous + 1 == cut else f'{previous + 1}–{cut}'
        previous = cut
    return f'>{cuts[-1]}'


def reached(rows, threshold):
    return sum(row['final_floor'] >= threshold for row in rows) / len(rows)


def metrics(label, rows, floor):
    rates = [reached(rows, floor + offset) for offset in (2, 5, 10)]
    cells = [label, f'{len(rows):,}', f'{mean(row["final_floor"] for row in rows):.2f}',
             *(f'{rate:.2%}' for rate in rates)]
    return '| ' + ' | '.join(cells) + ' |'


def report(csv_path, output):
    import matplotlib

    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    groups = defaultdict(list)
    with csv_path.open(newline='', encoding='utf-8') as file:
        for values in csv.DictReader(file):
            row = {field: int(value) for field, value in values.items() if field not in {'slots', 'outcome'}}
            row['slots'] = json.loads(values['slots'])
            row['gold_spent'] = row['entry_gold'] - row['exit_gold']
            groups[row['shop_floor']].append(row)
    output.mkdir(parents=True, exist_ok=True)
    total_games = len({row['game_id'] for rows in groups.values() for row in rows})
    exits = sum(map(len, groups.values()))
    lines = ['# Random baseline shop exit analysis', '',
             f'Replayed all **{total_games:,}** episodes whose historical furthest floor was at least 11. '
             'The original game and policy seeds, action cap and wall limit were used. Every replay matched its recorded '
             f'furthest floor, action count, outcome and termination detail. We captured **{exits:,}** shop exits. '
             'The source SQLite was opened read-only.', '',
             'The observer reads gold on entry and snapshots the player immediately before leaving. HP here is the base HP '
             'stat allocation. Used mana is the sum of allotted uses × current skill cost across the equipped slots. '
             'Zero-use skills count equipped skills with no allotted uses. Locked slots are excluded. The local CSV includes '
             'the ordered loadout with each skill’s use allocation and current cost, the two seeds and the final floor.', '',
             'Survival means reaching a later floor. Reaching the first floor after a shop is guaranteed by inclusion. '
             'All rates condition on the exit of the named shop. Later shops contain a selected subset of survivors. '
             'Univariate groups and skill cohorts can differ in other build decisions and encounter opportunities. '
             'These are descriptive associations. Use the group counts when interpreting sparse outcomes.', '',
             '![Survival by shop floor](shop_exit_survival.png)', '']
    if 10 in groups:
        first_shop = groups[10]
        lines += ['## Main findings', '']
        comparisons = (
            ('No unused stat points', lambda row: row['unused_stat_points'] == 0,
             'More than 100 unused stat points', lambda row: row['unused_stat_points'] > 100),
            ('No remaining mana', lambda row: row['remaining_mana'] == 0,
             'More than 40 remaining mana', lambda row: row['remaining_mana'] > 40),
            ('No zero-use skill', lambda row: row['zero_use_skills'] == 0,
             'Three or more zero-use skills', lambda row: row['zero_use_skills'] >= 3),
        )
        for left, left_filter, right, right_filter in comparisons:
            a, b = [r for r in first_shop if left_filter(r)], [r for r in first_shop if right_filter(r)]
            if a and b:
                lines.append(f'- **{left}** (n={len(a):,}) reached floor 12 at {reached(a, 12):.2%}. '
                             f'**{right}** (n={len(b):,}) reached it at {reached(b, 12):.2%}.')
        lines.append('')
    figure, axes = plt.subplots(1, len(groups), figsize=(6.1 * len(groups), 4), squeeze=False)
    for ax, (floor, rows) in zip(axes[0], sorted(groups.items())):
        highest = max(row['final_floor'] for row in rows)
        floors = list(range(floor + 1, highest + 2))
        ax.step(floors, [reached(rows, target) for target in floors], where='post')
        ax.set(title=f'Floor {floor} shop · {len(rows):,} exits', xlabel='Floor reached', ylabel='Share of exits')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(output / 'shop_exit_survival.png', dpi=150)
    plt.close(figure)

    cuts = {**{key: (99, 100) for key in STATS}, 'unused_stat_points': (0, 25, 100),
            'max_mana': (170, 180, 190), 'used_mana': (120, 150, 170), 'remaining_mana': (0, 10, 40),
            'zero_use_skills': (0, 1, 2), 'entry_gold': (2000, 5000, 10000),
            'exit_gold': (1000, 3000, 6000), 'gold_spent': (0, 1000, 3000)}
    use_cohorts = []
    for floor, rows in sorted(groups.items()):
        header = f'| Group | Exits | Mean final floor | Reach {floor+2} | Reach {floor+5} | Reach {floor+10} |'
        rule = '|---|---:|---:|---:|---:|---:|'
        lines += [f'## Floor {floor} shop', '', header, rule, metrics('All exits', rows, floor), '',
                  '| Measure | Minimum | Median | Maximum |', '|---|---:|---:|---:|']
        for key in FEATURES:
            values = [row[key] for row in rows]
            lines.append(f'| {NAMES[key]} | {min(values):,} | {median(values):,} | {max(values):,} |')
        lines += ['', '### Allocation, mana and gold cohorts', '', header, rule]
        panels = {}
        for key in FEATURES:
            by_band = defaultdict(list)
            for row in sorted(rows, key=lambda row: row[key]):
                by_band[band(row[key], cuts[key])].append(row)
            panels[key] = by_band
            lines.extend(metrics(f'{NAMES[key]} {label}', cohort, floor) for label, cohort in by_band.items())
        figure, axes = plt.subplots(3, 4, figsize=(13, 8))
        for ax, key in zip(axes.flat, FEATURES):
            by_band = panels[key]
            ax.bar(range(len(by_band)), [reached(group, floor + 2) for group in by_band.values()], color='#397c92')
            ax.set_xticks(range(len(by_band)), [f'{label}\n(n={len(group):,})' for label, group in by_band.items()],
                          fontsize=7)
            ax.set(title=NAMES[key], ylim=(0, 1))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        for ax in list(axes.flat)[len(FEATURES):]:
            ax.set_visible(False)
        figure.suptitle(f'Share reaching floor {floor + 2} after the floor {floor} shop')
        figure.tight_layout()
        chart = f'shop_{floor}_cohorts.png'
        figure.savefig(output / chart, dpi=150)
        plt.close(figure)
        lines += ['', f'![Floor {floor} shop cohorts]({chart})', '', '### Loadouts and use allocation', '']
        combos = defaultdict(list)
        skills = defaultdict(list)
        for row in rows:
            combos[tuple(name for name, _, _ in row['slots'])].append(row)
            for name, uses, _ in row['slots']:
                skills[name].append((row, uses))
        for name, allocations in sorted(skills.items()):
            by_uses = defaultdict(list)
            for row, uses in allocations:
                by_uses[uses].append(row)
            for uses, cohort in sorted(by_uses.items()):
                use_cohorts.append([floor, name, uses, len(cohort), mean(row['final_floor'] for row in cohort),
                                    *(reached(cohort, floor + n) for n in (2, 5, 10))])
        lines += [f'There were {len(combos):,} distinct ordered skill combinations. The table shows the ten most '
                  'frequent combinations, ordered by count, then by skill names.', '', header, rule]
        for names, cohort in sorted(combos.items(), key=lambda item: (-len(item[1]), item[0]))[:10]:
            lines.append(metrics(', '.join(names), cohort, floor))
        lines += ['', f'For the 15 most frequently equipped skills, the table compares reach to floor {floor+2} '
                  'between zero and positive allotted uses.', '',
                  '| Skill | Equipped | Median uses | Zero uses | Reach with zero uses | Reach with positive uses |',
                  '|---|---:|---:|---:|---:|---:|']
        for name, allocations in sorted(skills.items(), key=lambda item: (-len(item[1]), item[0]))[:15]:
            zero = [row for row, uses in allocations if uses == 0]
            positive = [row for row, uses in allocations if uses > 0]
            zero_rate = f'{reached(zero, floor + 2):.2%}' if zero else '—'
            positive_rate = f'{reached(positive, floor + 2):.2%}' if positive else '—'
            lines.append(f'| {name} | {len(allocations):,} | {median(uses for _, uses in allocations):g} | '
                         f'{len(zero):,} | {zero_rate} | {positive_rate} |')
        lines.append('')
    with (output / 'skill_use_cohorts.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['shop_floor', 'skill', 'allocated_uses', 'exits', 'mean_final_floor',
                         'reach_shop_plus_2', 'reach_shop_plus_5', 'reach_shop_plus_10'])
        writer.writerows(use_cohorts)
    lines += ['[All per-skill and exact-use cohorts](skill_use_cohorts.csv)', '',
              '## Reproduce locally', '', '```console',
              'python -m experiments.shop_exit_replay data/baseline_random_1m/baseline.sqlite3 data/shop_exit_replay/shops.csv',
              'python -m experiments.shop_exit_analysis data/shop_exit_replay/shops.csv docs/shop_exit_replay',
              '```', '', '[Original baseline report](../baseline_random_1m_results.md)', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report(args.csv, args.output)


if __name__ == '__main__':
    main()

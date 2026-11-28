"""Stream local metrics into English reports without exporting raw episodes."""

from collections import Counter, defaultdict
from contextlib import closing
import csv
import json
from pathlib import Path
import sqlite3

from combat_rl.environment import REWARD_CHOICES
from combat_rl.report import EpisodeStats
from .configurations import HEADS


def read_only(path):
    return closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True))


def csv_table(directory, filename, columns, rows):
    with (directory / filename).open('w', newline='', encoding='utf-8') as file:
        out = csv.writer(file)
        out.writerow(columns.split(','))
        out.writerows(rows)


def allocations(counter, row):
    values = [('unused_stats', row['unused_stat_points']), ('unused_mana', row['remaining_mana'])]
    stats = row['base_stats'] if 'base_stats' in row else row
    values.extend((stat, stats[stat]) for stat in ('attack', 'defense', 'hp', 'speed'))
    values.extend((('gold_retained', row['gold']), ('max_mana', row['max_mana'])))
    if 'entry_gold' in row:
        values.append(('gold_spent', row['entry_gold'] - row['gold']))
    slots = [(slot['skill'], slot['uses']) for slot in row['slots'] if slot['skill']]
    values.append(('loadout', ' / '.join(skill for skill, _ in slots)))
    values.extend(('memorize:' + skill, uses) for skill, uses in slots)
    counter.update(values)


def report(database, directory):
    groups = defaultdict(EpisodeStats)
    configurations = defaultdict(Counter)
    contexts = defaultdict(Counter)
    probes = defaultdict(Counter)
    work = defaultdict(Counter)
    panel = defaultdict(dict)
    labels = defaultdict(Counter)
    segment_updates = {}
    losses = []
    with read_only(database) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='comparisons'").fetchone():
            raise ValueError('Select a baseline-relative Shop run directory')
        for split, step, payload in db.execute('SELECT split,step,payload FROM episodes ORDER BY id'):
            row = json.loads(payload)
            cohort = split, row['segment'] if split == 'train' else step
            groups[cohort].add(row)
            work[cohort].update(row['shop_work'])
            if split != 'train':
                panel[cohort][row['game_seed'], row['policy_seed']] = row['floor']
            for context in row['reward_context']:
                k = context.pop('axis'), context.pop('bucket'), context.pop('source')
                contexts[(*cohort, *k)].update(context)
            for exit_state in row['shop_exits']:
                key =(*cohort, exit_state['floor'])
                allocations(configurations[key], exit_state)
                for probe in exit_state.get('allocation_probes', []):
                    tag = (*key, probe['resource'], ' '.join(map(str, probe['action'])),
                           probe['before'], probe['after'])
                    probes[tag].update(n=1, value=probe['score'])
        for step, version, head, label, has_features, payload in db.execute(
                'SELECT step,version,head,label,features IS NOT NULL,payload FROM comparisons'):
            row = json.loads(payload)
            key = (row['segment'], version, HEADS[head], row['shop_floor'])
            counts = labels[key]
            counts['censored' if label is None else ('loss','tie','win')[label + 1]] += 1
            counts['noop'] += row['noop']
            counts['fit_rows'] += has_features
            if label is not None:
                counts['sum_delta'] += row['delta']
                counts['resolved'] += 1
            candidate = row['configuration']
            candidate['entry_gold'] = row['entry_gold']
            allocations(configurations[('paired_' + HEADS[head], row['segment'], row['shop_floor'])], candidate)
        for step, payload in db.execute('SELECT step,payload FROM updates ORDER BY id'):
            row = json.loads(payload)
            segment_updates[row['segment']] = row
            losses.extend((step, row['version'], key, value) for key,value in row.items()
                          if key.startswith('loss_') or key in ('policy_loss', 'value_loss'))
        eval_times = [(step, policy, json.loads(payload)) for step, policy, payload in
                      db.execute('SELECT step,policy,payload FROM evaluations ORDER BY id')]

    if not groups:
        raise ValueError('The experiment has no committed episode telemetry yet')
    directory.mkdir(parents=True, exist_ok=True)
    refs = {}
    reference_path = Path(__file__).resolve().parents[1] / 'data/combat_reward_rl_optimized/training.sqlite3'
    if reference_path.exists():
        with read_only(reference_path) as db:
            ref = db.execute("SELECT game_seed,policy_seed,floor FROM episodes WHERE split='eval' AND step=1000000")
            refs = {(int(g), int(p)): f for g,p,f in ref}
    scores = []
    for (split, step), data in sorted(panel.items()):
        for kind, baseline in (('no_edit', panel.get(('eval_baseline', step), {})), ('phase_3b_2', refs)):
            shared = data.keys() & baseline.keys()
            if shared:
                deltas = [data[k] - baseline[k] for k in shared]
                scores.append((split, step, kind, len(deltas), sum(x > 0 for x in deltas),
                               sum(x == 0 for x in deltas), sum(x < 0 for x in deltas), sum(deltas)/len(deltas)))
    csv_table(directory, 'paired_panel.csv', 'policy,step,reference,seed_pairs,wins,ties,losses,mean_floor_difference', scores)
    csv_table(directory, 'shop_comparisons.csv', 'segment,ppo_version,head,shop_floor,wins,ties,losses,censored,noop,fit_rows,mean_resolved_floor_difference',
              ((*k, *(c[x] for x in ('win','tie','loss','censored','noop','fit_rows')),
                c['sum_delta'] / c['resolved'] if c['resolved'] else None) for k,c in sorted(labels.items())))
    csv_table(directory, 'reward_context.csv', 'split,segment,axis,bucket,source,reward,choices,selections,selection_rate,mean_ppo_probability',
              ((*k, r, c['choices'], c['chose:'+r], c['chose:'+r] / c['choices'],
                c['ppo_prob:'+r] / c['choices']) for k,c in sorted(contexts.items()) for r in REWARD_CHOICES))
    csv_table(directory, 'shop_allocations.csv', 'split,segment,shop_floor,metric,value,count',
              ((*k, metric, value, n) for k,c in sorted(configurations.items()) for (metric,value),n in c.items()))
    csv_table(directory, 'allocation_probes.csv', 'split,segment,shop_floor,resource,edit,before,after,count,mean_score_change',
              ((*k,c['n'],c['value']/c['n']) for k,c in sorted(probes.items())))
    csv_table(directory, 'loss.csv', 'episodes,ppo_version,metric,loss', losses)
    csv_table(directory, 'floor_reach.csv', 'split,step,floor,reached,episodes,rate',
              ((*k, floor, sum(n for reached,n in c.floors.items() if reached>=floor), c.episodes, c.reach(floor))
               for k,c in sorted(groups.items()) for floor in range(1,101)))
    csv_table(directory, 'shop_compute.csv', 'split,segment,metric,value',
              ((*k,metric,value) for k,c in sorted(work.items()) for metric,value in sorted(c.items())))
    csv_table(directory, 'evaluation_timing.csv', 'step,policy,episodes,actions,seconds',
              ((step,mode,*(c[k] for k in ('episodes','actions','seconds'))) for step,mode,c in eval_times))
    csv_table(directory, 'resource_snapshots.csv', 'segment,metric,value',
              ((k,m,v) for k,c in sorted(segment_updates.items()) for m,v in sorted(c['resources'].items())))
    csv_table(directory, 'timings.csv', 'segment,metric,wall_seconds,calls',
              ((k,m,p['seconds'] - prior.get(m,{}).get('seconds',0),p['calls'] - prior.get(m,{}).get('calls',0))
               for (k,c),prior in zip(sorted(segment_updates.items()), [{}, *(c['profile'] for _,c in sorted(segment_updates.items()))])
               for m,p in sorted(c['profile'].items())))
    prior_seconds = 0
    perf = []
    for step, update in sorted(segment_updates.items()):
        elapsed = update['train_seconds'] - prior_seconds
        prior_seconds = update['train_seconds']
        cohort = groups['train', step]
        perf.append((step, update['mode'], elapsed, cohort.episodes/elapsed, cohort.actions/elapsed))
    csv_table(directory, 'throughput.csv', 'segment,mode,training_seconds,episodes_per_second,actions_per_second', perf)

    lines = ['# Baseline-relative Shop learning', '',
             'Training segments and fixed evaluation panels are shown separately. Historical comparisons pair the exact game and policy seeds when the Phase 3B-2 SQLite file is locally available.', '',
             '| Split | Episodes at boundary | Games | Mean floor | Median | Max | Clears | Clear rate | Timeouts |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (split, step), c in sorted(groups.items()):
        summary = c.summary()
        lines.append(f"| {split} | {step:,} | {c.episodes:,} | {summary['mean_floor']:.3f} | "
                     f"{summary['median_floor']:g} | {max(c.floors)} | {c.outcomes['cleared']} | "
                     f"{c.outcomes['cleared']/c.episodes:.3%} | {c.outcomes['timeout']} |")
    lines.extend(['', '## Paired fixed-panel comparisons', '',
                  '| Policy | Boundary | Reference | Pairs | Wins | Ties | Losses | Mean floor change |',
                  '|---|---:|---|---:|---:|---:|---:|---:|'])
    for split, step, kind, n, win, tie, loss, delta in scores:
        lines.append(f'| {split} | {step:,} | {kind} | {n} | {win/n:.1%} | {tie/n:.1%} | {loss/n:.1%} | {delta:+.3f} |')
    lines.extend(['', '## Independent random configurations used for Shop learning', '',
                  '| Segment | Head | Resolved | Win | Tie | Loss | Censored | Eligible | Mean floor change |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|'])
    totals = defaultdict(Counter)
    for (step, _, head, _), c in labels.items():
        totals[step, head].update(c)
    for (step, head), c in sorted(totals.items()):
        n = c['resolved']
        rates = [f'{c[k]/n:.1%}' if n else '—' for k in ('win','tie','loss')]
        delta = f"{c['sum_delta']/n:+.3f}" if n else '—'
        lines.append(f"| {step:,} | {head} | {n} | {' | '.join(rates)} | {c['censored']} | {c['fit_rows']} | {delta} |")
    lines.extend(['', '- [Floor reach](floor_reach.csv) and [paired evaluation results](paired_panel.csv)',
                  '- [Shop comparisons by version and floor](shop_comparisons.csv) and [configuration distributions](shop_allocations.csv)',
                  '- [Reward selections and model probabilities by context](reward_context.csv)',
                  '- [One-edit allocation probes](allocation_probes.csv) and [learner losses](loss.csv)',
                  '- [Throughput](throughput.csv), [worker and learner timings](timings.csv), [branch costs](shop_compute.csv), [evaluation timing](evaluation_timing.csv), and [resource snapshots](resource_snapshots.csv)',
                  '', '![Fixed-panel depth and learning losses](progress.png)', '',
                  'The allocation probes hold previous edits fixed. Each probe adds one legal point or one legal Memorize, so its resource total and associated allocation move together. Its score is a model output, with the unchanged configuration anchored at zero. The Shop loss uses class-balanced sampling and its scores are uncalibrated. Censored pairs and unmodified configurations are excluded from fitting.'])
    summary = directory / 'report.md'
    summary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    plots(groups, losses, directory)
    return summary


def plots(groups, losses, directory):
    from matplotlib import use
    use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for split in sorted({k[0] for k in groups}):
        points = sorted((step,c.summary()['mean_floor']) for (s,step),c in groups.items() if s==split)
        axes[0].plot(*zip(*points), marker='.', label=split)
    for name in sorted({row[2] for row in losses}):
        points = [(step,value) for step,_,key,value in losses if key==name]
        axes[1].plot(*zip(*points), label=name)
    for ax,label in zip(axes,('Mean furthest floor','Loss')):
        ax.set(xlabel='Completed episodes',ylabel=label)
        if ax.lines:
            ax.legend(fontsize=7)
        ax.grid(alpha=.25)
    fig.savefig(directory/'progress.png',dpi=140)
    plt.close(fig)

"""V2 boundary, experience-transfer and resource reports."""

from collections import Counter, defaultdict
import csv
import json

from baseline_shop.report import csv_table, read_only, report as common_report
from combat_rl.environment import REWARD_CHOICES
from combat_rl.report import _weighted_median


def pct(value):
    return f'{value:.2%}'


def report(directory):
    database = directory/'training.sqlite3'
    output = directory/'report'
    with read_only(database) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='boundaries'").fetchone():
            raise ValueError('Choose a V2 experiment directory')
    summary = common_report(database,output)
    allocations = defaultdict(Counter)
    choices = defaultdict(Counter)
    probes = defaultdict(Counter)
    timeouts = Counter()
    results = defaultdict(Counter)
    purchases = Counter()
    branch_work = defaultdict(Counter)

    with read_only(database) as db, (output/'timeout_details.jsonl').open('w',encoding='utf-8') as details:
        boundaries = [(step,json.loads(p)) for step,p in db.execute('SELECT step,payload FROM boundaries ORDER BY step')]
        experience = list(db.execute('SELECT segment,shop_version,source,action,censored,count(*) '
                                     'FROM reward_experience GROUP BY segment,shop_version,source,action,censored'))
        for split, step, payload in db.execute('SELECT split,step,payload FROM episodes'):
            row = json.loads(payload)
            result = results[split,step]
            result.update(games=1, floors=row['floor'], clears=row['outcome']=='cleared')
            if row.get('timeout'):
                item = row['timeout']
                timeouts[split,step,item['floor'],item['monster'],item['branch'],item['signature']] += 1
                details.write(json.dumps(dict(item,split=split,step=step))+'\n')
            segment = row['segment']
            branch_work[split,segment].update(row['shop_work'])
            for item in row['shop_actions']:
                action = item['action']
                if action[0] in ('buy','reduce_skill_cost','change_type','reroll'):
                    purchases[split,segment,item['floor'],' '.join(action)] += 1
            if split=='train':
                continue
            for visit in row['shop_exits']:
                for metric in ('unused_stat_points','remaining_mana'):
                    allocations[split,step,visit['floor'],metric][visit[metric]] += 1
                selections = visit.get('configuration_choices',{})
                for head, decision in selections.items():
                    choices[split,step,visit['floor'],head].update(n=1,noop=decision['edits']==0)
                choices[split,step,visit['floor'],'combined'].update(n=1,noop=not any(d['edits'] for d in selections.values()))
            for probe in row.get('strategy_checks',[]):
                key = (step,probe['resource'],' '.join(map(str,probe['action'])),probe['before'],probe['after'])
                counts = probes[key]
                counts.update(total=1,censored=probe['censored'])
                if probe['score'] is not None:
                    counts.update(scored=1,score=probe['score'])
                if not probe['censored']:
                    delta = probe['floor']-probe['baseline_floor']
                    counts.update(resolved=1,delta=delta,wins=delta>0,ties=delta==0,losses=delta<0)
                for item in probe['timeouts']:
                    timeouts['probe',step,item['floor'],item['monster'],item['branch'],item['signature']] += 1
                    details.write(json.dumps(dict(item,split='probe',step=step))+'\n')
        for payload, in db.execute('SELECT payload FROM comparisons'):
            row = json.loads(payload)
            for item in row['timeouts']:
                timeouts['counterfactual',row['segment'],item['floor'],item['monster'],item['branch'],item['signature']] += 1
                details.write(json.dumps(dict(item,split='counterfactual',step=row['segment']))+'\n')
        size = db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]

    csv_table(output,'checkpoint_unused_resources.csv','policy,checkpoint,shop_floor,resource,unused,count',
              ((*k,value,n) for k,c in sorted(allocations.items()) for value,n in sorted(c.items())))
    csv_table(output,'shop_no_edit.csv','policy,checkpoint,shop_floor,head,selections,no_edit',
              ((*k,c['n'],c['noop']) for k,c in sorted(choices.items())))
    csv_table(output,'shop_purchases.csv','split,checkpoint_or_segment,shop_floor,action,count',
              ((*k,n) for k,n in sorted(purchases.items())))
    csv_table(output,'counterfactual_throughput.csv','split,segment,comparison_actions,worker_seconds,actions_per_worker_second',
              ((*k,c['paired_actions'],c['paired_seconds'],c['paired_actions']/c['paired_seconds'])
               for k,c in sorted(branch_work.items()) if c['paired_seconds']))
    csv_table(output,'reward_transfer.csv','checkpoint,reward,forced_fit_records,mean_probability_change',
              ((step,name,t['forced_training_counts'][name],change) for step,data in boundaries
               if (t:=data.get('transfer')) and 'probability_changes' in t
               for name,change in zip(REWARD_CHOICES,t['probability_changes'])))
    csv_table(output,'strategy_verification.csv',
              'checkpoint,resource,edit,before,after,observations,censored,mean_prediction,resolved,wins,ties,losses,mean_floor_delta',
              ((*k,c['total'],c['censored'],c['score']/c['scored'] if c['scored'] else None,c['resolved'],
                c['wins'],c['ties'],c['losses'],c['delta']/c['resolved'] if c['resolved'] else None)
               for k,c in sorted(probes.items())))
    csv_table(output,'timeout_counts.csv','split,checkpoint_or_ordinal,floor,monster,branch,signature,count',
              ((*k,n) for k,n in sorted(timeouts.items())))
    csv_table(output,'reward_experience_counts.csv','segment,shop_version,source,reward,censored,records',
              ((s,v,o,REWARD_CHOICES[a],c,n) for s,v,o,a,c,n in experience))
    csv_table(output,'reward_q_distribution.csv','checkpoint,reward,states,p10,p50,p90',
              ((step,name,row['n'],*row['quantiles']) for step,data in boundaries
               for name,row in data.get('transfer',{}).get('q_distribution',{}).items()))
    (output/'boundary_decisions.json').write_text(json.dumps(dict(boundaries),indent=2)+'\n',encoding='utf-8')

    lines = ['','## V2 boundary decisions','',
             '| Episodes | Previous → retained Shop version | Promoted | Development / validation lower bounds |'
             ' Reward rows (fit / guide) | Forced Gold examples | Mean Gold probability change | Q / auxiliary seconds |',
             '|---:|---|---|---|---:|---:|---:|---|']
    for step, data in boundaries:
        p, t = data.get('promotion'),data.get('transfer',{})
        lower = f"{p['development']['lower']:+.3f} / {p['heldout']['lower']:+.3f}" if p else '—'
        lines.append(f"| {step:,} | {data['baseline_before']} → {data['baseline_after']} | "
                     f"{p['promoted'] if p else '—'} | {lower} | {t.get('training_rows',0):,} / {t.get('heldout_rows',0):,} | "
                     f"{t.get('forced_training_counts',{}).get('Gold',0):,} | "
                     f"{pct(t['gold_probability_change']) if 'gold_probability_change' in t else '—'} | "
                     f"{t.get('q_seconds',0):.3f} / {t.get('aux_seconds',0):.3f} |")
    lines.extend(['','## Unused resources by evaluation checkpoint','',
                  '| Checkpoint | Policy | Shop exits | Unused stat mean / median | Unused mana mean / median |',
                  '|---:|---|---:|---:|---:|'])
    resource = defaultdict(Counter)
    for (split,step,_,metric), hs in allocations.items():
        if split in ('eval_baseline','eval_adaptive','eval_candidate','eval_accepted_post'):
            resource[step,split,metric].update(hs)
    for step, split in sorted({(s,p) for s,p,_ in resource}):
        values = []
        for metric in ('unused_stat_points','remaining_mana'):
            c = resource[step,split,metric]
            n = c.total()
            values.append(f'{sum(value*count for value,count in c.items())/n:.2f} / {_weighted_median(c):g}' if n else '—')
        lines.append(f"| {step:,} | {split} | {n:,} | {' | '.join(values)} |")
    lines.extend(['','## Reward policies before and after guidance','',
                  '| Episodes | Policy | Gold choice rate | Mean Gold PPO probability | Potion choice rate at 90–100% HP |',
                  '|---:|---|---:|---:|---:|'])
    with (output/'reward_context.csv').open(newline='',encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    hp = {(r['segment'],r['split']):float(r['selection_rate']) for r in rows
          if r['axis']=='hp' and r['bucket']=='90–100%' and r['reward']=='Potion'}
    for r in rows:
        split = r['split']
        if split in ('eval_accepted_pre','eval_accepted_post') and r['axis']=='overall' and r['reward']=='Gold':
            key = (r['segment'],split)
            lines.append(f"| {int(r['segment']):,} | {split} | {pct(float(r['selection_rate']))} | "
                         f"{pct(float(r['mean_ppo_probability']))} | {pct(hp[key]) if key in hp else '—'} |")
    final = [(step,c) for (split,step),c in results.items()
             if (split,step) in (('eval_accepted_post',1000000),('eval_adaptive',1025000))]
    if len(final)==2:
        lines.extend(['','## Final PPO adaptation',''])
        for step,c in sorted(final):
            lines.append(f"- {step:,} episodes · mean floor {c['floors']/c['games']:.3f} · {c['clears']}/{c['games']} clears")
    lines.extend(['',f'SQLite allocated pages: {size/2**20:,.1f} MiB. Raw observations, checkpoints and timeout snapshots are local.',
                 '', '- [Boundary and promotion details](boundary_decisions.json)',
                 '- [Unused resource distributions](checkpoint_unused_resources.csv), [no-edit rates](shop_no_edit.csv) and [purchases](shop_purchases.csv)',
                 '- [Reward source counts](reward_experience_counts.csv), [Reward Q distribution](reward_q_distribution.csv) and [Reward transfer](reward_transfer.csv)',
                 '- [Counterfactual action throughput](counterfactual_throughput.csv)',
                 '- [Resource probes with actual continuations](strategy_verification.csv) and [timeout counts](timeout_counts.csv)',
                 '', 'The same baseline-branch timeout can appear in multiple treatment comparisons. A Mimic-compatible record is a signature of the recorded suffix. The suffix alone cannot establish the full internal game state or a mathematical deadlock.'])
    contents = summary.read_text(encoding='utf-8')
    contents = contents.replace('Baseline-relative Shop learning','Adaptive Shop V2 results',1)
    contents = contents.replace('Independent random configurations used for Shop learning','Single-head treatments against the incumbent')
    summary.write_text(contents+'\n'.join(lines)+'\n',encoding='utf-8')
    return summary

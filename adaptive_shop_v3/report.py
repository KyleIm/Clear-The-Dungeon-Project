"""Publish the V2 metrics plus V3 exploration, selection and long-run diagnostics."""

from collections import Counter, defaultdict
import json

from adaptive_shop.report import report as v2_report
from baseline_shop.report import csv_table, read_only
from combat_rl.environment import REWARD_CHOICES
from combat_rl.report import EpisodeStats, _baseline


def report(directory):
    dbpath = directory/'training.sqlite3'
    output = directory/'report'
    with read_only(dbpath) as db:
        boundaries = [(step,json.loads(payload)) for step,payload in
                      db.execute('SELECT step,payload FROM boundaries ORDER BY step')]
        if not boundaries or 'disk' not in boundaries[0][1]:
            raise ValueError('Select an experiment started with V3')
    summary = v2_report(directory)
    exploration = defaultdict(Counter)
    selections = defaultdict(Counter)
    combat = defaultdict(Counter)
    final = defaultdict(EpisodeStats)
    train_seconds = wall_seconds = 0.
    with read_only(dbpath) as db:
        for step,head,label,fitted,payload in db.execute(
                'SELECT step,head,label,features IS NOT NULL,payload FROM comparisons'):
            row = json.loads(payload)
            key = row['segment'], row['baseline_version'], ('gold','stats','skills')[head], row['exploration']
            counts = exploration[key]
            counts.update(observations=1,unchanged=int(row['noop']),eligible=fitted)
            counts['censored' if label is None else {-1:'loss',0:'tie',1:'win'}[label]] += 1
            if label is not None:
                counts.update(resolved=1,delta=row['delta'])
        for split,step,payload in db.execute('SELECT split,step,payload FROM episodes'):
            row = json.loads(payload)
            boundary = row['segment'] if split=='train' else step
            key = split,boundary
            combat[key+('slot',)].update(row.get('combat_actions',{}))
            combat[key+('chosen_skill',)].update(row.get('skill_actions',{}))
            combat[key+('resolved_skill',)].update(row.get('skill_casts',{}))
            if (split,step) in (('eval_accepted_post',1_000_000),('eval_adaptive',1_025_000)):
                final[key].add(row)
            for visit in row['shop_exits']:
                for head,value in visit.get('configuration_choices',{}).items():
                    source = value.get('source','unchanged')
                    count = selections[(*key,visit['floor'],head)]
                    count.update(decisions=1)
                    count['selected_'+source] += 1
                    offered = value.get('offered',(0,0))
                    for typ,n in zip(('global','local'),offered):
                        count.update({'offered_'+typ:n,'opportunity_'+typ:int(n>0)})
        latest = db.execute('SELECT payload FROM updates ORDER BY id DESC LIMIT 1').fetchone()
        if latest:
            state = json.loads(latest[0])
            train_seconds = state['train_seconds']
            wall_seconds = max(state['wall_seconds'], state['train_seconds'])

    csv_table(output,'exploration_by_head.csv',
              'segment,baseline_version,head,search,comparisons,resolved,wins,ties,losses,censored,unchanged,fit_rows,win_rate,tie_rate,loss_rate,mean_floor_change',
              ((*k,*(c[a] for a in ('observations','resolved','win','tie','loss','censored','unchanged','eligible')),
                *[c[label]/c['resolved'] if c['resolved'] else None for label in ('win','tie','loss','delta')])
               for k,c in sorted(exploration.items())))
    csv_table(output,'configuration_adoption.csv',
              'split,segment,shop_floor,head,search,head_decisions,offered_configurations,decisions_with_offer,selected,selection_share,selection_rate_when_offered',
              ((*k,src,c['decisions'],c['offered_'+src],c['opportunity_'+src],c['selected_'+src],
                c['selected_'+src]/c['decisions'],
                c['selected_'+src]/c['opportunity_'+src] if c['opportunity_'+src] else None)
               for k,c in sorted(selections.items()) for src in ('global','local','anchor','unchanged')))
    csv_table(output,'head_promotion.csv','checkpoint,head,before,after,adopted,reason',
              ((step,head,d['before'],d['after'],d['adopted'],d['reason']) for step,p in boundaries
               for head,d in p.get('promotion',{}).get('head_changes',{}).items()))
    csv_table(output,'promotion_variants.csv',
              'checkpoint,variant,replaced,selected,reason,dev_pairs,dev_mean,dev_lower,validation_pairs,validation_mean,validation_lower',
              ((step,mode,' '.join(d['replaced_heads']),d['selected'],d['reason'],
                *(d[p][key] for p in ('development','heldout') for key in ('pairs','mean_floor_change','lower')))
               for step,data in boundaries for mode,d in data.get('promotion',{}).get('attempts',{}).items()))
    csv_table(output,'reward_q_context.csv','checkpoint,axis,bucket,reward,states,p10,p50,p90',
              ((step,row['axis'],row['bucket'],row['reward'],row['states'],*row['quantiles'])
               for step,data in boundaries for row in data.get('transfer',{}).get('q_by_context',[])))
    csv_table(output,'reward_probability_shift.csv','checkpoint,reward,mean_before,mean_after,change,kl,mean_total_variation,step_fraction',
              ((step,action,before,after,delta,t['kl'],t['mean_total_variation'],t['accepted_step_fraction'])
               for step,data in boundaries if (t:=data.get('transfer',{})) and 'probability_before' in t
               for action,before,after,delta in zip(REWARD_CHOICES,t['probability_before'],t['probability_after'],t['probability_changes'])))
    csv_table(output,'combat_strategy.csv','split,segment,action_type,action,selections,share',
              ((*k,action,n,n/c.total()) for k,c in sorted(combat.items()) for action,n in c.most_common()))
    csv_table(output,'disk_usage.csv','checkpoint,database_and_wal_bytes,free_bytes,reserve_bytes',
              ((step,d['database_bytes'],d['free_bytes'],d['reserve_bytes']) for step,row in boundaries if (d:=row.get('disk'))))

    wall_seconds = max(wall_seconds,boundaries[-1][1].get('wall_seconds',0.))
    baseline = _baseline(None)
    b = baseline.summary()
    lines = ['','## V3 comparison and resources','',
             f"Historical Uniform Random · {b['episodes']:,} episodes · {b['mean_floor']:.2f} mean floor · "
             f"{b['median_floor']:g} median · {b['max_floor']} maximum · {b['clear_rate']:.2%} clears.",
             'Its historical seed set differs from the fixed V3 panel. The matched Phase 3B-2 PPO baseline appears in the paired table.',
             '',f'Measured training time: {train_seconds:,.1f} seconds. Latest sampled elapsed lower bound: {wall_seconds:,.1f} seconds.',
             '', 'At a successful final boundary the 1,025,000-episode panel is compared with the 1,000,000-episode post-transfer panel.']
    if len(final)==2:
        a = final['eval_accepted_post',1_000_000].summary()
        z = final['eval_adaptive',1_025_000].summary()
        lines.extend(['','| Adaptation checkpoint | Episodes | Mean | Median | Max | Clear rate |',
                      '|---:|---:|---:|---:|---:|---:|'])
        for step,m in ((1_000_000,a),(1_025_000,z)):
            lines.append(f"| {step:,} | {m['episodes']} | {m['mean_floor']:.3f} | {m['median_floor']:g} | "
                         f"{m['max_floor']} | {m['clear_rate']:.2%} |")
    lines.extend(['','- [Global and Local training outcomes](exploration_by_head.csv) and [configuration adoption](configuration_adoption.csv)',
                  '- [Head history](head_promotion.csv) and [paired promotion variants](promotion_variants.csv)',
                  '- [Contextual Q estimates](reward_q_context.csv) and [Reward probability shifts and KL](reward_probability_shift.csv)',
                  '- [Combat usage](combat_strategy.csv) and [disk usage](disk_usage.csv)',
                  '', 'Origin labels refer to the immediate candidate choice. An anchor can itself contain earlier accepted edits.',
                  'A proposal attempt can return an unchanged configuration when legal moves are unavailable.'])
    text = summary.read_text(encoding='utf-8').replace('Adaptive Shop V2 results','Adaptive Shop V3 results',1)
    text = text.replace('## V2 boundary decisions','## Shop and Reward boundary decisions')
    summary.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
    return summary

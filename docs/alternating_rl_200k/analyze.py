import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alternating_rl.dqn import ShopDQN
from alternating_rl.report import HEADS, ShopStats, _q_band, _q_band_label
from alternating_rl.runner import DEFAULT, VERSION, read_config
from alternating_rl.shop import FEATURE_COUNT, HEAD_SIZES
from combat_rl.ppo import ActorCritic
from combat_rl.report import _weighted_median

OUT = Path(__file__).resolve().parent
RUN = ROOT / 'data/alternating_ppo_dqn'
SPAN = 25_000
STATS = ('attack', 'defense', 'hp', 'speed')


def db_at(path):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    return closing(db)


def write(name, header, rows):
    with (OUT / (name + '.csv')).open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header.split(','))
        writer.writerows(rows)


def measure(counter):
    n = sum(counter.values())
    return (n, min(counter), _weighted_median(counter),
            sum(value * count for value, count in counter.items()) / n,
            max(counter)) if n else (0, None, None, None, None)


def checkpoint_audit(db, points):
    torch.set_num_threads(1)
    cfg = read_config(DEFAULT)
    latest = torch.load(RUN / 'latest.pt', map_location='cpu', weights_only=False)
    state = latest['state']
    assert latest['version'] == VERSION and cfg == latest['config']
    maxima = {table: db.execute(f'SELECT coalesce(max(id),0) FROM {table}').fetchone()[0]
              for table in latest['marks']}
    replay = db.execute('SELECT min(id), max(id), count(*) FROM shop_replay').fetchone()
    n = state['episodes']
    train = db.execute("SELECT count(*), min(step), max(step), count(DISTINCT step), sum(actions) "
                       "FROM episodes WHERE split='train'").fetchone()
    assert train == (n, 1, n, n, state['steps'])
    assert maxima == latest['marks'] and replay[1] == latest['replay_mark']
    assert replay[2] == cfg['shop']['replay_capacity']
    assert state['last_eval'] == n and not latest['workers'] and not latest['observations']
    assert state['started'] == n
    active_shape = []
    for key, model in [('model', ActorCritic([128, 128])),
                       ('shop', ShopDQN(FEATURE_COUNT, HEAD_SIZES, cfg['shop']['hidden_dims'])),
                       ('shop_target', ShopDQN(FEATURE_COUNT, HEAD_SIZES, cfg['shop']['hidden_dims']))]:
        model.load_state_dict(latest[key], strict=True)
        assert all(torch.isfinite(x).all() for x in model.state_dict().values())
        if key != 'shop_target':
            opt = torch.optim.Adam(model.parameters())
            opt.load_state_dict(latest['combat_optimizer' if key == 'model' else 'shop_optimizer'])
            assert len(opt.state) == len(list(model.parameters()))
            for param, moments in opt.state.items():
                for name in ('exp_avg', 'exp_avg_sq'):
                    assert moments[name].shape == param.shape and torch.isfinite(moments[name]).all()
            active_shape.append((key, len(opt.state)))
    assert torch.cuda.is_available() and len(latest['cuda_rng']) == torch.cuda.device_count()
    for device, rng in enumerate(latest['cuda_rng']):
        torch.Generator(device=f'cuda:{device}').set_state(rng)
    torch.Generator().set_state(latest['torch_rng'])
    import random
    import numpy as np
    random.Random().setstate(latest['exploration_rng'])
    np.random.default_rng().bit_generator.state = latest['data_rng']
    archive = RUN / f'episode-{n:07d}.pt'
    sha = hashlib.sha256((RUN / 'latest.pt').read_bytes()).hexdigest()
    assert sha == hashlib.sha256(archive.read_bytes()).hexdigest()
    def changes(a, b):
        return sum(not torch.equal(a[k], b[k]) for k in a)
    old = None
    boundaries = []
    for step in points:
        row = torch.load(RUN / f'episode-{step:07d}.pt', map_location='cpu', weights_only=False)
        st = row['state']
        assert st['episodes'] == st['started'] == st['last_eval'] == step
        assert not row['workers'] and not row['observations']
        for table, mark in row['marks'].items():
            assert mark <= maxima[table]
            if mark:
                assert db.execute(f'SELECT 1 FROM {table} WHERE id=?', (mark,)).fetchone()
        ppo = changes(old['model'], row['model']) if old else 0
        shop = changes(old['shop'], row['shop']) if old else 0
        boundaries.append((step, st['version'], st['shop_updates'], ppo, shop, st['steps'],
                           st['train_seconds'], st['wall_seconds'], row['replay_mark']))
        old = row
    write('checkpoints', 'step,ppo_version,dqn_updates,ppo_tensors_changed,dqn_tensors_changed,actions,train_seconds,wall_seconds,replay_watermark', boundaries)
    return dict(state=state, marks=maxima, replay=replay, schedule=latest['schedule'],
                optimizer_parameter_counts=active_shape, checkpoint_sha256=sha,
                action_cap=cfg['action_cap'])


def strategy_features(row):
    allocation = [max(0, row[k] - 1) for k in STATS]
    mana = [s['uses'] * s['mana_cost'] for s in row['slots']]
    return {'any_gold_spent': row['entry_gold'] > row['exit_gold'],
            'no_gold_at_entry': row['entry_gold'] == 0,
            'any_gold_retained': row['exit_gold'] > 0,
            'no_unused_stats': row['unused_stat_points'] == 0,
            'all_stats_at_minimum': not sum(allocation),
            'most_points_unallocated': row['unused_stat_points'] > sum(allocation),
            'single_stat_majority': bool(sum(allocation) and max(allocation) * 2 > sum(allocation)),
            **{f'{k}_highest': x == max(allocation) and allocation.count(x) == 1
               for k, x in zip(STATS, allocation)},
            'no_unused_mana': row['remaining_mana'] == 0,
            'single_skill_mana_majority': bool(sum(mana) and max(mana) * 2 > sum(mana)),
            'any_zero_use_skill': any(not s['uses'] for s in row['slots']),
            'slot_unlocked': row.get('slot_unlocked', False)}


def analyze(db):
    groups = defaultdict(Counter)
    depth = defaultdict(Counter)
    labels = defaultdict(Counter)
    causes = Counter()
    visits = defaultdict(ShopStats)
    decisions = defaultdict(Counter)
    gold = defaultdict(Counter)
    associations = defaultdict(Counter)
    routes = defaultdict(Counter)
    route_depth = defaultdict(Counter)
    tails = []
    panel = defaultdict(dict)
    for split, step, payload in db.execute('SELECT split,step,payload FROM episodes ORDER BY id'):
        row = json.loads(payload)
        end = row['segment'] * SPAN if split == 'train' else step
        cohort = (split, end)
        floor = row['floor']
        counts = groups[cohort]
        combat = sum(row['combat_actions'].values())
        counts.update(n=1, actions=row['actions'], combat_actions=combat,
                      entries=len(row['shop_entries']), exits=len(row['shop_exits']))
        counts[row['outcome']] += 1
        depth[cohort][floor] += 1
        if row['outcome'] != 'cleared':
            cause = (row['timeout_phase'], row['timeout_monster']) if row['outcome'] == 'timeout' else ('combat', row['death_monster'])
            causes[(*cohort, row['outcome'], row['detail'] or '', floor, *[v or '' for v in cause])] += 1
        for field in ('combat_actions', 'skill_actions', 'skill_casts', 'reward_actions'):
            labels[(*cohort, field)].update(row[field])
        if split == 'train' and step > end - 12:
            tails.append((end, step, floor, row['actions'], combat, row['outcome'], row['timeout_phase'],
                          row['timeout_monster'], len(row['shop_actions']), row['death_monster']))
        if split == 'eval':
            seed = (str(row['game_seed']), str(row['policy_seed']))
            assert seed not in panel[step]
            panel[step][seed] = floor
        if row['event_route']:
            r = (*cohort, row['event_route'])
            routes[r].update(n=1, floor_sum=floor, clears=row['outcome'] == 'cleared',
                             shop99_entered=row['special_shop_entered'], shop99_exited=row['special_shop_exited'])
            route_depth[r][floor] += 1
        remaining = {v['floor']: v for v in row['shop_exits']}
        for entry in row['shop_entries']:
            shop_floor = entry['floor']
            before_after = remaining.pop(shop_floor, None)
            key = (*cohort, shop_floor)
            stat = visits[key]
            if before_after is None:
                stat.add_visit({'entry': entry, 'exit': None})
                stat.add_dqn((row['unfinished_shop'] or {}).get('dqn_actions', []))
                continue
            end_state = before_after
            edits = row['shop_actions'][end_state['action_start']:end_state['action_stop']]
            stat.add_visit({'entry': entry, 'exit': end_state, 'actions': edits, 'dqn_actions': end_state['dqn_actions']})
            y = end_state['target']
            assert y in (None, 0, 1)
            stat.add_target(y, end_state['prediction'], end_state['exit_q'])
            counts['censored'] += y is None
            counts['resolved'] += y is not None
            counts['milestones'] += y == 1
            delta = entry['gold'] - end_state['gold']
            gold[key].update(spent=delta, saved=end_state['gold'], entry=entry['gold'],
                             with_gold=int(entry['gold'] > 0), spend_visits=int(delta > 0),
                             slots_unlocked=end_state['unlocked_slots'] - entry['unlocked_slots'],
                             engine_actions=len(edits), dqn_selections=len(end_state['dqn_actions']))
            for act in edits:
                a = act['action']
                decisions[(*key, 'engine')][' '.join(map(str, a))] += 1
            end_state['slot_unlocked'] = end_state['unlocked_slots'] > entry['unlocked_slots']
            for feature, present in strategy_features(end_state).items():
                associations[(*key, end_state['version'], feature, present)][y] += 1
            if y is not None:
                head_values = defaultdict(list)
                for action in end_state['dqn_actions']:
                    head_values[action['head']].append(action['q'])
                for h, values in head_values.items():
                    band = _q_band(values[0])
                    stat.heads[h, band].update(n=1, successes=y, q_sum=values[0])
        assert not remaining
    assert all(set(seeds) == set(panel[0]) for seeds in panel.values())
    write('episodes_by_segment', 'split,end,episodes,mean_floor,median_floor,max_floor,clears,defeats,timeouts,actions,combat_actions,shop_entries,shop_exits,shop_targets_resolved,shop_targets_success,shop_targets_censored',
          ((*k, c['n'], measure(depth[k])[3], measure(depth[k])[2], max(depth[k]), c['cleared'], c['game_over'], c['timeout'],
            c['actions'], c['combat_actions'], c['entries'], c['exits'], c['resolved'], c['milestones'], c['censored']) for k, c in sorted(groups.items())))
    write('floor_reach', 'split,end,floor,reached,episodes',
          ((*k, f, sum(n for reached, n in c.items() if reached >= f), groups[k]['n'])
           for k,c in sorted(depth.items()) for f in range(1, 101)))
    write('termination_causes', 'split,end,outcome,detail,floor,phase,monster,count',
          ((*k,n) for k,n in sorted(causes.items())))
    write('training_boundary_tail', 'boundary,completion_ordinal,floor,actions,combat_actions,outcome,timeout_phase,timeout_monster,shop_actions,death_monster', tails)
    write('policy_actions', 'split,end,kind,action,count,group_total',
          ((*k,a,n,c.total()) for k,c in sorted(labels.items()) for a,n in c.most_common()))
    write('shop_by_floor', 'split,end,floor,entries,exits,successes,failures,censored,entry_gold,spent_gold,saved_gold,visits_with_gold,visits_spending_gold,slots_unlocked,engine_actions,dqn_selections',
          ((*k, *(s.counts[x] for x in ('entries','exits','success','failure','censored')),
            *(gold[k][x] for x in ('entry','spent','saved','with_gold','spend_visits','slots_unlocked','engine_actions','dqn_selections')))
           for k,s in sorted(visits.items())))
    write('shop_numeric_distributions', 'split,end,floor,period,metric,value,count',
          ((*k, phase, name, value,n) for k,s in sorted(visits.items()) for (phase,name), c in sorted(s.fields.items()) for value,n in sorted(c.items())))
    write('shop_reroll_distributions', 'split,end,floor,metric,value,count',
          ((*k, name, value,n) for k,s in sorted(visits.items()) for name,c in sorted(s.reroll.items()) for value,n in sorted(c.items())))
    write('shop_actions', 'split,end,floor,layer,action,count',
          ((*k,a,n) for k,c in sorted(decisions.items()) for a,n in c.most_common()))
    write('shop_loadouts', 'split,end,floor,ordered_skills,exits',
          ((*k,' / '.join(a),n) for k,s in sorted(visits.items()) for a,n in s.loadouts.most_common()))
    write('shop_memorize', 'split,end,floor,skill,assigned_uses,slots',
          ((*k,a,uses,n) for k,s in sorted(visits.items()) for (a,uses),n in sorted(s.uses.items())))
    write('shop_associations', 'split,end,floor,ppo_version,feature,present,successes,failures,censored',
          ((*k,c[1],c[0],c[None]) for k,c in sorted(associations.items())))
    adjusted = defaultdict(Counter)
    for (split,end,floor,ver,name,present), exposed in associations.items():
        if not present:
            continue
        other = associations.get((split,end,floor,ver,name,False), Counter())
        n, m = exposed[0] + exposed[1], other[0] + other[1]
        if n and m:
            w = n*m/(n+m)
            adjusted[split,end,floor,name].update(strata=1, present=n, absent=m, weight=w,
                                                  diff=w*(exposed[1]/n-other[1]/m))
    write('shop_associations_stratified', 'split,end,floor,feature,overlapping_versions,present_resolved,absent_resolved,weighted_rate_difference',
          ((*k,c['strata'],c['present'],c['absent'],c['diff']/c['weight']) for k,c in sorted(adjusted.items())))
    write('shop_head_activity', 'split,end,floor,head,selections,modifications,explored,early_finishes,cap_finishes,finished_modifications,adjacent_modification_pairs,repeat,inverse',
          ((*k,h,*(s.heads[h][x] for x in ('selections','modifications','explored','early','cap','finished_modifications','adjacent_modifications','repeat','inverse')))
           for k,s in sorted(visits.items()) for h in HEADS if s.heads[h]))
    write('shop_head_actions', 'split,end,floor,head,action,count',
          ((*k,h,a,n) for k,s in sorted(visits.items()) for h,c in s.head_actions.items() for a,n in c.most_common()))
    write('shop_head_q', 'split,end,floor,head,raw_q_band,decisions',
          ((*k,h,_q_band_label(b),n) for k,s in sorted(visits.items()) for h,c in s.head_q.items() for b,n in sorted(c.items())))
    write('shop_q_outcomes', 'split,end,floor,decision,raw_q_band,resolved,mean_q,successes,mean_squared_error',
          (values for k,s in sorted(visits.items()) for values in [
              *((*k,'exit',_q_band_label(b), c['n'],c['predictions']/c['n'],c['success'],c['squared_error']/c['n'])
                for b,c in sorted(s.calibration.items())),
              *((*k,HEADS[h]+' first',_q_band_label(b),c['n'],c['q_sum']/c['n'],c['successes'],None)
                for (h,b),c in sorted((k2,c2) for k2,c2 in s.heads.items() if isinstance(k2,tuple)))]))
    write('routes', 'split,end,route,episodes,mean_floor,max_floor,shop99_entered,shop99_exited,clears',
          ((*k,c['n'],c['floor_sum']/c['n'],max(route_depth[k]),c['shop99_entered'],c['shop99_exited'],c['clears'])
           for k,c in sorted(routes.items())))
    write('route_floor_reach', 'split,end,route,floor,reached,episodes',
          ((*k,f,sum(n for reached,n in c.items() if reached>=f),routes[k]['n'])
           for k,c in sorted(route_depth.items()) for f in range(35,101)))
    return groups, depth, labels, panel



def timings(db, groups):
    prior = None
    last = {}
    boundary = {}
    tail_stalls = defaultdict(Counter)
    losses = []
    replay = []
    for step, payload in db.execute('SELECT step,payload FROM updates ORDER BY id'):
        r = json.loads(payload)
        if r.get('target') == 'boundary_replay':
            replay.extend((step, row['cycle'], HEADS[row['head']], row['source'],
                           row['transitions'], row['terminals'], row['successes'])
                          for row in r['replay']['groups'])
            continue
        if 'rounds' not in r:
            continue
        end = (r['cycle'] * 2 - (r['target'] == 'combat_reward')) * SPAN
        if end not in last:
            last[end] = prior
        if prior is not None and prior['episodes'] >= end - 12:
            ts = tail_stalls[end]
            dt = r['train_seconds'] - prior['train_seconds']
            ts.update(seconds=dt, actions=r['steps']-prior['steps'], rounds=1)
            if step == prior['episodes']:
                ts.update(no_completion_seconds=dt, no_completion_rounds=1)
        prior = r
        if step == end:
            boundary[end] = r
        losses.append((end, r['rounds'], step, r['epsilon'], r['dqn_updates'], r.get('dqn_loss'),
                       r.get('dqn_mean_q'), r.get('dqn_mean_target'), r.get('dqn_terminal'),
                       r.get('policy_loss'), r.get('value_loss'), r['episodes_per_second'],
                       r['actions_per_second'], r['elapsed_seconds']))
    write('rollout_metrics', 'end,round,episodes,epsilon,dqn_updates,dqn_loss,dqn_mean_q,dqn_mean_target,dqn_minibatch_terminal_count,ppo_policy_loss,ppo_value_loss,episodes_per_sec,actions_per_sec,wall_elapsed_seconds', losses)
    write('replay_composition', 'step,cycle,head,source,transitions,terminals,successes', replay)
    phases, comp, resources = [], [], []
    for end, r in sorted(boundary.items()):
        p = last[end] or {}
        t = r['train_seconds'] - p.get('train_seconds', 0)
        c = groups['train',end]
        a = r['steps'] - p.get('steps', 0)
        assert a == c['actions']
        comp.append((end, t, c['n']/t, a/t, a/c['n'], c['combat_actions']/c['n'],
                     c['entries']/c['n'], r['version'] - p.get('version',0),
                     r['shop_updates'] - p.get('shop_updates',0),
                     tail_stalls[end]['seconds'], tail_stalls[end]['actions'],
                     tail_stalls[end]['no_completion_seconds'],tail_stalls[end]['no_completion_rounds']))
        for name,row in sorted(r['phases'].items()):
            previous = p.get('phases', {}).get(name, {})
            phases.append((end,name,row['seconds']-previous.get('seconds',0),
                           row['calls']-previous.get('calls',0)))
        resources.extend((end,key,value) for key,value in sorted(r['resources'].items()))
    write('performance', 'end,training_seconds,episodes_per_second,actions_per_second,actions_per_episode,combat_per_episode,shops_per_episode,ppo_updates,dqn_updates,tail_seconds,tail_actions,tail_no_completion_seconds,tail_no_completion_rounds',comp)
    write('profile', 'end,region,seconds,calls', phases)
    write('resources_cumulative', 'end,metric,value', resources)
    ev = []
    for step, p in db.execute('SELECT step,payload FROM evaluations ORDER BY step'):
        r = json.loads(p)
        c = groups['eval',step]
        assert r['episodes'] == c['n'] == 128 and r['actions'] == c['actions']
        ev.append((step,r['episodes'],r['actions'],r['seconds'],
                   r.get('shop_inference',{}).get('seconds',0),
                   r.get('shop_application',{}).get('seconds',0),
                   r.get('shop_random',{}).get('seconds',0)))
    assert set(step for step,*_ in ev) == {k for split,k in groups if split=='eval'}
    write('evaluation_timing', 'step,episodes,actions,seconds,shop_inference_seconds,shop_application_seconds,random_shop_seconds', ev)
    return comp


def comparison(panel):
    import numpy as np
    rng = np.random.default_rng(20261128)
    rows = []
    refs = {}
    overlap = {}
    for dirname in ('combat_reward_rl','combat_reward_rl_optimized'):
        with db_at(ROOT / 'data' / dirname / 'training.sqlite3') as db:
            for split in ('eval','random'):
                last = db.execute('SELECT max(step) FROM episodes WHERE split=?',(split,)).fetchone()[0]
                saved = {(a,b): floor for a,b,floor in db.execute('SELECT game_seed,policy_seed,floor FROM episodes WHERE split=? AND step=?',(split,last))}
                name = f'{dirname} {split}'
                overlap[name] = len(saved.keys() & panel[0].keys())
                refs[name] = saved
    base = panel[0]
    for step, values in sorted(panel.items()):
        for name, reference in [('alternating 0',base), ('previous checkpoint',panel.get(step-SPAN,base)), *refs.items()]:
            keys = sorted(values.keys() & reference.keys())
            if len(keys) != len(values):
                continue
            d = np.array([values[s]-reference[s] for s in keys])
            samples = rng.choice(d, size=(10000,len(d)), replace=True).mean(axis=1)
            lo, hi = np.quantile(samples,(.025,.975))
            rows.append((step,name,len(d),d.mean(),lo,hi,sum(d>0),sum(d==0),sum(d<0)))
    write('paired_evaluations', 'step,reference,pairs,delta_mean_floor,bootstrap_low,bootstrap_high,farther,tied,shorter', rows)
    return overlap


def plots(groups, depth, labels, comp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axs = plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for split, color in [('train','#2475a5'),('eval','#b84159')]:
        points = [(k,c) for (s,k),c in sorted(groups.items()) if s==split]
        x = [k/1000 for k,_ in points]
        ds = [depth[split,k] for k,_ in points]
        axs[0,0].plot(x,[measure(d)[3] for d in ds], 'o-', label=split.title(), color=color)
        axs[0,0].plot(x,[measure(d)[2] for d in ds], ':', color=color, alpha=.65)
        axs[0,1].plot(x,[sum(n for f,n in d.items() if f>=20)/c['n'] for d,(_,c) in zip(ds,points)],'o-',label=split.title(),color=color)
        for step,d in zip([k for k,_ in points],ds):
            if split=='eval' and step in [0,50000,100000,150000,200000]:
                axs[1,0].step(range(1,41),[sum(n for f,n in d.items() if f>=j)/d.total() for j in range(1,41)],
                              where='post',label=f'{step//1000}k')
    for name in ['Gold','Potion','Initialize','Stat','Exchange']:
        pts = [(k,c) for (s,k,f),c in sorted(labels.items()) if s=='eval' and f=='reward_actions']
        axs[1,1].plot([k/1000 for k,_ in pts],[c[name]/c.total() for _,c in pts],'o-',label=name)
    axs[0,0].set(ylabel='Floor',title='Mean (solid) and median (dotted)')
    axs[0,1].set(ylabel='Reached floor 20',title='Late-stage reach')
    axs[1,0].set(xlabel='Floor',ylabel='Fraction reaching floor',title='Fixed-panel reach by floor')
    axs[1,1].set(ylabel='Share of reward choices',title='Fixed-panel reward selection')
    for ax in axs.flat:
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    for ax in (axs[0,0],axs[0,1],axs[1,1]):
        ax.set_xlabel('Completed training episodes (thousands)')
    for ax in (axs[0,1],axs[1,0],axs[1,1]):
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.savefig(OUT/'performance_and_rewards.png',dpi=150)
    plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    X=[r[0]/1000 for r in comp]
    axs[0,0].bar(X,[r[2] for r in comp],width=11,color='#2475a5')
    axs[0,0].set(title='Training throughput by segment',ylabel='Episodes / second')
    for i,label in [(4,'All simulator actions'),(5,'Combat actions')]:
        axs[0,1].plot(X,[r[i] for r in comp],'o-',label=label)
    axs[0,1].set(title='Actions per completed training episode',ylabel='Actions / episode')
    axs[0,1].legend(fontsize=8)
    e = [groups['eval',k*1000] for k in [0,*map(int,X)]]
    axs[1,0].plot([0,*X],[c['actions'] for c in e],'o-',color='#b84159')
    axs[1,0].set(title='Work per fixed evaluation',ylabel='Simulator actions / 128 games')
    axs[1,1].bar(X,[r[11] for r in comp],width=11,color='#b84159')
    axs[1,1].set(title='Zero-completion rollouts near boundaries',ylabel='Seconds (last 12 episode range)')
    for ax in axs.flat:
        ax.set_xlabel('Completed training episodes (thousands)')
        ax.grid(axis='y', alpha=.3)
    fig.savefig(OUT/'compute_and_boundary_tails.png',dpi=150)
    plt.close(fig)



def shop_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    def rows(name):
        with (OUT / (name + '.csv')).open(newline='', encoding='utf-8') as source:
            return list(csv.DictReader(source))

    fig, ax = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    shops = [r for r in rows('shop_by_floor') if r['floor'] == '10']
    for split, color in [('train', '#2475a5'), ('eval', '#b84159')]:
        subset = [r for r in shops if r['split'] == split]
        x = [int(r['end']) / 1000 for r in subset]
        ax[0, 0].plot(x, [int(r['successes']) / (int(r['exits']) - int(r['censored']))
                         for r in subset], 'o-', color=color, label=split.title())
        ax[0, 1].plot(x, [int(r['engine_actions']) / int(r['exits'])
                         for r in subset], 'o-', color=color, label=split.title())
    activity = [r for r in rows('shop_head_activity') if r['split'] == 'eval' and r['floor'] == '10']
    for head in HEADS:
        subset = [r for r in activity if r['head'] == head]
        ax[1, 0].plot([int(r['end']) / 1000 for r in subset],
                      [int(r['cap_finishes']) / (int(r['cap_finishes']) + int(r['early_finishes']))
                       for r in subset], 'o-', label=head)
    q_rows = defaultdict(Counter)
    for r in rows('shop_q_outcomes'):
        if r['split'] == 'eval' and r['floor'] == '10' and r['decision'] == 'exit':
            n = int(r['resolved'])
            q_rows[int(r['end'])].update(n=n, q=n * float(r['mean_q']), y=int(r['successes']))
    x = [step / 1000 for step in sorted(q_rows)]
    points = [q_rows[step] for step in sorted(q_rows)]
    ax[1, 1].plot(x, [c['q'] / c['n'] for c in points], 'o-', label='Mean raw exit Q')
    ax[1, 1].plot(x, [c['y'] / c['n'] for c in points], 'o-', label='Observed milestone rate')
    for a, title in zip(ax.flat, ('Milestone after a floor-10 shop exit', 'Primitive actions per floor-10 exit',
                                  'Floor-10 head caps during evaluation', 'Floor-10 exit Q and outcome')):
        a.set_title(title)
        a.set_xlabel('Completed training episodes (thousands)')
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    ax[0, 0].yaxis.set_major_formatter(PercentFormatter(1))
    ax[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    fig.savefig(OUT / 'shop_diagnostics.png', dpi=150)
    plt.close(fig)



def main():
    with db_at(RUN / 'training.sqlite3') as db:
        points = [r[0] for r in db.execute('SELECT step FROM evaluations ORDER BY step')]
        audit = checkpoint_audit(db,points)
        groups, depth, labels, panel = analyze(db)
        for split,f,n in db.execute('SELECT split,target,count(*) FROM shop_samples GROUP BY split,target'):
            metric = {None:'censored',0:'resolved',1:'milestones'}[f]
            expected = sum(c[metric] - (c['milestones'] if f==0 else 0)
                           for (s,_),c in groups.items() if s==split)
            assert n == expected
        comp = timings(db,groups)
    audit['historical_eval_seed_pair_overlap'] = comparison(panel)
    (OUT/'checkpoint_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    plots(groups, depth, labels, comp)
    shop_figure()
    print('Verified', audit['state']['episodes'], 'training episodes and', len(panel), 'fixed panels')


if __name__ == '__main__':
    main()

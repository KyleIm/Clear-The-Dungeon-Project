"""Complete configurations, paired continuations, and live Reward experience."""

from collections import Counter, deque
from random import Random
from time import perf_counter

import numpy as np

from alternating_rl.features import summarize
from baseline_shop.configurations import HEADS, clone_shop, configuration_features, configurations, edit, replay
from baseline_shop.environment import Branch, Run as OldRun, frozen_policy
from combat_rl.environment import REWARD_CHOICES
from simulator import shop as engine_shop


def net(name, x, weights):
    return np.einsum('bi,oi->bo', x, weights[name+'.weight']) + weights[name+'.bias']


def scores(head, observations, weights):
    x = np.stack(observations)
    for layer in ('encoder.0','encoder.2'):
        x = np.tanh(net(layer,x,weights))
    x = net(f'heads.{head}',x,weights)
    p = np.exp(x-x.max(1,keepdims=True))
    return (p[:,2]-p[:,0])/p.sum(1)


def planning(run, policy, count, override=None, alternate=None):
    game = run.game
    actions, choices, treatment = [], {}, None
    for head in range(3):
        weights, trained, anchor_noop = alternate[1] if alternate is not None and alternate[0]==head else policy
        if head != override and not trained[head]:
            choices[HEADS[head]] = dict(edits=0, value=0.)
            continue
        started = perf_counter()
        rng = Random(f'v2-config-1:{run.policy_seed}:{game.floor}:{head}')
        options = configurations(game, head, rng, count, *run.context())
        if head == override:
            index = rng.randrange(int(not trained[head]),len(options)) if len(options)>1 else 0
            estimate = None
        else:
            values = scores(head, [c.features for c in options], weights)
            if anchor_noop:
                values[0] = 0.
            index = int(values.argmax())
            estimate = float(values[index])
        selected = options[index]
        if head == override:
            treatment = selected.features
        if selected.actions:
            game = replay(game, selected.actions)
            actions.extend(selected.actions)
        choices[HEADS[head]] = dict(edits=len(selected.actions), value=estimate)
        run.shop_work[f'plan_{HEADS[head]}_seconds'] += perf_counter()-started
    return tuple(actions), game, choices, treatment


def recent_turn(trace, game, action):
    p, m = game.player, game.monster
    trace.append((action[1], m.ram_action, p.hp, m.hp,
                  tuple((k,p.memorized_skills.get(k,0)) for k in p.current_skills if k!='lock'),
                  tuple(x['Name'] for x in p.status_conditions), tuple(x['Name'] for x in m.status_conditions)))


def timeout_snapshot(game, trace, name):
    patterns = Counter((row[0], row[1]) for row in trace)
    stable = len(trace) >= 8 and len(set(trace)) == 1
    pair = next(iter(patterns), (None,None))
    signature = ('Mimic/Sacrifice/Counter compatible' if stable and game.monster.name.lower()=='mimic'
                 and tuple(map(str.lower,map(str,pair))) == ('sacrifice','counter') else
                 'identical recorded combat snapshots' if stable else 'undetermined')
    state = game.get_state()
    return dict(branch=name, floor=game.floor, monster=game.monster.name, phase=state['phase'],
                player=state['player'], enemy=state['monster'], last_turns=list(trace),
                patterns=[(*k,n) for k,n in patterns.items()], cause='action_limit', signature=signature)


class Run(OldRun):
    def __init__(self, *args, shop_version=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.shop_version = shop_version
        self.experience = []
        self.latest_probability = None
        self.tail = deque(maxlen=24)
        self.tail_floor = None
        self.strategy_checks = []
        self.enabled_heads = (0,1,2)

    def learned(self, selected, probabilities, version):
        self.latest_probability = probabilities
        return super().learned(selected, probabilities, version)

    def select_action(self, selected):
        action, source = super().select_action(selected)
        if self.phase == 'reward' and self.mode == 'paired':
            from baseline_shop.environment import hp_bucket, next_shop_distance
            p = self.state['player']
            obs, _, mask = self.observation()
            name = REWARD_CHOICES[action]
            probability = float(self.latest_probability[action])
            meta = dict(action=name, source=source, floor=self.floor, hp=p['hp'], max_hp=p['max_hp'],
                        hp_bucket=hp_bucket(p['hp'],p['max_hp']),
                        shop_distance=next_shop_distance(self.state),
                        gold=self.state['gold'], shop_version=self.shop_version, ppo_version=self.version,
                        probability=probability, behavior_probability=.8*probability+.2/int(mask.sum()),
                        current_stats={k:p[k] for k in ('attack','defense','speed','ACC','EVA','DR')},
                        base_stats={k:p['base_'+k] for k in ('attack','defense','hp','speed','ACC','EVA','DR')},
                        skills=[(k,p['memorized_skills'].get(k,0)) for k in p['current_skills']])
            self.experience.append((action,obs,mask[:5],meta))
        return action, source

    def _advance(self, action):
        if self.phase == 'combat':
            if self.floor != self.tail_floor:
                self.tail.clear()
                self.tail_floor = self.floor
            super()._advance(action)
            # The simulator can replace the encounter during transitions such as a reward.
            recent_turn(self.tail, self.game, action)
        else:
            super()._advance(action)

    def result(self):
        row = super().result()
        return dict(row, _experience=self.experience, strategy_checks=self.strategy_checks,
                    shop_version=self.shop_version,
                    timeout=timeout_snapshot(self.game,self.tail,self.mode) if row['outcome']=='timeout' else None)


class Continuation(Branch):
    def __init__(self, *args):
        super().__init__(*args)
        self.tail = deque(maxlen=24)
        self.tail_floor = None

    def advance(self, action):
        super().advance(action)
        if action[0]=='skill':
            if self.floor != self.tail_floor:
                self.tail.clear()
                self.tail_floor = self.floor
            recent_turn(self.tail,self.game,action)


def outcomes(run, paths, weights, budget, exploration):
    key = f'v2-paired:{run.policy_seed}:{run.game.floor}'
    branches = [Continuation(run, path, key, budget) for _,path in paths]
    policy = frozen_policy(weights)
    while alive := [b for b in branches if not b.complete()]:
        for b in alive:
            if b.state['phase']=='event':
                b.advance(b.tapes[2].choice(b.game.get_legal_actions()))
        batch = [b for b in alive if not b.complete() and b.state['phase']!='event']
        if not batch:
            continue
        observations = [b.observation() for b in batch]
        for b, (_,phase,mask), probs in zip(batch,observations,policy(observations)):
            legal = np.flatnonzero(mask)
            draw = b.tapes[phase].random()
            if phase==1 and b.tapes[3].random()<exploration:
                selected = b.tapes[4].choice(legal)
            elif exploration == 0:
                selected = legal[int(np.argmax(probs[legal]))]
            else:
                cdf = np.cumsum(probs[legal])
                selected = legal[min(len(legal)-1,int(np.searchsorted(cdf,draw*cdf[-1],side='right')))]
            b.advance(dict(b._decision_actions())[selected])
    results = []
    for branch,(label,_) in zip(branches,paths):
        floor,actions,censored = branch.result()
        results.append(dict(floor=floor, actions=actions, clear=branch.game.is_cleared(),
                            timeout=timeout_snapshot(branch.game,branch.tail,label) if censored else None))
    return results


def compare(run, snapshot, count, budget, base_plan):
    start = perf_counter()
    ppo, baseline = snapshot
    plans = [planning(run,baseline,count,head) for head in range(3)]
    result = outcomes(run,[('adaptive',base_plan), *((name,row[0]) for name,row in zip(HEADS,plans))],
                      ppo,budget,.20)
    entry = result[0]
    rows = []
    for head, (path,game,choices,features) in enumerate(plans):
        end = result[head+1]
        delta = end['floor']-entry['floor']
        utility = delta+int(end['clear'])-int(entry['clear'])
        row = dict(shop_floor=run.floor, baseline_floor=entry['floor'], floor=end['floor'],
                   delta=delta, label=None if entry['timeout'] or end['timeout'] else int(np.sign(utility)),
                   noop=path==base_plan, treatment_no_edit=not choices[HEADS[head]]['edits'],
                   baseline_version=run.shop_version,
                   baseline_clear=entry['clear'], clear=end['clear'], timeouts=[x for x in
                        (entry['timeout'],end['timeout']) if x], branch_actions=end['actions'],
                   entry_gold=run.game.gold, baseline_choices=choices,
                   configuration=summarize(game, previous_area=run.context()[0],event_choice=run.event_route))
        rows.append((head, features, row))
    run.shop_work['paired_seconds'] += perf_counter()-start
    run.shop_work['paired_actions'] += sum(row['actions'] for row in result)
    run.shop_work['paired_branches'] += 4
    return rows


def probes(run, snapshot, path, budget, value_policy, *, predict=None):
    """Probe an incremental allocation or refund, with a spare equal-spend alternative."""
    ppo, _ = snapshot
    final = replay(run.game,path)
    picked = []
    for head, resource, field in ((1,'stat_points','available_points'),(2,'mana','current_mana')):
        enter = ('shop_menu',HEADS[head])
        menu = clone_shop(final)
        edit(menu,enter)
        legal = engine_shop.legal_actions(menu)
        if head==1:
            moves = [a for a in legal if a[0]=='adjust_stat' and a[2]==1]
            moves = moves or [a for a in legal if a[0]=='adjust_stat' and a[2]==-1]
        else:
            moves = [a for a in legal if a[0]=='memorize'] or [a for a in legal if a[0]=='forget']
        options = []
        before = getattr(final.player,field)
        for action in moves:
            edits = (enter,action,('shop_back',))
            changed = replay(final,edits)
            remaining = getattr(changed.player,field)
            options.append((head,resource,action,before,remaining,
                            configuration_features(final,changed,*run.context()),edits))
        if options:
            picked.append(options[0])
            picked.extend([x for x in options[1:] if x[4]==options[0][4]][:1])
    if not picked:
        return []
    paths = [('unchanged allocation',path)]
    for _,resource,_,_,_,_,edits in picked:
        paths.append((resource,path+edits))
    result = outcomes(run,paths,ppo,budget,0.)
    base = result[0]
    unchanged = configuration_features(final,final,*run.context())
    weights, trained, anchor = value_policy
    run.shop_work['probe_actions'] += sum(row['actions'] for row in result)
    records = []
    for (head,resource,move,before,after,feature,_),observed in zip(picked,result[1:]):
        change = None
        if predict is not None:
            change = predict(head,feature,unchanged)
        elif trained[head]:
            changed, prior = scores(head,[feature,unchanged],weights)
            change = float(changed if anchor else changed-prior)
        records.append(dict(resource=resource,action=move,before=before,after=after,score=change,
                            floor=observed['floor'],baseline_floor=base['floor'],
                            censored=bool(observed['timeout'] or base['timeout']),
                            timeouts=[x for x in (observed['timeout'],base['timeout']) if x]))
    return records

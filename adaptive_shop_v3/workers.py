"""Apply a frozen per-head stack and compare one changed head at a time."""

from time import perf_counter

import numpy as np

from adaptive_shop.gameplay import outcomes, probes, scores
from adaptive_shop.workers import handle as v2
from alternating_rl.features import summarize
from baseline_shop.configurations import HEADS, configuration_features, replay
from baseline_shop.workers import pack

from .planning import choose, treatment


def plan(run, stack, count, override=None, *, cache=None):
    if cache is None:
        cache = {}
    game, path, choices, observed = run.game, (), {}, None
    for head, policy in enumerate(stack):
        started = perf_counter()
        key = head,path
        if key not in cache:
            cache[key] = choose(game,head,run.policy_seed,policy,run.context())
        selected = cache[key]
        if head==override:
            selected = treatment(game,head,run.policy_seed,policy,count,run.context(),anchor=selected)
        final = replay(game,selected.actions) if selected.actions else game
        if head==override:
            observed = configuration_features(game,final,*run.context())
        game = final
        path += selected.actions
        choices[HEADS[head]] = dict(edits=len(selected.actions),value=selected.estimate,
                                   source=selected.source,offered=list(selected.offered))
        run.shop_work['plan_'+HEADS[head]+'_seconds'] += perf_counter()-started
    return path, game, choices, observed


def compare(run, stack, weights, baseline, config, cache):
    begin = perf_counter()
    treatments = [plan(run,stack,config['candidates'],head,cache=cache) for head in range(3)]
    paths = [('adaptive',baseline[0]), *((label,row[0]) for label,row in zip(HEADS,treatments))]
    results = outcomes(run,paths,weights,config['comparison_budget'],.20)
    base = results[0]
    rows = []
    for head,(path,final,choices,encoded) in enumerate(treatments):
        outcome = results[head+1]
        delta = outcome['floor']-base['floor']
        relative = delta+int(outcome['clear'])-int(base['clear'])
        timeouts = [value for value in (base['timeout'],outcome['timeout']) if value]
        details = dict(shop_floor=run.floor,baseline_floor=base['floor'],floor=outcome['floor'],
                       delta=delta,label=None if timeouts else int(np.sign(relative)),timeouts=timeouts,
                       noop=path==baseline[0],baseline_version=run.shop_version,entry_gold=run.game.gold,
                       baseline_clear=base['clear'],clear=outcome['clear'],branch_actions=outcome['actions'],
                       exploration=choices[HEADS[head]]['source'],
                       configuration=summarize(final,previous_area=run.context()[0],event_choice=run.event_route))
        rows.append((head,encoded,details))
    run.shop_work['paired_seconds'] += perf_counter()-begin
    run.shop_work['paired_actions'] += sum(row['actions'] for row in results)
    run.shop_work['paired_branches'] += 4
    return rows


def handle(runs,slot,operation,arg):
    if operation!='shop' or arg[0]=='uniform':
        return v2(runs,slot,operation,arg)
    run = runs[slot]
    mode, config = arg
    before = run.actions
    models = runs[-1]
    cache = {}
    if mode=='baseline':
        path, choices = (), {}
    else:
        stack = models['variants'][mode] if mode.startswith('variant_') else models['adaptive']
        result = plan(run,stack,config['candidates'],cache=cache)
        path, _, choices, _ = result
    rows = compare(run,stack,models['ppo'],result,config,cache) if mode=='paired' else []
    if mode=='probe' and not run.strategy_checks:
        value_stack = models['variants'].get('variant_combined',stack)

        def predict(head,changed,unchanged):
            policy = value_stack[head]
            if policy is None:
                return None
            a,b = scores(head,[changed,unchanged],policy.weights)
            return float(a-b)

        run.strategy_checks.extend(probes(run,(models['ppo'],None),path,config['comparison_budget'],
                                         (None,(),False),predict=predict))
    for action in path:
        if run.done:
            break
        run._advance(action)
    run.applied = choices
    reward,_ = run.leave()
    return pack(run,before,reward), rows

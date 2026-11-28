"""Apply the versioned incumbent and collect isolated configuration treatments."""

from baseline_shop.workers import pack

from .gameplay import Run, compare, planning, probes


def handle(runs, slot, operation, arg):
    if operation == 'policy':
        runs[-1] = arg
        return None
    if operation == 'restore':
        runs[slot] = arg
        return None
    if operation == 'reset':
        args, kwargs = arg
        runs[slot] = Run(*args,**kwargs)
        return pack(runs[slot],0)
    run = runs[slot]
    before = run.actions
    if operation == 'snapshot':
        return run
    if operation == 'step':
        return pack(run,before,run.learned(*arg))
    if operation != 'shop':
        raise ValueError(f'Unknown V2 worker operation {operation}')
    mode, config = arg
    if mode == 'uniform':
        return pack(run,before,run.choose_random_shop()), []
    treatments = []
    if mode == 'baseline':
        path, choices = (), {}
    else:
        models = runs[-1]
        snapshot = models['candidate' if mode=='candidate' else 'adaptive']
        alternate = (int(mode[-1]), models['candidate'][1]) if mode.startswith('candidate_') else None
        path, _, choices, _ = planning(run,snapshot[1],config['candidates'],alternate=alternate)
        if mode == 'paired':
            treatments = compare(run,snapshot,config['candidates'],config['comparison_budget'],path)
        if mode == 'probe' and not run.strategy_checks:
            value_policy = models.get('candidate',snapshot)[1]
            run.strategy_checks.extend(probes(run,snapshot,path,config['comparison_budget'],value_policy))
    for action in path:
        if run.done:
            break
        run._advance(action)
    run.applied = choices
    reward, _ = run.leave()
    return pack(run,before,reward), treatments

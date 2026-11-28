"""Simulator-side application and batched configuration selection."""

from typing import NamedTuple

from .configurations import HEADS, allocation_probes
from .environment import Run, compare


class Reply(NamedTuple):
    reward: float
    actions: int
    phase: str
    observation: object
    result: object


def pack(run, before, reward=0):
    obs = run.observation() if run.phase in ('combat', 'reward') else None
    return Reply(reward, run.actions - before, run.phase, obs, run.result() if run.done else None)


def handle(runs, slot, op, arg):
    if op == 'policy':
        runs[-1] = arg
        return None
    if op == 'reset':
        args, kwargs = arg
        run = runs[slot] = Run(*args, **kwargs)
        return pack(run, 0)
    if op == 'restore':
        runs[slot] = arg
        return None
    run = runs[slot]
    before = run.actions
    if op == 'snapshot':
        return run
    if op == 'step':
        return pack(run, before, run.learned(*arg))
    if op == 'random_shop':
        return pack(run, before, run.choose_random_shop())
    if op == 'compare':
        return compare(run, runs[-1], arg)
    if op == 'offer':
        return run.offer(*arg)
    if op == 'apply':
        run.apply_candidate(*arg)
        return pack(run, before)
    if op == 'probe':
        return list(allocation_probes(run.game, *run.context()))
    if op == 'leave':
        run.probes = arg
        reward, _ = run.leave()
        return pack(run, before, reward)
    raise ValueError(f'Unknown baseline shop worker operation: {op}')


def bridge(pool, replies, actor, trained, mode, config, profile):
    pending = [slot for slot, reply in replies.items() if reply.phase == 'shop']
    if not pending:
        return replies, []

    def merge(additions):
        for slot, current in additions.items():
            previous = replies[slot]
            replies[slot] = current._replace(reward=previous.reward + current.reward,
                                             actions=previous.actions + current.actions)

    if mode == 'uniform':
        with profile.measure('uniform_shop'):
            merge(pool.call('random_shop', dict.fromkeys(pending)))
        return replies, []
    labels = []
    if mode == 'paired':
        with profile.measure('baseline_comparison'):
            compared = pool.call('compare', dict.fromkeys(pending, config['comparison_budget']))
        labels = [row for rows in compared.values() for row in rows]
    heads = range(3) if mode in ('paired', 'combined') else (
        (HEADS.index(mode),) if mode in HEADS else ())
    for head in heads:
        valid = [slot for slot in pending if replies[slot].phase == 'shop']
        if not valid or not trained[head]:
            continue
        with profile.measure('generate_' + HEADS[head]):
            batches = pool.call('offer', dict.fromkeys(valid, (head, config['candidates'])))
        flat = [features for candidates in batches.values() for features in candidates]
        with profile.measure('score_' + HEADS[head]):
            scores = actor.scores(head, flat)
        chosen = {}
        offset = 0
        for slot, candidates in batches.items():
            values = scores[offset:offset + len(candidates)]
            values[0] = 0  # A deterministic no-op has paired relative value zero.
            index = int(values.argmax())
            chosen[slot] = (head, index, float(values[index]))
            offset += len(candidates)
        with profile.measure('apply_' + HEADS[head]):
            merge(pool.call('apply', chosen))
    continuing = [slot for slot in pending if replies[slot].phase == 'shop']
    diagnostics = {slot: [] for slot in continuing}
    if mode == 'combined' and config.get('probe'):
        with profile.measure('allocation_probes'):
            for slot, options in pool.call('probe', dict.fromkeys(continuing)).items():
                for head, field, action, before, after, feature in options:
                    if trained[head]:
                        score = float(actor.scores(head, [feature])[0])
                        diagnostics[slot].append(dict(resource=field, action=action, before=before,
                                                      after=after, score=score))
    merge(pool.call('leave', diagnostics))
    return replies, labels

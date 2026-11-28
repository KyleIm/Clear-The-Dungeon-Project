"""Bounded global and local proposals with independently frozen Shop heads."""

from dataclasses import dataclass
from random import Random
from types import SimpleNamespace

import numpy as np

from adaptive_shop.gameplay import scores
from baseline_shop.configurations import HEADS, build, clone_shop, configurations, edit, features, replay
from simulator import shop as engine

GENERATOR = 'v3-mixed-1'


@dataclass(frozen=True)
class HeadPolicy:
    step: int
    width: int
    weights: dict
    anchor: object


@dataclass(frozen=True)
class Choice:
    actions: tuple = ()
    source: str = 'unchanged'
    estimate: float = 0.
    offered: tuple = (0, 0)


def snapshot(model, trained, step, width, anchors):
    weights = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    for value in weights.values():
        value.flags.writeable = False
    encoder = {name: w for name,w in weights.items() if name.startswith('encoder.')}
    return tuple(HeadPolicy(step,width,encoder | {name:w for name,w in weights.items()
                                                if name.startswith(f'heads.{i}.')},anchors[i])
                 if trained[i] else anchors[i] for i in range(3))


def origin_rng(game, head, seed, label, step=0):
    return Random(f'{GENERATOR}:{seed}:{game.floor}:{head}:{label}:{step}')


def local_candidate(game, head, rng):
    """Return one short, engine-legal path relative to an incumbent configuration."""
    shadow = clone_shop(game)
    path = []

    def do(action):
        edit(shadow,action)
        path.append(action)

    if head==0:
        legal = []
        for menu in ('merchandise','costs'):
            view = SimpleNamespace(data=game.data,player=game.player,gold=game.gold,
                                   reroll_cost=game.reroll_cost,shop_menu=menu)
            legal.extend((menu,a) for a in engine.legal_actions(view)
                         if a[0] in ('buy','reduce_skill_cost','change_type'))
        if not legal:
            return ()
        menu,move = rng.choice(legal)
        do(('shop_menu',menu))
        do(move)
    else:
        do(('shop_menu',HEADS[head]))
        legal = engine.legal_actions(shadow)
        valid = ('adjust_stat',) if head==1 else ('replace_skill','memorize','forget')
        groups = [moves for kind in valid if (moves := [a for a in legal if a[0]==kind])]
        if not groups:
            return ()
        move = rng.choice(rng.choice(groups))
        do(move)
        if head==1 and move[2]<0 and rng.random()<.5:
            transfers = [a for a in engine.legal_actions(shadow) if a[0]=='adjust_stat'
                         and a[1]!=move[1] and a[2]==-move[2]]
            if transfers:
                do(rng.choice(transfers))
        elif head==2 and move[0] in ('memorize','forget'):
            for _ in range(rng.randrange(3)):
                if move in engine.legal_actions(shadow):
                    do(move)
            if move[0]=='forget' and rng.random()<.5:
                additions = [a for a in engine.legal_actions(shadow)
                             if a[0]=='memorize' and a[1]!=move[1]]
                if additions:
                    do(rng.choice(additions))
    do(('shop_back',))
    return tuple(path)


def global_options(game, head, rng, width, context):
    return [row.actions for row in configurations(game,head,rng,width,*context)]


def proposals(game, head, seed, policy, context, cache=None):
    """Generate at most width unique configurations using a frozen parent's choice."""
    if cache is None:
        cache = {}
    anchor = choose(game,head,seed,policy.anchor,context,cache).actions
    base = replay(game,anchor) if anchor else game
    entry = cache.get('entry')
    if entry is None:
        entry = cache['entry'] = features(game,*context)
    anchor_features = features(base,*context) if anchor else entry
    seen = {anchor_features.tobytes()}
    result = [(anchor,'anchor',np.concatenate((entry,anchor_features)))]
    rng = origin_rng(game,head,seed,'pool',policy.step)
    for _ in range(policy.width-1):
        source = 'local' if rng.random()<.5 else 'global'
        if source=='local':
            actions = anchor+local_candidate(base,head,rng)
        else:
            slot = rng.randrange(policy.width)
            if slot not in cache:
                generator = origin_rng(game,head,seed,'global',slot)
                cache[slot] = build(clone_shop(game),head,generator,slot) if slot else ()
            actions = cache[slot]
        end = replay(game,actions) if actions else game
        encoded = features(end,*context) if actions else entry
        key = encoded.tobytes()
        if key not in seen:
            result.append((actions,source,np.concatenate((entry,encoded))))
            seen.add(key)
    return result


def choose(game, head, seed, policy, context, cache=None):
    if policy is None:
        return Choice()
    if cache is None:
        cache = {}
    options = proposals(game,head,seed,policy,context,cache)
    values = scores(head,[row[2] for row in options],policy.weights)
    values[0] = 0.  # The same configuration has exact relative value zero.
    index = int(values.argmax())
    actions,source,_ = options[index]
    return Choice(actions,source,float(values[index]),
                  tuple(sum(row[1]==src for row in options) for src in ('global','local')))


def treatment(game, head, seed, policy, count, context, *, anchor=None):
    """Flip one fair coin per head treatment and sample a legal changed proposal."""
    if anchor is None:
        anchor = choose(game,head,seed,policy,context)
    rng = origin_rng(game,head,seed,'treatment')
    source = 'local' if rng.random()<.5 else 'global'
    if source=='local':
        base = replay(game,anchor.actions) if anchor.actions else game
        actions = anchor.actions+local_candidate(base,head,rng)
    else:
        options = [row for row in global_options(game,head,rng,count,context) if row!=anchor.actions]
        actions = rng.choice(options) if options else anchor.actions
    return Choice(actions,source)

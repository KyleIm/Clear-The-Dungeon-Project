"""Conservative paired promotion on development and fresh validation seeds."""

import numpy as np


def superiority(incumbent, candidate, *, minimum, alpha, seed):
    if not incumbent or incumbent.keys() != candidate.keys():
        raise ValueError('Promotion requires matching paired evaluation seeds')
    base, edit = zip(*[(incumbent[key], candidate[key]) for key in sorted(incumbent)])
    # A clear outranks any failed run. Non-clears retain their observed furthest floor.
    a = np.asarray(base, dtype=np.int64)
    b = np.asarray(edit, dtype=np.int64)
    d = b[:,0] + b[:,1] - a[:,0] - a[:,1]
    rng = np.random.default_rng(seed)
    samples = np.concatenate([rng.choice(d, (1000,len(d)), replace=True).mean(1) for _ in range(10)])
    lower = float(np.quantile(samples, alpha))
    return dict(pairs=len(d), lower=lower, mean_floor_change=float(np.mean(b[:,0]-a[:,0])),
                mean_utility_change=float(d.mean()), wins=int((d>0).sum()), ties=int((d==0).sum()),
                losses=int((d<0).sum()), incumbent_clears=int(a[:,1].sum()), candidate_clears=int(b[:,1].sum()),
                incumbent_timeouts=int(a[:,2].sum()), candidate_timeouts=int(b[:,2].sum()),
                passed=bool(len(d)>=minimum and lower>0 and b[:,1].sum()>=a[:,1].sum()
                            and b[:,2].sum()<=a[:,2].sum()))


def decision(incumbent, candidate, heldout_incumbent, heldout_candidate, *, minimum, alpha=.0025, seed=0):
    development = superiority(incumbent,candidate,minimum=minimum,alpha=alpha,seed=seed)
    heldout = superiority(heldout_incumbent,heldout_candidate,minimum=minimum,alpha=alpha,seed=seed+1)
    return dict(promoted=development['passed'] and heldout['passed'],
                development=development, heldout=heldout, alpha=alpha, minimum_pairs=minimum)

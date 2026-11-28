"""Bounded integration run, interruption recovery and planning microbenchmark."""

import argparse
from contextlib import closing
from pathlib import Path
import json
import sqlite3
from statistics import median
from tempfile import mkdtemp
from time import perf_counter

import numpy as np
import torch

from adaptive_shop.gameplay import Run
from baseline_shop.model import ConfigurationValue
from baseline_shop.report import read_only
from baseline_shop.validate import fingerprint

from .planning import snapshot
from .report import report
from .runner import DEFAULT, INITIAL, ROOT, configuration, train
from .workers import plan


def benchmark():
    run = Run(17,42,10000,mode='adaptive')
    run.game.floor = run.floor = 10
    run.game.phase = 'shop'
    run.game.gold = 1500
    run.state = run.game.get_state()
    model = ConfigurationValue()
    stack = (None,)*3
    for depth in range(1,5):
        stack = snapshot(model,(True,True,True),depth*50000,8,stack)
        if depth in (1,4):
            samples = []
            for _ in range(20):
                start = perf_counter()
                plan(run,stack,8)
                samples.append((perf_counter()-start)*1000)
            print(f'{depth} saved versions per head: median complete three-head planning {median(samples):.3f} ms')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--initial',type=Path,default=INITIAL)
    args = parser.parse_args()
    cfg = configuration(DEFAULT)
    assert cfg['first_stop']==cfg['episodes']==1_025_000
    cfg.update(device=args.device,episodes=48,alternating_episodes=32,first_stop=48,segment_episodes=16,
               workers=4,processes=2,eval_episodes=4,heldout_episodes=4,promotion_minimum=4,
               rollout_rounds=16,checkpoint_rounds=2,action_cap=512,comparison_budget=512,
               candidates=4,batch_size=24,reward_rows=2048,reward_epochs=2,reward_support=1,probe_episodes=2)
    root = ROOT/'data'
    directory = Path(mkdtemp(prefix='adaptive_shop_v3_validation_',dir=root))
    protected = [args.initial,*(root/subdir/name for subdir in
                                ('adaptive_shop_v2','baseline_relative_shop','alternating_ppo_dqn')
                                for name in ('latest.pt','training.sqlite3'))]
    before = [fingerprint(p) for p in protected]
    print(directory,flush=True)
    (directory/'settings.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    train(cfg,directory,args.initial,stop=32)
    dbpath = directory/'training.sqlite3'
    # Mimic a committed SQLite tail left by a process that stopped before replacing its checkpoint.
    with closing(sqlite3.connect(dbpath)) as db, db:
        db.execute("INSERT INTO updates(step,payload) VALUES (33,?)", ('{\"interrupted_tail\":true}',))
    train(cfg,directory,args.initial,resume=True)

    def load(name):
        return torch.load(directory/name,map_location='cpu',weights_only=False)

    pre,post,last = (load(name) for name in ('episode-0000032-pre.pt','episode-0000032.pt','latest.pt'))
    assert last['state']['episodes']==last['state']['last_eval']==48
    assert all(torch.equal(pre['model'][k],post['model'][k]) for k in pre['model']
               if not k.startswith('reward_head.'))
    assert any(not torch.equal(pre['model'][k],post['model'][k]) for k in pre['model'] if k.startswith('reward_head.'))
    assert all(torch.equal(last['shop'][k],post['shop'][k]) for k in last['shop'])
    assert last['state']['baseline_version']==post['state']['baseline_version']
    with read_only(dbpath) as db:
        assert db.execute("SELECT COUNT(*) FROM updates WHERE payload LIKE '%interrupted_tail%'").fetchone()[0]==0
        for table,mark in last['marks'].items():
            assert db.execute(f'SELECT coalesce(max(id),0) FROM {table}').fetchone()[0]==mark
        assert db.execute("SELECT count(*) FROM episodes WHERE split='train'").fetchone()[0]==48
        origin = db.execute("SELECT json_extract(payload,'$.exploration'),count(*) FROM comparisons GROUP BY 1").fetchall()
        assert {kind for kind,_ in origin}=={'global','local'}
        guid = json.loads(db.execute('SELECT payload FROM boundaries WHERE step=32').fetchone()[0])['transfer']
        assert guid['applied'] and guid['kl']<=.05 and guid['forced_training_counts']['Gold']>0
        assert np.isclose(sum(guid['probability_before']),1)
        assert np.isclose(sum(guid['probability_after']),1)
    assert [fingerprint(p) for p in protected]==before
    print('Exploration:',origin,flush=True)
    print(f"Reward transfer KL: {guid['kl']:.5f} | TV: {guid['mean_total_variation']:.5f}",flush=True)
    print(report(directory),flush=True)
    benchmark()
    print('Verified final adaptation, interrupted-tail recovery, model isolation, local reports and prior artifact preservation.')


if __name__=='__main__':
    main()

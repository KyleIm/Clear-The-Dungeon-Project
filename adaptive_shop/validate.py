"""Bounded CUDA validation of V2 on the original Phase 3B-2 checkpoint."""

import argparse
import json
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch

from baseline_shop.report import read_only
from baseline_shop.validate import fingerprint
from .report import report
from .runner import DEFAULT, INITIAL, ROOT, configuration, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--initial',type=Path,default=INITIAL)
    args = parser.parse_args()
    cfg = configuration(DEFAULT)
    cfg.update(device=args.device,episodes=48,alternating_episodes=32,first_stop=32,
               segment_episodes=16,workers=4,processes=2,eval_episodes=4,heldout_episodes=4,
               promotion_minimum=4,rollout_rounds=16,checkpoint_rounds=2,action_cap=512,
               comparison_budget=512,candidates=4,batch_size=24,reward_rows=2048,
               reward_epochs=2,reward_support=1,probe_episodes=2)
    root = ROOT/'data'
    directory = Path(mkdtemp(prefix='adaptive_shop_validation_',dir=root))
    paths = [args.initial,*(root/subdir/file for subdir in ('baseline_relative_shop','alternating_ppo_dqn')
                            for file in ('latest.pt','training.sqlite3'))]
    preserved = [fingerprint(p) for p in paths]
    print(directory,flush=True)
    (directory/'settings.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    train(cfg,directory,args.initial)
    train(cfg,directory,args.initial,resume=True)

    def load(name):
        return torch.load(directory/name,map_location='cpu',weights_only=False)
    pre,post,last = (load(name) for name in ('episode-0000032-pre.pt','episode-0000032.pt','latest.pt'))
    assert last['state']['episodes']==last['state']['last_eval']==48
    assert pre['state']['baseline_version']==0
    assert all(torch.equal(pre['model'][key],post['model'][key]) for key in pre['model']
               if not key.startswith('reward_head.'))
    assert any(not torch.equal(pre['model'][key],post['model'][key]) for key in pre['model']
               if key.startswith('reward_head.'))
    assert all(torch.equal(last['shop'][key],post['shop'][key]) for key in last['shop'])
    assert last['state']['baseline_version']==post['state']['baseline_version']
    with read_only(directory/'training.sqlite3') as db:
        for table,mark in last['marks'].items():
            assert (db.execute(f'SELECT coalesce(max(id),0) FROM {table}').fetchone()[0])==mark
        counts = db.execute('SELECT source,count(*) FROM reward_experience GROUP BY source').fetchall()
        assert {s for s,_ in counts}=={'ppo','random'}
        for obs,mask,payload in db.execute('SELECT observation,legal,payload FROM reward_experience'):
            row = json.loads(payload)
            assert len(obs)==583*4 and len(mask)==5
            assert np.isclose(row['behavior_probability'],.8*row['probability']+.2/sum(mask))
            assert row['final_floor']>=row['floor']
        boundary = json.loads(db.execute('SELECT payload FROM boundaries WHERE step=32').fetchone()[0])
        assert boundary['transfer']['applied'] and boundary['transfer']['forced_training_counts']['Gold']>0
        assert db.execute("SELECT count(*) FROM episodes WHERE split='train'").fetchone()[0]==48
    print('Reward experience:',counts,flush=True)
    print('Boundary:',json.dumps(boundary),flush=True)
    print(report(directory),flush=True)
    assert preserved==[fingerprint(p) for p in paths]
    print('Verified 48 episodes, boundary transfer, final adaptation, resume and source preservation.')


if __name__=='__main__':
    main()

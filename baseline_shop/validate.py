"""Run a bounded local smoke test and benchmark from the real Phase 3B-2 model."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import mkdtemp

import torch

from .report import read_only, report
from .runner import DEFAULT, INITIAL, ROOT, configuration, train


def fingerprint(path):
    if path.is_file():
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, sha256(path.read_bytes()).hexdigest() if stat.st_size < 10**7 else None
    return None


def verify(directory):
    checkpoints = [torch.load(directory / f'episode-{step:07d}.pt', map_location='cpu', weights_only=False)
                   for step in (0, 16, 32, 48)]
    for left, right, active in zip(checkpoints, checkpoints[1:], ('model', 'shop', 'model')):
        for key in ('model', 'shop'):
            changed = any(not torch.equal(left[key][k], right[key][k]) for k in left[key])
            assert changed == (key == active), (key, active)
    db_path = directory / 'training.sqlite3'
    with read_only(db_path) as db:
        final = checkpoints[-1]
        for table, mark in final['marks'].items():
            assert db.execute(f'SELECT max(id) FROM {table}').fetchone()[0] == mark
        assert db.execute("SELECT count(DISTINCT step) FROM episodes WHERE split='train'").fetchone()[0] == 48
        labels = db.execute('SELECT label,count(*) FROM comparisons GROUP BY label').fetchall()
        assert any(n for label, n in labels if label in (-1, 1))
        assert db.execute("SELECT count(*) FROM evaluations WHERE step=48").fetchone()[0] == 5
        rewards = db.execute("SELECT payload FROM episodes WHERE split='train' AND step BETWEEN 17 AND 32")
        sources = {context['source'] for (payload,) in rewards for context in json.loads(payload)['reward_context']}
        assert sources == {'ppo', 'random'}
        assert sum(final['state']['trained']) > 0
        for name, value in db.execute('SELECT head,count(*) FROM comparisons GROUP BY head'):
            print(f'Head {name}: {value} paired labels')
    print('Verified three segments, frozen weights, 48 episodes, resume, label telemetry, and evaluation.')
    print('Shop updates by head:', final['state']['trained'], 'labels:', labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--initial', type=Path, default=INITIAL)
    args = parser.parse_args()
    cfg = configuration(DEFAULT)
    cfg.update(device=args.device, episodes=48, segment_episodes=16, workers=4, processes=2,
               eval_episodes=4, rollout_rounds=16, checkpoint_rounds=2,
               action_cap=512, comparison_budget=512, candidates=4, batch_size=24)
    root = ROOT / 'data'
    directory = Path(mkdtemp(prefix='baseline_shop_validation_', dir=root))
    files = [args.initial, root/'alternating_ppo_dqn/latest.pt', root/'alternating_ppo_dqn/training.sqlite3']
    before = [fingerprint(p) for p in files]
    print(directory, flush=True)
    (directory/'settings.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    train(cfg, directory, args.initial, stop=16)
    train(cfg, directory, args.initial, resume=True)
    verify(directory)
    print(report(directory/'training.sqlite3',directory/'report'))
    assert before == [fingerprint(p) for p in files]


if __name__ == '__main__':
    main()

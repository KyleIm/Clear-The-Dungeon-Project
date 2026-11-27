"""Run short isolated comparisons of equivalent Combat and Reward PPO runtimes."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

from .performance import Profile, Resources
from .runner import DEFAULT, config, runtime_settings, train


HYBRID = dict(inference_device='cpu', reuse_critic=True)
VARIANTS = {
    'cuda_baseline': ('cuda', {}),
    'cuda_reuse': ('cuda', dict(reuse_critic=True)),
    'cuda_bundled': ('cuda', dict(reuse_critic=True, bundle_inference=True)),
    'cpu_reuse': ('cpu', dict(reuse_critic=True)),
    'cpu_6x2': ('cpu', dict(reuse_critic=True, processes=6)),
    'cpu_4x3': ('cpu', dict(reuse_critic=True, processes=4)),
    'hybrid': ('cuda', HYBRID),
    'hybrid_6x2': ('cuda', dict(HYBRID, processes=6)),
    'hybrid_4x3': ('cuda', dict(HYBRID, processes=4)),
    'hybrid_6x2_batch6': ('cuda', dict(HYBRID, processes=6, inference_batch=6)),
    'hybrid_6x2_sqlite1024': ('cuda', dict(HYBRID, processes=6, sqlite_batch=1024)),
}


def warm_learner(device, cfg):
    """Initialize backend kernels on disposable weights before measured runs."""
    import numpy as np
    import torch

    from .ppo import ActorCritic, Transition, update

    torch.set_num_threads(cfg['cpu_threads'])
    model = ActorCritic(cfg['hidden_dims']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['ppo']['learning_rate'])
    count = cfg['ppo']['minibatch']
    features = np.zeros((count, 583), dtype=np.float32)
    phases = np.arange(count) % 2
    masks = np.ones((count, 7), dtype=np.bool_)
    masks[phases == 1, 5:] = False
    chosen, logs, values = model.act((features, phases, masks))
    rollout = [Transition(features[i], int(phases[i]), masks[i], int(chosen[i]), float(logs[i]),
                          float(values[i]), float(i % 2), 0.0, True, i) for i in range(count)]
    update(model, optimizer, rollout, cfg['ppo'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=list(VARIANTS))
    parser.add_argument('--detailed', action='store_true')
    args = parser.parse_args()
    base = config(args.config)
    if not 1 <= args.episodes <= 3000 or base['workers'] != 12:
        parser.error('Use 1–3,000 episodes and 12 logical game slots for this benchmark')
    if len(set(args.variants)) != len(args.variants):
        parser.error('Select each variant at most once')
    args.output.mkdir(parents=True, exist_ok=False)
    for device in sorted({VARIANTS[name][0] for name in args.variants}):
        warm_learner(device, base)
    for name in args.variants:
        device, runtime = VARIANTS[name]
        settings = deepcopy(base)
        settings.update(device=device, runtime=runtime, episodes=args.episodes)
        print(f'Running {name} | {args.episodes} episodes', flush=True)
        profile = Profile(detailed=args.detailed)
        started = perf_counter()
        with Resources() as resources:
            state = train(settings, args.output / name, profile=profile, benchmark=True)
        elapsed = perf_counter() - started
        result = dict(variant=name, arguments=runtime_settings(settings), update_device=device,
                      episodes=state['episodes'], actions=state['steps'], decisions=state['decisions'],
                      updates=state['updates'], training_seconds=state['seconds'], wall_seconds=elapsed,
                      episodes_per_second=state['episodes'] / state['seconds'],
                      actions_per_second=state['steps'] / state['seconds'], detailed=args.detailed,
                      phases=profile.summary(), resources=resources.summary())
        (args.output / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(f'{name}: {result["episodes_per_second"]:.2f} episodes/s | '
              f'{result["actions_per_second"]:.1f} actions/s', flush=True)


if __name__ == '__main__':
    main()

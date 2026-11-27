"""Train, evaluate and resume the local two-head PPO experiment."""

import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np

from experiments.random_baseline_large import run_lock
from .report import Store, generate_report
from .workers import Workers


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'combat_rl' / 'training.json'
VERSION = 1


def seed(master, stream, index=0):
    raw = hashlib.sha256(f'{master}:{stream}:{index}'.encode()).digest()
    return int.from_bytes(raw[:8], 'big') >> 1


def config(path):
    with path.open(encoding='utf-8') as file:
        cfg = json.load(file)
    for key in ('episodes', 'workers', 'action_cap', 'rollout_rounds', 'checkpoint_updates',
                'eval_every_episodes', 'eval_episodes', 'cpu_threads'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('minibatch', 'epochs'):
        if type(cfg['ppo'][key]) is not int or cfg['ppo'][key] < 1:
            raise ValueError(f'ppo.{key} must be a positive integer')
    if (not cfg['hidden_dims'] or any(type(width) is not int or width < 1 for width in cfg['hidden_dims'])
            or not 0 < cfg['ppo']['gamma'] <= 1 or not 0 <= cfg['ppo']['gae_lambda'] <= 1
            or not 0 < cfg['ppo']['clip'] < 1 or not cfg['ppo']['entropy'] >= 0
            or not 0 < cfg['ppo']['learning_rate'] < 1):
        raise ValueError('Invalid model or PPO settings')
    return cfg


def batch(observations):
    return tuple(np.stack(items) for items in zip(*observations))


def runtime_settings(cfg):
    runtime = dict(processes=cfg['workers'], inference_device=cfg['device'],
                   inference_batch=cfg['workers'], reuse_critic=False,
                   bundle_inference=False, sqlite_batch=256)
    runtime.update(cfg.get('runtime', {}))
    processes = runtime['processes']
    if (type(processes) is not int or processes < 1 or cfg['workers'] % processes
            or any(type(runtime[key]) is not int or runtime[key] < 1
                   for key in ('inference_batch', 'sqlite_batch'))):
        raise ValueError('Processes must divide the logical worker count and batch sizes must be positive')
    return runtime


def act_batch(actor, observations, runtime, profile):
    size = runtime['inference_batch']
    if len(observations) <= size:
        return actor.act(batch(observations), profile=profile, bundle=runtime['bundle_inference'])
    parts = [actor.act(batch(observations[start:start + size]), profile=profile,
                       bundle=runtime['bundle_inference']) for start in range(0, len(observations), size)]
    return tuple(np.concatenate(values) for values in zip(*parts))


def fill_bootstraps(rollout, pending, values):
    for slot, value in values.items():
        if slot in pending:
            index = pending.pop(slot)
            rollout[index] = rollout[index]._replace(next_value=float(value))


def seeds(cfg, split, index):
    master = cfg['master_seed']
    return seed(master, f'{split}-game', index), seed(master, f'{split}-policy', index), cfg['action_cap']


def evaluate(cfg, model, store, step):
    count = cfg['eval_episodes']
    size = min(cfg['workers'], count)
    for split in ('eval', 'random'):
        started = last_message = perf_counter()
        next_index = completed = actions = 0
        active = {}

        def refill(pool, slots):
            nonlocal next_index
            requests = {}
            for slot in slots:
                if next_index == count:
                    break
                requests[slot] = seeds(cfg, 'evaluation', next_index)
                next_index += 1
            active.update(pool.call('reset', requests))

        print(f'{split} evaluation at training episode {step:,} | {count} games', flush=True)
        with Workers(size) as pool:
            refill(pool, range(size))
            while active:
                if split == 'eval':
                    selected, _, _ = model.act(batch(active.values()), deterministic=True)
                    results = pool.call('step', dict(zip(active, map(int, selected))))
                else:
                    results = pool.call('random', dict.fromkeys(active))
                finished = []
                for slot, (_, delta, observation, result) in results.items():
                    actions += delta
                    if result is None:
                        active[slot] = observation
                    else:
                        store.episode(split, step, result)
                        completed += 1
                        del active[slot]
                        finished.append(slot)
                refill(pool, finished)
                if perf_counter() - last_message >= 2:
                    last_message = perf_counter()
                    print(f'{split}: {completed}/{count} games | {actions:,} actions | '
                          f'{last_message-started:.1f}s', flush=True)
        seconds = perf_counter() - started
        store.evaluation(step, split, dict(episodes=completed, actions=actions, seconds=seconds))
        print(f'{split}: {completed}/{count} games in {seconds:.1f}s', flush=True)


def load(path):
    import torch

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['version'] != VERSION:
        raise ValueError('Unsupported Combat/Reward checkpoint')
    return checkpoint


def train(cfg, directory, *, resume=False, baseline=None, profile=None, benchmark=False):
    import torch

    from .ppo import ActorCritic, Transition, update

    runtime = runtime_settings(cfg)
    measure = profile.measure if profile is not None else lambda _: nullcontext()

    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / 'latest.pt'
    database = directory / 'training.sqlite3'
    with run_lock(directory):
        if resume and not (checkpoint_path.is_file() and database.is_file()):
            raise ValueError('Resume needs latest.pt and training.sqlite3 from the same run')
        if not resume and (checkpoint_path.exists() or database.exists()):
            raise ValueError('Experiment exists. Resume it or choose another output directory.')
        old = load(checkpoint_path) if resume else None
        if old:
            fixed = ('workers', 'device', 'master_seed', 'action_cap', 'hidden_dims', 'eval_episodes')
            if any(old['config'][key] != cfg[key] for key in fixed):
                raise ValueError('Resume needs matching workers, device, master seed, action cap, model and evaluation panel size')
            if runtime_settings(old['config']) != runtime:
                raise ValueError('Resume requires the same runtime profile')
            if cfg['episodes'] < old['state']['started']:
                raise ValueError('Episode budget cannot be smaller than the number already started')
        torch.set_num_threads(cfg['cpu_threads'])
        device = torch.device(cfg['device'])
        inference_device = torch.device(runtime['inference_device'])
        if inference_device != device and inference_device.type != 'cpu':
            raise ValueError('Inference must use the learner device or CPU')
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use a CUDA PyTorch wheel or set device to cpu.')
        random.seed(seed(cfg['master_seed'], 'python'))
        np.random.seed(seed(cfg['master_seed'], 'numpy') % 2**32)
        torch.manual_seed(seed(cfg['master_seed'], 'torch'))
        model = ActorCritic(cfg['hidden_dims']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['ppo']['learning_rate'])
        state = dict(steps=0, decisions=0, episodes=0, started=0, updates=0,
                     seconds=0.0, next_eval=cfg['eval_every_episodes'], last_eval=-1)
        active = {}
        with (Store(database, batch_size=runtime['sqlite_batch']) as store,
              Workers(runtime['processes'], cfg['workers'] // runtime['processes'], profile) as pool):
            if old:
                state.update(old['state'])
                model.load_state_dict(old['model'])
                optimizer.load_state_dict(old['optimizer'])
                for group in optimizer.param_groups:
                    group['lr'] = cfg['ppo']['learning_rate']
                torch.set_rng_state(old['torch_rng'])
                if device.type == 'cuda':
                    torch.cuda.set_rng_state_all(old['cuda_rng'])
                random.setstate(old['python_rng'])
                np.random.set_state(old['numpy_rng'])
                store.restore(old['marks'])
                pool.call('restore', old['workers'])
                active.update(old['observations'])
            actor = model if inference_device == device else deepcopy(model).to(inference_device)

            def refill(slots):
                requests = {}
                for slot in slots:
                    if state['started'] == cfg['episodes']:
                        break
                    requests[slot] = seeds(cfg, 'training', state['started'])
                    state['started'] += 1
                if requests:
                    with measure('worker_reset'):
                        active.update(pool.call('reset', requests))

            def save():
                with measure('sqlite'):
                    marks = store.marks()
                with measure('checkpoint'):
                    payload = dict(version=VERSION, config=cfg, state=state, observations=active,
                                   workers=pool.call('snapshot', dict.fromkeys(active)), marks=marks,
                                   model=model.state_dict(), optimizer=optimizer.state_dict(),
                                   torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                                   numpy_rng=np.random.get_state(),
                                   cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
                    pending = checkpoint_path.with_suffix('.tmp')
                    torch.save(payload, pending)
                    pending.replace(checkpoint_path)

            refill(slot for slot in range(cfg['workers']) if slot not in active)
            if benchmark:
                with measure('warmup'):
                    actor.act(batch(active.values()), deterministic=True)
                    model.values(batch(active.values()))
            save()
            while True:
                due = state['episodes'] >= state['next_eval'] or not active
                if not benchmark and due and state['last_eval'] != state['episodes']:
                    save()
                    evaluate(cfg, model, store, state['episodes'])
                    state['last_eval'] = state['episodes']
                    while state['next_eval'] <= state['episodes']:
                        state['next_eval'] += cfg['eval_every_episodes']
                    save()
                if not active:
                    break
                started = perf_counter()
                starting_steps = state['steps']
                starting_episodes = state['episodes']
                rollout = []
                pending_values = {}
                for _ in range(cfg['rollout_rounds']):
                    slots = list(active)
                    observations = list(active.values())
                    selected, log_prob, values = act_batch(actor, observations, runtime, profile)
                    if pending_values:
                        fill_bootstraps(rollout, pending_values, dict(zip(slots, values)))
                    results = pool.call('step', dict(zip(slots, map(int, selected))))
                    next_values = {}
                    if not runtime['reuse_critic']:
                        ongoing = {slot: result[2] for slot, result in results.items() if result[3] is None}
                        if ongoing:
                            next_values = dict(zip(ongoing, actor.values(batch(ongoing.values()), profile=profile)))
                    finished = []
                    for position, slot in enumerate(slots):
                        reward, steps, observation, result = results[slot]
                        done = result is not None
                        rollout.append(Transition(*observations[position], int(selected[position]),
                                                  float(log_prob[position]), float(values[position]), reward,
                                                  float(next_values.get(slot, 0)), done, slot))
                        state['steps'] += steps
                        state['decisions'] += 1
                        if done:
                            state['episodes'] += 1
                            with measure('sqlite'):
                                store.episode('train', state['episodes'], result)
                            del active[slot]
                            finished.append(slot)
                        else:
                            active[slot] = observation
                            if runtime['reuse_critic']:
                                pending_values[slot] = len(rollout) - 1
                    refill(finished)
                    if not active or state['episodes'] >= state['next_eval']:
                        break
                if pending_values:
                    observations = batch([active[slot] for slot in pending_values])
                    fill_bootstraps(rollout, pending_values,
                                    dict(zip(pending_values, actor.values(observations, profile=profile))))
                metrics = update(model, optimizer, rollout, cfg['ppo'], profile=profile)
                if actor is not model:
                    with measure('actor_weight_sync'):
                        actor.load_state_dict(model.state_dict())
                duration = perf_counter() - started
                state['seconds'] += duration
                state['updates'] += 1
                rate = (state['steps'] - starting_steps) / duration
                episode_rate = (state['episodes'] - starting_episodes) / duration
                with measure('sqlite'):
                    store.update(state['episodes'], dict(state, **metrics, steps_per_second=rate,
                                                        episodes_per_second=episode_rate))
                if not benchmark:
                    print(f'{state["episodes"]:,}/{cfg["episodes"]:,} episodes | {state["steps"]:,} actions | '
                          f'{rate:,.1f} actions/s | {episode_rate:.1f} episodes/s | {state["seconds"]:.1f}s train',
                          flush=True)
                if state['updates'] % cfg['checkpoint_updates'] == 0:
                    save()
            if benchmark:
                save()
                return state
        return generate_report(database, directory / 'report', baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'resume', 'report'))
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'data/baseline_random_1m/baseline.sqlite3')
    args = parser.parse_args()
    cfg = config(args.config)
    directory = args.output or ROOT / cfg['output']
    baseline = args.baseline if args.baseline.is_file() else None
    if args.command == 'report':
        result = generate_report(directory / 'training.sqlite3', directory / 'report', baseline)
    else:
        result = train(cfg, directory, resume=args.command == 'resume', baseline=baseline)
    print(result)


if __name__ == '__main__':
    main()

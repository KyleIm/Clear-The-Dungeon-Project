"""Standalone alternating learner with exact, restartable episode boundaries."""

import argparse
from collections import deque
from copy import deepcopy
import json
import logging
from pathlib import Path
import random
from time import perf_counter

import numpy as np

from combat_rl.performance import Profile, Resources
from combat_rl.runner import (ROOT, act_batch, batch, fill_bootstraps, load as load_combat, seed)
from combat_rl.workers import Workers
from experiments.random_baseline_large import run_lock
from .environment import Run
from .report import Store, generate_report


DEFAULT = ROOT / 'alternating_rl' / 'training.json'
VERSION = 1


def read_config(path):
    cfg = json.loads(path.read_text(encoding='utf-8'))
    for name in ('episodes', 'segment_episodes', 'eval_episodes', 'workers', 'processes',
                 'action_cap', 'checkpoint_rounds', 'rollout_rounds'):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if cfg['episodes'] % cfg['segment_episodes'] or cfg['workers'] % cfg['processes']:
        raise ValueError('Episode budget must divide into whole segments and processes must divide game slots')
    shop = cfg['shop']
    for key in ('candidates', 'max_edits', 'batch_size', 'replay_window', 'warmup_updates'):
        if type(shop[key]) is not int or shop[key] < 1:
            raise ValueError(f'shop.{key} must be a positive integer')
    if (not 0 <= shop['epsilon'] <= 1 or not 0 < shop['learning_rate'] < 1
            or not shop['hidden_dims'] or any(type(x) is not int or x < 1 for x in shop['hidden_dims'])):
        raise ValueError('Invalid shop learner settings')
    if shop.get('planner', 'uniform_menu') not in ('uniform_menu', 'uniform_action'):
        raise ValueError('Unknown shop candidate planner')
    return cfg


def segment(episodes, length):
    index = episodes // length
    return index, index // 2 + 1, 'shop' if index % 2 else 'combat_reward'


def pack(run, reward=0.0, delta=0):
    return (reward, delta, run.phase, None if run.done or run.phase == 'shop' else run.observation(),
            run.result() if run.done else None, run.drain_samples())


def advance_worker(runs, slot, op, arg):
    if op == 'reset':
        args, options = arg
        run = runs[slot] = Run(*args, **options)
        return pack(run)
    if op == 'restore':
        runs[slot] = arg
        return None
    run = runs[slot]
    if op == 'snapshot':
        return run
    if op == 'candidates':
        return run.generate_candidates()
    before = run.actions
    if op == 'step':
        action, version = arg
        run.version = version
        reward = run.step(action)
    elif op == 'shop_random':
        reward = run.choose_random_shop()
    elif op == 'shop_apply':
        reward = run.choose_shop(*arg)
    else:
        raise ValueError(f'Unknown alternating worker operation {op}')
    return pack(run, reward, run.actions - before)


def bridge(pool, replies, shop_model, profile, rng, *, random_shop, epsilon, trained):
    pending = [slot for slot, reply in replies.items() if reply[4] is None and reply[2] == 'shop']
    if not pending:
        return replies
    if random_shop:
        with profile.measure('shop_random'):
            processed = pool.call('shop_random', dict.fromkeys(pending))
    else:
        with profile.measure('candidate_generation'):
            proposals = pool.call('candidates', dict.fromkeys(pending))
        with profile.measure('shop_inference'):
            import torch

            packed = np.concatenate([proposals[slot] for slot in pending])
            with torch.inference_mode():
                probabilities = shop_model(packed).cpu().numpy()
        choices = {}
        offset = 0
        for slot in pending:
            count = len(proposals[slot])
            scores = probabilities[offset:offset + count]
            explore = not trained or rng.random() < epsilon
            choice = rng.randrange(count) if explore else int(scores.argmax())
            choices[slot] = (choice, float(scores[choice]) if trained else None, explore)
            offset += count
        with profile.measure('shop_application'):
            processed = pool.call('shop_apply', choices)
    for slot, (reward, delta, phase, observation, result, samples) in processed.items():
        earlier_reward, earlier_steps, _, _, _, earlier_samples = replies[slot]
        replies[slot] = (earlier_reward + reward, earlier_steps + delta, phase, observation, result,
                         earlier_samples + samples)
    if any(item[4] is None and item[2] == 'shop' for item in replies.values()):
        raise RuntimeError('Shop bridge did not advance to a decision or episode end')
    return replies


class Experiment:
    def __init__(self, cfg, initial, old, store, profile, resources, directory):
        import torch

        from combat_rl.ppo import ActorCritic
        from .shop_value import ShopValue

        self.torch = torch
        self.cfg, self.initial, self.store = cfg, initial, store
        self.profile, self.resources, self.directory = profile, resources, directory
        self.started_at = perf_counter()
        self.device = torch.device(cfg['device'])
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in the project environment')
        torch.set_num_threads(initial['config']['cpu_threads'])
        torch.manual_seed(seed(cfg['master_seed'], 'alternating-torch'))
        self.rng = random.Random(seed(cfg['master_seed'], 'alternating-exploration'))
        self.data_rng = np.random.default_rng(seed(cfg['master_seed'], 'alternating-samples'))
        self.model = ActorCritic([128, 128]).to(self.device)
        self.combat_optimizer = torch.optim.Adam(self.model.parameters(), lr=initial['config']['ppo']['learning_rate'])
        self.shop = ShopValue(cfg['shop']['hidden_dims']).to('cpu')
        self.shop_optimizer = torch.optim.Adam(self.shop.parameters(), lr=cfg['shop']['learning_rate'])
        self.state = dict(episodes=0, started=0, steps=0, decisions=0, rounds=0, version=0,
                          shop_updates=0, train_seconds=0.0, wall_seconds=0.0, last_eval=-1)
        self.active = {}
        self.old = old
        self.samples = deque(maxlen=cfg['shop']['replay_window'])
        self.last_data_segment = None
        self.resource_summary = {}
        self.model.load_state_dict((old or initial)['model'])
        self.combat_optimizer.load_state_dict((old or initial)['combat_optimizer'] if old else initial['optimizer'])
        if old:
            self.shop.load_state_dict(old['shop'])
            self.shop_optimizer.load_state_dict(old['shop_optimizer'])
            self.state.update(old['state'])
            self.active.update(old['observations'])
            torch.set_rng_state(old['torch_rng'])
            if self.device.type == 'cuda':
                torch.cuda.set_rng_state_all(old['cuda_rng'])
            self.rng.setstate(old['exploration_rng'])
            self.data_rng.bit_generator.state = old['data_rng']
            self.store.restore(old['marks'])
        self.prior_wall = self.state['wall_seconds']
        self.actor = self.model if self.device.type == 'cpu' else deepcopy(self.model).to('cpu')
        self.runtime = dict(inference_batch=cfg['workers'], bundle_inference=False)

    def pool(self):
        return Workers(self.cfg['processes'], self.cfg['workers'] // self.cfg['processes'],
                       self.profile, apply_handler=advance_worker)

    def request(self, split, index, cycle, random_shop):
        master = self.cfg['master_seed'] if split == 'train' else self.cfg['evaluation_seed']
        prefix = 'alternating-training' if split == 'train' else 'evaluation'
        shop = self.cfg['shop']
        args = (seed(master, f'{prefix}-game', index), seed(master, f'{prefix}-policy', index),
                self.cfg['action_cap'], shop['candidates'], cycle, self.state['version'])
        return args, dict(random_shop=random_shop, max_edits=shop['max_edits'],
                          planner=shop.get('planner', 'uniform_menu'))

    def refill(self, pool, slots, boundary, cycle, random_shop):
        requests = {}
        for slot in slots:
            if self.state['started'] >= boundary:
                break
            requests[slot] = self.request('train', self.state['started'], cycle, random_shop)
            self.state['started'] += 1
        with self.profile.measure('worker_reset'):
            self.active.update({slot: reply[3] for slot, reply in pool.call('reset', requests).items()})

    def save(self, pool, archive=False):
        number, cycle, target = segment(self.state['episodes'], self.cfg['segment_episodes'])
        if self.state['episodes'] == self.cfg['episodes']:
            schedule = dict(segment=number, cycle=(number + 1) // 2, target='complete')
        else:
            schedule = dict(segment=number + 1, cycle=cycle, target=target)
        self.state['wall_seconds'] = self.prior_wall + perf_counter() - self.started_at
        payload = dict(version=VERSION, config=self.cfg, combat_config=self.initial['config'],
                       schedule=schedule,
                       state=self.state, observations=self.active,
                       workers=pool.call('snapshot', dict.fromkeys(self.active)), marks=self.store.marks(),
                       model=self.model.state_dict(), shop=self.shop.state_dict(),
                       combat_optimizer=self.combat_optimizer.state_dict(),
                       shop_optimizer=self.shop_optimizer.state_dict(),
                       torch_rng=self.torch.get_rng_state(),
                       cuda_rng=self.torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None,
                       exploration_rng=self.rng.getstate(), data_rng=self.data_rng.bit_generator.state)
        pending = self.directory / 'latest.tmp'
        with self.profile.measure('checkpoint'):
            self.torch.save(payload, pending)
            pending.replace(self.directory / 'latest.pt')
            if archive:
                import shutil

                checkpoint = self.directory / f'episode-{self.state["episodes"]:07d}.pt'
                temp = checkpoint.with_suffix('.tmp')
                shutil.copyfile(self.directory / 'latest.pt', temp)
                temp.replace(checkpoint)

    def evaluate(self, pool, step):
        cfg = self.cfg
        started = perf_counter()
        count = cfg['eval_episodes']
        index = completed = actions = 0
        active = {}
        cycle = max(1, (max(0, step - 1) // cfg['segment_episodes']) // 2 + 1)
        random_shop = step <= cfg['segment_episodes']
        rng = random.Random(seed(cfg['evaluation_seed'], 'evaluation-exploration'))

        def refill(slots):
            nonlocal index
            requests = {}
            for slot in slots:
                if index == count:
                    break
                requests[slot] = self.request('eval', index, cycle, random_shop)
                index += 1
            active.update({slot: reply[3] for slot, reply in pool.call('reset', requests).items()})

        refill(range(cfg['workers']))
        profile = Profile()
        while active:
            chosen, _, _ = self.actor.act(batch(active.values()), deterministic=True)
            requests = {slot: (int(action), self.state['version']) for slot, action in zip(active, chosen)}
            replies = bridge(pool, pool.call('step', requests), self.shop, profile, rng,
                             random_shop=random_shop, epsilon=0, trained=self.state['shop_updates'] > 0)
            finished = []
            for slot, (_, delta, _, observation, result, samples) in replies.items():
                actions += delta
                for sample in samples:
                    self.store.sample('eval', step, sample)
                if result is None:
                    active[slot] = observation
                else:
                    self.store.episode('eval', step, dict(result, segment=(step + cfg['segment_episodes'] - 1)
                                                          // cfg['segment_episodes']))
                    del active[slot]
                    finished.append(slot)
                    completed += 1
            refill(finished)
        self.store.evaluation(step, 'eval', dict(episodes=completed, actions=actions,
                                                seconds=perf_counter() - started, **profile.summary()))
        self.state['last_eval'] = step
        print(f'Evaluation {step:,} | {completed}/{count} games | {actions:,} actions', flush=True)

    def learn_shop(self, iterations, segment_index):
        from .shop_value import update_shop

        if segment_index != self.last_data_segment:
            with self.profile.measure('shop_replay_read'):
                recent = self.store.recent_samples(self.cfg['shop']['replay_window'], self.state['version'])
            self.samples.clear()
            self.samples.extend(recent)
            self.last_data_segment = segment_index
        if not self.samples:
            return {}
        metrics = {}
        with self.profile.measure('shop_learning'):
            for _ in range(iterations):
                metrics = update_shop(self.shop, self.shop_optimizer, self.samples,
                                      self.cfg['shop']['batch_size'], self.data_rng)
                self.state['shop_updates'] += 1
        return metrics

    def rounds(self, pool, boundary, cycle, target, random_shop):
        from combat_rl.ppo import Transition, update

        train_combat = target == 'combat_reward'
        rollout, pending = [], {}
        starting_actions, starting_episodes = self.state['steps'], self.state['episodes']
        started = perf_counter()
        for _ in range(self.cfg['rollout_rounds']):
            slots = list(self.active)
            observations = list(self.active.values())
            chosen, logs, values = act_batch(self.actor, observations, self.runtime, self.profile)
            if pending:
                fill_bootstraps(rollout, pending, dict(zip(slots, values)))
            requests = {slot: (int(action), self.state['version']) for slot, action in zip(slots, chosen)}
            replies = bridge(pool, pool.call('step', requests), self.shop, self.profile,
                             self.rng, random_shop=random_shop, epsilon=0 if train_combat else self.cfg['shop']['epsilon'],
                             trained=self.state['shop_updates'] > 0)
            finished = []
            for i, slot in enumerate(slots):
                reward, delta, _, observation, result, samples = replies[slot]
                self.state['steps'] += delta
                self.state['decisions'] += 1
                for sample in samples:
                    self.store.sample('train', self.state['episodes'], sample)
                    if not train_combat and sample['target'] is not None:
                        self.samples.appendleft(sample)
                if train_combat:
                    rollout.append(Transition(*observations[i], int(chosen[i]), float(logs[i]),
                                              float(values[i]), reward, 0, result is not None, slot))
                    if result is None:
                        pending[slot] = len(rollout) - 1
                if result is None:
                    self.active[slot] = observation
                else:
                    self.state['episodes'] += 1
                    self.store.episode('train', self.state['episodes'],
                                       dict(result, segment=boundary // self.cfg['segment_episodes'],
                                            training_target=target))
                    del self.active[slot]
                    finished.append(slot)
            self.refill(pool, finished, boundary, cycle, random_shop)
            if not self.active:
                break
        if train_combat:
            if pending:
                values = self.actor.values(batch([self.active[slot] for slot in pending]))
                fill_bootstraps(rollout, pending, dict(zip(pending, values)))
            metrics = update(self.model, self.combat_optimizer, rollout, self.initial['config']['ppo'],
                             profile=self.profile)
            self.state['version'] += 1
            if self.actor is not self.model:
                self.actor.load_state_dict(self.model.state_dict())
        else:
            metrics = self.learn_shop(1, boundary // self.cfg['segment_episodes'] - 1)
        seconds = perf_counter() - started
        self.state['train_seconds'] += seconds
        self.state['rounds'] += 1
        if self.state['rounds'] % 64 == 1 or not self.active:
            self.resource_summary = self.resources.summary()
        rate = (self.state['episodes'] - starting_episodes) / seconds
        action_rate = (self.state['steps'] - starting_actions) / seconds
        metrics.update(self.state, cycle=cycle, target=target, episodes_per_second=rate,
                       actions_per_second=action_rate, elapsed_seconds=self.prior_wall + perf_counter() - self.started_at,
                       phases=self.profile.summary(), resources=self.resource_summary)
        self.store.update(self.state['episodes'], metrics)
        print(f'{self.state["episodes"]:,}/{self.cfg["episodes"]:,} | cycle {cycle} {target} | '
              f'{rate:.1f} episodes/s | {action_rate:.0f} actions/s', flush=True)

    def run(self):
        cfg = self.cfg
        length = cfg['segment_episodes']
        with self.pool() as train_pool:
            if self.old:
                train_pool.call('restore', self.old['workers'])
            while True:
                completed = self.state['episodes']
                if completed % length == 0 and self.state['last_eval'] != completed:
                    assert not self.active
                    self.save(train_pool)
                    with self.pool() as eval_pool:
                        self.evaluate(eval_pool, completed)
                    self.save(train_pool, archive=True)
                if completed == cfg['episodes']:
                    return
                index, cycle, target = segment(completed, length)
                boundary = (index + 1) * length
                assert self.state['started'] <= boundary
                if not self.active and target == 'shop' and self.state['shop_updates'] == 0:
                    warmup = self.learn_shop(cfg['shop']['warmup_updates'], index)
                    if warmup:
                        self.store.update(completed, dict(warmup, cycle=cycle, target='shop_warmup'))
                self.refill(train_pool, (slot for slot in range(cfg['workers']) if slot not in self.active),
                            boundary, cycle, index == 0)
                self.rounds(train_pool, boundary, cycle, target, index == 0)
                if self.state['rounds'] % cfg['checkpoint_rounds'] == 0:
                    self.save(train_pool)


def train(cfg, directory, initial_path, *, resume=False, baseline=None):
    import torch

    if directory.resolve() == initial_path.resolve().parent:
        raise ValueError('Choose a new output directory outside the initial checkpoint directory')
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / 'training.sqlite3'
    checkpoint = directory / 'latest.pt'
    with run_lock(directory):
        if resume and not (database.is_file() and checkpoint.is_file()):
            raise ValueError('Resume requires this experiment’s database and checkpoint')
        if not resume and (database.exists() or checkpoint.exists()):
            raise ValueError('Experiment exists. Resume it or select a new directory.')
        old = torch.load(checkpoint, map_location='cpu', weights_only=False) if resume else None
        if old and (old['version'] != VERSION or old['config'] != cfg):
            raise ValueError('Use the same alternating experiment configuration to resume')
        initial = {'config': old['combat_config']} if old else load_combat(initial_path)
        if not old and (initial['state']['episodes'] != 1_000_000 or initial['config']['hidden_dims'] != [128, 128]):
            raise ValueError('The Phase 3B-2 final million-episode 128/128 checkpoint is required')
        with Resources() as resources, Store(database, batch_size=1024) as store:
            experiment = Experiment(cfg, initial, old, store, Profile(), resources, directory)
            experiment.run()
        return generate_report(database, directory / 'report', baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'resume', 'report'])
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--initial', type=Path, default=ROOT / 'data/combat_reward_rl_optimized/latest.pt')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'data/baseline_random_1m/baseline.sqlite3')
    args = parser.parse_args()
    cfg = read_config(args.config)
    directory = args.output or ROOT / cfg['output']
    baseline = args.baseline if args.baseline.is_file() else None
    if args.command == 'report':
        result = generate_report(directory / 'training.sqlite3', directory / 'report', baseline)
    else:
        directory.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=directory / 'errors.log', encoding='utf-8', level=logging.ERROR)
        try:
            result = train(cfg, directory, args.initial, resume=args.command == 'resume', baseline=baseline)
        except (Exception, KeyboardInterrupt):
            logging.exception('Alternating run interrupted')
            raise
    print(result)


if __name__ == '__main__':
    main()

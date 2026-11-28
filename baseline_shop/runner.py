"""Independent local baseline-relative Shop and Combat/Reward PPO experiment."""

import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path
from random import Random
from shutil import copyfile
from time import perf_counter
from uuid import uuid4

from combat_rl.performance import Profile, Resources
from combat_rl.runner import ROOT, batch, fill_bootstraps, load as load_ppo, seed
from combat_rl.workers import Workers
from experiments.random_baseline_large import run_lock
from .configurations import HEADS
from .storage import Store
from .workers import bridge, handle

DEFAULT = ROOT / 'baseline_shop/training.json'
INITIAL = ROOT / 'data/combat_reward_rl_optimized/latest.pt'
VERSION = 'baseline-shop-v1'
HISTORICAL = tuple(ROOT / 'data' / name for name in (
    'alternating_ppo_dqn', 'alternating_combat_shop', 'combat_reward_rl',
    'combat_reward_rl_optimized', 'baseline_random_1m'))


def configuration(path):
    cfg = json.loads(path.read_text(encoding='utf-8'))
    for k in ('episodes', 'segment_episodes', 'workers', 'processes', 'rollout_rounds',
              'checkpoint_rounds', 'eval_episodes', 'action_cap', 'comparison_budget',
              'candidates', 'class_window', 'batch_size'):
        if type(cfg[k]) is not int or cfg[k] < 1:
            raise ValueError(f'{k} must be a positive integer')
    if (cfg['episodes'] % cfg['segment_episodes'] or cfg['workers'] % cfg['processes']
            or cfg['candidates'] < 2 or not 3 <= cfg['batch_size'] <= 256
            or cfg['device'] not in ('cpu', 'cuda') or not 0 < cfg['shop_lr'] < 1):
        raise ValueError('Inconsistent baseline-relative configuration')
    return cfg


def mode_for(completed, length):
    segment = completed // length
    return 'uniform' if segment == 0 else 'paired' if segment % 2 else 'combined'


def require_isolated_output(directory, initial_path):
    target = directory.resolve()
    protected = (*HISTORICAL, initial_path.resolve().parent)
    if any(target.is_relative_to(path.resolve()) for path in protected):
        raise ValueError('Select an output directory outside historical experiments and the initial checkpoint')


class Experiment:
    def __init__(self, cfg, initial, old, store, directory, resources):
        import torch
        from combat_rl.ppo import ActorCritic
        from .model import ConfigurationValue

        self.torch = torch
        self.cfg, self.directory, self.store, self.resources = cfg, directory, store, resources
        self.profile = Profile()
        self.started = perf_counter()
        self.ppo_config = (old['ppo_config'] if old else initial['config'])
        torch.set_num_threads(self.ppo_config['cpu_threads'])
        torch.manual_seed(seed(cfg['master_seed'], 'torch'))
        self.rng = Random(seed(cfg['master_seed'], 'shop-sample'))
        device = torch.device(cfg['device'])
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in the project environment')
        self.ppo = ActorCritic(self.ppo_config['hidden_dims']).to(device)
        self.shop = ConfigurationValue().to(device)
        self.ppo_opt = torch.optim.Adam(self.ppo.parameters(), lr=self.ppo_config['ppo']['learning_rate'])
        self.shop_opt = torch.optim.Adam(self.shop.parameters(), lr=cfg['shop_lr'])
        self.active = {}
        self.state = dict(episodes=0, started=0, actions=0, rounds=0, version=0,
                          trained=[0, 0, 0], last_eval=-1, train_seconds=0., wall_seconds=0.)
        self.run_id = old['run_id'] if old else uuid4().hex
        store.identity(self.run_id, bool(old))
        self.ppo.load_state_dict((old or initial)['model'])
        self.ppo_opt.load_state_dict(old['ppo_opt'] if old else initial['optimizer'])
        if old:
            self.active.update(old['active'])
            self.state.update(old['state'])
            self.shop.load_state_dict(old['shop'])
            self.shop_opt.load_state_dict(old['shop_opt'])
            torch.set_rng_state(old['torch_rng'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(old['cuda_rng'])
            self.rng.setstate(old['rng'])
            store.restore(old['marks'])
            self.profile.totals.update(old['profile_totals'])
            self.profile.counts.update(old['profile_counts'])
        self.old = old
        self.prior_wall = self.state['wall_seconds']
        self.actor = self.ppo if device.type == 'cpu' else deepcopy(self.ppo).to('cpu')
        self.shop_actor = self.shop if device.type == 'cpu' else deepcopy(self.shop).to('cpu')

    def pool(self):
        return Workers(self.cfg['processes'], self.cfg['workers'] // self.cfg['processes'],
                       self.profile, apply_handler=handle)

    def request(self, split, number, mode):
        master = self.cfg['evaluation_seed' if split == 'evaluation' else 'master_seed']
        args = seed(master, split + '-game', number), seed(master, split + '-policy', number), self.cfg['action_cap']
        return args, dict(mode=mode, version=self.state['version'])

    def save(self, pool, archive=False):
        torch = self.torch
        self.state['wall_seconds'] = self.prior_wall + perf_counter() - self.started
        active_workers = pool.call('snapshot', dict.fromkeys(self.active))
        with self.profile.measure('checkpoint'):
            payload = dict(version=VERSION, run_id=self.run_id, config=self.cfg, ppo_config=self.ppo_config,
                           state=self.state, active=self.active, workers=active_workers, marks=self.store.marks(),
                           model=self.ppo.state_dict(), ppo_opt=self.ppo_opt.state_dict(),
                           shop=self.shop.state_dict(), shop_opt=self.shop_opt.state_dict(), rng=self.rng.getstate(),
                           profile_totals=dict(self.profile.totals), profile_counts=dict(self.profile.counts),
                           torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()
                           if self.cfg['device'] == 'cuda' else None)
            temp = self.directory / 'latest.tmp'
            torch.save(payload, temp)
            latest = self.directory / 'latest.pt'
            temp.replace(latest)
            if archive:
                dest = self.directory / f"episode-{self.state['episodes']:07d}.pt"
                copyfile(latest, temp)
                temp.replace(dest)

    def choose(self, active, deterministic):
        with self.profile.measure('ppo_inference'), self.torch.inference_mode():
            inputs = self.actor._inputs(batch(active.values()))
            dist, value = self.actor(*inputs)
            chosen = dist.logits.argmax(dim=-1) if deterministic else dist.sample()
            probs = dist.probs[:, :5].numpy()
            logs = dist.log_prob(chosen).numpy()
            return chosen.numpy(), logs, value.numpy(), probs

    def step(self, pool, active, mode, deterministic):
        choices, logs, values, probs = self.choose(active, deterministic)
        requests = {slot: (int(action), probability, self.state['version'])
                    for slot, action, probability in zip(active, choices, probs)}
        replies = pool.call('step', requests)
        replies, labels = bridge(pool, replies, self.shop_actor, self.state['trained'], mode, self.cfg, self.profile)
        return replies, labels, choices, logs, values

    def rollout(self, pool, boundary, mode):
        from combat_rl.ppo import Transition, update
        from .model import fit

        start = perf_counter()
        initial_actions, initial_count = self.state['actions'], self.state['episodes']
        trajectory, pending = [], {}
        for _ in range(self.cfg['rollout_rounds']):
            if not self.active:
                break
            slots = list(self.active)
            replies, labels, choices, logs, values = self.step(pool, self.active, mode, False)
            if pending:
                fill_bootstraps(trajectory, pending, dict(zip(slots, values)))
            for head, features, row in labels:
                with self.profile.measure('sqlite_comparison'):
                    self.store.comparison(self.state['episodes'], self.state['version'], head, features,
                                          dict(row, segment=boundary))
            finished = []
            for i, slot in enumerate(slots):
                reply = replies[slot]
                done = reply.result is not None
                self.state['actions'] += reply.actions
                if mode != 'paired':
                    trajectory.append(Transition(*self.active[slot], int(choices[i]), float(logs[i]),
                                                 float(values[i]), reply.reward, 0, done, slot))
                    if not done:
                        pending[slot] = len(trajectory) - 1
                if done:
                    self.state['episodes'] += 1
                    with self.profile.measure('sqlite_episode'):
                        self.store.episode('train', self.state['episodes'], dict(reply.result, segment=boundary))
                    del self.active[slot]
                    finished.append(slot)
                else:
                    self.active[slot] = reply.observation
            self.refill(pool, finished, boundary, mode)
        metrics = {}
        if mode == 'paired':
            for head in range(len(HEADS)):
                with self.profile.measure('paired_sample'):
                    sample = self.store.sample(self.state['version'], head, self.cfg['batch_size'],
                                               self.cfg['class_window'], self.rng)
                if sample:
                    with self.profile.measure('fit_' + HEADS[head]):
                        metrics['loss_' + HEADS[head]] = fit(self.shop, self.shop_opt, head, sample)
                    self.state['trained'][head] += 1
            if metrics and self.shop_actor is not self.shop:
                self.shop_actor.load_state_dict(self.shop.state_dict())
        else:
            if pending:
                with self.profile.measure('critic_final_bootstrap'):
                    last_values = self.actor.values(batch(self.active[slot] for slot in pending))
                fill_bootstraps(trajectory, pending, dict(zip(pending, last_values)))
            metrics.update(update(self.ppo, self.ppo_opt, trajectory, self.ppo_config['ppo'], profile=self.profile))
            if self.actor is not self.ppo:
                self.actor.load_state_dict(self.ppo.state_dict())
            self.state['version'] += 1
        seconds = perf_counter() - start
        self.state['rounds'] += 1
        self.state['train_seconds'] += seconds
        eps = (self.state['episodes'] - initial_count) / seconds
        aps = (self.state['actions'] - initial_actions) / seconds
        self.store.update(self.state['episodes'], dict(self.state, **metrics, mode=mode, segment=boundary,
                          episodes_per_second=eps, actions_per_second=aps, profile=self.profile.summary(),
                          resources=self.resources.summary() if self.state['rounds'] % 32 == 1 or not self.active else {}))
        print(f"{self.state['episodes']:,}/{self.cfg['episodes']:,} | {mode} | {eps:.1f} episodes/s | {aps:.0f} actions/s",
              flush=True)

    def refill(self, pool, slots, boundary, mode):
        requests = {}
        for slot in slots:
            index = self.state['started']
            if index >= boundary:
                break
            requests[slot] = self.request('training', index, mode)
            self.state['started'] += 1
        self.active.update({slot: reply.observation for slot, reply in pool.call('reset', requests).items()})

    def evaluation(self, pool, step):
        count = self.cfg['eval_episodes']
        modes = ('baseline', *HEADS, 'combined')
        if step <= self.cfg['segment_episodes']:
            modes = ('uniform', *modes)
        for mode in modes:
            active = {}
            started = completed = actions = 0
            begin = perf_counter()

            def refill(slots):
                nonlocal started
                requests = {}
                for slot in slots:
                    if started >= count:
                        break
                    requests[slot] = self.request('evaluation', started, mode)
                    started += 1
                active.update({slot: reply.observation for slot, reply in pool.call('reset', requests).items()})

            refill(range(self.cfg['workers']))
            while active:
                replies, _, _, _, _ = self.step(pool, active, mode, True)
                finished = []
                for slot, reply in replies.items():
                    actions += reply.actions
                    if reply.result is None:
                        active[slot] = reply.observation
                    else:
                        self.store.episode('eval_' + mode, step, dict(reply.result, segment=step))
                        completed += 1
                        finished.append(slot)
                        del active[slot]
                refill(finished)
            self.store.evaluation(step, mode, dict(episodes=completed, actions=actions,
                                                   seconds=perf_counter() - begin))
            print(f'Evaluation {step:,} | {mode} | {completed}/{count} | {actions:,} actions', flush=True)

    def run(self, stop):
        length = self.cfg['segment_episodes']
        with self.pool() as pool:
            if self.old:
                pool.call('restore', self.old['workers'])
            published = None
            while True:
                complete = self.state['episodes']
                if complete % length == 0 and self.state['last_eval'] != complete:
                    assert not self.active
                    self.save(pool)
                    # Evaluation does not consume the training Torch RNG or run-level RNG.
                    training_profile = self.profile
                    try:
                        self.profile = pool.profile = Profile()
                        self.evaluation(pool, complete)
                    finally:
                        self.profile = pool.profile = training_profile
                    self.state['last_eval'] = complete
                    self.save(pool, archive=True)
                if complete >= stop:
                    return
                mode = mode_for(complete, length)
                boundary = (complete // length + 1) * length
                if mode == 'paired' and published != self.state['version']:
                    weights = {k: value.detach().numpy().copy() for k, value in self.actor.state_dict().items()}
                    pool.call('policy', dict.fromkeys(range(self.cfg['processes']), weights))
                    published = self.state['version']
                self.refill(pool, (slot for slot in range(self.cfg['workers']) if slot not in self.active), boundary, mode)
                self.rollout(pool, boundary, mode)
                if self.state['rounds'] % self.cfg['checkpoint_rounds'] == 0:
                    self.save(pool)


def train(cfg, directory, initial_path, resume=False, stop=None):
    import torch

    require_isolated_output(directory, initial_path)
    stop = cfg['episodes'] if stop is None else stop
    if not 0 < stop <= cfg['episodes'] or stop % cfg['segment_episodes']:
        raise ValueError('--stop-at must be a positive segment boundary within the budget')
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint, db_path = directory / 'latest.pt', directory / 'training.sqlite3'
    with run_lock(directory):
        if resume:
            if not checkpoint.is_file() or not db_path.is_file():
                raise ValueError('Resume needs both the checkpoint and its SQLite database')
            old = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if old.get('version') != VERSION or old.get('config') != cfg:
                raise ValueError('Resume needs the baseline-relative checkpoint and identical settings')
            if old['state']['started'] > stop:
                raise ValueError('The stop boundary precedes already-started episodes')
            initial = None
        else:
            if checkpoint.exists() or db_path.exists():
                raise ValueError('This run exists. Resume it or select another directory')
            old, initial = None, load_ppo(initial_path)
            if initial['state']['episodes'] != 1_000_000 or initial['config']['hidden_dims'] != [128, 128]:
                raise ValueError('Expected the final Phase 3B-2 PPO checkpoint')
        with Store(db_path, batch_size=1024) as store, Resources() as resources:
            Experiment(cfg, initial, old, store, directory, resources).run(stop)


def main():
    from .report import report
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'resume', 'report'))
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--initial', type=Path, default=INITIAL)
    parser.add_argument('--stop-at', type=int)
    args = parser.parse_args()
    cfg = configuration(args.config)
    directory = args.output or ROOT / cfg['output']
    require_isolated_output(directory, args.initial)
    if args.command != 'report':
        directory.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=directory / 'errors.log', encoding='utf-8', level=logging.ERROR)
        try:
            train(cfg, directory, args.initial, args.command == 'resume', args.stop_at)
        except (Exception, KeyboardInterrupt):
            logging.exception('Baseline-relative run interrupted')
            raise
    print(report(directory / 'training.sqlite3', directory / 'report'))


if __name__ == '__main__':
    main()

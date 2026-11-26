"""Standalone local training, validation, evaluation and reporting commands."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import logging
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from experiments.random_baseline_large import run_lock
from .environment import DungeonEnv
from .reporting import EpisodeStats, ExperimentStore, Telemetry, generate_report
from .vector import open_vector


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'rl' / 'training.json'
CHECKPOINT_VERSION = 1


def seed_for(master, domain, index=0):
    key = json.dumps([master, domain, index], separators=(',', ':')).encode()
    return int.from_bytes(sha256(key).digest()[:8], 'big') >> 1


def read_config(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def check_config(cfg):
    if not 1 <= cfg['workers'] or cfg['action_limit'] < 1 or cfg['checkpoint_every_updates'] < 1:
        raise ValueError('Worker count, action limit and checkpoint interval must be positive')
    if cfg['initial_stage'] not in range(1, 10) or len(cfg['curriculum']) != 9:
        raise ValueError('The curriculum must contain nine stages and a valid initial stage')
    if any(not 0 <= stage['advance_rate'] <= 1 for stage in cfg['curriculum']):
        raise ValueError('Curriculum advancement rates must be between 0 and 1')
    if any(value is not None and value <= 0 for value in cfg['budget'].values()) or not any(cfg['budget'].values()):
        raise ValueError('Set at least one positive training budget')
    ppo = cfg['ppo']
    if (any(ppo[key] < 1 for key in ('rollout_size', 'batch_size', 'update_epochs'))
            or not ppo['hidden_dims'] or any(n < 1 for n in ppo['hidden_dims'])
            or ppo['learning_rate'] <= 0 or not 0 < ppo['clip_range'] < 1
            or min(ppo['gamma'], ppo['gae_lambda'], ppo['entropy_coef']) < 0
            or max(ppo['gamma'], ppo['gae_lambda']) > 1):
        raise ValueError('Invalid PPO settings')
    evaluation = cfg['evaluation']
    if min(evaluation[key] for key in ('interval_steps', 'validation_episodes', 'test_episodes', 'action_limit')) < 1:
        raise ValueError('Evaluation episode counts, interval and action limit must be positive')


def batch_observations(observations):
    return {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}


def tensors(observations, device):
    import torch

    return {key: torch.as_tensor(value, device=device) for key, value in observations.items()}


def model_for(env, cfg, device):
    from .ppo import ActorCritic

    return ActorCritic(env.observation_space['features'].shape[0], env.action_phases,
                       cfg['ppo']['hidden_dims']).to(device)


def evaluate(cfg, model, stage, store, *, split, step, elapsed, train_episodes,
             device, policies=('ppo', 'random')):
    """Use frozen, paired manifests and the faithful action cap for every policy."""
    evaluation = cfg['evaluation']
    env = DungeonEnv(cfg['curriculum'][stage - 1], action_limit=evaluation['action_limit'])
    n = evaluation[f'{split}_episodes']
    rows = {}
    for policy in policies:
        stats = EpisodeStats()
        for index in range(n):
            seed = seed_for(cfg['master_seed'], split, index)
            policy_seed = seed_for(cfg['master_seed'], f'{split}-random-policy', index)
            rng = random.Random(policy_seed)
            observation, _ = env.reset(seed=seed)
            reward_sum = 0.0
            started = perf_counter()
            finished = False
            while not finished:
                if policy == 'random':
                    action = rng.choice(env.legal_ids_in_engine_order())
                else:
                    action, _, _ = model.act(tensors(batch_observations([observation]), device),
                                             deterministic=True)
                    action = int(action.item())
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += reward
                finished = terminated or truncated
            row = dict(info, split=split, policy=policy, stage=stage, step=step, seed=seed,
                       policy_seed=policy_seed if policy == 'random' else None,
                       actions=env.actions, reward=reward_sum, elapsed=perf_counter() - started)
            stats.add(row)
            store.episode(row)
        rows[policy] = dict(stats.summary(), policy=policy, split=split, step=step, stage=stage,
                            elapsed=elapsed, train_episodes=train_episodes)
        store.evaluation(rows[policy])
        print(f'{split} {policy}: {n} games, clear {rows[policy]["clear_rate"]:.1%}, '
              f'target {rows[policy]["target_success_rate"]:.1%}', flush=True)
    return rows


def save_checkpoint(path, payload):
    import torch

    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def read_checkpoint(path):
    import torch

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['version'] != CHECKPOINT_VERSION:
        raise ValueError('Unsupported checkpoint version')
    return checkpoint


def _expired(cfg, state):
    counters = {'steps': state['step'], 'episodes': state['episodes'], 'seconds': state['elapsed']}
    return any(limit is not None and counters[name] >= limit
               for name, limit in cfg['budget'].items())


class TrainingRun:
    """Own one learner, its active games and a checkpoint-aligned experiment store."""

    def __init__(self, cfg, directory, store, checkpoint):
        import torch

        self.cfg, self.directory, self.store = cfg, directory, store
        self.device = torch.device(cfg['device'])
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Install a compatible PyTorch wheel or set device to cpu.')
        torch.set_num_threads(cfg['torch_cpu_threads'])
        random.seed(seed_for(cfg['master_seed'], 'python'))
        np.random.seed(seed_for(cfg['master_seed'], 'numpy') % 2**32)
        torch.manual_seed(seed_for(cfg['master_seed'], 'torch'))
        self.state = dict(stage=cfg['initial_stage'], step=0, episodes=0, elapsed=0.0,
                          update=0, next_game=0, stage_changes=0,
                          next_eval=cfg['evaluation']['interval_steps'])
        self.stats = EpisodeStats()
        self.active = {}
        self.checkpoint = checkpoint
        if checkpoint:
            self.state.update(checkpoint['state'])
            self.stats = checkpoint['stats']
            self.active = checkpoint['active']
            store.restore_watermarks(checkpoint['watermarks'])
        probe = DungeonEnv(**self.options())
        self.model = model_for(probe, cfg, self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg['ppo']['learning_rate'])
        if checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for group in self.optimizer.param_groups:
                group['lr'] = cfg['ppo']['learning_rate']
            random.setstate(checkpoint['python_rng'])
            np.random.set_state(checkpoint['numpy_rng'])
            torch.set_rng_state(checkpoint['torch_rng'])
            if self.device.type == 'cuda' and checkpoint['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        from .ppo import Rollout

        self.rollout = Rollout(cfg['ppo']['rollout_size'], cfg['workers'],
                               probe.observation_space['features'].shape[0], probe.action_space.n)
        self.telemetry = Telemetry()

    def options(self):
        return dict(stage=self.cfg['curriculum'][self.state['stage'] - 1],
                    action_limit=self.cfg['action_limit'], early_deadlock=self.cfg['early_deadlock'])

    def reset_slots(self, vector, slots):
        seeds = {}
        for slot in slots:
            seeds[slot] = seed_for(self.cfg['master_seed'], 'game', self.state['next_game'])
            self.state['next_game'] += 1
            self.active[slot] = dict(seed=seeds[slot], reward=0.0, actions=0)
        for slot, (observation, _) in vector.call('reset', seeds).items():
            self.active[slot]['observation'] = observation

    def record_step(self, vector):
        import torch

        obs = batch_observations([row['observation'] for row in self.active.values()])
        slots = list(self.active)
        actions, logs, values = self.model.act(tensors(obs, self.device))
        action_ids = actions.cpu().numpy()
        transitions = list(vector.call('step', dict(zip(slots, action_ids.tolist()))).values())
        next_obs, rewards, terminals, truncations, infos = zip(*transitions)
        with torch.no_grad():
            next_values = self.model.value(tensors(batch_observations(next_obs), self.device)).cpu().numpy()
        episode_end = np.logical_or(terminals, truncations)
        for index, info in enumerate(infos):
            if terminals[index] or info['deadlock_mimic_counter']:
                next_values[index] = 0.0
        self.rollout.add(obs, action_ids, logs.cpu().numpy(), values.cpu().numpy(),
                         rewards, next_values, episode_end)
        self.state['step'] += len(slots)
        reset = []
        for index, slot in enumerate(slots):
            active = self.active[slot]
            active['reward'] += rewards[index]
            active['actions'] += 1
            active['observation'] = next_obs[index]
            if episode_end[index]:
                row = {key: active[key] for key in ('seed', 'reward', 'actions')}
                row.update(infos[index], split='train', policy='ppo', step=self.state['step'],
                           stage=self.state['stage'])
                self.stats.add(row)
                self.store.episode(row)
                self.state['episodes'] += 1
                reset.append(slot)
        self.reset_slots(vector, reset)

    def save(self, vector, label=None):
        import torch

        payload = dict(version=CHECKPOINT_VERSION, config=self.cfg, state=self.state,
                       stats=self.stats, active=self.active,
                       environments=vector.call('snapshot', dict.fromkeys(self.active)),
                       model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                       python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                       torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None,
                       watermarks=self.store.watermarks())
        if label:
            save_checkpoint(self.directory / f'{label}.pt', payload)
        save_checkpoint(self.directory / 'latest.pt', payload)

    def _collect_update(self, vector):
        started = perf_counter()
        self.rollout.clear()
        while len(self.rollout) < self.cfg['ppo']['rollout_size']:
            self.record_step(vector)
            collection_time = perf_counter() - started
            if _expired(self.cfg, dict(self.state, elapsed=self.state['elapsed'] + collection_time)):
                break
        stats = self.rollout.update(self.model, self.optimizer, self.cfg['ppo'], self.device)
        duration = perf_counter() - started
        self.state['elapsed'] += duration
        self.state['update'] += 1
        throughput = len(self.rollout) * self.cfg['workers'] / duration
        self.store.update(dict(self.stats.summary(), **stats, step=self.state['step'],
                               stage=self.state['stage'], elapsed=self.state['elapsed'],
                               throughput=throughput, telemetry=self.telemetry.sample()))
        budget = self.cfg['budget']
        print(f'Stage {self.state["stage"]} | steps {self.state["step"]:,}/{budget["steps"] or "unlimited"} '
              f'| episodes {self.state["episodes"]:,}/{budget["episodes"] or "unlimited"} '
              f'| elapsed {self.state["elapsed"]:.1f}s | {throughput:,.0f} steps/s', flush=True)

    def run(self, switch_to):
        initial = self.checkpoint
        switch_from = None
        if initial and switch_to is None:
            switch_to = self.state.get('pending_stage')
        while True:
            with open_vector(self.cfg['workers'], self.options()) as vector:
                if initial:
                    vector.call('restore', initial['environments'])
                    initial = None
                else:
                    self.active.clear()
                    self.reset_slots(vector, range(self.cfg['workers']))
                stage = self.state['stage']
                if switch_from is not None:
                    self.state.pop('pending_stage', None)
                    self.store.log_event(dict(step=self.state['step'], stage=stage, kind='stage_change',
                                              previous_stage=switch_from, discarded_active=self.cfg['workers']))
                    self.save(vector, f'stage-{stage}-start-step-{self.state["step"]}-change-{self.state["stage_changes"]}')
                    switch_from = None
                elif self.state['update'] == 0 and not self.checkpoint:
                    self.save(vector)
                if switch_to is None or switch_to == stage:
                    switch_to = None
                    while not _expired(self.cfg, self.state):
                        self._collect_update(vector)
                        if self.state['step'] >= self.state['next_eval']:
                            validation = evaluate(
                                self.cfg, self.model, stage, self.store, split='validation',
                                step=self.state['step'], elapsed=self.state['elapsed'],
                                train_episodes=self.state['episodes'], device=self.device, policies=('ppo',))
                            self.state['next_eval'] = self.state['step'] + self.cfg['evaluation']['interval_steps']
                            if (self.cfg['evaluation']['auto_advance'] and stage < 9 and
                                    validation['ppo']['target_success_rate'] >= self.options()['stage']['advance_rate']):
                                switch_to = stage + 1
                                break
                            if (stage == 9 and not self.state.get('stage_9_target_met') and
                                    validation['ppo']['target_success_rate'] >= self.options()['stage']['advance_rate']):
                                self.state['stage_9_target_met'] = True
                                self.store.log_event(dict(step=self.state['step'], stage=stage, kind='stage_9_target_met',
                                                          clear_rate=validation['ppo']['clear_rate']))
                        if self.state['update'] % self.cfg['checkpoint_every_updates'] == 0:
                            self.save(vector)
                if switch_to is None:
                    self.save(vector)
                    return self.state
                if 'pending_stage' not in self.state:
                    self.state['stage_changes'] += 1
                self.state['pending_stage'] = switch_to
                self.save(vector, f'stage-{stage}-end-step-{self.state["step"]}-change-{self.state["stage_changes"]}')
                self.state['stage'] = switch_to
                self.state['next_eval'] = self.state['step'] + self.cfg['evaluation']['interval_steps']
                switch_from, switch_to = stage, None


def train(cfg, directory, *, resume=False, stage=None):
    check_config(cfg)
    if stage is not None and stage not in range(1, 10):
        raise ValueError('Stage must be between 1 and 9')
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        db_path = directory / 'training.sqlite3'
        checkpoint = read_checkpoint(directory / 'latest.pt') if resume else None
        if not resume and (db_path.exists() or (directory / 'latest.pt').exists()):
            raise ValueError('Experiment already exists. Resume it or choose a new output directory.')
        if resume:
            if not db_path.exists():
                raise ValueError('The experiment database for this checkpoint is missing')
            original = checkpoint['config']
            fixed = ('master_seed', 'workers', 'action_limit', 'early_deadlock', 'curriculum')
            if any(original[key] != cfg[key] for key in fixed) or original['ppo']['hidden_dims'] != cfg['ppo']['hidden_dims']:
                raise ValueError('Resume requires matching seed, workers, environment, curriculum and hidden dimensions')
        with ExperimentStore(db_path) as store:
            result = TrainingRun(cfg, directory, store, checkpoint).run(stage)
        generate_report(db_path, directory / 'report')
        return result


def evaluate_checkpoint(cfg, directory, *, split='test'):
    import torch

    check_config(cfg)
    torch.set_num_threads(cfg['torch_cpu_threads'])
    # Keep independent test results outside the training DB rollback boundary.
    db_path = directory / f'{split}_evaluation.sqlite3'
    with run_lock(directory):
        checkpoint = read_checkpoint(directory / 'latest.pt')
        saved_cfg = checkpoint['config']
        if cfg['master_seed'] != saved_cfg['master_seed']:
            raise ValueError('The evaluation master seed must match the training checkpoint')
        stage, state = checkpoint['state']['stage'], checkpoint['state']
        env = DungeonEnv(saved_cfg['curriculum'][stage - 1])
        model = model_for(env, saved_cfg, cfg['device'])
        model.load_state_dict(checkpoint['model'])
        model.eval()
        with ExperimentStore(db_path) as store:
            result = evaluate(dict(cfg, curriculum=saved_cfg['curriculum']), model, stage, store,
                              split=split, step=state['step'], elapsed=state['elapsed'],
                              train_episodes=state['episodes'], device=cfg['device'])
        generate_report(db_path, directory / f'{split}_report')
    return result


def validate():
    """Run eight training decisions and tiny capped evaluation fixtures."""
    cfg = read_config(DEFAULT_CONFIG)
    cfg.update(workers=2, device='cpu', action_limit=2, torch_cpu_threads=1)
    cfg['budget'] = dict(steps=4, episodes=None, seconds=None)
    cfg['ppo'].update(hidden_dims=[8], rollout_size=2, batch_size=4, update_epochs=1)
    cfg['evaluation'].update(interval_steps=4, validation_episodes=1, test_episodes=1, action_limit=2)
    cfg['curriculum'][0]['advance_rate'] = 0.0  # Exercise the transition without winning a real floor.
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        first = train(cfg, directory)
        assert first['stage'] == 2 and len(list(directory.glob('stage-*.pt'))) == 2
        resumed = deepcopy(cfg)
        resumed['budget']['steps'] = 8
        resumed['evaluation']['interval_steps'] = 100
        final = train(resumed, directory, resume=True)
        result = evaluate_checkpoint(cfg, directory)
        assert result['ppo']['episodes'] == result['random']['episodes'] == 1
        changed = train(resumed, directory, resume=True, stage=3)
        assert changed['stage'] == 3 and changed['step'] == 8 and len(list(directory.glob('stage-*.pt'))) == 4
        return dict(steps=final['step'], stage=final['stage'],
                    report=any((directory / 'test_report').glob('*.png')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'resume', 'evaluate', 'report', 'validate'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    args = parser.parse_args()
    if args.command == 'validate':
        import unittest

        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_rl_*.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    cfg = read_config(args.config)
    directory = args.output or ROOT / cfg['output']
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=directory / 'errors.log', encoding='utf-8', level=logging.ERROR)
    try:
        if args.command in ('train', 'resume'):
            train(cfg, directory, resume=args.command == 'resume', stage=args.stage)
        elif args.command == 'evaluate':
            evaluate_checkpoint(cfg, directory, split=args.split)
        else:
            for stem in ('training', 'validation_evaluation', 'test_evaluation'):
                if (directory / f'{stem}.sqlite3').exists():
                    name = 'report' if stem == 'training' else f'{stem.split("_")[0]}_report'
                    print(generate_report(directory / f'{stem}.sqlite3', directory / name))
    except (KeyboardInterrupt, Exception):
        logging.exception('Local RL command ended before completion')
        raise


if __name__ == '__main__':
    main()

"""Standalone local training, validation, evaluation and reporting commands."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import logging
import math
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from experiments.random_baseline_large import run_lock
from .decision_stats import DecisionStats
from .environment import DungeonEnv
from .reporting import EpisodeStats, ExperimentStore, Telemetry, generate_report
from .vector import open_vector


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'rl' / 'training.json'
SHOP_CONFIG = ROOT / 'rl' / 'shop_training.json'
CHECKPOINT_VERSION = 1
EVAL_MODES = ('deterministic', 'stochastic')


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
    for stage in cfg['curriculum']:
        epsilon = stage.get('shop_repeat_epsilon', 0.0001)
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError('shop_repeat_epsilon must be finite and nonnegative in every stage')
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
    if cfg.get('shop_policy', 'legacy') not in ('legacy', 'strategic'):
        raise ValueError('shop_policy must be legacy or strategic')
    if any(evaluation.get(f'{split}_policy', 'deterministic') not in EVAL_MODES
           for split in ('validation', 'test')):
        raise ValueError('Evaluation policy must be deterministic or stochastic')
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


def evaluation_mode(cfg, split, policy):
    if policy == 'random':
        return 'random'
    if policy == 'ppo':
        default = ('stochastic' if split == 'validation' and cfg.get('shop_policy') == 'strategic'
                   else 'deterministic')
        return cfg['evaluation'].get(f'{split}_policy', default)
    if policy in ('ppo_deterministic', 'ppo_stochastic'):
        return policy.removeprefix('ppo_')
    raise ValueError(f'Unknown evaluation policy {policy!r}')


def evaluate(cfg, model, stage, store, *, split, step, elapsed, train_episodes,
             device, policies=('ppo', 'random'), evaluation_elapsed=0.0):
    """Run paired seed panels with one GPU action batch per active worker round."""
    evaluation = cfg['evaluation']
    profile = cfg.get('shop_policy', 'legacy')
    options = dict(stage=cfg['curriculum'][stage - 1], action_limit=evaluation['action_limit'],
                   shop_policy=profile)
    n = evaluation[f'{split}_episodes']
    workers = min(cfg['workers'], n)
    catalog = DungeonEnv(**options).action_catalog
    rows = {}
    for policy in policies:
        mode = evaluation_mode(cfg, split, policy)
        label = policy if mode == 'random' else f'ppo_{mode}'
        stats = EpisodeStats()
        decisions = DecisionStats()
        active = {}
        next_index = actions = rounds = max_uncapped_actions = 0
        policy_seconds = worker_seconds = 0.0
        started = last_progress = perf_counter()

        def worker_call(command, args):
            nonlocal worker_seconds
            before = perf_counter()
            result = vector.call(command, args)
            worker_seconds += perf_counter() - before
            return result

        def refill(slots):
            nonlocal next_index
            seeds = {}
            for slot in slots:
                if next_index == n:
                    break
                seed = seed_for(cfg['master_seed'], split, next_index)
                stream = 'random' if mode == 'random' else 'ppo'
                policy_seed = seed_for(cfg['master_seed'], f'{split}-{stream}-policy', next_index)
                decisions.reset_slot(slot)
                active[slot] = dict(seed=seed, policy_seed=policy_seed,
                                    rng=random.Random(policy_seed),
                                    reward=0.0, actions=0, started=perf_counter(), recent=[])
                seeds[slot] = seed
                next_index += 1
            if seeds:
                for slot, ((obs, info), labels) in worker_call('eval_reset', seeds).items():
                    active[slot].update(observation=obs, labels=labels, floor=info['floor'])

        def progress():
            seconds = perf_counter() - started
            done = stats.episodes
            remaining = None
            if done:
                typical = stats.action_sum / done
                lengths = [max(1, evaluation['action_limit'] - ep['actions'])
                           if ep['actions'] >= max(1, max_uncapped_actions) else max(1, typical - ep['actions'])
                           for ep in active.values()]
                typical = (actions + sum(lengths)) / next_index
                pending = sum(lengths) + (n - next_index) * typical
                remaining = max(max(lengths, default=0), pending / workers) * seconds / max(1, rounds)
            eta = f'{remaining:.1f}s' if remaining is not None else 'pending first completion'
            longest = max((episode['actions'] for episode in active.values()), default=0)
            print(f'{split} {label} | {done}/{n} episodes | {actions:,} actions '
                  f'| longest active {longest:,}/{evaluation["action_limit"]:,} '
                  f'| elapsed {seconds:.1f}s | ETA {eta}', flush=True)

        progress()
        with open_vector(workers, options) as vector:
            refill(range(workers))
            while active:
                slots = list(active)
                if mode == 'random':
                    legal = worker_call('legal', dict.fromkeys(slots))
                    before = perf_counter()
                    choices = [active[slot]['rng'].choice(legal[slot]) for slot in slots]
                    probabilities = [{action: 1 / len(legal[slot]) for action in legal[slot]}
                                     for slot in slots]
                else:
                    before = perf_counter()
                    observations = batch_observations([active[slot]['observation'] for slot in slots])
                    draws = [active[slot]['rng'].random() for slot in slots] if mode == 'stochastic' else None
                    batch_actions, scores = model.evaluation_actions(tensors(observations, device), draws)
                    choices = batch_actions.cpu().tolist()
                    probabilities = scores.cpu().numpy()
                policy_seconds += perf_counter() - before
                rounds += 1
                transitions = worker_call('eval_step', dict(zip(slots, choices)))
                actions += len(slots)
                finished = []
                for slot, choice, scores in zip(slots, choices, probabilities):
                    (obs, reward, terminal, truncated, info), labels = transitions[slot]
                    episode = active[slot]
                    decisions.record(episode['observation']['phase'], episode['floor'],
                                     episode['labels'], choice, scores, slot=slot)
                    decisions.transition(info, slot=slot)
                    episode['reward'] += reward
                    episode['actions'] += 1
                    episode.update(observation=obs, labels=labels, floor=info['floor'])
                    episode['recent'] = [*episode['recent'][-7:], catalog[choice]]
                    if terminal or truncated:
                        if info['outcome'] != 'timeout':
                            max_uncapped_actions = max(max_uncapped_actions, episode['actions'])
                        row = dict(info, split=split, policy=label, stage=stage, step=step,
                                   seed=episode['seed'],
                                   policy_seed=episode['policy_seed'] if mode != 'deterministic' else None,
                                   actions=episode['actions'], reward=episode['reward'],
                                   elapsed=perf_counter() - episode['started'],
                                   last_actions=episode['recent'])
                        stats.add(row)
                        decisions.episode(row, slot=slot)
                        store.episode(row)
                        finished.append(slot)
                for slot in finished:
                    del active[slot]
                refill(finished)
                if perf_counter() - last_progress >= 2.0:
                    progress()
                    last_progress = perf_counter()
        progress()
        seconds = perf_counter() - started
        evaluation_elapsed += seconds
        rows[policy] = dict(stats.summary(), policy=label, split=split, step=step, stage=stage,
                            shop_policy=profile, decision_stats=decisions.summary(),
                            elapsed=elapsed, training_seconds=elapsed, train_episodes=train_episodes,
                            evaluation_seconds=seconds,
                            evaluation_elapsed=evaluation_elapsed, eval_actions=actions,
                            eval_workers=workers, decision_batches=rounds, policy_seconds=policy_seconds,
                            worker_seconds=worker_seconds, action_limit=evaluation['action_limit'])
        store.evaluation(rows[policy])
        print(f'{split} {label}: {n} games, clear {rows[policy]["clear_rate"]:.1%}, '
              f'target {rows[policy]["target_success_rate"]:.1%} | '
              f'{seconds:.2f}s ({policy_seconds:.2f}s policy, {worker_seconds:.2f}s workers/IPC)', flush=True)
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
        self.state = dict(stage=cfg['initial_stage'], step=0, episodes=0, elapsed=0.0, evaluation_elapsed=0.0,
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
                    action_limit=self.cfg['action_limit'], early_deadlock=self.cfg['early_deadlock'],
                    shop_policy=self.cfg.get('shop_policy', 'legacy'))

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
                               training_seconds=self.state['elapsed'],
                               evaluation_elapsed=self.state['evaluation_elapsed'],
                               throughput=throughput, telemetry=self.telemetry.sample()))
        budget = self.cfg['budget']
        print(f'Stage {self.state["stage"]} | steps {self.state["step"]:,}/{budget["steps"] or "unlimited"} '
              f'| episodes {self.state["episodes"]:,}/{budget["episodes"] or "unlimited"} '
              f'| train {self.state["elapsed"]:.1f}s | evaluation {self.state["evaluation_elapsed"]:.1f}s '
              f'| {throughput:,.0f} steps/s', flush=True)

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
                    while True:
                        if self.state['step'] >= self.state['next_eval']:
                            # Keep next_eval due in this checkpoint so an interrupted panel restarts on resume.
                            self.save(vector)
                            validation = evaluate(
                                self.cfg, self.model, stage, self.store, split='validation',
                                step=self.state['step'], elapsed=self.state['elapsed'],
                                train_episodes=self.state['episodes'], device=self.device, policies=('ppo',),
                                evaluation_elapsed=self.state['evaluation_elapsed'])
                            self.state['evaluation_elapsed'] = validation['ppo']['evaluation_elapsed']
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
                            self.save(vector)
                        if _expired(self.cfg, self.state):
                            break
                        self._collect_update(vector)
                        if (self.state['step'] < self.state['next_eval'] and
                                self.state['update'] % self.cfg['checkpoint_every_updates'] == 0):
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
            if original.get('shop_policy', 'legacy') != cfg.get('shop_policy', 'legacy'):
                raise ValueError('Checkpoint Shop Head and observation schema differ. Start a new run or use the matching configuration.')
            fixed = ('master_seed', 'workers', 'action_limit', 'early_deadlock', 'curriculum')
            if any(original[key] != cfg[key] for key in fixed) or original['ppo']['hidden_dims'] != cfg['ppo']['hidden_dims']:
                raise ValueError('Resume requires matching seed, workers, environment, curriculum and hidden dimensions')
        with ExperimentStore(db_path) as store:
            result = TrainingRun(cfg, directory, store, checkpoint).run(stage)
        generate_report(db_path, directory / 'report')
        return result


def evaluate_checkpoint(cfg, directory, *, split='test', policies=('ppo', 'random')):
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
        profile = saved_cfg.get('shop_policy', 'legacy')
        env = DungeonEnv(saved_cfg['curriculum'][stage - 1], shop_policy=profile)
        model = model_for(env, saved_cfg, cfg['device'])
        model.load_state_dict(checkpoint['model'])
        model.eval()
        with ExperimentStore(db_path) as store:
            result = evaluate(dict(cfg, curriculum=saved_cfg['curriculum'], shop_policy=profile), model, stage, store,
                              split=split, step=state['step'], elapsed=state['elapsed'],
                              train_episodes=state['episodes'], device=cfg['device'], policies=policies)
        generate_report(db_path, directory / f'{split}_report')
    return result


def validate():
    """Run eight training decisions and tiny capped evaluation fixtures."""
    from unittest.mock import patch

    cfg = read_config(DEFAULT_CONFIG)
    cfg.update(workers=2, device='cpu', action_limit=2, torch_cpu_threads=1)
    cfg['budget'] = dict(steps=4, episodes=None, seconds=None)
    cfg['ppo'].update(hidden_dims=[8], rollout_size=2, batch_size=4, update_epochs=1)
    cfg['evaluation'].update(interval_steps=4, validation_episodes=1, test_episodes=1, action_limit=2)
    cfg['curriculum'][0]['advance_rate'] = 0.0  # Exercise the transition without winning a real floor.
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        with patch(f'{__package__}.runner.evaluate', side_effect=KeyboardInterrupt):
            try:
                train(cfg, directory)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError('Expected to interrupt the pending validation')
        before = read_checkpoint(directory / 'latest.pt')['state']
        assert before['step'] == before['next_eval'] == 4 and before['evaluation_elapsed'] == 0
        first = train(cfg, directory, resume=True)
        assert first['stage'] == 2 and len(list(directory.glob('stage-*.pt'))) == 2
        assert first['evaluation_elapsed'] > 0
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
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument('--policy', choices=('both', 'ppo', 'random', 'ppo_deterministic', 'ppo_stochastic',
                                            'compare'), default='both')
    args = parser.parse_args()
    if args.command == 'validate':
        import unittest

        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_rl_*.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    cfg = read_config(args.config or (SHOP_CONFIG if args.command == 'train' else DEFAULT_CONFIG))
    directory = args.output or ROOT / cfg['output']
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=directory / 'errors.log', encoding='utf-8', level=logging.ERROR)
    try:
        if args.command in ('train', 'resume'):
            train(cfg, directory, resume=args.command == 'resume', stage=args.stage)
        elif args.command == 'evaluate':
            policies = {'both': ('ppo', 'random'), 'compare': ('ppo_deterministic', 'ppo_stochastic')}.get(
                args.policy, (args.policy,))
            evaluate_checkpoint(cfg, directory, split=args.split, policies=policies)
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

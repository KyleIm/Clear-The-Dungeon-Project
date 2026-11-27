"""Two-head actor–critic and masked PPO updates for Combat and Reward."""

from contextlib import contextmanager
from time import perf_counter
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@contextmanager
def _timed(profile, name, device):
    if not profile:
        yield
        return
    synchronize = profile.detailed and device.type == 'cuda'
    if synchronize:
        torch.cuda.synchronize(device)
    start = perf_counter()
    yield
    if synchronize:
        torch.cuda.synchronize(device)
    profile.add(name, perf_counter() - start)


class Transition(NamedTuple):
    features: np.ndarray
    phase: int
    mask: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    next_value: float
    done: bool
    slot: int


class ActorCritic(nn.Module):
    """Share the encoder and critic across the two learned phases."""

    def __init__(self, hidden_dims):
        super().__init__()
        width = 583
        layers = []
        for units in hidden_dims:
            layers.extend((nn.Linear(width, units), nn.Tanh()))
            width = units
        self.encoder = nn.Sequential(*layers)
        self.combat_head = nn.Linear(width, 7)
        self.reward_head = nn.Linear(width, 5)
        self.critic = nn.Linear(width, 1)

    @property
    def device(self):
        return self.critic.weight.device

    def _inputs(self, obs):
        features, phases, masks = obs
        return (torch.as_tensor(features, dtype=torch.float32, device=self.device),
                torch.as_tensor(phases, dtype=torch.long, device=self.device),
                torch.as_tensor(masks, dtype=torch.bool, device=self.device))

    def forward(self, features, phases, masks):
        encoded = self.encoder(features)
        combat_logits = self.combat_head(encoded)
        reward_logits = nn.functional.pad(self.reward_head(encoded), (0, 2), value=-torch.inf)
        logits = torch.where(phases[:, None] == 0, combat_logits, reward_logits)
        distribution = Categorical(logits=logits.masked_fill(~masks, -torch.inf), validate_args=False)
        return distribution, self.critic(encoded).squeeze(-1)

    @torch.inference_mode()
    def act(self, obs, deterministic=False, *, profile=None, bundle=False):
        """Sample legal actions or take the masked argmax for a batch."""
        with _timed(profile, 'host_to_device', self.device):
            inputs = self._inputs(obs)
        with _timed(profile, 'actor_critic_inference', self.device):
            distribution, value = self(*inputs)
            action = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
            log_prob = distribution.log_prob(action)
            # Action IDs are 0–6 and are exactly representable in the value dtype.
            if bundle and self.device.type == 'cuda':
                outputs = torch.stack((action.to(value.dtype), log_prob, value))
        with _timed(profile, 'device_to_host', self.device):
            if bundle and self.device.type == 'cuda':
                host = outputs.cpu().numpy()
                return host[0].astype(np.int64), host[1], host[2]
            return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    @torch.inference_mode()
    def values(self, obs, *, profile=None):
        with _timed(profile, 'host_to_device', self.device):
            features = torch.as_tensor(obs[0], dtype=torch.float32, device=self.device)
        with _timed(profile, 'critic_inference', self.device):
            values = self.critic(self.encoder(features)).squeeze(-1)
        with _timed(profile, 'device_to_host', self.device):
            return values.cpu().numpy()


def advantages(rollout, gamma, gae_lambda):
    """Use one backward GAE trace per worker slot, cutting at episode boundaries."""
    result = np.empty(len(rollout), dtype=np.float32)
    by_slot = {}
    for index in range(len(rollout) - 1, -1, -1):
        step = rollout[index]
        delta = float(step.reward) + gamma * float(step.next_value) - float(step.value)
        future = 0.0 if step.done else by_slot.get(step.slot, 0.0)
        estimate = delta + gamma * gae_lambda * future
        result[index] = by_slot[step.slot] = estimate
    return result


def update(model, optimizer, rollout, cfg, *, profile=None):
    """Optimize the clipped objective, shared value head, and masked policy entropy."""
    if not rollout:
        raise ValueError("A PPO update requires at least one decision")

    with _timed(profile, 'ppo_staging', model.device):
        observations = (np.stack([step.features for step in rollout]),
                        [step.phase for step in rollout], np.stack([step.mask for step in rollout]))
        actions_array = [step.action for step in rollout]
        logp_array = [step.log_prob for step in rollout]
        value_array = [step.value for step in rollout]
        advantages_array = advantages(rollout, cfg['gamma'], cfg['gae_lambda'])
    with _timed(profile, 'host_to_device', model.device):
        features, phases, masks = model._inputs(observations)
        actions = torch.as_tensor(actions_array, device=model.device)
        old_logp = torch.as_tensor(logp_array, dtype=torch.float32, device=model.device)
        old_value = torch.as_tensor(value_array, dtype=torch.float32, device=model.device)
        estimates = torch.as_tensor(advantages_array, device=model.device)
    with _timed(profile, 'ppo_update', model.device):
        averages = _optimize(model, optimizer, features, phases, masks, actions, old_logp, old_value,
                             estimates, cfg)
    with _timed(profile, 'device_to_host', model.device):
        return dict(zip(('policy_loss', 'value_loss', 'entropy'), averages.cpu().tolist()))


def _optimize(model, optimizer, features, phases, masks, actions, old_logp, old_value, estimates, cfg):
    returns = estimates + old_value
    if len(features) > 1:
        estimates = (estimates - estimates.mean()) / (estimates.std(unbiased=False) + 1e-8)

    totals = torch.zeros(3, device=model.device)
    decisions = 0
    for _ in range(cfg["epochs"]):
        permutation = torch.randperm(len(features), device=model.device)
        for batch in permutation.split(cfg["minibatch"]):
            distribution, values = model(features[batch], phases[batch], masks[batch])
            ratio = (distribution.log_prob(actions[batch]) - old_logp[batch]).exp()
            advantage = estimates[batch]
            unclipped = ratio * advantage
            clipped = ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(values, returns[batch])
            entropy = distribution.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - cfg["entropy"] * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            totals += torch.stack((policy_loss.detach(), value_loss.detach(), entropy.detach())) * batch.numel()
            decisions += batch.numel()
    return totals / decisions

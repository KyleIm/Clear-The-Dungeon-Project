"""Two-head actor–critic and masked PPO updates for Combat and Reward."""

from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


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
    def act(self, obs, deterministic=False):
        """Sample legal actions or take the masked argmax for a batch."""
        distribution, value = self(*self._inputs(obs))
        action = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action.cpu().numpy(), distribution.log_prob(action).cpu().numpy(), value.cpu().numpy()

    @torch.inference_mode()
    def values(self, obs):
        features = torch.as_tensor(obs[0], dtype=torch.float32, device=self.device)
        return self.critic(self.encoder(features)).squeeze(-1).cpu().numpy()


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


def update(model, optimizer, rollout, cfg):
    """Optimize the clipped objective, shared value head, and masked policy entropy."""
    if not rollout:
        raise ValueError("A PPO update requires at least one decision")

    features, phases, masks = model._inputs((np.stack([step.features for step in rollout]),
                                             [step.phase for step in rollout],
                                             np.stack([step.mask for step in rollout])))
    actions = torch.as_tensor([step.action for step in rollout], device=model.device)
    old_logp = torch.as_tensor([step.log_prob for step in rollout], dtype=torch.float32, device=model.device)
    old_value = torch.as_tensor([step.value for step in rollout], dtype=torch.float32, device=model.device)
    estimates = torch.as_tensor(advantages(rollout, cfg["gamma"], cfg["gae_lambda"]), device=model.device)
    returns = estimates + old_value
    if len(rollout) > 1:
        estimates = (estimates - estimates.mean()) / (estimates.std(unbiased=False) + 1e-8)

    totals = torch.zeros(3, device=model.device)
    decisions = 0
    for _ in range(cfg["epochs"]):
        permutation = torch.randperm(len(rollout), device=model.device)
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
    return dict(zip(("policy_loss", "value_loss", "entropy"), (totals / decisions).cpu().tolist()))

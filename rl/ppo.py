"""Shared phase policy, bounded rollout storage, and masked PPO updates."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


PHASE_NAMES = ("combat", "reward", "shop", "event")


class ActorCritic(nn.Module):
    """Score the active phase's actions with a critic shared across phases."""

    def __init__(self, feature_dim, action_phases, hidden_dims):
        super().__init__()
        phases = torch.as_tensor(action_phases, dtype=torch.long)
        if phases.ndim != 1 or torch.any((phases < 0) | (phases >= len(PHASE_NAMES))):
            raise ValueError("Action phases must be a vector of valid phase indices")

        width = feature_dim + len(PHASE_NAMES)
        layers = []
        for next_width in hidden_dims:
            layers.extend((nn.Linear(width, next_width), nn.Tanh()))
            width = next_width
        self.encoder = nn.Sequential(*layers)
        self.policy_heads = nn.ModuleDict()
        for phase, name in enumerate(PHASE_NAMES):
            ids = torch.where(phases == phase)[0]
            if not ids.numel():
                raise ValueError(f"The action catalog has no {name} actions")
            self.register_buffer(f"{name}_ids", ids)
            self.policy_heads[name] = nn.Linear(width, ids.numel())
        self.critic = nn.Linear(width, 1)
        self.num_actions = len(phases)

    def _encode(self, obs):
        phase = nn.functional.one_hot(obs["phase"].long(), num_classes=len(PHASE_NAMES))
        features = obs["features"]
        return self.encoder(torch.cat((features, phase.to(features.dtype)), dim=-1))

    def value(self, obs):
        return self.critic(self._encode(obs)).squeeze(-1)

    def forward(self, obs):
        encoding = self._encode(obs)
        batch_rows = obs["phase"].reshape(-1)
        logits = encoding.new_full((len(batch_rows), self.num_actions), -torch.inf)
        for phase, name in enumerate(PHASE_NAMES):
            rows = torch.where(batch_rows == phase)[0]
            if rows.numel():
                action_ids = getattr(self, f"{name}_ids")
                logits[rows[:, None], action_ids] = self.policy_heads[name](encoding[rows])
        logits = logits.masked_fill(~obs["action_mask"].bool(), -torch.inf)
        return Categorical(logits=logits, validate_args=True), self.critic(encoding).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, *, deterministic=False):
        distribution, value = self(obs)
        action = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), value


class Rollout:
    """Store projected decisions in reusable CPU arrays for one PPO rollout."""

    def __init__(self, capacity, num_envs, feature_dim, num_actions):
        shape = (capacity, num_envs)
        self.features = np.empty((*shape, feature_dim), dtype=np.float32)
        self.phase = np.empty(shape, dtype=np.uint8)
        self.mask = np.empty((*shape, num_actions), dtype=np.bool_)
        self.action = np.empty(shape, dtype=np.int64)
        self.log_prob = np.empty(shape, dtype=np.float32)
        self.value = np.empty(shape, dtype=np.float32)
        self.reward = np.empty(shape, dtype=np.float32)
        self.next_value = np.empty(shape, dtype=np.float32)
        self.episode_end = np.empty(shape, dtype=np.bool_)
        self.size = 0

    def __len__(self):
        return self.size

    def clear(self):
        self.size = 0

    def add(self, obs, action, log_prob, value, reward, next_value, episode_end):
        index = self.size
        if index == len(self.features):
            raise ValueError("Rollout capacity exhausted")
        self.features[index] = obs["features"]
        self.phase[index] = obs["phase"]
        self.mask[index] = obs["action_mask"]
        self.action[index] = action
        self.log_prob[index] = log_prob
        self.value[index] = value
        self.reward[index] = reward
        self.next_value[index] = next_value
        self.episode_end[index] = episode_end
        self.size += 1

    def compute_advantages(self, gamma, gae_lambda):
        values = self.value[:self.size]
        advantages = np.empty_like(values)
        carry = np.zeros(values.shape[1], dtype=np.float32)
        for index in range(self.size - 1, -1, -1):
            error = self.reward[index] + gamma * self.next_value[index] - values[index]
            continuation = ~self.episode_end[index]
            carry = error + gamma * gae_lambda * continuation * carry
            advantages[index] = carry
        return advantages, advantages + values

    def update(self, model, optimizer, config, device):
        if not self.size:
            raise ValueError("Cannot update PPO from an empty rollout")

        advantages, returns = self.compute_advantages(config["gamma"], config["gae_lambda"])
        num_samples = advantages.size
        mean, std = advantages.mean(), advantages.std()
        if num_samples > 1 and std > 1e-8:
            advantages -= mean
            advantages /= std
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        features = self.features[:self.size].reshape(num_samples, -1)
        action_mask = self.mask[:self.size].reshape(num_samples, -1)
        phases = self.phase[:self.size].reshape(-1)
        actions = self.action[:self.size].reshape(-1)
        log_probs = self.log_prob[:self.size].reshape(-1)
        phase_counts = np.bincount(phases, minlength=len(PHASE_NAMES))
        phase_entropy = torch.zeros(len(PHASE_NAMES), device=device)
        losses = torch.zeros(2, device=device)

        def batch(array, indices):
            return torch.as_tensor(array[indices], device=device)

        clip = config["clip_range"]
        batch_size = config["batch_size"]
        epochs = config["update_epochs"]
        for _ in range(epochs):
            permutation = torch.randperm(num_samples).numpy()
            for offset in range(0, num_samples, batch_size):
                indices = permutation[offset:offset + batch_size]
                phase = batch(phases, indices).long()
                obs = {
                    "features": batch(features, indices),
                    "action_mask": batch(action_mask, indices),
                    "phase": phase,
                }
                distribution, value = model(obs)
                ratio = (distribution.log_prob(batch(actions, indices)) - batch(log_probs, indices)).exp()
                advantage = batch(advantages, indices)
                policy_loss = -torch.minimum(
                    ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage,
                ).mean()
                value_loss = nn.functional.mse_loss(value, batch(returns, indices))
                entropy = distribution.entropy()
                loss = policy_loss + 0.5 * value_loss - config["entropy_coef"] * entropy.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                losses += len(indices) * torch.stack((policy_loss.detach(), value_loss.detach()))
                phase_entropy.index_add_(0, phase, entropy.detach())

        policy_loss, value_loss = (losses / (num_samples * epochs)).cpu().tolist()
        entropy_sums = phase_entropy.cpu().tolist()
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "phase_steps": {name: int(count) for name, count in zip(PHASE_NAMES, phase_counts)},
            "phase_entropy": {
                name: entropy / (int(count) * epochs) if count else None
                for name, entropy, count in zip(PHASE_NAMES, entropy_sums, phase_counts)
            },
        }

"""Three-head masked Double DQN for ordered shop decisions."""

from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _by_head(observations):
    groups = defaultdict(list)
    for index, (_, head, _) in enumerate(observations):
        groups[head].append(index)
    return groups.items()


class ShopDQN(nn.Module):
    """Use a shared encoder and separate Q-value heads for the three shop stages."""

    def __init__(self, input_dim, head_sizes, widths):
        super().__init__()
        if len(head_sizes) != 3 or any(size < 1 for size in head_sizes):
            raise ValueError('Expected three nonempty action heads')
        self.input_dim = input_dim
        self.head_sizes = tuple(head_sizes)
        layers = []
        for width in widths:
            layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
            input_dim = width
        self.encoder = nn.Sequential(*layers)
        self.heads = nn.ModuleList(nn.Linear(input_dim, size) for size in head_sizes)

    @property
    def device(self):
        return self.heads[0].weight.device

    def encode(self, features):
        return self.encoder(torch.as_tensor(features, dtype=torch.float32, device=self.device))

    def q_values(self, features, head):
        return self.heads[head](self.encode(features))

    @torch.inference_mode()
    def act(self, observations, rng, epsilon):
        """Score every legal action and optionally sample uniformly from the mask."""
        if not 0 <= epsilon <= 1:
            raise ValueError('Epsilon must be between zero and one')
        if not observations:
            return []
        encoded = self.encode(np.stack([obs[0] for obs in observations]))
        results = [None] * len(observations)
        explored = [epsilon > 0 and rng.random() < epsilon for _ in observations]
        for head, indices in _by_head(observations):
            values = self.heads[head](encoded[indices])
            masks = np.stack([observations[index][2] for index in indices])
            if masks.shape != values.shape or not masks.any(axis=1).all():
                raise ValueError('Each shop decision needs a nonempty mask matching its head')
            legal_mask = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
            actions = values.masked_fill(~legal_mask, -torch.inf).argmax(dim=1).cpu().numpy()
            for position, index in enumerate(indices):
                if explored[index]:
                    allowed = np.flatnonzero(masks[position])
                    actions[position] = allowed[int(rng.random() * len(allowed))]
            action_ids = torch.as_tensor(actions, dtype=torch.long, device=self.device)
            chosen_q = values.gather(1, action_ids[:, None]).squeeze(1).cpu().numpy()
            for position, index in enumerate(indices):
                results[index] = int(actions[position]), float(chosen_q[position]), explored[index]
        return results


@torch.no_grad()
def bellman_targets(online, target, rows):
    """Select the next legal action online and evaluate it on the target network."""
    values = torch.tensor([row['reward'] for row in rows], dtype=torch.float32, device=online.device)
    ongoing = [index for index, row in enumerate(rows) if not row['terminal']]
    if not ongoing:
        return values
    observations = [rows[index]['next_state'] for index in ongoing]
    features = np.stack([obs[0] for obs in observations])
    online_features, target_features = online.encode(features), target.encode(features)
    for head, indices in _by_head(observations):
        scores = online.heads[head](online_features[indices])
        masks = np.stack([observations[index][2] for index in indices])
        if masks.shape != scores.shape or not masks.any(axis=1).all():
            raise ValueError('A nonterminal next observation needs legal actions in its target head')
        legal_mask = torch.as_tensor(masks, dtype=torch.bool, device=online.device)
        actions = scores.masked_fill(~legal_mask, -torch.inf).argmax(dim=1)
        next_scores = target.heads[head](target_features[indices])
        chosen = next_scores.gather(1, actions.to(target.device)[:, None]).squeeze(1)
        values[[ongoing[index] for index in indices]] += chosen.to(online.device)
    return values


def update_dqn(online, target, optimizer, rows):
    """Fit a minibatch with Huber loss and an un-discounted masked Bellman target."""
    if not rows:
        raise ValueError('An empty replay batch cannot update the DQN')
    observations = [row['state'] for row in rows]
    encoded = online.encode(np.stack([obs[0] for obs in observations]))
    labels = bellman_targets(online, target, rows)
    predictions = torch.empty(len(rows), device=online.device)
    for head, indices in _by_head(observations):
        actions = torch.tensor([rows[index]['action'] for index in indices], device=online.device)
        values = online.heads[head](encoded[indices]).gather(1, actions[:, None]).squeeze(1)
        predictions[indices] = values
    loss = F.smooth_l1_loss(predictions, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return dict(dqn_loss=float(loss.detach()), dqn_mean_q=float(predictions.detach().mean()),
                dqn_mean_target=float(labels.mean()), dqn_batch=len(rows),
                dqn_terminal=sum(bool(row['terminal']) for row in rows))

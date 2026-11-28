"""Off-policy Reward return estimation and a constrained Reward-only transfer."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

EXPLORATION = .20


def mixture_probability(ppo_probability, legal_count):
    return (1 - EXPLORATION) * ppo_probability + EXPLORATION / legal_count


class RewardValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.fit_net = nn.Sequential(nn.Linear(583, 128), nn.Tanh(), nn.Linear(128, 5))

    def forward(self, x):
        return self.fit_net(x) * 10


def fit(model, rows, device, rng, epochs):
    x = np.stack([row[0] for row in rows])
    actions = np.asarray([row[2] for row in rows])
    returns = np.asarray([row[3] for row in rows], dtype=np.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    average = []
    for _ in range(epochs):
        order = rng.permutation(len(rows))
        for ids in np.array_split(order, (len(order) + 255) // 256):
            obs = torch.as_tensor(x[ids], device=device)
            selected = torch.as_tensor(actions[ids], device=device)
            target = torch.as_tensor(returns[ids], device=device)
            predicted = model(obs).gather(1, selected[:, None]).squeeze(1)
            loss = F.mse_loss(predicted / 10, target / 10)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            average.append(float(loss.detach()))
    return float(np.mean(average))


def guide(model, observations, masks, q_values, strength, supported=None, learning_rate=.003,
          *, updates=4, kl_limit=.005, temperature=2.):
    """Distill an anchored soft target with a legal mask and an empirical KL limit."""
    device = model.device
    x = torch.as_tensor(observations, dtype=torch.float32, device=device)
    mask = torch.as_tensor(masks, dtype=torch.bool, device=device)
    q = torch.as_tensor(q_values, dtype=torch.float32, device=device)
    allowed = mask if supported is None else mask & torch.as_tensor(supported, device=device)
    active = allowed.sum(1) >= 2
    empty = dict(kl=0., gold_probability_change=0., probability_changes=[0.] * 5, applied=False)
    if not active.any():
        return empty
    x, mask, q, allowed = x[active], mask[active], q[active], allowed[active]
    head = model.reward_head
    weights = [p.detach().clone() for p in head.parameters()]

    def policy(hidden):
        return head(hidden).masked_fill(~mask, -1e9).log_softmax(dim=1)

    with torch.no_grad():
        hidden = model.encoder(x)
        prior = policy(hidden)
        probs = prior.exp()
        teacher = (q / temperature).masked_fill(~allowed, -1e9).softmax(1)
        target = (1-strength)*probs + strength*teacher
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    for _ in range(updates):
        loss = -(target * policy(hidden)).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        proposal = [p.detach().clone() for p in head.parameters()]
        for fraction in (1., .5, .25, .125, 0.):
            for current, before, after in zip(head.parameters(), weights, proposal):
                current.copy_(before.lerp(after, fraction))
            updated = policy(hidden)
            kl = float((probs*(prior-updated)).sum(1).mean())
            if kl <= kl_limit:
                change = (updated.exp()-probs).mean(0).tolist()
                return dict(kl=kl, probability_changes=change, gold_probability_change=change[0],
                            applied=fraction > 0, probability_before=probs.mean(0).tolist(),
                            probability_after=updated.exp().mean(0).tolist(),
                            mean_total_variation=float((updated.exp()-probs).abs().sum(1).mean()/2),
                            accepted_step_fraction=fraction)
    return empty

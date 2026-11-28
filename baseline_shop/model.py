"""Balanced supervised prediction of paired loss, tie and win labels."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .configurations import FEATURE_COUNT


class ConfigurationValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(FEATURE_COUNT, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh())
        self.heads = nn.ModuleList(nn.Linear(64, 3) for _ in range(3))

    @property
    def device(self):
        return self.heads[0].weight.device

    def forward(self, head, features):
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        return self.heads[head](self.encoder(x))

    @torch.inference_mode()
    def scores(self, head, features):
        probabilities = self(head, np.stack(features)).softmax(dim=-1)
        return (probabilities[:, 2] - probabilities[:, 0]).cpu().numpy()


def fit(model, optimizer, head, examples):
    observations, labels = zip(*examples)
    x = np.stack(observations)
    y = torch.tensor(labels, device=model.device) + 1
    loss = F.cross_entropy(model(head, x), y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())

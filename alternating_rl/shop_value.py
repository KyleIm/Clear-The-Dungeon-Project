"""Shop survival probability model and bounded binary classification update."""

import numpy as np
import torch
from torch import nn

from .candidates import FEATURE_COUNT


class ShopValue(nn.Module):
    def __init__(self, widths):
        super().__init__()
        layers = []
        previous = FEATURE_COUNT
        for width in widths:
            layers.extend((nn.Linear(previous, width), nn.Tanh()))
            previous = width
        self.network = nn.Sequential(*layers, nn.Linear(previous, 1))

    def logits(self, features):
        device = next(self.parameters()).device
        values = torch.as_tensor(features, dtype=torch.float32, device=device)
        return self.network(values).squeeze(-1)

    def forward(self, features):
        return torch.sigmoid(self.logits(features))


def update_shop(model, optimizer, samples, batch_size, rng):
    if not samples:
        raise ValueError('Shop training requires resolved samples')
    # Input order is newest first. Every eligible row retains a nonzero sampling probability.
    ranks = np.arange(len(samples), dtype=np.float64)
    weights = np.exp2(-4 * ranks / max(1, len(samples) - 1))
    indices = rng.choice(len(samples), min(batch_size, len(samples)), replace=False, p=weights / weights.sum())
    batch = [samples[i] for i in indices]
    features = np.stack([row['features'] for row in batch])
    logits = model.logits(features)
    targets = logits.new_tensor([row['target'] for row in batch])
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return dict(shop_loss=float(loss.detach().item()), shop_batch=len(batch),
                shop_batch_successes=sum(row['target'] for row in batch))

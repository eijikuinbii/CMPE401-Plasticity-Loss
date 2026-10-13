"""
Networks used in the experiments.

Both models are plain `nn.Sequential`s with a *distinct* `nn.ReLU` module after
every hidden layer. The diagnostics and mitigations rely on that structure:
they find hidden units by hooking each ReLU, and they find a unit's incoming /
outgoing weights by looking at the nearest Linear/Conv2d before / after it.
"""

from __future__ import annotations

import torch.nn as nn


def mlp(in_dim: int = 784, hidden: int = 256, depth: int = 3, out_dim: int = 10) -> nn.Sequential:
    """MLP for permuted MNIST: `depth` hidden ReLU layers of width `hidden`."""
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def minatar_qnet(in_channels: int, n_actions: int, conv: int = 16, hidden: int = 128) -> nn.Sequential:
    """The reference MinAtar DQN architecture (Young & Tian, 2019).

    10x10xC grid -> conv(16, 3x3) -> ReLU -> fc(128) -> ReLU -> Q-values.
    `conv` / `hidden` change the widths (defaults = the reference network).
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, conv, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(conv * 8 * 8, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )

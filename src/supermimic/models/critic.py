"""Critic network; parameter names and computation are unchanged."""

from __future__ import annotations
import torch.nn as nn


from .common import RunningNorm, mlp


class Critic(nn.Module):
    def __init__(self, in_dim: int, layer_norm: bool = False):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1, layer_norm=layer_norm)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        # fp32 even under --amp bf16: bf16 resolves values near G_MAX = 10 only to 1/16, about a step reward
        return self.net(self.norm(x)).squeeze(-1)

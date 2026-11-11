"""Shared MLP layers, running statistics and autocast policy."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``.

    Under torch.distributed every rank must call ``update`` together; the batch statistics are merged
    across ranks so all ranks keep identical normalizers. Dims pinned with ``fix`` keep their values.
    """

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("fixed", torch.zeros(dim, dtype=torch.bool), persistent=False)
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def fix(self, idx, mean: torch.Tensor, std: torch.Tensor):
        """Pin dims ``idx`` to ``mean`` / ``std``; ``update`` leaves them unchanged."""
        self.mean[idx] = mean
        self.var[idx] = std**2
        self.fixed[idx] = True

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        # rows with a non-finite entry are left out, since one would spread to every rank's statistics
        ok = torch.isfinite(x).all(-1, keepdim=True)
        x = torch.where(ok, x, 0.0)
        n = ok.sum().float()
        m = x.sum(0) / n.clamp(min=1)
        v = (((x - m) ** 2) * ok).sum(0) / n.clamp(min=1)
        if dist.is_initialized() and dist.get_world_size() > 1:
            s = torch.cat((n.view(1), m * n, (v + m * m) * n)).double()
            dist.all_reduce(s)
            mg = s[1 : 1 + m.numel()] / s[0].clamp(min=1)
            n, m, v = s[0].float(), mg.float(), (s[1 + m.numel() :] / s[0].clamp(min=1) - mg * mg).clamp(min=0).float()
        tot = (self.count + n).clamp(min=1)
        delta = m - self.mean
        mean = self.mean + delta * n / tot
        var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.mean.copy_(torch.where(self.fixed, self.mean, mean))
        self.var.copy_(torch.where(self.fixed, self.var, var))
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)

AMP_DTYPE: torch.dtype | None = None

def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()

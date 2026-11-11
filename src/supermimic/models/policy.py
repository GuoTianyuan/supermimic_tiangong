"""Policy network; parameter names and computation are unchanged."""

from __future__ import annotations
import torch
import torch.nn as nn
from supermimic.tasks.tracking.mdp import NUM_J, soft_clip


from .common import RunningNorm, mlp, run_mlp


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh. With ``soft_clip_actions``
    the output passes :func:`supermimic.tasks.tracking.mdp.actions.soft_clip`, so the simulator's hard clip never acts and
    imagination sees the same bounded action."""

    def __init__(self, obs_dim: int, goal_dim: int, hist: int = 10, z_dim: int = 32, soft_clip_actions: bool = False):
        super().__init__()
        self.hist, self.soft_clip_actions = hist, soft_clip_actions
        self.enc = mlp(goal_dim, [2048, 1024, 512, 256], z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, [2048, 1024, 512, 256], NUM_J)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last."""
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        o = self.obs_norm(obs_hist).flatten(-2)
        a = run_mlp(self.dec, torch.cat((z, o), -1))
        return soft_clip(a) if self.soft_clip_actions else a

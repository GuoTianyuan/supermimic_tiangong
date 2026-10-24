"""World model, policy (motion encoder + shared decoder), and critic (paper Table A.4)."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from .mdp import INV_DIM, NUM_J, XI_DIM, invariant_features, soft_clip
from .rotations import so3_exp, yaw_mat


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


WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J

# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments.

    With ``a_prev`` the previous action is an input as well: under actuator delay its target still
    drives the first physics substeps of the control step.
    """

    def __init__(self, hidden=(2048, 2048, 2048), xi_dim: int = XI_DIM, a_prev: bool = False):
        super().__init__()
        self.a_prev = a_prev
        self.in_dim = INV_DIM + NUM_J * (2 if a_prev else 1) + xi_dim
        self.net = mlp(self.in_dim, list(hidden), WM_OUT, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)

    def inputs(self, S, a, xi_n, a_prev=None):
        if self.a_prev:
            return torch.cat((invariant_features(S), a, a_prev, xi_n), -1)
        return torch.cat((invariant_features(S), a, xi_n), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ R1),
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, S, a, xi_n, a_prev=None):
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n, a_prev)))
        d = y * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return (p1, R1, v1, w1, q + dq, qd + dqd)

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (ph - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ Rh),
                (Rt @ (vh - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (wh - w).unsqueeze(-1)).squeeze(-1),
                qh - q,
                qdh - qd,
            ),
            -1,
        )


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh. With ``soft_clip_actions``
    the output passes :func:`supermimic.mdp.soft_clip`, so the simulator's hard clip never acts and
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


class Critic(nn.Module):
    def __init__(self, in_dim: int, layer_norm: bool = False):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1, layer_norm=layer_norm)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        # fp32 even under --amp bf16: bf16 resolves values near G_MAX = 10 only to 1/16, about a step reward
        return self.net(self.norm(x)).squeeze(-1)

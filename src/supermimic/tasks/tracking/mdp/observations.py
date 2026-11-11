"""Actor observations, shared history noise and privileged Critic features."""

from __future__ import annotations
import torch
from supermimic.utils.rotations import mat_to_6d, yaw_mat


from .state import invariant_features

OBS_DIM = 29 + 29 + 3 + 3 + 29 + 6

OBS_A_PREV = 29 + 29 + 3 + 3  # where a_prev starts in obs()

OBS_NOISE = torch.cat(
    (
        torch.full((29,), 0.01),
        torch.full((29,), 0.5),
        torch.full((3,), 0.2),
        torch.full((3,), 0.05),
        torch.zeros(29),
        torch.full((6,), 0.05),
    )
)

OBS_SCALE = torch.cat(
    (torch.ones(29), torch.full((29,), 0.1), torch.full((3,), 0.25), torch.ones(3), torch.ones(29), torch.ones(6))
)

CRITIC_EXTRA = 29 + 14 * 3 + 3 + 6  # followed by xi

def obs(self, S, a_prev, xi, f) -> torch.Tensor:
    """Noise-free proprioceptive observation (+ reference-relative root orientation)."""
    p, R, v, w, q, qd = S
    Rt = R.transpose(-1, -2)
    dq0 = xi[..., 1:30]
    g_b = -R[..., 2, :]  # R^T (0,0,-1)
    rel = mat_to_6d(Rt @ self.motion.root_R[f])
    return torch.cat((q - self.default_q - dq0, qd, (Rt @ w.unsqueeze(-1)).squeeze(-1), g_b, a_prev, rel), -1)

def add_noise(self, o: torch.Tensor) -> torch.Tensor:
    return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

def add_history_noise(self, o: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Noise (B,H,D) history, sharing one draw across copies of a padded frame.

    ``source`` comes from ReplayBuffer.sample_starts. Genuine predecessor
    frames keep independent noise; reset/buffer-boundary padding shares the
    oldest retained frame's draw, including the all-padded reset case.
    """
    noisy = self.add_noise(o)
    return noisy.gather(1, source.unsqueeze(-1).expand_as(noisy))

def critic_features(self, S, f, xi, ref_body_p=None) -> torch.Tensor:
    """Critic input: invariant state, goal, sim context, and robot-vs-reference errors."""
    m = self.motion
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Ht = H.transpose(-1, -2)
    rp, rR = m.root_p[f], m.root_R[f]
    bp = self.kin.forward(p, R, v, w, q, qd)[0]
    rbp = m.body_reference(f, positions_only=True)[0] if ref_body_p is None else ref_body_p
    dR = yaw_mat(R @ rR.transpose(-1, -2))
    anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
    ref_bp = anchor.unsqueeze(-2) + ((dR.unsqueeze(-3) @ (rbp - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1))
    e_body = (Ht.unsqueeze(-3) @ (ref_bp - bp).unsqueeze(-1)).squeeze(-1).flatten(-2)
    e_root = (Ht @ (rp - p).unsqueeze(-1)).squeeze(-1)
    rel = mat_to_6d(R.transpose(-1, -2) @ rR)
    return torch.cat((invariant_features(S), m.q[f] - q, e_body, e_root, rel, xi), -1)

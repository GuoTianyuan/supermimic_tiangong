"""State layout, invariant features and physical-context scaling."""

from __future__ import annotations
import torch
from supermimic.utils.rotations import mat_to_6d, quat_to_mat, mat_to_quat, yaw_mat


STATE_DIM = 71

XI_DIM = 33

NUM_J = 29

INV_DIM = 1 + 6 + 3 + 3 + 29 + 29

def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)

def unflatten_state(x: torch.Tensor):
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13:42], x[..., 42:71]

def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )

def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    """Base context to unit scale; entries past XI_DIM (src/supermimic/envs/domain_randomization.py) are stored normalized."""
    base = torch.cat(((xi[..., :1] - 0.75) / 0.45, xi[..., 1:30] / 0.01, xi[..., 30:33] / 0.05), -1)
    return torch.cat((base, xi[..., XI_DIM:]), -1) if xi.shape[-1] > XI_DIM else base

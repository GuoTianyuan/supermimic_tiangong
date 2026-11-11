"""WorldModel network; parameter names and computation are unchanged."""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from supermimic.tasks.tracking.mdp import INV_DIM, NUM_J, XI_DIM, invariant_features
from supermimic.utils.rotations import so3_exp, yaw_mat


from .common import RunningNorm, mlp, run_mlp

WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J

class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments.

    With ``a_prev`` the previous action is an input as well: under actuator delay its target still
    drives the first physics substeps of the control step. ``state_delta`` retains
    the original 70-output predictor. ``velocity_integrated`` predicts 35 velocity
    increments and integrates pose using the control dt and physics substep count.
    """

    def __init__(self, hidden=(2048, 2048, 2048), xi_dim: int = XI_DIM, a_prev: bool = False,
                 prediction: str = "state_delta", control_dt: float | None = None,
                 physics_substeps: int | None = None):
        super().__init__()
        if prediction not in ("state_delta", "velocity_integrated"):
            raise ValueError(f"Unknown WM prediction mode: {prediction}")
        if prediction == "velocity_integrated":
            if control_dt is None or not math.isfinite(control_dt) or control_dt <= 0:
                raise ValueError("velocity_integrated requires a positive finite control_dt")
            if type(physics_substeps) is not int or physics_substeps < 1:
                raise ValueError("velocity_integrated requires a positive integer physics_substeps")
        self.prediction, self.control_dt = prediction, control_dt
        self.substep_coeff = ((physics_substeps + 1) / (2 * physics_substeps)
                              if prediction == "velocity_integrated" else None)
        out_dim = WM_OUT if prediction == "state_delta" else 6 + NUM_J
        self.a_prev = a_prev
        self.in_dim = INV_DIM + NUM_J * (2 if a_prev else 1) + xi_dim
        self.net = mlp(self.in_dim, list(hidden), out_dim, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(out_dim, eps=1e-4)
        # Keep the 70-channel TG loss scaling in both ablations. The integrated
        # predictor has only 35 outputs, so it needs separate error statistics.
        # None adds no buffers/parameters: legacy checkpoints load strictly.
        self.error_norm = (RunningNorm(WM_OUT, eps=1e-4)
                           if prediction == "velocity_integrated" else None)

    def output_normalizers(self):
        return (self.out_norm,) if self.error_norm is None else (self.out_norm, self.error_norm)

    @torch.no_grad()
    def update_output_normalization(self, S, S1):
        delta = self.targets(S, S1)
        if self.error_norm is None:
            self.out_norm.update(delta)
        else:
            self.out_norm.update(torch.cat((delta[..., 6:12], delta[..., 12 + NUM_J:]), -1))
            self.error_norm.update(delta)

    def error_std(self):
        """Scale for the unchanged 70 physical-state error channels."""
        return (self.out_norm if self.error_norm is None else self.error_norm).std()

    def inputs(self, S, a, xi_n, a_prev=None):
        if self.a_prev:
            return torch.cat((invariant_features(S), a, a_prev, xi_n), -1)
        return torch.cat((invariant_features(S), a, xi_n), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        from supermimic.utils.rotations import so3_log

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
        if self.prediction == "velocity_integrated":
            dv, dw, dqd = d.split((3, 3, NUM_J), -1)
            dv_w = (R @ dv.unsqueeze(-1)).squeeze(-1)
            dw_w = (R @ dw.unsqueeze(-1)).squeeze(-1)
            w_b = (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
            dt, c = self.control_dt, self.substep_coeff
            # Constant-acceleration semi-implicit substep approximation. This
            # constrains pose/velocity consistency, not rigid-body dynamics.
            p1 = p + dt * (v + c * dv_w)
            R1 = R @ so3_exp(dt * (w_b + c * dw))
            return p1, R1, v + dv_w, w + dw_w, q + dt * (qd + c * dqd), qd + dqd
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
        from supermimic.utils.rotations import so3_log

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

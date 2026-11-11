"""Hard simulator termination and analytic sigmoid survival for imagination."""

from __future__ import annotations
import torch


def termination(self, S, f) -> torch.Tensor:
    p, R = S[0], S[1]
    rp, rR = self.motion.root_p[f], self.motion.root_R[f]
    e_z = (rp[..., 2] - p[..., 2]).abs()
    e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
    return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

def soft_termination(self, S, f, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
    """Differentiable termination probability for imagined rollouts."""
    p, R = S[0], S[1]
    rp, rR = self.motion.root_p[f], self.motion.root_R[f]
    e_z = (rp[..., 2] - p[..., 2]).abs()
    e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
    alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt))
    return 1 - alive

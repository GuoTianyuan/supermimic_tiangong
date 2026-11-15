"""Tracking and smoothness reward terms, shared by real and imagined steps."""

from __future__ import annotations
import torch
from supermimic.utils.rotations import rot_angle_sq, yaw_mat


from supermimic.motions.library import CONTROL_DT

def reward(self, S, f, a, a_prev, terms: bool = False, qd_prev=None, ref_bodies=None):
    """Per-step reward (already multiplied by dt) comparing S with reference frame f; ``qd_prev``: the joint
    velocities one control step before S (needed by ``w_joint_acc``)."""
    c, m = self.cfg, self.motion
    p, R, v, w, q, qd = S
    bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
    rp, rR = m.root_p[f], m.root_R[f]
    rbp, rbR, rbv, rbw = m.body_reference(f) if ref_bodies is None else ref_bodies
    # XY + yaw alignment; optionally align root Z while retaining relative link heights.
    dR = yaw_mat(R @ rR.transpose(-1, -2))
    anchor_z = p[..., 2:3] if c.body_pos_z_mode == "relative_root" else rp[..., 2:3]
    anchor = torch.cat((p[..., :2], anchor_z), -1)
    ref_bp = anchor.unsqueeze(-2) + ((dR.unsqueeze(-3) @ (rbp - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1))
    ref_bR = dR.unsqueeze(-3) @ rbR
    e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
    e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
    e_lv = ((rbv - bv) ** 2).sum(-1).mean(-1)
    e_av = ((rbw - bw) ** 2).sum(-1).mean(-1)
    root_error_sq = (rp - p).square()
    e_rp = root_error_sq.sum(-1)
    # Keep XYZ error in physical diagnostics, even when Z has reduced reward weight.
    e_rp_reward = e_rp if c.root_pos_z_weight == 1.0 else (
        root_error_sq[..., :2].sum(-1) + c.root_pos_z_weight * root_error_sq[..., 2])
    e_ro = rot_angle_sq(rR, R)
    r_track = (
        torch.exp(-e_pos / c.body_pos_std**2)
        + torch.exp(-e_ori / c.body_ori_std**2)
        + torch.exp(-e_lv / c.body_linvel_std**2)
        + torch.exp(-e_av / c.body_angvel_std**2)
        + 0.5 * torch.exp(-e_rp_reward / c.root_pos_std**2)
        + 0.5 * torch.exp(-e_ro / c.root_ori_std**2)
    )
    r_rate = ((a - a_prev) ** 2).sum(-1)
    r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
    pen_smooth = c.w_action_rate * r_rate
    r_acc = None
    if c.w_joint_acc and qd_prev is not None:
        r_acc = (((qd - qd_prev) / CONTROL_DT) ** 2).sum(-1)
        pen_smooth = pen_smooth + c.w_joint_acc * r_acc
    r = (r_track + pen_smooth + c.w_joint_limit * r_lim) * CONTROL_DT
    if terms:
        relative_z = (rbp[..., 2] - rp[..., 2:3]) - (bp[..., 2] - p[..., 2:3])
        # Reference-Z position error remains comparable to the baseline metric.
        ref_z_error = (rbp[..., 2] - bp[..., 2]).square().mean(-1)
        ref_z_pos = (ref_bp[..., :2] - bp[..., :2]).square().sum(-1).mean(-1) + ref_z_error
        return r, {
            "track": r_track * CONTROL_DT,
            "pen_smooth": pen_smooth * CONTROL_DT,
            "e_body_pos": e_pos,
            "e_body_pos_ref_z": ref_z_pos,
            "e_body_z_relative": relative_z.square().mean(-1),
            "e_body_ori": e_ori,
            "e_root_pos": e_rp,
            "e_root_ori": e_ro,
            "e_root_xy": ((rp[..., :2] - p[..., :2]) ** 2).sum(-1),
            "e_root_z": (rp[..., 2] - p[..., 2]) ** 2,
            "e_body_linvel": e_lv,
            "e_body_angvel": e_av,
            "e_joint_pos": (m.q[f] - q).square().mean(-1),
            "e_joint_vel": (m.qd[f] - qd).square().mean(-1),
            "reward_body_pos": torch.exp(-e_pos / c.body_pos_std**2) * CONTROL_DT,
            "reward_body_ori": torch.exp(-e_ori / c.body_ori_std**2) * CONTROL_DT,
            "reward_body_linvel": torch.exp(-e_lv / c.body_linvel_std**2) * CONTROL_DT,
            "reward_body_angvel": torch.exp(-e_av / c.body_angvel_std**2) * CONTROL_DT,
            "reward_root_pos": 0.5 * torch.exp(-e_rp_reward / c.root_pos_std**2) * CONTROL_DT,
            "reward_root_ori": 0.5 * torch.exp(-e_ro / c.root_ori_std**2) * CONTROL_DT,
            "pen_action_rate": c.w_action_rate * r_rate * CONTROL_DT,
            "pen_joint_acc": c.w_joint_acc * (r_acc if r_acc is not None else torch.zeros_like(r_rate)) * CONTROL_DT,
            "pen_joint_limit": c.w_joint_limit * r_lim * CONTROL_DT,
            "action_rate": r_rate,
            "joint_acc": r_acc if r_acc is not None else torch.zeros_like(r_rate),
            "joint_limit": r_lim,
        }
    return r

"""Read-only real-simulator reward and tracking diagnostics, reduced once per log window."""
from __future__ import annotations

import torch
import torch.distributed as dist

from .mdp import MAX_STEP_REWARD

REWARD_KEYS = ("reward_body_pos", "reward_body_ori", "reward_body_linvel", "reward_body_angvel",
               "reward_root_pos", "reward_root_ori", "pen_action_rate", "pen_joint_acc", "pen_joint_limit")
ERRORS = {
    "error_root_xy_m": "e_root_xy", "error_root_z_m": "e_root_z", "error_root_pos_m": "e_root_pos",
    "error_root_ori_rad": "e_root_ori", "error_body_pos_m": "e_body_pos", "error_body_ori_rad": "e_body_ori",
    "error_body_linvel_mps": "e_body_linvel", "error_body_angvel_radps": "e_body_angvel",
    "error_joint_pos_rad": "e_joint_pos", "error_joint_vel_radps": "e_joint_vel",
}
POOL_FIELDS = ("pool_active", "pool_resident", "pool_legacy", "pool_frames", "pool_gpu_gib", "pool_cpu_gib",
               "pool_refresh_s", "pool_loaded", "pool_catalog", "pool_selected", "pool_starts_seen",
               "pool_starts_drawn", "pool_refreshes")
EXTRA_FIELDS = ("wm_nse1", "success_frac", "episodes_ended", "episodes_failed", "ended_clip", "ended_time",
                "track_fraction", "sim_reward_normalized", "imag_reward_normalized", *REWARD_KEYS,
                *ERRORS, "action_delta_rms", "joint_acc_rms", *POOL_FIELDS)


class RealMetrics:
    def __init__(self, device):
        self.keys = (*REWARD_KEYS, *ERRORS.values(), "action_rate", "joint_acc", "track", "net")
        self.sums = torch.zeros(len(self.keys)+1, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, out):
        terms = {**out["terms"], "net": out["r"]}
        self.sums[:-1].add_(torch.stack([terms[k].double().sum() for k in self.keys]))
        self.sums[-1].add_(out["r"].numel())

    def flush(self):
        values = self.sums.clone()
        if dist.is_initialized():
            dist.all_reduce(values)
        values = values.cpu().tolist()
        d = dict(zip(self.keys, [v/max(values[-1], 1) for v in values[:-1]]))
        self.sums.zero_()
        return {**{k: d[k] for k in REWARD_KEYS},
                **{k: max(d[v], 0)**.5 for k, v in ERRORS.items()},
                "action_delta_rms": max(d["action_rate"]/29, 0)**.5,
                "joint_acc_rms": max(d["joint_acc"]/29, 0)**.5,
                "track_fraction": d["track"]/MAX_STEP_REWARD,
                "sim_reward_normalized": d["net"]/MAX_STEP_REWARD}

"""Read-only simulator, WM and critic diagnostics, reduced once per log window."""
from __future__ import annotations

import torch
import torch.distributed as dist

from supermimic.tasks.tracking.mdp import MAX_STEP_REWARD

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
CRITIC_FIELDS = ("fit_weighted_huber", "fit_weighted_mse", "fit_survival_mean")
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


class CriticMetrics:
    """Aggregate pre-update fitting errors over all minibatches/epochs/ranks.

    These are errors encountered while fitting, not a separate post-fit eval.
    Reduce numerators and survival mass, not unweighted minibatch means.
    """
    def __init__(self, device):
        self.sums = torch.zeros(4, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, value, target, weight, huber):
        value, target, weight, huber = (x.detach() for x in (value, target, weight, huber))
        self.sums[:3].add_(torch.stack(((weight * huber).sum(),
                                      (weight * (value - target).square()).sum(), weight.sum())).double())
        self.sums[3].add_(value.numel())

    def flush(self):
        sums = self.sums.clone()
        if dist.is_initialized():
            dist.all_reduce(sums)
        huber, mse, mass, count = sums.cpu().tolist()
        self.sums.zero_()
        return dict(zip(CRITIC_FIELDS, (huber / max(mass, 1e-6), mse / max(mass, 1e-6),
                                       mass / max(count, 1))))


# Layout of WorldModel.state_error, not the network output layout: both prediction
# parameterizations are supervised on these same six physical state groups.
WM_GROUPS = ("root_pos", "root_ori", "root_linvel", "root_angvel", "joint_pos", "joint_vel")
WM_UNITS = ("m", "rad", "m/s", "rad/s", "rad", "rad/s")
WM_GROUP_METRICS = ("loss_contribution", "loss_share", "normalized_mse", "raw_rmse", "scale_mean", "floor_fraction")
WM_FIELDS = ("wm_fit_loss_per_step", "wm_fit_valid_fraction",
             *(f"wm_{group}_{metric}" for group in WM_GROUPS for metric in WM_GROUP_METRICS))
WM_TB_TAGS = {
    "wm_fit_loss_per_step": "2_world_model/fit_all_batches/loss_per_step",
    "wm_fit_valid_fraction": "2_world_model/fit_all_batches/valid_fraction",
    **{f"wm_{group}_{metric}": f"2_world_model/groups/{group}/{metric}"
       for group in WM_GROUPS for metric in WM_GROUP_METRICS},
}


@torch.no_grad()
def wm_error_summary(raw_error, normalized_sq, mask, scale, floor):
    """Compact detached sufficient statistics; no rollout or host synchronization.

    Input B x H x D. Padding is zero-weighted exactly as in the training loss.
    The loss denominator remains B*H*D, while physical RMS uses valid samples.
    All state axes in one sample share the same mask.
    """
    valid = mask.sum()
    weight = mask.unsqueeze(-1)
    return torch.cat(((normalized_sq.detach() * weight).sum((0, 1)),
                      (raw_error.detach().square() * weight).sum((0, 1)),
                      scale.detach() * valid, (scale.detach() <= floor).to(scale.dtype) * valid,
                      valid.reshape(1), mask.new_full((1,), mask.numel())))


class WorldModelMetrics:
    """All minibatches/updates/ranks, evaluated BEFORE each optimizer update.

    The group contributions add to the masked training loss / horizon. This is
    a training diagnostic, not held-out evaluation or a gradient attribution.
    Per-channel scale is a valid-sample-weighted mean if RunningNorm changes.
    """
    def __init__(self, device, joint_names):
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("WM diagnostic joint names must be unique")
        widths = (3, 3, 3, 3, len(joint_names), len(joint_names))
        self.groups = []
        self.channel_names = []
        offset = 0
        for group, width in zip(WM_GROUPS, widths):
            self.groups.append((group, offset, offset + width))
            labels = joint_names if group.startswith("joint_") else ("x", "y", "z")
            self.channel_names.extend(f"{group}/{label}" for label in labels)
            offset += width
        self.dim = offset
        self.sums = torch.zeros(4 * offset + 2, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, summary):
        if summary.shape != self.sums.shape:
            raise ValueError(f"WM diagnostic layout mismatch: {summary.shape} vs {self.sums.shape}")
        self.sums.add_(summary.detach().double())

    def flush(self):
        sums = self.sums.clone()
        if dist.is_initialized():
            dist.all_reduce(sums)
        values = sums.cpu().tolist()
        self.sums.zero_()
        d = self.dim
        normalized, raw, scales, floors = (values[i*d:(i+1)*d] for i in range(4))
        valid, slots = values[-2:]
        norm_total = sum(normalized)
        loss_den = max(slots * d, 1)
        valid_den = max(valid, 1)
        result = {"wm_fit_loss_per_step": norm_total / loss_den,
                  "wm_fit_valid_fraction": valid / max(slots, 1)}
        for group, start, end in self.groups:
            width = end - start
            norm_sum = sum(normalized[start:end])
            result.update({
                f"wm_{group}_loss_contribution": norm_sum / loss_den,
                f"wm_{group}_loss_share": norm_sum / norm_total if norm_total > 0 else 0.,
                f"wm_{group}_normalized_mse": norm_sum / (valid_den * width),
                f"wm_{group}_raw_rmse": (max(sum(raw[start:end]), 0.) / (valid_den * width)) ** .5,
                f"wm_{group}_scale_mean": sum(scales[start:end]) / (valid_den * width),
                f"wm_{group}_floor_fraction": sum(floors[start:end]) / (valid_den * width),
            })
        channels = {}
        for i, name in enumerate(self.channel_names):
            prefix = f"2_world_model/channels/{name}"
            channels.update({f"{prefix}/normalized_mse": normalized[i] / valid_den,
                             f"{prefix}/raw_rmse": (max(raw[i], 0.) / valid_den) ** .5,
                             f"{prefix}/loss_share": normalized[i] / norm_total if norm_total > 0 else 0.,
                             f"{prefix}/scale": scales[i] / valid_den,
                             f"{prefix}/floor_fraction": floors[i] / valid_den})
        return result, channels

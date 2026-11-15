"""Tracking reward and termination parameters shared by simulation and imagination."""

from dataclasses import asdict, dataclass, replace
import math
from supermimic.motions.library import CONTROL_DT

MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT

@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_pos_z_weight: float = 1.0  # coefficient of squared Z error, not of the whole reward
    body_pos_z_mode: str = "reference"  # reference absolute Z, or relative_root height
    root_ori_std: float = 0.4
    w_action_rate: float = -0.4
    w_joint_acc: float = 0.0  # sum of squared joint accelerations over the control step [(rad/s^2)^2]
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8

    def __post_init__(self):
        if not math.isfinite(self.root_pos_z_weight) or self.root_pos_z_weight < 0:
            raise ValueError("root_pos_z_weight must be finite and nonnegative")
        if self.body_pos_z_mode not in ("reference", "relative_root"):
            raise ValueError("body_pos_z_mode must be reference or relative_root")

def reward_cfg(dynamics: str = "default") -> RewardCfg:
    """100 Hz smoothing: action-rate -0.4 and DexEVT acceleration -3e-6; tracking unchanged."""
    return RewardCfg(w_joint_acc=-3e-6) if dynamics == "dexevt" else RewardCfg()


def checkpoint_reward_cfg(saved_args: dict) -> RewardCfg:
    """Restore saved task semantics; checkpoints predating the flags keep absolute Z."""
    cfg = reward_cfg(saved_args.get("dynamics", "default"))
    saved = saved_args.get("reward_cfg", {})
    overrides = {k: saved_args[k] for k in ("root_pos_z_weight", "body_pos_z_mode")
                 if saved_args.get(k) is not None}
    return replace(cfg, **(saved | overrides))


def configure_reward(args, checkpoint=None) -> RewardCfg:
    """Select a fresh-run task or restore it exactly on ordinary resume."""
    previous = checkpoint_reward_cfg(checkpoint["args"]) if checkpoint else None
    cfg = previous if previous and checkpoint["args"].get("dynamics", "default") == args.dynamics else reward_cfg(args.dynamics)
    choices = {}
    for key in ("root_pos_z_weight", "body_pos_z_mode"):
        value = getattr(args, key, None)
        choices[key] = getattr(cfg, key) if value is None else value
        if previous and choices[key] != getattr(previous, key):
            raise ValueError(f"Cannot resume with different {key}; start a separate reward ablation")
        setattr(args, key, choices[key])
    cfg = replace(cfg, **choices)
    args.reward_cfg = asdict(cfg)
    return cfg

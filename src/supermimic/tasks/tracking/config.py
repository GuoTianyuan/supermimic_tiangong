"""Tracking reward and termination parameters shared by simulation and imagination."""

from dataclasses import dataclass
from supermimic.motions.library import CONTROL_DT

MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT

@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.4
    w_joint_acc: float = 0.0  # sum of squared joint accelerations over the control step [(rad/s^2)^2]
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8

def reward_cfg(dynamics: str = "default") -> RewardCfg:
    """100 Hz smoothing: action-rate -0.4 and DexEVT acceleration -3e-6; tracking unchanged."""
    return RewardCfg(w_joint_acc=-3e-6) if dynamics == "dexevt" else RewardCfg()

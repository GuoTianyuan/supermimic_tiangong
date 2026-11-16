"""Bind tracking terms to the same kinematics, motion library and reward configuration."""

from __future__ import annotations

import torch
from supermimic.robots.kinematics import G1Kinematics
from supermimic.motions.library import CONTROL_DT, MotionLib


from supermimic.tasks.tracking.config import RewardCfg
from .observations import OBS_NOISE, OBS_SCALE, obs, add_noise, add_history_noise, critic_features
from .rewards import reward
from .terminations import termination, soft_termination


class MDP:
    def __init__(self, kin: G1Kinematics, motion: MotionLib, default_q: torch.Tensor, cfg: RewardCfg = RewardCfg()):
        self.kin, self.motion, self.cfg = kin, motion, cfg
        self.max_step_reward = cfg.validate().max_step_reward(CONTROL_DT)
        self.default_q = default_q
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        dev = default_q.device
        self.obs_noise = OBS_NOISE.to(dev)
        self.obs_scale = OBS_SCALE.to(dev)

    # Binding these functions as methods preserves the MDP interface used by both environments.
    obs = obs
    add_noise = add_noise
    add_history_noise = add_history_noise
    reward = reward
    termination = termination
    soft_termination = soft_termination
    critic_features = critic_features

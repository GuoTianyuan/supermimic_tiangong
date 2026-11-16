"""Tracking reward and termination parameters shared by simulation and imagination."""

from dataclasses import asdict, dataclass, fields
import math
from .tracking_env_cfg import RewardsCfg, TerminationsCfg

MAX_STEP_REWARD = 0.05  # legacy default bound at 100 Hz; runtime uses cfg.max_step_reward(dt)

@dataclass
class RewardCfg:
    # Flat compatibility format for existing checkpoints; edit grouped declarations in tracking_env_cfg.py.
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_linvel: float = 1.0
    w_body_angvel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
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

    def max_step_reward(self, control_dt):
        return sum(getattr(self, "w_" + name) for name in TRACKING_TERMS) * control_dt

    def validate(self):
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Reward/termination values must be finite")
        for name in TRACKING_TERMS:
            if getattr(self, "w_" + name) < 0 or getattr(self, name + "_std") <= 0:
                raise ValueError(f"{name}: weight must be nonnegative and std positive")
        if self.max_step_reward(1.0) <= 0:
            raise ValueError("At least one positive tracking weight is required")
        if any(getattr(self, "w_" + name) > 0 for name in PENALTY_TERMS):
            raise ValueError("Penalty weights must be <= 0")
        if not 0 < self.soft_limit_factor <= 1 or self.term_height <= 0 or self.term_tilt <= 0:
            raise ValueError("Invalid soft limits or termination thresholds")
        return self


TRACKING_TERMS = ("body_pos", "body_ori", "body_linvel", "body_angvel", "root_pos", "root_ori")
PENALTY_TERMS = ("action_rate", "joint_acc", "joint_limit")


def from_task_cfg(rewards: RewardsCfg, terminations: TerminationsCfg) -> RewardCfg:
    values = {"soft_limit_factor": rewards.soft_limit_factor, **asdict(terminations)}
    for name in TRACKING_TERMS:
        term = getattr(rewards, name)
        values.update({"w_" + name: term.weight, name + "_std": term.std})
    for name in PENALTY_TERMS:
        values["w_" + name] = getattr(rewards, name).weight
    return RewardCfg(**values).validate()


def reward_cfg(dynamics: str = "default", saved: dict | None = None) -> RewardCfg:
    """Legacy calls retain their values; newer checkpoints restore all declared terms."""
    rewards = RewardsCfg()
    if dynamics == "dexevt":
        rewards.joint_acc.weight = -3e-6
    cfg = from_task_cfg(rewards, TerminationsCfg())
    if saved is not None:
        unknown = set(saved) - {f.name for f in fields(RewardCfg)}
        if unknown:
            raise ValueError(f"Reward settings require another implementation: {sorted(unknown)}")
        cfg = RewardCfg(**{**asdict(cfg), **saved})
    return cfg.validate()

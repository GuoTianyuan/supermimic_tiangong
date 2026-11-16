"""Readable tracking task settings; no simulator imports or initialization.

Like BeyondMimic, declarations live here and tensor formulas live in mdp/.
Names on leaf fields retain the existing command-line/checkpoint spelling.
"""
from dataclasses import dataclass, field
from supermimic.utils.paths import REPO_ROOT


@dataclass
class ExponentialRewardCfg:
    weight: float
    std: float  # exp(-mean_squared_error / std**2)


@dataclass
class PenaltyCfg:
    weight: float


@dataclass
class RewardsCfg:
    # Signed weights BEFORE multiplication by the control period (0.01 seconds).
    body_pos: ExponentialRewardCfg = field(default_factory=lambda: ExponentialRewardCfg(1.0, 0.3))
    body_ori: ExponentialRewardCfg = field(default_factory=lambda: ExponentialRewardCfg(1.0, 0.4))
    body_linvel: ExponentialRewardCfg = field(default_factory=lambda: ExponentialRewardCfg(1.0, 1.0))
    body_angvel: ExponentialRewardCfg = field(default_factory=lambda: ExponentialRewardCfg(1.0, 3.14))
    root_pos: ExponentialRewardCfg = field(default_factory=lambda: ExponentialRewardCfg(0.5, 0.3))
    root_ori: ExponentialRewardCfg = field(default_factory=lambda: ExponentialRewardCfg(0.5, 0.4))
    action_rate: PenaltyCfg = field(default_factory=lambda: PenaltyCfg(-0.4))
    joint_acc: PenaltyCfg = field(default_factory=lambda: PenaltyCfg(0.0))
    joint_limit: PenaltyCfg = field(default_factory=lambda: PenaltyCfg(-2.0))
    soft_limit_factor: float = 0.9


@dataclass
class TerminationsCfg:
    term_height: float = 0.25  # absolute root-height error, meters
    term_tilt: float = 0.8     # absolute difference of R[2, 2], not radians


@dataclass
class SceneCfg:
    robot: str = "g1"
    num_envs: int = 4096  # per rank / GPU, as are all training batch sizes
    dynamics: str | None = None  # new: default; resume: saved choice


@dataclass
class CommandsCfg:
    motion_dir: str | None = None
    motion_exclude: str = ""
    goal_frames: int = 10
    goal_start: int | None = None   # new: 1; resume: saved/legacy value
    goal_stride: int | None = None  # new: 5; resume: saved/legacy value
    goal_drop: str | None = None
    allow_reference_timing_change: bool = False
    reset_temp: float = 25.0
    motion_active_ratio: float = 1.0
    motion_refresh_iters: int = 1500
    motion_replace_ratio: float = 0.25  # fraction of catalog, not active pool
    motion_stats_cache: str = str(REPO_ROOT / ".cache" / "motion_pool")


@dataclass
class ObservationsCfg:
    hist: int = 10
    act_norm: str | None = None  # new: ref; old checkpoint: running


@dataclass
class ActionsCfg:
    action_noise: float = 0.1
    act_soft_clip: bool | None = None  # new: True; old checkpoint: False
    reset_a_prev: str | None = None   # new: pose; old checkpoint: zero


@dataclass
class EventsCfg:
    # Ranges and physical implementation: envs/domain_randomization.py::DRCfg.
    dr: str = ""
    dr_curriculum: str = ""
    dr_ramp: int = 2000


@dataclass
class TrackingEnvCfg:
    scene: SceneCfg = field(default_factory=SceneCfg)
    commands: CommandsCfg = field(default_factory=CommandsCfg)
    observations: ObservationsCfg = field(default_factory=ObservationsCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    events: EventsCfg = field(default_factory=EventsCfg)
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)

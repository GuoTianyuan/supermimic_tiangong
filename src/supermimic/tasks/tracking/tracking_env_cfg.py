"""Readable tracking task settings; no simulator imports or initialization.

Like BeyondMimic, declarations live here and tensor formulas live in mdp/.
Names on leaf fields retain the existing command-line/checkpoint spelling.
"""
from dataclasses import dataclass, field, replace
import math
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
    root_pos_z_weight: float = 1.0  # coefficient of Z error squared in root-position exponential
    body_pos_z_mode: str = "reference"  # absolute reference Z, or relative_root
    # Replace link-origin positions/linear velocities for these reward points only.
    # Rotation/angular velocity, FK caches, commands, obs and critic features stay at their original frames.
    body_point_offsets: dict[str, tuple[float, float, float]] = field(default_factory=dict)


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


@dataclass(frozen=True, kw_only=True)
class RandomizationRangesCfg:
    # Base DR: always active during randomized training, even if dr="".
    # Friction is selected by robot: G1 (0.3,1.2); Tiangong3 (0.6,1.6).
    friction: tuple[float, float] = (0.3, 1.2)  # uniform, once per environment
    joint_offset: tuple[float, float] = (-0.01, 0.01)  # rad, resample each episode
    torso_com: tuple[float, float, float] = (0.025, 0.05, 0.05)  # +/- xyz meters, startup

    # Reset-state perturbations, separate from the physical parameter curriculum.
    reset_position: tuple[float, float, float] = (0.05, 0.05, 0.01)  # +/- m
    reset_rotation: tuple[float, float, float] = (0.1, 0.1, 0.2)  # rotation-vector +/- rad
    reset_lin_vel: tuple[float, float, float] = (0.5, 0.5, 0.2)  # +/- m/s
    reset_ang_vel: tuple[float, float, float] = (0.52, 0.52, 0.78)  # +/- rad/s
    reset_joint_pos: float = 0.1  # +/- rad; joint velocities unchanged

    # Extra terms below are enabled by EventsCfg.dr and widened by its curriculum.
    kp_scale: tuple[float, float] = (0.75, 1.25)  # per joint, each episode
    kd_scale: tuple[float, float] = (0.75, 1.25)
    push_interval_s: tuple[float, float] = (1.0, 5.0)
    push_lin_vel: tuple[float, float, float] = (0.5, 0.5, 0.2)  # +/- world m/s
    push_ang_vel: tuple[float, float, float] = (0.52, 0.52, 0.78)  # +/- world rad/s
    mass_scale: tuple[float, float] = (0.85, 1.15)  # mass AND inertia per segment, startup
    armature: tuple[float, float] = (0.002, 0.06)  # kg*m^2, startup
    ankle_armature: tuple[float, float] = (0.01, 0.06)  # DexEVT: same range as armature
    joint_friction: tuple[float, float] = (0.001, 0.6)  # Coulomb friction, N*m, startup
    viscous_friction: bool = False  # DexEVT: same draw also used as N*m*s/rad damping
    max_delay: int = 4  # 0..4 physical substeps; at 100/400 Hz this is 0..10 ms

    # Foot collision shapes, startup. One correlated draw: time constant is
    # log-uniform, impedance endpoints linear. Not a restitution coefficient.
    # Tuples: (solref time constant [s], solimp dmin, solimp dmax).
    # Damping ratio and remaining solimp parameters retain the asset values.
    contact_soft: tuple[float, float, float] = (0.02, 0.9, 0.95)
    contact_hard: tuple[float, float, float] = (0.01, 0.99, 0.99)

    def validate(self):
        for name in ("friction", "joint_offset", "kp_scale", "kd_scale", "push_interval_s",
                     "mass_scale", "armature", "ankle_armature", "joint_friction"):
            value = getattr(self, name)
            if len(value) != 2 or not all(math.isfinite(x) for x in value) or not value[0] < value[1]:
                raise ValueError(f"{name} needs finite lo < hi")
            if name != "joint_offset" and value[0] < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.push_interval_s[0] <= 0:
            raise ValueError("push_interval_s must be positive")
        for name in ("torso_com", "reset_position", "reset_rotation", "reset_lin_vel", "reset_ang_vel",
                     "push_lin_vel", "push_ang_vel"):
            value = getattr(self, name)
            if len(value) != 3 or not all(math.isfinite(x) and x >= 0 for x in value):
                raise ValueError(f"{name} needs three nonnegative finite half widths")
        if not math.isfinite(self.reset_joint_pos) or self.reset_joint_pos < 0:
            raise ValueError("reset_joint_pos must be nonnegative and finite")
        if type(self.max_delay) is not int or not 1 <= self.max_delay <= 4:
            raise ValueError("max_delay must be 1..4; remove delay from dr to disable it")
        if type(self.viscous_friction) is not bool:
            raise ValueError("viscous_friction must be a boolean")
        for value in (self.contact_soft, self.contact_hard):
            if len(value) != 3 or not all(math.isfinite(x) for x in value):
                raise ValueError("contact endpoints need three finite values")
            if not value[0] >= 0.005 or not 0 < value[1] <= value[2] < 1:
                raise ValueError("contact time constant must be >= 2 physical steps (5 ms); 0 < dmin <= dmax < 1")
        return self


def randomization_ranges(robot="g1", dynamics="default"):
    """Historical presets; values stay centralized here, without importing assets."""
    if robot not in ("g1", "tiangong3") or dynamics not in ("default", "dexevt"):
        raise ValueError(f"Unknown robot/dynamics: {robot}/{dynamics}")
    cfg = RandomizationRangesCfg(friction=(0.6, 1.6) if robot == "tiangong3" else (0.3, 1.2))
    if dynamics == "dexevt":
        cfg = replace(cfg, ankle_armature=cfg.armature, viscous_friction=True)
    return cfg.validate()


@dataclass
class EventsCfg:
    # dr selects extra terms; an empty string retains base DR and reset noise.
    # "all" = pd,push,mass,armature,joint_friction,delay; contact must be explicit.
    dr: str = ""
    dr_curriculum: str = ""  # "A,B": extras from A; push/delay from B; empty = full immediately
    dr_ramp: int = 2000  # each stage widens over this many learning iterations, in 5% steps
    ranges: RandomizationRangesCfg = field(default_factory=RandomizationRangesCfg)


@dataclass
class TrackingEnvCfg:
    scene: SceneCfg = field(default_factory=SceneCfg)
    commands: CommandsCfg = field(default_factory=CommandsCfg)
    observations: ObservationsCfg = field(default_factory=ObservationsCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    events: EventsCfg = field(default_factory=EventsCfg)
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)

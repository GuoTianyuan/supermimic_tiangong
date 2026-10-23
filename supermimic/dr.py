"""Opt-in domain randomization for deployment (``train.py --dr``, Tiangong3 only), with the ranges of
DexEVT (cheerleading_competition, tracking_env_cfg.py EventCfg) and mjlab TK3.

Every randomized parameter is also written into the simulator context ``xi``. The world model and the
critic see ``xi`` while the policy does not, so imagination keeps each environment's own dynamics and
the policy has to work across the whole range; a parameter missing from ``xi`` would only blur the
world model towards average dynamics. Any term switches to one extended layout whose size does not
depend on which terms are on, so checkpoints resume across term sets; terms that are off keep their
nominal value. The extended entries are stored normalized to [-1, 1] over their range.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .mdp import XI_DIM
from .robot import JOINT_NAMES, ROBOT_NAME

TERMS = ("pd", "push", "mass", "armature", "joint_friction", "delay")

if ROBOT_NAME == "tiangong3":
    from .tiangong3 import MASS_SEGMENTS, mass_segment
else:
    MASS_SEGMENTS, mass_segment = [], None

# extended xi layout, after the 33 base entries
NJ = len(JOINT_NAMES)
KP = slice(XI_DIM, XI_DIM + NJ)
KD = slice(KP.stop, KP.stop + NJ)
ARMATURE = slice(KD.stop, KD.stop + NJ)
JOINT_FRICTION = slice(ARMATURE.stop, ARMATURE.stop + NJ)
MASS = slice(JOINT_FRICTION.stop, JOINT_FRICTION.stop + len(MASS_SEGMENTS))
DELAY = MASS.stop
XI_DIM_DR = DELAY + 1


def parse_terms(spec: str | None) -> frozenset[str]:
    """``"pd,push"`` or ``"all"``; ``""``, ``"none"`` and ``"base"`` add nothing to the paper's randomization."""
    if not spec or spec in ("none", "base"):
        return frozenset()
    terms = frozenset(TERMS) if spec == "all" else frozenset(t.strip() for t in spec.split(",") if t.strip())
    if unknown := terms - set(TERMS):
        raise ValueError(f"unknown --dr term(s) {sorted(unknown)}; choose from {', '.join(TERMS)} or all")
    if terms and ROBOT_NAME != "tiangong3":
        raise ValueError("--dr is only available for --robot tiangong3 (G1 keeps the 33-dim xi of its checkpoints)")
    return terms


def describe(terms: frozenset[str]) -> str:
    return ", ".join(t for t in TERMS if t in terms)


@dataclass(frozen=True)
class DRCfg:
    terms: frozenset[str] = frozenset()
    kp_scale: tuple[float, float] = (0.75, 1.25)  # per joint, every episode
    kd_scale: tuple[float, float] = (0.75, 1.25)
    push_interval_s: tuple[float, float] = (1.0, 5.0)  # per environment
    push_lin_vel: tuple[float, float, float] = (0.5, 0.5, 0.2)  # half-widths added to the root velocity (world) [m/s]
    push_ang_vel: tuple[float, float, float] = (0.52, 0.52, 0.78)  # [rad/s]
    mass_scale: tuple[float, float] = (0.85, 1.15)  # per rigid segment with its inertia, at start-up
    armature: tuple[float, float] = (0.002, 0.06)  # [kg m^2], at start-up
    ankle_armature: tuple[float, float] = (0.01, 0.06)
    joint_friction: tuple[float, float] = (0.001, 0.6)  # MuJoCo frictionloss [N m], at start-up; PhysX's is unitless
    max_delay: int = 4  # 5 ms physics substeps that still track the previous target, every episode

    @property
    def xi_dim(self) -> int:
        return XI_DIM_DR if self.terms else XI_DIM


def to_unit(x, lo, hi):
    """Map [lo, hi] to [-1, 1]."""
    return (x - 0.5 * (lo + hi)) / (0.5 * (hi - lo))


def uniform(lo, hi, shape, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, device=device)

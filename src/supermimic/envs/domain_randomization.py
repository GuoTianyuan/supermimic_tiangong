"""Opt-in domain randomization for deployment (``train.py --dr``, Tiangong3 only), with the ranges of
DexEVT (cheerleading_competition, tracking_env_cfg.py EventCfg) and mjlab TK3.

Every randomized parameter is also written into the simulator context ``xi``. The world model and the
critic see ``xi`` while the policy does not, so imagination keeps each environment's own dynamics and
the policy has to work across the whole range; a parameter missing from ``xi`` would only blur the
world model towards average dynamics. Any term switches to one extended layout whose size does not
depend on which terms are on, so checkpoints resume across term sets; terms that are off keep their
nominal value. The extended entries are stored normalized to [-1, 1] over their range.

``contact`` (foot-ground contact stiffness) came later: it appends one more entry, is not part of
``all`` (whose meaning old checkpoints store) and is asked for as ``all,contact``. train.py widens the
first layers of a checkpoint without it when resuming with it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch

from supermimic.tasks.tracking.tracking_env_cfg import RandomizationRangesCfg, randomization_ranges
from supermimic.tasks.tracking.mdp import XI_DIM
from supermimic.robots import FRICTION_RANGE, JOINT_NAMES, ROBOT_NAME

TERMS = ("pd", "push", "mass", "armature", "joint_friction", "delay", "contact")
ALL_TERMS = frozenset(TERMS) - {"contact"}  # what "all" meant when the checkpoints stored it

if ROBOT_NAME == "tiangong3":
    from supermimic.robots.tiangong3 import MASS_SEGMENTS, mass_segment
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
CONTACT = XI_DIM_DR  # only with the contact term


def parse_terms(spec: str | None) -> frozenset[str]:
    """``"pd,push"``, ``"all"`` (every term but contact) or ``"all,contact"``; ``""``, ``"none"`` and ``"base"``
    add nothing to the paper's randomization."""
    if not spec or spec in ("none", "base"):
        return frozenset()
    names = {t.strip() for t in spec.split(",") if t.strip()}
    terms = (names - {"all"}) | (ALL_TERMS if "all" in names else frozenset())
    if unknown := terms - set(TERMS):
        raise ValueError(f"unknown --dr term(s) {sorted(unknown)}; choose from {', '.join(TERMS)}, all or all,contact")
    if terms and ROBOT_NAME != "tiangong3":
        raise ValueError("--dr is only available for --robot tiangong3 (G1 keeps the 33-dim xi of its checkpoints)")
    return terms


def describe(terms: frozenset[str]) -> str:
    return ", ".join(t for t in TERMS if t in terms)


def describe_scale(scale: dict[str, float]) -> str:
    return ", ".join(f"{t} {100 * scale[t]:.0f}%" for t in TERMS if scale.get(t, 0.0) > 0) or "base only"


LATE_TERMS = frozenset({"push", "delay"})


def parse_curriculum(spec: str | None) -> tuple[int, int] | None:
    """``"A,B"``: base randomization only before iteration A, the terms other than push/delay widen from A,
    push/delay from B (see :func:`curriculum_scale`)."""
    if not spec:
        return None
    try:
        a, b = (int(x) for x in spec.split(","))
    except ValueError:
        raise ValueError(f"--dr_curriculum takes two iterations A,B, got {spec!r}") from None
    if not 0 <= a <= b:
        raise ValueError(f"--dr_curriculum needs 0 <= A <= B, got {spec!r}")
    return a, b


def curriculum_scale(terms: frozenset[str], curriculum: tuple[int, int] | None, ramp: int, it: int) -> dict[str, float]:
    """Fraction of each term's range in use at iteration ``it`` (0: nominal, 1: full). Each curriculum stage
    widens linearly over ``ramp`` iterations (0: at once), in 5 % steps since every step rewrites the
    start-up terms of all environments."""
    out = {}
    for t in terms:
        if curriculum is None:
            out[t] = 1.0
            continue
        start = curriculum[1] if t in LATE_TERMS else curriculum[0]
        x = (it - start + 1) / ramp if ramp > 0 else float(it >= start)
        out[t] = math.floor(min(max(x, 0.0), 1.0) * 20) / 20
    return out


@dataclass(frozen=True)
class DRCfg(RandomizationRangesCfg):
    """Runtime context layout plus the task's declared physical ranges.

    Range fields are keyword-only; DRCfg(terms) remains compatible.
    """
    terms: frozenset[str] = frozenset()
    # Direct legacy DRCfg() callers also use the selected robot's base friction.
    friction: tuple[float, float] = field(default=FRICTION_RANGE, kw_only=True)

    @property
    def xi_dim(self) -> int:
        return XI_DIM_DR + ("contact" in self.terms) if self.terms else XI_DIM

    @classmethod
    def for_dynamics(cls, terms: frozenset[str], dynamics: str = "default",
                     ranges: RandomizationRangesCfg | None = None) -> "DRCfg":
        cfg = randomization_ranges(ROBOT_NAME, dynamics) if ranges is None else ranges.validate()
        return cls(terms=terms, **asdict(cfg))


def to_unit(x, lo, hi):
    """Map [lo, hi] to [-1, 1]."""
    return (x - 0.5 * (lo + hi)) / (0.5 * (hi - lo))


def uniform(lo, hi, shape, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, device=device)

"""Scene and Isaac Lab / Newton / MuJoCo-Warp timing configuration."""

from __future__ import annotations
from supermimic.utils.paths import DATA_ROOT
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, replace
from supermimic.envs.domain_randomization import parse_terms
from supermimic.motions.library import CONTROL_DT
from supermimic.robots import G1_CFG


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(usd_path=str(DATA_ROOT / "ground" / "default_ground_plane.usda")))
    robot = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/Robot")

def make_sim(device: str, args) -> SimulationContext:
    # With actuator delay (args.dr, src/supermimic/envs/domain_randomization.py) TrackingSim runs the four 2.5 ms physics substeps
    # itself, so it can switch joint targets between them.
    k = 4 if "delay" in parse_terms(getattr(args, "dr", None)) else 1
    physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=500, nconmax=128, cone="pyramidal", integrator="implicitfast", use_mujoco_contacts=True),
        num_substeps=4 // k,
    )
    return sim_utils.SimulationCfg(dt=CONTROL_DT / k, device=device, physics=physics)

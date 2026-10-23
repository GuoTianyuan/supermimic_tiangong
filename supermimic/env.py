"""Motion-tracking simulator on Isaac Lab 3 (kit-less, Newton physics with the MuJoCo-Warp solver).

The environment is a thin batched wrapper: it applies the joint setpoint, steps MuJoCo-Warp, labels
the transition with the analytic reward/termination of :mod:`supermimic.mdp`, and resets episodes to
reference-motion states drawn by return-based bin sampling (paper Appendix D.1).
"""

from __future__ import annotations

import math

import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from newton import ModelFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace

from .dr import ARMATURE, DELAY, JOINT_FRICTION, KD, KP, MASS, MASS_SEGMENTS, DRCfg, mass_segment, parse_terms, to_unit, uniform
from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import G1_CFG, G1_JOINT_NAMES, TORSO_BODY, action_scale
from .rotations import mat_to_quat, quat_to_mat, so3_exp


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/Robot")


def make_sim(device: str, args) -> SimulationContext:
    # With actuator delay (args.dr, supermimic/dr.py) TrackingSim runs the four 5 ms physics substeps
    # itself, so it can switch joint targets between them.
    k = 4 if "delay" in parse_terms(getattr(args, "dr", None)) else 1
    physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=500, nconmax=128, cone="pyramidal", integrator="implicitfast", use_mujoco_contacts=True),
        num_substeps=4 // k,
    )
    return sim_utils.SimulationCfg(dt=CONTROL_DT / k, device=device, physics=physics)


class TrackingSim:
    """Batched G1 motion-tracking simulator.

    Must be constructed inside ``launch_simulation(sim_cfg, ...)`` with ``sim_cfg`` from :func:`make_sim`.
    ``randomize`` (default: not ``eval_mode``) draws the paper's context randomization and the ``dr``
    terms; otherwise every parameter stays nominal, with friction 1.
    """

    def __init__(self, sim_cfg, num_envs: int, motion: MotionLib, mdp: MDP, gamma: float = 0.99, eval_mode: bool = False,
                 max_episode_steps: int = 500, reset_noise: bool = True, dr: DRCfg | None = None, randomize: bool | None = None):
        self.device = sim_cfg.device
        self.sim = SimulationContext(sim_cfg)
        self.scene = instantiate(TrackingSceneCfg(num_envs=num_envs, env_spacing=0.0, replicate_physics=True))
        self.sim.reset()
        self.robot = self.scene["robot"]
        assert list(self.robot.joint_names) == G1_JOINT_NAMES, self.robot.joint_names
        self.N = num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.randomize = not eval_mode if randomize is None else randomize
        self.dr = dr or DRCfg()
        self.dr_on = self.dr.terms if self.randomize else frozenset()
        self.substeps = round(CONTROL_DT / sim_cfg.dt)
        if "delay" in self.dr.terms and self.substeps < self.dr.max_delay:
            raise ValueError("actuator delay needs the 5 ms physics steps of make_sim(device, args) with args.dr")
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        dev = self.device
        self.default_q = self.robot.data.default_joint_pos.torch[0].clone()
        self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=dev)
        self.torso_id = self.robot.find_bodies(TORSO_BODY)[0][0]

        # ---- sim context xi = [friction, joint offsets(29), torso com offset(3)] (+ supermimic/dr.py layout) ----
        self.xi = torch.zeros(self.N, self.dr.xi_dim, device=dev)
        self._setup_friction()
        if not self.randomize:
            self.xi[:, 0] = 1.0
        else:
            self.xi[:, 0] = torch.empty(self.N, device=dev).uniform_(0.3, 1.2)
            com = torch.stack(
                (
                    torch.empty(self.N, device=dev).uniform_(-0.025, 0.025),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                ),
                -1,
            )
            self.xi[:, 30:33] = com
            default_com = self.robot.data.body_com_pose_b.torch[:, self.torso_id, :3].clone()
            self.robot.set_coms_index(coms=(default_com + com).unsqueeze(1), body_ids=[self.torso_id])
        self._apply_friction()
        if self.dr.terms:
            self._setup_dr()

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, 29, device=dev)
        self.target_prev = torch.zeros(self.N, 29, device=dev)
        self.delay = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.push_timer = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    # ------------------------------------------------------------------ friction
    def _setup_friction(self):
        mgr = self.sim.backend_manager
        model = mgr.get_model()
        self._mu_view = wp.to_torch(self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
        # ground shapes (body == -1): lowest friction of the range, since MuJoCo takes the max of a pair
        shape_body = wp.to_torch(model.shape_body)
        mu_all = wp.to_torch(model.shape_material_mu)
        mu_all[shape_body < 0] = 0.3 if self.randomize else 1.0

    def _apply_friction(self):
        self._mu_view[:] = self.xi[:, :1]
        self.sim.backend_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    # ------------------------------------------------------------------ supermimic/dr.py terms
    def _setup_dr(self):
        """Start-up terms; every extended xi entry starts at its nominal value."""
        c, r, d, dev, N = self.dr, self.robot, self.robot.data, self.device, self.N
        self.kp0, self.kd0 = d.joint_stiffness.torch.clone(), d.joint_damping.torch.clone()
        arm_lo, arm_hi = (torch.tensor([(c.ankle_armature if "ankle" in n else c.armature)[i] for n in G1_JOINT_NAMES], device=dev)
                          for i in (0, 1))  # fmt: skip
        if "armature" in self.dr_on:
            r.write_joint_armature_to_sim_index(armature=uniform(arm_lo, arm_hi, (N, 29), dev))
        if "joint_friction" in self.dr_on:
            r.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=uniform(*c.joint_friction, (N, 29), dev))
        mass = torch.ones(N, len(MASS_SEGMENTS), device=dev)
        if "mass" in self.dr_on:
            mass = uniform(*c.mass_scale, mass.shape, dev)
            s = mass[:, [mass_segment(b) for b in r.body_names]]
            r.set_masses_index(masses=d.body_mass.torch * s)
            r.set_inertias_index(inertias=d.body_inertia.torch * s.unsqueeze(-1))
        self.xi[:, KP] = to_unit(1.0, *c.kp_scale)
        self.xi[:, KD] = to_unit(1.0, *c.kd_scale)
        self.xi[:, ARMATURE] = to_unit(d.joint_armature.torch, arm_lo, arm_hi)
        self.xi[:, JOINT_FRICTION] = to_unit(d.joint_friction_coeff.torch, *c.joint_friction)
        self.xi[:, MASS] = to_unit(mass, *c.mass_scale)
        self.xi[:, DELAY] = to_unit(0.0, 0, c.max_delay)

    def _reset_dr(self, env_ids: torch.Tensor):
        """Per-episode terms."""
        c, n, dev = self.dr, len(env_ids), self.device
        if "pd" in self.dr_on:
            kp, kd = uniform(*c.kp_scale, (n, 29), dev), uniform(*c.kd_scale, (n, 29), dev)
            mgr = self.sim.backend_manager
            pending = ModelFlags.JOINT_DOF_PROPERTIES in mgr._model_changes
            self.robot.write_joint_stiffness_to_sim_index(stiffness=self.kp0[env_ids] * kp, env_ids=env_ids)
            self.robot.write_joint_damping_to_sim_index(damping=self.kd0[env_ids] * kd, env_ids=env_ids)
            if not pending:
                # The joint writers request JOINT_DOF_PROPERTIES, which makes MuJoCo-Warp recompute all model
                # constants at the next step (~10 ms at 1k envs); gains only need the actuator parameters.
                mgr._model_changes.discard(ModelFlags.JOINT_DOF_PROPERTIES)
                mgr.add_model_change(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
            self.xi[env_ids, KP] = to_unit(kp, *c.kp_scale)
            self.xi[env_ids, KD] = to_unit(kd, *c.kd_scale)
        if "delay" in self.dr_on:
            self.delay[env_ids] = torch.randint(0, c.max_delay + 1, (n,), device=dev)
            self.xi[env_ids, DELAY] = to_unit(self.delay[env_ids].float(), 0, c.max_delay)
        if "push" in self.dr_on:
            self.push_timer[env_ids] = self._push_interval(n)

    def _push_interval(self, n: int) -> torch.Tensor:
        return (uniform(*self.dr.push_interval_s, (n,), self.device) / CONTROL_DT).round().long()

    def _push(self, S1) -> torch.Tensor:
        """Add a random root velocity every 1-5 s per environment (DexEVT's push_by_setting_velocity)."""
        self.push_timer -= 1
        push = self.push_timer <= 0
        ids = push.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            half = torch.tensor(self.dr.push_lin_vel + self.dr.push_ang_vel, device=self.device)
            vel = torch.cat((S1[2][ids], S1[3][ids]), -1) + (2 * torch.rand(len(ids), 6, device=self.device) - 1) * half
            self.robot.write_root_link_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
            self.push_timer[ids] = self._push_interval(len(ids))
        return push

    # ------------------------------------------------------------------ state
    def state(self):
        d = self.robot.data
        pose = d.root_link_pose_w.torch.clone()
        vel = d.root_link_vel_w.torch.clone()
        p = pose[:, :3] - self.scene.env_origins
        quat = pose[:, 3:7]  # Isaac Lab 3 uses xyzw
        R = quat_to_mat(torch.cat((quat[:, 3:4], quat[:, :3]), -1))
        return (p, R, vel[:, :3], vel[:, 3:6], d.joint_pos.torch.clone(), d.joint_vel.torch.clone())

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m = self.motion
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            p = p + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                 torch.empty(n, device=dev).uniform_(-0.01, 0.01)), -1)  # fmt: skip
            rv = torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            R = so3_exp(rv) @ R
            v = v + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.5, 0.5), torch.empty(n, device=dev).uniform_(-0.5, 0.5),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            w = w + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.52, 0.52), torch.empty(n, device=dev).uniform_(-0.52, 0.52),
                 torch.empty(n, device=dev).uniform_(-0.78, 0.78)), -1)  # fmt: skip
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        quat = mat_to_quat(R)
        pose = torch.cat((p + self.scene.env_origins[env_ids], quat[:, 1:4], quat[:, 0:1]), -1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)
        self.robot.set_joint_position_target_index(target=q, env_ids=env_ids)
        self.target_prev[env_ids] = q  # a delayed first step holds the reset pose
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        if self.randomize:
            self.xi[env_ids, 1:30] = torch.empty(n, 29, device=dev).uniform_(-0.01, 0.01)
        if self.dr.terms:
            self._reset_dr(env_ids)
        self.scene.reset(env_ids)

    def refresh(self):
        """Propagate written states so that ``state()`` is consistent after a reset."""
        self.sim.forward() if hasattr(self.sim, "forward") else None
        self.scene.update(0.0)

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1:30] + self.alpha * a
        for k in range(self.substeps):
            # under actuator delay the first `delay` substeps still track the previous target
            tgt = target if self.substeps == 1 else torch.where((k < self.delay).unsqueeze(-1), self.target_prev, target)
            self.robot.set_joint_position_target_index(target=tgt)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
        self.target_prev = target
        self.scene.update(CONTROL_DT)
        S1 = self.state()
        f1 = f + 1
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        cut = self._push(S1) if "push" in self.dr_on else torch.zeros_like(done)
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "cut": cut, "ep_step": ep_step, "xi": xi, "terms": terms,
        }  # fmt: skip
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out

"""Motion-tracking simulator on Isaac Lab 3 (kit-less, Newton physics with the MuJoCo-Warp solver).

The environment is a thin batched wrapper: it applies the joint setpoint, steps MuJoCo-Warp, labels
the transition with the analytic reward/termination of :mod:`supermimic.tasks.tracking.mdp`, and resets episodes to
reference-motion states drawn by return-based bin sampling (paper Appendix D.1).
"""

from __future__ import annotations


import re

import torch
import warp as wp
from newton import ModelFlags

from isaaclab.sim import SimulationContext
from isaaclab.utils import instantiate, replace

from supermimic.envs.domain_randomization import ARMATURE, CONTACT, DELAY, JOINT_FRICTION, KD, KP, MASS, MASS_SEGMENTS, DRCfg, mass_segment, to_unit, uniform
from supermimic.tasks.tracking.mdp import ACTION_CLIP, MAX_STEP_REWARD, MDP, flatten_state
from supermimic.motions.library import BIN_FRAMES, CONTROL_DT, MotionLib
from supermimic.robots import FRICTION_RANGE, G1_CFG, G1_JOINT_NAMES, TORSO_BODY, action_scale
from supermimic.utils.rotations import mat_to_quat, quat_to_mat, so3_exp
from supermimic.motions.sampling import update_bin_returns_

from .sim_context import TrackingSceneCfg

SIM_STATE_BOUND = 1e3  # |entries| of a sane state (positions in m, velocities in m/s and rad/s)


class TrackingSim:
    """Batched G1 motion-tracking simulator.

    Must be constructed inside ``launch_simulation(sim_cfg, ...)`` with ``sim_cfg`` from :func:`make_sim`.
    ``randomize`` (default: not ``eval_mode``) draws the paper's context randomization and the ``dr``
    terms; otherwise every parameter stays nominal, with friction 1. ``dr_scale`` maps each ``dr`` term
    to the fraction of its range in use (0: nominal, 1: full; default all 1); :meth:`set_dr_scale` changes
    it later. ``reset_a_prev``: ``"pose"`` starts each episode with the previous action that holds the
    reset pose, ``"zero"`` with 0 (checkpoints trained before the pose option existed). ``reset_temp`` is
    the inverse temperature of the return-based start sampling.
    """

    def __init__(self, sim_cfg, num_envs: int, motion: MotionLib, mdp: MDP, gamma: float = 0.99, eval_mode: bool = False,
                 max_episode_steps: int = 500, reset_noise: bool = True, dr: DRCfg | None = None, randomize: bool | None = None,
                 dr_scale: dict[str, float] | None = None, reset_a_prev: str = "pose", reset_temp: float = 25.0):
        self.device = sim_cfg.device
        self.sim = SimulationContext(sim_cfg)
        scene_cfg = TrackingSceneCfg(num_envs=num_envs, env_spacing=0.0, replicate_physics=True)
        scene_cfg.robot = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/Robot")  # the actuators of set_dynamics()
        self.scene = instantiate(scene_cfg)
        self.sim.reset()
        self.robot = self.scene["robot"]
        assert list(self.robot.joint_names) == G1_JOINT_NAMES, self.robot.joint_names
        self.N = num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.randomize = not eval_mode if randomize is None else randomize
        self.dr = dr or DRCfg()
        dr_scale = {t: 1.0 for t in self.dr.terms} if dr_scale is None else dict(dr_scale)
        if not set(dr_scale) <= self.dr.terms or not all(0.0 <= s <= 1.0 for s in dr_scale.values()):
            raise ValueError(f"dr_scale {dr_scale} needs dr terms {sorted(self.dr.terms)} and scales in [0, 1]")
        self.dr_scale = {t: float(dr_scale.get(t, 0.0)) if self.randomize else 0.0 for t in self.dr.terms}
        self.dr_on = frozenset(t for t, s in self.dr_scale.items() if s > 0)
        if reset_a_prev not in ("pose", "zero"):
            raise ValueError(f"reset_a_prev must be 'pose' or 'zero', got {reset_a_prev!r}")
        self.reset_a_prev = reset_a_prev
        self.reset_temp = reset_temp
        self.substeps = round(CONTROL_DT / sim_cfg.dt)
        if "delay" in self.dr.terms and self.substeps < self.dr.max_delay:
            raise ValueError("actuator delay needs the 2.5 ms physics steps of make_sim(device, args) with args.dr")
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        dev = self.device
        self.default_q = self.robot.data.default_joint_pos.torch[0].clone()
        self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=dev)
        self.torso_id = self.robot.find_bodies(TORSO_BODY)[0][0]

        # ---- sim context xi = [friction, joint offsets(29), torso com offset(3)] (+ src/supermimic/envs/domain_randomization.py layout) ----
        self.xi = torch.zeros(self.N, self.dr.xi_dim, device=dev)
        self._setup_friction()
        if not self.randomize:
            self.xi[:, 0] = 1.0
        else:
            self.xi[:, 0] = torch.empty(self.N, device=dev).uniform_(*FRICTION_RANGE)
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
        if hasattr(motion, "bin_returns"):
            # Restore before the constructor's first reset, so it uses the saved sampler too.
            self.bin_G = motion.bin_returns[motion.bin_global.cpu()].to(dev)
            self.bin_w_avg = motion.bin_weights[motion.bin_global.cpu()].to(dev)
        else:
            self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
            self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.n_nonfinite = 0  # diverged transitions so far
        self.reset(torch.arange(self.N, device=dev))

    # ------------------------------------------------------------------ friction
    def _setup_friction(self):
        mgr = self.sim.backend_manager
        model = mgr.get_model()
        self._mu_view = wp.to_torch(self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
        # ground shapes (body == -1): lowest friction of the range, since MuJoCo takes the max of a pair
        shape_body = wp.to_torch(model.shape_body)
        mu_all = wp.to_torch(model.shape_material_mu)
        mu_all[shape_body < 0] = FRICTION_RANGE[0] if self.randomize else 1.0

    def _apply_friction(self):
        self._mu_view[:] = self.xi[:, :1]
        self.sim.backend_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    # ------------------------------------------------------------------ src/supermimic/envs/domain_randomization.py terms
    def _setup_dr(self):
        """Every extended xi entry starts at its nominal value; then the start-up terms in use are drawn."""
        c, d, dev, N = self.dr, self.robot.data, self.device, self.N
        self.kp0, self.kd0 = d.joint_stiffness.torch.clone(), d.joint_damping.torch.clone()
        self.mass0, self.inertia0 = d.body_mass.torch.clone(), d.body_inertia.torch.clone()
        self.arm0, self.fric0 = d.joint_armature.torch.clone(), d.joint_friction_coeff.torch.clone()
        self.arm_lo, self.arm_hi = (
            torch.tensor([(c.ankle_armature if "ankle" in n else c.armature)[i] for n in G1_JOINT_NAMES], device=dev)
            for i in (0, 1))  # fmt: skip
        self.xi[:, KP] = to_unit(1.0, *c.kp_scale)
        self.xi[:, KD] = to_unit(1.0, *c.kd_scale)
        self.xi[:, ARMATURE] = to_unit(self.arm0, self.arm_lo, self.arm_hi)
        self.xi[:, JOINT_FRICTION] = to_unit(self.fric0, *c.joint_friction)
        self.xi[:, MASS] = to_unit(1.0, *c.mass_scale)
        self.xi[:, DELAY] = to_unit(0.0, 0, c.max_delay)
        # One full-range draw per environment; a scale s uses nominal + s * (draw - nominal).
        self.u_arm = uniform(self.arm_lo, self.arm_hi, (N, 29), dev)
        self.u_fric = uniform(*c.joint_friction, (N, 29), dev)
        self.u_mass = uniform(*c.mass_scale, (N, len(MASS_SEGMENTS)), dev)
        if "contact" in c.terms:
            self._setup_contact()
            self.xi[:, CONTACT] = -1.0
            self.u_contact = torch.rand(N, device=dev)  # 0: contact_soft, 1: contact_hard
        self._apply_startup_dr(self.dr_on)

    def _setup_contact(self):
        """Per-environment views of the MuJoCo contact parameters of the foot collision shapes. The feet get a
        large solmix, so that their solref / solimp, not the shared ground's, apply to foot-ground contacts."""
        model = self.sim.backend_manager.get_model()
        mj = model.mujoco
        ids = [[] for _ in range(self.N)]
        for s, label in enumerate(model.shape_label):
            parts = label.split("/")
            env = re.search(r"/env_(\d+)/", label)
            if env and len(parts) > 1 and re.fullmatch(r".*ankle_roll.*link", parts[-2]):
                ids[int(env.group(1))].append(s)
        if not ids[0] or any(len(i) != len(ids[0]) for i in ids):
            raise RuntimeError(f"found {sorted({len(i) for i in ids})} foot collision shapes per environment")
        self.foot_shapes = torch.tensor(ids, device=self.device)  # (N, shapes per env)
        if bool((wp.to_torch(mj.solref_mode)[self.foot_shapes] != 1).any()):  # SOLREF_MODE_RAW
            raise RuntimeError("the foot shapes need raw MuJoCo solref values (mjc:solref in the robot USD)")
        self._solref, self._solimp = wp.to_torch(mj.solref), wp.to_torch(mj.geom_solimp)
        wp.to_torch(mj.geom_solmix)[self.foot_shapes] = 1e3
        phys_dt = CONTROL_DT / 4
        if self.dr.contact_hard[0] < 2 * phys_dt - 1e-9:
            raise ValueError(f"contact_hard time constant {self.dr.contact_hard[0]} s is below 2 physics steps ({phys_dt} s)")

    def _apply_startup_dr(self, terms: frozenset[str]):
        """Start-up terms, fixed per environment until their scale changes."""
        c, r, d, s = self.dr, self.robot, self.robot.data, self.dr_scale
        if "armature" in terms:
            r.write_joint_armature_to_sim_index(armature=self.arm0 + s["armature"] * (self.u_arm - self.arm0))
            self.xi[:, ARMATURE] = to_unit(d.joint_armature.torch, self.arm_lo, self.arm_hi)
        if "joint_friction" in terms:
            fric = self.fric0 + s["joint_friction"] * (self.u_fric - self.fric0)
            r.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=fric, joint_viscous_friction_coeff=fric if c.viscous_friction else None)
            self.xi[:, JOINT_FRICTION] = to_unit(d.joint_friction_coeff.torch, *c.joint_friction)
        if "mass" in terms:
            mass = 1.0 + s["mass"] * (self.u_mass - 1.0)
            m = mass[:, [mass_segment(b) for b in r.body_names]]
            r.set_masses_index(masses=self.mass0 * m)
            r.set_inertias_index(inertias=self.inertia0 * m.unsqueeze(-1))
            self.xi[:, MASS] = to_unit(mass, *c.mass_scale)
        if "contact" in terms:
            h = (s["contact"] * self.u_contact).unsqueeze(-1)
            lo, hi = (torch.tensor(x, device=self.device) for x in (c.contact_soft, c.contact_hard))
            tc = lo[0] * (hi[0] / lo[0]) ** h  # log-uniform in the time constant
            imp = lo[1:] + h * (hi[1:] - lo[1:])
            k = self.foot_shapes.shape[1]
            self._solref[self.foot_shapes, 0] = tc.expand(-1, k)
            self._solimp[self.foot_shapes, 0] = imp[:, :1].expand(-1, k)
            self._solimp[self.foot_shapes, 1] = imp[:, 1:].expand(-1, k)
            self.sim.backend_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
            self.xi[:, CONTACT] = 2 * h.squeeze(-1) - 1

    def set_dr_scale(self, scale: dict[str, float]):
        """Change the range scale of ``dr`` terms mid-training (missing terms: 0): start-up terms and push
        timers now, pd/delay at each environment's next reset."""
        if not self.randomize:
            return
        new = {t: float(scale.get(t, 0.0)) for t in self.dr.terms}
        changed = frozenset(t for t in new if new[t] != self.dr_scale[t])
        if not changed:
            return
        was_on = self.dr_on
        self.dr_scale = new
        self.dr_on = frozenset(t for t, s in new.items() if s > 0)
        self._apply_startup_dr(changed)
        if "push" in self.dr_on - was_on:
            self.push_timer[:] = self._push_interval(self.N)

    def _reset_dr(self, env_ids: torch.Tensor):
        """Per-episode terms."""
        c, n, dev, s = self.dr, len(env_ids), self.device, self.dr_scale
        if "pd" in self.dr_on:
            kp = 1.0 + s["pd"] * (uniform(*c.kp_scale, (n, 29), dev) - 1.0)
            kd = 1.0 + s["pd"] * (uniform(*c.kd_scale, (n, 29), dev) - 1.0)
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
            self.delay[env_ids] = torch.randint(0, round(s["delay"] * c.max_delay) + 1, (n,), device=dev)
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
            half = self.dr_scale["push"] * torch.tensor(self.dr.push_lin_vel + self.dr.push_ang_vel, device=self.device)
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
        eligible = (m.active_bins if hasattr(m, "active_bins") else
                    torch.arange(m.num_bins, device=self.device))
        # A final partial bin can contain only the clip's last frame. It has no
        # successor transition, so keep its stable ID but exclude it from resets.
        eligible = eligible[m.bin_start[eligible] < m.clip_end[m.bin_clip[eligible]]]
        if eligible.numel() == 0:
            raise RuntimeError("No active motion bin has a valid successor frame for reset")
        soft = torch.softmax(-self.reset_temp * self.bin_G[eligible] / self.G_max, 0)
        avg = self.bin_w_avg[eligible].mul(0.999).add(0.001 * soft)
        self.bin_w_avg[eligible] = avg
        probs = 0.5 / len(eligible) + 0.4 * soft + 0.1 * avg / avg.sum()
        bins = eligible[torch.multinomial(probs, n, replacement=True)]
        if hasattr(m, "active_bins"):
            m.record_starts(bins)
        start = m.bin_start[bins]
        end = torch.minimum(start + BIN_FRAMES, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
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
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        if self.randomize:
            self.xi[env_ids, 1:30] = torch.empty(n, 29, device=dev).uniform_(-0.01, 0.01)
        if self.reset_a_prev == "pose":
            # inverse of the joint target in step()
            a0 = (q - self.default_q - self.xi[env_ids, 1:30]) / self.alpha
            self.a_prev[env_ids] = a0.clamp(-ACTION_CLIP, ACTION_CLIP)
        else:
            self.a_prev[env_ids] = 0.0
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
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True, qd_prev=S[5])
        term = self.mdp.termination(S1, f1)
        s, s1 = flatten_state(S), flatten_state(S1)
        # A diverged physics state (rare) would poison the reset sampler and, through the normalizers merged
        # across ranks, every GPU: it ends the episode with reward 0 and is stored as a standing still step.
        bad = ~((torch.isfinite(s1) & (s1.abs() < SIM_STATE_BOUND)).all(-1) & torch.isfinite(r))
        if bool(bad.any()):
            s = torch.where(bad.unsqueeze(-1), torch.nan_to_num(s).clamp(-SIM_STATE_BOUND, SIM_STATE_BOUND), s)
            s1 = torch.where(bad.unsqueeze(-1), s, s1)
            S1 = tuple(torch.where(bad.view(-1, *[1] * (x.dim() - 1)), x0, x) for x0, x in zip(S, S1))
            r = torch.where(bad, 0.0, r)
            terms = {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in terms.items()}
            term = term | bad
            self.n_nonfinite += int(bad.sum())
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        cut = self._push(S1) if "push" in self.dr_on else torch.zeros_like(done)
        out = {
            "s": s, "a": a.clone(), "a_prev": a_prev, "f": f, "s1": s1, "r": r,
            "term": term, "done": done, "cut": cut | bad, "ep_step": ep_step, "xi": xi, "terms": terms,
            "end_clip": (f1 >= clip_end) & ~term,
            "end_time": (ep_step + 1 >= self.max_episode_steps) & (f1 < clip_end) & ~term,
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
            update_bin_returns_(self.bin_G, b, G)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out

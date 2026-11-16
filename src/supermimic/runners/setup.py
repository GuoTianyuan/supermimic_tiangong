"""Construct and restore training dependencies in the original RNG/collective order.

Called only after the CLI has selected SUPERMIMIC_ROBOT. Numerical algorithms
live in learners/, while checkpoint persistence lives in checkpoints.py.
"""
from __future__ import annotations
import os
import json
import time
import datetime
from dataclasses import dataclass
from argparse import Namespace
import copy

import torch
import torch.distributed as dist

from supermimic.configs.experiment import resolve_reward_config, config_snapshot
from supermimic.motions.pool import MotionPool
from supermimic.motions.timing import configure_reference_timing
from supermimic.envs.domain_randomization import DRCfg, parse_curriculum, parse_terms
from supermimic.robots.kinematics import G1Kinematics
from supermimic.tasks.tracking.mdp import CRITIC_EXTRA, INV_DIM, MDP, OBS_A_PREV, OBS_DIM, XI_DIM
from supermimic.motions.library import CONTROL_DT, MotionLib, goal_frame_dim, parse_goal_drop
import supermimic.models.common as networks
from supermimic.models import Critic, Policy, WorldModel
from supermimic.robots import DEFAULT_MOTION, G1_CFG, G1_JOINT_NAMES, action_scale, default_joint_pos_list, set_dynamics


@dataclass
class TrainingComponents:
    """Explicit dependencies shared by the runner, collector and checkpoint writer."""
    args: Namespace
    dev: str
    dr: DRCfg
    dr_curriculum: tuple | None
    motion: MotionLib
    mdp: MDP
    wm: WorldModel
    pi: Policy
    critic: Critic
    critic_targ: Critic
    opt_wm: torch.optim.Optimizer
    opt_pi: torch.optim.Optimizer
    opt_v: torch.optim.Optimizer
    start_it: int
    wall_offset: float
    sim_steps: int
    num_clips: int
    num_frames: int
    goal_drop: tuple
    max_return: float
    world: int
    rank: int
    is_main: bool


def build_training(args, rank=0, world=1):
    RANK, WORLD, MAIN = rank, world, rank == 0
    args.motion_dir = args.motion_dir or DEFAULT_MOTION
    dr_curriculum = parse_curriculum(args.dr_curriculum)
    if dr_curriculum and not parse_terms(args.dr):
        raise ValueError("--dr_curriculum needs --dr terms")
    if args.dr_ramp < 0:
        raise ValueError("--dr_ramp must be >= 0")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    dev = "cuda:0"
    if WORLD > 1:
        # Ranks load their motion shards at different speeds before the first collective.
        dist.init_process_group("nccl", device_id=torch.device(dev), timeout=datetime.timedelta(hours=2))
    args.world_size = WORLD
    os.makedirs(args.log_dir, exist_ok=True)
    ck = torch.load(args.resume, map_location=dev) if args.resume else None
    configure_reference_timing(args, ck)
    # checkpoints from before these options existed were trained with the second value
    for k, new, old in (("critic_layer_norm", True, False), ("act_soft_clip", True, False), ("act_norm", "ref", "running"),
                        ("reset_a_prev", "pose", "zero"), ("dynamics", "default", "default"), ("goal_drop", "", ""),
                        ("wm_prediction", "state_delta", "state_delta")):  # fmt: skip
        if getattr(args, k) is None:
            setattr(args, k, ck["args"].get(k, old) if ck else new)
    if ck is not None and ck["args"].get("wm_prediction", "state_delta") != args.wm_prediction:
        raise ValueError("Cannot resume a checkpoint with a different --wm_prediction; start a separate ablation")
    GOAL_DROP = parse_goal_drop(args.goal_drop)
    args.goal_drop = ",".join(GOAL_DROP)
    if ck is not None and parse_goal_drop(ck["args"].get("goal_drop", "")) != GOAL_DROP:
        raise ValueError(f"Checkpoint was trained with --goal_drop '{ck['args'].get('goal_drop', '')}'; the policy's goal "
                         "input size differs, so resume it with the same --goal_drop")  # fmt: skip
    if ck is not None and ck["args"].get("dynamics", "default") != args.dynamics:
        print(f"[rank {RANK}] Checkpoint was trained with --dynamics {ck['args'].get('dynamics', 'default')}; "
              f"continuing with --dynamics {args.dynamics} (the world model has to re-adapt).", flush=True)
    set_dynamics(args.dynamics)
    dr = DRCfg.for_dynamics(parse_terms(args.dr), args.dynamics)
    task_reward_cfg = resolve_reward_config(args, ck)
    args.max_step_reward = task_reward_cfg.max_step_reward(CONTROL_DT)
    G_MAX = args.max_step_reward / (1 - args.gamma)
    args.simulation_timing = dict(control_dt=CONTROL_DT, physics_dt=CONTROL_DT/4, physics_substeps=4)
    if ck is not None and args.wm_prediction == "velocity_integrated":
        if ck["args"].get("simulation_timing") != args.simulation_timing:
            raise ValueError("Integrated WM resume requires the checkpoint's control/physics timing")
    args.self_collisions = bool(G1_CFG.spawn.articulation_props.enabled_self_collisions)
    if ck is not None and ck["args"].get("self_collisions", False) != args.self_collisions:
        print(f"[rank {RANK}] Self-collision setting changed from {ck['args'].get('self_collisions', False)} "
              f"to {args.self_collisions}; replay is rebuilt from the new simulator and the world model has to re-adapt.",
              flush=True)
    args.training_config = config_snapshot(args)
    if MAIN:
        json.dump(args.training_config, open(os.path.join(args.log_dir, "config.json"), "w"), indent=2)
        json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

    kin = G1Kinematics(dev)
    t_load = time.time()
    pool_states = ck.get("motion_pool") if ck else None
    if not 0 < args.motion_active_ratio <= 1 or args.motion_refresh_iters < 0 or not 0 <= args.motion_replace_ratio <= 1:
        raise ValueError("Invalid active ratio / motion refresh interval / replacement ratio")
    if pool_states and args.motion_active_ratio == 1:
        raise ValueError("Resume a pooled run with its original motion_active_ratio")
    if pool_states and (len(pool_states) != WORLD or
                       ck["args"].get("motion_replace_ratio") != args.motion_replace_ratio or
                       ck["args"].get("motion_refresh_iters") != args.motion_refresh_iters):
        raise ValueError("Resume pooled runs with their original world size and pool schedule")
    if args.motion_active_ratio < 1:
        motion = MotionPool.from_dir(args.motion_dir, kin, dev, RANK, WORLD, exclude=args.motion_exclude,
            active_ratio=args.motion_active_ratio, replace_ratio=args.motion_replace_ratio, seed=args.seed+RANK,
            stats_cache=args.motion_stats_cache, state=pool_states[RANK] if pool_states else None)
        if int(motion.clip_len.max()) <= args.horizon:
            raise ValueError("The active subset has no clip long enough for imagination; increase active ratio")
    else:
        motion = MotionLib.from_dir(args.motion_dir, kin, dev, RANK, WORLD, exclude=args.motion_exclude)
    print(f"[rank {RANK}] motion shard: {len(motion.names)} clips, {motion.num_frames} frames, "
          f"loaded in {time.time() - t_load:.0f} s", flush=True)
    motion.goal_stride = args.goal_stride
    motion.goal_start = args.goal_start
    print(f"[rank {RANK}] future reference offsets: {args.goal_offsets} control frames "
          f"({args.goal_offsets[0] * CONTROL_DT:g}..{args.goal_offsets[-1] * CONTROL_DT:g} s)", flush=True)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), task_reward_cfg)
    K, Hh, HZ = args.goal_frames, args.hist, args.horizon
    GOAL_DIM = goal_frame_dim(GOAL_DROP) * K
    CRITIC_IN = INV_DIM + CRITIC_EXTRA + dr.xi_dim + GOAL_DIM

    wm = WorldModel(xi_dim=dr.xi_dim, a_prev=bool(dr.terms), prediction=args.wm_prediction,
                    control_dt=CONTROL_DT, physics_substeps=args.simulation_timing["physics_substeps"]).to(dev)
    pi = Policy(OBS_DIM, GOAL_DIM, hist=Hh, soft_clip_actions=args.act_soft_clip).to(dev)
    # RunningNorm freezes after 2e8 samples, sized for one GPU with 4096 environments and pi_batch 4096;
    # scaled so that the normalizers still freeze at the same iteration
    for norm in (pi.obs_norm, pi.goal_norm, wm.in_norm, *wm.output_normalizers()):
        norm.max_count *= WORLD * args.num_envs / 4096
    # The extended xi entries (src/supermimic/envs/domain_randomization.py) lie in [-1, 1] and stay constant until a --dr_curriculum stage
    # reaches them; running statistics would be frozen around that constant by then.
    N_XI_EXT = dr.xi_dim - XI_DIM
    XI_EXT_STATS = (torch.zeros(N_XI_EXT, device=dev), torch.full((N_XI_EXT,), 3**-0.5, device=dev))
    if N_XI_EXT:
        wm.in_norm.fix(slice(wm.in_dim - N_XI_EXT, wm.in_dim), *XI_EXT_STATS)


    def make_critic() -> Critic:
        c = Critic(CRITIC_IN, layer_norm=args.critic_layer_norm).to(dev)
        c.norm.max_count *= WORLD * args.pi_batch / 4096
        if N_XI_EXT:
            i0 = INV_DIM + CRITIC_EXTRA + XI_DIM
            c.norm.fix(slice(i0, i0 + N_XI_EXT), *XI_EXT_STATS)
        return c


    critic = make_critic()


    def widen_xi(ck: dict, old_xi: int, n: int):
        """Insert ``n`` xi inputs after the checkpoint's ``old_xi`` into its world model and critics, with zero
        first-layer weights and Adam moments, so that they start out unused, and the fixed statistics of the
        extended xi entries."""

        def widen(sd: dict, at: int, norm: str, opt: dict | None):
            cat = lambda x, fill: torch.cat((x[..., :at], x.new_full((*x.shape[:-1], n), fill), x[..., at:]), -1)  # noqa: E731
            sd["net.0.weight"] = cat(sd["net.0.weight"], 0.0)
            sd[f"{norm}.mean"] = cat(sd[f"{norm}.mean"], float(XI_EXT_STATS[0][0]))
            sd[f"{norm}.var"] = cat(sd[f"{norm}.var"], float(XI_EXT_STATS[1][0]) ** 2)
            if opt is not None and 0 in opt["state"]:  # net.0.weight is the first parameter
                for k in ("exp_avg", "exp_avg_sq"):
                    opt["state"][0][k] = cat(opt["state"][0][k], 0.0)

        widen(ck["wm"], wm.in_dim - n, "in_norm", ck.get("opt_wm"))
        at = INV_DIM + CRITIC_EXTRA + old_xi
        for key, opt in (("critic", ck.get("opt_v")), ("critic_targ", None)):
            if key in ck:
                widen(ck[key], at, "norm", opt)


    def reference_action_stats() -> tuple[torch.Tensor, torch.Tensor]:
        """Per-joint mean / std of the actions that hold the reference joint positions, over every rank's motions."""
        if isinstance(motion, MotionPool):
            var, mean, n = motion.reference_joint_stats()
        else:
            var, mean = (x.double() for x in torch.var_mean(motion.q, dim=0, correction=0))
            n = float(motion.num_frames)
        if WORLD > 1:
            s = torch.cat((mean.new_tensor([n]), mean * n, (var + mean**2) * n))
            dist.all_reduce(s)
            mean = s[1:30] / s[0]
            var = (s[30:] / s[0] - mean**2).clamp(min=0)
        alpha = torch.tensor([action_scale(j) for j in G1_JOINT_NAMES], device=dev)
        return (mean.float() - mdp.default_q) / alpha, var.float().sqrt() / alpha


    if args.act_norm == "ref":
        # floor: the policy also moves the joints that the references barely move
        a_mean, a_std = reference_action_stats()
        a_std = a_std.clamp(min=0.5)
        pi.obs_norm.fix(slice(OBS_A_PREV, OBS_A_PREV + 29), a_mean, a_std)
        for i0 in (INV_DIM, INV_DIM + 29) if wm.a_prev else (INV_DIM,):
            wm.in_norm.fix(slice(i0, i0 + 29), a_mean, a_std)
    opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
    start_it, wall_offset, sim_steps = 0, 0.0, 0
    if ck is not None:
        if ck["args"].get("robot", "g1") != args.robot:
            raise ValueError("Checkpoint robot does not match --robot")
        ck_dr = ck["args"].get("dr", "")
        if bool(parse_terms(ck_dr)) != bool(dr.terms):
            # all term sets share one xi layout (src/supermimic/envs/domain_randomization.py), which differs from the paper's 33 entries
            raise ValueError(f"Checkpoint was trained {f'with --dr {ck_dr}' if ck_dr else 'without --dr'}; resume it "
                             f"{'with some --dr terms' if ck_dr else 'without --dr'} (the xi layouts differ)")  # fmt: skip
        ck_xi = DRCfg(parse_terms(ck_dr)).xi_dim
        if ck_xi > dr.xi_dim:
            raise ValueError(f"Checkpoint was trained with --dr {ck_dr}; keep contact in --dr (the xi layouts differ)")
        if ck_xi < dr.xi_dim:
            widen_xi(ck, ck_xi, dr.xi_dim - ck_xi)
            print(f"[rank {RANK}] Added {dr.xi_dim - ck_xi} xi input(s) with zero weights to the checkpoint's world model "
                  "and critic.", flush=True)
        wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"])
        opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"])
        try:
            critic.load_state_dict(ck["critic"]), opt_v.load_state_dict(ck["opt_v"])
        except (RuntimeError, ValueError):
            print(f"[rank {RANK}] The checkpoint critic does not match --critic_layer_norm={args.critic_layer_norm}; "
                  "starting a new critic.", flush=True)
            critic = make_critic()
            opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
            ck.pop("critic_targ", None)
        start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
    if WORLD > 1:
        for t in (*wm.state_dict().values(), *pi.state_dict().values(), *critic.state_dict().values()):
            dist.broadcast(t, 0)
        torch.manual_seed(args.seed + RANK)  # different envs, noise and minibatches on every rank
        shard = torch.tensor([getattr(motion, "catalog_num_clips", len(motion.names)),
                              getattr(motion, "catalog_num_frames", motion.num_frames)], device=dev)
        dist.all_reduce(shard)
        num_clips, num_frames = shard.tolist()
    else:
        num_clips = getattr(motion, "catalog_num_clips", len(motion.names))
        num_frames = getattr(motion, "catalog_num_frames", motion.num_frames)
    critic_targ = copy.deepcopy(critic).requires_grad_(False)
    critic_targ.norm = critic.norm
    if ck is not None and "critic_targ" in ck:
        critic_targ.load_state_dict(ck["critic_targ"])
    del ck

    return TrainingComponents(args=args, dev=dev, dr=dr, dr_curriculum=dr_curriculum, motion=motion, mdp=mdp, wm=wm, pi=pi, critic=critic, critic_targ=critic_targ, opt_wm=opt_wm, opt_pi=opt_pi, opt_v=opt_v, start_it=start_it, wall_offset=wall_offset, sim_steps=sim_steps, num_clips=num_clips, num_frames=num_frames, goal_drop=GOAL_DROP, max_return=G_MAX, world=WORLD, rank=RANK, is_main=MAIN)

"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update.

Multi-GPU: one process per GPU (RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT, each process seeing a
single GPU, as started by ``sm_native.sh train --gpu 0,1,...``). Every rank simulates its own
``--num_envs`` environments on its own shard of the motion files; gradients and normalizers are
averaged over ranks, so the networks stay identical and see all motions.
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import os
import signal
import sys
import time

from supermimic.paths import REPO_ROOT
from supermimic.configs.train import build_parser

REPO = str(REPO_ROOT)
RANK, WORLD = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
MAIN = RANK == 0

from isaaclab.app import add_launcher_args, launch_simulation

parser = build_parser(add_launcher_args)
args = parser.parse_args()
os.environ["SUPERMIMIC_ROBOT"] = args.robot

STOP = {"signals": 0}


def _request_stop(sig, _frame):
    STOP["signals"] += 1
    if STOP["signals"] > 1:
        raise KeyboardInterrupt
    print(f"\n[{signal.Signals(sig).name}] Stopping after the current iteration and saving a checkpoint "
          "(send it again to quit without saving).", flush=True)


# The launcher's shell ignores SIGINT/SIGTERM so tee keeps logging; these handlers replace the inherited SIG_IGN.
signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from supermimic.buffer import ReplayBuffer
from supermimic.diagnostics import (CRITIC_FIELDS, EXTRA_FIELDS, POOL_FIELDS, WM_FIELDS, CriticMetrics, RealMetrics, WorldModelMetrics, wm_error_summary)
from supermimic.motion_pool import MotionPool
from supermimic.dr import LATE_TERMS, DRCfg, curriculum_scale, describe, describe_scale, parse_curriculum, parse_terms
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import (ACTION_CLIP, CRITIC_EXTRA, INV_DIM, MAX_STEP_REWARD, MDP, OBS_A_PREV, OBS_DIM, XI_DIM,
                            invariant_features, normalize_xi, reward_cfg, unflatten_state)  # fmt: skip
from supermimic.motion import CONTROL_DT, MotionLib, goal_frame_dim, parse_goal_drop
import supermimic.networks as networks
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.robot import DEFAULT_MOTION, G1_CFG, G1_JOINT_NAMES, action_scale, default_joint_pos_list, set_dynamics
args.motion_dir = args.motion_dir or DEFAULT_MOTION
dr_curriculum = parse_curriculum(args.dr_curriculum)
if dr_curriculum and not parse_terms(args.dr):
    raise ValueError("--dr_curriculum needs --dr terms")
if args.dr_ramp < 0:
    raise ValueError("--dr_ramp must be >= 0")
G_MAX = MAX_STEP_REWARD / (1 - args.gamma)

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
args.reward_cfg = vars(reward_cfg(args.dynamics))
args.simulation_timing = dict(control_dt=CONTROL_DT, physics_dt=CONTROL_DT/4, physics_substeps=4)
if ck is not None and args.wm_prediction == "velocity_integrated":
    if ck["args"].get("simulation_timing") != args.simulation_timing:
        raise ValueError("Integrated WM resume requires the checkpoint's control/physics timing")
args.self_collisions = bool(G1_CFG.spawn.articulation_props.enabled_self_collisions)
if ck is not None and ck["args"].get("self_collisions", False) != args.self_collisions:
    print(f"[rank {RANK}] Self-collision setting changed from {ck['args'].get('self_collisions', False)} "
          f"to {args.self_collisions}; replay is rebuilt from the new simulator and the world model has to re-adapt.",
          flush=True)
if MAIN:
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
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), reward_cfg(args.dynamics))
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
# The extended xi entries (src/supermimic/dr.py) lie in [-1, 1] and stay constant until a --dr_curriculum stage
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
        # all term sets share one xi layout (src/supermimic/dr.py), which differs from the paper's 33 entries
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


def sync_grads(m: nn.Module):
    """Average gradients over ranks before clipping, so every rank takes the same optimizer step."""
    if WORLD == 1:
        return
    ps = list(m.parameters())
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in ps])
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    for p, g in zip(ps, flat.split([p.numel() for p in ps])):
        p.grad = g.view_as(p)


def clip_finite_grads(m: nn.Module, max_norm: float):
    """clip_grad_norm_, and zero gradients when their norm is not finite (without a host sync, unlike
    the policy's skipped step): a NaN step would reach every rank's weights."""
    gn = nn.utils.clip_grad_norm_(m.parameters(), max_norm)
    bad = ~torch.isfinite(gn)
    for p in m.parameters():
        if p.grad is not None:
            p.grad.masked_fill_(bad, 0.0)


def stop_requested() -> bool:
    """A stop signal on any rank stops all ranks after the same iteration."""
    if WORLD == 1:
        return bool(STOP["signals"])
    flag = torch.tensor(STOP["signals"], device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag)


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, a_prev, xi):
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], normalize_xi(xi[:, j]), a_prev[:, j])
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.error_std(), ERR_FLOOR)
    raw = wm.state_error(S_hat, unflatten_state(s1))
    squared = (raw / std) ** 2  # (B, k, D)
    summary = wm_error_summary(raw, squared, mask, std, ERR_FLOOR)
    e = squared.mean(-1)
    return (e * mask).sum() / s1.shape[0], e[:, -1], e[:, 0], summary


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g):
    a = pi(o_hist, g)
    S1 = wm.step(S, a, xi_n, a_prev)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a.detach() if args.detach_a_prev else a, xi, f1))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, motion.goal(f1, K, GOAL_DROP)


def _reward_term(S, f, a, a_prev, qd_prev, ref_bodies):
    return mdp.reward(S, f, a, a_prev, qd_prev=qd_prev, ref_bodies=ref_bodies), mdp.soft_termination(S, f)


def _critic_input(S, f, xi_n, g, ref_body_p=None):
    return torch.cat((mdp.critic_features(S, f, xi_n, ref_body_p=ref_body_p), g), -1)


wm_rollout = torch.compile(_wm_rollout)
wm_err = torch.compile(_wm_err)
imag_step = torch.compile(_imag_step)
reward_term_c = torch.compile(_reward_term)
critic_input_gpu = torch.compile(_critic_input)


def reward_term(S, f, a, a_prev, qd_prev):
    # One batched transfer for the entire H x B imagined trajectory, outside torch.compile.
    reference = motion.body_reference(f)
    return reward_term_c(S, f, a, a_prev, qd_prev, reference)


def critic_input_c(S, f, xi_n, g):
    # Critic errors only use body positions, so do not transfer rotations or velocities.
    reference_p = motion.body_reference(f, positions_only=True)[0]
    return critic_input_gpu(S, f, xi_n, g, reference_p)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


from supermimic.runners.progress import CSV_FIELDS, SMOOTH_KEYS, build_tb_tags, print_progress, _hms
TB_TAGS = build_tb_tags(args.wm_window)


# ------------------------------------------------------------------------------------------------
def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    S_hat = wm_rollout(w["s"], w["a"], w["a_prev"], w["xi"])
    loss, e_last, e_first, wm_summary = wm_err(S_hat, w["s1"], w["mask"])
    wm_metrics.update(wm_summary)
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    sync_grads(wm)
    clip_finite_grads(wm, 1.0)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1), e_first.detach().mean()


def sample_valid_starts(B: int):
    return buffer.sample_valid_starts(B, Hh, HZ, motion.clip_id, motion.clip_end)


PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))


def policy_update():
    if PROFILE:
        torch.cuda.synchronize(); tq0 = time.time()
    s, h = sample_valid_starts(args.pi_batch)
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    xi_n = normalize_xi(xi)
    with torch.no_grad():
        o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
    o_hist = mdp.add_history_noise(o_hist, h["noise_source"])
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    g0 = motion.goal(f, K, GOAL_DROP)
    g = g0
    S0 = S
    Ss, As, Aps, Fs, Gs = [], [], [], [], []
    for t in range(HZ):
        Aps.append(a_prev)
        S, a_prev, o_hist, f, g = imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
        Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    f_all = flat(Fs)
    qd_prev = flat([S0[5]] + [St[5] for St in Ss[:-1]])
    r, d = reward_term(S_all, f_all, flat(As), flat(Aps), qd_prev)
    r, d = r.view(HZ, B), d.view(HZ, B)
    alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    J = (disc * c_t * r).sum(0)
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    v_end = torch.zeros(B, device=dev)
    if not args.no_critic:
        # the clamp at the largest possible return stops the policy from chasing critic overestimates
        v_end = critic(_critic_input(S, f, xi_n, g))
        J = J + (args.gamma**HZ) * c * v_end.clamp(-G_MAX, G_MAX)
        v_end = v_end.detach()
    loss = -J.mean()
    if PROFILE:
        torch.cuda.synchronize(); tq1 = time.time()
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    sync_grads(pi)
    if PROFILE:
        torch.cuda.synchronize(); tq2 = time.time()
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    if torch.isfinite(gn):
        opt_pi.step()
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    v_mse = torch.zeros((), device=dev)
    if not args.no_critic:
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        with torch.no_grad():
            V = critic_targ(X).clamp(-G_MAX, G_MAX)
            R = V[-1]
            targets = []
            for t in range(HZ - 1, -1, -1):
                R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1]).clamp(-G_MAX, G_MAX)  # (H, B)
        # weighted by the probability of being alive: states after a soft termination never bootstrap anything
        Xf, Yf, Wf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1), c_t.detach().reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v, y, w = critic(Xf[idx]), Yf[idx], Wf[idx]
                w_sum = w.sum().clamp(min=1e-6)
                huber = F.huber_loss(v, y, reduction="none", delta=1.0)
                v_loss = (w * huber).sum() / w_sum
                critic_metrics.update(v, y, w, huber)
                opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                sync_grads(critic)
                clip_finite_grads(critic, 1.0)
                opt_v.step()
        v_mse = (w * (v.detach() - y) ** 2).sum() / w_sum
        with torch.no_grad():
            for p_targ, p in zip(critic_targ.net.parameters(), critic.net.parameters()):
                p_targ.lerp_(p, args.critic_tau)
    if PROFILE:
        torch.cuda.synchronize()
        print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
    v_stats = (v_end.mean(), v_end.max(), (v_end > G_MAX).float().mean())
    return loss.detach(), v_mse, (r_sum / HZ / MAX_STEP_REWARD).mean(), gn.detach(), c.mean().detach(), *v_stats


def a_prev_gain(n: int = 256) -> torch.Tensor:
    """Largest singular value of da_t / da_{t-1} through the newest observation, at n buffer states:
    above 1 the imagined action loop (and its BPTT gradient) can grow from step to step."""
    s, h = buffer.sample_starts(n, Hh)
    with torch.no_grad():
        o = mdp.add_history_noise(
            mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"]), h["noise_source"])
    x = o[:, -1, OBS_A_PREV : OBS_A_PREV + 29].clone().requires_grad_(True)
    newest = torch.cat((o[:, -1, :OBS_A_PREV], x, o[:, -1, OBS_A_PREV + 29 :]), -1)
    a = pi(torch.cat((o[:, :-1], newest.unsqueeze(1)), 1), motion.goal(s["f"], K, GOAL_DROP))
    jac = torch.stack([torch.autograd.grad(a[:, j].sum(), x, retain_graph=j < 28)[0] for j in range(29)], 1)
    return torch.linalg.matrix_norm(jac, ord=2)


# ------------------------------------------------------------------------------------------------
sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    dr_scale = curriculum_scale(dr.terms, dr_curriculum, args.dr_ramp, start_it)
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma, dr=dr, dr_scale=dr_scale,
                      reset_a_prev=args.reset_a_prev, reset_temp=args.reset_temp)
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, xi_dim=dr.xi_dim)
    real_metrics = RealMetrics(dev)
    critic_metrics = CriticMetrics(dev)
    wm_metrics = WorldModelMetrics(dev, G1_JOINT_NAMES)
    obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)

    @torch.no_grad()
    def collect(n_steps: int, stats: dict):
        global obs_hist, sim_steps
        for _ in range(n_steps):
            S = env.state()
            o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
            fresh = env.ep_step == 0
            obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
            g = motion.goal(env.f, K, GOAL_DROP)
            pi.obs_norm.update(o)
            pi.goal_norm.update(g)
            a = pi(obs_hist, g)
            a = torch.clamp(a + args.action_noise * torch.randn_like(a), -ACTION_CLIP, ACTION_CLIP)
            out = env.step(a)
            buffer.add(out)
            real_metrics.update(out)
            S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
            wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"]), out["a_prev"]))
            wm.update_output_normalization(S0, S1)
            sim_steps += args.num_envs * WORLD
            stats["r"].append(out["r"].mean() / MAX_STEP_REWARD)
            for k in SMOOTH_KEYS:
                stats[k].append(out["terms"][k].mean())
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].sum())
                stats["ep_ret"].append(out["ep_ret_norm"].sum())
                stats["ep_count"].append(out["ep_len"].numel())
                stats["end_clip"].append(out["end_clip"].sum())
                stats["end_time"].append(out["end_time"].sum())
                stats["term"].append(out["term"].float().sum())
                stats["done"].append(out["done"].float().sum())

    def save_checkpoint(n_it: int) -> str:
        path = os.path.join(args.log_dir, f"ckpt_{n_it:06d}.pt")
        pool_states = None
        if isinstance(motion, MotionPool):
            state = motion.state_dict(env)
            if WORLD > 1:
                pool_states = [None] * WORLD if MAIN else None
                dist.gather_object(state, pool_states, dst=0)
            else:
                pool_states = [state]
        if not MAIN:
            return path
        ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "critic_targ": critic_targ.state_dict(),
              "opt_wm": opt_wm.state_dict(), "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": n_it,
              "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "cuda_memory": dict(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)}  # fmt: skip
        if pool_states is not None:
            ck["motion_pool"] = pool_states
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
        if pool_states is not None:
            small["motion_pool"] = pool_states
        torch.save(small, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"Saved checkpoint {path} (and last.pt)", flush=True)
        return path

    tb_url = os.environ.get("SUPERMIMIC_TB_URL")
    if MAIN:
        envs = f"{args.num_envs} envs" + (f" per GPU x {WORLD} GPUs = {args.num_envs * WORLD}" if WORLD > 1 else "")
        print("=" * 80)
        print(f" SuperMimic training | run {os.path.basename(os.path.normpath(args.log_dir))} | robot {args.robot} | {torch.cuda.get_device_name(0)}")
        print(f" Motion {args.motion_dir}: {num_clips} clips, {num_frames} frames" + (f" (split over {WORLD} GPUs)" if WORLD > 1 else ""))
        print(f" {envs} | iterations {start_it} -> {args.iterations} | world-model warm-up until {args.wm_only_iters}"
              f" | checkpoint every {args.ckpt_every}")
        print(f" Joint dynamics: {args.dynamics}")
        print(f" Robot self-collisions: {args.self_collisions}")
        print(f" Reward action-rate {mdp.cfg.w_action_rate:g}, joint-acc {mdp.cfg.w_joint_acc:g}; gamma {args.gamma:g}; "
              f"WM {args.wm_window} steps, Actor {HZ} steps; control {CONTROL_DT:g}s, physics {CONTROL_DT/4:g}s")
        print(" Domain randomization: friction, joint offsets, torso COM" + (f" + {describe(dr.terms)} (xi {dr.xi_dim})" if dr.terms else ""))
        if dr_curriculum:
            print(f"   curriculum: + {describe(dr.terms - LATE_TERMS) or '-'} from iteration {dr_curriculum[0]}, "
                  f"+ {describe(dr.terms & LATE_TERMS) or '-'} from {dr_curriculum[1]}, each widening over "
                  f"{args.dr_ramp} iterations; now {describe_scale(dr_scale)}")
        print(f" Stabilization: action inputs normalized by {args.act_norm}, actions "
              f"{'soft-clipped' if args.act_soft_clip else 'hard-clipped'} at {ACTION_CLIP:.0f}, reset a_prev "
              f"{args.reset_a_prev}{', a_prev detached in imagination' if args.detach_a_prev else ''}, critic "
              f"{'LayerNorm, ' if args.critic_layer_norm else ''}target EMA {args.critic_tau}, values clamped to +-{G_MAX:.0f}")
        if args.motion_exclude or args.reset_temp != 25.0:
            print(f" Motions excluded by {args.motion_exclude!r}" if args.motion_exclude else " All motions",
                  f"| reset sampling temperature {args.reset_temp:g}")
        print(f" Logs {args.log_dir}" + (f" | TensorBoard {tb_url}" if tb_url else ""))
        print(" Ctrl+C: stop after the current iteration and save a checkpoint")
        print("=" * 80, flush=True)

    stats = {"r": [], "ep_len": [], "ep_ret": [], "ep_count": [], "end_clip": [], "end_time": [], "term": [], "done": [], **{k: [] for k in SMOOTH_KEYS}}
    nonfinite_logged = 0
    # The replay buffer is not checkpointed, so a resumed run refills it before sampling world-model windows.
    if MAIN:
        print(f"Collecting {args.warmup_steps} warm-up steps; the first iterations compile kernels with torch.compile...", flush=True)
    collect(args.warmup_steps, stats)
    log_f = tb = None
    if MAIN:
        csv_path = os.path.join(args.log_dir, "train_log.csv")
        header = ",".join(CSV_FIELDS) + "\n"
        new_csv = not os.path.isfile(csv_path)
        if start_it and not new_csv:
            with open(csv_path) as fh:
                reader = csv.DictReader(fh)
                rows = [row for row in reader if row.get("iteration", "").isdigit()
                        and int(row["iteration"]) <= start_it]
            # Preserve the old columns by name; old rows have no new diagnostic observations.
            with open(csv_path + ".tmp", "w") as fh:
                writer = csv.DictWriter(fh, CSV_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows({k: row.get(k, "nan") for k in CSV_FIELDS} for row in rows)
            os.replace(csv_path + ".tmp", csv_path)
        log_f = open(csv_path, "a")
        if start_it == 0 or new_csv:
            log_f.write(header)
        tb = SummaryWriter(os.path.join(args.log_dir, "tensorboard"), purge_step=start_it or None, flush_secs=10)
    t_start = time.time() - wall_offset
    t_session = t_log = time.time()
    steps_log = sim_steps
    last_log_it = start_it
    step_acc = 0.0
    agg = {"wm": [], "wm8": [], "wm1": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": [], "vm": [], "vx": [], "vo": []}
    done_it = saved_it = start_it
    last_ckpt = "none yet"
    for it in range(start_it, args.iterations):
        if stop_requested():
            break
        if it == args.wm_only_iters and it > start_it and MAIN:
            print(f"World-model warm-up done; policy and critic updates start at iteration {it + 1} (first ones compile kernels).", flush=True)
        scale = curriculum_scale(dr.terms, dr_curriculum, args.dr_ramp, it)
        if scale != dr_scale:
            env.set_dr_scale(scale)
            if MAIN:
                print(f"Domain randomization ranges at iteration {it}: {describe_scale(scale)}", flush=True)
            dr_scale = scale
        lr = lr_at(it)
        for o in (opt_wm, opt_pi, opt_v):
            for gparam in o.param_groups:
                gparam["lr"] = lr
        step_acc += 1.0 / args.utd
        n = int(step_acc)
        step_acc -= n
        prof = PROFILE and it % 20 == 0 and it > args.wm_only_iters + 5
        if prof:
            torch.cuda.synchronize(); tp0 = time.time()
        collect(n, stats)
        if prof:
            torch.cuda.synchronize(); tp1 = time.time()
        for _ in range(args.wm_batches):
            l, e8, e1 = wm_update()
        if prof:
            torch.cuda.synchronize(); tp2 = time.time()
        agg["wm"].append(l), agg["wm8"].append(e8), agg["wm1"].append(e1)
        if it >= args.wm_only_iters:
            pl, vl, ir, gn, surv, vm, vx, vo = policy_update()
            if prof:
                torch.cuda.synchronize(); tp3 = time.time()
                print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
            agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
            agg["vm"].append(vm), agg["vx"].append(vx), agg["vo"].append(vo)
        done_it = it + 1
        if isinstance(motion, MotionPool) and args.motion_refresh_iters and done_it % args.motion_refresh_iters == 0:
            motion.refresh(env, buffer)
            if int(motion.catalog_len[motion.active_ids].max()) <= HZ:
                raise ValueError("No active clip can supply the imagination horizon; increase active ratio")
            print(f"[rank {RANK}] motion pool refreshed: {motion.metrics()}", flush=True)
        if it == start_it and MAIN:
            print(f"First iteration done after {time.time() - t_session:.0f} s; progress is printed every {args.log_every} iterations.", flush=True)
        if done_it % args.log_every == 0:
            mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
            total = lambda x: float(sum(float(v) for v in x))  # noqa: E731
            top = lambda x: max((float(v) for v in x), default=0.0)  # noqa: E731
            now = time.time()
            ips = (done_it - last_log_it) / (now - t_log)
            last_log_it = done_it
            sps = (sim_steps - steps_log) / (now - t_log)
            t_log, steps_log = now, sim_steps
            # losses and per-step rewards are averaged over ranks, maxima taken over ranks; episode statistics are pooled
            vals = [mean(agg[k]) for k in ("wm", "wm8", "pi", "v", "ir", "gn", "surv")] + [mean(stats["r"])]
            gain = a_prev_gain() if it >= args.wm_only_iters else torch.zeros(1, device=dev)
            vals += [mean(agg["vm"]), mean(agg["vo"]), float(gain.median())] + [mean(stats[k]) for k in SMOOTH_KEYS]
            maxes = [top(agg["vx"]), top(agg["gn"]), float(gain.max())]
            pooled = [total(stats["ep_len"]), total(stats["ep_ret"]), total(stats["ep_count"]), total(stats["term"]), total(stats["done"]),
                      float(env.n_nonfinite - nonfinite_logged), total(stats["end_clip"]), total(stats["end_time"])]  # fmt: skip
            nonfinite_logged = env.n_nonfinite
            if WORLD > 1:
                t = torch.tensor(vals + pooled, dtype=torch.float64, device=dev)
                dist.all_reduce(t)
                vals, pooled = [x / WORLD for x in t[: len(vals)].tolist()], t[len(vals) :].tolist()
                t = torch.tensor(maxes, dtype=torch.float64, device=dev)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                maxes = t.tolist()
            ep_len_sum, ep_ret_sum, n_ep, n_term, n_done, n_nonfinite, n_clip, n_time = pooled
            (v_end_mean, v_end_over, gain_med), (v_end_max, gn_max, gain_max) = vals[8:11], maxes
            dr_mean = sum(dr_scale.values()) / max(len(dr_scale), 1)
            row = [done_it, (now - t_start) / 3600, sim_steps, *vals[:8], ep_len_sum / max(n_ep, 1), ep_ret_sum / max(n_ep, 1),
                   n_term / max(n_done, 1), ips, lr, v_end_mean, v_end_max, v_end_over, gn_max, gain_med, gain_max, dr_mean,
                   int(n_nonfinite), -vals[11] / max(vals[12], 1e-9), vals[13]]  # fmt: skip
            extra = real_metrics.flush()
            pool_metrics = motion.metrics() if isinstance(motion, MotionPool) else {k: 0.0 for k in POOL_FIELDS}
            if WORLD > 1 and isinstance(motion, MotionPool):
                pm = torch.tensor([pool_metrics[k] for k in POOL_FIELDS], device=dev, dtype=torch.float64)
                dist.all_reduce(pm)
                pool_metrics = dict(zip(POOL_FIELDS, pm.tolist()))
                # Memory/counts are global sums; refresh latency is mean per rank.
                pool_metrics["pool_refresh_s"] /= WORLD
                pool_metrics["pool_refreshes"] /= WORLD
            extra.update(pool_metrics)
            extra.update(wm_nse1=mean(agg["wm1"]), success_frac=1-n_term/n_done if n_done else 0.0,
                         episodes_ended=n_done, episodes_failed=n_term, ended_clip=n_clip, ended_time=n_time,
                         imag_reward_normalized=vals[4])
            if WORLD > 1:
                e1 = torch.tensor(extra["wm_nse1"], device=dev)
                dist.all_reduce(e1, op=dist.ReduceOp.AVG)
                extra["wm_nse1"] = float(e1)
            extra.update(critic_metrics.flush())
            wm_group_metrics, wm_channel_metrics = wm_metrics.flush()
            extra.update(wm_group_metrics)
            row += [extra[k] for k in (*EXTRA_FIELDS, *CRITIC_FIELDS, *WM_FIELDS)]
            if MAIN:
                log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
                log_f.flush()
                for k, v in zip(CSV_FIELDS[1:], row[1:]):
                    tb.add_scalar(TB_TAGS[k], v, done_it)
                for tag, value in wm_channel_metrics.items():
                    tb.add_scalar(tag, value, done_it)
                eta = (now - t_session) / (done_it - start_it) * (args.iterations - done_it)
                print_progress(dict(zip(CSV_FIELDS, row)), n_done, sps, now - t_start, eta, last_ckpt,
                               args=args, g_max=G_MAX, control_dt=CONTROL_DT, dr_terms=dr.terms,
                               world=WORLD, pooled=isinstance(motion, MotionPool))
            stats = {k: [] for k in stats}
            agg = {k: [] for k in agg}
        if done_it % args.ckpt_every == 0 or done_it == args.iterations:
            last_ckpt, saved_it = save_checkpoint(done_it), done_it
    if done_it > saved_it:
        save_checkpoint(done_it)
    if MAIN:
        tb.close()
        log_f.close()
        finished = done_it == args.iterations
        print(f"Training {'finished' if finished else 'stopped'} at iteration {done_it}/{args.iterations} "
              f"({_hms(time.time() - t_start)} total).", flush=True)
        if not finished and done_it > 0:
            print(f"Resume with: --resume {os.path.join(args.log_dir, 'last.pt')}", flush=True)
    if WORLD > 1:
        dist.destroy_process_group()

"""Sim2sim check of a Tiangong3 SuperMimic policy in plain (CPU) MuJoCo.

Training runs Newton with the MuJoCo-Warp solver, implicit PD and 5 ms physics steps. Here the
generated MJCF runs in MuJoCo with an explicit 1 kHz PD loop clipped to the effort limits and
MuJoCo's default contact parameters, optionally with actuator delay, joint friction, gain errors and
pushes. Observations, goals, rewards and termination are the training ones (src/supermimic/tasks/tracking/mdp/context.py), so
the scores compare with scripts/evaluation/evaluate.py. Without --ckpt the reference joint angles are played
back open loop, which checks the physics setup alone.

Usage (in the container, e.g. ./sm.sh run --robot tiangong3 python scripts/evaluation/sim2sim_mujoco.py ...):
    python scripts/evaluation/sim2sim_mujoco.py --ckpt logs/<run>/ckpt_016000.pt --motion_dir data/wbc_two \
        [--delay_ms 10] [--joint_friction 0.3] [--kp_scale 0.8] [--push] [--record traj.npz]
"""

from __future__ import annotations

from pathlib import Path

import argparse
import json
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
os.environ["SUPERMIMIC_ROBOT"] = "tiangong3"

import mujoco
import numpy as np
import torch

import supermimic.models.common as networks
from supermimic.envs.domain_randomization import DRCfg
from supermimic.robots.kinematics import G1Kinematics
from supermimic.tasks.tracking.mdp import ACTION_CLIP, MAX_STEP_REWARD, MDP, OBS_DIM, XI_DIM, reward_cfg
from supermimic.motions.library import CONTROL_DT, MotionLib, goal_frame_dim, parse_goal_drop
from supermimic.motions.timing import checkpoint_goal_start, checkpoint_goal_stride
from supermimic.models import Policy
from supermimic.robots import DEFAULT_MOTION, JOINT_NAMES, MJCF_PATH, action_scale, default_joint_pos_list
from supermimic.robots.tiangong3 import get_dynamics, motor_params, set_dynamics

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=None, help="policy checkpoint; without it the reference is played back open loop")
parser.add_argument("--motion_dir", default=None, help="motion directory or a single NPZ")
parser.add_argument("--clips", nargs="*", default=None, help="motion names (default: all)")
parser.add_argument("--physics_dt", type=float, default=0.001)
parser.add_argument("--delay_ms", type=float, default=0.0, help="the previous joint target still holds this long into each step")
parser.add_argument("--joint_friction", type=float, default=0.0, help="frictionloss on every joint [N m]")
parser.add_argument("--kp_scale", type=float, default=1.0)
parser.add_argument("--kd_scale", type=float, default=1.0)
parser.add_argument("--push", action="store_true", help="random root velocity changes every 1-5 s, as with train.py --dr push")
parser.add_argument("--self_collision", action="store_true", help="training disables self-collision")
parser.add_argument("--dynamics", choices=["default", "dexevt"], default=None,
                    help="joint dynamics preset of src/supermimic/robots/tiangong3.py (default: the checkpoint's)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--record", default=None, help="save trajectories (npz) for scripts/evaluation/render.py")
parser.add_argument("--out", default=None, help="write the results as JSON")
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(MJCF_PATH)
    spec.option.timestep = args.physics_dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for name in JOINT_NAMES:
        j = spec.joint(name)
        j.armature = motor_params(name)[4]
        j.frictionloss = args.joint_friction
    if not args.self_collision:
        for g in spec.geoms:
            g.conaffinity = 0  # robot geoms then only collide with the floor added below
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    return spec.compile()


dev = args.device
set_dynamics(args.dynamics or (torch.load(args.ckpt, map_location="cpu")["args"].get("dynamics", "default") if args.ckpt else "default"))
kin = G1Kinematics(dev)
motion = MotionLib.from_dir(args.motion_dir or DEFAULT_MOTION, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), reward_cfg(get_dynamics()))
clips = [motion.names.index(c) for c in args.clips] if args.clips else list(range(len(motion.names)))

m = build_model()
d = mujoco.MjData(m)
jid = [m.joint(n).id for n in JOINT_NAMES]
qadr, vadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
effort, kp, kd = (np.array([motor_params(n)[i] for n in JOINT_NAMES]) for i in (0, 2, 3))
kp, kd = kp * args.kp_scale, kd * args.kd_scale
alpha = np.array([action_scale(n) for n in JOINT_NAMES])
default_q = np.array(default_joint_pos_list())
q_lo, q_hi = kin.q_lo.cpu().numpy(), kin.q_hi.cpu().numpy()
n_sub = round(CONTROL_DT / args.physics_dt)
n_delay = min(round(args.delay_ms * 1e-3 / args.physics_dt), n_sub)
push_cfg = DRCfg()
push_half = np.array(push_cfg.push_lin_vel + push_cfg.push_ang_vel)
rng = np.random.default_rng(args.seed)
xi = torch.zeros(1, XI_DIM, device=dev)  # no joint offsets

pi = None
if args.ckpt:
    ck = torch.load(args.ckpt, map_location=dev)
    if ck["args"].get("robot", "g1") != "tiangong3":
        raise ValueError("not a tiangong3 checkpoint")
    K, Hh = ck["args"]["goal_frames"], ck["args"]["hist"]
    motion.goal_stride = checkpoint_goal_stride(ck["args"])
    motion.goal_start = checkpoint_goal_start(ck["args"])
    networks.AMP_DTYPE = torch.bfloat16 if ck["args"].get("amp", "none") == "bf16" and dev.startswith("cuda") else None
    GOAL_DROP = parse_goal_drop(ck["args"].get("goal_drop", ""))
    pi = Policy(OBS_DIM, goal_frame_dim(GOAL_DROP) * K, hist=Hh, soft_clip_actions=ck["args"].get("act_soft_clip", False)).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
# previous action at reset, as in training (src/supermimic/envs/tracking_env.py)
RESET_A_PREV = ck["args"].get("reset_a_prev", "zero") if args.ckpt else "pose"


def rotation() -> np.ndarray:
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, d.qpos[3:7])
    return R.reshape(3, 3)


def state():
    """Training state (p, R, v, w, q, qd); MuJoCo's free joint has the angular velocity in the body frame."""
    R = rotation()
    S = (d.qpos[0:3], R, d.qvel[0:3], R @ d.qvel[3:6], d.qpos[qadr], d.qvel[vadr])
    return tuple(torch.tensor(np.array(x), dtype=torch.float32, device=dev).unsqueeze(0) for x in S)


def reset(f: int):
    p, R, v, w, q, qd = (x[0].double().cpu().numpy() for x in motion.ref_state(torch.tensor([f], device=dev)))
    mujoco.mj_resetData(m, d)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, R.reshape(-1))
    d.qpos[0:3], d.qpos[3:7] = p, quat
    d.qpos[qadr] = np.clip(q, q_lo, q_hi)
    d.qvel[0:3], d.qvel[3:6], d.qvel[vadr] = v, R.T @ w, qd
    mujoco.mj_forward(m, d)


@torch.no_grad()
def rollout(c: int):
    f, end = int(motion.clip_start[c]), int(motion.clip_end[c])
    reset(f)
    tgt_prev = d.qpos[qadr].copy()
    a_prev = torch.zeros(1, 29, device=dev)
    if RESET_A_PREV == "pose":
        a_prev[0] = torch.tensor((tgt_prev - default_q) / alpha, dtype=torch.float32, device=dev).clamp(-ACTION_CLIP, ACTION_CLIP)
    hist, ret, failed, steps = None, 0.0, False, 0
    next_push = rng.uniform(*push_cfg.push_interval_s) / CONTROL_DT
    rec = {"root_p": [], "root_quat": [], "q": [], "f": []}
    while True:
        S = state()
        ft = torch.tensor([f], device=dev)
        rec["root_p"].append(d.qpos[0:3].copy()), rec["root_quat"].append(d.qpos[3:7].copy())
        rec["q"].append(d.qpos[qadr].copy()), rec["f"].append(f)
        if pi is None:
            a = (motion.q[ft + 1] - mdp.default_q) / torch.tensor(alpha, dtype=torch.float32, device=dev)
        else:
            o = mdp.obs(S, a_prev, xi, ft)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            a = pi(hist, motion.goal(ft, K, GOAL_DROP))
        target = default_q + alpha * a[0].double().cpu().numpy()
        for i in range(n_sub):
            tgt = tgt_prev if i < n_delay else target
            d.qfrc_applied[vadr] = np.clip(kp * (tgt - d.qpos[qadr]) - kd * d.qvel[vadr], -effort, effort)
            mujoco.mj_step(m, d)
        tgt_prev = target
        f1, steps = f + 1, steps + 1
        S1, f1t = state(), torch.tensor([f1], device=dev)
        ret += mdp.reward(S1, f1t, a, a_prev, qd_prev=S[5]).item()
        if bool(mdp.termination(S1, f1t)):
            failed = True
            break
        if f1 >= end:
            break
        if args.push and steps >= next_push:
            dv = rng.uniform(-1.0, 1.0, 6) * push_half
            d.qvel[0:3] += dv[:3]
            d.qvel[3:6] += rotation().T @ dv[3:]
            next_push = steps + rng.uniform(*push_cfg.push_interval_s) / CONTROL_DT
        a_prev, f = a, f1
    return ret / ((end - int(motion.clip_start[c])) * MAX_STEP_REWARD), failed, steps, rec


print(f"MuJoCo {mujoco.__version__} sim2sim | {'policy ' + args.ckpt if pi is not None else 'open-loop reference playback'}")
print(f"physics {1e3 * args.physics_dt:g} ms explicit PD | delay {1e3 * n_delay * args.physics_dt:g} ms | joint friction "
      f"{args.joint_friction:g} N m | kp x{args.kp_scale:g} kd x{args.kd_scale:g} | push {'on' if args.push else 'off'} | "
      f"self-collision {'on' if args.self_collision else 'off'} | {len(clips)} motion(s)", flush=True)  # fmt: skip
results, recs, t0 = {}, [], time.time()
for c in clips:
    norm, failed, steps, rec = rollout(c)
    results[motion.names[c]] = {"norm_return": round(norm, 4), "failed": failed, "steps": steps}
    recs.append(rec)
    print(f"  {motion.names[c]}: norm return {norm:.4f}" + (f", fell after {steps * CONTROL_DT:.1f} s" if failed else ""), flush=True)
failed = [n for n, r in results.items() if r["failed"]]
summary = {
    "norm_return": float(np.mean([r["norm_return"] for r in results.values()])),
    "n_failed": len(failed),
    "failed": failed,
    "settings": {k: getattr(args, k) for k in ("ckpt", "physics_dt", "delay_ms", "joint_friction", "kp_scale", "kd_scale", "push", "self_collision", "seed")},
}
print(json.dumps(summary, indent=1) + f"\n({time.time() - t0:.0f} s)")
if args.out:
    json.dump(summary | {"per_clip": results}, open(args.out, "w"), indent=1)
if args.record:
    T = max(len(r["f"]) for r in recs)
    pad = lambda x: np.concatenate((np.array(x), np.repeat(np.array(x)[-1:], T - len(x), 0)))  # noqa: E731
    traj = {k: np.stack([pad(r[k]) for r in recs], 1) for k in recs[0]}
    traj["alive"] = np.stack([np.arange(T) < len(r["f"]) for r in recs], 1)
    sel = torch.tensor(clips)
    np.savez_compressed(args.record, control_dt=CONTROL_DT, names=np.array([motion.names[c] for c in clips]), **traj,
                        ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                        clip_start=motion.clip_start.cpu()[sel].numpy(), clip_len=motion.clip_len.cpu()[sel].numpy())  # fmt: skip
    print("wrote", args.record)

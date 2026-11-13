"""Checkpoint evaluation (paper Appendix F.1) on all motions of the library.

Each motion plays once from its first frame with friction 1.0, no domain randomization, no
observation or action noise. A motion fails if the rollout triggers the training termination.
Normalized return is the undiscounted reward sum divided by its maximum attainable value.

``--dr`` keeps randomization, for deployment checks: ``base`` is the training randomization of
friction, joint offsets and torso COM plus observation noise, and src/supermimic/envs/domain_randomization.py terms (or ``all``)
come on top of it. Each motion then plays ``--repeats`` times with its own draws, and every checkpoint
evaluated with the same ``--seed`` (several ``--ckpt``, or a run's checkpoints) meets the same draws.

Run as a watcher: ``python scripts/evaluation/evaluate.py --run_dir logs/<run> --watch``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

from supermimic.utils.paths import REPO_ROOT

REPO = str(REPO_ROOT)
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--ckpt", type=str, nargs="+", default=None, help="evaluate these checkpoints")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
parser.add_argument("--motion_dir", type=str, default=None, help="Motion directory or a single CSV/NPZ file")
parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
parser.add_argument("--robot", choices=["g1", "tiangong3"], default=os.environ.get("SUPERMIMIC_ROBOT", "g1"))
parser.add_argument("--dr", type=str, default="none", help="none: nominal | base: training randomization and observation "
                    "noise | base plus src/supermimic/envs/domain_randomization.py terms (comma list, or all)")  # fmt: skip
parser.add_argument("--repeats", type=int, default=1, help="rollouts per motion, each with its own random draws (with --dr)")
parser.add_argument("--seed", type=int, default=0, help="seed of the random draws, the same for every checkpoint")
parser.add_argument("--dynamics", choices=["default", "dexevt"], default=None,
                    help="joint dynamics preset (default: from the run's args.json, else from the first --ckpt)")
add_launcher_args(parser)
args = parser.parse_args()
os.environ["SUPERMIMIC_ROBOT"] = args.robot

import numpy as np
import torch

import supermimic.models.common as networks
from supermimic.envs.domain_randomization import DRCfg, describe, parse_terms
from supermimic.envs.tracking_env import TrackingSim
from supermimic.envs.sim_context import make_sim
from supermimic.robots.kinematics import G1Kinematics
from supermimic.tasks.tracking.mdp import MAX_STEP_REWARD, OBS_DIM, MDP, reward_cfg
from supermimic.motions.library import CONTROL_DT, MotionLib, goal_frame_dim, motion_group, parse_goal_drop
from supermimic.motions.timing import checkpoint_goal_stride
from supermimic.models import Policy
from supermimic.robots import DEFAULT_MOTION, default_joint_pos_list, set_dynamics
args.motion_dir = args.motion_dir or DEFAULT_MOTION


def run_dynamics() -> str:
    if args.dynamics:
        return args.dynamics
    path = os.path.join(args.run_dir, "args.json")
    if os.path.isfile(path):
        return json.load(open(path)).get("dynamics", "default")
    if args.ckpt:
        return torch.load(args.ckpt[0], map_location="cpu")["args"].get("dynamics", "default")
    return "default"


torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
DYNAMICS = run_dynamics()
set_dynamics(DYNAMICS)
kin = G1Kinematics(dev)
motion = MotionLib.from_dir(args.motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), reward_cfg(DYNAMICS))
C = len(motion.names)
dr = DRCfg.for_dynamics(parse_terms(args.dr), DYNAMICS)
RANDOMIZE = args.dr not in ("", "none")
R = args.repeats
SUFFIX = "" if not RANDOMIZE else "_dr-" + args.dr.replace(",", "+")
GROUPS: dict[str, list[int]] = {}
for _i, _name in enumerate(motion.names):
    GROUPS.setdefault(motion_group(_name), []).append(_i)


def group_table(res: dict, top: int = 25) -> str:
    """Motion groups by failure rate, worst first."""
    rows = sorted(res["per_group"].items(), key=lambda kv: (-kv[1]["failed"] / kv[1]["rollouts"], kv[1]["norm_return"]))
    lines = [f"  {'group':<40} {'failed':>12} {'norm return':>12}"]
    lines += [f"  {g[:40]:<40} {v['failed']:>5}/{v['rollouts']:<6} {v['norm_return']:>12.4f}" for g, v in rows[:top]]
    if len(rows) > top:
        lines.append(f"  ... {len(rows) - top} more groups in the json")
    return "\n".join(lines)


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
    a_ = ck["args"]
    if a_.get("robot", "g1") != args.robot:
        raise ValueError("Checkpoint robot does not match --robot")
    K, Hh = a_["goal_frames"], a_["hist"]
    motion.goal_stride = checkpoint_goal_stride(a_)
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    goal_drop = parse_goal_drop(a_.get("goal_drop", ""))
    pi = Policy(OBS_DIM, goal_frame_dim(goal_drop) * K, hist=Hh, soft_clip_actions=a_.get("act_soft_clip", False)).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    torch.manual_seed(args.seed)  # per-episode draws, pushes and observation noise repeat for every checkpoint
    ids = torch.arange(C * R, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset_a_prev = a_.get("reset_a_prev", "zero")  # as in training
    env.reset(ids, motion.clip_start.repeat(R))
    clip_end = motion.clip_end.repeat(R)
    ret = torch.zeros(C * R, device=dev)
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "qd": [], "a": [], "f": [], "alive": []}
    while not bool(env.done_eval.all()):
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        if RANDOMIZE:
            o = mdp.add_noise(o)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K, goal_drop))
        alive = ~env.done_eval.clone()
        if record:  # the first rollout of every motion
            from supermimic.utils.rotations import mat_to_quat

            traj["root_p"].append(S[0][:C].cpu()), traj["root_quat"].append(mat_to_quat(S[1][:C]).cpu())
            traj["q"].append(S[4][:C].cpu()), traj["qd"].append(S[5][:C].cpu()), traj["a"].append(a[:C].cpu())
            traj["f"].append(env.f[:C].cpu()), traj["alive"].append(alive[:C].cpu())
        out = env.step(a)
        ret += torch.where(alive, out["r"], 0.0)
        env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
    norm = (ret / ((motion.clip_len.repeat(R) - 1).float() * MAX_STEP_REWARD)).view(R, C)
    n_fail = env.failed_eval.view(R, C).sum(0)
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / (R * (motion.clip_len - 1).sum() * MAX_STEP_REWARD)).item(),
        "n_failed": int(n_fail.sum().item()),
        "failed": [motion.names[i] + (f" x{int(n_fail[i])}" if R > 1 else "") for i in n_fail.nonzero().flatten().tolist()],
        "per_clip": {motion.names[i]: round(norm[:, i].mean().item(), 4) for i in range(C)},
        "per_group": {g: {"rollouts": R * len(ix), "failed": int(n_fail[ix].sum().item()), "norm_return": round(norm[:, ix].mean().item(), 4)}
                      for g, ix in GROUPS.items()},  # fmt: skip
    }
    if RANDOMIZE:
        res |= {"dr": args.dr, "repeats": R, "seed": args.seed}
    if record:
        np.savez_compressed(record, control_dt=CONTROL_DT, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
    return res


sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    torch.manual_seed(args.seed)  # start-up draws
    env = TrackingSim(sim_cfg, C * R, motion, mdp, eval_mode=True, dr=dr, randomize=RANDOMIZE)
    if RANDOMIZE:
        print(f"Randomized evaluation: friction, joint offsets, torso COM, observation noise"
              + (f", {describe(dr.terms)}" if dr.terms else "") + f"; {R} rollout(s) per motion, seed {args.seed}", flush=True)  # fmt: skip
    if args.ckpt:
        results = []
        for i, path in enumerate(args.ckpt):
            record = args.record and (args.record if len(args.ckpt) == 1 else args.record.replace(".npz", f"_{i}.npz"))
            res = evaluate(env, torch.load(path, map_location=dev), record)
            summary = {k: v for k, v in res.items() if k not in ("per_clip", "per_group")}
            print(f"== {path}\n" + json.dumps(summary, indent=1) + "\n" + group_table(res), flush=True)
            results.append((path, res))
        if len(results) > 1:
            print("\nSame draws for all checkpoints:")
            for path, res in results:
                print(f"  norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C * R}  {path}")
        sys.exit(0)
    out_csv = os.path.join(args.run_dir, f"eval{SUFFIX}.csv")
    done = set()
    if os.path.exists(out_csv):
        done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:]}
    else:
        open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
    while True:
        ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
        todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
        todo = [c for c in todo if (int(os.path.basename(c)[5:11]) // 2000) % args.every == 0]
        for c in todo:
            it = int(os.path.basename(c)[5:11])
            ck = torch.load(c, map_location=dev)
            t0 = time.time()
            res = evaluate(env, ck)
            line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n"
            open(out_csv, "a").write(line)
            json.dump(res, open(c.replace(".pt", f"_eval{SUFFIX}.json"), "w"), indent=1)
            print(f"[eval{SUFFIX}] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C * R} ({time.time() - t0:.0f}s)", flush=True)
            print(group_table(res, top=10), flush=True)
            done.add(it)
        if not args.watch:
            break
        time.sleep(60)

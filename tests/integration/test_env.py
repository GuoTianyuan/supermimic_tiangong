"""Check FK vs simulator, state round-trip, and reward on reference playback."""

from pathlib import Path

import argparse
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--robot", choices=["g1", "tiangong3"], default=os.environ.get("SUPERMIMIC_ROBOT", "g1"))
parser.add_argument("--motion_dir", default=None)
add_launcher_args(parser)
args = parser.parse_args()
os.environ["SUPERMIMIC_ROBOT"] = args.robot

import torch

from supermimic.envs.tracking_env import TrackingSim
from supermimic.envs.sim_context import make_sim
from supermimic.robots.kinematics import G1Kinematics
from supermimic.tasks.tracking.mdp import MDP, MAX_STEP_REWARD
from supermimic.motions.library import CONTROL_DT, MotionLib
from supermimic.robots import TRACKED_BODIES, DEFAULT_MOTION, default_joint_pos_list
from supermimic.utils.rotations import quat_to_mat, rot_angle_sq

dev = "cuda:0"
kin = G1Kinematics(dev)
t0 = time.time()
motion = MotionLib.from_dir(args.motion_dir or DEFAULT_MOTION, kin, dev)
print(f"motion: {len(motion.names)} clips, {motion.num_frames} frames ({motion.num_frames * CONTROL_DT / 3600:.2f} h), bins {motion.num_bins}, load {time.time() - t0:.1f}s")
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, eval_mode=True)
    clips = torch.arange(args.num_envs, device=dev) % len(motion.names)
    frames = motion.clip_start[clips] + torch.minimum(motion.clip_len[clips] - 2, torch.full_like(clips, 100))
    env.reset(torch.arange(args.num_envs, device=dev), frames)
    env.scene.update(0.0)
    S = env.state()
    ref = motion.ref_state(frames)
    names = ["p", "R", "v", "w", "q", "qd"]
    for n, a, b in zip(names, S, ref):
        print(f"roundtrip {n}: max err {(a - b).abs().max().item():.2e}")
    # FK vs sim body frames
    d = env.robot.data
    ids = [env.robot.body_names.index(b) for b in TRACKED_BODIES]
    sp = d.body_link_pose_w.torch[:, ids, :3]
    sq = d.body_link_pose_w.torch[:, ids, 3:7]
    sR_xyzw = quat_to_mat(torch.cat((sq[..., 3:4], sq[..., :3]), -1))
    sR_wxyz = quat_to_mat(sq)
    bp, bR, bv, bw = kin.forward(*S)
    print("FK pos err", (bp - sp).norm(dim=-1).max().item())
    print("FK rot err xyzw", rot_angle_sq(bR, sR_xyzw).sqrt().max().item(), " wxyz", rot_angle_sq(bR, sR_wxyz).sqrt().max().item())
    sv = d.body_link_vel_w.torch[:, ids]
    print("FK linvel err", (bv - sv[..., :3]).norm(dim=-1).max().item(), " angvel err", (bw - sv[..., 3:]).norm(dim=-1).max().item())
    print("ref body vs FK(sim) pos err", (motion.body_reference(frames)[0] - bp).norm(dim=-1).max().item())
    # open-loop reference playback via joint targets
    rs = []
    # Evaluation does not reset completed clips; stop before short samples run out.
    steps = min(150, int((motion.clip_end[clips] - frames).min().item()))
    for t in range(steps):
        f1 = env.f + 1
        a = (motion.q[f1] - env.default_q) / env.alpha
        out = env.step(a)
        rs.append(out["r"].mean().item() / MAX_STEP_REWARD)
        if t % 25 == 0:
            print(t, "norm reward", rs[-1], "term frac", env.failed_eval.float().mean().item(), "e_body_pos", out["terms"]["e_body_pos"].mean().item())

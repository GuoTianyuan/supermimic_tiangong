"""Play a checkpoint in this checkout's original Newton/MuJoCo-Warp environment.

Only display, wall-clock pacing and episode repetition are added. Physics, actor
observations, goals, rewards, resets and termination come from the training code.
Use the matching 50/100 Hz checkout; --control-hz guards against a wrong checkout.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ["SUPERMIMIC_ROBOT"] = "tiangong3"

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--motion", required=True, help="A single NPZ, not the full dataset")
parser.add_argument("--control-hz", type=int, choices=(50, 100), required=True)
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--loops", type=int, default=0, help="0 repeats until Ctrl+C")
parser.add_argument("--speed", type=float, default=1.0, help="Requested maximum playback speed; 0 is unpaced")
parser.add_argument("--render-hz", type=float, default=50.0, help="Maximum display refresh rate in simulated time; control rate is unchanged")
parser.add_argument("--no-wait", action="store_true", help="Start without waiting for Enter")
add_launcher_args(parser)
args = parser.parse_args()
args.dr = "none"
if args.visualizer is None:
    args.visualizer = "viser"
if args.speed < 0 or args.loops < 0:
    parser.error("--speed and --loops must be nonnegative")
if args.render_hz <= 0:
    parser.error("--render-hz must be positive")
if not Path(args.motion).is_file():
    parser.error("--motion must point to one motion file")

import torch
import supermimic.models.common as networks
from supermimic.envs.domain_randomization import DRCfg
from supermimic.envs.tracking_env import TrackingSim
from supermimic.envs.sim_context import make_sim
from supermimic.robots.kinematics import G1Kinematics
from supermimic.tasks.tracking.mdp import MDP, MAX_STEP_REWARD, OBS_DIM, reward_cfg
from supermimic.motions.library import CONTROL_DT, MotionLib, goal_frame_dim, parse_goal_drop
from supermimic.motions.timing import checkpoint_goal_start, checkpoint_goal_stride, goal_offsets
from supermimic.tasks.tracking.config import checkpoint_reward_cfg
from supermimic.models import Policy
from supermimic.robots import default_joint_pos_list, set_dynamics

if abs(CONTROL_DT - 1 / args.control_hz) > 1e-9:
    parser.error(f"This checkout uses {1 / CONTROL_DT:g} Hz, but --control-hz is {args.control_hz}")
dev = args.device
torch.backends.cuda.matmul.allow_tf32 = True
ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
cfg = ck["args"]
if cfg.get("robot", "g1") != "tiangong3":
    parser.error("This player expects a tiangong3 checkpoint")
set_dynamics(cfg.get("dynamics", "default"))
networks.AMP_DTYPE = torch.bfloat16 if cfg.get("amp", "none") == "bf16" else None
kin = G1Kinematics(dev)
motion = MotionLib.from_dir(args.motion, kin, dev)
motion.goal_stride = checkpoint_goal_stride(cfg)
motion.goal_start = checkpoint_goal_start(cfg)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), checkpoint_reward_cfg(cfg))
K, H = cfg["goal_frames"], cfg["hist"]
drop = parse_goal_drop(cfg.get("goal_drop", ""))
pi = Policy(OBS_DIM, goal_frame_dim(drop) * K, hist=H, soft_clip_actions=cfg.get("act_soft_clip", False)).to(dev)
pi.load_state_dict(ck["pi"])
pi.eval()
sim_cfg = make_sim(dev, args)
selection = args.visualizer.split(",") if isinstance(args.visualizer, str) else args.visualizer
if selection == ["viser"]:
    from isaaclab.visualizers import VisualizerCfg
    from isaaclab_visualizers.viser import ViserBackendCfg
    sim_cfg.visualizer_cfgs = [VisualizerCfg(backend=ViserBackendCfg(port=args.port, bind_address="127.0.0.1"))]

try:
    with launch_simulation(sim_cfg, args):
        torch.manual_seed(0)
        env = TrackingSim(sim_cfg, 1, motion, mdp, eval_mode=True, dr=DRCfg(), randomize=False)
        env.reset_a_prev = cfg.get("reset_a_prev", "zero")
        print(f"[play] {args.ckpt}\n[play] control dt={CONTROL_DT:g}s; nominal training physics; no DR/noise; WM is not used", flush=True)
        print(f"[play] future offsets={goal_offsets(K, motion.goal_stride, motion.goal_start)} control frames", flush=True)
        print(f"[play] requested speed={args.speed:g} (0=unpaced), display cap={args.render_hz:g} Hz; actual speed is reported below", flush=True)
        ids = torch.zeros(1, dtype=torch.long, device=dev)
        episode = 0
        with torch.no_grad():
            while args.loops == 0 or episode < args.loops:
                torch.manual_seed(0)
                env.done_eval[:] = False
                env.failed_eval[:] = False
                env.reset(ids, motion.clip_start[:1])
                env.sim.render()
                if episode == 0 and not args.no_wait:
                    input(f"Open the viewer (Viser port {args.port}), then press Enter to play: ")
                hist, total, steps = None, 0.0, 0
                wall_start = time.perf_counter()
                report_wall, report_steps = wall_start, 0
                next_render = 1.0 / args.render_hz
                while not bool(env.done_eval.all()):
                    state = env.state()
                    obs = mdp.obs(state, env.a_prev, env.xi, env.f)
                    hist = obs[:, None].repeat(1, H, 1) if hist is None else torch.cat((hist[:, 1:], obs[:, None]), 1)
                    action = pi(hist, motion.goal(env.f, K, drop))
                    out = env.step(action)
                    total += out["r"].item()
                    steps += 1
                    simulated = steps * CONTROL_DT
                    if simulated + 1e-9 >= next_render or bool(env.done_eval.all()):
                        env.sim.render()
                        next_render = (int(simulated * args.render_hz + 1e-9) + 1) / args.render_hz
                    if args.speed:
                        remaining = wall_start + simulated / args.speed - time.perf_counter()
                        if remaining > 0:
                            time.sleep(remaining)
                    now = time.perf_counter()
                    if now - report_wall >= 2.0:
                        rate = (steps - report_steps) * CONTROL_DT / (now - report_wall)
                        print(f"[play timing] simulated={simulated:.2f}s wall={now-wall_start:.2f}s recent_speed={rate:.2f}x", flush=True)
                        report_wall, report_steps = now, steps
                episode += 1
                wall_elapsed = time.perf_counter() - wall_start
                norm = total / ((int(motion.clip_len[0]) - 1) * MAX_STEP_REWARD)
                print(f"[play] episode={episode} simulated={steps * CONTROL_DT:.2f}s wall={wall_elapsed:.2f}s actual_speed={steps * CONTROL_DT / wall_elapsed:.2f}x reward={norm:.6f} failed={bool(env.failed_eval[0])}", flush=True)
                if args.speed and (args.loops == 0 or episode < args.loops):
                    time.sleep(0.5)
except KeyboardInterrupt:
    print("\n[play] stopped", flush=True)

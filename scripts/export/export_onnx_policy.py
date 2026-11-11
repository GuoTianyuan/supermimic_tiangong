"""Export a SuperMimic policy WITHOUT a motion baked in to ONNX, for xmigcs policy/smwbc.

The deployment computes the goal from a motion NPZ itself (xmigcs policy/smwbc/motion/), so one ONNX
serves every motion the policy can track:

    inputs   obs   (1, hist * 99)   observation history, oldest first; each frame is
                                    [q - default_q (29), qd (29), base ang vel in body frame (3),
                                     projected gravity (3), previous action (29),
                                     first two columns of R_robot^T R_ref(f), row-major (6)]
             goal  (1, K * 73)      MotionLib.goal(f, K): for frames f+1..f+K (clamped to the clip),
                                    [yaw-frame root offset (3), 6d yaw-frame root rotation (6),
                                     root lin vel in root frame (3), root ang vel in root frame (3),
                                     q (29), 0.1 * qd (29)]
    outputs  actions (1, 29)        joint target = default_joint_pos + action_scale * actions

Joint order, default pose, action scale, PD gains, history length and goal frames are stored as metadata.
Sample inputs and the PyTorch actions are saved as <out>_check.npz (compare with onnxruntime). With --motion,
the goal table of that motion is saved as <out>_goals.npz so the deployment's goal computation can be checked
against MotionLib.
Usage:
    python scripts/export/export_onnx_policy.py --ckpt logs/<run>/ckpt_100000.pt [--motion data/wbc_full/<m>.npz] [--out x.onnx]
"""

from __future__ import annotations

from pathlib import Path

import argparse
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--motion", default=None, help="optional NPZ motion; its goal table is saved for checking the deployment")
ap.add_argument("--out", default=None, help="default: <run>_<ckpt>_policy.onnx next to the checkpoint")
args = ap.parse_args()
os.environ["SUPERMIMIC_ROBOT"] = "tiangong3"

import numpy as np
import onnx
import torch
import torch.nn as nn

import supermimic.networks as networks
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import ACTION_CLIP, OBS_DIM
from supermimic.motion import CONTROL_DT, MotionLib
from supermimic.networks import Policy
from supermimic.robot import JOINT_NAMES, action_scale, default_joint_pos_list
from supermimic.tiangong3 import motor_params, set_dynamics


class Deploy(nn.Module):
    def __init__(self, pi: Policy):
        super().__init__()
        self.pi = pi

    def forward(self, obs: torch.Tensor, goal: torch.Tensor):
        a = self.pi(obs.reshape(obs.shape[0], self.pi.hist, OBS_DIM), goal)
        return a.clamp(-ACTION_CLIP, ACTION_CLIP)


def main():
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a_ = ck["args"]
    if a_.get("robot", "g1") != "tiangong3":
        raise ValueError(f"not a tiangong3 checkpoint (robot {a_.get('robot', 'g1')})")
    dynamics = a_.get("dynamics", "default")
    set_dynamics(dynamics)
    networks.AMP_DTYPE = None
    K, H = a_["goal_frames"], a_["hist"]
    goal_dim = 73 * K
    pi = Policy(OBS_DIM, goal_dim, hist=H, soft_clip_actions=a_.get("act_soft_clip", False))
    pi.load_state_dict(ck["pi"])
    pi.eval()
    model = Deploy(pi).eval()

    run_dir = os.path.dirname(os.path.abspath(args.ckpt))
    out = args.out or os.path.join(
        run_dir, f"{os.path.basename(run_dir)}_{os.path.splitext(os.path.basename(args.ckpt))[0]}_policy.onnx")
    torch.onnx.export(model, (torch.zeros(1, H * OBS_DIM), torch.zeros(1, goal_dim)), out,
                      input_names=["obs", "goal"], output_names=["actions"], opset_version=17, dynamo=False)

    meta = {
        "policy": "supermimic_goal",
        "dynamics": dynamics,
        "run": os.path.basename(run_dir),
        "checkpoint": os.path.basename(args.ckpt),
        "iteration": str(ck.get("iteration", "")),
        "control_dt": str(CONTROL_DT),
        "history_length": str(H),
        "obs_frame_dim": str(OBS_DIM),
        "future_frames": str(K),
        "goal_dim": str(goal_dim),
        "joint_names": ",".join(JOINT_NAMES),
        "default_joint_pos": ",".join(f"{v:.6g}" for v in default_joint_pos_list()),
        "action_scale": ",".join(f"{action_scale(n):.6g}" for n in JOINT_NAMES),
        "joint_stiffness": ",".join(f"{motor_params(n)[2]:.6g}" for n in JOINT_NAMES),
        "joint_damping": ",".join(f"{motor_params(n)[3]:.6g}" for n in JOINT_NAMES),
    }
    m = onnx.load(out)
    for k, v in meta.items():
        m.metadata_props.add(key=k, value=v)
    onnx.save(m, out)

    # the exported graph must reproduce the PyTorch policy
    g = torch.Generator().manual_seed(0)
    o = pi.obs_norm.mean + torch.randn(8, H, OBS_DIM, generator=g) * pi.obs_norm.std()
    gl = pi.goal_norm.mean + torch.randn(8, goal_dim, generator=g) * pi.goal_norm.std()
    with torch.no_grad():
        ref = pi(o, gl).clamp(-ACTION_CLIP, ACTION_CLIP).numpy()
    check = os.path.splitext(out)[0] + "_check.npz"
    np.savez(check, obs=o.reshape(8, -1).numpy(), goal=gl.numpy(), actions=ref)
    print(f"wrote {out}  (obs {H}x{OBS_DIM}, goal {K}x73)")
    print(f"check data: {check} (compare with onnxruntime)")

    if args.motion:
        motion = MotionLib.from_dir(args.motion, G1Kinematics("cpu"), "cpu")
        if len(motion.names) != 1:
            raise ValueError(f"--motion must be one motion file, got {len(motion.names)}")
        f = torch.arange(int(motion.clip_len[0]))
        with torch.no_grad():
            goals = motion.goal(f, K).numpy()
        path = os.path.splitext(out)[0] + "_goals.npz"
        np.savez(path, motion=motion.names[0], goals=goals, ref_q=motion.q[f].numpy(), ref_root_R=motion.root_R[f].numpy())
        print(f"goal table of {motion.names[0]} ({len(f)} frames): {path}")


if __name__ == "__main__":
    main()

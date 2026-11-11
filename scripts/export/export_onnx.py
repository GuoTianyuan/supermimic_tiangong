"""Export a SuperMimic policy with one motion baked in to ONNX for deployment (xmigcs policy/smdance).

A policy trained on many motions tracks whichever goal it is given, so one checkpoint can be exported once per
motion. The motion's goal table is baked in, so the graph needs only the robot's observation history:

    inputs   obs        (1, hist * 99)  observation history, oldest first; each frame is
                                        [q - default_q (29), qd (29), base ang vel in body frame (3),
                                         projected gravity (3), previous action (29),
                                         first two columns of R_robot^T R_ref(f), row-major (6)]
             time_step  (1, 1)          reference frame f (checkpoint control rate), clamped to the motion
    outputs  actions        (1, 29)     joint target = default_joint_pos + action_scale * actions
             ref_joint_pos  (1, 29)     reference joint angles of frame f
             ref_root_quat  (1, 4)      reference pelvis orientation of frame f (w, x, y, z)

Joint order, default pose, action scale, PD gains and the motion length are stored as metadata.
Usage:
    python scripts/export/export_onnx.py --ckpt logs/<run>/ckpt_100000.pt --motion data/wbc_full/<motion>.npz [--out x.onnx]
"""

from __future__ import annotations

from pathlib import Path

import argparse
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--motion", required=True, help="one NPZ motion; it should be one the policy was trained on")
ap.add_argument("--out", default=None, help="default: <ckpt>_<motion>.onnx next to the checkpoint")
args = ap.parse_args()
os.environ["SUPERMIMIC_ROBOT"] = "tiangong3"

import numpy as np
import onnx
import torch
import torch.nn as nn

import supermimic.models.common as networks
from supermimic.robots.kinematics import G1Kinematics
from supermimic.tasks.tracking.mdp import ACTION_CLIP, OBS_DIM
from supermimic.motions.library import CONTROL_DT, MotionLib, parse_goal_drop
from supermimic.models import Policy
from supermimic.robots import JOINT_NAMES, action_scale, default_joint_pos_list
from supermimic.utils.rotations import mat_to_quat
from supermimic.robots.tiangong3 import motor_params, set_dynamics


class Deploy(nn.Module):
    def __init__(self, pi: Policy, goals: torch.Tensor, ref_q: torch.Tensor, ref_quat: torch.Tensor):
        super().__init__()
        self.pi = pi
        self.register_buffer("goals", goals)
        self.register_buffer("ref_q", ref_q)
        self.register_buffer("ref_quat", ref_quat)

    def forward(self, obs: torch.Tensor, time_step: torch.Tensor):
        f = time_step.reshape(-1).round().clamp(0, self.goals.shape[0] - 1).long()
        a = self.pi(obs.reshape(obs.shape[0], self.pi.hist, OBS_DIM), self.goals[f])
        return a.clamp(-ACTION_CLIP, ACTION_CLIP), self.ref_q[f], self.ref_quat[f]


def main():
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a_ = ck["args"]
    if a_.get("robot", "g1") != "tiangong3":
        raise ValueError(f"not a tiangong3 checkpoint (robot {a_.get('robot', 'g1')})")
    dynamics = a_.get("dynamics", "default")
    set_dynamics(dynamics)
    networks.AMP_DTYPE = None
    kin = G1Kinematics("cpu")
    motion = MotionLib.from_dir(args.motion, kin, "cpu")
    if len(motion.names) != 1:
        raise ValueError(f"--motion must be one motion file, got {len(motion.names)}")
    K, H = a_["goal_frames"], a_["hist"]

    T = int(motion.clip_len[0])
    f = torch.arange(T)
    with torch.no_grad():
        goals = motion.goal(f, K, parse_goal_drop(a_.get("goal_drop", "")))
    pi = Policy(OBS_DIM, goals.shape[1], hist=H, soft_clip_actions=a_.get("act_soft_clip", False))
    pi.load_state_dict(ck["pi"])
    pi.eval()
    model = Deploy(pi, goals, motion.q[f].clone(), mat_to_quat(motion.root_R[f]).clone()).eval()

    out = args.out or os.path.splitext(args.ckpt)[0] + "_" + os.path.splitext(os.path.basename(args.motion))[0] + ".onnx"
    obs = torch.zeros(1, H * OBS_DIM)
    ts = torch.zeros(1, 1)
    torch.onnx.export(model, (obs, ts), out, input_names=["obs", "time_step"],
                      output_names=["actions", "ref_joint_pos", "ref_root_quat"], opset_version=17, dynamo=False)

    meta = {
        "policy": "supermimic",
        "tracking": "paper",
        "dynamics": dynamics,
        "checkpoint": os.path.basename(args.ckpt),
        "iteration": str(ck.get("iteration", "")),
        "motion": motion.names[0],
        "motion_length": str(T),
        "control_dt": str(CONTROL_DT),
        "history_length": str(H),
        "obs_frame_dim": str(OBS_DIM),
        "future_frames": str(K),
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
    fr = torch.randint(0, T, (8,), generator=g)
    with torch.no_grad():
        ref = pi(o, goals[fr]).clamp(-ACTION_CLIP, ACTION_CLIP).numpy()
    np.savez(out.replace(".onnx", "_check.npz"), obs=o.reshape(8, -1).numpy(), time_step=fr.float().numpy()[:, None], actions=ref)
    print(f"wrote {out}  (motion {motion.names[0]}, {T} frames = {T * CONTROL_DT:.1f} s, obs {H}x{OBS_DIM}, goal {goals.shape[1]})")
    print(f"check data: {out.replace('.onnx', '_check.npz')} (compare with onnxruntime)")


if __name__ == "__main__":
    main()

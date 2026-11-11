"""Verify motion reordering, preserved NPZ velocities, and differentiable FK vs MuJoCo."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))

import mujoco
import numpy as np
import torch
from supermimic.kinematics import G1Kinematics
from supermimic.motion import BIN_FRAMES, CONTROL_DT, MotionLib
from supermimic.robot import JOINT_NAMES, MJCF_PATH, TRACKED_BODIES

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--motion_file',required=True)
a=ap.parse_args()
kin=G1Kinematics('cpu')
motion=MotionLib.from_dir(a.motion_file,kin,'cpu')
if a.motion_file.endswith('.npz'):
    raw=np.load(a.motion_file)
    meta=json.loads(Path(a.motion_file+'.metadata.json').read_text())
    order=[meta['joint_names'].index(n) for n in JOINT_NAMES]
    fps = float(raw['fps'].reshape(-1)[0])
    duration = (len(raw['joint_pos']) - 1) / fps
    assert motion.num_frames == int(np.floor(duration / CONTROL_DT)) + 1
    assert np.isclose(BIN_FRAMES * CONTROL_DT, 1.0)
    # Match the loader's float32 query times, including rounding in long clips.
    phase = np.arange(motion.num_frames, dtype=np.float32) * np.float32(CONTROL_DT) * np.float32(fps)
    def interpolated(values):
        return np.stack([np.interp(phase, np.arange(len(values)), values[:, j])
                         for j in range(values.shape[1])], axis=-1)
    root = meta['root_body_index']
    for actual, source, atol in (
        (motion.q, raw['joint_pos'][:, order], 5e-5),
        (motion.qd, raw['joint_vel'][:, order], 1e-3),
        (motion.root_p, raw['body_pos_w'][:, root], 5e-5),
        (motion.root_v, raw['body_lin_vel_w'][:, root], 5e-5),
        (motion.root_w, raw['body_ang_vel_w'][:, root], 5e-5),
    ):
        np.testing.assert_allclose(actual.numpy(), interpolated(source), atol=atol, rtol=1e-4)
model=mujoco.MjModel.from_xml_path(MJCF_PATH);data=mujoco.MjData(model)
ids=torch.linspace(0,motion.num_frames-1,19).long()
S=motion.ref_state(ids)
pred=kin.forward(*S)[0]
for k,frame in enumerate(ids):
    data.qpos[:3]=motion.root_p[frame]
    mujoco.mju_mat2Quat(data.qpos[3:7],motion.root_R[frame].numpy().astype(float).ravel())
    for j,name in enumerate(JOINT_NAMES):data.qpos[model.jnt_qposadr[model.joint(name).id]]=motion.q[frame,j]
    mujoco.mj_forward(model,data)
    target=np.stack([data.xpos[model.body(name).id] for name in TRACKED_BODIES])
    np.testing.assert_allclose(pred[k].numpy(),target,atol=2e-6,rtol=1e-5)
q=S[4].clone().requires_grad_()
bp=kin.forward(*S[:4],q,S[5])[0]
bp.square().sum().backward()
assert torch.isfinite(q.grad).all() and q.grad.abs().max()>0
print('PASS: motion layout/velocity preservation, MuJoCo FK, and finite FK gradients')

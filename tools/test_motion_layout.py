"""Verify motion reordering, preserved NPZ velocities, and differentiable FK vs MuJoCo."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib
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
    assert float(raw['fps'].reshape(-1)[0])==50
    assert motion.num_frames==len(raw['joint_pos'])
    torch.testing.assert_close(motion.q,torch.from_numpy(raw['joint_pos'][:,order].copy()),atol=5e-5,rtol=1e-5)
    torch.testing.assert_close(motion.qd,torch.from_numpy(raw['joint_vel'][:,order].copy()),atol=1e-3,rtol=1e-4)
    torch.testing.assert_close(motion.root_v,torch.from_numpy(raw['body_lin_vel_w'][:,0].copy()),atol=5e-5,rtol=1e-5)
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

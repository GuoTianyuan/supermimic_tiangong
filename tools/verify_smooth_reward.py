"""Check the analytic reward change and its physical-time scaling without running a simulator."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["SUPERMIMIC_ROBOT"] = "tiangong3"

import numpy as np
import torch

from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MAX_STEP_REWARD, MDP, reward_cfg
from supermimic.motion import CONTROL_DT
from supermimic.robot import default_joint_pos_list, set_dynamics

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
set_dynamics("dexevt")
torch.manual_seed(0)
kin = G1Kinematics("cpu")
q0 = torch.tensor(default_joint_pos_list())
n = 8
p = torch.tensor([0., 0., .9]).expand(n, 3).clone()
R = torch.eye(3).expand(n, 3, 3).clone()
v = torch.zeros(n, 3)
w = torch.zeros(n, 3)
q = q0.expand(n, 29).clone()
qd = (.2 * torch.randn(n, 29)).requires_grad_()
S = p, R, v, w, q, qd
bp, bR, bv, bw = kin.forward(*S)
motion = SimpleNamespace(root_p=p, root_R=R, q=q, body_p=bp, body_R=bR, body_v=bv, body_w=bw)
f = torch.arange(n)
cfg = reward_cfg("dexevt")
assert cfg.w_action_rate == -.4 and cfg.w_joint_acc == -3e-6
old_cfg = replace(cfg, w_action_rate=-.1, w_joint_acc=-2e-6)
new = MDP(kin, motion, q0, cfg)
old = MDP(kin, motion, q0, old_cfg)
a = (.1 * torch.randn(n, 29)).requires_grad_()
prev = a.detach() + .025 * torch.randn(n, 29)
qd_prev = qd.detach() + .02 * torch.randn(n, 29)
r_new, t_new = new.reward(S, f, a, prev, terms=True, qd_prev=qd_prev)
r_old, t_old = old.reward(S, f, a, prev, terms=True, qd_prev=qd_prev)
expected = (-.3 * (a-prev).square().sum(-1) - 1e-6 * ((qd-qd_prev)/CONTROL_DT).square().sum(-1)) * CONTROL_DT
torch.testing.assert_close(r_new-r_old, expected, rtol=1e-4, atol=1e-8)
rate_only_diff = new.reward(S, f, a, prev) - old.reward(S, f, a, prev)
acc_only_diff = new.reward(S, f, a, a, qd_prev=qd_prev) - old.reward(S, f, a, a, qd_prev=qd_prev)
torch.testing.assert_close(rate_only_diff, -.3 * (a-prev).square().sum(-1) * CONTROL_DT, rtol=1e-4, atol=1e-8)
torch.testing.assert_close(acc_only_diff, -1e-6 * ((qd-qd_prev)/CONTROL_DT).square().sum(-1) * CONTROL_DT, rtol=1e-4, atol=1e-8)
torch.testing.assert_close(t_new["track"], t_old["track"])
torch.testing.assert_close(t_new["joint_limit"], t_old["joint_limit"])
grads = torch.autograd.grad(r_new.sum(), (a, qd))
assert all(torch.isfinite(x).all() and x.abs().max() > 0 for x in grads)
perfect = new.reward(S, f, a, a, qd_prev=qd)
torch.testing.assert_close(perfect, torch.full_like(perfect, MAX_STEP_REWARD))

def cost(dt, weight, derivative=False):
    t = np.arange(round(2/dt)+1)*dt
    signal = np.sin(2*math.pi*t)
    diff = np.diff(signal)/(dt if derivative else 1.)
    return -weight * dt * np.sum(diff**2)

action_ratio = cost(.01, -.4)/cost(.02, -.1)
acc_ratio = cost(.01, -2e-6, True)/cost(.02, -2e-6, True)
assert abs(action_ratio-1) < .002 and abs(acc_ratio-1) < .002
gamma_old, gamma_new = .99, .995
report = {
    "passed": True, "reward_cfg": vars(cfg), "control_dt": CONTROL_DT,
    "reward_difference_max_abs_error": (r_new-r_old-expected).abs().max().item(),
    "perfect_reference_reward": perfect[0].item(),
    "finite_nonzero_action_and_velocity_gradients": True,
    "same_continuous_action_cost_ratio_100_to_50": action_ratio,
    "same_continuous_acceleration_cost_ratio_100_to_50": acc_ratio,
    "one_second_discount_50hz": gamma_old**50,
    "one_second_discount_100hz": gamma_new**100,
    "discount_time_constant_seconds_50hz": -.02/math.log(gamma_old),
    "discount_time_constant_seconds_100hz": -.01/math.log(gamma_new),
    "max_return_100hz": MAX_STEP_REWARD/(1-gamma_new),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))

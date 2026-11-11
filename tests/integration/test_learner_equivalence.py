"""Compare extracted learner factories with the actual pre-split runner on fixed replay.

No physical simulation is run: a reference clip provides deterministic transitions.
Compile is disabled for both paths so code generation/RNG cache effects are excluded.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("SUPERMIMIC_ROBOT", "tiangong3")

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from supermimic.configs.train import build_parser
from supermimic.learners.policy_learner import build_policy_learner
from supermimic.learners.wm_learner import build_world_model_learner
from supermimic.models import Critic, Policy, WorldModel
from supermimic.motions.library import MotionLib, goal_frame_dim
from supermimic.robots import G1_JOINT_NAMES, default_joint_pos_list
from supermimic.robots.kinematics import G1Kinematics
from supermimic.storage.replay_buffer import ReplayBuffer
from supermimic.tasks.tracking.mdp import (
    CRITIC_EXTRA, INV_DIM, MAX_STEP_REWARD, MDP, OBS_A_PREV, OBS_DIM,
    flatten_state, normalize_xi, unflatten_state,
)
from supermimic.utils.metrics import CriticMetrics, WorldModelMetrics, wm_error_summary


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b, (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    dev = cli.device
    args = build_parser().parse_args([])
    args.hist, args.goal_frames, args.horizon, args.wm_window = 3, 2, 4, 3
    args.wm_batch = args.pi_batch = 4
    args.critic_epochs, args.critic_minibatches = 2, 2
    kin = G1Kinematics(dev)
    motion = MotionLib.from_dir(str(ROOT / "data/tiangong_motion/motion1.npz"), kin, dev)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    buffer = ReplayBuffer(16, 4, dev, xi_dim=181)
    for t in range(16):
        row = {key: torch.zeros_like(value[0]) for key, value in buffer.data.items()}
        f = torch.arange(4, device=dev) + 100 + t
        row.update(f=f, s=flatten_state(motion.ref_state(f)), s1=flatten_state(motion.ref_state(f+1)))
        row["a"] = torch.randn(4, 29, device=dev) * .1
        row["a_prev"] = row["a"] * .9
        row["xi"][:, 0] = 1
        row["ep_step"].fill_(t)
        buffer.add(row)
    wm = WorldModel(hidden=(32, 32), xi_dim=181, a_prev=True).to(dev)
    pi = Policy(OBS_DIM, goal_frame_dim(()) * args.goal_frames, hist=args.hist, soft_clip_actions=True).to(dev)
    critic = Critic(INV_DIM + CRITIC_EXTRA + 181 + goal_frame_dim(()) * args.goal_frames, layer_norm=True).to(dev)
    target = copy.deepcopy(critic).requires_grad_(False)
    target.norm = critic.norm
    initial = (wm, pi, critic, target)
    # Deepcopy preserves the intentional shared normalizer inside each replica.
    def context():
        w, p, c, ct = copy.deepcopy(initial)
        return dict(
            args=args, wm=w, pi=p, critic=c, critic_targ=ct,
            opt_wm=torch.optim.Adam(w.parameters(), lr=args.lr),
            opt_pi=torch.optim.Adam(p.parameters(), lr=args.lr, betas=(.9, .95)),
            opt_v=torch.optim.Adam(c.parameters(), lr=args.lr),
            buffer=buffer, mdp=mdp, motion=motion, dev=dev,
            wm_metrics=WorldModelMetrics(dev, G1_JOINT_NAMES), critic_metrics=CriticMetrics(dev),
            GOAL_DROP=(), G_MAX=MAX_STEP_REWARD/(1-args.gamma),
        )
    before, after = context(), context()
    fixture = json.loads((ROOT / "tests/fixtures/package_layout_before.json").read_text())
    tree = ast.parse(fixture["runner_source"])
    function_names = {
        "sync_grads", "clip_finite_grads", "set_requires_grad", "_wm_rollout", "_wm_err",
        "_imag_step", "_reward_term", "_critic_input", "reward_term", "critic_input_c",
        "wm_update", "sample_valid_starts", "policy_update", "a_prev_gain",
    }
    assignment_names = {"ERR_FLOOR", "wm_rollout", "wm_err", "imag_step", "reward_term_c", "critic_input_gpu"}
    nodes = [node for node in tree.body if
             (isinstance(node, ast.FunctionDef) and node.name in function_names) or
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in assignment_names for t in node.targets))]
    before.update(torch=torch, nn=nn, F=F, dist=dist, time=time, WORLD=1, PROFILE=False,
                  K=args.goal_frames, Hh=args.hist, HZ=args.horizon, OBS_A_PREV=OBS_A_PREV,
                  MAX_STEP_REWARD=MAX_STEP_REWARD, normalize_xi=normalize_xi,
                  unflatten_state=unflatten_state, wm_error_summary=wm_error_summary)
    original_compile = torch.compile
    try:
        torch.compile = lambda fn, **kwargs: fn
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<pre-split-runner>", "exec"), before)
        wm_new = build_world_model_learner(**{k: after[k] for k in
            ("args", "wm", "opt_wm", "buffer", "wm_metrics", "dev")})
        pi_new = build_policy_learner(**{k: after[k] for k in
            ("args", "wm", "pi", "critic", "critic_targ", "opt_pi", "opt_v", "buffer", "mdp",
             "motion", "critic_metrics", "dev", "GOAL_DROP", "G_MAX")})
    finally:
        torch.compile = original_compile
    for step in range(3):
        for old_update, new_update in ((before["wm_update"], wm_new.update), (before["policy_update"], pi_new.update)):
            torch.manual_seed(1000 + step)
            old_result = old_update()
            torch.manual_seed(1000 + step)
            new_result = new_update()
            assert_equal(old_result, new_result)
        for key in ("wm", "pi", "critic", "critic_targ", "opt_wm", "opt_pi", "opt_v"):
            assert_equal(before[key].state_dict(), after[key].state_dict())
        for key in ("wm", "pi", "critic"):
            for old, new in zip(before[key].parameters(), after[key].parameters()):
                assert_equal(old.grad, new.grad)
        for key in ("wm_metrics", "critic_metrics"):
            assert_equal(before[key].flush(), after[key].flush())
    report = dict(device=dev, updates=3, fixed_replay=True, compile=False,
                  loss_gradients_parameters_normalizers_optimizer_and_ema_bitwise_equal=True)
    cli.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Exercise 12-step differentiable WM rollout and replay boundaries before the long run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
os.environ["SUPERMIMIC_ROBOT"] = "tiangong3"
import torch

from supermimic.storage.replay_buffer import ReplayBuffer
from supermimic.envs.domain_randomization import DRCfg, parse_terms
from supermimic.tasks.tracking.mdp import flatten_state
from supermimic.models import WorldModel
from supermimic.robots import default_joint_pos_list, set_dynamics

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(1)
torch.manual_seed(0)
set_dynamics("dexevt")
xi_dim = DRCfg.for_dynamics(parse_terms("all,contact"), "dexevt").xi_dim
buf = ReplayBuffer(19, 2, "cpu", xi_dim=xi_dim)
for t in range(60):
    tr = {k: torch.zeros_like(v[0]) for k, v in buf.data.items()}
    tr["s"][:, 0] = t
    tr["s1"][:, 0] = t+1
    tr["xi"][:, 0] = torch.arange(2)
    tr["done"] = (t+torch.arange(2)) % 7 == 6
    tr["cut"] = (t+torch.arange(2)) % 11 == 10
    buf.add(tr)
w = buf.sample_windows(1024, 12)
torch.testing.assert_close(w["s1"][..., 0], w["s"][..., 0]+1)
torch.testing.assert_close(torch.diff(w["s"][..., 0], dim=1), torch.ones(1024, 11))
assert w["s"][..., 0].min() >= 41 and w["s"][..., 0].max() <= 59
assert (w["xi"][:, :, 0] == w["xi"][:, :1, 0]).all()
for i in range(len(w["mask"])):
    valid = True
    for j in range(12):
        assert bool(w["mask"][i, j]) == valid
        if bool(w["done"][i, j] | w["cut"][i, j]): valid = False
assert (w["mask"][:, 0] == 1).all() and (w["mask"][:, -1] == 0).any()

B, K = 8, 12
S0 = (torch.tensor([0., 0., .9]).expand(B, 3).clone(), torch.eye(3).expand(B, 3, 3).clone(),
      torch.zeros(B, 3), torch.zeros(B, 3), torch.tensor(default_joint_pos_list()).expand(B, 29).clone(),
      torch.zeros(B, 29))
wm = WorldModel(xi_dim=xi_dim, a_prev=True)
S = S0
preds = []
actions = .01*torch.randn(B, K, 29)
for j in range(K):
    prev = actions[:, j-1] if j else torch.zeros(B, 29)
    S = wm.step(S, actions[:, j], torch.zeros(B, xi_dim), prev)
    preds.append(S)
S_hat = tuple(torch.stack([x[i] for x in preds], 1) for i in range(6))
target = [x[:, None].expand(-1, K, *x.shape[1:]).clone() for x in S0]
# The initialized WM is the identity map; use a moving target to exercise a nonzero loss/gradient.
steps = torch.arange(1, K+1).float()[None, :]
target[0][:, :, 0] += .001 * steps
target[2][:, :, 0] = .1
target[4] += .0002 * steps.unsqueeze(-1)
target[5] += .02
target = tuple(target)
e = wm.state_error(S_hat, target).square().mean(-1)
loss = e.sum()/B
assert e.shape == (B, K) and torch.isfinite(e).all()
loss.backward()
grads = [p.grad for p in wm.parameters() if p.grad is not None]
assert grads and all(torch.isfinite(g).all() for g in grads)
assert any(g.abs().max() > 0 for g in grads)
report = {"passed": True, "wm_window": K, "actor_horizon": 48, "xi_dim": xi_dim,
          "replay_wraparound_episode_and_push_boundaries": True,
          "wm_recursive_prediction_shape": list(e.shape), "wm_loss": loss.item(),
          "finite_nonzero_wm_gradients": True,
          "loss_reduction": "time-sum, batch-mean (unchanged)",
          "scope": "CPU numerical check; actual full-batch GPU training is checked after launch"}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))

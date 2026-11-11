"""World-model rollout, normalized supervised loss and optimizer update."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from functools import partial
import torch
from supermimic.tasks.tracking.mdp import unflatten_state, normalize_xi
from supermimic.utils.metrics import wm_error_summary
from .learner_utils import sync_grads as _sync_grads, clip_finite_grads



@dataclass
class WorldModelLearner:
    update: Callable
    rollout: Callable
    error: Callable


def build_world_model_learner(*, args, wm, opt_wm, buffer, wm_metrics, dev, world=1):
    sync_grads = partial(_sync_grads, world=world)

    ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                           torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(dev)  # fmt: skip

    def _wm_rollout(s, a, a_prev, xi):
        S = unflatten_state(s[:, 0])
        preds = []
        for j in range(args.wm_window):
            S = wm.step(S, a[:, j], normalize_xi(xi[:, j]), a_prev[:, j])
            preds.append(S)
        return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)

    def _wm_err(S_hat, s1, mask):
        std = torch.maximum(wm.error_std(), ERR_FLOOR)
        raw = wm.state_error(S_hat, unflatten_state(s1))
        squared = (raw / std) ** 2  # (B, k, D)
        summary = wm_error_summary(raw, squared, mask, std, ERR_FLOOR)
        e = squared.mean(-1)
        return (e * mask).sum() / s1.shape[0], e[:, -1], e[:, 0], summary

    wm_rollout = torch.compile(_wm_rollout)
    wm_err = torch.compile(_wm_err)

    def wm_update():
        w = buffer.sample_windows(args.wm_batch, args.wm_window)
        S_hat = wm_rollout(w["s"], w["a"], w["a_prev"], w["xi"])
        loss, e_last, e_first, wm_summary = wm_err(S_hat, w["s1"], w["mask"])
        wm_metrics.update(wm_summary)
        opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        sync_grads(wm)
        clip_finite_grads(wm, 1.0)
        opt_wm.step()
        m = w["mask"][:, -1]
        return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1), e_first.detach().mean()

    return WorldModelLearner(wm_update, wm_rollout, wm_err)

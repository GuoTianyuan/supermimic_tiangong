"""Differentiable imagined rollouts, Actor update and action-loop diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from functools import partial
import time
import torch
import torch.nn as nn
from supermimic.tasks.tracking.mdp import unflatten_state, normalize_xi, OBS_A_PREV
from .learner_utils import sync_grads as _sync_grads, set_requires_grad
from .critic_learner import build_critic_learner


@dataclass
class PolicyLearner:
    update: Callable
    action_loop_gain: Callable


def build_policy_learner(*, args, wm, pi, critic, critic_targ, opt_pi, opt_v, buffer, mdp, motion,
                         critic_metrics, dev, GOAL_DROP, G_MAX, world=1, PROFILE=False):
    sync_grads = partial(_sync_grads, world=world)
    K, Hh, HZ = args.goal_frames, args.hist, args.horizon

    def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g):
        a = pi(o_hist, g)
        S1 = wm.step(S, a, xi_n, a_prev)
        f1 = f + 1
        o1 = mdp.add_noise(mdp.obs(S1, a.detach() if args.detach_a_prev else a, xi, f1))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return S1, a, o_hist, f1, motion.goal(f1, K, GOAL_DROP)

    def _reward_term(S, f, a, a_prev, qd_prev, ref_bodies):
        return mdp.reward(S, f, a, a_prev, qd_prev=qd_prev, ref_bodies=ref_bodies), mdp.soft_termination(S, f)

    def _critic_input(S, f, xi_n, g, ref_body_p=None):
        return torch.cat((mdp.critic_features(S, f, xi_n, ref_body_p=ref_body_p), g), -1)

    imag_step = torch.compile(_imag_step)
    reward_term_c = torch.compile(_reward_term)
    critic_input_gpu = torch.compile(_critic_input)

    def reward_term(S, f, a, a_prev, qd_prev):
        # One batched transfer for the entire H x B imagined trajectory, outside torch.compile.
        reference = motion.body_reference(f)
        return reward_term_c(S, f, a, a_prev, qd_prev, reference)

    def critic_input_c(S, f, xi_n, g):
        # Critic errors only use body positions, so do not transfer rotations or velocities.
        reference_p = motion.body_reference(f, positions_only=True)[0]
        return critic_input_gpu(S, f, xi_n, g, reference_p)

    def sample_valid_starts(B: int):
        return buffer.sample_valid_starts(B, Hh, HZ, motion.clip_id, motion.clip_end)

    critic_update = build_critic_learner(args=args, critic=critic, critic_targ=critic_targ, opt_v=opt_v,
        critic_input_c=critic_input_c, critic_metrics=critic_metrics, dev=dev, G_MAX=G_MAX, world=world)

    def policy_update():
        if PROFILE:
            torch.cuda.synchronize(); tq0 = time.time()
        s, h = sample_valid_starts(args.pi_batch)
        S = unflatten_state(s["s"])
        xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
        xi_n = normalize_xi(xi)
        with torch.no_grad():
            o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
        o_hist = mdp.add_history_noise(o_hist, h["noise_source"])
        set_requires_grad(wm, False)
        set_requires_grad(critic, False)
        B = args.pi_batch
        g0 = motion.goal(f, K, GOAL_DROP)
        g = g0
        S0 = S
        Ss, As, Aps, Fs, Gs = [], [], [], [], []
        for t in range(HZ):
            Aps.append(a_prev)
            S, a_prev, o_hist, f, g = imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
            Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        f_all = flat(Fs)
        qd_prev = flat([S0[5]] + [St[5] for St in Ss[:-1]])
        r, d = reward_term(S_all, f_all, flat(As), flat(Aps), qd_prev)
        r, d = r.view(HZ, B), d.view(HZ, B)
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        c = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        v_end = torch.zeros(B, device=dev)
        if not args.no_critic:
            # the clamp at the largest possible return stops the policy from chasing critic overestimates
            v_end = critic(_critic_input(S, f, xi_n, g))
            J = J + (args.gamma**HZ) * c * v_end.clamp(-G_MAX, G_MAX)
            v_end = v_end.detach()
        loss = -J.mean()
        if PROFILE:
            torch.cuda.synchronize(); tq1 = time.time()
        opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        sync_grads(pi)
        if PROFILE:
            torch.cuda.synchronize(); tq2 = time.time()
        gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(pi.parameters(), 0.1)
        if torch.isfinite(gn):
            opt_pi.step()
        set_requires_grad(wm, True)
        set_requires_grad(critic, True)
        # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
        v_mse = critic_update(S0, Ss, s, Fs, Gs, g0, xi_n, rews, dones, c_t)
        if PROFILE:
            torch.cuda.synchronize()
            print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
        v_stats = (v_end.mean(), v_end.max(), (v_end > G_MAX).float().mean())
        return loss.detach(), v_mse, (r_sum / HZ / mdp.max_step_reward).mean(), gn.detach(), c.mean().detach(), *v_stats

    def a_prev_gain(n: int = 256) -> torch.Tensor:
        """Largest singular value of da_t / da_{t-1} through the newest observation, at n buffer states:
        above 1 the imagined action loop (and its BPTT gradient) can grow from step to step."""
        s, h = buffer.sample_starts(n, Hh)
        with torch.no_grad():
            o = mdp.add_history_noise(
                mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"]), h["noise_source"])
        x = o[:, -1, OBS_A_PREV : OBS_A_PREV + 29].clone().requires_grad_(True)
        newest = torch.cat((o[:, -1, :OBS_A_PREV], x, o[:, -1, OBS_A_PREV + 29 :]), -1)
        a = pi(torch.cat((o[:, :-1], newest.unsqueeze(1)), 1), motion.goal(s["f"], K, GOAL_DROP))
        jac = torch.stack([torch.autograd.grad(a[:, j].sum(), x, retain_graph=j < 28)[0] for j in range(29)], 1)
        return torch.linalg.matrix_norm(jac, ord=2)

    return PolicyLearner(policy_update, a_prev_gain)

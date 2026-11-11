"""Survival-weighted Huber fit, lambda returns and EMA target update."""

from __future__ import annotations

from functools import partial
import torch
import torch.nn.functional as F
from .learner_utils import sync_grads as _sync_grads, clip_finite_grads


def build_critic_learner(*, args, critic, critic_targ, opt_v, critic_input_c, critic_metrics, dev, G_MAX, world=1):
    sync_grads = partial(_sync_grads, world=world)
    HZ, B = args.horizon, args.pi_batch

    def critic_update(S0, Ss, s, Fs, Gs, g0, xi_n, rews, dones, c_t):
        v_mse = torch.zeros((), device=dev)
        if not args.no_critic:
            with torch.no_grad():
                S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
                f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
                g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
                xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
                X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
            critic.norm.update(X)
            with torch.no_grad():
                V = critic_targ(X).clamp(-G_MAX, G_MAX)
                R = V[-1]
                targets = []
                for t in range(HZ - 1, -1, -1):
                    R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                    targets.append(R)
                targets = torch.stack(targets[::-1]).clamp(-G_MAX, G_MAX)  # (H, B)
            # weighted by the probability of being alive: states after a soft termination never bootstrap anything
            Xf, Yf, Wf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1), c_t.detach().reshape(-1)
            n = Xf.shape[0]
            mb = n // args.critic_minibatches
            for _ in range(args.critic_epochs):
                perm = torch.randperm(n, device=dev)
                for i in range(args.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    v, y, w = critic(Xf[idx]), Yf[idx], Wf[idx]
                    w_sum = w.sum().clamp(min=1e-6)
                    huber = F.huber_loss(v, y, reduction="none", delta=1.0)
                    v_loss = (w * huber).sum() / w_sum
                    critic_metrics.update(v, y, w, huber)
                    opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    sync_grads(critic)
                    clip_finite_grads(critic, 1.0)
                    opt_v.step()
            v_mse = (w * (v.detach() - y) ** 2).sum() / w_sum
            with torch.no_grad():
                for p_targ, p in zip(critic_targ.net.parameters(), critic.net.parameters()):
                    p_targ.lerp_(p, args.critic_tau)

        return v_mse

    return critic_update

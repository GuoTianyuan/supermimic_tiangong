"""Real environment sampling and replay insertion; no optimizer or file I/O."""
import torch
from supermimic.tasks.tracking.mdp import ACTION_CLIP, MAX_STEP_REWARD, OBS_DIM, normalize_xi, unflatten_state
from supermimic.logging.legacy import SMOOTH_KEYS


class RolloutCollector:
    def __init__(self, components, env, buffer, real_metrics, episode_metrics):
        self.components, self.env, self.buffer = components, env, buffer
        self.real_metrics, self.episode_metrics = real_metrics, episode_metrics
        self.sim_steps = components.sim_steps
        self.obs_hist = torch.zeros(components.args.num_envs, components.args.hist, OBS_DIM, device=components.dev)

    @torch.no_grad()
    def collect(self, n_steps: int, stats: dict):
        c = self.components
        args, env, mdp, motion = c.args, self.env, c.mdp, c.motion
        pi, wm, buffer = c.pi, c.wm, self.buffer
        K, GOAL_DROP, WORLD = args.goal_frames, c.goal_drop, c.world
        for _ in range(n_steps):
            S = env.state()
            o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
            fresh = env.ep_step == 0
            self.obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(self.obs_hist), torch.cat((self.obs_hist[:, 1:], o.unsqueeze(1)), 1))
            g = motion.goal(env.f, K, GOAL_DROP)
            pi.obs_norm.update(o)
            pi.goal_norm.update(g)
            a = pi(self.obs_hist, g)
            a = torch.clamp(a + args.action_noise * torch.randn_like(a), -ACTION_CLIP, ACTION_CLIP)
            out = env.step(a)
            buffer.add(out)
            self.real_metrics.update(out)
            self.episode_metrics.update(out)
            S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
            wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"]), out["a_prev"]))
            wm.update_output_normalization(S0, S1)
            self.sim_steps += args.num_envs * WORLD
            stats["r"].append(out["r"].mean() / MAX_STEP_REWARD)
            for k in SMOOTH_KEYS:
                stats[k].append(out["terms"][k].mean())
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].sum())
                stats["ep_ret"].append(out["ep_ret_norm"].sum())
                stats["ep_count"].append(out["ep_len"].numel())
                stats["end_clip"].append(out["end_clip"].sum())
                stats["end_time"].append(out["end_time"].sum())
                stats["term"].append(out["term"].float().sum())
                stats["done"].append(out["done"].float().sum())

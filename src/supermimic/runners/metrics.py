"""Reduce each logging window once, with the original DDP and sampling order."""
import torch
import torch.distributed as dist
from supermimic.motions.pool import MotionPool
from supermimic.utils.metrics import CRITIC_FIELDS, EXTRA_FIELDS, POOL_FIELDS, WM_FIELDS
from supermimic.logging.legacy import CSV_FIELDS, SMOOTH_KEYS


class TrainingMetrics:
    def __init__(self):
        self.nonfinite_logged = 0

    def reduce(self, *, components, env, real_metrics, critic_metrics, wm_metrics,
               policy_learner, stats, agg, it, done_it, now, t_start, sim_steps, ips, lr, dr_scale):
        args, WORLD, dev, motion = components.args, components.world, components.dev, components.motion
        mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
        total = lambda x: float(sum(float(v) for v in x))  # noqa: E731
        top = lambda x: max((float(v) for v in x), default=0.0)  # noqa: E731
        # losses and per-step rewards are averaged over ranks, maxima taken over ranks; episode statistics are pooled
        vals = [mean(agg[k]) for k in ("wm", "wm8", "pi", "v", "ir", "gn", "surv")] + [mean(stats["r"])]
        gain = policy_learner.action_loop_gain() if it >= args.wm_only_iters else torch.zeros(1, device=dev)
        vals += [mean(agg["vm"]), mean(agg["vo"]), float(gain.median())] + [mean(stats[k]) for k in SMOOTH_KEYS]
        maxes = [top(agg["vx"]), top(agg["gn"]), float(gain.max())]
        pooled = [total(stats["ep_len"]), total(stats["ep_ret"]), total(stats["ep_count"]), total(stats["term"]), total(stats["done"]),
                  float(env.n_nonfinite - self.nonfinite_logged), total(stats["end_clip"]), total(stats["end_time"])]  # fmt: skip
        self.nonfinite_logged = env.n_nonfinite
        if WORLD > 1:
            t = torch.tensor(vals + pooled, dtype=torch.float64, device=dev)
            dist.all_reduce(t)
            vals, pooled = [x / WORLD for x in t[: len(vals)].tolist()], t[len(vals) :].tolist()
            t = torch.tensor(maxes, dtype=torch.float64, device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            maxes = t.tolist()
        ep_len_sum, ep_ret_sum, n_ep, n_term, n_done, n_nonfinite, n_clip, n_time = pooled
        (v_end_mean, v_end_over, gain_med), (v_end_max, gn_max, gain_max) = vals[8:11], maxes
        dr_mean = sum(dr_scale.values()) / max(len(dr_scale), 1)
        row = [done_it, (now - t_start) / 3600, sim_steps, *vals[:8], ep_len_sum / max(n_ep, 1), ep_ret_sum / max(n_ep, 1),
               n_term / max(n_done, 1), ips, lr, v_end_mean, v_end_max, v_end_over, gn_max, gain_med, gain_max, dr_mean,
               int(n_nonfinite), -vals[11] / max(vals[12], 1e-9), vals[13]]  # fmt: skip
        extra = real_metrics.flush()
        pool_metrics = motion.metrics() if isinstance(motion, MotionPool) else {k: 0.0 for k in POOL_FIELDS}
        if WORLD > 1 and isinstance(motion, MotionPool):
            pm = torch.tensor([pool_metrics[k] for k in POOL_FIELDS], device=dev, dtype=torch.float64)
            dist.all_reduce(pm)
            pool_metrics = dict(zip(POOL_FIELDS, pm.tolist()))
            # Memory/counts are global sums; refresh latency is mean per rank.
            pool_metrics["pool_refresh_s"] /= WORLD
            pool_metrics["pool_refreshes"] /= WORLD
        extra.update(pool_metrics)
        extra.update(wm_nse1=mean(agg["wm1"]), success_frac=1-n_term/n_done if n_done else 0.0,
                     episodes_ended=n_done, episodes_failed=n_term, ended_clip=n_clip, ended_time=n_time,
                     imag_reward_normalized=vals[4])
        if WORLD > 1:
            e1 = torch.tensor(extra["wm_nse1"], device=dev)
            dist.all_reduce(e1, op=dist.ReduceOp.AVG)
            extra["wm_nse1"] = float(e1)
        extra.update(critic_metrics.flush())
        wm_group_metrics, wm_channel_metrics = wm_metrics.flush()
        extra.update(wm_group_metrics)
        row += [extra[k] for k in (*EXTRA_FIELDS, *CRITIC_FIELDS, *WM_FIELDS)]
        return dict(zip(CSV_FIELDS, row)), wm_channel_metrics

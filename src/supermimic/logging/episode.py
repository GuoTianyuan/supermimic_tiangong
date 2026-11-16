"""Detached episode statistics, without modifying simulator or reset sampling."""
import torch
import torch.distributed as dist
from .schema import REWARD_NAMES, EPISODE_FIELDS


class EpisodeMetrics:
    def __init__(self, num_envs, max_episode_seconds, device):
        if max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds must be positive")
        self.max_seconds = max_episode_seconds
        self.keys = tuple(REWARD_NAMES)
        # Terms followed by actual (possibly sanitized) total reward.
        self.running = torch.zeros(num_envs, len(self.keys) + 1, dtype=torch.float64, device=device)
        self.completed = torch.zeros(len(EPISODE_FIELDS), dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, out):
        self.running.add_(torch.stack([out["terms"][k] for k in self.keys] + [out["r"]], -1))
        done = out["done"].bool()
        self.completed[:len(self.keys) + 1].add_((self.running * done[:, None]).sum(0))
        # ep_step is the pre-transition age; termination includes this final step.
        self.completed[-2].add_(((out["ep_step"] + 1) * done).sum())
        self.completed[-1].add_(done.sum())
        self.running.masked_fill_(done[:, None], 0.)

    def flush(self, control_dt=.01):
        values = self.completed.clone()
        if dist.is_initialized():
            dist.all_reduce(values)
        self.completed.zero_()
        values = values.cpu().tolist()
        count = values[-1]
        if not count:
            return {}  # No finished episode is missing data, not zero reward.
        normalized = [v / count / self.max_seconds for v in values[:len(self.keys)]]
        normalized += [values[-3] / count, values[-2] / count * control_dt, count]
        return dict(zip(EPISODE_FIELDS, normalized))

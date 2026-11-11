"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch

from .mdp import STATE_DIM, XI_DIM

FIELDS = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (29, torch.float32),
    "a_prev": (29, torch.float32),
    "xi": (XI_DIM, torch.float32),
    "r": ((), torch.float32),
    "f": ((), torch.long),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
    "cut": ((), torch.bool),  # a push changed the state after s1, before the next transition
}


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str, xi_dim: int = XI_DIM):
        self.T, self.N, self.device = T, N, device
        self.fields = {**FIELDS, "xi": (xi_dim, torch.float32)}
        self.data = {}
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def reference_frames(self) -> torch.Tensor:
        """A writable view of every valid reference index, including circular-buffer history.

        Before the first wrap rows [0,size) are valid; afterwards every row is valid.
        Used only between updates when the motion pool remaps resident offsets.
        """
        return self.data["f"][:self.size]

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "a_prev", "xi", "done", "cut")}
        # transition j is valid if no episode ended and no push moved the state at transitions 0..j-1
        d = (out["done"] | out["cut"]).float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int):
        """Start points for imagination plus the indices of their observation history (oldest first)."""
        age = torch.randint(0, self.size, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        return self._starts_at(age, env, hist)

    def _starts_at(self, age: torch.Tensor, env: torch.Tensor, hist: int):
        """Read starts/history at valid logical time/env indices (shared by both samplers)."""
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "f")}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f")}
        # A padded slot refers to the oldest retained real observation. Reuse that
        # observation's noise, just as collect() repeats one noisy frame at reset.
        hist_d["noise_source"] = hist - 1 - back
        return start, hist_d

    def sample_valid_starts(self, B: int, hist: int, horizon: int,
                            clip_id: torch.Tensor, clip_end: torch.Tensor,
                            max_attempts: int = 8):
        """Uniform replay starts conditioned on enough remaining reference frames.

        Keep the original rejection sampler's fast path, but bound its attempts.
        On rare low-acceptance batches enumerate eligible replay cells in chunks
        and fill the remainder uniformly with replacement. No replay data or
        reference is changed; an entirely ineligible replay raises explicitly.
        """
        if B < 1 or hist < 1 or horizon < 1 or max_attempts < 0:
            raise ValueError("Invalid imagination sampling size/history/horizon/attempts")
        if self.size == 0:
            raise RuntimeError("Cannot sample imagination starts from an empty replay")
        starts, histories, n = [], [], 0
        for _ in range(max_attempts):
            s, h = self.sample_starts(2 * B, hist)
            f = s["f"]
            ok = f + horizon <= clip_end[clip_id[f]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[:B - n]
            if idx.numel():
                starts.append({k: v[idx] for k, v in s.items()})
                histories.append({k: v[idx] for k, v in h.items()})
                n += len(idx)
            if n == B:
                break
        if n < B:
            # Physical storage rows are all valid when full, or [0,size) before
            # the first wrap. Convert selected rows back to age before history.
            frames = self.reference_frames().reshape(-1)
            candidates = []
            for start in range(0, frames.numel(), 65536):
                f = frames[start:start + 65536]
                ok = f + horizon <= clip_end[clip_id[f]]
                candidates.append(ok.nonzero(as_tuple=False).squeeze(-1) + start)
            candidates = torch.cat(candidates)
            if candidates.numel() == 0:
                raise RuntimeError(
                    f"No replay start has {horizon} remaining reference transitions "
                    f"(replay={self.size}x{self.N}). Check motion lengths/active pool "
                    "and imagination horizon; rejection sampling was stopped.")
            ids = candidates[torch.randint(candidates.numel(), (B - n,), device=self.device)]
            age = (ids // self.N - (self.ptr - self.size)) % self.T
            s, h = self._starts_at(age, ids % self.N, hist)
            starts.append(s)
            histories.append(h)
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}
        return cat(starts), cat(histories)

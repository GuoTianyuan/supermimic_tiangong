"""Tensor-only reset-sampler updates, independent of the simulator backend."""
from __future__ import annotations

import torch


@torch.no_grad()
def update_bin_returns_(values: torch.Tensor, bins: torch.Tensor, returns: torch.Tensor,
                        alpha: float = 0.2) -> None:
    """One EMA update per touched bin, using the mean of this batch's returns.

    Multiple finished environments may share a start bin. Aggregate before the
    indexed write so each bin is written exactly once, rather than last-writer
    wins. The .8/.2 EMA cadence is retained; this is not the official sampler.
    """
    if bins.numel() == 0:
        return
    unique, inverse, counts = torch.unique(bins, return_inverse=True, return_counts=True)
    sums = torch.zeros_like(unique, dtype=returns.dtype).scatter_add_(0, inverse, returns)
    mean = sums / counts
    values[unique] = (1.0 - alpha) * values[unique] + alpha * mean

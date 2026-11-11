"""Raw policy-action bounds and differentiable clipping."""

from __future__ import annotations
import torch


ACTION_CLIP = 20.0

ACTION_SOFT = 15.0

def soft_clip(a: torch.Tensor) -> torch.Tensor:
    """Identity for |a| <= ACTION_SOFT, beyond it a C1 tanh saturation towards ACTION_CLIP."""
    k, w = ACTION_SOFT, ACTION_CLIP - ACTION_SOFT
    m = a.abs()
    return torch.where(m <= k, a, a.sign() * (k + w * torch.tanh((m - k) / w)))

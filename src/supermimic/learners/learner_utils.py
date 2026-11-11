"""Distributed gradient averaging and finite-gradient guards."""

import torch
import torch.nn as nn
import torch.distributed as dist

def sync_grads(m: nn.Module, world: int):
    """Average gradients over ranks before clipping, so every rank takes the same optimizer step."""
    if world == 1:
        return
    ps = list(m.parameters())
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in ps])
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    for p, g in zip(ps, flat.split([p.numel() for p in ps])):
        p.grad = g.view_as(p)

def clip_finite_grads(m: nn.Module, max_norm: float):
    """clip_grad_norm_, and zero gradients when their norm is not finite (without a host sync, unlike
    the policy's skipped step): a NaN step would reach every rank's weights."""
    gn = nn.utils.clip_grad_norm_(m.parameters(), max_norm)
    bad = ~torch.isfinite(gn)
    for p in m.parameters():
        if p.grad is not None:
            p.grad.masked_fill_(bad, 0.0)

def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)

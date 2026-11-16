"""Atomic checkpoint writes, preserving the legacy load/export schema."""
import os
import time
import torch
import torch.distributed as dist
from supermimic.motions.pool import MotionPool


def save_checkpoint(n_it: int, *, components, env, sim_steps, t_start) -> str:
    args = components.args
    motion = components.motion
    wm = components.wm
    pi = components.pi
    critic = components.critic
    critic_targ = components.critic_targ
    opt_wm = components.opt_wm
    opt_pi = components.opt_pi
    opt_v = components.opt_v
    WORLD = components.world
    MAIN = components.is_main
    path = os.path.join(args.log_dir, f"ckpt_{n_it:06d}.pt")
    pool_states = None
    if isinstance(motion, MotionPool):
        state = motion.state_dict(env)
        if WORLD > 1:
            pool_states = [None] * WORLD if MAIN else None
            dist.gather_object(state, pool_states, dst=0)
        else:
            pool_states = [state]
    if not MAIN:
        return path
    ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "critic_targ": critic_targ.state_dict(),
          "opt_wm": opt_wm.state_dict(), "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": n_it,
          "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
          "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
          "cuda_memory": dict(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)}  # fmt: skip
    if pool_states is not None:
        ck["motion_pool"] = pool_states
    last = os.path.join(args.log_dir, "last.pt")
    torch.save(ck, last + ".tmp")
    os.replace(last + ".tmp", last)
    small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
    if pool_states is not None:
        small["motion_pool"] = pool_states
    torch.save(small, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"Saved checkpoint {path} (and last.pt)", flush=True)
    return path

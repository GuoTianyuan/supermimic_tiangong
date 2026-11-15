"""Reference slots in control frames; old checkpoints retain their original timing."""
from copy import deepcopy
from datetime import datetime, timezone

DEFAULT_GOAL_STRIDE = 5
DEFAULT_GOAL_START = 1


def positive_frames(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def checkpoint_goal_stride(config):
    """Checkpoints without timing metadata used consecutive future frames."""
    return positive_frames(config.get("goal_stride", 1), "goal_stride")


def checkpoint_goal_start(config):
    """Before goal_start existed, the first slot was exactly one stride ahead."""
    return positive_frames(config.get("goal_start", checkpoint_goal_stride(config)), "goal_start")


def goal_offsets(frames, stride, start=None):
    positive_frames(frames, "goal_frames")
    positive_frames(stride, "goal_stride")
    start = stride if start is None else positive_frames(start, "goal_start")
    return [start + stride * i for i in range(frames)]


def configure_reference_timing(args, checkpoint=None):
    """New runs use 10..460 ms; changing a resumed run needs explicit opt-in."""
    saved = checkpoint["args"] if checkpoint is not None else None
    if args.goal_stride is None:
        args.goal_stride = checkpoint_goal_stride(saved) if saved is not None else DEFAULT_GOAL_STRIDE
    if getattr(args, "goal_start", None) is None:
        args.goal_start = checkpoint_goal_start(saved) if saved is not None else DEFAULT_GOAL_START
    args.goal_offsets = goal_offsets(args.goal_frames, args.goal_stride, args.goal_start)
    args.reference_timing_changes = deepcopy(saved.get("reference_timing_changes", [])) if saved is not None else []
    if saved is None:
        return
    saved_frames = saved.get("goal_frames", args.goal_frames)
    if args.goal_frames != saved_frames:
        raise ValueError("Cannot resume with a different goal_frames: policy input dimensions would change")
    old = goal_offsets(saved_frames, checkpoint_goal_stride(saved), checkpoint_goal_start(saved))
    if args.goal_offsets != old:
        if not getattr(args, "allow_reference_timing_change", False):
            raise ValueError(f"Cannot resume with reference offsets {args.goal_offsets}; checkpoint uses {old}. "
                             "Use --allow_reference_timing_change only for intentional fine-tuning.")
        change = dict(from_offsets=old, to_offsets=list(args.goal_offsets),
                      checkpoint_iteration=checkpoint.get("iteration"),
                      checkpoint_path=getattr(args, "resume", None),
                      recorded_at=datetime.now(timezone.utc).isoformat(),
                      state_policy="Preserve networks, optimizers, normalizers and iteration; rebuild replay as on normal resume")
        args.reference_timing_changes.append(change)
        print(f"[reference timing] explicit resume change at iteration {checkpoint.get('iteration')}: "
              f"{old} -> {args.goal_offsets}; model/optimizer/normalizer states are retained", flush=True)

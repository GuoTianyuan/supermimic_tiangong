"""Future reference timing in control frames (independent of physics/WM time steps)."""

DEFAULT_GOAL_STRIDE = 5


def positive_frames(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def checkpoint_goal_stride(config):
    """Checkpoints without timing metadata always used consecutive future frames."""
    return positive_frames(config.get("goal_stride", 1), "goal_stride")


def goal_offsets(frames, stride):
    positive_frames(frames, "goal_frames")
    positive_frames(stride, "goal_stride")
    return [stride * i for i in range(1, frames + 1)]


def configure_reference_timing(args, checkpoint=None):
    """New runs use 50..500 ms; resume must preserve the learned input semantics."""
    saved = checkpoint_goal_stride(checkpoint["args"]) if checkpoint is not None else None
    if args.goal_stride is None:
        args.goal_stride = saved if saved is not None else DEFAULT_GOAL_STRIDE
    args.goal_offsets = goal_offsets(args.goal_frames, args.goal_stride)
    if saved is not None and args.goal_stride != saved:
        raise ValueError(f"Cannot resume with --goal_stride {args.goal_stride}; checkpoint uses {saved}. "
                         "Start a new run to change future reference timing.")

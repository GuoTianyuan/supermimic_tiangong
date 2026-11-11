"""Timing helpers for evaluation recordings, independent of the simulator."""
import math
import numpy as np


def recording_dt(record, control_hz=None):
    dt = float(np.asarray(record["control_dt"]).item()) if "control_dt" in record else None
    if control_hz is not None:
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("--control-hz must be finite and positive")
        requested = 1.0 / control_hz
        if dt is not None and not math.isclose(dt, requested, rel_tol=1e-6):
            raise ValueError("--control-hz conflicts with the recording's control_dt")
        dt = requested
    if dt is None:
        raise ValueError("Recording has no control_dt; supply --control-hz 50 or --control-hz 100")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Recording control_dt must be finite and positive")
    return dt


def render_indices(total_steps, dt, start, seconds, fps):
    if not all(math.isfinite(x) for x in (dt, start, seconds, fps)):
        raise ValueError("Video timing parameters must be finite")
    if total_steps < 0 or dt <= 0 or start < 0 or seconds <= 0 or fps <= 0:
        raise ValueError("Invalid video duration, start, frame rate or control period")
    duration = min(seconds, total_steps * dt - start)
    if duration <= 0:
        return np.empty(0, dtype=np.int64)
    times = start + np.arange(math.ceil(duration * fps - 1e-9)) / fps
    return np.minimum(np.floor(times / dt + 1e-9).astype(np.int64), total_steps - 1)

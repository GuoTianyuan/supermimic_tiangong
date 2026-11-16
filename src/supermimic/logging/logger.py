"""CSV persistence, compact TensorBoard and BeyondMimic-style terminal summaries."""
from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from supermimic.logging.legacy import CSV_FIELDS, build_tb_tags, _hms, _rss_gb
from .schema import REWARD_NAMES, EPISODE_FIELDS, dashboard_scalars, custom_layout

LOG_FIELDS = [*CSV_FIELDS, *EPISODE_FIELDS]


def open_csv(path, start_it):
    """Keep historical columns/values, discard rows newer than the resume checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        if not start_it:
            raise FileExistsError(f"Training log exists: {path}; use a new log_dir or --resume")
        with path.open() as stream:
            reader = csv.DictReader(stream)
            rows = [row for row in reader if row.get("iteration", "").isdigit()
                    and int(row["iteration"]) <= start_it]
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, LOG_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({k: row.get(k, "nan") for k in LOG_FIELDS} for row in rows)
        os.replace(tmp, path)
    stream = path.open("a", newline="")
    writer = csv.DictWriter(stream, LOG_FIELDS, extrasaction="ignore", restval="nan")
    if path.stat().st_size == 0:
        writer.writeheader()
    return stream, writer


def format_progress(row, *, args, control_dt, steps_per_s, elapsed_s, eta_s, last_ckpt):
    """One aligned line per term, matching the names shown in TensorBoard."""
    def v(key): return float(row.get(key, float("nan")))
    lines = ["#" * 88, f" Learning iteration {int(v('iteration')):,}/{args.iterations:,} | {Path(args.log_dir).name}", ""]
    def line(key, value): lines.append(f"{key + ':':>46} {value}")
    line("Computation", f"{steps_per_s:,.0f} steps/s | {v('it_per_s'):.2f} it/s")
    line("Loss/WM / Actor / Critic Huber", f"{v('wm_fit_loss_per_step'):.4f} / {v('pi_loss'):.4f} / {v('fit_weighted_huber'):.4f}")
    line("WM online 1 / replay horizon", f"{v('wm_nse1'):.4f} / {v('wm_err8'):.4f} ({args.wm_window} steps)")
    line("Mean net reward / max tracking", f"{v('sim_reward_normalized'):.4f}")
    line("Mean episode length", f"{v('ep_len'):.1f} steps ({v('ep_len')*control_dt:.2f} s)")
    line("Failed / ended episodes", f"{v('episodes_failed'):.0f} / {v('episodes_ended'):.0f} ({100*v('term_frac'):.2f}%)")
    lines += ["-" * 88, " Reward contribution per second     | Episode_Reward = episode sum / max seconds"]
    for key, name in REWARD_NAMES.items():
        episode = v("episode_" + key)
        suffix = f"{episode:9.4f}" if math.isfinite(episode) else "      n/a"
        line("Reward/" + name, f"{v(key)/control_dt:9.4f} | {suffix}")
    lines += ["-" * 88]
    line("Metrics/root XY / Z RMSE (m)", f"{v('error_root_xy_m'):.4f} / {v('error_root_z_m'):.4f}")
    line("Metrics/body position / orientation RMSE", f"{v('error_body_pos_m'):.4f} m / {v('error_body_ori_rad'):.4f} rad")
    line("Metrics/body linear / angular velocity RMSE", f"{v('error_body_linvel_mps'):.4f} m/s / {v('error_body_angvel_radps'):.4f} rad/s")
    line("Metrics/action delta / joint accel RMS", f"{v('action_delta_rms'):.4f} / {v('joint_acc_rms'):.2f} rad/s2")
    if v("pool_catalog") > 0:
        line("Motion pool active / resident", f"{v('pool_active'):,.0f} / {v('pool_resident'):,.0f}")
    line("DR range / nonfinite resets", f"{100*v('dr_scale'):.0f}% / {v('sim_nonfinite'):.0f}")
    line("Memory GPU reserved / host RSS (rank 0)", f"{v('gpu_reserved_gib'):.2f} / {v('host_rss_gib'):.2f} GiB")
    line("Learning rate", f"{v('lr'):.2e}")
    line("Total sim steps", f"{v('sim_steps'):,.0f}")
    line("Elapsed / ETA", f"{_hms(elapsed_s)} / {_hms(eta_s)}")
    line("Checkpoint", last_ckpt)
    return "\n".join(lines)


class TrainingLogger:
    def __init__(self, args, start_it=0, *, enabled=True, control_dt=.01):
        self.args, self.enabled, self.dt = args, enabled, control_dt
        self.stream = self.tb = self.details = None
        if not enabled:
            return
        directory = Path(args.log_dir)
        self.stream, self.csv = open_csv(directory / "train_log.csv", start_it)
        # Migrate old event streams without deleting them, then rebuild the compact
        # historical view from CSV. Otherwise old channel tags still clutter resumes.
        legacy = None
        event_dir = directory / "tensorboard"
        if start_it and event_dir.exists() and not (directory / "logging_schema.json").exists():
            legacy = directory / "tensorboard_legacy"
            index = 1
            while legacy.exists():
                legacy = directory / f"tensorboard_legacy_{index}"
                index += 1
            event_dir.rename(legacy)
        # Sibling directories: detailed channels do not clutter the default dashboard.
        self.tb = SummaryWriter(str(directory / "tensorboard"), purge_step=start_it or None, flush_secs=10)
        self.details = SummaryWriter(str(directory / "tensorboard_diagnostics"), purge_step=start_it or None, flush_secs=10)
        self.tb.add_custom_scalars(custom_layout())
        if legacy is not None:
            with (directory / "train_log.csv").open() as stream:
                for row in csv.DictReader(stream):
                    step = int(row["iteration"])
                    for tag, value in dashboard_scalars(row, control_dt, args.wm_window).items():
                        self.tb.add_scalar(tag, value, step)
        self.tags = build_tb_tags(args.wm_window)
        (directory / "logging_schema.json").write_text(json.dumps({
            "version": 2, "control_dt": control_dt,
            "legacy_events": str(legacy) if legacy else None,
            "reward": "weighted per-second contribution; CSV terms retain per-step units",
            "episode_reward": "mean completed-episode term sum / max episode seconds; missing if none ended",
            "episode_window": "all episodes ended since last log, pooled over ranks; not last-100 deque",
            "diagnostics": "tensorboard_diagnostics; full legacy and per-channel WM metrics",
        }, indent=2) + "\n")

    def log(self, row, channels, episode, **display):
        if not self.enabled:
            return
        row = {**row, **episode}
        self.csv.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()})
        self.stream.flush()
        step = int(row["iteration"])
        row["gpu_reserved_gib"] = torch.cuda.memory_reserved() / 2**30 if torch.cuda.is_available() else 0.
        row["host_rss_gib"] = _rss_gb()
        self.tb.add_scalar("Perf/gpu_reserved_gib_rank0", row["gpu_reserved_gib"], step)
        self.tb.add_scalar("Perf/host_rss_gib_rank0", row["host_rss_gib"], step)
        for tag, value in dashboard_scalars(row, self.dt, self.args.wm_window).items():
            self.tb.add_scalar(tag, value, step)
        for key, tag in self.tags.items():
            if key in row and math.isfinite(float(row[key])):
                self.details.add_scalar(tag, row[key], step)
        for tag, value in channels.items():
            self.details.add_scalar(tag, value, step)
        print(format_progress(row, args=self.args, control_dt=self.dt, **display), flush=True)

    def close(self):
        for attr in ("tb", "details", "stream"):
            value = getattr(self, attr)
            if value is not None:
                value.close()
                setattr(self, attr, None)

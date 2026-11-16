"""Build a readable TensorBoard view from an archived training CSV (read-only input).

Separate output is required: existing events, checkpoints and code are untouched.
Old CSVs have only per-step term means, so Episode_Reward is intentionally absent.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from supermimic.logging.schema import dashboard_scalars, custom_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty; original event streams are never overwritten")
    cfg = json.loads((args.run_dir / "args.json").read_text())
    dt = cfg.get("simulation_timing", {}).get("control_dt")
    if dt is None:
        parser.error("args.json is missing simulation_timing.control_dt; cannot infer reward units safely")
    from torch.utils.tensorboard import SummaryWriter
    count, final_step = 0, 0
    with SummaryWriter(str(args.output)) as writer:
        writer.add_custom_scalars(custom_layout())
        with (args.run_dir / "train_log.csv").open() as stream:
            for row in csv.DictReader(stream):
                if not row.get("iteration", "").isdigit():
                    continue
                step = int(row["iteration"])
                for tag, value in dashboard_scalars(row, dt, cfg["wm_window"], cfg.get("max_step_reward", 5.0 * dt) / dt).items():
                    writer.add_scalar(tag, value, step)
                count += 1
                final_step = step
    manifest = dict(source=str(args.run_dir.resolve()), rows=count, last_step=final_step,
                    control_dt=dt, wm_window=cfg["wm_window"],
                    episode_terms="Only if present in the original CSV; never synthesized from step averages")
    (args.output / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

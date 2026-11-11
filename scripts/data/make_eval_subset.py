"""Stratified evaluation subset of a motion directory, for supermimic/evaluate.py --motion_dir.

Up to ``--per_group`` clips of every motion group (the name before ``__``, see supermimic.motion.motion_group),
drawn with a fixed seed and symlinked with their validation metadata into ``--out``.

    python scripts/data/make_eval_subset.py --src data/wbc_full --out data/wbc_full_eval --per_group 32
"""

from __future__ import annotations

from pathlib import Path

import argparse
import os
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from supermimic.motion import motion_group  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--per_group", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

if os.path.exists(args.out) and os.listdir(args.out):
    sys.exit(f"{args.out} exists and is not empty")
groups: dict[str, list[str]] = {}
for f in sorted(os.listdir(args.src)):
    if f.endswith(".npz"):
        groups.setdefault(motion_group(f[:-4]), []).append(f)
os.makedirs(args.out, exist_ok=True)
rng = random.Random(args.seed)
total = 0
for g, files in sorted(groups.items()):
    pick = rng.sample(files, min(args.per_group, len(files)))
    for f in pick:
        for name in (f, f + ".metadata.json"):
            os.symlink(os.path.realpath(os.path.join(args.src, name)), os.path.join(args.out, name))
    total += len(pick)
    print(f"{g:<40} {len(pick):>4} of {len(files)}")
print(f"{total} clips from {len(groups)} groups -> {args.out}")

"""Validate a directory of Tiangong3 NPZ motions in parallel and link the valid ones into a training directory.

The source directory is only read. Each motion that passes tools/prepare_tiangong3.py's checks is
symlinked into --out next to its .metadata.json, so --out can be used as --motion_dir. Motions that
fail are left out and listed in <out>/prepare_failures.txt. Re-running skips finished motions.

    python tools/prepare_motion_dir.py --src /media/wbc_partial --out data/wbc_partial
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import contextlib
import io
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_tiangong3 as prep  # noqa: E402

ROBOT = MODEL = None


def _validate(job):
    src, dst = job
    meta = Path(str(dst) + ".metadata.json")
    if meta.is_file() and dst.is_symlink():
        return "skip", src.name, ""
    try:
        if not dst.is_symlink():
            dst.symlink_to(src)
        with contextlib.redirect_stdout(io.StringIO()):
            prep.validate(ROBOT, MODEL, dst)
        return "ok", src.name, ""
    except Exception as e:  # noqa: BLE001
        dst.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
        return "fail", src.name, f"{type(e).__name__}: {e}"


def main():
    global ROBOT, MODEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 96))
    args = ap.parse_args()
    src, out = args.src.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files in {src}")
    ROBOT, MODEL, _ = prep.convert()
    jobs = [(f, out / f.name) for f in files]
    counts, failures, t0 = {"ok": 0, "skip": 0, "fail": 0}, [], time.time()
    print(f"Validating {len(jobs)} motions from {src} into {out} with {args.workers} workers", flush=True)
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, (status, name, err) in enumerate(pool.imap_unordered(_validate, jobs, chunksize=16), 1):
            counts[status] += 1
            if err:
                failures.append(f"{name}\t{err}")
            if i % 2000 == 0 or i == len(jobs):
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(jobs)}  ok {counts['ok']}  skipped {counts['skip']}  failed {counts['fail']}"
                      f"  ({rate:.0f} files/s, ETA {(len(jobs) - i) / rate / 60:.1f} min)", flush=True)
    (out / "prepare_failures.txt").write_text("".join(f + "\n" for f in sorted(failures)))
    print(f"Done in {(time.time() - t0) / 60:.1f} min: {counts['ok'] + counts['skip']} usable, {counts['fail']} failed"
          + (f" (see {out / 'prepare_failures.txt'})" if failures else ""), flush=True)


if __name__ == "__main__":
    main()

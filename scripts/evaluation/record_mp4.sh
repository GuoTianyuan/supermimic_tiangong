#!/usr/bin/env bash
# Roll out a checkpoint and render it to MP4.
# Usage: scripts/evaluation/record_mp4.sh [ckpt] [motion] [out.mp4] [seconds]
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT=${1:-logs/wbc_full_dr_v3/ckpt_010000.pt}
MOTION=${2:-data/tiangong_motion/motion1.npz}
# a raw NPZ (e.g. /media/wbc_full/x.npz) has no validation metadata; use its prepared copy under data/
if [[ "$MOTION" == *.npz && ! -f "$MOTION.metadata.json" ]]; then
    for m in data/*/"$(basename "$MOTION")"; do
        if [[ -f "$m.metadata.json" ]]; then echo "using validated copy $m"; MOTION=$m; break; fi
    done
fi
RUN_DIR=$(dirname "$CKPT")
NAME=$(basename "$CKPT" .pt)_$(basename "$MOTION" .npz)
OUT=${3:-$RUN_DIR/$NAME.mp4}
SECONDS_=${4:-20}
TRAJ=$RUN_DIR/$NAME.traj.npz

export SUPERMIMIC_ROBOT=${SUPERMIMIC_ROBOT:-tiangong3}
export MUJOCO_GL=egl

# local: ./sm.sh (docker); on the server: LAUNCHER=./sm_native.sh
RUN="${LAUNCHER:-./sm_native.sh} run --robot $SUPERMIMIC_ROBOT"
$RUN python scripts/evaluation/evaluate.py --run_dir "$RUN_DIR" --ckpt "$CKPT" \
    --motion_dir "$MOTION" --robot "$SUPERMIMIC_ROBOT" --record "$TRAJ"
$RUN python scripts/evaluation/render.py "$TRAJ" "$OUT" --seconds "$SECONDS_" --title "$NAME"
echo "saved: $OUT"

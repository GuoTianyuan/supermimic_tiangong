#!/usr/bin/env bash
# Export a policy with one motion baked in to ONNX for xmigcs (policy/smdance); a multi-motion
# checkpoint is exported once per motion.
# Usage: scripts/export/export_onnx.sh <ckpt> <motion.npz> [out.onnx]
#   out.onnx defaults to <run>_<ckpt>_<motion>.onnx next to the checkpoint.
#   On the server: LAUNCHER=./sm_native.sh scripts/export/export_onnx.sh ...
#   XMIGCS=<xmigcs repo> also copies it to src/xmigcs/policy/smdance/model/ (then set onnx_path in
#   policy/smdance/config/smdance.yaml to the printed file name).
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT=${1:?usage: scripts/export/export_onnx.sh <ckpt> <motion.npz> [out.onnx]}
MOTION=${2:?usage: scripts/export/export_onnx.sh <ckpt> <motion.npz> [out.onnx]}
# a raw NPZ (e.g. /media/wbc_full/x.npz) has no validation metadata; use its prepared copy under data/
if [[ "$MOTION" == *.npz && ! -f "$MOTION.metadata.json" ]]; then
    for m in data/*/"$(basename "$MOTION")"; do
        if [[ -f "$m.metadata.json" ]]; then echo "using validated copy $m"; MOTION=$m; break; fi
    done
fi
RUN_DIR=$(dirname "$CKPT")
OUT=${3:-$RUN_DIR/$(basename "$RUN_DIR")_$(basename "$CKPT" .pt)_$(basename "$MOTION" .npz).onnx}

# local: ./sm.sh (docker); on the server: LAUNCHER=./sm_native.sh
${LAUNCHER:-./sm_native.sh} run --robot tiangong3 python scripts/export/export_onnx.py --ckpt "$CKPT" --motion "$MOTION" --out "$OUT"
rm -f "${OUT%.onnx}_check.npz"

if [[ -n "${XMIGCS:-}" ]]; then
    DEST=$XMIGCS/src/xmigcs/policy/smdance/model
    mkdir -p "$DEST"
    cp "$OUT" "$DEST/"
    echo "copied to $DEST/$(basename "$OUT")"
    echo "set onnx_path: \"$(basename "$OUT")\" in $XMIGCS/src/xmigcs/policy/smdance/config/smdance.yaml"
fi
echo "saved: $OUT"

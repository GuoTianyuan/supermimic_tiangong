#!/bin/bash
# Start the long-lived 'supermimic' container with one GPU and the repository mounted.
# Usage: docker/run_container.sh [GPU_INDEX] [MEMORY_LIMIT] [CONTAINER_NAME] [ROBOT]
set -euo pipefail
GPU=${1:-0}
MEM=${2:-128g}
NAME=${3:-supermimic}
ROBOT=${4:-g1}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
docker run -d --name "$NAME" --gpus "\"device=${GPU}\"" --memory="$MEM" --shm-size=16g --user root \
  -e HOME=/root -e SUPERMIMIC_ROBOT="$ROBOT" -e WARP_CACHE_PATH=/workspace/supermimic/.cache/warp -e MPLCONFIGDIR=/tmp/mpl \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/supermimic/.cache/inductor \
  -v "$REPO":/workspace/supermimic -w /workspace/supermimic supermimic:latest sleep infinity

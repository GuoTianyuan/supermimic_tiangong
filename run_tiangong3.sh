#!/bin/bash
# Usage: ./run_tiangong3.sh [run_name] [additional train.py arguments]
# Container supermimic-tiangong3 maps host GPU 0 to container cuda:0.
set -euo pipefail
cd "$(dirname "$0")"
RUN=${1:-tiangong3_balance}
if [ "$#" -gt 0 ]; then shift; fi
CONTAINER=supermimic-tiangong3 bash run_train.sh "$RUN" \
  --robot tiangong3 \
  --motion_dir data/tiangong_motion/dex_evt_balance__balance_phzs5_chr01__c2475856d1.npz \
  --num_envs 1024 --buffer_steps 1000 --wm_batch 1024 --pi_batch 1024 \
  --iterations 16000 "$@"

#!/usr/bin/env bash
# Usage: ./run_tiangong3.sh [run_name] [sm.sh train options, e.g. --gpu 1 -d] [train.py arguments]
set -euo pipefail
cd "$(dirname "$0")"
RUN=${1:-tiangong3_balance}
if [ "$#" -gt 0 ]; then shift; fi
exec ./sm_native.sh train "$RUN" --robot tiangong3 --gpu 0 \
  --motion_dir data/tiangong_motion/dex_evt_balance__balance_phzs5_chr01__c2475856d1.npz \
  --num_envs 1024 --buffer_steps 1000 --wm_batch 1024 --pi_batch 1024 \
  --iterations 16000 "$@"

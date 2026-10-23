#!/bin/bash
# Usage: [GPU=0] [CPUS=0-23] ./run_train.sh <run_name> [sm.sh train options] [train.py args...]
# Kept for compatibility; see ./sm.sh for status/logs/stop.
exec "$(dirname "$0")/sm.sh" train "$@"

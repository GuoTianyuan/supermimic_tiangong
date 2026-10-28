#!/usr/bin/env bash
# Default entry point: native Conda. Legacy Docker launcher: sm_docker.sh.
set -euo pipefail
cd "$(dirname "$0")"
exec ./sm_native.sh "$@"

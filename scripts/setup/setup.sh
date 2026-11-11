#!/usr/bin/env bash
# Native Conda install. No Docker, GitHub checkout, or online robot conversion.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ ${1:-} == --help ]]; then
    cat <<'EOF'
Usage: bash scripts/setup/setup.sh
Linux x86_64, NVIDIA driver compatible with CUDA 13, Conda required.
SM_CONDA_PREFIX     environment path (default: <repo>/.conda)
CONDA_EXE          conda executable (default: conda on PATH)
CONDA_CHANNEL      default: Tsinghua conda-forge mirror
PIP_INDEX_URL      default: Tsinghua PyPI mirror
PIP_EXTRA_INDEX_URL default: NVIDIA index and official PyPI fallback
SM_WHEELHOUSE      optional directory of wheels; pip then uses --no-index
SM_SKIP_GPU_CHECK=1 skips CUDA checks for installation on a login node
SM_CONDA_PREFIX may be set to an existing Python 3.12 Conda environment.
EOF
    exit 0
fi
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'Requires Linux x86_64.' >&2; exit 1; }
CONDA_BIN=${CONDA_EXE:-conda}
command -v "$CONDA_BIN" >/dev/null || { echo 'Install Miniforge/Conda first; see docs/DEPLOY_CONDA.md.' >&2; exit 1; }
PREFIX=${SM_CONDA_PREFIX:-$PWD/.conda}
[[ $PREFIX == /* ]] || PREFIX="$PWD/$PREFIX"
CHANNEL=${CONDA_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge}
if [[ ! -x $PREFIX/bin/python ]]; then
    "$CONDA_BIN" create -y -p "$PREFIX" --override-channels -c "$CHANNEL" python=3.12 pip tmux ffmpeg cxx-compiler
fi
PY="$PREFIX/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 required"'
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
export PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL:-https://pypi.nvidia.com https://pypi.org/simple}
export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-180} PIP_RETRIES=${PIP_RETRIES:-10}
PIP_ARGS=()
if [[ -n ${SM_WHEELHOUSE:-} ]]; then
    PIP_ARGS+=(--no-index --find-links "$SM_WHEELHOUSE")
fi
PIP_EXTRA_INDEX_URL= "$PY" -m pip install "${PIP_ARGS[@]}" --index-url "$PIP_INDEX_URL" 'pip>=25' wheel uv==0.11.16
# uv only accelerates wheel downloads; the environment itself remains Conda.
UV_ARGS=(--python "$PY" --no-deps --prerelease allow)
if [[ -n ${SM_WHEELHOUSE:-} ]]; then
    UV_ARGS+=(--no-index --find-links "$SM_WHEELHOUSE")
else
    # First mirror containing the exact locked version wins. Do not prefer an
    # overseas wheel just because the same version also exists on another index.
    UV_ARGS+=(--index "$PIP_INDEX_URL" --index-strategy unsafe-first-match)
    read -r -a EXTRA_INDEXES <<< "$PIP_EXTRA_INDEX_URL"
    for index in "${EXTRA_INDEXES[@]}"; do UV_ARGS+=(--index "$index"); done
fi
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-180} UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-10}
# Install the known working graph verbatim, avoiding upstream metadata conflicts.
# PyPI's Linux Torch 2.11 build uses CUDA 13; NVIDIA runtime pins are in the lock.
"$PREFIX/bin/uv" pip install "${UV_ARGS[@]}" torch==2.11.0 torchvision==0.26.0 -r requirements-native.lock
"$PY" scripts/setup/unpack_vendor.py
LOCAL_PACKAGES=()
for p in isaaclab isaaclab_assets isaaclab_contrib isaaclab_experimental isaaclab_newton isaaclab_ov isaaclab_physx isaaclab_ppisp isaaclab_rl isaaclab_tasks isaaclab_tasks_experimental isaaclab_visualizers; do
    LOCAL_PACKAGES+=(-e "$PWD/IsaacLab/source/$p")
done
"$PY" -m pip install --no-deps --no-build-isolation -e "$PWD/.vendor/newton" "${LOCAL_PACKAGES[@]}" -e "$PWD"
SM_PYTHON="$PY" ./sm_native.sh run python scripts/setup/check_install.py
echo "Ready. Use ./sm.sh (native Conda launcher). Environment: $PREFIX"
echo "Custom prefix: export SM_PYTHON='$PY' before future commands."

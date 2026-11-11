#!/usr/bin/env bash
# SuperMimic launcher for hosts without Docker (e.g. the H800 server); same commands as sm.sh.
# Each training runs in its own tmux session sm-<run>, so an SSH disconnect does not stop it.
#
#   ./sm_native.sh train <run> [--gpu N|N,M,...|all] [--robot g1|tiangong3] [--tb_port P | --no_tb] [-d] [train.py args...]
#       Shows the training in the terminal; Ctrl+C stops after the current iteration and saves a
#       checkpoint; Ctrl+B then D leaves it running. -d starts it without attaching.
#       Several GPUs train one policy with one process per GPU; --num_envs is per GPU and every GPU
#       loads its own share of the motion files. Ranks 1.. log to logs/<run>/rank<N>.log.
#   ./sm_native.sh status            trainings, TensorBoard ports, GPU memory per process, progress
#   ./sm_native.sh logs <run>        follow logs/<run>/stdout.log (Ctrl+C only stops viewing)
#   ./sm_native.sh attach <run>      back into the training's terminal
#   ./sm_native.sh stop <run>|--all  graceful stop (checkpoint is saved); TensorBoard exits with it
#   ./sm_native.sh tb [run...]       TensorBoard for existing runs (Ctrl+C closes it)
#   ./sm_native.sh run [--gpu N] [--robot R] <command...>
#
# SM_PYTHON selects the interpreter (default .conda/bin/python, then activated Conda environment).
# TensorBoard binds to 127.0.0.1; view it with: ssh -p <port> -L 16006:127.0.0.1:16006 <host>
set -euo pipefail
cd "$(dirname "$0")"
SELF=$(pwd)/$(basename "$0")
if [ -n "${SM_PYTHON:-}" ]; then
    PY=$SM_PYTHON
elif [ -x "$PWD/.conda/bin/python" ]; then
    PY=$PWD/.conda/bin/python
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PY=$CONDA_PREFIX/bin/python
else
    echo "Run bash scripts/setup/setup.sh first, or set SM_PYTHON to your Conda Python." >&2
    exit 1
fi
export SM_PYTHON=$PY
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
[ -x "$PY" ] || { echo "error: Python not found at $PY (set SM_PYTHON)" >&2; exit 1; }
export PATH="$(dirname "$PY"):$PATH" PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-$PWD/.cache/matplotlib}
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$PWD/.cache/warp}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$PWD/.cache/inductor}
mkdir -p "$MPLCONFIGDIR" "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR"

die() { echo "error: $*" >&2; exit 1; }

free_port() {
    local p=$1
    while (echo >"/dev/tcp/127.0.0.1/$p") 2>/dev/null; do p=$((p + 1)); done
    echo "$p"
}

train_pids() { pgrep -f "(scripts/training/train.py|supermimic/train.py) --log_dir logs/$1( |\$)" | paste -sd, || true; }

cmd_train() {
    [ $# -gt 0 ] || die "usage: ./sm_native.sh train <run> [options] [train.py args...]"
    local run=$1; shift
    [[ $run =~ ^[A-Za-z0-9_.-]+$ ]] || die "run name may only contain letters, digits, '_', '.', '-'"
    local detach=0 resume=0 a fwd=()
    for a in "$@"; do
        case $a in
            -d | --detach) detach=1 ;;
            *) [ "$a" = --resume ] && resume=1; fwd+=("$a") ;;
        esac
    done
    tmux has-session -t "sm-$run" 2>/dev/null && die "sm-$run is already running (./sm_native.sh attach|logs|stop $run)"
    [ -e "logs/$run/train_log.csv" ] && [ $resume = 0 ] && die "logs/$run already has a run; pick a new name or add --resume logs/$run/last.pt"
    mkdir -p "logs/$run"
    tmux new-session -d -s "sm-$run" -x 200 -y 50 "$(printf '%q ' env \
        "SM_PYTHON=$PY" "MPLCONFIGDIR=$MPLCONFIGDIR" "WARP_CACHE_PATH=$WARP_CACHE_PATH" \
        "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR" "$SELF" _train "$run" "${fwd[@]}")"
    if [ $detach = 1 ] || ! [ -t 1 ]; then
        echo "Started tmux session sm-$run. ./sm_native.sh logs $run | attach $run | stop $run"
        return
    fi
    tmux attach -t "sm-$run"
    if tmux has-session -t "sm-$run" 2>/dev/null; then
        echo "Detached; sm-$run keeps training. ./sm_native.sh logs $run | attach $run | stop $run"
    else
        tail -n 4 "logs/$run/stdout.log"
    fi
}

cmd__train() {  # body of the tmux session
    local run=$1; shift
    local gpu=${GPU:-0} robot=${SUPERMIMIC_ROBOT:-g1} tb_port="" no_tb=0 pass=()
    while [ $# -gt 0 ]; do
        case $1 in
            --gpu) gpu=$2; shift 2 ;;
            --tb_port) tb_port=$2; shift 2 ;;
            --no_tb) no_tb=1; shift ;;
            --robot) robot=$2; pass+=("$1" "$2"); shift 2 ;;
            *) pass+=("$1"); shift ;;
        esac
    done
    [ "$gpu" = all ] && gpu=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)
    local gpus
    IFS=, read -ra gpus <<<"$gpu"
    export SUPERMIMIC_ROBOT=$robot SUPERMIMIC_TB_URL=""
    # The shell and tee ignore Ctrl+C so the log stays complete; train.py installs its own handler.
    trap '' INT
    local tb_pid=""
    if [ $no_tb = 0 ]; then
        tb_port=${tb_port:-$(free_port 16006)}
        tensorboard --logdir "logs/$run/tensorboard" --host 127.0.0.1 --port "$tb_port" >"logs/$run/tensorboard.log" 2>&1 &
        tb_pid=$!
        export SUPERMIMIC_TB_URL="http://127.0.0.1:$tb_port"
    fi
    trap "[ -n '$tb_pid' ] && kill '$tb_pid' 2>/dev/null; true" EXIT
    echo "SuperMimic train: run $run | robot $robot | GPU $gpu | TensorBoard ${SUPERMIMIC_TB_URL:-off}"
    echo "  Ctrl+C: stop after the current iteration and save a checkpoint | Ctrl+B then D: leave it running"
    local cmd=("$PY" scripts/training/train.py --log_dir "logs/$run" "${pass[@]}")
    if [ ${#gpus[@]} = 1 ]; then
        CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" 2>&1 | tee -a "logs/$run/stdout.log"
        return
    fi
    # The pod environment sets NCCL_DEBUG=INFO, which floods the training log. Each rank defaulting
    # to one CPU thread per core makes loading the motion files about ten times slower.
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=$(free_port 29500) WORLD_SIZE=${#gpus[@]} NCCL_DEBUG=${SM_NCCL_DEBUG:-WARN}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( $(nproc) / WORLD_SIZE > 0 ? $(nproc) / WORLD_SIZE : 1 ))}
    local r pids=()
    for r in $(seq 1 $((WORLD_SIZE - 1))); do
        CUDA_VISIBLE_DEVICES=${gpus[$r]} RANK=$r LOCAL_RANK=$r "${cmd[@]}" >>"logs/$run/rank$r.log" 2>&1 &
        pids+=($!)
    done
    (CUDA_VISIBLE_DEVICES=${gpus[0]} RANK=0 LOCAL_RANK=0 "${cmd[@]}" 2>&1 | tee -a "logs/$run/stdout.log") &
    pids+=($!)
    # A rank that dies leaves the others blocked in a collective, so then all of them are killed.
    local ranks=("${pids[@]}") alive p st=0
    while [ ${#pids[@]} -gt 0 ] && [ $st = 0 ]; do
        sleep 2
        alive=()
        for p in "${pids[@]}"; do
            if kill -0 "$p" 2>/dev/null; then alive+=("$p"); else wait "$p" || st=$?; fi
        done
        pids=("${alive[@]}")
    done
    if [ $st != 0 ]; then
        echo "A training process exited with status $st; stopping all ranks (see logs/$run/rank*.log)"
        pkill -KILL -f "(scripts/training/train.py|supermimic/train.py) --log_dir logs/$run( |\$)" || true
        wait "${ranks[@]}" 2>/dev/null || true
        return "$st"
    fi
}

progress() {
    local csv=logs/$1/train_log.csv total
    [ -f "$csv" ] || { echo "no progress yet"; return; }
    total=$(grep -o '"iterations": [0-9]*' "logs/$1/args.json" | grep -o '[0-9]*$')
    tail -n 1 "$csv" | awk -F, -v total="$total" '$1 ~ /^[0-9]+$/ {
        eta = ($15 > 0) ? (total - $1) / $15 : 0
        printf "it %d/%d (%.0f%%) | ep_len %.0f | term %.0f%% | %.2f it/s | ETA %02d:%02d:%02d", $1, total, 100 * $1 / total, $12, 100 * $14, $15, eta / 3600, eta % 3600 / 60, eta % 60
        found = 1 } END { if (!found) printf "warming up" }'
}

cmd_status() {
    echo "== SuperMimic trainings"
    local s run pid found=0
    for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep '^sm-' || true); do
        found=1; run=${s#sm-}; pid=$(train_pids "$run")
        local rss="-" tb
        [ -n "$pid" ] && rss=$(ps -o rss= -p "$pid" | awk '{s += $1} END {printf "%.1f GB", s / 2^20}')
        tb=$(grep -o 'http://127.0.0.1:[0-9]*' "logs/$run/stdout.log" 2>/dev/null | tail -1 || true)
        echo "  $s  pid ${pid:-none}  RAM $rss  ${tb:-no TensorBoard}"
        echo "      $(progress "$run")"
    done
    [ $found = 1 ] || echo "  none running"
    echo "== GPUs"
    local uuid idx used
    declare -A gpu_idx
    while IFS=', ' read -r idx uuid; do gpu_idx[$uuid]=$idx; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
        awk -F', ' '{printf "  GPU %s (%s): %.1f / %.1f GB used, %s%% util\n", $1, $2, $3/1024, $4/1024, $5}'
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | while IFS=', ' read -r uuid pid used; do
        printf "    GPU %s  pid %-8s %5s GB  %s\n" "${gpu_idx[$uuid]:-?}" "$pid" "$(awk -v u="$used" 'BEGIN { printf "%.1f", u / 1024 }')" \
            "$(ps -o args= -p "$pid" 2>/dev/null | cut -c1-80)"
    done
}

cmd_logs() {
    [ $# -gt 0 ] || die "usage: ./sm_native.sh logs <run>"
    [ -f "logs/$1/stdout.log" ] || die "no log for $1"
    if tmux has-session -t "sm-$1" 2>/dev/null; then
        echo "(Ctrl+C only stops viewing; training keeps running)"
        exec tail -n 60 -f "logs/$1/stdout.log"
    fi
    tail -n 60 "logs/$1/stdout.log"
}

cmd_attach() {
    [ $# -gt 0 ] || die "usage: ./sm_native.sh attach <run>"
    exec tmux attach -t "sm-$1"
}

cmd_stop() {
    [ $# -gt 0 ] || die "usage: ./sm_native.sh stop <run>|--all"
    local runs r pid
    if [ "$1" = --all ]; then runs=$(tmux ls -F '#{session_name}' 2>/dev/null | grep '^sm-' | sed 's/^sm-//' || true)
    else runs=$1; tmux has-session -t "sm-$1" 2>/dev/null || die "sm-$1 is not running"; fi
    [ -n "$runs" ] || { echo "nothing running"; return; }
    for r in $runs; do
        pid=$(train_pids "$r")
        echo "Stopping $r ..."
        [ -n "$pid" ] && kill -INT ${pid//,/ }
    done
    for r in $runs; do
        for _ in $(seq 300); do tmux has-session -t "sm-$r" 2>/dev/null || break; sleep 1; done
        if tmux has-session -t "sm-$r" 2>/dev/null; then
            echo "$r did not exit within 300 s; killing it"; tmux kill-session -t "sm-$r"
        fi
        echo "$r stopped"
    done
}

cmd_tb() {
    local port=${TB_PORT:-$(free_port 16006)} spec="" r
    for r in "$@"; do [ -d "logs/$r/tensorboard" ] || die "logs/$r/tensorboard not found"; spec+="$r:logs/$r/tensorboard,"; done
    local logdir=(--logdir logs)
    [ -n "$spec" ] && logdir=(--logdir_spec "${spec%,}")
    echo "TensorBoard http://127.0.0.1:$port  (Ctrl+C closes it)"
    exec tensorboard "${logdir[@]}" --host 127.0.0.1 --port "$port"
}

cmd_run() {
    local gpu=${GPU:-0} robot=${SUPERMIMIC_ROBOT:-g1}
    while [ $# -gt 0 ]; do
        case $1 in
            --gpu) gpu=$2; shift 2 ;;
            --robot) robot=$2; shift 2 ;;
            *) break ;;
        esac
    done
    [ $# -gt 0 ] || die "usage: ./sm_native.sh run [--gpu N] [--robot R] <command...>"
    CUDA_VISIBLE_DEVICES=$gpu SUPERMIMIC_ROBOT=$robot MUJOCO_GL=egl exec "$@"
}

case ${1:-} in
    train | _train | status | logs | attach | stop | tb | run) c=$1; shift; "cmd_${c}" "$@" ;;
    *) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac

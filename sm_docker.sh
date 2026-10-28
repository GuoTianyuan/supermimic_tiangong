#!/usr/bin/env bash
# SuperMimic Docker launcher. Every command runs in a fresh, labelled container that is removed
# when it exits, so nothing keeps GPU memory, RAM or ports after a run ends.
#
#   ./sm.sh train <run> [--gpu N] [--robot g1|tiangong3] [--mem 24g] [--tb_port P | --no_tb] [-d] [train.py args...]
#       Foreground training; Ctrl+C stops after the current iteration, saves a checkpoint and
#       removes the container. -d trains in the background and only follows the output.
#       TensorBoard for the run is served from the same container on 127.0.0.1 (free port >= 16006).
#   ./sm.sh status             running SuperMimic containers, ports, RAM, GPU memory and progress
#   ./sm.sh logs <run>         follow a training's output (Ctrl+C only stops viewing)
#   ./sm.sh stop <run>|--all   graceful stop (checkpoint is saved), container is removed
#   ./sm.sh tb [run...]        TensorBoard for existing runs (Ctrl+C closes it)
#   ./sm.sh run [--gpu N] [--robot R] <command...>   one-off command, e.g. evaluate, render, tests
#   ./sm.sh cleanup            remove the old long-lived containers (supermimic, supermimic-tiangong3, ...)
#
# Environment defaults: GPU (0), SUPERMIMIC_ROBOT (g1), MEM (24g), CPUS (cpuset, e.g. 0-23).
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(pwd)
IMAGE=supermimic:latest
LABEL=supermimic.kind

die() { echo "error: $*" >&2; exit 1; }

free_port() {
    local p=$1
    while ss -ltnH "sport = :$p" | grep -q . || docker ps --format '{{.Ports}}' | grep -q "127.0.0.1:$p->"; do p=$((p + 1)); done
    echo "$p"
}

gpu_line() {
    nvidia-smi -i "$1" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null |
        awk -F', ' '{printf "GPU %s (%s): %.1f / %.1f GB used, %s%% util", $1, $2, $3/1024, $4/1024, $5}'
}

# Arguments shared by every container. The image's uid 1000 user (isaaclab) is the host user.
base_args() {
    local home=/home/isaaclab
    [ "$(id -u)" = 1000 ] || home=/tmp
    mkdir -p .cache/remote_assets
    BASE=(--rm --init --user "$(id -u):$(id -g)" -e HOME="$home" -e TINI_KILL_PROCESS_GROUP=1
        -e WARP_CACHE_PATH=/workspace/supermimic/.cache/warp -e TORCHINDUCTOR_CACHE_DIR=/workspace/supermimic/.cache/inductor
        -e MPLCONFIGDIR=/tmp/mpl -v "$REPO":/workspace/supermimic -v "$REPO/.cache/remote_assets":/tmp/https
        -w /workspace/supermimic)
    if [ -t 0 ] && [ -t 1 ]; then TTY=(-it); else TTY=(-i); fi
}

cmd_train() {
    [ $# -gt 0 ] || die "usage: ./sm.sh train <run> [options] [train.py args...]"
    local run=$1; shift
    [[ $run =~ ^[A-Za-z0-9_.-]+$ ]] || die "run name may only contain letters, digits, '_', '.', '-'"
    local gpu=${GPU:-0} robot=${SUPERMIMIC_ROBOT:-g1} mem=${MEM:-24g} tb_port="" no_tb=0 detach=0 resume=0 pass=()
    while [ $# -gt 0 ]; do
        case $1 in
            --gpu) gpu=$2; shift 2 ;;
            --mem) mem=$2; shift 2 ;;
            --tb_port) tb_port=$2; shift 2 ;;
            --no_tb) no_tb=1; shift ;;
            -d | --detach) detach=1; shift ;;
            --robot) robot=$2; pass+=("$1" "$2"); shift 2 ;;
            --resume) resume=1; pass+=("$1" "$2"); shift 2 ;;
            *) pass+=("$1"); shift ;;
        esac
    done
    local name=sm-train-$run
    docker ps -a --format '{{.Names}}' | grep -qx "$name" && die "$name is already running (./sm.sh logs $run, ./sm.sh stop $run)"
    if [ -e "logs/$run/train_log.csv" ] && [ $resume = 0 ]; then
        die "logs/$run already has a run; pick a new name or add --resume logs/$run/last.pt"
    fi
    mkdir -p "logs/$run"
    base_args

    local ports=() inner="trap '' INT TERM; " tb_msg="off"
    if [ $no_tb = 0 ]; then
        tb_port=${tb_port:-$(free_port 16006)}
        ports=(-p "127.0.0.1:$tb_port:6006")
        tb_msg="http://127.0.0.1:$tb_port"
        inner+="tensorboard --logdir logs/$run/tensorboard --host 0.0.0.0 --port 6006 >/tmp/tensorboard.log 2>&1 & "
    fi
    local quoted=""
    [ ${#pass[@]} -gt 0 ] && quoted=$(printf '%q ' "${pass[@]}")
    inner+="python supermimic/train.py --log_dir logs/$run ${quoted}2>&1 | tee -a logs/$run/stdout.log; exit \${PIPESTATUS[0]}"
    local cpus=()
    [ -n "${CPUS:-}" ] && cpus=(--cpuset-cpus "$CPUS")

    echo "SuperMimic train: run $run | robot $robot | container $name"
    echo "  $(gpu_line "$gpu") before start"
    echo "  RAM limit $mem | TensorBoard $tb_msg | logs logs/$run/"
    local run_args=("${BASE[@]}" --name "$name" --label "$LABEL=train" --label "supermimic.run=$run"
        --label "supermimic.gpu=$gpu" --gpus "\"device=$gpu\"" --memory "$mem" --memory-swap "$mem" --shm-size 16g
        -e SUPERMIMIC_ROBOT="$robot" -e SUPERMIMIC_TB_URL="${tb_msg#off}" "${ports[@]}" "${cpus[@]}")
    if [ $detach = 1 ]; then
        docker run -d "${run_args[@]}" "$IMAGE" bash -c "$inner" >/dev/null
        echo "  Running in the background. Ctrl+C below only stops viewing; stop training with ./sm.sh stop $run"
        exec docker logs -f "$name"
    fi
    echo "  Ctrl+C: stop after the current iteration, save a checkpoint and remove the container"
    exec docker run "${TTY[@]}" "${run_args[@]}" "$IMAGE" bash -c "$inner"
}

progress() {  # one-line progress of logs/<run> from its CSV
    local csv=logs/$1/train_log.csv total
    [ -f "$csv" ] || { echo "no progress yet"; return; }
    total=$(grep -o '"iterations": [0-9]*' "logs/$1/args.json" | grep -o '[0-9]*$')
    tail -n 1 "$csv" | awk -F, -v total="$total" '$1 ~ /^[0-9]+$/ {
        eta = ($15 > 0) ? (total - $1) / $15 : 0
        printf "it %d/%d (%.0f%%) | ep_len %.0f | term %.0f%% | %.2f it/s | ETA %02d:%02d:%02d", $1, total, 100 * $1 / total, $12, 100 * $14, $15, eta / 3600, eta % 3600 / 60, eta % 60
        found = 1 } END { if (!found) printf "warming up" }'
}

cmd_status() {
    echo "== SuperMimic containers"
    local ids
    ids=$(docker ps -q --filter "label=$LABEL")
    if [ -z "$ids" ]; then
        echo "  none running"
    else
        declare -A mem
        while read -r n m; do mem[$n]=$m; done < <(docker stats --no-stream --format '{{.Name}} {{.MemUsage}}' $ids | sed 's/ \/ /\//')
        local id n kind run gpu ports up
        for id in $ids; do
            IFS='|' read -r n kind run gpu < <(docker inspect -f '{{slice .Name 1}}|{{index .Config.Labels "'$LABEL'"}}|{{index .Config.Labels "supermimic.run"}}|{{index .Config.Labels "supermimic.gpu"}}' "$id")
            ports=$(docker ps -f id="$id" --format '{{.Ports}}' | grep -o '127.0.0.1:[0-9]*' | sed 's|^|http://|' || true)
            up=$(docker ps -f id="$id" --format '{{.Status}}')
            echo "  $n  [$kind]  $up  RAM ${mem[$n]:-?}  GPU ${gpu:--}  ${ports:-no ports}"
            [ "$kind" = train ] && echo "      $(progress "$run")"
        done
    fi
    local legacy
    legacy=$(docker ps -a --filter name=^supermimic --format '{{.Names}} ({{.State}})' | tr '\n' ' ')
    [ -n "$legacy" ] && echo "  old long-lived containers: $legacy-> ./sm.sh cleanup removes them"

    echo "== GPUs"
    local uuid idx
    declare -A gpu_idx
    while IFS=', ' read -r idx uuid; do gpu_idx[$uuid]=$idx; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    nvidia-smi --query-gpu=index --format=csv,noheader | while read -r idx; do echo "  $(gpu_line "$idx")"; done
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | while IFS=', ' read -r uuid pid used; do
        local owner cid
        cid=$(grep -o 'docker-[0-9a-f]\{12\}' "/proc/$pid/cgroup" 2>/dev/null | head -1 | cut -c8- || true)
        if [ -n "$cid" ]; then owner="container $(docker inspect -f '{{slice .Name 1}}' "$cid" 2>/dev/null || echo "$cid")"
        else owner="host: $(ps -o user=,args= -p "$pid" 2>/dev/null | cut -c1-70)"; fi
        printf "    GPU %s  pid %-8s %5s GB  %s\n" "${gpu_idx[$uuid]:-?}" "$pid" "$(awk -v u="$used" 'BEGIN { printf "%.1f", u / 1024 }')" "$owner"
    done
}

cmd_logs() {
    [ $# -gt 0 ] || die "usage: ./sm.sh logs <run>"
    if docker ps --format '{{.Names}}' | grep -qx "sm-train-$1"; then
        echo "(Ctrl+C only stops viewing; training keeps running)"
        exec docker logs -f --tail 200 "sm-train-$1"
    fi
    [ -f "logs/$1/stdout.log" ] || die "no running container or log for $1"
    echo "($1 is not running; last lines of logs/$1/stdout.log)"
    tail -n 60 "logs/$1/stdout.log"
}

cmd_stop() {
    [ $# -gt 0 ] || die "usage: ./sm.sh stop <run>|--all"
    local names n
    if [ "$1" = --all ]; then names=$(docker ps --filter "label=$LABEL" --format '{{.Names}}')
    else names=sm-train-$1; docker ps --format '{{.Names}}' | grep -qx "$names" || die "$names is not running"; fi
    [ -n "$names" ] || { echo "nothing running"; return; }
    for n in $names; do echo "Stopping $n ..."; docker kill --signal INT "$n" >/dev/null 2>&1 || true; done
    for n in $names; do
        if ! timeout 300 docker wait "$n" >/dev/null 2>&1 && docker ps -q --filter "name=^$n\$" | grep -q .; then
            echo "$n did not exit within 300 s; forcing it"; docker stop -t 5 "$n" >/dev/null 2>&1 || true
        fi
        echo "$n stopped and removed"
    done
}

cmd_tb() {
    base_args
    local port=${TB_PORT:-$(free_port 16006)} spec="" r
    for r in "$@"; do [ -d "logs/$r/tensorboard" ] || die "logs/$r/tensorboard not found"; spec+="$r:logs/$r/tensorboard,"; done
    local logdir=(--logdir logs)
    [ -n "$spec" ] && logdir=(--logdir_spec "${spec%,}")
    echo "TensorBoard http://127.0.0.1:$port  (Ctrl+C closes it and removes the container)"
    exec docker run "${TTY[@]}" "${BASE[@]}" --name "sm-tb-$port" --label "$LABEL=tb" -p "127.0.0.1:$port:6006" \
        "$IMAGE" tensorboard "${logdir[@]}" --host 0.0.0.0 --port 6006
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
    [ $# -gt 0 ] || die "usage: ./sm.sh run [--gpu N] [--robot R] <command...>"
    base_args
    exec docker run "${TTY[@]}" "${BASE[@]}" --name "sm-run-$$" --label "$LABEL=run" --label "supermimic.gpu=$gpu" \
        --gpus "\"device=$gpu\"" --memory "${MEM:-24g}" --shm-size 16g -e SUPERMIMIC_ROBOT="$robot" -e MUJOCO_GL=egl \
        "$IMAGE" "$@"
}

cmd_cleanup() {
    local legacy
    legacy=$(docker ps -a --filter name=^supermimic --format '{{.Names}}')
    [ -n "$legacy" ] || { echo "no old containers"; return; }
    echo "Old long-lived containers (the repository is bind-mounted, so no code, data or logs are lost):"
    docker ps -a --filter name=^supermimic --format '  {{.Names}}\t{{.Status}}'
    read -r -p "Remove them? [y/N] " ok
    [[ $ok =~ ^[Yy]$ ]] || { echo "kept"; return; }
    docker rm -f $legacy >/dev/null
    echo "removed"
}

case ${1:-} in
    train | status | logs | stop | tb | run | cleanup) c=$1; shift; "cmd_$c" "$@" ;;
    *) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac

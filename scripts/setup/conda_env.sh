#!/usr/bin/env bash
# Shared by the installer and launcher. Does not activate or mutate environments.
sm_conda_prefix() {
    if [[ -n ${SM_CONDA_PREFIX:-} ]]; then
        realpath -m "$SM_CONDA_PREFIX"
        return
    fi
    local name=${SM_CONDA_ENV:-tg_supermimic} conda_bin=${CONDA_EXE:-conda}
    if [[ ${CONDA_DEFAULT_ENV:-} == "$name" && -x ${CONDA_PREFIX:-}/bin/python ]]; then
        printf '%s\n' "$CONDA_PREFIX"
        return
    fi
    command -v "$conda_bin" >/dev/null || return 1
    "$conda_bin" env list --json | python3 -c '
import json, pathlib, sys
name = sys.argv[1]
matches = [p for p in json.load(sys.stdin)["envs"] if pathlib.Path(p).name == name]
if len(matches) != 1:
    sys.exit(1)
print(matches[0])
' "$name"
}

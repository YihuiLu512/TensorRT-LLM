#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Collect the exact pytest node ids of the R10 test files on a CPU node, inside the sm100 image
# with the Python overlay of SRC (no GPU: the CUDA compat library lets the conftest import).
# Usage (on the host of a Slurm CPU job): collect.sh <result parent>
# Site paths come from the environment: LLM_MODELS_ROOT (the model store), MODELS_MOUNT (the shared
# directory that holds it, mounted read-only) and LOCAL_CANDIDATES (space-separated node-local
# directories tried in order for the image copy; the first with 80 GB free is used).
set -uo pipefail
export TZ=Asia/Shanghai GIT_OPTIONAL_LOCKS=0
not_run() { printf 'NOT RUN: %s\n' "$*" >&2; exit 65; }
SRC=${SRC:-/home/scratch.yihlu_coreai/tmp/20261113-wt31-r10/src}
HEAD_COMMIT=${HEAD_COMMIT:-9b4022dd0dfe7581dd0580750d79f3261d7fc5be}  # the commit the shards run
FILES=(
    disaggregated/test_disaggregated.py
    disaggregated/test_disaggregated_etcd.py
    disaggregated/test_auto_scaling.py
    disaggregated/test_workers.py
    stress_test/disagg_cancel/test_disagg_cancel_stress.py
    accuracy/test_disaggregated_serving.py
    accuracy/test_disaggregated_multinode.py
    accuracy/test_dwdp_disaggregated_serving.py
    test_e2e.py
    examples/test_ray.py
)
UNIT_FILES=(
    llmapi/apps/test_disagg_serving_perf_metrics.py
    llmapi/apps/_test_disagg_serving_multi_nodes.py
    llmapi/apps/_test_disagg_serving_multi_nodes_service_discovery.py
)

if [[ ${1:-} == --container ]]; then
    RES=$2 LLM_MODELS_ROOT=$3
    PURE=$(/usr/bin/python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') || not_run 'no python'
    while IFS= read -r file; do
        [[ $file == tensorrt_llm/*.py && $file != *..* && -f $SRC/$file ]] || not_run "invalid overlay entry: $file"
        mkdir -p "$PURE/$(dirname "$file")" && cp "$SRC/$file" "$PURE/$file" && cmp -s "$SRC/$file" "$PURE/$file" \
            || not_run "overlay failed: $file"
    done < "$RES/overlay-files.txt"
    printf 'OVERLAY_COUNT %s\n' "$(wc -l < "$RES/overlay-files.txt")"
    for var in $(compgen -e); do case "$var" in PMIX*|PMI*|SLURM*) unset "$var" ;; esac; done
    [[ ! -d /opt/hpcx/ompi ]] || export OPAL_PREFIX=/opt/hpcx/ompi
    for compat in /usr/local/cuda*/compat/lib.real; do
        [[ -d $compat ]] && export LD_LIBRARY_PATH="$compat:${LD_LIBRARY_PATH:-}" && break
    done
    export LLM_ROOT=$SRC LLM_MODELS_ROOT PYTHONDONTWRITEBYTECODE=1
    export PYTEST_ADDOPTS='-p no:cacheprovider'
    rc=0
    cd "$SRC/tests/integration/defs" || not_run 'defs dir'
    for f in "${FILES[@]}"; do
        name=${f//\//_}
        /usr/bin/python3 -m pytest --collect-only -q --rootdir "$PWD" "$f" > "$RES/defs-$name.txt" 2> "$RES/defs-$name.err"
        r=$?; printf 'COLLECT defs %s rc=%s lines=%s\n' "$f" "$r" "$(grep -c '::' "$RES/defs-$name.txt")"
        [[ $r == 0 ]] || rc=1
    done
    cd "$SRC/tests/unittest" || not_run 'unittest dir'
    for f in "${UNIT_FILES[@]}"; do
        name=${f//\//_}
        /usr/bin/python3 -m pytest --collect-only -q "$f" > "$RES/unit-$name.txt" 2> "$RES/unit-$name.err"
        r=$?; printf 'COLLECT unit %s rc=%s lines=%s\n' "$f" "$r" "$(grep -c '::' "$RES/unit-$name.txt")"
        [[ $r == 0 ]] || rc=1
    done
    exit $rc
fi

PARENT=${1:?result parent}
[[ -n ${LLM_MODELS_ROOT:-} && -n ${MODELS_MOUNT:-} && -n ${LOCAL_CANDIDATES:-} ]] \
    || not_run 'LLM_MODELS_ROOT, MODELS_MOUNT and LOCAL_CANDIDATES must be set'
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || not_run 'no Slurm allocation'
SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
RES=$PARENT/collect-$SLURM_JOB_ID-r${SLURM_RESTART_COUNT:-0}
mkdir -p "$PARENT" && mkdir "$RES" || not_run 'result dir exists'
# tee ignores TERM, so that a TERM sent to every process of the job does not close the log
# before the shell's last lines are written.
exec > >(trap '' TERM; exec tee "$RES/driver.log") 2>&1
IMAGE_COMMIT=9fd3822a0019d23a2a6b372a219e0e3d51a1a95d
READY=/home/scratch.yihlu_coreai/build/e17503-9fd3822a00-x86-sm100-cv2.ready.sqsh
READY_SHA=a098ea8bed607cad155f5078e5b041ff81bc6ecc07ed360adf6184749f357b38
printf 'START %s job=%s node=%s src_head=%s\n' "$(date -Is)" "$SLURM_JOB_ID" "$(hostname)" "$(git -C "$SRC" rev-parse HEAD)"
[[ $(git -C "$SRC" rev-parse HEAD) == "$HEAD_COMMIT" ]] || not_run "source HEAD is not $HEAD_COMMIT"
[[ -z $(git -C "$SRC" status --porcelain --untracked-files=no) ]] || not_run 'source modified'
git -C "$SRC" diff --name-only "$IMAGE_COMMIT" HEAD -- 'cpp/**' > "$RES/cpp-diff.txt" || not_run 'cpp diff'
[[ ! -s $RES/cpp-diff.txt ]] || not_run 'C++ changed since image'
git -C "$SRC" diff --name-only "$IMAGE_COMMIT" HEAD -- 'tensorrt_llm/**' ':!tensorrt_llm/**.py' ':!tensorrt_llm/**.md' > "$RES/non-python-diff.txt" || not_run 'pkg diff'
[[ ! -s $RES/non-python-diff.txt ]] || not_run 'non-Python package data changed'
git -C "$SRC" diff --name-status --diff-filter=DR "$IMAGE_COMMIT" HEAD -- 'tensorrt_llm/**.py' > "$RES/removed-python.txt" || not_run 'rm diff'
[[ ! -s $RES/removed-python.txt ]] || not_run 'deleted or renamed package files'
git -C "$SRC" diff --name-only --diff-filter=AM "$IMAGE_COMMIT" HEAD -- 'tensorrt_llm/**.py' > "$RES/overlay-files.txt" || not_run 'overlay list'
LOCAL=''
for candidate in $LOCAL_CANDIDATES; do
    [[ -d $candidate ]] || continue
    avail=$(df -Pk "$candidate" | awk 'NR==2 {print $4}')
    [[ $avail =~ ^[0-9]+$ && $avail -ge 80000000 ]] || continue
    mkdir -m 700 "$candidate/j17874-collect-$USER-$SLURM_JOB_ID" && LOCAL=$candidate/j17874-collect-$USER-$SLURM_JOB_ID && break
done
[[ -n $LOCAL ]] || not_run 'no local dir with 80 GB free'
printf 'local=%s (no automatic deletion)\n' "$LOCAL" > "$RES/container.txt"
export ENROOT_DATA_PATH="$LOCAL/data" ENROOT_CACHE_PATH="$LOCAL/cache" ENROOT_TEMP_PATH=/tmp NVIDIA_VISIBLE_DEVICES=void
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH"
timeout 900s cp "$READY" "$LOCAL/ready.sqsh" || not_run 'image copy'
[[ $(timeout 900s sha256sum "$LOCAL/ready.sqsh" | cut -d' ' -f1) == "$READY_SHA" ]] || not_run 'image checksum'
ROOTFS=j17874-collect-$SLURM_JOB_ID
timeout 900s enroot create --name "$ROOTFS" "$LOCAL/ready.sqsh" || not_run 'enroot create'
timeout 1800s enroot start --mount /home/scratch.yihlu_coreai:/home/scratch.yihlu_coreai \
    --mount /home/scratch.yihlu_coreai_1:/home/scratch.yihlu_coreai_1 \
    --mount "$MODELS_MOUNT:$MODELS_MOUNT:none:bind,ro" \
    --mount "$SRC:$SRC:none:bind,ro" --root --rw "$ROOTFS" -- bash "$SCRIPT" --container "$RES" "$LLM_MODELS_ROOT"
rc=$?
printf 'END %s rc=%s\n' "$(date -Is)" "$rc"
exit $rc

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# WT-31 (R10): run one shard of the control-form baseline (no chain switch) on the allocated
# node(s). Partition, account, QoS, GPUs, nodes and time belong to the submitter (submit.sh).
# Site paths come from the environment: LLM_MODELS_ROOT (the model store), MODELS_MOUNT (the
# shared directory that holds it, mounted read-only) and LOCAL_PARENT (node-local disk for /tmp).
# With CASE_FILE set this is an array job: line SLURM_ARRAY_TASK_ID+1 of CASE_FILE is
# "<shard name>|<list file>". Exit codes: 64 bad input, 65 NOT RUN, 70 source changed,
# 73 result directory unavailable, 75 ENV_INVALID (a node precheck failed before any test ran:
# GPU busy, wrong GPU model or a missing shared mount; or the image does not match), 143
# terminated, 124 (or 137 after the kill-after grace) when the container step reached RUN_TIMEOUT;
# otherwise the exit code srun reports for the container step, which can also be an entry's 124 or
# 137 from timeout(1) (with 2 nodes, the highest of the two, so a node 0 that fails its setup (65)
# shows as 67 when node 1 then started no entry; the codes of r10_shard.sh are listed in its header
# and kept per node in node*/exit-code.txt).
# SRC_MODE (ro|rw) and CLEAN_SLURM_ENV (0|1) come from submit.sh --src-rw / --clean-slurm-env.
set -uo pipefail
export TZ=Asia/Shanghai GIT_OPTIONAL_LOCKS=0
if [[ -n ${CASE_FILE:-} ]]; then
    [[ ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-9]+$ && -r $CASE_FILE ]] || exit 64
    record=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CASE_FILE")
    [[ -n $record ]] || exit 64
    IFS='|' read -r SHARD LIST <<< "$record"
    export SHARD LIST
fi
for var in SRC HEAD_COMMIT LIST SHARD RESULT_ROOT IMAGE IMAGE_SHA256 GPU_COUNT EXPECT_GPU DRIVER_DIR \
    LLM_MODELS_ROOT MODELS_MOUNT LOCAL_PARENT; do
    [[ -n ${!var:-} ]] || { printf 'NOT RUN: missing %s\n' "$var" >&2; exit 64; }
done
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || { echo 'NOT RUN: no Slurm allocation'; exit 64; }
[[ $HEAD_COMMIT =~ ^[0-9a-f]{40}$ && $SHARD =~ ^[A-Za-z0-9_.-]+$ && $GPU_COUNT =~ ^[1-8]$ ]] || exit 64
[[ $IMAGE_SHA256 =~ ^[0-9a-f]{64}$ ]] || exit 64
[[ $SRC == /* && $LIST == /* && $RESULT_ROOT == /* && $IMAGE == /* && $DRIVER_DIR == /* ]] || exit 64
[[ $MODELS_MOUNT == /* && $LOCAL_PARENT == /* && $LLM_MODELS_ROOT == "$MODELS_MOUNT"/* ]] || exit 64
for path in "$SRC" "$LIST" "$RESULT_ROOT" "$IMAGE" "$DRIVER_DIR" "$MODELS_MOUNT" "$LOCAL_PARENT"; do
    [[ $path != *[,:]* ]] || { echo 'NOT RUN: unsupported mount path'; exit 64; }
done
export IMAGE_COMMIT=${IMAGE_COMMIT:-9fd3822a0019d23a2a6b372a219e0e3d51a1a95d}
# JOB_BUDGET: seconds from job start until the container step must have ended (--time is 4 h; the
# rest is for the kill-after grace and the bookkeeping below).
export TEST_TIMEOUT=${TEST_TIMEOUT:-3600} JOB_BUDGET=${JOB_BUDGET:-14100} MODE=${MODE:-run}
[[ $TEST_TIMEOUT =~ ^[1-9][0-9]*$ && $JOB_BUDGET =~ ^[1-9][0-9]*$ ]] || exit 64
[[ $MODE == run || $MODE == preflight || $MODE == probe ]] || exit 64
# SRC_MODE=rw: for tests that write into their working directory (the defs directory of SRC).
SRC_MODE=${SRC_MODE:-ro}
[[ $SRC_MODE == ro || $SRC_MODE == rw ]] || exit 64
export CLEAN_SLURM_ENV=${CLEAN_SLURM_ENV:-0}
[[ $CLEAN_SLURM_ENV == 0 || $CLEAN_SLURM_ENV == 1 ]] || exit 64
NNODES=${SLURM_JOB_NUM_NODES:-1}
[[ $NNODES == 1 || $NNODES == 2 ]] || exit 64
TAG=${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}
RUN="$RESULT_ROOT/$SHARD/job-${TAG:-$SLURM_JOB_ID}-r${SLURM_RESTART_COUNT:-0}"
export RUN
mkdir -p "$(dirname "$RUN")" || exit 73
mkdir "$RUN" || { echo "NOT RUN: result path exists: $RUN"; exit 73; }
# tee ignores TERM, so that a TERM sent to every process of the step or job does not close the
# log before finish() writes the exit status.
exec > >(trap '' TERM; exec tee "$RUN/job.log") 2>&1
START_SECONDS=$(date +%s)
PHASE=preflight
finish() {
    local code=$?
    trap - EXIT
    printf 'END %s phase=%s rc=%s elapsed=%ss\n' "$(date -Is)" "$PHASE" "$code" "$(( $(date +%s) - START_SECONDS ))"
    printf '%s\t%s\t%s\n' "$PHASE" "$code" "$(date -Is)" > "$RUN/final-status.tsv"
    exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
fail() { printf 'NOT RUN: %s\n' "$*"; exit 65; }
invalid() { printf 'ENV_INVALID: %s\n' "$*"; exit 75; }
printf 'START %s job=%s tag=%s shard=%s nodes=%s gpus/node=%s mode=%s\n' "$(date -Is)" "$SLURM_JOB_ID" \
    "${TAG:-none}" "$SHARD" "$SLURM_JOB_NODELIST" "$GPU_COUNT" "$MODE"
printf 'SRC=%s HEAD_COMMIT=%s IMAGE=%s\n' "$SRC" "$HEAD_COMMIT" "$IMAGE"
nproc; free -h
# Every node of the allocation: GPU model, free of other compute processes, shared mounts present.
srun --nodes="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
    printf "NODE %s id=%s\n" "$(hostname)" "$SLURM_NODEID"
    for mount in /home/scratch.yihlu_coreai /home/scratch.yihlu_coreai_1 "$1"; do
        [[ -n $(ls -A "$mount" 2>/dev/null | head -n 1) ]] || { echo "NOT RUN: $(hostname) lacks $mount"; exit 65; }
    done
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || exit 75
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    [[ $busy == 0 ]] || { echo "ENV_INVALID: $(hostname) GPU has $busy compute process(es)"; exit 75; }
    nvidia-smi --query-gpu=name --format=csv,noheader | grep -Eqv -- "$0" && { echo "ENV_INVALID: GPU is not $0"; exit 75; }
    exit 0' "$EXPECT_GPU" "$MODELS_MOUNT" > "$RUN/node-precheck.log" 2>&1
precheck=$?
cat "$RUN/node-precheck.log"
[[ $precheck == 0 ]] || invalid "node precheck rc=$precheck"
[[ -r $LIST && -s $LIST && -r $DRIVER_DIR/r10_shard.sh && -r $IMAGE ]] || fail 'missing list, driver or image'
[[ $(git -C "$SRC" rev-parse HEAD) == "$HEAD_COMMIT" ]] || fail 'source HEAD differs'
[[ -z $(git -C "$SRC" status --porcelain --untracked-files=no) ]] || fail 'source has tracked modifications'
git -C "$SRC" diff --name-only "$IMAGE_COMMIT" "$HEAD_COMMIT" -- 'cpp/**' > "$RUN/cpp-diff.txt" || fail 'cannot compare C++'
[[ ! -s $RUN/cpp-diff.txt ]] || fail 'C++ differs from image'
git -C "$SRC" diff --name-only "$IMAGE_COMMIT" "$HEAD_COMMIT" -- 'tensorrt_llm/**' ':!tensorrt_llm/**.py' \
    ':!tensorrt_llm/**.md' > "$RUN/non-python-diff.txt" || fail 'cannot compare package'
[[ ! -s $RUN/non-python-diff.txt ]] || fail 'non-Python package content differs'
git -C "$SRC" diff --name-only --diff-filter=DR "$IMAGE_COMMIT" "$HEAD_COMMIT" -- 'tensorrt_llm/**.py' \
    > "$RUN/removed-python.txt" || fail 'cannot compare deletions'
[[ ! -s $RUN/removed-python.txt ]] || fail 'deleted or renamed Python files'
git -C "$SRC" diff --name-only --diff-filter=AM "$IMAGE_COMMIT" "$HEAD_COMMIT" -- 'tensorrt_llm/**.py' \
    > "$RUN/overlay.txt" || fail 'cannot enumerate overlay'
cp -- "$LIST" "$RUN/selected.txt" || fail 'cannot freeze list'
# The batch script Slurm runs is its spooled copy ($0); the other inputs are read from DRIVER_DIR.
sha256sum "$LIST" "$0" "$DRIVER_DIR"/r10_shard.sh "$DRIVER_DIR"/r10_probe.py "$DRIVER_DIR"/hook/*.py \
    > "$RUN/inputs.sha256" || fail 'hash'
# Image identity: size and mtime against the reference values in submit.sh (IMAGE_STAT, fixed when
# the image's sha256 was registered); the full sha256 runs only with IMAGE_FULL_CHECK=1 or without
# IMAGE_STAT (28 GB over NFS per task).
stat -c '%s:%Y' "$IMAGE" > "$RUN/image-stat.txt" || fail 'image stat'
[[ -z ${IMAGE_STAT:-} || $(cat "$RUN/image-stat.txt") == "$IMAGE_STAT" ]] || invalid 'image size/mtime differ'
if [[ ${IMAGE_FULL_CHECK:-0} == 1 || -z ${IMAGE_STAT:-} ]]; then
    printf '%s  %s\n' "$IMAGE_SHA256" "$IMAGE" | timeout 900s sha256sum --check - > "$RUN/image-check.log" 2>&1 \
        || invalid 'image checksum failed or timed out (slow NFS on this node?)'
fi
[[ -d $LOCAL_PARENT && -w $LOCAL_PARENT ]] || fail 'local temporary parent unavailable'
LOCAL_RUN="$LOCAL_PARENT/j17874-r10/$USER-$SLURM_JOB_ID-${SLURM_ARRAY_TASK_ID:-0}-r${SLURM_RESTART_COUNT:-0}"
# One private /tmp per node, owner-only (it is the container's /tmp and its working directory);
# a pre-existing run directory or a parent owned by someone else is refused. Never deleted
# automatically: the paths are listed in local-run.txt.
srun --nodes="$NNODES" --ntasks-per-node=1 bash -c '
    mkdir -p -- "${0%/*}" && [[ -O ${0%/*} && ! -L ${0%/*} ]] && mkdir -m 700 -- "$0" "$0/tmp"' "$LOCAL_RUN" \
    || fail 'cannot create private tmp'
printf '%s (no automatic deletion)\n' "$LOCAL_RUN" > "$RUN/local-run.txt"
MOUNTS="/home/scratch.yihlu_coreai:/home/scratch.yihlu_coreai,/home/scratch.yihlu_coreai_1:/home/scratch.yihlu_coreai_1"
MOUNTS+=",$MODELS_MOUNT:$MODELS_MOUNT:ro,$SRC:$SRC:$SRC_MODE,$LOCAL_RUN/tmp:/tmp"
PHASE=container
RUN_TIMEOUT=$(( START_SECONDS + JOB_BUDGET - 150 - $(date +%s) ))
(( RUN_TIMEOUT > 600 )) || fail "no time left for the container step ($RUN_TIMEOUT s)"
# The container stops starting new entries 600 s before this point (r10_shard.sh).
export NNODES DEADLINE=$(( $(date +%s) + RUN_TIMEOUT ))
printf 'COMMAND timeout %ss srun --nodes=%s --ntasks-per-node=1 ... bash %s\n' "$RUN_TIMEOUT" "$NNODES" "$DRIVER_DIR/r10_shard.sh"
timeout --signal=TERM --kill-after=120s "${RUN_TIMEOUT}s" \
    srun --nodes="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=0 \
    --container-image="$IMAGE" --container-mounts="$MOUNTS" \
    --no-container-mount-home --container-remap-root --container-writable --container-workdir=/tmp --export=ALL \
    bash "$DRIVER_DIR/r10_shard.sh"
rc=$?
for node_rc in "$RUN"/node*/exit-code.txt; do
    [[ -f $node_rc ]] && printf 'NODE_RC %s %s\n' "$node_rc" "$(cat "$node_rc")"
done
[[ $(git -C "$SRC" rev-parse HEAD) == "$HEAD_COMMIT" ]] || { echo 'SOURCE_CHANGED'; exit 70; }
[[ -z $(git -C "$SRC" status --porcelain --untracked-files=no) ]] || { echo 'SOURCE_CHANGED'; exit 70; }
exit "$rc"

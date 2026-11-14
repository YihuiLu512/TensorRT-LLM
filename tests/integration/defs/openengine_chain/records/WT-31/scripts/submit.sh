#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# WT-31 (R10): submit one shard array of the control-form baseline from the cluster's login node.
#   submit.sh <group key> <cases psv> [--test-only] [--array <spec>] [--mode preflight|probe]
#             [--full-image-check] [--src-rw] [--clean-slurm-env]
#             [--time <H:MM:SS>] [--qos <qos>] [--partition <list>]
# Group keys come from build_shards.py: b200-<N>g[-ports], hopper-<N>g, 2node-8g (a rerun key keeps
# such a prefix, e.g. b200-4g-rerun). --mode probe runs
# r10_probe.py once (array index 0 unless --array is given) and needs a key with at least 2 GPUs.
# --src-rw mounts SRC writable, for tests that write into their working directory.
# --clean-slurm-env runs the entries without the Slurm and PMI variables of the job step, for tests
# that start their own mpirun (it would otherwise take its slots from the one-task step).
# The cases file, given by absolute path, has one "<shard name>|<absolute list path>" line per array
# task and no blank lines.
# --time is the job's time limit (default 4:00:00, at least 0:45:00, which leaves room for a full
# image check); the container step must end 300 s before it.
# --qos and --partition replace R10_QOS and the group's partition list for this submission.
# Site values come from the environment and have no defaults here: the group's partition list
# (R10_B200_PARTITIONS for 2node and b200 keys, R10_B200_1G_PARTITIONS for b200-1g,
# R10_HOPPER_PARTITIONS; comma-separated -p lists), its account (R10_B200_ACCOUNT or
# R10_HOPPER_ACCOUNT), R10_QOS, R10_EXCLUDE (nodes to avoid; may be empty), LLM_MODELS_ROOT,
# MODELS_MOUNT and LOCAL_PARENT (see r10_shard.sbatch).
# Optional (unset: the built-in values below): R10_BASE (the work directory: drivers/, results/
# and src/, the source checkout), R10_HEAD_COMMIT (the commit SRC must be at), R10_B200_IMAGE and
# R10_B200_IMAGE_STAT (for b200 and 2node keys: a copy of the B200 image, which must have the same
# sha256, and the copy's size:mtime; a job checks the full sha256 only with --full-image-check or
# without R10_B200_IMAGE_STAT, so check a new copy once that way), R10_GRES_GPU_TYPE (the GPU type
# in --gres of b200 and 2node keys; set and empty for a cluster whose GPUs have no type) and
# R10_SHARED_MOUNTS (colon-separated directories each node must have, bound into the container at
# the same path; R10_BASE must be inside one of them also after resolving links, and neither it
# nor any of them may be / or MODELS_MOUNT or lie inside MODELS_MOUNT, since that one is bound
# read-only).
# The jobs run a frozen, read-only copy of this directory, $BASE/drivers/<commit>/, exported from
# the commit with git archive (a directory with uncommitted changes is refused, so that what is
# submitted is what was reviewed); later edits do not reach running jobs.
set -euo pipefail
KEY=$1 CASES=$2; shift 2
TEST_ONLY='' ARRAY='' MODE=run FULL_CHECK=0 SRC_MODE=ro CLEAN_SLURM_ENV=0
TIME=4:00:00 QOS='' PARTITION=''
while [[ $# -gt 0 ]]; do
    case $1 in
        --array|--mode|--time|--qos|--partition)
            [[ $# -ge 2 ]] || { echo "$1 needs a value" >&2; exit 64; } ;;
    esac
    case $1 in
        --test-only) TEST_ONLY=--test-only; shift ;;
        --array) ARRAY=$2; shift 2 ;;
        --mode) MODE=$2; shift 2 ;;
        --full-image-check) FULL_CHECK=1; shift ;;
        --src-rw) SRC_MODE=rw; shift ;;
        --clean-slurm-env) CLEAN_SLURM_ENV=1; shift ;;
        --time) TIME=$2; shift 2 ;;
        --qos) QOS=$2; shift 2 ;;
        --partition) PARTITION=$2; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 64 ;;
    esac
done
[[ -r $CASES ]] || { echo "cases file unreadable: $CASES" >&2; exit 64; }
case $KEY in
    b200-1g*) PART_VAR=R10_B200_1G_PARTITIONS ACCOUNT_VAR=R10_B200_ACCOUNT ;;
    b200-*|2node-*) PART_VAR=R10_B200_PARTITIONS ACCOUNT_VAR=R10_B200_ACCOUNT ;;
    hopper-*) PART_VAR=R10_HOPPER_PARTITIONS ACCOUNT_VAR=R10_HOPPER_ACCOUNT ;;
    *) echo "unknown group $KEY" >&2; exit 64 ;;
esac
[[ $KEY =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "group key has other characters than [A-Za-z0-9_.-]: $KEY" >&2; exit 64; }
[[ $MODE == run || $MODE == preflight || $MODE == probe ]] || { echo "unknown mode $MODE" >&2; exit 64; }
PARTITION=${PARTITION:-${!PART_VAR:-}} QOS=${QOS:-${R10_QOS:-}} ACCOUNT=${!ACCOUNT_VAR:-}
[[ -n $PARTITION ]] || { echo "missing site setting $PART_VAR (or --partition)" >&2; exit 64; }
[[ -n $QOS ]] || { echo 'missing site setting R10_QOS (or --qos)' >&2; exit 64; }
[[ -n $ACCOUNT ]] || { echo "missing site setting $ACCOUNT_VAR" >&2; exit 64; }
for var in LLM_MODELS_ROOT MODELS_MOUNT LOCAL_PARENT; do
    [[ -n ${!var:-} ]] || { echo "missing site setting $var" >&2; exit 64; }
done
[[ -v R10_EXCLUDE ]] || { echo 'missing site setting R10_EXCLUDE (may be empty)' >&2; exit 64; }
[[ $TIME =~ ^([0-9]{1,2}):([0-5][0-9]):([0-5][0-9])$ ]] || { echo "--time is not H:MM:SS: $TIME" >&2; exit 64; }
TIME_S=$(( 10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]} ))
(( TIME_S >= 2700 )) || { echo "--time below 0:45:00: $TIME" >&2; exit 64; }
JOB_BUDGET=$(( TIME_S - 300 ))
COUNT=$(grep -c . "$CASES" || true)
(( COUNT > 0 )) || { echo "cases file is empty: $CASES" >&2; exit 64; }
[[ $MODE == probe ]] && ARRAY=${ARRAY:-0}
ARRAY=${ARRAY:-0-$((COUNT - 1))%4}
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
BASE=${R10_BASE:-/home/scratch.yihlu_coreai/tmp/20261113-wt31-r10}
HEAD_COMMIT=${R10_HEAD_COMMIT:-9b4022dd0dfe7581dd0580750d79f3261d7fc5be}
SHARED_MOUNTS=${R10_SHARED_MOUNTS:-/home/scratch.yihlu_coreai:/home/scratch.yihlu_coreai_1}
[[ $HEAD_COMMIT =~ ^[0-9a-f]{40}$ ]] || { echo "R10_HEAD_COMMIT is not a full sha: $HEAD_COMMIT" >&2; exit 64; }
[[ -z ${R10_B200_IMAGE_STAT:-} || ( -n ${R10_B200_IMAGE:-} && $R10_B200_IMAGE_STAT =~ ^[0-9]+:[0-9]+$ ) ]] \
    || { echo 'R10_B200_IMAGE_STAT needs R10_B200_IMAGE and the form <size>:<mtime>' >&2; exit 64; }
[[ ${R10_GRES_GPU_TYPE-b200} =~ ^[A-Za-z0-9_-]*$ ]] || { echo 'R10_GRES_GPU_TYPE invalid' >&2; exit 64; }
# Checked here rather than after the allocation: these values go into the comma-separated --export
# list and into the job's --container-mounts list (colon and comma separated); the batch script
# rejects the same values after the allocation.
plain_path() { [[ $1 == /* && $1 != *[,:$'\n']* ]]; }
[[ $SHARED_MOUNTS != *$'\n'* ]] || { echo 'R10_SHARED_MOUNTS has a newline' >&2; exit 64; }
IFS=: read -ra SHARED <<< "$SHARED_MOUNTS"
for path in "$CASES" "$BASE" "$LLM_MODELS_ROOT" "$MODELS_MOUNT" "$LOCAL_PARENT" "${R10_B200_IMAGE:-/}" \
    "${SHARED[@]}"; do
    plain_path "$path" || { echo "path is relative or has a comma, colon or newline: '$path'" >&2; exit 64; }
done
[[ $LLM_MODELS_ROOT == "$MODELS_MOUNT"/* ]] || { echo 'LLM_MODELS_ROOT is not inside MODELS_MOUNT' >&2; exit 64; }
inside() { [[ $1/ == "${2%/}"/* ]]; }  # <path> <directory>: path is the directory or lies in it
BASE_REAL=$(realpath -m -- "$BASE") MODELS_REAL=$(realpath -m -- "$MODELS_MOUNT")
IN_SHARED=0 REAL_IN_SHARED=0
for mount in "${SHARED[@]}"; do
    mount_real=$(realpath -m -- "$mount")
    [[ $mount_real != / ]] || { echo "shared mount $mount is /" >&2; exit 64; }
    inside "$BASE" "$mount" && IN_SHARED=1
    inside "$BASE_REAL" "$mount_real" && REAL_IN_SHARED=1
    ! inside "$mount" "$MODELS_MOUNT" && ! inside "$mount_real" "$MODELS_REAL" \
        || { echo "shared mount $mount is or lies in MODELS_MOUNT" >&2; exit 64; }
done
(( IN_SHARED && REAL_IN_SHARED )) || { echo 'R10_BASE is not inside R10_SHARED_MOUNTS' >&2; exit 64; }
! inside "$BASE" "$MODELS_MOUNT" && ! inside "$BASE_REAL" "$MODELS_REAL" \
    || { echo 'R10_BASE is or lies in MODELS_MOUNT' >&2; exit 64; }
! grep -qv . "$CASES" || { echo "cases file has a blank line: $CASES" >&2; exit 64; }
while IFS='|' read -r shard list; do
    [[ $shard =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "shard name has other characters than [A-Za-z0-9_.-]: $shard" >&2; exit 64; }
    plain_path "$list" && [[ -r $list && -s $list ]] \
        || { echo "shard list relative, with a comma, colon or newline, unreadable or empty: $list" >&2; exit 64; }
done < <(grep . "$CASES")
[[ -z $(git -C "$SCRIPTS" status --porcelain -- .) ]] || { echo "uncommitted changes in $SCRIPTS" >&2; exit 64; }
DRIVER_COMMIT=$(git -C "$SCRIPTS" rev-parse HEAD)
DRIVER_DIR=$BASE/drivers/$DRIVER_COMMIT
if [[ ! -d $DRIVER_DIR ]]; then
    # A unique staging name, so that an interrupted copy never blocks the next submission.
    STAGE="$DRIVER_DIR.partial.$$"
    mkdir -p "$BASE/drivers" && mkdir "$STAGE" \
        && git -C "$(git -C "$SCRIPTS" rev-parse --show-toplevel)" archive --format=tar \
            "$DRIVER_COMMIT:$(git -C "$SCRIPTS" rev-parse --show-prefix)" \
            | tar -x -C "$STAGE" \
        && chmod -R a-w "$STAGE" && mv -T "$STAGE" "$DRIVER_DIR" \
        || { echo "cannot freeze the drivers into $DRIVER_DIR" >&2; exit 73; }
fi
# The jobs execute this directory: it must be the submitter's own, not a link.
[[ -O $DRIVER_DIR && ! -L $DRIVER_DIR ]] || { echo "$DRIVER_DIR is not owned by $(id -un)" >&2; exit 64; }
ENV="ALL,CASE_FILE=$CASES,SRC=$BASE/src,HEAD_COMMIT=$HEAD_COMMIT"
ENV+=",RESULT_ROOT=$BASE/results/$KEY,DRIVER_DIR=$DRIVER_DIR,MODE=$MODE,SRC_MODE=$SRC_MODE,CLEAN_SLURM_ENV=$CLEAN_SLURM_ENV"
ENV+=",LLM_MODELS_ROOT=$LLM_MODELS_ROOT,MODELS_MOUNT=$MODELS_MOUNT,LOCAL_PARENT=$LOCAL_PARENT"
ENV+=",JOB_BUDGET=$JOB_BUDGET,SHARED_MOUNTS=$SHARED_MOUNTS"
case $KEY in
    b200-*|2node-*)
        IMAGE=/home/scratch.yihlu_coreai/build/e17503-9fd3822a00-x86-sm100-cv2.ready.sqsh
        SHA=a098ea8bed607cad155f5078e5b041ff81bc6ecc07ed360adf6184749f357b38
        IMAGE_STAT=29234315264:1794216097  # size:mtime of the file whose sha256 is SHA
        if [[ -n ${R10_B200_IMAGE:-} ]]; then IMAGE=$R10_B200_IMAGE IMAGE_STAT=${R10_B200_IMAGE_STAT:-}; fi
        EXPECT='B200' ;;
    hopper-*)
        IMAGE=/home/scratch.yihlu_coreai/build/e17503-9fd3822a00-x86-sm90-cv2.ready.sqsh
        SHA=aacc73567223f2536e044e6bbf231198fe5be8467c7c56d9803865114521733a
        IMAGE_STAT=28794388480:1794576492
        EXPECT='H100|H200' ;;
esac
GPUS=$(sed -E 's/^[a-z0-9]+-([0-9])g.*/\1/' <<< "$KEY")
[[ $GPUS =~ ^[1-8]$ ]] || { echo "no GPU count in group key: $KEY" >&2; exit 64; }
[[ $MODE != probe || $GPUS -ge 2 ]] || { echo "probe needs at least 2 GPUs: $KEY" >&2; exit 64; }
[[ $CLEAN_SLURM_ENV == 0 || ( $MODE == run && $KEY != 2node-* ) ]] \
    || { echo '--clean-slurm-env is for single-node run shards' >&2; exit 64; }
ENV+=",IMAGE=$IMAGE,IMAGE_SHA256=$SHA,IMAGE_STAT=$IMAGE_STAT,IMAGE_FULL_CHECK=$FULL_CHECK"
ENV+=",EXPECT_GPU=$EXPECT,GPU_COUNT=$GPUS"
GPU_TYPE=${R10_GRES_GPU_TYPE-b200}
GRES="gpu:${GPU_TYPE:+$GPU_TYPE:}"
ARGS=(--job-name="j17874-r10-$KEY" --array="$ARRAY" -A "$ACCOUNT" -q "$QOS" --time="$TIME"
      --export="$ENV"
      --output="$BASE/results/slurm-$KEY-%A_%a.log")
[[ -z $R10_EXCLUDE ]] || ARGS+=(--exclude="$R10_EXCLUDE")
case $KEY in
    2node-*) ARGS+=(-N 2 --ntasks-per-node=1 --gres="${GRES}8" --exclusive --mem=0 -p "$PARTITION") ;;
    hopper-*) ARGS+=(-N 1 --gres=gpu:$GPUS -c $((GPUS * 24)) --mem=$((GPUS * 200))G -p "$PARTITION") ;;
    b200-8g*) ARGS+=(-N 1 --gres="${GRES}8" --exclusive --mem=0 -p "$PARTITION") ;;
    b200-1g*) ARGS+=(-N 1 --gres="${GRES}1" -c 16 --mem=192G -p "$PARTITION") ;;
    b200-*) ARGS+=(-N 1 --gres="${GRES}$GPUS" -c $((GPUS * 24)) --mem=$((GPUS * 200))G -p "$PARTITION") ;;
esac
mkdir -p "$BASE/results/$KEY"
printf 'SUBMIT %s cases=%s array=%s drivers=%s\n' "$KEY" "$COUNT" "$ARRAY" "$DRIVER_DIR"
sbatch $TEST_ONLY "${ARGS[@]}" "$DRIVER_DIR/r10_shard.sbatch"

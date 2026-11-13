#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# WT-31 (R10): submit one shard array of the control-form baseline from the cluster's login node.
#   submit.sh <group key> <cases psv> [--test-only] [--array <spec>] [--mode preflight|probe]
#             [--full-image-check] [--src-rw] [--clean-slurm-env]
# Group keys come from build_shards.py: b200-<N>g[-ports], hopper-<N>g, 2node-8g (a rerun key keeps
# such a prefix, e.g. b200-4g-rerun). --mode probe runs
# r10_probe.py once (array index 0 unless --array is given) and needs a key with at least 2 GPUs.
# --src-rw mounts SRC writable, for tests that write into their working directory.
# --clean-slurm-env runs the entries without the Slurm and PMI variables of the job step, for tests
# that start their own mpirun (it would otherwise take its slots from the one-task step).
# Site values come from the environment and have no defaults here: R10_B200_PARTITIONS,
# R10_B200_1G_PARTITIONS, R10_HOPPER_PARTITIONS (comma-separated -p lists), R10_B200_ACCOUNT,
# R10_HOPPER_ACCOUNT, R10_QOS, R10_EXCLUDE (nodes to avoid; may be empty), LLM_MODELS_ROOT,
# MODELS_MOUNT and LOCAL_PARENT (see r10_shard.sbatch).
# The jobs run a frozen, read-only copy of this directory, $BASE/drivers/<commit>/, exported from
# the commit with git archive (a directory with uncommitted changes is refused, so that what is
# submitted is what was reviewed); later edits do not reach running jobs.
set -euo pipefail
KEY=$1 CASES=$2; shift 2
TEST_ONLY='' ARRAY='' MODE=run FULL_CHECK=0 SRC_MODE=ro CLEAN_SLURM_ENV=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-only) TEST_ONLY=--test-only; shift ;;
        --array) ARRAY=$2; shift 2 ;;
        --mode) MODE=$2; shift 2 ;;
        --full-image-check) FULL_CHECK=1; shift ;;
        --src-rw) SRC_MODE=rw; shift ;;
        --clean-slurm-env) CLEAN_SLURM_ENV=1; shift ;;
        *) echo "unknown option $1" >&2; exit 64 ;;
    esac
done
[[ -r $CASES ]] || { echo "cases file unreadable: $CASES" >&2; exit 64; }
for var in R10_B200_PARTITIONS R10_B200_1G_PARTITIONS R10_HOPPER_PARTITIONS R10_B200_ACCOUNT \
    R10_HOPPER_ACCOUNT R10_QOS LLM_MODELS_ROOT MODELS_MOUNT LOCAL_PARENT; do
    [[ -n ${!var:-} ]] || { echo "missing site setting $var" >&2; exit 64; }
done
[[ -v R10_EXCLUDE ]] || { echo 'missing site setting R10_EXCLUDE (may be empty)' >&2; exit 64; }
COUNT=$(grep -c . "$CASES" || true)
(( COUNT > 0 )) || { echo "cases file is empty: $CASES" >&2; exit 64; }
[[ $MODE == probe ]] && ARRAY=${ARRAY:-0}
ARRAY=${ARRAY:-0-$((COUNT - 1))%4}
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
BASE=/home/scratch.yihlu_coreai/tmp/20261113-wt31-r10
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
ENV="ALL,CASE_FILE=$CASES,SRC=$BASE/src,HEAD_COMMIT=9b4022dd0dfe7581dd0580750d79f3261d7fc5be"
ENV+=",RESULT_ROOT=$BASE/results/$KEY,DRIVER_DIR=$DRIVER_DIR,MODE=$MODE,SRC_MODE=$SRC_MODE,CLEAN_SLURM_ENV=$CLEAN_SLURM_ENV"
ENV+=",LLM_MODELS_ROOT=$LLM_MODELS_ROOT,MODELS_MOUNT=$MODELS_MOUNT,LOCAL_PARENT=$LOCAL_PARENT"
case $KEY in
    b200-*|2node-*)
        IMAGE=/home/scratch.yihlu_coreai/build/e17503-9fd3822a00-x86-sm100-cv2.ready.sqsh
        SHA=a098ea8bed607cad155f5078e5b041ff81bc6ecc07ed360adf6184749f357b38
        IMAGE_STAT=29234315264:1794216097  # size:mtime of the file whose sha256 is SHA
        EXPECT='B200' ACCOUNT=$R10_B200_ACCOUNT ;;
    hopper-*)
        IMAGE=/home/scratch.yihlu_coreai/build/e17503-9fd3822a00-x86-sm90-cv2.ready.sqsh
        SHA=aacc73567223f2536e044e6bbf231198fe5be8467c7c56d9803865114521733a
        IMAGE_STAT=28794388480:1794576492
        EXPECT='H100|H200' ACCOUNT=$R10_HOPPER_ACCOUNT ;;
    *) echo "unknown group $KEY" >&2; exit 64 ;;
esac
GPUS=$(sed -E 's/^[a-z0-9]+-([0-9])g.*/\1/' <<< "$KEY")
[[ $GPUS =~ ^[1-8]$ ]] || { echo "no GPU count in group key: $KEY" >&2; exit 64; }
[[ $MODE != probe || $GPUS -ge 2 ]] || { echo "probe needs at least 2 GPUs: $KEY" >&2; exit 64; }
[[ $CLEAN_SLURM_ENV == 0 || ( $MODE == run && $KEY != 2node-* ) ]] \
    || { echo '--clean-slurm-env is for single-node run shards' >&2; exit 64; }
ENV+=",IMAGE=$IMAGE,IMAGE_SHA256=$SHA,IMAGE_STAT=$IMAGE_STAT,IMAGE_FULL_CHECK=$FULL_CHECK"
ENV+=",EXPECT_GPU=$EXPECT,GPU_COUNT=$GPUS"
ARGS=(--job-name="j17874-r10-$KEY" --array="$ARRAY" -A "$ACCOUNT" -q "$R10_QOS" --time=04:00:00
      --export="$ENV"
      --output="$BASE/results/slurm-$KEY-%A_%a.log")
[[ -z $R10_EXCLUDE ]] || ARGS+=(--exclude="$R10_EXCLUDE")
case $KEY in
    2node-*) ARGS+=(-N 2 --ntasks-per-node=1 --gres=gpu:b200:8 --exclusive --mem=0 -p "$R10_B200_PARTITIONS") ;;
    hopper-*) ARGS+=(-N 1 --gres=gpu:$GPUS -c $((GPUS * 24)) --mem=$((GPUS * 200))G -p "$R10_HOPPER_PARTITIONS") ;;
    b200-8g*) ARGS+=(-N 1 --gres=gpu:b200:8 --exclusive --mem=0 -p "$R10_B200_PARTITIONS") ;;
    b200-1g*) ARGS+=(-N 1 --gres=gpu:b200:1 -c 16 --mem=192G -p "$R10_B200_1G_PARTITIONS") ;;
    b200-*) ARGS+=(-N 1 --gres=gpu:b200:$GPUS -c $((GPUS * 24)) --mem=$((GPUS * 200))G -p "$R10_B200_PARTITIONS") ;;
esac
mkdir -p "$BASE/results/$KEY"
printf 'SUBMIT %s cases=%s array=%s drivers=%s\n' "$KEY" "$COUNT" "$ARRAY" "$DRIVER_DIR"
sbatch $TEST_ONLY "${ARGS[@]}" "$DRIVER_DIR/r10_shard.sbatch"

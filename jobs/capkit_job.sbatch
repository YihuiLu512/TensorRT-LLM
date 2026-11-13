#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# capkit batch step (runs on the node, outside the container). Submitted only by
# `python3.11 -m capjobs.submit submit` as: sbatch <resource flags> <snap>/jobs/capkit_job.sbatch <snap>
# All resources come from the sbatch command line; this file has no #SBATCH lines and no
# hard-coded paths. Every value comes from <snap>/job.env (generated and quoted by capjobs.submit).
#
# Steps: job dir runs/J<job>-r<restart>/ -> image size + mtime check against the values recorded
# when capjobs.submit hashed the image (coreutils only; the image is not re-hashed here) ->
# main srun step into the container (capkit_inner.sh) -> fallback finalize step when the main
# step did not leave finalize.done. Exit code = main step's exit code (6 = ENV_INVALID).
set -uo pipefail

SNAP=${1:?usage: capkit_job.sbatch <snapshot dir>}
[[ $# -eq 1 ]] || { echo "capkit_job: exactly one argument expected" >&2; exit 2; }
# shellcheck source=/dev/null
source "$SNAP/job.env" || { echo "capkit_job: cannot source $SNAP/job.env" >&2; exit 3; }
[[ $CAPJOBS_SNAP == "$SNAP" ]] || { echo "capkit_job: job.env belongs to $CAPJOBS_SNAP" >&2; exit 3; }
for v in CAPJOBS_PLAN_ID CAPJOBS_RUN_ROOT CAPJOBS_IMAGE CAPJOBS_IMAGE_SHA256 CAPJOBS_IMAGE_SIZE \
         CAPJOBS_IMAGE_MTIME_NS CAPJOBS_MOUNTS CAPJOBS_SRUN_MPI CAPJOBS_TIME_LIMIT_S; do
    [[ -n ${!v:-} ]] || { echo "capkit_job: $v is empty" >&2; exit 3; }
done

JOB_ID=${SLURM_JOB_ID:?not inside a Slurm job}
RESTART=${SLURM_RESTART_COUNT:-0}
NODE=${SLURMD_NODENAME:-unknown}
CPT=${SLURM_CPUS_PER_TASK:?--cpus-per-task was not given}
JOB_DIR="$CAPJOBS_RUN_ROOT/J${JOB_ID}-r${RESTART}"
START_EPOCH=$(date +%s)
DEADLINE_EPOCH=$((START_EPOCH + CAPJOBS_TIME_LIMIT_S))

# A requeued job gets a new restart count, hence a new job dir; an existing dir is never reused.
mkdir "$JOB_DIR" || { echo "capkit_job: job dir $JOB_DIR exists or cannot be created" >&2; exit 3; }
mkdir "$JOB_DIR/env" || exit 3
LOG="$JOB_DIR/batch.log"

ts() { date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }
log() { printf '%s %s\n' "$(ts)" "$*" | tee -a "$LOG"; }
state() {  # job-level STATE line (same format as a launch tree's STATE, INTERFACES.md 5.7)
    printf '%s\t%s\t%s\t%s\n' "$(ts)" "$1" "capkit_job.sbatch" "$2" >> "$JOB_DIR/STATE"
}

{
    printf 'plan_id\t%s\n' "$CAPJOBS_PLAN_ID"
    printf 'job_id\t%s\n' "$JOB_ID"
    printf 'restart_count\t%s\n' "$RESTART"
    printf 'node\t%s\n' "$NODE"
    printf 'snapshot_dir\t%s\n' "$SNAP"
    printf 'start_epoch\t%s\n' "$START_EPOCH"
    printf 'deadline_epoch\t%s\n' "$DEADLINE_EPOCH"
    printf 'cpus_per_task\t%s\n' "$CPT"
} > "$JOB_DIR/batch.tsv"
log "start job=$JOB_ID restart=$RESTART node=$NODE plan=$CAPJOBS_PLAN_ID"

# Image check (plan RESOURCES 6): capjobs.submit hashed the image and recorded its sha256, size
# and mtime; here size and mtime (ns) must still be those values, so the sha256 still holds.
log "image check: $CAPJOBS_IMAGE"
MEAS_SIZE=$(stat -c %s -- "$CAPJOBS_IMAGE" 2>>"$LOG") || MEAS_SIZE=""
MEAS_MTIME_NS=$(date -r "$CAPJOBS_IMAGE" +%s%N 2>>"$LOG") || MEAS_MTIME_NS=""
{
    printf 'path\t%s\n' "$CAPJOBS_IMAGE"
    printf 'expected_sha256\t%s\n' "$CAPJOBS_IMAGE_SHA256"
    printf 'sha256_source\t%s\n' "capjobs.submit (hashed at submission)"
    printf 'expected_size\t%s\n' "$CAPJOBS_IMAGE_SIZE"
    printf 'expected_mtime_ns\t%s\n' "$CAPJOBS_IMAGE_MTIME_NS"
    printf 'measured_size\t%s\n' "$MEAS_SIZE"
    printf 'measured_mtime_ns\t%s\n' "$MEAS_MTIME_NS"
} > "$JOB_DIR/env/image_check.tsv"
if [[ $MEAS_SIZE != "$CAPJOBS_IMAGE_SIZE" || $MEAS_MTIME_NS != "$CAPJOBS_IMAGE_MTIME_NS" ]]; then
    log "ENV_INVALID: image size/mtime differ (size=$MEAS_SIZE mtime_ns=$MEAS_MTIME_NS)"
    state ENV_INVALID "image size or mtime differs from the submission"
    exit 6
fi

SRUN_COMMON=(--nodes=1 --ntasks=1 --cpus-per-task="$CPT" --cpu-bind=none --mpi="$CAPJOBS_SRUN_MPI"
             --container-image="$CAPJOBS_IMAGE" --container-mounts="$CAPJOBS_MOUNTS"
             --no-container-mount-home --container-remap-root --container-writable
             --container-workdir="$SNAP" --export=ALL)
export CAPJOBS_JOB_DIR="$JOB_DIR" CAPJOBS_DEADLINE_EPOCH="$DEADLINE_EPOCH" CAPJOBS_NODE="$NODE"

log "main step"
srun "${SRUN_COMMON[@]}" bash "$SNAP/jobs/capkit_inner.sh" "$SNAP" "$JOB_DIR" run
rc=$?
log "main step rc=$rc"

if [[ ! -e $JOB_DIR/finalize.done ]]; then
    log "finalize.done missing: running the fallback finalize step"
    srun "${SRUN_COMMON[@]}" bash "$SNAP/jobs/capkit_inner.sh" "$SNAP" "$JOB_DIR" finalize-only
    log "fallback finalize rc=$?"
fi
[[ -e $JOB_DIR/finalize.done ]] || log "WARNING: finalize did not complete; node-local data may remain"
log "end rc=$rc"
exit "$rc"

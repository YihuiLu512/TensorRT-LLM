#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# capkit inner script: runs inside the container, in the same srun step as every measured process
# (plan RESOURCES 3, 5). Started only by capkit_job.sbatch:
#   bash <snap>/jobs/capkit_inner.sh <snap> <job dir> {run|finalize-only}
#
# run: node-local dirs <base>/<user>-<job>-r<restart> -> overlay into site-packages
#      (byte-verified) -> PYTHONPATH unset; import origins checked -> job/image/mounts/local-disk/
#      gpu-idle files -> (srun --mpi=none: unset PMIX_* PMI_* SLURM_*) -> hardware profile ->
#      env gate -> launches (capjobs.launches).
#      Any failed check: job-level STATE ENV_INVALID, exit 6, no measurement. A job submitted with
#      --registered-hw-sha UNREGISTERED stops at the gate this way after writing the profile.
# EXIT trap (both modes): sweep copy-back + verify of the node-local dirs this run owns, then
#      delete each only when capcommon.scratchpath accepts the exact path and the verification
#      passed. A dir that mkdir did not create in this run (run mode), or that is not a real
#      directory owned by this user (finalize-only mode), is never swept or deleted.
set -uo pipefail

SNAP=${1:?usage: capkit_inner.sh <snap> <job dir> <run|finalize-only>}
JOB_DIR=${2:?job dir required}
MODE=${3:?mode required}
[[ $MODE == run || $MODE == finalize-only ]] || { echo "capkit_inner: bad mode $MODE" >&2; exit 2; }
# shellcheck source=/dev/null
source "$SNAP/job.env" || { echo "capkit_inner: cannot source job.env" >&2; exit 3; }

JOB_ID=${SLURM_JOB_ID:?not inside a Slurm job}
RESTART=${SLURM_RESTART_COUNT:-0}
JOB_USER=${SLURM_JOB_USER:?SLURM_JOB_USER unset}
PY=${CAPJOBS_PYTHON:?}
LOCAL_BASE=${CAPJOBS_LOCAL_BASE:?}
SHM_BASE=${CAPJOBS_SHM_BASE:?}
# The restart count is part of the name: a requeued run on the same node never meets the dirs
# an unclean earlier run left behind (capcommon.scratchpath sees "<job>-r<restart>" as the id).
SCRATCH_ID="$JOB_ID-r$RESTART"
LOCAL_ROOT="$LOCAL_BASE/$JOB_USER-$SCRATCH_ID"
SHM_ROOT="$SHM_BASE/$JOB_USER-$SCRATCH_ID"
LOCAL_OWNED=0
SHM_OWNED=0
BOOT_PP="$SNAP/tools"
LOG="$JOB_DIR/inner.$MODE.log"
export PYTHONDONTWRITEBYTECODE=1
IDENT=(--experiment "$CAPJOBS_EXPERIMENT" --arm "$CAPJOBS_JOB_ARM" --launch-seq 0
       --job-id "$JOB_ID" --restart-count "$RESTART" --register-sha "$CAPJOBS_REGISTER_SHA")

ts() { date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }
log() { printf '%s %s\n' "$(ts)" "$*" >> "$LOG"; }
state() { printf '%s\t%s\t%s\t%s\n' "$(ts)" "$1" "capkit_inner.sh" "$2" >> "$JOB_DIR/STATE"; }
env_invalid() { log "ENV_INVALID: $1"; state ENV_INVALID "$1"; exit 6; }

delete_scratch() {  # $1 = path, $2 = base, $3 = verification record
    local p=$1 base=$2 v=$3
    [[ -e $p || -L $p ]] || { log "nothing to delete at $p"; return 0; }
    # CONSTRAINTS 6: the path must be exactly <base>/<user>-<job>-r<restart>; checked here and by
    # scratchpath.
    if [[ $p != "$base/$JOB_USER-$SCRATCH_ID" ]]; then
        log "refuse to delete $p (not <base>/<user>-<job>-r<restart>)"
        return 1
    fi
    if PYTHONPATH="$BOOT_PP" "$PY" -m capcommon.scratchpath --path "${p:?}" --base "${base:?}" \
            --user "${JOB_USER:?}" --job-id "${SCRATCH_ID:?}" --verified-file "${v:?}" \
            >> "$LOG" 2>&1; then
        log "rm -rf -- ${p:?}"
        rm -rf -- "${p:?}"
        local rc=$?
        log "rm rc=$rc"
        return "$rc"
    fi
    log "scratchpath refused deletion of $p"
    return 1
}

owned_by_us() {  # finalize-only mode: an existing real directory owned by this user
    [[ -d $1 && ! -L $1 && -O $1 ]]
}

finalize() {
    trap - EXIT
    local v="$JOB_DIR/copyback/final.verify.json" ok=1 roots=()
    if [[ $MODE == finalize-only ]]; then
        owned_by_us "$LOCAL_ROOT" && LOCAL_OWNED=1
        owned_by_us "$SHM_ROOT" && SHM_OWNED=1
    fi
    [[ $LOCAL_OWNED == 1 ]] && roots+=(--local-root "$LOCAL_ROOT")
    [[ $SHM_OWNED == 1 ]] && roots+=(--shm-root "$SHM_ROOT")
    log "finalize: sweep ${roots[*]:-(no node-local dir owned by this run)}"
    mkdir -p "$JOB_DIR/copyback"
    if PYTHONPATH="$BOOT_PP" "$PY" -m capjobs.copyback sweep "${roots[@]}" \
            --dest-root "$CAPJOBS_RUN_ROOT" --job-dir "$JOB_DIR" --out "$v" >> "$LOG" 2>&1; then
        [[ $LOCAL_OWNED == 1 ]] && { delete_scratch "$LOCAL_ROOT" "$LOCAL_BASE" "$v" || ok=0; }
        [[ $SHM_OWNED == 1 ]] && { delete_scratch "$SHM_ROOT" "$SHM_BASE" "$v" || ok=0; }
    else
        log "finalize: copy-back sweep did not verify; node-local dirs kept"
        ok=0
    fi
    [[ $ok == 1 ]] && ts > "$JOB_DIR/finalize.done"
    log "finalize done ok=$ok"
}
trap finalize EXIT
trap 'log "signal TERM"; exit 143' TERM
trap 'log "signal INT"; exit 130' INT

if [[ $MODE == finalize-only ]]; then
    exit 0  # the EXIT trap does the work
fi

log "run: job=$JOB_ID restart=$RESTART user=$JOB_USER"
mkdir -p "$JOB_DIR/copyback" "$JOB_DIR/env" || env_invalid "cannot create job dir subdirectories"
mkdir -m 700 "$LOCAL_ROOT" || env_invalid "node-local dir $LOCAL_ROOT exists or cannot be created"
LOCAL_OWNED=1
mkdir -m 700 "$SHM_ROOT" || env_invalid "shm dir $SHM_ROOT exists or cannot be created"
SHM_OWNED=1

# Overlay (INTERFACES.md 1.2, 6.6): the only step that runs with PYTHONPATH (the snapshot copy).
unset PYTHONPATH
PURELIB=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    || env_invalid "cannot resolve site-packages of $PY"
PYTHONPATH="$BOOT_PP" "$PY" -m capjobs.overlay --manifest "$SNAP/overlay_manifest.json" \
    --source-root "$SNAP" --dest "$PURELIB" --out "$JOB_DIR/env/overlay_verify.json" \
    "${IDENT[@]}" --role jobs >> "$LOG" 2>&1 || env_invalid "overlay copy/verification failed"
unset PYTHONPATH

ef() { "$PY" -m capjobs.envfiles "$@" "${IDENT[@]}" --role jobs >> "$LOG" 2>&1; }
ef imports --manifest "$SNAP/overlay_manifest.json" --dest "$PURELIB" \
    --out "$JOB_DIR/env/import_origin.json" || env_invalid "capkit packages not imported from $PURELIB"
ef job --snap "$SNAP" --plan-id "$CAPJOBS_PLAN_ID" --node "${CAPJOBS_NODE:-unknown}" \
    --deadline-epoch "$CAPJOBS_DEADLINE_EPOCH" --out "$JOB_DIR/job.json" \
    || env_invalid "snapshot verification failed"
ef image --check-file "$JOB_DIR/env/image_check.tsv" --out "$JOB_DIR/env/image.json" \
    || env_invalid "image check failed"
ef mounts --mounts-json "$SNAP/mounts.json" --must-read "$SNAP" --must-read "$JOB_DIR" \
    --out "$JOB_DIR/env/mounts.json" || env_invalid "mount check failed"
ef local-disk --path "$LOCAL_BASE" --path "$SHM_BASE" \
    --min-free-bytes "$CAPJOBS_MIN_LOCAL_FREE_BYTES" --out "$JOB_DIR/env/local_disk.json" \
    || env_invalid "local disk check failed"
ef gpu-idle --nvidia-smi "$CAPJOBS_NVIDIA_SMI" --max-alloc-gpus "$CAPJOBS_MAX_ALLOC_GPUS" \
    --out "$JOB_DIR/env/gpu_idle.json" || env_invalid "GPU idle check failed"

if [[ $CAPJOBS_SRUN_MPI == none ]]; then
    # RESOURCES 6: MPI singleton start; keep OPAL_PREFIX. Values needed later were captured above.
    while IFS= read -r name; do
        unset "$name"
    done < <(compgen -e | command grep -E '^(PMIX_|PMI_|SLURM_)')
    log "unset PMIX_*/PMI_*/SLURM_* for --mpi=none"
fi

# Hardware profile (INTERFACES.md 5.5.3) for the gate; the chassis source is the registered one.
# With chassis_source slurm_view the profile needs env/slurm_view.json, which the front end writes
# after the job starts: wait for it up to the registered slurm_view_wait_s.
SVIEW="$JOB_DIR/env/slurm_view.json"
if [[ $CAPJOBS_CHASSIS_SOURCE == slurm_view ]]; then
    wait_end=$(( $(date +%s) + ${CAPJOBS_SLURM_VIEW_WAIT_S%.*} + 1 ))
    while [[ ! -s $SVIEW && $(date +%s) -lt $wait_end ]]; do sleep 1; done
    [[ -s $SVIEW ]] || env_invalid "$SVIEW not written within ${CAPJOBS_SLURM_VIEW_WAIT_S} s"
fi
SVIEW_ARG=()
[[ -s $SVIEW ]] && SVIEW_ARG=(--slurm-view-file "$SVIEW")
log "hardware profile (chassis source $CAPJOBS_CHASSIS_SOURCE)"
"$PY" -m capenv.hwprofile "${IDENT[@]}" --role hwprofile --out "$JOB_DIR/env/hw_profile.json" \
    --raw-out "$JOB_DIR/env/hw_raw.json" --local-dir "$LOCAL_ROOT" \
    --chassis-source "$CAPJOBS_CHASSIS_SOURCE" "${SVIEW_ARG[@]}" \
    --image-sha256 "$CAPJOBS_IMAGE_SHA256" --code-baseline "$CAPJOBS_CODE_SHA" \
    --container-runtime pyxis-enroot --nvidia-smi "$CAPJOBS_NVIDIA_SMI" >> "$LOG" 2>&1 \
    || env_invalid "hardware profile failed"

GATE_CMP=(--no-comparable)
[[ -n ${CAPJOBS_COMPARABLE_FILE:-} ]] && GATE_CMP=(--comparable-file "$CAPJOBS_COMPARABLE_FILE")
log "gate: $CAPJOBS_GATE_MODULE"
"$PY" -m "$CAPJOBS_GATE_MODULE" --run-dir "$JOB_DIR" --phase pre \
    --registered-hw-sha "$CAPJOBS_REGISTERED_HW_SHA" "${GATE_CMP[@]}" \
    --hw-profile "$JOB_DIR/env/hw_profile.json" \
    --slurm-view-file "$SVIEW" --slurm-view-wait-s "$CAPJOBS_SLURM_VIEW_WAIT_S" \
    --max-alloc-gpus "$CAPJOBS_MAX_ALLOC_GPUS" --idle-window-s "$CAPJOBS_IDLE_WINDOW_S" \
    --idle-busy-max-frac "$CAPJOBS_IDLE_BUSY_MAX_FRAC" --nvidia-smi "$CAPJOBS_NVIDIA_SMI" \
    "${IDENT[@]}" --role gate >> "$LOG" 2>&1
grc=$?
if [[ $grc != 0 && $CAPJOBS_REGISTERED_HW_SHA == UNREGISTERED ]]; then
    env_invalid "environment gate exit $grc (hardware profile not registered: profile-only job)"
fi
[[ $grc == 0 ]] || env_invalid "environment gate exit $grc"

log "launches"
"$PY" -m capjobs.launches --plan "$SNAP/launch_plan.json" \
    --overlay-manifest "$SNAP/overlay_manifest.json" --nvidia-smi "$CAPJOBS_NVIDIA_SMI" \
    --snap "$SNAP" --job-dir "$JOB_DIR" \
    --local-root "$LOCAL_ROOT" --shm-root "$SHM_ROOT" --run-root "$CAPJOBS_RUN_ROOT" \
    --job-id "$JOB_ID" --restart-count "$RESTART" --python "$PY" \
    --deadline-epoch "$CAPJOBS_DEADLINE_EPOCH" --out "$JOB_DIR/launches.json" >> "$LOG" 2>&1
lrc=$?
log "launches rc=$lrc"
exit "$lrc"

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tokenizer copy job (purpose=prep_tokenizer; user decision 2026-11-13, INTERFACES erratum R1b).
# The only job type allowed to mount part of the model library. Submitted only by
# `capjobs.submit prep-tokenizer` (see jobs/README.md), which writes <plan dir>/job.env and copies this
# file to <plan dir>/jobs/prep_tokenizer.sbatch:
#   sbatch <minimal flags: no GPU, no --exclusive> <plan dir>/jobs/prep_tokenizer.sbatch <plan dir>
# No Python anywhere, no #SBATCH lines, no hard-coded paths: every value comes from job.env.
#
# Batch step (node host): job dir <plan dir>/J<job>-r<restart>/ (never reused) -> image size and
# mtime still the recorded ones -> one srun step into the container with the registered model
# dir mounted read-only:   prep_tokenizer.sbatch inner <plan dir> <job dir> <mountinfo file>
# Inner step (container, bash + coreutils only): the model dir is a read-only mount and nothing else of
# the model library is visible -> copy whitelisted top-level non-weight files into
# <dest>.partial (weight suffixes refused, any whitelisted file above the registered size limit
# fails the job) -> source and copy sha256 equal -> COPY_MANIFEST.tsv (name, size, sha256) ->
# re-check the copy (regular files only, no weight file, nothing unlisted) -> read-only, rename
# to <dest> -> sha256sum -c. Exit 0 ok, 2 usage, 3 config, 4 copy check failed, 6 environment.
set -uo pipefail
export LC_ALL=C
shopt -s nullglob dotglob

ts() { date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }

load_env() {
    local plan=$1 v
    # shellcheck source=/dev/null
    source "$plan/job.env" || { echo "prep_tokenizer: cannot source $plan/job.env" >&2; exit 3; }
    [[ $CAPJOBS_RUN_ROOT == "$plan" ]] || { echo "prep_tokenizer: job.env belongs to $CAPJOBS_RUN_ROOT" >&2; exit 3; }
    for v in CAPJOBS_PLAN_ID CAPJOBS_IMAGE CAPJOBS_IMAGE_SIZE CAPJOBS_IMAGE_MTIME_NS CAPJOBS_MOUNTS \
             CAPJOBS_PREP_MODEL_DIR CAPJOBS_PREP_DEST CAPJOBS_PREP_ALLOW_GLOBS \
             CAPJOBS_PREP_VOCAB_GLOBS CAPJOBS_PREP_WEIGHT_SUFFIXES CAPJOBS_PREP_MAX_FILE_BYTES \
             CAPJOBS_PREP_LIBRARY_ROOTS; do
        [[ -n ${!v:-} ]] || { echo "prep_tokenizer: $v is empty" >&2; exit 3; }
    done
    [[ $CAPJOBS_PREP_MODEL_DIR == /* && $CAPJOBS_PREP_DEST == /* ]] \
        || { echo "prep_tokenizer: model dir and destination must be absolute" >&2; exit 3; }
    [[ $CAPJOBS_PREP_MAX_FILE_BYTES =~ ^[1-9][0-9]*$ ]] \
        || { echo "prep_tokenizer: bad CAPJOBS_PREP_MAX_FILE_BYTES" >&2; exit 3; }
}

# ---- batch step (host) ---------------------------------------------------------------------------
batch() {
    local plan=$1 job rst jd rc
    load_env "$plan"
    job=${SLURM_JOB_ID:?not inside a Slurm job}
    rst=${SLURM_RESTART_COUNT:-0}
    jd="$plan/J${job}-r${rst}"
    mkdir "$jd" || { echo "prep_tokenizer: job dir $jd exists or cannot be created" >&2; exit 3; }
    {
        printf 'plan_id\t%s\n' "$CAPJOBS_PLAN_ID"
        printf 'job_id\t%s\n' "$job"
        printf 'restart_count\t%s\n' "$rst"
        printf 'node\t%s\n' "${SLURMD_NODENAME:-unknown}"
        printf 'started_at\t%s\n' "$(ts)"
    } > "$jd/batch.tsv"
    local size mtime
    size=$(stat -c %s -- "$CAPJOBS_IMAGE" 2>/dev/null) || size=""
    mtime=$(date -r "$CAPJOBS_IMAGE" +%s%N 2>/dev/null) || mtime=""
    printf 'image\t%s\nexpected_size\t%s\nmeasured_size\t%s\nexpected_mtime_ns\t%s\nmeasured_mtime_ns\t%s\n' \
        "$CAPJOBS_IMAGE" "$CAPJOBS_IMAGE_SIZE" "$size" "$CAPJOBS_IMAGE_MTIME_NS" "$mtime" \
        > "$jd/image_check.tsv"
    if [[ $size != "$CAPJOBS_IMAGE_SIZE" || $mtime != "$CAPJOBS_IMAGE_MTIME_NS" ]]; then
        echo "prep_tokenizer: image size/mtime differ from the submission" >&2
        printf '%s\tENV_INVALID\timage size or mtime differs\n' "$(ts)" >> "$jd/STATE"
        exit 6
    fi
    srun --nodes=1 --ntasks=1 --container-image="$CAPJOBS_IMAGE" \
        --container-mounts="$CAPJOBS_MOUNTS" --no-container-mount-home \
        --container-workdir="$plan" \
        bash "$plan/jobs/prep_tokenizer.sbatch" inner "$plan" "$jd" /proc/self/mountinfo
    rc=$?
    printf '%s\t%s\tinner step rc=%s\n' "$(ts)" "$([[ $rc == 0 ]] && echo DONE || echo FAILED)" \
        "$rc" >> "$jd/STATE"
    exit "$rc"
}

# ---- inner step (container) ----------------------------------------------------------------------
fail() { echo "prep_tokenizer FAIL($1): $2" >&2; exit "$1"; }

ro_mount() {  # is $1 a mount point whose mount options contain ro? (field 5 path, field 6 opts)
    local want=$1 info=$2 f1 f2 f3 f4 mp opts rest found=""
    while read -r f1 f2 f3 f4 mp opts rest; do
        [[ $mp == "$want" ]] && found=$opts
    done < "$info"
    [[ -n $found && ,$found, == *,ro,* ]]
}

only_path_visible() {  # nothing of library root $1 is visible except the path to $2
    local root=$1 target=$2 cur rest comp entries
    if [[ $target == "$root"/* ]]; then
        cur=$root
        rest=${target#"$root"/}
        while [[ -n $rest ]]; do
            comp=${rest%%/*}
            entries=$(ls -A -- "$cur") || return 1
            [[ $entries == "$comp" ]] || return 1
            cur="$cur/$comp"
            if [[ $rest == */* ]]; then rest=${rest#*/}; else rest=""; fi
        done
        return 0
    fi
    [[ ! -e $root ]] || [[ -z $(ls -A -- "$root" 2>/dev/null) ]]
}

has_weight_suffix() {
    local n=${1,,} s
    for s in "${WEIGHT[@]}"; do [[ $n == *"${s,,}" ]] && return 0; done
    return 1
}

matches_any() {  # name, globs...
    local n=$1 g
    shift
    for g in "$@"; do
        # shellcheck disable=SC2053
        [[ $n == $g ]] && return 0
    done
    return 1
}

inner() {
    local plan=$1 jd=$2 info=$3
    load_env "$plan"
    local SRC=$CAPJOBS_PREP_MODEL_DIR DEST=$CAPJOBS_PREP_DEST MAX=$CAPJOBS_PREP_MAX_FILE_BYTES
    local -a ALLOW VOCAB WEIGHT ROOTS
    read -r -a ALLOW <<< "$CAPJOBS_PREP_ALLOW_GLOBS"
    read -r -a VOCAB <<< "$CAPJOBS_PREP_VOCAB_GLOBS"
    read -r -a WEIGHT <<< "$CAPJOBS_PREP_WEIGHT_SUFFIXES"
    read -r -a ROOTS <<< "$CAPJOBS_PREP_LIBRARY_ROOTS"
    local LOG="$jd/prep.log"
    log() { printf '%s %s\n' "$(ts)" "$*" >> "$LOG"; }
    log "inner start src=$SRC dest=$DEST max_file_bytes=$MAX"
    [[ -d $SRC ]] || fail 6 "model dir $SRC is not visible"
    ro_mount "$SRC" "$info" || fail 6 "model dir $SRC is not a read-only mount ($info)"
    local root
    for root in "${ROOTS[@]}"; do
        only_path_visible "$root" "$SRC" || fail 6 "more of the model library $root is visible than $SRC"
    done
    [[ ! -e $DEST && ! -L $DEST && ! -e $DEST.partial && ! -L $DEST.partial ]] \
        || fail 3 "$DEST or $DEST.partial exists"
    [[ -d $(dirname -- "$DEST") ]] || fail 3 "destination parent of $DEST does not exist"
    mkdir -- "$DEST.partial" || fail 3 "cannot create $DEST.partial"
    local -a copied=() skipped=() refused=()
    local f n real size have_vocab=0 a b
    : > "$jd/copy_manifest.tsv"
    for f in "$SRC"/*; do
        n=${f##*/}
        if has_weight_suffix "$n"; then refused+=("$n:weight"); continue; fi
        if ! matches_any "$n" "${ALLOW[@]}"; then skipped+=("$n"); continue; fi
        if [[ -L $f ]]; then
            real=$(readlink -f -- "$f") || real=""
            [[ $real == "$SRC"/* && -f $real ]] || { refused+=("$n:link-outside"); continue; }
        fi
        [[ -f $f ]] || { skipped+=("$n:not-a-file"); continue; }
        size=$(stat -L -c %s -- "$f") || fail 4 "cannot stat $f"
        (( size <= MAX )) || fail 4 "whitelisted $n is $size bytes > limit $MAX"
        cp -L -- "$f" "$DEST.partial/$n" || fail 4 "copy of $n failed"
        a=$(sha256sum -- "$f" | cut -d' ' -f1)
        b=$(sha256sum -- "$DEST.partial/$n" | cut -d' ' -f1)
        [[ -n $a && $a == "$b" ]] || fail 4 "sha256 of the copy of $n differs from the source"
        printf '%s\t%s\t%s\n' "$n" "$size" "$b" >> "$jd/copy_manifest.tsv"
        copied+=("$n")
        matches_any "$n" "${VOCAB[@]}" && have_vocab=1
    done
    log "copied ${#copied[@]}: ${copied[*]}"
    log "skipped: ${skipped[*]}"
    log "refused: ${refused[*]}"
    (( ${#copied[@]} > 0 )) || fail 4 "no whitelisted file in $SRC"
    (( have_vocab == 1 )) || fail 4 "no tokenizer vocabulary file copied"
    cp -- "$jd/copy_manifest.tsv" "$DEST.partial/COPY_MANIFEST.tsv" || fail 4 "manifest copy failed"
    {
        printf 'source\t%s\n' "$SRC"
        printf 'plan_id\t%s\n' "$CAPJOBS_PLAN_ID"
        printf 'job_dir\t%s\n' "$jd"
        printf 'copied_at\t%s\n' "$(ts)"
        printf 'copied\t%s\n' "${copied[*]}"
        printf 'skipped\t%s\n' "${skipped[*]}"
        printf 'refused\t%s\n' "${refused[*]}"
    } > "$DEST.partial/PROVENANCE.txt"
    verify_copy "$DEST.partial" "${#copied[@]}"
    chmod -R a-w -- "$DEST.partial" || fail 4 "chmod failed"
    mv -T -- "$DEST.partial" "$DEST" || fail 4 "rename to $DEST failed"
    verify_copy "$DEST" "${#copied[@]}"
    log "OK ${#copied[@]} files -> $DEST"
    echo "OK: ${#copied[@]} files -> $DEST"
}

verify_copy() {  # dir, expected file count: only listed regular files, no weight, sums match
    local d=$1 want=$2 e n count=0 name size sum
    local -A listed=()
    while IFS=$'\t' read -r name size sum; do listed[$name]=$sum; done < "$d/COPY_MANIFEST.tsv"
    (( ${#listed[@]} == want )) || fail 4 "manifest lists ${#listed[@]} files, expected $want"
    for e in "$d"/*; do
        n=${e##*/}
        [[ -f $e && ! -L $e ]] || fail 4 "non-regular entry in the copy: $e"
        has_weight_suffix "$n" && fail 4 "weight file in the copy: $e"
        [[ $n == COPY_MANIFEST.tsv || $n == PROVENANCE.txt ]] && continue
        [[ -n ${listed[$n]+x} ]] || fail 4 "unlisted file in the copy: $n"
        (( $(stat -c %s -- "$e") <= CAPJOBS_PREP_MAX_FILE_BYTES )) || fail 4 "oversize file in the copy: $n"
        count=$((count + 1))
    done
    (( count == want )) || fail 4 "copy has $count listed files, expected $want"
    for name in "${!listed[@]}"; do printf '%s  %s\n' "${listed[$name]}" "$name"; done \
        | (cd -- "$d" && sha256sum -c --quiet -) || fail 4 "sha256sum -c of $d failed"
}

case ${1:-} in
    inner) [[ $# -eq 4 ]] || { echo "usage: prep_tokenizer.sbatch inner <plan dir> <job dir> <mountinfo>" >&2; exit 2; }
           inner "$2" "$3" "$4" ;;
    /*) [[ $# -eq 1 ]] || { echo "usage: prep_tokenizer.sbatch <plan dir>" >&2; exit 2; }
        batch "$1" ;;
    *) echo "usage: prep_tokenizer.sbatch <plan dir> | inner <plan dir> <job dir> <mountinfo>" >&2
       exit 2 ;;
esac

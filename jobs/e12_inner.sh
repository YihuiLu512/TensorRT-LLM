#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# E12-L2a launch body (capjobs script job body; design 12.6(a)(c)). Adapted from WT-21's
# wt21_inner_v4.sh: same per-step runner (tests/integration/defs/openengine_chain/runner.py over
# perf/test_perf_sanity.py::test_e2e[aggr-<case>]), same per-step deadline check and leftover
# clean-up by an inherited marker variable; differences: identity, launch tree and budget come
# from capjobs.launches (CAPKIT_*), the tensorrt_llm overlay is capjobs' (not copied here), the
# client PATH shim is tools/cape12/bin (E12_REQLOG_DIR -> cape12.reqlog_l2a), no GLM chain, and a
# preflight refuses the whole launch when steps x the registered expected step duration
# (--step-expect) exceeds 80% of the launch budget (12.6(c): a job missing any step is out of the
# main analysis, so it should not start); --step-timeout still bounds each step.
#
# E12_PYTHON (default python3) is the interpreter for the CUDA probe and the runner (tests only).
# usage: e12_inner.sh --src DIR --src-head SHA --config-dir DIR --gpus N --step-timeout S --step-expect S \
#            --steps "ARM=CASE ARM=CASE ..."
#   ARM in A-def A-1f0 B-s1 B-full; CASE = test_e2e[aggr-CASE] (the YAML stem and server name).
# Exit: 0 every step ran (each step's own rc is in its step.txt); 6 GPU count / CUDA probe wrong
# (ENV_INVALID); 3 refused before any step (bad arguments, source tree, preflight); 4 the node
# was left contaminated by a step (later steps NOT_RUN).
set -uo pipefail
SRC= SRC_HEAD= CFG= GPUS= STEP_TIMEOUT= STEP_EXPECT= STEPS=
while [[ $# -gt 0 ]]; do
    case $1 in
        --src) SRC=$2;; --src-head) SRC_HEAD=$2;; --config-dir) CFG=$2;; --gpus) GPUS=$2;;
        --step-timeout) STEP_TIMEOUT=$2;; --step-expect) STEP_EXPECT=$2;; --steps) STEPS=$2;;
        *) echo "e12_inner: unknown argument $1" >&2; exit 3;;
    esac
    shift 2
done
export GIT_OPTIONAL_LOCKS=0
OUT=${CAPKIT_RUN_DIR:?}/raw
mkdir -p "$OUT"
refuse() { echo "REFUSED: $*" | tee "$OUT/REFUSED"; exit 3; }
for v in SRC SRC_HEAD CFG GPUS STEP_TIMEOUT STEP_EXPECT STEPS; do [[ -n ${!v} ]] || refuse "missing --${v,,}"; done
[[ $(git -C "$SRC" rev-parse HEAD 2>/dev/null) == "$SRC_HEAD" ]] || refuse "source HEAD differs"
[[ -z $(git -C "$SRC" status --porcelain --untracked-files=no) ]] || refuse "source tree modified"
RUNNER=$SRC/tests/integration/defs/openengine_chain/runner.py
[[ -f $RUNNER && -d $CFG ]] || refuse "runner or config dir missing"
read -r -a STEP_LIST <<< "$STEPS"
T0=$(date +%s)
BUDGET=${CAPKIT_LAUNCH_BUDGET_S:?}
DEADLINE=$(( T0 + BUDGET - 60 ))
NEED=$(( ${#STEP_LIST[@]} * STEP_EXPECT ))
(( NEED * 10 <= BUDGET * 8 )) || refuse "steps need $NEED s > 80% of the launch budget $BUDGET s"
for step in "${STEP_LIST[@]}"; do
    case ${step%%=*} in A-def|A-1f0|B-s1|B-full) ;; *) refuse "bad arm in step $step";; esac
    [[ $step == *=?* ]] || refuse "bad step $step"
done
PY=${E12_PYTHON:-python3}
{ echo "experiment=${CAPKIT_EXPERIMENT:-} arm=${CAPKIT_ARM:-} job=${CAPKIT_JOB_ID:-}"
  echo "src=$SRC head=$SRC_HEAD config_dir=$CFG gpus=$GPUS step_timeout=$STEP_TIMEOUT"
  echo "steps=$STEPS"; echo "start=$(date -Is) budget_s=$BUDGET"; } > "$OUT/inner.txt"
unset PYTHONPATH TRTLLM_TEST_OPENENGINE_CHAIN TRTLLM_OPENENGINE_RESULTS_DIR TRTLLM_PERF_ARM
export TZ=UTC TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 LLM_ROOT=$SRC AGG_CONFIG_FOLDER=$CFG
export LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models LLM_TEST_PARALLEL_OVERRIDE=1
export HOME=${TMPDIR:-/tmp}/e12-home-${CAPKIT_JOB_ID:-x} PYTEST_ADDOPTS='-p no:cacheprovider'
mkdir -p "$HOME"
# The PATH shim runs from a run-local executable copy: the job's code snapshot is read-only and
# without exec bits (capjobs.snapshot), and a non-executable `python` on PATH is skipped silently,
# so the client would run without reqlog (E12 smoke 6148825). The copy keeps the shim's layout
# (bin/python next to ../reqlog_l2a.py); `command -v python` must resolve to it.
CAPE12="$(cd "$(dirname "${BASH_SOURCE[0]}")/../tools/cape12" && pwd)" || refuse "tools/cape12 missing"
SHIM=$CAPKIT_RUN_DIR/e12shim
mkdir -p "$SHIM/bin" && cp "$CAPE12/bin/python" "$SHIM/bin/python" \
    && cp "$CAPE12/reqlog_l2a.py" "$SHIM/reqlog_l2a.py" && chmod 0755 "$SHIM/bin/python" \
    || refuse "cannot set up the PATH shim"
export PATH="$SHIM/bin:$PATH"
[[ $(command -v python) == "$SHIM/bin/python" ]] || refuse "PATH shim not first: $(command -v python)"
"$PY" - "$GPUS" > "$OUT/cuda_probe.txt" 2>&1 <<'PY' || { echo "ENV_INVALID: CUDA probe"; exit 6; }
import sys, torch
n = torch.cuda.device_count(); print('CUDA_DEVICE_COUNT', n, flush=True)
assert n == int(sys.argv[1]), 'allocated GPU count differs'
for i in range(n):
    assert 'B200' in torch.cuda.get_device_name(i)
PY

marked() {  # pids whose environment carries E12_MARK=$1 (inherited by everything a step starts)
    local p
    for p in /proc/[0-9]*; do
        tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -qx "E12_MARK=$1" && echo "${p#/proc/}"
    done
}
gpu_apps() { "${CAPKIT_NVIDIA_SMI:-nvidia-smi}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null; }
clean_leftovers() {  # wait for GPU memory to drain, then TERM / KILL what carries the marker
    local mark=$1 sd=$2 i left
    for i in $(seq 1 24); do [[ -z $(gpu_apps) ]] && break; sleep 5; done
    left=$(marked "$mark"); echo "$left" > "$sd/leftovers-before.txt"
    [[ -z $left && -z $(gpu_apps) ]] && return 0
    # shellcheck disable=SC2086
    kill -TERM $left 2>/dev/null; sleep 20
    left=$(marked "$mark")
    # shellcheck disable=SC2086
    [[ -n $left ]] && kill -KILL $left 2>/dev/null
    sleep 10
    left=$(marked "$mark"); echo "$left" > "$sd/leftovers-after.txt"
    [[ -z $left && -z $(gpu_apps) ]] && return 0
    echo "after step $sd: $(date -Is)" > "$OUT/CONTAMINATED"
    return 1
}

i=0
for step in "${STEP_LIST[@]}"; do
    i=$((i + 1))
    arm=${step%%=*}; case_=${step#*=}
    case $arm in A-def|A-1f0) chain=none;; *) chain=trtllm;; esac
    sd=$OUT/$(printf '%02d' "$i")_${arm}__${case_}
    mkdir -p "$sd"
    if [[ -e $OUT/CONTAMINATED ]]; then echo "NOT_RUN: node contaminated" > "$sd/NOT_RUN"; continue; fi
    now=$(date +%s)
    if (( now + STEP_TIMEOUT + 150 > DEADLINE )); then
        echo "NOT_RUN: $(( DEADLINE - now )) s left < step $STEP_TIMEOUT s + 150 s" > "$sd/NOT_RUN"; continue
    fi
    echo "perf/test_perf_sanity.py::test_e2e[aggr-$case_]" > "$sd/list.txt"
    export TRTLLM_PERF_ARM=$arm TRTLLM_PERF_RUN=$i E12_REQLOG_DIR=$sd/reqlog
    mark="${CAPKIT_JOB_ID:-x}-$i-$RANDOM$RANDOM"
    echo "=== STEP $i $arm $case_ chain=$chain start $(date -Is)" | tee "$sd/step.txt"
    t0=$(date +%s.%N)
    E12_MARK=$mark timeout --signal=TERM --kill-after=60 "$STEP_TIMEOUT" "$PY" "$RUNNER" \
        --chain "$chain" --out "$sd/run" --list "$sd/list.txt" --timeout $((STEP_TIMEOUT - 120)) \
        > "$sd/runner.log" 2>&1
    rc=$?
    echo "rc=$rc t0=$t0 t1=$(date +%s.%N) end=$(date -Is) reqlog_files=$(ls "$sd/reqlog" 2>/dev/null | wc -l)" \
        | tee -a "$sd/step.txt"
    clean_leftovers "$mark" "$sd"
done
echo "=== INNER DONE $(date -Is)" | tee -a "$OUT/inner.txt"
[[ -e $OUT/CONTAMINATED ]] && exit 4
exit 0

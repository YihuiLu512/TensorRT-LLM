#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# WT-31 (R10): inside the writable Pyxis container of one node of a shard. Overlays the Python
# files of SRC on the image, checks the GPUs, records the transport facts the KV-path conclusion
# needs, runs each entry of the shard list through its own runner.py --chain none (no chain switch,
# no waives file, so waived tests run too), and records, then ends, the processes each entry left
# behind. The nodes of a 2-node shard go through the list in step (files under $RUN/sync): node 0
# waits up to RK_WAIT seconds for node 1 before each entry, node 1 waits for node 0's decision until
# 60 s past the deadline; either stops starting entries once the other node has exited.
# Exit codes: 64 bad input, 65 NOT RUN, 66 incomplete probe, 67 some entries not started (all
# started ones returned 0), 73 no result directory, 143 terminated; otherwise the first non-zero
# code of an entry: runner.py's, or 124 / 137 when timeout(1) ended the entry.
set -uo pipefail
export TZ=Asia/Shanghai GIT_OPTIONAL_LOCKS=0
for var in SRC HEAD_COMMIT RUN GPU_COUNT EXPECT_GPU MODE TEST_TIMEOUT NNODES DRIVER_DIR DEADLINE \
    LLM_MODELS_ROOT; do
    [[ -n ${!var:-} ]] || { printf 'NOT RUN: missing %s\n' "$var"; exit 64; }
done
NODE_ID=${SLURM_NODEID:-0}
NODE_DIR="$RUN/node$NODE_ID"
SYNC="$RUN/sync"
RK_WAIT=7200  # longest wait of node 0 for node 1 in a 2-node shard (RK-19)
CURRENT=''    # the entry whose runner is running
mkdir -p "$NODE_DIR" "$SYNC" || exit 73
# tee ignores TERM, so that a TERM sent to every process of the step or job does not close the
# log before finish() writes the exit status. srun may instead end the step's tasks with KILL
# (at its time limit), and then neither trap runs.
exec > >(trap '' TERM; exec tee "$NODE_DIR/container.log") 2>&1
finish() {
    local code=$?
    trap - EXIT
    printf '%s\n' "$code" > "$SYNC/exit.node$NODE_ID"  # tells the other node to stop waiting
    printf '%s\n' "$code" > "$NODE_DIR/exit-code.txt"
    printf 'NODE_END %s node=%s rc=%s\n' "$(date -Is)" "$NODE_ID" "$code"
    exit "$code"
}
on_term() {
    # Bash runs this once the foreground runner has ended (it gets the TERM as well).
    trap - TERM
    if [[ -n $CURRENT ]] && declare -F reap > /dev/null; then
        printf '%s\t%s\n' "$CURRENT" "$(reap "$CURRENT")" > "$NODE_DIR/reap-on-term.tsv"
    fi
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
        > "$NODE_DIR/gpu-apps-at-end.txt" 2>&1
    exit 143
}
trap finish EXIT
trap on_term TERM
fail() { printf 'NOT RUN: %s\n' "$*"; exit 65; }
printf 'NODE_START %s host=%s node=%s\n' "$(date -Is)" "$(hostname)" "$NODE_ID"
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1
export LLM_ROOT="$SRC" LLM_MODELS_ROOT
# One attempt per test: the wrappers would otherwise rerun failures in parallel mode.
export LLM_TEST_PARALLEL_OVERRIDE=1
# Every process started below inherits R10_RUN_TAG (set per entry below); the leftover check looks
# for it in /proc/*/environ, because the container shares the node's PID namespace with other jobs.
export R10_RUN_TAG="$NODE_DIR/setup"
unset PYTHONPATH PYTEST_ADDOPTS TRTLLM_TEST_OPENENGINE_CHAIN TRTLLM_TEST_OPENENGINE_TOPOLOGY \
    TRTLLM_OPENENGINE_RESULTS_DIR
export PYTEST_ADDOPTS='-p no:cacheprovider'
[[ -r $SRC/tensorrt_llm/version.py && -r $RUN/overlay.txt && -r $RUN/selected.txt ]] || fail 'inputs unavailable'
[[ -d $LLM_MODELS_ROOT/datasets ]] || fail 'model root lacks datasets'
cat /opt/trtllm-env/ENV_READY || fail 'ready marker missing'
/usr/bin/python3 -I - "$SRC" "$RUN/overlay.txt" <<'PYOVERLAY'
import hashlib
import shutil
import sys
import sysconfig
from pathlib import Path

source = Path(sys.argv[1])
installed = Path(sysconfig.get_paths()["purelib"])
paths = Path(sys.argv[2]).read_text().splitlines()
for value in paths:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "tensorrt_llm" \
            or relative.suffix != ".py":
        raise SystemExit("invalid overlay path: " + value)
    target = installed / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / relative, target)
    if target.read_bytes() != (source / relative).read_bytes():
        raise SystemExit("overlay content differs: " + value)
    print("OVERLAID", value, hashlib.sha256(target.read_bytes()).hexdigest()[:16], flush=True)
print("OVERLAY_COUNT", len(paths), flush=True)
PYOVERLAY
[[ $? == 0 ]] || fail 'package overlay failed'
/usr/bin/python3 -I - "$GPU_COUNT" "$EXPECT_GPU" <<'PYPROBE'
import re
import sys

import torch

expected, pattern = int(sys.argv[1]), sys.argv[2]
actual = torch.cuda.device_count()
print("CUDA_DEVICE_COUNT", actual, "EXPECTED", expected, flush=True)
if actual != expected:
    raise SystemExit("allocated GPU count differs")
for index in range(actual):
    name = torch.cuda.get_device_name(index)
    free, total = torch.cuda.mem_get_info(index)
    print("GPU", index, name, "total_mib", total // 2**20, "free_mib", free // 2**20, flush=True)
    if not re.search(pattern, name):
        raise SystemExit("unexpected GPU " + name)
    value = torch.ones(1, device=f"cuda:{index}")
    torch.cuda.synchronize(index)
    if value.item() != 1:
        raise SystemExit("CUDA probe failed")
PYPROBE
[[ $? == 0 ]] || fail 'CUDA probe failed'
# Facts for the KV transfer path conclusion (DQ-24 / UV-08 / UV-75): topology, NICs, UCX transports.
{
    nvidia-smi topo -m
    echo '--- /sys/class/net speed'
    for n in /sys/class/net/*; do printf '%s %s\n' "${n##*/}" "$(cat "$n/speed" 2>/dev/null || echo na)"; done
    echo '--- ibstat'
    command -v ibstat >/dev/null && ibstat 2>&1 | grep -E "^CA '|State:|Rate:|Link layer" || echo 'ibstat unavailable'
    echo '--- ucx_info -d (transports)'
    command -v ucx_info >/dev/null && ucx_info -d 2>&1 | grep -E '^#\s+(Transport|Device):' || echo 'ucx_info unavailable'
    echo '--- UCX/NIXL env'
    env | grep -E '^(UCX_|NIXL_|TRTLLM_.*KV|TLLM_.*KV)' | sort
} > "$NODE_DIR/transport-facts.txt" 2>&1
if [[ $MODE == preflight ]]; then
    printf 'PREFLIGHT_COMPLETED %s; tests NOT RUN\n' "$(date -Is)"
    exit 0
fi
leftovers() {  # <output file>: "pid tag cmdline" of live processes tagged with this node's run
    local pid tag
    for pid in $(ls /proc | grep -E '^[0-9]+$'); do
        [[ $pid == "$$" || $pid == "$BASHPID" ]] && continue
        tag=$({ tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null | grep -m1 '^R10_RUN_TAG=') || continue
        tag=${tag#R10_RUN_TAG=}
        [[ $tag == "$NODE_DIR/"* ]] || continue
        printf '%s %s %s\n' "$pid" "${tag#"$NODE_DIR/"}" \
            "$({ tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null | cut -c1-300)"
    done > "$1"
}
reap() {  # <name>: record the leftovers, TERM them, KILL what is left 10 s later; prints counts
    local found survived
    leftovers "$NODE_DIR/leftover-$1.txt"
    found=$(grep -c . < "$NODE_DIR/leftover-$1.txt")
    if (( found > 0 )); then
        kill -TERM $(cut -d' ' -f1 "$NODE_DIR/leftover-$1.txt") 2>/dev/null
        sleep 10
        leftovers "$NODE_DIR/leftover-$1.kill.txt"  # listed again, so that a reused pid is not hit
        [[ -s $NODE_DIR/leftover-$1.kill.txt ]] && kill -KILL $(cut -d' ' -f1 "$NODE_DIR/leftover-$1.kill.txt") 2>/dev/null
        sleep 2
    fi
    leftovers "$NODE_DIR/leftover-$1.after.txt"
    survived=$(grep -c . < "$NODE_DIR/leftover-$1.after.txt")
    printf '%s\t%s\n' "$found" "$survived"
}
if [[ $MODE == probe ]]; then
    # KV paths and the three behaviors of the control form T (r10_probe.py); no shard list.
    cd "$SRC/tests/integration/defs" || fail 'defs directory'
    start=$(date +%s)
    R10_RUN_TAG="$NODE_DIR/probe" /usr/bin/python3 "$DRIVER_DIR/r10_probe.py" --out "$NODE_DIR/probe" \
        --hook-dir "$DRIVER_DIR/hook" > "$NODE_DIR/probe.log" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start ))
    # Not in a pipeline: the other commands of a pipeline carry this node's tag and would be reaped.
    counts=$(reap probe)
    printf 'PROBE_FINISHED rc=%s elapsed=%ss leftovers(found, after kill)=%s\n' "$rc" "$elapsed" \
        "${counts/$'\t'/,}"
    [[ -s $NODE_DIR/probe/probe.json ]] || { echo 'INCOMPLETE: no probe.json'; exit 66; }
    exit "$rc"
fi
# One runner.py (one pytest) per entry: a per-test timeout ends its pytest with os._exit
# (--timeout-method=thread) before any junit file is written, which would lose the whole list.
# Each entry's timeout is its list bound (TIMEOUT mark, else TEST_TIMEOUT) cut to the time left
# before DEADLINE. runner.py gets it as --timeout; a test with its own @pytest.mark.timeout
# overrides that, so the runner also runs under timeout(1) 60 s past it. Entries thus end before
# the container step does (srun ends the step's tasks with KILL at its time limit, so no trap runs).
# entries.tsv per entry: name, entry, start, elapsed s, runner rc, timeout s, leftover processes
# found, leftovers still alive after TERM/KILL; start, elapsed, rc and timeout are "-" for an entry
# not started (job budget, or the other node of a 2-node shard stopped or did not get there).
wait_for() {  # <limit epoch> <file>...: 0 when one of the files exists; 1 at the limit or when
    local limit=$1 f   # the other node has exited
    shift
    while (( $(date +%s) < limit )); do
        for f in "$@"; do [[ -e $f ]] && return 0; done
        [[ -e $SYNC/exit.node$(( 1 - NODE_ID )) ]] && return 1
        sleep 5
    done
    return 1
}
STOPPED=0
decide() {  # <index> <name>: 0 to run the entry; node 0 decides, node 1 follows
    local prev limit
    (( STOPPED )) && return 1
    prev=$(printf 'e%02d' "$(( $1 - 1 ))")
    limit=$(( $(date +%s) + RK_WAIT ))
    (( limit < DEADLINE )) || limit=$DEADLINE
    if [[ $NODE_ID == 0 ]]; then
        local need=''
        # Node 1 must be set up before the first entry and done with the previous one after that.
        if (( NNODES > 1 )); then
            if (( $1 == 0 )); then need="$SYNC/ready.node1"; else need="$SYNC/$prev.done.node1"; fi
            if ! wait_for "$limit" "$need"; then
                STOPPED=1
                printf 'STOP node 1 exited or did not reach %s\n' "${need##*/}"
            fi
        fi
        if (( ! STOPPED && $(date +%s) <= DEADLINE - 600 )); then : > "$SYNC/$2.go"; return 0; fi
        : > "$SYNC/$2.skip"
        return 1
    fi
    if ! wait_for "$(( DEADLINE + 60 ))" "$SYNC/$2.go" "$SYNC/$2.skip"; then
        STOPPED=1
        printf 'STOP node 0 exited before deciding %s\n' "$2"
        return 1
    fi
    [[ -e $SYNC/$2.go ]]
}
if [[ ${CLEAN_SLURM_ENV:-0} == 1 ]]; then
    # For tests that start their own mpirun: with the step's variables it takes its slots from the
    # one-task step. Single-node shards only (multi-node tests read SLURM_NODEID and the node list);
    # NODE_ID was read above.
    (( NNODES == 1 )) || { printf 'NOT RUN: CLEAN_SLURM_ENV needs a single-node shard\n'; exit 64; }
    for var in $(compgen -e); do case $var in SLURM_*|PMIX*|PMI_*) unset "$var" ;; esac; done
    printf 'CLEAN_SLURM_ENV: Slurm and PMI variables removed for the entries\n'
fi
mapfile -t ENTRIES < <(grep -v '^[[:space:]]*$' "$RUN/selected.txt")
: > "$NODE_DIR/entries.tsv"
: > "$SYNC/ready.node$NODE_ID"
rc=0 skipped=0
for i in "${!ENTRIES[@]}"; do
    entry=${ENTRIES[$i]} name=$(printf 'e%02d' "$i")
    if ! decide "$i" "$name"; then
        printf '%s\t%s\t-\t-\t-\t-\t0\t0\n' "$name" "$entry" >> "$NODE_DIR/entries.tsv"
        printf 'NOT_STARTED %s %s\n' "$name" "$entry"
        skipped=$(( skipped + 1 ))
        [[ $NODE_ID != 0 ]] && : > "$SYNC/$name.done.node$NODE_ID"
        continue
    fi
    base=${entry%% TIMEOUT *}
    bound=$TEST_TIMEOUT
    [[ $entry =~ \ TIMEOUT\ \(?([0-9]+)\)? ]] && bound=$(( BASH_REMATCH[1] * 60 ))
    left=$(( DEADLINE - $(date +%s) - 120 ))
    timeout_s=$(( bound < left ? bound : left ))
    # Marks this job as the one whose results count for the shard (summarize_r10.py); written at
    # the first entry started, so that a rerun that starts none does not hide an earlier job.
    [[ -e $NODE_DIR/test-started.txt ]] || printf '%s\n' "$(date -Is)" > "$NODE_DIR/test-started.txt"
    printf '%s\n' "$base" > "$NODE_DIR/list-$name.txt"
    extra=("--workspace=$NODE_DIR/workspace/$name")
    # The CI Ray stages run test_ray.py with --run-ray (Ray instead of MPI, local Ray cluster).
    [[ $entry == examples/test_ray.py::* ]] && extra+=(--run-ray)
    RUNNER=(/usr/bin/python3 "$SRC/tests/integration/defs/openengine_chain/runner.py"
        --chain none --out "$NODE_DIR/results/$name" --list "$NODE_DIR/list-$name.txt"
        --timeout "$timeout_s" -- "${extra[@]}")
    printf '%q ' "${RUNNER[@]}" >> "$NODE_DIR/command.txt"; echo >> "$NODE_DIR/command.txt"
    started=$(date -Is) start=$(date +%s) CURRENT=$name
    R10_RUN_TAG="$NODE_DIR/$name" timeout --signal=TERM --kill-after=30s "$(( timeout_s + 60 ))s" \
        "${RUNNER[@]}" > "$NODE_DIR/runner-$name.log" 2>&1
    entry_rc=$?
    elapsed=$(( $(date +%s) - start ))
    # Servers or MPI processes the entry left behind would hold GPUs and ports for the next one.
    counts=$(reap "$name")
    CURRENT=''
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$entry" "$started" "$elapsed" "$entry_rc" \
        "$timeout_s" "$counts" >> "$NODE_DIR/entries.tsv"
    printf 'ENTRY %s rc=%s elapsed=%ss timeout=%ss leftovers(found, after kill)=%s %s\n' "$name" \
        "$entry_rc" "$elapsed" "$timeout_s" "${counts/$'\t'/,}" "$entry"
    (( rc != 0 )) || rc=$entry_rc
    [[ $NODE_ID != 0 ]] && : > "$SYNC/$name.done.node$NODE_ID"
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "$NODE_DIR/gpu-apps-at-end.txt" 2>&1
printf 'RUNNER_FINISHED rc=%s entries=%s not_started=%s\n' "$rc" "${#ENTRIES[@]}" "$skipped"
(( rc == 0 && skipped > 0 )) && exit 67
exit "$rc"

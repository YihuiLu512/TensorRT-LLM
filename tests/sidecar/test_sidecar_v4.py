# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: sidecar v4 + canary against spawned busy / idle processes registered in
roles.json. Functional only (no performance numbers are results)."""

import json
import os
import signal
import subprocess
import time

import pytest

from b4_sidecar_util import (IDLE_SRC, PY, SHA, THREADS_SRC, capenv_env, id_args, kill_all,
                             make_ident, spawn_py, wait_for)
from capcommon.jsonio import read_json, read_jsonl, write_json_stamped
from capcommon.runtree import RunTree
from capcommon.snapshot import SlotRegistry, SnapshotWriter
from capenv import procfs

ROLES = "capkit.roles/1"


def _cpus(n):
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < n + 1:
        pytest.skip("needs %d cpus" % (n + 1))
    return cpus[-n:]


def _write_roles(tree, gen, procs, unattributed=()):
    write_json_stamped(tree.roles_json, ROLES, make_ident("launch"),
                       {"generation": gen, "procs": procs, "unattributed": list(unattributed)})


def _entry(pid, role, start=None):
    return {"pid": pid, "starttime_ticks": procfs.starttime_ticks(pid) if start is None else start,
            "role": role, "worker": 0, "parent_pid": os.getpid(), "cmdline": [],
            "cpus": [], "discovered_via": "launched", "registered_mono_ns": time.monotonic_ns()}


def _by_tick(rows):
    out = {}
    for r in rows:
        out.setdefault(r[0], []).append(r)
    return out


def test_sidecar_and_canary_end_to_end(tmp_path):
    c_busy, c_idle, c_canary, c_side = _cpus(4)
    tree = RunTree(str(tmp_path / "T")).make_skeleton()
    shm = str(tmp_path / "shm")
    os.makedirs(shm)
    busy = spawn_py(THREADS_SRC, [c_busy])
    idle = spawn_py(IDLE_SRC, [c_idle])
    # an unregistered spinner whose command line contains the v3 match string: must NOT appear
    decoy = spawn_py("# trtllm-serve glm5-serve\nwhile True:\n    pass\n", [c_idle])
    procs = [busy, idle, decoy]
    side = can = None
    try:
        origin = time.monotonic_ns() + 300_000_000
        env = capenv_env()
        can = subprocess.Popen(
            [PY, "-m", "capenv.canary"] + id_args("canary") + [
                "--run-dir", tree.root, "--cpus", str(c_canary), "--stop-grace-s", "1",
                "--origin-ns", str(origin), "--interval-ms", "200",
                "--out", tree.canary("canary.jsonl"), "--shm-dir", shm,
                "--ready-file", tree.ready_file("canary")], env=env)
        procs.append(can)
        assert wait_for(lambda: os.path.exists(tree.ready_file("canary")), 20)
        # thread inventory for the busy process lists only its main thread -> the spin thread
        # must be reported as not in the inventory
        write_json_stamped(tree.threads_file("busy", busy.pid), "capkit.threads/1",
                           make_ident("launch"),
                           {"pid": busy.pid, "taken_mono_ns": time.monotonic_ns(),
                            "threads": [{"tid": busy.pid, "comm": "python", "py_name": None,
                                         "class": "python", "identified_as": "main",
                                         "source": "comm"}]})
        # idle registered with a wrong starttime -> pid-reuse detection
        _write_roles(tree, 1, [_entry(busy.pid, "busy"), _entry(can.pid, "canary"),
                               _entry(idle.pid, "idle", start=1)])
        # a snapshot from an unregistered pid must be reported, not sampled
        reg = SlotRegistry()
        reg.counter("count.X1.msgs")
        SnapshotWriter(reg, shm, 100_000_000, pid=4_000_000).write_now()
        with open(tree.log_file("busy", 0, busy.pid, "out"), "w") as f:
            f.write("x" * 10)
        side = subprocess.Popen(
            [PY, "-m", "capenv.sidecar"] + id_args("sidecar") + [
                "--run-dir", tree.root, "--cpus", str(c_side), "--stop-grace-s", "1",
                "--shm-dir", shm, "--origin-ns", str(origin), "--roles-file", tree.roles_json,
                "--interval-ms", "200", "--out-dir", tree.sidecar(), "--cpus-watch", "all",
                "--gpu-every-s", "0.5", "--numa-every-s", "0.5", "--logs-dir",
                os.path.join(tree.raw, "logs"), "--ready-file", tree.ready_file("sidecar"),
                "--gpu-timeout-s", "5"], env=env)
        procs.append(side)
        assert wait_for(lambda: os.path.exists(tree.ready_file("sidecar")), 20)
        time.sleep(1.0)
        # generation 2: fix idle's registration; sidecar must reload
        _write_roles(tree, 2, [_entry(busy.pid, "busy"), _entry(can.pid, "canary"),
                               _entry(idle.pid, "idle2")])
        with open(tree.log_file("busy", 0, busy.pid, "out"), "a") as f:
            f.write("y" * 20)
        time.sleep(1.4)
        side.send_signal(signal.SIGTERM)
        can.send_signal(signal.SIGTERM)
        assert side.wait(timeout=30) == 0
        assert can.wait(timeout=30) == 0
    finally:
        kill_all(procs)

    hdr, th = read_jsonl(tree.sidecar("threads.jsonl"), "capkit.sidecar.threads/1")
    assert hdr["identity"]["role"] == "sidecar" and hdr["identity"]["register_sha"] == SHA
    pids = {r[1] for r in th}
    assert decoy.pid not in pids  # never matched by command line
    assert busy.pid in pids and can.pid in pids
    busy_rows = [r for r in th if r[1] == busy.pid]
    tids = {r[2] for r in busy_rows}
    assert len(tids) >= 2
    spin_tid = max(tids, key=lambda t: max(r[4] for r in busy_rows if r[2] == t))
    spin = [r for r in busy_rows if r[2] == spin_tid]
    assert spin[-1][4] - spin[0][4] >= 50  # >= 0.5 s of user CPU at 100 Hz over ~2 s
    assert all(r[10] == str(c_busy) for r in busy_rows)  # Cpus_allowed_list every row
    assert spin[-1][9] == c_busy  # last-run CPU
    assert all(r[6] is not None for r in busy_rows)  # schedstat readable on this kernel
    assert all(r[7] is not None and r[8] is not None for r in busy_rows)
    # idle was first registered with a wrong starttime -> never sampled under that entry
    meta = read_json(tree.sidecar("meta.json"))
    assert meta["final"] is True and meta["stats"]["ticks"] >= 8
    gone = [g for g in meta["pids_gone"] if g["pid"] == idle.pid]
    assert len(gone) == 1 and "pid reused" in gone[0]["why"] and gone[0]["role"] == "idle"
    assert [g["generation"] for g in meta["roles_generations"]] == [1, 2]
    assert 4_000_000 in meta["unregistered_snapshot_pids"]
    # cpus.jsonl: every online cpu on every tick, same t_rel_ns as the thread rows
    _h, cp = read_jsonl(tree.sidecar("cpus.jsonl"), "capkit.sidecar.cpus/1")
    online = procfs.online_cpus()
    ticks = _by_tick(cp)
    assert all(sorted(r[1] for r in rows) == online for rows in ticks.values())
    assert {r[0] for r in th} <= set(ticks)
    # absolute schedule: tick times are origin + k * 200 ms plus a small lateness
    assert all((t % 200_000_000) < 150_000_000 for t in ticks)
    # procs.jsonl role comes from the registry only
    _h, pr = read_jsonl(tree.sidecar("procs.jsonl"), "capkit.sidecar.procs/1")
    assert {(r[1], r[2]) for r in pr} >= {(busy.pid, "busy"), (can.pid, "canary"),
                                          (idle.pid, "idle2")}
    # snapshots: the canary's counters land on the same ticks
    _h, sn = read_jsonl(tree.sidecar("snapshots.jsonl"), "capkit.sidecar.snapshots/1")
    can_sn = [r for r in sn if r[1] == can.pid]
    assert len(can_sn) >= 5
    its = [r[4]["canary.loop.iters"] for r in can_sn]
    assert its == sorted(its) and its[-1] > its[0]
    assert all(r[0] in ticks for r in can_sn)
    # tids: spin thread appeared and is not in the inventory; main thread is
    _h, td = read_jsonl(tree.sidecar("tids.jsonl"), "capkit.sidecar.tids/1")
    ev = {(r[1], r[3]): r for r in td if r[4] == "appear"}
    assert ev[(busy.pid, busy.pid)][6] is True and ev[(busy.pid, busy.pid)][7] == "main"
    assert ev[(busy.pid, spin_tid)][6] is False
    inv = {(r[1], r[3]): r for r in td if r[4] == "inventory"}
    assert inv[(busy.pid, busy.pid)][6] is True and inv[(busy.pid, spin_tid)][6] is False
    assert not any(r[1] == can.pid and r[4] == "inventory" for r in td)  # no inventory file
    # gpu: no nvidia-smi here -> positive control fails, header says so, rows carry query_rc
    gh, gr = read_jsonl(tree.sidecar("gpu.jsonl"), "capkit.sidecar.gpu/1")
    assert gh["gpu_count"] == 0 and gh["positive_control_ok"] is False
    assert gr and all(r[5] != 0 for r in gr)
    assert "gpu" in meta["unreadable"]
    _h, nu = read_jsonl(tree.sidecar("numa.jsonl"), "capkit.sidecar.numa/1")
    assert any(r[1] == busy.pid and isinstance(r[2], dict) for r in nu)
    _h, lg = read_jsonl(tree.sidecar("logs.jsonl"), "capkit.sidecar.logs/1")
    sizes = [r[2] for r in lg]
    assert 10 in sizes and 30 in sizes
    # canary output
    chd, cr = read_jsonl(tree.canary("canary.jsonl"), "capkit.canary/1")
    assert chd["identity"]["role"] == "canary" and len(cr) >= 5
    assert all(r[1] > 0 and r[3] == c_canary for r in cr)
    assert all(r[2] > 0 for r in cr[1:])
    cm = read_json(tree.canary("canary.meta.json"))
    assert cm["unit_ops"] == cm["unit_ops_default"]


def test_sidecar_rejects_bad_args(tmp_path):
    tree = RunTree(str(tmp_path / "T")).make_skeleton()
    base = [PY, "-m", "capenv.sidecar"] + id_args("sidecar") + [
        "--run-dir", tree.root, "--cpus", str(sorted(os.sched_getaffinity(0))[0]),
        "--stop-grace-s", "1", "--shm-dir", str(tmp_path), "--origin-ns", "0",
        "--roles-file", tree.roles_json, "--interval-ms", "200", "--out-dir", tree.sidecar(),
        "--gpu-every-s", "0", "--numa-every-s", "0", "--logs-dir", str(tmp_path)]
    # pin table needs cpu 99 but watch list is 0 -> exit 3
    write_json_stamped(str(tmp_path / "pins.json"), "capkit.pin_table/1", make_ident("pins"),
                       {"roles": {"a": {"cpus": [99], "numa_node": 0}},
                        "siblings_left_idle": []})
    r = subprocess.run(base + ["--cpus-watch", "0", "--pin-table", str(tmp_path / "pins.json")],
                       env=capenv_env(), capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "does not cover" in r.stderr
    r = subprocess.run(base[:-2], env=capenv_env(), capture_output=True, text=True, timeout=60)
    assert r.returncode == 2  # missing required --logs-dir / --cpus-watch


def test_canary_requires_exactly_one_cpu(tmp_path):
    cpus = sorted(os.sched_getaffinity(0))
    r = subprocess.run([PY, "-m", "capenv.canary"] + id_args("canary") + [
        "--run-dir", str(tmp_path), "--cpus", "%d,%d" % (cpus[0], cpus[1]), "--stop-grace-s",
        "1", "--origin-ns", "0", "--interval-ms", "100", "--out", str(tmp_path / "c.jsonl")],
        env=capenv_env(), capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "exactly one" in r.stderr


def test_canary_rate_drops_when_core_shared(tmp_path):
    """Positive control for the canary as a frequency/contention reading: a spinner on the same
    core must lower the per-interval iteration count (functional check, not a measurement)."""
    (cpu,) = _cpus(1)
    env = capenv_env()

    def run_canary(out):
        origin = time.monotonic_ns()
        return subprocess.run([PY, "-m", "capenv.canary"] + id_args("canary") + [
            "--run-dir", str(tmp_path), "--cpus", str(cpu), "--stop-grace-s", "1",
            "--origin-ns", str(origin), "--interval-ms", "200", "--out", out,
            "--max-rows", "6"], env=env, capture_output=True, text=True, timeout=60)

    a = str(tmp_path / "alone.jsonl")
    assert run_canary(a).returncode == 0
    spinner = spawn_py("while True:\n    pass\n", [cpu])
    try:
        b = str(tmp_path / "shared.jsonl")
        assert run_canary(b).returncode == 0
    finally:
        kill_all([spinner])
    ra = [r[1] for r in read_jsonl(a)[1][1:]]
    rb = [r[1] for r in read_jsonl(b)[1][1:]]
    assert sum(rb) / len(rb) < 0.8 * (sum(ra) / len(ra)), (ra, rb)
    json.dumps([ra, rb])

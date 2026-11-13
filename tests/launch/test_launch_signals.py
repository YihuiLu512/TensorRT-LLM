# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launches that are signalled or stalled from outside (stop signals, a stopped orchestrator)."""

import json
import os
import signal
import time

import conftest as cf
from capcommon.runtree import read_state
from caplaunch import procfs

MS = 1_000_000


def _j(*p):
    with open(os.path.join(*p)) as f:
        return json.load(f)


def _wait_file(path, proc, timeout_s=30):
    end = time.monotonic() + timeout_s
    while not os.path.exists(path):
        assert proc.poll() is None, "launch exited before %s appeared" % path
        assert time.monotonic() < end, "timed out waiting for %s" % path
        time.sleep(0.02)


def _sleep_until(t_mono_ns):
    d = t_mono_ns - time.monotonic_ns()
    if d > 0:
        time.sleep(d / 1e9)


def _kill_marked(T):
    """Test hygiene: kill whatever still carries this launch's marker (a broken run)."""
    try:
        mark = _j(T, "raw", "launch.json")["launch_mark"]
    except (OSError, ValueError):
        return []
    left = procfs.marked_pids(mark)
    for pid in left:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return left


def test_second_signal_during_cleanup_still_finalizes(tmp_path):
    """K19: SIGTERM, then another SIGTERM while a role ignores SIGTERM in its stop grace. The
    second one must not escape run(): every role is stopped (SIGKILL without the rest of the
    grace) and launch_end.json / STATE are written."""
    roles = cf.default_roles(product_args=["--ignore-term"])
    wd = cf.make_workdir(tmp_path, cf.tier_doc(cf.gate([("nullsse", "backlog.active_streams"),
                                                         ("capload", "backlog.inflight")])),
                         roles, stop_order=["product", "nullsse", "capload", "sidecar",
                                            "canary"])
    T = os.path.join(str(tmp_path), "T")
    p = cf.start_launch(cf.launch_argv(wd, T, os.path.join(str(tmp_path), "shm"), lead_ms=2500),
                        str(tmp_path))
    try:
        _wait_file(os.path.join(T, "raw", "launch.json"), p)
        origin = _j(T, "raw", "launch.json")["origin_ns"]
        _sleep_until(origin + 300 * MS)
        p.send_signal(signal.SIGTERM)
        time.sleep(0.6)  # product ignores SIGTERM; its grace is stop_grace 1 s + kill_after 2 s
        assert p.poll() is None
        p.send_signal(signal.SIGTERM)
        rc = p.wait(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        left = _kill_marked(T)
    assert rc == 5, open(os.path.join(str(tmp_path), "launch.err")).read()
    assert left == []
    st = read_state(T)
    assert st[-1][1] == "NOT_RUN" and "signal 15" in st[-1][3]
    end = _j(T, "raw", "launch_end.json")
    assert [s["during_cleanup"] for s in end["signals"]] == [False, True]
    assert "stop signal during cleanup" in end["forced_kills"]["product"]
    assert end["exit_codes"]["product"] == -9
    assert all(rc is not None for rc in end["exit_codes"].values())
    assert end["leftovers"]["after"] == []
    assert os.path.isfile(os.path.join(T, "raw", "dims.json"))


def test_gate_survives_orchestrator_stalls(tmp_path):
    """K21: the orchestrator is stopped (SIGSTOP) for longer than two poll periods, repeatedly,
    during a drain whose backlog is already 0. Each resume finds several planned polls due; they
    must not run back to back (unchanged seq -> unsatisfied -> settle count reset). A 600 ms run
    window holds at most two on-schedule polls, so back-to-back polls keep the count below 4."""
    gcfg = cf.gate([("nullsse", "backlog.active_streams"), ("capload", "backlog.inflight")],
                   poll_ms=400, settle=4, max_age_ms=300, guard_ms=200)
    tdoc = cf.tier_doc(gcfg, load_ms=600, drain_ms=3000)
    roles = cf.default_roles(with_observers=False)
    wd = cf.make_workdir(tmp_path, tdoc, roles)
    T = os.path.join(str(tmp_path), "T")
    p = cf.start_launch(cf.launch_argv(wd, T, os.path.join(str(tmp_path), "shm"), lead_ms=2500,
                                       snap_ms=100), str(tmp_path))
    try:
        _wait_file(os.path.join(T, "raw", "launch.json"), p)
        origin = _j(T, "raw", "launch.json")["origin_ns"]
        drain = origin + 600 * MS  # tier 1 (first drain) start
        t = drain + 50 * MS
        while t < drain + 2800 * MS:  # stop 1000 ms, run 600 ms, until the gate deadline
            _sleep_until(t)
            p.send_signal(signal.SIGSTOP)
            _sleep_until(t + 1000 * MS)
            p.send_signal(signal.SIGCONT)
            t += 1600 * MS
        rc = p.wait(timeout=120)
    finally:
        if p.poll() is None:
            p.send_signal(signal.SIGCONT)
            p.kill()
            p.wait()
        _kill_marked(T)
    g = _j(T, "raw", "tiers", "001_drain-1", "drain_gate.json")
    lag = [x["actual_rel_ns"] - x["planned_rel_ns"] for x in g["polls"]]
    assert max(lag) > 500 * MS, lag  # a stall really made several planned polls due at once
    assert g["passed"] is True, [(x["planned_rel_ns"], x["actual_rel_ns"], x["satisfied"],
                                  x["errors"]) for x in g["polls"]]
    assert g["n_polls"] == len(g["polls"]) and \
        [x["planned_rel_ns"] for x in g["polls"]] == \
        list(range(g["start_rel_ns"], g["start_rel_ns"] + 400 * MS * len(g["polls"]), 400 * MS))
    acts = [x["actual_rel_ns"] for x in g["polls"]]
    assert all(b - a >= 150 * MS for a, b in zip(acts, acts[1:])), acts
    assert rc == 0, open(os.path.join(str(tmp_path), "launch.err")).read()

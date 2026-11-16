# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D14 loopback tool, clock check, D13 switchinterval report (functional only)."""

import json
import os
import subprocess

import pytest

from b4_envgate_util import PY, capenv_env, id_args, kill_all, wait_for
from capcommon.jsonio import read_json
from capenv import switchinterval


def _cpus(n):
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < n:
        pytest.skip("needs %d cpus" % n)
    return cpus[-n:]


def test_loopback_cli(tmp_path):
    a, b = _cpus(2)
    out = tmp_path / "loopback_softirq.json"
    r = subprocess.run([PY, "-m", "capenv.loopback"] + id_args("loopback") + [
        "--sender-cpu", str(a), "--receiver-cpu", str(b), "--duration-s", "0.5",
        "--msg-bytes", "65536", "--out", str(out)], env=capenv_env(), capture_output=True,
        text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    doc = read_json(str(out))
    assert doc["schema"] == "capkit.loopback_softirq/1" and doc["identity"]["role"] == "loopback"
    assert doc["bytes_sent"] == doc["bytes_received"] > 0
    snd, rcv = doc["sides"]["sender"], doc["sides"]["receiver"]
    assert snd["cpu"] == a and rcv["cpu"] == b
    assert snd["thread_last_cpu"] == a and rcv["thread_last_cpu"] == b
    # the sender spins for the whole window: most of a core in utime + stime
    assert snd["thread_cpu_frac"] > 0.5
    for side in (snd, rcv):
        assert set(side["core_ticks"]) >= {"softirq", "irq", "user", "system", "idle"}
        assert side["core_busy_minus_thread_ticks"] == side["core_busy_ticks"] - (
            side["thread_utime_ticks"] + side["thread_stime_ticks"])


def test_loopback_same_cpu_rejected(tmp_path):
    (a,) = _cpus(1)
    r = subprocess.run([PY, "-m", "capenv.loopback"] + id_args("loopback") + [
        "--sender-cpu", str(a), "--receiver-cpu", str(a), "--duration-s", "0.1",
        "--msg-bytes", "10", "--out", str(tmp_path / "x.json")], env=capenv_env(),
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 3


def test_clockcheck_server_client(tmp_path):
    a, b = _cpus(2)
    pf = str(tmp_path / "clock.port.json")
    out = str(tmp_path / "clock_check.json")
    common = ["--run-dir", str(tmp_path), "--stop-grace-s", "1", "--port-file", pf,
              "--round-trips", "200", "--back-to-back", "1000"]
    srv = subprocess.Popen([PY, "-m", "capenv.clockcheck"] + id_args("clocksrv") + common + [
        "--cpus", str(a), "--peer-mode", "server", "--out", str(tmp_path / "unused.json")],
        env=capenv_env())
    try:
        assert wait_for(lambda: os.path.exists(pf), 20)
        r = subprocess.run([PY, "-m", "capenv.clockcheck"] + id_args("clockcli") + common + [
            "--cpus", str(b), "--peer-mode", "client", "--out", out], env=capenv_env(),
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        assert srv.wait(timeout=30) == 0
    finally:
        kill_all([srv])
    doc = read_json(out)
    assert doc["schema"] == "capkit.clock_check/1"
    assert doc["round_trips"] == 200 and doc["one_way_ns"]["n"] == 200
    assert doc["consistent"] is True and doc["one_way_outside_0_rtt"] == 0
    assert doc["same_pid_ns"] is True and doc["back_to_back_ns"]["n"] == 1000
    assert doc["peer"]["pid"] == srv.pid


def test_switchinterval_counts(tmp_path):
    d = tmp_path / "versions"
    d.mkdir()
    (d / "a.1.json").write_text(json.dumps({"schema": "x", "identity": {"role": "a",
                                                                        "worker": 0},
                                            "versions": {"pid": 1, "switchinterval_s": 0.005}}))
    (d / "b.2.json").write_text(json.dumps({"pid": 2, "switchinterval_s": 0.005}))
    assert switchinterval.collect(str(d), 2)["verdict"] == "PASS"
    assert switchinterval.collect(str(d), 3)["verdict"] == "RED"  # one process missing
    assert switchinterval.collect(str(d), 0)["verdict"] == "NOT_RUN"
    (d / "c.3.json").write_text(json.dumps({"pid": 3}))
    res = switchinterval.collect(str(d), 3)
    assert res["verdict"] == "RED" and res["problems"][0]["file"] == "c.3.json"
    assert switchinterval.collect(str(tmp_path / "nope"), 1)["verdict"] == "RED"

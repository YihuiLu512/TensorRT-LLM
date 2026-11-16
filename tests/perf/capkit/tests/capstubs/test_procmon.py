# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-NX core-set split (capstubs.affinity), the process-tree sampler (capstubs.procmon), the
worker_cpus config key and the affinity / thread-name fields of the NullExecutor reports, and
the --worker-cores / --procmon-* options of serve_nx and nx_role."""

import json
import os
import subprocess
import sys
import time
import types

import pytest

from capcommon.exitcodes import ConfigError

from capstubs import affinity, procmon
from capstubs.null_executor import NullExecutor, SnxConfig

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, "..", "..", "tools"))
MS = 1_000_000


def test_split_cpus():
    assert affinity.split_cpus([5, 1, 3, 2, 4, 0], worker=3, monitor=1) == ([0, 1], [2], [3, 4, 5])
    assert affinity.split_cpus(range(4), worker=0, monitor=0) == ([0, 1, 2, 3], [], [])
    assert affinity.split_cpus(range(4), worker=2) == ([0, 1], [], [2, 3])
    with pytest.raises(ValueError):
        affinity.split_cpus(range(4), worker=3, monitor=1)
    with pytest.raises(ValueError):
        affinity.split_cpus(range(4), worker=-1)


def test_pin_all_tasks_to_current_set_is_a_no_op():
    cur = sorted(os.sched_getaffinity(0))
    got = affinity.pin_all_tasks(cur)
    assert got["before"] == cur and got["after"] == cur and got["tasks"] >= 1


def test_worker_cpus_config_key():
    c = SnxConfig(worker_cpus=[3, 4]).validate()
    assert SnxConfig.from_doc(c.to_doc()).worker_cpus == [3, 4]
    for bad in ([-1], ["a"], [True]):
        with pytest.raises(ConfigError):
            SnxConfig(worker_cpus=bad).validate()


class _Adapter:
    name = "fake"

    def convert(self, req_id, ereq):
        return types.SimpleNamespace(req_id=req_id)


def test_meta_and_summary_carry_affinity_and_thread_names(tmp_path):
    pin = {"before": [0, 1], "after": [1], "tasks": 3}
    ex = NullExecutor(SnxConfig(beat_ns=2 * MS, out_dir=str(tmp_path)).validate(), _Adapter(),
                      pin_info=pin)
    meta_p = os.path.join(str(tmp_path), "snx.%d.meta.json" % os.getpid())
    end = time.monotonic() + 5
    while not os.path.exists(meta_p) and time.monotonic() < end:
        time.sleep(0.01)
    ex.shutdown()
    meta = json.load(open(meta_p))
    assert meta["affinity"]["pin"] == pin
    assert meta["affinity"]["now"] == sorted(os.sched_getaffinity(0))
    summ = json.load(open(os.path.join(str(tmp_path), "snx.%d.summary.json" % os.getpid())))
    assert summ["affinity_end"] == sorted(os.sched_getaffinity(0))
    assert str(os.getpid()) in summ["thread_names"]  # the main thread's native id = pid
    assert "snx_step" in summ["thread_names"].values()  # seen while alive, kept after exit


TREE = r"""
import subprocess, sys, threading, time
child = subprocess.Popen([sys.executable, "-c",
    "import threading, time\n"
    "threading.Thread(target=lambda: sum(range(10**7)), daemon=True).start()\n"
    "time.sleep(30)"])
print(child.pid, flush=True)
time.sleep(float(sys.argv[1]))
child.kill()
"""


def test_procmon_samples_the_descendants_and_ends_with_the_root(tmp_path):
    out = str(tmp_path / "pm.jsonl")
    root = subprocess.Popen([sys.executable, "-c", TREE, "1.5"], stdout=subprocess.PIPE,
                            text=True)
    child = int(root.stdout.readline())
    env = dict(os.environ, PYTHONPATH=TOOLS)
    pm = subprocess.run([sys.executable, "-m", "capstubs.procmon", "--root-pid", str(root.pid),
                         "--out", out, "--interval-ms", "100"], env=env, timeout=30,
                        capture_output=True, text=True)
    root.wait(timeout=10)
    assert pm.returncode == 0, pm.stderr
    hdr, rows, procs, tasks, selfs = procmon.load(out)
    assert hdr["schema"] == procmon.SCHEMA and hdr["root_pid"] == root.pid
    seen = {p["proc"] for p in procs}
    assert {root.pid, child} <= seen
    assert all(r[1] in seen for r in rows) and len(rows) >= 4
    child_tids = {r[2] for r in rows if r[1] == child}
    assert len(child_tids) >= 2  # main thread + the busy thread
    assert {t["task"] for t in tasks} >= child_tids
    assert all(p["cpus_allowed"] for p in procs)
    end = selfs[-1]
    assert end["_end"] and end["reason"] == "root_gone" and end["samples"] >= 2
    assert end["cpu_s"] >= 0 and end["rows"] == len(rows)
    assert all(p["proc"] != end["self"] for p in procs)  # the sampler leaves itself out


def test_procmon_refuses_a_missing_root(tmp_path):
    env = dict(os.environ, PYTHONPATH=TOOLS)
    r = subprocess.run([sys.executable, "-m", "capstubs.procmon", "--root-pid", "999999999",
                        "--out", str(tmp_path / "x.jsonl")], env=env, capture_output=True,
                       text=True, timeout=30)
    assert r.returncode == 3


def test_split_role_cpus_starts_the_sampler_on_its_own_cpus(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", TOOLS)
    cur = sorted(os.sched_getaffinity(0))
    if len(cur) < 3:
        pytest.skip("needs >= 3 cpus")
    out = str(tmp_path / "pm.jsonl")
    a = types.SimpleNamespace(worker_cores=1, procmon_out=out, procmon_interval_ms=50,
                              procmon_cores=1)
    rec = None
    try:
        wk, rec = affinity.split_role_cpus(a, sys.executable)
        assert wk == cur[-1:] and rec["procmon_cpus"] == cur[-2:-1]
        assert sorted(os.sched_getaffinity(0)) == cur[:-2] == rec["own_cpus"]
        end = time.monotonic() + 10
        procs = []
        while time.monotonic() < end and not procs:
            time.sleep(0.1)
            if os.path.exists(out):
                procs = procmon.load(out)[2]
        assert procs and procs[0]["proc"] == os.getpid()
        with open("/proc/%d/status" % rec["procmon_pid"]) as f:
            line = [x for x in f if x.startswith("Cpus_allowed_list:")][0]
        assert line.split(":")[1].strip() == str(cur[-2])
    finally:
        affinity.pin_all_tasks(cur)
        if rec is not None:
            try:
                os.kill(rec["procmon_pid"], 15)
            except ProcessLookupError:
                pass


def test_role_option_refusals():
    from capstubs import nx_role, serve_nx
    p = serve_nx.build_parser()
    a = p.parse_args(["--worker-cores", "3", "--procmon-out", "/x", "--procmon-interval-ms", "500"])
    assert (a.worker_cores, a.procmon_out, a.procmon_interval_ms, a.procmon_cores) == \
        (3, "/x", 500, 1)
    ns = types.SimpleNamespace(mode="standalone", model=None, lanes=1, consumer="raw", poll_ms=10,
                               tail_s=1.0, worker_cores=2, procmon_cores=1,
                               procmon_interval_ms=1000)
    with pytest.raises(ConfigError):
        nx_role.validate_args(ns)
    ns.worker_cores, ns.procmon_interval_ms = 0, 5
    with pytest.raises(ConfigError):
        nx_role.validate_args(ns)

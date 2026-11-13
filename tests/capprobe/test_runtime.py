# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capprobe runtime in fresh subprocesses: counters, snapshot, injection heartbeat, miscount
control, lag probe, version self-report, fork and exit behaviour."""

import json
import os

from probe_testutil import DECODE_SELF, IDENTITY, run_json, run_py

from capcommon.inject import expected_applied
from capcommon.snapshot import list_snapshot_files, read_snapshot_file
from capprobe import sites


def test_counts_and_snapshot_layout(probe_env):
    env = probe_env("count")
    out = run_json(DECODE_SELF + """
for _ in range(3): capprobe.H1()
for _ in range(10): capprobe.F2()
s = snap()
s["on"] = {k: getattr(capprobe, "ON_" + k) for k in ("H1", "H3", "F2", "F4")}
s["pid"] = os.getpid()
print(json.dumps(s))
""", env)
    v = out["values"]
    assert out["on"] == {"H1": True, "H3": True, "F2": True, "F4": True}
    assert v["count.H1.arrivals"] == 3 and v["count.F2.msgs"] == 10
    # never-passed sites are registered eagerly: present with 0 (unwired != undeclared)
    assert v["count.H3.chunks"] == 0 and v["count.F4.dequeued"] == 0
    count_slots = {k for k, d in sites.SLOTS.items() if d["group"] == "count"}
    assert count_slots <= set(v)
    assert not any(k.startswith(("lag.", "inject.")) for k in v)  # groups off -> not registered
    meta = out["meta"]
    assert meta["identity"] == dict(IDENTITY, pid=out["pid"])
    assert meta["pid"] == out["pid"] and meta["schema"] == "capkit.snapshot/1"
    ver = meta["versions"]
    assert ver["probe_tier"] == "P2" and ver["probe_patch_sha"] == "c" * 40
    assert ver["capkit_code_sha"] == "d" * 40 and ver["probe_groups"] == ["count"]
    assert ver["loop_impl"] is None  # no loop installed in this process
    assert meta["probe"]["sites_on"] == ["F2", "F4", "H1", "H3"]
    assert sorted(os.listdir(probe_env.shm)) == ["%d.bin" % out["pid"]]


def test_p1_everything_off_writes_nothing(probe_env):
    env = probe_env("")
    out = run_json("""
import json, capprobe
r = {"active": capprobe.ACTIVE, "tier": capprobe.TIER,
     "on": [getattr(capprobe, "ON_" + k) for k in ("H1", "H3", "F2", "F4")],
     "noop": capprobe.F2 is capprobe._noop, "path": capprobe.snapshot_path(),
     "lag": capprobe.LAG}
capprobe.F2(); capprobe.flush()
print(json.dumps(r))
""", env)
    assert out == {"active": False, "tier": "P1", "on": [False] * 4, "noop": True, "path": None,
                   "lag": False}
    assert not os.path.exists(probe_env.shm)


def test_busy_injection_heartbeat(probe_env):
    delta = 200_000
    env = probe_env("count", CAPPROBE_INJECT="site=F2,mode=busy,delta_ns=%d,every_n=3,phase=1"
                    % delta)
    out = run_json(DECODE_SELF + """
for _ in range(10): capprobe.F2()
for _ in range(4): capprobe.H1()
print(json.dumps(snap()))
""", env)
    v = out["values"]
    want = expected_applied(10, 3, 1)
    assert want == 3
    assert v["inject.F2.hits"] == 10 and v["inject.F2.applied"] == want
    assert v["inject.F2.applied_ns"] >= want * delta
    assert v["count.F2.msgs"] == 10  # busy does not change the count
    # sites without a spec still count passes (heartbeat of the switch wiring)
    assert v["inject.H1.hits"] == 4 and v["inject.H1.applied"] == 0
    assert out["meta"]["probe"]["inject"][0]["site"] == "F2"


def test_sleep_injection_without_count_group(probe_env):
    delta = 2_000_000
    env = probe_env("lag", CAPPROBE_INJECT="site=H1,mode=sleep,delta_ns=%d" % delta)
    out = run_json(DECODE_SELF + """
r = {"on": capprobe.ON_H1}
for _ in range(3): capprobe.H1()
r.update(snap()); print(json.dumps(r))
""", env)
    v = out["values"]
    assert out["on"] is True
    assert "count.H1.arrivals" not in v  # count group off
    assert v["inject.H1.hits"] == 3 and v["inject.H1.applied"] == 3
    assert v["inject.H1.applied_ns"] >= 3 * delta


def test_miscount_control_is_visible(probe_env):
    env = probe_env("count", CAPPROBE_INJECT="site=H3,mode=miscount,miscount_delta=2,every_n=5;"
                    "site=F4,mode=miscount,miscount_delta=-1,every_n=2")
    out = run_json(DECODE_SELF + """
for _ in range(10): capprobe.H3()
for _ in range(10): capprobe.F4()
for _ in range(10): capprobe.F2()
print(json.dumps(snap()))
""", env)
    v = out["values"]
    assert v["count.H3.chunks"] == 10 + 2 * expected_applied(10, 5, 0) == 14  # over-count
    assert v["count.F4.dequeued"] == 10 - expected_applied(10, 2, 0) == 5  # under-count
    assert v["count.F2.msgs"] == 10  # untouched control site
    assert v["inject.H3.applied"] == 2 and v["inject.F4.applied"] == 5


def test_busy_on_aux_thread(probe_env):
    delta = 1_000_000
    env = probe_env("count", CAPPROBE_INJECT="site=F4,mode=busy,delta_ns=%d,where=aux_thread"
                    % delta)
    out = run_json(DECODE_SELF + """
import threading, time
names = sorted(t.name for t in threading.enumerate())
for _ in range(5): capprobe.F4()
deadline = time.monotonic() + 10
while capprobe.injector_marks()["F4"]["aux_pending"] and time.monotonic() < deadline:
    time.sleep(0.01)
r = snap(); r["threads"] = names; print(json.dumps(r))
""", env)
    v = out["values"]
    assert "capkit-inject-aux-F4" in out["threads"]  # pre-started at import, not lazily
    assert v["inject.F4.applied"] == 5 and v["inject.F4.applied_ns"] >= 5 * delta


LAG_CODE = DECODE_SELF + """
import asyncio, time
async def main():
    capprobe.install_loop_lag("Lfe")
    await asyncio.sleep(0.05)
    time.sleep(0.06)  # block the loop: the next timer fires >= ~50 ms late
    await asyncio.sleep(0.05)
    try:
        capprobe.install_loop_lag("Lfe")
        twice = "no error"
    except RuntimeError as e:
        twice = str(e)
    try:
        capprobe.install_loop_lag("Lnope")
        unknown = "no error"
    except RuntimeError as e:
        unknown = str(e)
    return twice, unknown
twice, unknown = asyncio.run(main())
r = snap(); r["twice"] = twice; r["unknown"] = unknown; print(json.dumps(r))
"""


def test_lag_probe_and_version_self_report(probe_env):
    out = run_json(LAG_CODE, probe_env("count,lag"))
    v = out["values"]
    assert v["lag.Lfe.installs"] == 1 and v["lag.Lmfl.installs"] == 0
    assert v["lag.Lfe.n"] >= 20
    assert v["lag.Lfe.max"] >= 40_000_000  # the 60 ms block is visible
    assert sum(v["lag.Lfe"]) == v["lag.Lfe.n"]
    assert v["lag.Lmfa.n"] == 0  # declared, not installed in this process
    assert "installed twice" in out["twice"] and "unknown lag site" in out["unknown"]
    ver = out["meta"]["versions"]
    assert ver["loop_impl"].startswith("asyncio.")
    assert ver["probe_loops"] == [{"site": "Lfe", "loop_impl": ver["loop_impl"]}]
    assert isinstance(ver["gc_enabled"], bool) and ver["switchinterval_s"] > 0


def test_install_loop_lag_refuses_when_lag_off(probe_env):
    r = run_py("""
import asyncio, capprobe
async def m(): capprobe.install_loop_lag("Lfe")
asyncio.run(m())
""", probe_env("count"))
    assert r.returncode != 0 and "lag group is off" in r.stderr


def test_grpc_version_refresh_after_late_import(probe_env):
    # grpc imported by the product after capprobe: the writer refreshes versions by itself.
    out = run_json(DECODE_SELF + """
r1 = snap()["meta"]["versions"]["grpc"]
try:
    import grpc
    have = grpc.__version__
except ImportError:
    have = None
r2 = snap()["meta"]["versions"]
print(json.dumps({"before": r1, "after": r2["grpc"], "have": have,
                  "pb_impl": r2["protobuf_impl"]}))
""", probe_env("count"))
    assert out["before"] is None
    assert out["after"] == out["have"]
    if out["have"] is not None:
        assert out["pb_impl"] in ("upb", "python", "cpp")


def test_background_writer_is_periodic(probe_env):
    env = probe_env("count", CAPPROBE_INTERVAL_MS="20")
    out = run_json("""
import json, time, capprobe
from capcommon.snapshot import read_snapshot_file
p = capprobe.snapshot_path()
s0 = read_snapshot_file(p).seq
capprobe.F2(); capprobe.F2()
time.sleep(0.3)
s1 = read_snapshot_file(p)
print(json.dumps({"s0": s0, "s1": s1.seq, "f2": s1.values["count.F2.msgs"]}))
""", env)
    assert out["s0"] == 1 and out["s1"] >= 6 and out["f2"] == 2


def test_final_snapshot_at_exit(probe_env):
    env = probe_env("count", CAPPROBE_INTERVAL_MS="600000")
    r = run_py("import capprobe\nfor _ in range(7): capprobe.H3()\nprint(capprobe.snapshot_path())",
               env)
    assert r.returncode == 0, r.stderr
    s = read_snapshot_file(r.stdout.strip())
    assert s.values["count.H3.chunks"] == 7 and s.seq == 2  # start + exit


def test_fork_child_gets_own_zeroed_probe(probe_env):
    out = run_json(DECODE_SELF + """
for _ in range(5): capprobe.F2()
pid = os.fork()
if pid == 0:
    capprobe.F2(); capprobe.F2(); capprobe.flush(); os._exit(0)
os.waitpid(pid, 0)
r = snap(); r["child"] = pid; r["parent"] = os.getpid(); print(json.dumps(r))
""", probe_env("count"))
    files = list_snapshot_files(probe_env.shm)
    assert set(files) == {out["parent"], out["child"]}
    child = read_snapshot_file(files[out["child"]])
    assert child.values["count.F2.msgs"] == 2  # zeroed at fork, not 7
    assert child.meta["identity"]["pid"] == out["child"] and child.meta["ppid"] == out["parent"]
    assert out["values"]["count.F2.msgs"] == 5


def test_bad_switch_stops_the_process(probe_env):
    r = run_py("import capprobe", probe_env("count,bytes"))
    assert r.returncode != 0 and "CapprobeConfigError" in r.stderr
    assert not os.path.exists(probe_env.shm)


def test_decode_cli(probe_env, tmp_path):
    r = run_py("import capprobe\ncapprobe.F2()\ncapprobe.flush()\nprint(capprobe.snapshot_path())",
               probe_env("count"))
    path = r.stdout.strip()
    # tool mode: the decoder runs under the same P2 environment and must not start a probe
    before = sorted(os.listdir(probe_env.shm))
    assert before == [os.path.basename(path)]
    import subprocess
    import sys
    d = subprocess.run([sys.executable, "-m", "capprobe.decode", path, "--no-hist"],
                       env=probe_env("count"), capture_output=True, text=True, timeout=60)
    assert d.returncode == 0, d.stderr
    doc = json.loads(d.stdout)
    assert doc["values"]["count.F2.msgs"] == 1 and doc["seq"] >= 1
    assert "lag.Lfe" not in doc["values"]
    assert sorted(os.listdir(probe_env.shm)) == before  # no snapshot from the tool process
    bad = tmp_path / "123.bin"
    bad.write_bytes(b"CAPSNAP1" + b"\0" * 40)
    d = subprocess.run([sys.executable, "-m", "capprobe.decode", str(bad)], env=probe_env("count"),
                       capture_output=True, text=True, timeout=60)
    assert d.returncode == 3 and "capprobe.decode" in d.stderr
    d = subprocess.run([sys.executable, "-m", "capprobe.decode", str(tmp_path / "none.bin")],
                       env=probe_env("count"), capture_output=True, text=True, timeout=60)
    assert d.returncode == 4

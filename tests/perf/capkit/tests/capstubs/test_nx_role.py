# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capstubs.nx_role (nx_driver as a capkit tool) without tensorrt_llm: standalone mode with the
fake adapter of test_null_executor through a real ToolRuntime (snapshots, ready file, per-tier
summaries), the NullExecutor live record and its reader, the slots it feeds, argument refusals
and the TOOL_TABLE entry. Proxy mode needs the product (container / local chroot runs)."""

import glob
import json
import os
import time

import pytest

from conftest import g
from test_null_executor import FakeAdapter

from capcommon.exitcodes import ConfigError, OK
from capcommon.lifecycle import StopController
from caplaunch import scenario as sc

from capstubs import nx_role
from capstubs.null_executor import LIVE_FIELDS, LiveReader, NullExecutor, SnxConfig

MS = 1_000_000


def _cpus():
    return ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))


def _table(launch):
    # warmup 300 ms at 40 req/s, drain, up 300 ms at 60 req/s, drain; 4 tokens, beat 5 ms
    kw = {"n": 4, "beat": 5 * MS}
    return launch.write_table(g.table_doc([("warmup", 300, 40.0, kw), ("drain", 200, 0, kw),
                                            ("up", 300, 60.0, kw), ("drain", 200, 0, kw)]))


def _role(tmp_path, extra=()):
    launch = g.Launch(tmp_path, "python3")
    _table(launch)
    launch.set_origin(1.0)
    argv = launch.common("nx", cpus=_cpus()) + [
        "--mode", "standalone", "--schedule", launch.sched_path,
        "--nxd-dir", str(tmp_path / "nxd"), "--snx-out-dir", str(tmp_path / "snx"),
        "--set", "beat_ns=5000000", "--tail-s", "0.3"] + list(extra)
    args = nx_role.build_parser().parse_args(argv)
    common = nx_role.common_from_args(args)
    nx_role.validate_args(args)
    return launch, args, common


def fake_parts():
    def make_request(plan, req_id, seq):
        return {"client_id": req_id, "max_tokens": plan.max_tokens[seq]}
    return FakeAdapter(), make_request


def test_standalone_role_end_to_end(tmp_path, monkeypatch):
    from capcommon.aio import run as aio_run
    monkeypatch.setattr(nx_role, "standalone_parts", fake_parts)
    launch, args, common = _role(tmp_path)
    role = nx_role.NxRole(args, common, StopController(install=False))
    role.build()
    assert aio_run(role.run, "asyncio") == OK
    tree = launch.tree
    ready = json.load(open(tree.ready_file("nx")))
    info = ready["info"]["workers"][0]["info"]
    assert info["mode"] == "standalone" and info["engine_pid"] == os.getpid()
    per = {int(k): v for k, v in launch.sched_meta["per_tier"].items()}
    n_all = sum(per.values())
    for seq, label in ((0, "warmup-0"), (2, "up-2")):
        doc = json.load(open(tree.tier_file(seq, label, "nx", 0, "summary", "json")))
        c = doc["counts"]
        assert c["planned"] == per.get(seq, 0) == c["sent"] == c["ok"] > 0, c
        assert c["failed"] == 0 and doc["failures"] == []
        assert doc["schedule_sha256"] and doc["n_workers"] == 1
    for seq, label in ((1, "drain-1"), (3, "drain-3")):
        c = json.load(open(tree.tier_file(seq, label, "nx", 0, "summary", "json")))["counts"]
        assert c["planned"] == c["sent"] == 0
    assert glob.glob(os.path.join(launch.shm, "*"))  # the snapshot writer wrote
    v, p = role.rt.registry.values, role.pos
    assert v[p["backlog.inflight"]] == 0 and v[p["backlog.engine_open"]] == 0
    assert v[p["self.sent"]] == v[p["self.finished"]] == n_all
    nxs = json.load(open(str(tmp_path / "nxd" / "nxd.standalone.summary.json")))
    assert nxs["engine_counters_at_stop"]["finished"] == n_all


def test_live_record_reader(tmp_path):
    ex = NullExecutor(SnxConfig(beat_ns=2 * MS, out_dir=str(tmp_path)).validate(),
                      FakeAdapter())
    try:
        lives = glob.glob(str(tmp_path / "snx.*.live"))
        assert len(lives) == 1
        rd = LiveReader(lives[0])
        for cid in range(3):
            ex.enqueue_request({"client_id": cid, "max_tokens": 2})
        end = time.monotonic() + 5
        got = 0
        while got < 6 and time.monotonic() < end:
            got += len(ex.await_responses(timeout=0.2))
        live = rd.read()
        assert set(live) == set(LIVE_FIELDS)
        assert live["enqueued"] == 3 and live["taken"] == 3 and live["finished"] == 3
        assert live["active"] == live["queued"] == live["pending_keys"] == 0
        assert live["gen"] % 2 == 0 and live["t_ns"] > 0
        rd.close()
    finally:
        ex.shutdown()
    assert LiveReader(str(tmp_path / "missing")).read() is None


def test_refusals(tmp_path):
    launch, args, common = _role(tmp_path)
    args.mode = "proxy"
    with pytest.raises(ConfigError):
        nx_role.validate_args(args)
    args.mode, args.lanes = "standalone", 2
    with pytest.raises(ConfigError):
        nx_role.validate_args(args)


def test_tool_table_entry():
    t = sc.TOOL_TABLE["capstubs.nx_role"]
    assert t["single"] and t["snapshot"] and t["ready"] and not t.get("port")
    flags = [a for a in vars(nx_role.build_parser().parse_args(
        ["--experiment", "E", "--arm", "a", "--launch-seq", "0", "--job-id", "j",
         "--restart-count", "0", "--register-sha", "s", "--role", "r", "--run-dir", "/",
         "--loop", "asyncio", "--tier-table", "/t", "--origin-ns", "1", "--shm-dir", "/",
         "--snapshot-interval-ms", "50", "--ready-file", "/r", "--cpus", "0",
         "--stop-grace-s", "1", "--mode", "standalone", "--schedule", "/s", "--nxd-dir", "/n",
         "--snx-out-dir", "/x", "--lag-period-us", "1000"]))]
    assert "out_dir" not in flags  # --out-dir is orchestrator-managed
    for k in nx_role.SLOTS:
        assert k.startswith(("backlog.", "self.", "lag."))


# ---- --engine-probe-ms (PLAN-R2R5 3.1 B5, E7 R axis) ----------------------------------------

def test_engine_probe_end_to_end(tmp_path, monkeypatch):
    """A healthy engine: one probe row per 20 ms grid point (plus the base sample before the
    origin), rows split by the tier of the sample time, per-tier summaries count them, counters
    never go back, the finished counter reaches every request, and the backlog lags stay small."""
    from capcommon.aio import run as aio_run
    from capcommon.capbin import read_capbin
    monkeypatch.setattr(nx_role, "standalone_parts", fake_parts)
    launch, args, common = _role(tmp_path, ["--engine-probe-ms", "20"])
    role = nx_role.NxRole(args, common, StopController(install=False))
    role.build()
    assert aio_run(role.run, "asyncio") == OK
    tree = launch.tree
    n_all = sum(int(v) for v in launch.sched_meta["per_tier"].values())
    rows = {}
    for seq, label in ((0, "warmup-0"), (1, "drain-1"), (2, "up-2"), (3, "drain-3")):
        h, c = read_capbin(tree.tier_file(seq, label, "nx", 0, "probe", "capbin"))
        assert h["kind"] == nx_role.PROBE_KIND and h["meta"]["overflow"] == 0
        assert h["meta"]["period_ns"] == 20 * MS and h["meta"]["planned_requests"] == n_all
        assert set(c["tier_seq"]) == {seq}
        win_ms = (300, 200, 300, 200)[seq]
        assert win_ms // 20 - 2 <= len(c["t_rel_ns"]) <= win_ms // 20 + 1, (seq, len(c["t_rel_ns"]))
        doc = json.load(open(tree.tier_file(seq, label, "nx", 0, "summary", "json")))
        assert doc["engine_probe"]["samples"] == len(c["t_rel_ns"])
        assert doc["engine_probe"]["period_ns"] == 20 * MS and doc["engine_probe"]["missed"] == 0
        rows[seq] = c
    up = rows[2]
    # grid samples: every sample time is at or just after a 20 ms grid point
    assert all(0 <= t % (20 * MS) < 15 * MS for t in up["t_rel_ns"])
    for k in ("enqueued", "taken", "finished", "due"):
        seq_vals = [v for s in (0, 1, 2, 3) for v in rows[s][k]]
        assert seq_vals == sorted(seq_vals), k
    assert rows[3]["finished"][-1] == n_all and rows[3]["due"][-1] == n_all
    # the engine keeps up (beat 5 ms, 4 tokens): the oldest unfinished request is never old
    assert max(up["lag_fin_ns"]) < 100 * MS and max(up["lag_enq_ns"]) < 50 * MS


class _Plan:
    def __init__(self, planned_abs, origin, tt):
        self.rows = [(i, p, 0) for i, p in enumerate(planned_abs)]
        self.origin = origin
        self.tt = tt


def _probe(period_ms=100, rate=1000.0, dur_ms=2000):
    from capcommon.tiers import TierTable
    tt = TierTable.from_doc(g.table_doc([("warmup", dur_ms, rate, {"n": 1, "beat": MS}),
                                          ("drain", 300, 0.0, {"beat": MS})]))
    origin = 10 ** 15
    planned = [origin + int(i * 1e9 / rate) for i in range(int(rate * dur_ms / 1000))]
    return nx_role.EngineProbe(_Plan(planned, origin, tt), period_ms, 0.0), origin, planned


def _counters(t_abs, enq, taken, fin, base=(5, 5, 5)):
    return {"t_ns": t_abs, "enqueued": enq + base[0], "taken": taken + base[1],
            "finished": fin + base[2], "active": 0, "queued": enq - taken, "pending_keys": 0}


def test_engine_probe_lags_keep_up_and_stall():
    """Positive control on hand-made counters (1000 req/s, one sample per 100 ms): an engine that
    keeps up gives lag 0; once the step thread stops at the 1 s point (taken / finished frozen,
    enqueued still following the plan) the take and finish lags grow with the sample time, one
    second of stall = one second of lag, and the enqueue lag stays 0."""
    pr, origin, planned = _probe()
    assert pr.sample(_counters(origin - 10 ** 9, 0, 0, 0), origin - 10 ** 9)  # base before origin
    assert pr.base == {"enqueued": 5, "taken": 5, "finished": 5}
    frozen = None
    for k in range(20):
        t = origin + k * 100 * MS
        due = sum(1 for p in planned if p <= t)
        if k < 10:
            assert pr.sample(_counters(t, due, due, due), t)
        else:
            frozen = frozen if frozen is not None else due_prev
            assert pr.sample(_counters(t - 5 * MS, due, frozen, frozen), t)
        due_prev = due
    c = {k: list(v[:pr.rec.n]) for k, v in pr.rec.cols.items()}
    assert c["lag_enq_ns"] == [0] * 21
    assert c["lag_fin_ns"][:11] == [0] * 11 and c["lag_take_ns"][:11] == [0] * 11
    stalled = c["lag_fin_ns"][11:]
    assert stalled == c["lag_take_ns"][11:]
    # frozen at the count due at 0.9 s: the oldest unfinished request was planned at 0.901 s
    assert stalled == [k * 100 * MS - (frozen * MS) for k in range(10, 20)]
    assert all(b - a == 100 * MS for a, b in zip(stalled, stalled[1:]))
    assert c["live_age_ns"][11:] == [5 * MS] * 10
    assert c["tier_seq"][0] == -1 and set(c["tier_seq"][1:]) == {0}


def test_engine_probe_missed_and_capacity():
    pr, origin, _planned = _probe(period_ms=500, dur_ms=1000)
    assert pr.rec.capacity == (1000 + 300) * MS // (500 * MS) + 16
    assert pr.sample(None, origin) is False and pr.missed == 1 and pr.base is None
    assert pr.lag(10, 3, 3) == 0 and pr.lag(10, 2, 3) == 10 - pr.planned[2]


def test_engine_probe_refusal(tmp_path):
    launch, args, common = _role(tmp_path)
    args.engine_probe_ms = -1
    with pytest.raises(ConfigError):
        nx_role.validate_args(args)

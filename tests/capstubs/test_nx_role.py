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

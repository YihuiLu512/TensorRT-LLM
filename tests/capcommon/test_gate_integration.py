# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate arithmetic, plus an end-to-end walk of the contract a load generator follows."""

import copy
import os
import time

import pytest

from capcommon import gate, jsonio, schedule, snapshot
from capcommon.capbin import RecordArray, read_capbin
from capcommon.runtree import RunTree
from capcommon.tiers import TierTable

from conftest import GATE


LK = ["E0", "unit-a", 1, "local-test", 0, "a" * 64]


def _snap(tmp_path, pid, launch_key=LK, **vals):
    reg = snapshot.SlotRegistry()
    for k, v in vals.items():
        name = k.replace("__", ".")
        reg.values[reg.gauge(name) if "inflight" in name else reg.counter(name)] = v
    ident = dict(zip(("experiment", "arm", "launch_seq", "job_id", "restart_count",
                      "register_sha"), launch_key))
    w = snapshot.SnapshotWriter(reg, str(tmp_path), 1_000_000, pid=pid,
                                meta_extra={"identity": ident})
    w.write_now()
    return w


@pytest.fixture
def fake_starttime(monkeypatch):
    monkeypatch.setattr(snapshot, "proc_starttime_ticks", lambda pid=None: 1000 + (pid or 0))


def _proc(pid, role):
    return {"pid": pid, "role": role, "starttime_ticks": 1000 + pid}


def test_snapshot_vetting(tmp_path, fake_starttime):
    w = _snap(tmp_path, 101, backlog__inflight=0)
    _snap(tmp_path, 102, backlog__inflight=0, launch_key=LK[:2] + [0] + LK[3:])  # other launch
    _snap(tmp_path, 103, backlog__inflight=0)
    procs = [_proc(101, "capload"), _proc(102, "capload"),
             dict(_proc(103, "capload"), starttime_ticks=7)]  # pid reused by another process
    now = time.monotonic_ns()
    snaps, errors, fields = gate.snapshots_by_role(str(tmp_path), procs, LK, 10 ** 9, now_ns=now)
    assert [s.pid for s in snaps["capload"]] == [101]
    assert "launch key" in errors[102] and "starttime" in errors[103]
    # stale: the same file read 2 s later is too old (a crashed writer leaves inflight=0 behind)
    snaps, errors, _f = gate.snapshots_by_role(str(tmp_path), procs[:1], LK, 10 ** 9,
                                               now_ns=now + 2 * 10 ** 9)
    assert snaps == {} and "stale" in errors[101]
    # seq must advance between polls of one gate
    prev = {101: fields[101]["seq"]}
    snaps, errors, _f = gate.snapshots_by_role(str(tmp_path), procs[:1], LK, 10 ** 9,
                                               prev_seqs=prev)
    assert snaps == {} and "did not advance" in errors[101]
    w.write_now()
    snaps, errors, _f = gate.snapshots_by_role(str(tmp_path), procs[:1], LK, 10 ** 9,
                                               prev_seqs=prev)
    assert errors == {} and len(snaps["capload"]) == 1
    # the judge re-applies the same rule to recorded fields
    assert gate.check_snapshot(fields[101], procs[0], now, 10 ** 9, None, LK) == (True, None)
    assert not gate.check_snapshot(fields[101], procs[0], now, 10 ** 9, fields[101]["seq"],
                                   LK)[0]


def test_planned_polls():
    assert gate.planned_polls(0, 250, 100) == [0, 100, 200]
    assert gate.planned_polls(0, 200, 100) == [0, 100, 200]
    assert gate.planned_polls(10, 5, 100) == []


def test_gate_readings(tmp_path, fake_starttime):
    _snap(tmp_path, 101, backlog__inflight=2)
    _snap(tmp_path, 102, backlog__inflight=0)
    _snap(tmp_path, 201, backlog__yielded=10)
    _snap(tmp_path, 301, backlog__read=10)
    procs = [_proc(101, "capload"), _proc(102, "capload"), _proc(201, "stub"),
             _proc(301, "frontend"), _proc(999, "ghost")]
    snaps, errors, _f = gate.snapshots_by_role(str(tmp_path), procs, LK, 10 ** 10)
    assert errors == {999: "no snapshot file"} and len(snaps["capload"]) == 2
    g = copy.deepcopy(GATE)
    g["readings"].append({"name": "grpc_in_flight", "abs_max": 0, "terms": [
        {"role": "stub", "slot": "backlog.yielded", "sign": 1, "agg": "sum"},
        {"role": "frontend", "slot": "backlog.read", "sign": -1, "agg": "sum"}]})
    ok, values, _d = gate.evaluate_gate(g, snaps)
    assert not ok and values == {"inflight": 2, "grpc_in_flight": 0}
    g["readings"][0]["terms"][0]["agg"] = "max"
    g["readings"][0]["abs_max"] = 2
    assert gate.evaluate_gate(g, snaps)[0]
    g["readings"].append({"name": "missing", "abs_max": 0, "terms": [
        {"role": "ghost", "slot": "backlog.x", "sign": 1, "agg": "sum"}]})
    ok, values, details = gate.evaluate_gate(g, snaps)
    assert not ok and values["missing"] is None and "error" in details["missing"]
    # role present but slot absent: must be unsatisfied, never read as 0
    g2 = copy.deepcopy(GATE)
    g2["readings"][0]["terms"][0]["slot"] = "backlog.nonexistent"
    ok, values, details = gate.evaluate_gate(g2, snaps)
    assert not ok and values["inflight"] is None and "no slot" in details["inflight"]["error"]
    band_gate = dict(g, rule="baseline_band", readings=g["readings"][:1])
    assert not gate.evaluate_gate(band_gate, snaps)[0]  # no band registered -> unsatisfied
    assert gate.evaluate_gate(band_gate, snaps, band={"inflight": (0, 2)})[0]
    tr = gate.GateTracker({"settle_polls": 3})
    assert [tr.update(s, i) for i, s in enumerate([True, True, False, True, True, True])] == \
        [False, False, False, False, False, True]
    assert tr.t_pass_rel_ns == 5 and tr.n_polls == 6


def test_loadgen_contract_walkthrough(tmp_path, table_path, ident):
    """Tier table -> schedule -> per-worker records split by tier -> summary files."""
    tt = TierTable.load(table_path())
    sched = str(tmp_path / "sched.capbin")
    meta = schedule.write_schedule(sched, tt, 10 ** 6, identity=ident.with_role("launch"))
    tree = RunTree(str(tmp_path / "T")).make_skeleton()
    W = 2
    expected = schedule.per_worker_tier_counts(meta, W)
    for w in range(W):
        my = ident.with_worker(w).with_pid()
        ids, planned, tseq, _m = schedule.load_schedule(sched, w, W, tier_table_sha=tt.sha256)
        rec = RecordArray([("req_id", "q"), ("planned_rel_ns", "q"), ("sent_rel_ns", "q"),
                           ("tier_seq", "h")], capacity=len(ids))
        for r, p, s in zip(ids, planned, tseq):
            rec.append(r, p, p + 5, s)  # pretend each request went out 5 ns late
        written = rec.dump_split(
            lambda s: tree.tier_file(s, tt.label_of(s), "capload", w, "requests", "capbin"),
            "loadgen.requests", lambda s: my.with_tier(s, tt.label_of(s)))
        for s, path in written.items():
            hdr, cols = read_capbin(path)
            assert hdr["identity"]["tier_seq"] == s and hdr["identity"]["worker"] == w
            assert len(cols["req_id"]) == expected[(w, s)]
            jsonio.write_json_stamped(
                tree.tier_file(s, tt.label_of(s), "capload", w, "summary", "json"),
                "capkit.tier_summary/1", my.with_tier(s, tt.label_of(s)),
                {"tool": "capload", "counts": {"planned": expected[(w, s)],
                                               "sent": len(cols["req_id"]), "ok": 0,
                                               "failed": 0, "not_sent": 0}})
    files = sorted(os.listdir(tree.tier_dir(2, "up-r200")))
    assert files == ["capload.w0.requests.capbin", "capload.w0.summary.json",
                     "capload.w1.requests.capbin", "capload.w1.summary.json"]
    with pytest.raises(FileNotFoundError):
        os.listdir(tree.tier_dir(1, "drain-1"))  # no arrivals planned in drain tiers

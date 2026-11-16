# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E0 (b) chain: beat source (nullgrpc --pacing beat --stamp) -> relay -> stamped drain
(capgrpc --stamped). Counts must reconcile at every hop and every tier, with the expected
numbers derived from the tier table and the schedule header (not from what the tools report)."""

import time

from capcommon import expect
from capcommon.gate import evaluate_gate, snapshots_by_role
from capcommon.jsonio import read_json_stamped
from capcommon.schedule import BeatSchedule
from capcommon.snapshot import list_snapshot_files, read_snapshot_file

from conftest import table_doc, per_tier

METHOD = "/capkit.Test/Stream"
CHAIN = {"name": "msgs", "kind": "messages", "gate": "product", "include_usage": None,
         "hops": [], "corrections": []}


def chain_table():
    return table_doc([("warmup", 600, 20.0, {}),
                      ("drain", 400, 0.0, {}),
                      ("up", 600, 40.0, {"n": 12, "k": 2, "s": 1}),
                      ("drain", 600, 0.0, {})])


def drain_args(launch, target_role, workers, extra=()):
    return (launch.common("drain", cpus=",".join(str(i) for i in range(workers)))
            + ["--target-port-file", launch.tree.port_file(target_role), "--method", METHOD,
               "--workers", str(workers), "--channels-per-worker", "2",
               "--schedule", launch.sched_path, "--request-format", "capkit",
               "--prompts", launch.prompts_path, "--msg-sample-mod", "1",
               "--max-records-per-worker", "1000"] + list(extra))


def snapshot_roles(shm):
    out = {}
    for pid, path in list_snapshot_files(shm).items():
        try:
            s = read_snapshot_file(path)
        except Exception:  # noqa: BLE001 -- a file mid-replace; skip this poll
            continue
        out.setdefault(s.meta["identity"]["role"], []).append(s)
    return out


def _term(role, slot, sign=1):
    return {"role": role, "slot": slot, "sign": sign, "agg": "sum"}


# Drain gate over this chain, evaluated with the shared capcommon.gate code (INTERFACES.md 8)
CHAIN_GATE = {"mode": "enforce", "rule": "abs_max", "readings": [
    {"name": "drain_inflight", "terms": [_term("drain", "backlog.inflight")], "abs_max": 0},
    {"name": "source_pending", "terms": [_term("source", "backlog.pending_msgs")], "abs_max": 0},
    {"name": "source_to_relay", "terms": [_term("source", "backlog.yielded"),
                                          _term("relay", "backlog.read", -1)], "abs_max": 0},
    {"name": "relay_queue", "terms": [_term("relay", "backlog.queue_msgs")], "abs_max": 0},
    {"name": "relay_to_drain", "terms": [_term("relay", "backlog.yielded"),
                                         _term("drain", "backlog.read", -1)], "abs_max": 0}],
    "poll_ns": 100_000_000, "settle_polls": 2, "deadline_guard_ns": 0,
    "max_snapshot_age_ns": 10 ** 15}


def gate_poll(shm):
    procs, key = [], None
    for pid, path in list_snapshot_files(shm).items():
        try:
            m = read_snapshot_file(path).meta
        except Exception:  # noqa: BLE001 -- mid-replace
            continue
        procs.append({"pid": pid, "role": m["identity"]["role"],
                      "starttime_ticks": m["starttime_ticks"]})
        ident = m["identity"]
        key = [ident[k] for k in ("experiment", "arm", "launch_seq", "job_id", "restart_count",
                                  "register_sha")]
    by_role, errors, _f = snapshots_by_role(shm, procs, key, CHAIN_GATE["max_snapshot_age_ns"])
    ok, values, _d = evaluate_gate(CHAIN_GATE, by_role)
    return ok, values, errors


def test_beat_relay_drain_reconciles(launch):
    tt = launch.write_table(chain_table())
    launch.set_origin(5.0)
    launch.server("source", "capgrpc.nullgrpc", ["--workers", "1", "--in-process", "--msg-bytes", "64",
                                         "--pacing", "beat", "--stamp"])
    launch.wait_file(launch.tree.port_file("source"), 15, "source")
    launch.server("relay", "capselfcheck.relay",
                  ["--upstream-port-file", launch.tree.port_file("source")], cpus="1")
    launch.wait_file(launch.tree.port_file("relay"), 15, "relay")
    launch.start("drain", "capgrpc.load", drain_args(launch, "relay", 2, ["--stamped"]))
    launch.wait_file(launch.tree.ready_file("drain"), 15, "drain")
    # positive control for the backlog readings: they must move during load
    peak = {"drain": 0, "source": 0}
    unsatisfied_polls = 0
    end = launch.origin + tt.end_ns
    while time.monotonic_ns() < end and launch.procs["drain"].poll() is None:
        snaps = snapshot_roles(launch.shm)
        peak["drain"] = max(peak["drain"], sum(s.values["backlog.active_streams"]
                                               for s in snaps.get("drain", [])))
        peak["source"] = max(peak["source"], sum(s.values["backlog.active_streams"]
                                                 for s in snaps.get("source", [])))
        if time.monotonic_ns() > launch.origin:
            ok, _v, errors = gate_poll(launch.shm)
            unsatisfied_polls += 0 if ok or errors else 1
        time.sleep(0.03)
    assert launch.wait_exit("drain", 20) == 0, launch.stderr("drain")
    assert launch.stop("relay") == 0, launch.stderr("relay")
    assert launch.stop("source") == 0, launch.stderr("source")
    assert peak["drain"] > 0 and peak["source"] > 0, peak
    assert unsatisfied_polls > 0   # the gate readings do register in-flight work ...
    ok, values, errors = gate_poll(launch.shm)
    assert not errors and ok and len(values) == 5 and set(values.values()) == {0}  # ... and drain

    per = {int(k): v for k, v in launch.sched_meta["per_tier"].items()}
    exp_req = {t.seq: per.get(t.seq, 0) for t in tt.tiers}
    exp_msgs = {t.seq: (expect.expected_hop_count(CHAIN, exp_req[t.seq], t.max_tokens,
                                                  t.tokens_per_msg, t.stream_interval, "x")
                        if t.is_load else 0) for t in tt.tiers}
    assert sum(exp_req.values()) > 0 and sum(exp_msgs.values()) > 0  # something to reconcile

    d, r, s = launch.summaries("drain"), launch.summaries("relay"), launch.summaries("source")
    assert sorted(d) == sorted(r) == sorted(s) == list(range(len(tt)))  # every tier reported
    assert per_tier(d, "counts.planned") == exp_req
    assert per_tier(d, "counts.sent") == exp_req
    assert per_tier(d, "counts.ok") == exp_req
    assert per_tier(d, "counts.failed") == {k: 0 for k in exp_req}
    assert per_tier(s, "counts.streams_accepted") == exp_req
    assert per_tier(s, "counts.streams_completed") == exp_req
    assert per_tier(r, "counts.streams_accepted") == exp_req
    assert per_tier(r, "counts.streams_completed") == exp_req
    # message hops: source yielded == relay read == relay forwarded == drain read == expected
    assert per_tier(s, "counts.msgs_sent") == exp_msgs
    assert per_tier(r, "counts.msgs_received") == exp_msgs
    assert per_tier(r, "counts.msgs_forwarded") == exp_msgs
    assert per_tier(d, "msgs.msgs") == exp_msgs
    for seq, docs in d.items():
        for doc in docs:
            assert doc["msgs"]["ok_requests_with_unexpected_msg_count"] in (0, None)
            assert doc["overflow"] == {"requests": 0, "msgs": 0, "lag": 0}
            assert set(doc["recv_types"]) <= {"bytes"}
            assert doc["n_workers"] == 2 and doc["local_subchannel_pool"] is True
    assert sum(sum(doc["recv_types"].get("bytes", 0) for doc in docs)
               for docs in d.values()) == sum(exp_req.values())
    assert all(set(doc["request_types"]) <= {"bytes"} for docs in s.values() for doc in docs)

    # beat source: one step row per planned step, stamped drain rows = every message
    steps, _ = launch.records("source", "steps")
    assert len(steps["step"]) == BeatSchedule(tt).n_steps
    msgs, mh = launch.records("drain", "msgs")
    assert len(msgs["stream_id"]) == sum(exp_msgs.values())
    assert all(h["meta"]["overflow"] == 0 for h in mh)
    assert min(rv - pl for rv, pl in zip(msgs["recv_rel_ns"], msgs["planned_rel_ns"])) >= 0
    streams, _ = launch.records("source", "streams")
    assert sorted(streams["req_id"]) == list(range(sum(exp_req.values())))
    reqs, _ = launch.records("drain", "requests")
    assert sorted(reqs["req_id"]) == list(range(sum(exp_req.values())))
    assert all(sent >= pl for sent, pl in zip(reqs["sent_rel_ns"], reqs["planned_rel_ns"]))

    # backlog readings are back to zero at the end, for every expected process
    snaps = snapshot_roles(launch.shm)
    assert len(snaps["drain"]) == 3 and len(snaps["relay"]) == 1 and len(snaps["source"]) == 1
    assert sum(x.values["backlog.inflight"] for x in snaps["drain"]) == 0
    assert sum(x.values["backlog.active_streams"] for x in snaps["drain"]) == 0
    assert snaps["relay"][0].values["backlog.queue_msgs"] == 0
    assert snaps["source"][0].values["backlog.pending_msgs"] == 0
    total_msgs = sum(exp_msgs.values())
    assert snaps["source"][0].values["backlog.yielded"] == total_msgs
    assert snaps["relay"][0].values["backlog.read"] == total_msgs
    assert snaps["relay"][0].values["backlog.yielded"] == total_msgs
    assert sum(x.values["backlog.read"] for x in snaps["drain"]) == total_msgs

    # ready file of the multi-process drain lists both children
    _ident, ready = read_json_stamped(launch.tree.ready_file("drain"), "capkit.ready/1")
    assert sorted(c["worker"] for c in ready["children"]) == [0, 1]


def test_drop_injection_breaks_reconciliation(launch):
    """Positive control for the reconciliation above: drop every 5th message at the drain.
    The drain's count must fall short of the source's by exactly the applied drops."""
    tt = launch.write_table(table_doc([("warmup", 400, 20.0, {}), ("drain", 400, 0.0, {})]))
    launch.set_origin(4.0)
    launch.server("source", "capgrpc.nullgrpc", ["--workers", "1", "--in-process", "--msg-bytes", "32",
                                         "--pacing", "beat", "--stamp"])
    launch.wait_file(launch.tree.port_file("source"), 15, "source")
    launch.start("drain", "capgrpc.load",
                 drain_args(launch, "source", 1, ["--stamped", "--in-process",
                                                  "--inject", "site=drop.msg,mode=drop,every_n=5"]))
    assert launch.wait_exit("drain", 20) == 0, launch.stderr("drain")
    assert launch.stop("source") == 0
    s, d = launch.summaries("source"), launch.summaries("drain")
    sent = sum(per_tier(s, "counts.msgs_sent").values())
    got = sum(per_tier(d, "msgs.msgs").values())
    drops = sum(i["applied_delta"] for docs in d.values() for doc in docs for i in doc["inject"]
                if i["site"] == "drop.msg")
    assert sent > 0 and drops > 0
    assert got == sent - drops
    assert sum(doc["msgs"]["ok_requests_with_unexpected_msg_count"] or 0
               for docs in d.values() for doc in docs) > 0
    hits = sum(i["hits_delta"] for docs in d.values() for doc in docs for i in doc["inject"]
               if i["site"] == "drop.msg")
    assert hits == sent
    assert drops == (hits - 1) // 5 + 1  # every_n=5, phase 0, counted since activation

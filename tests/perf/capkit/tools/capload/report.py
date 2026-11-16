# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-of-run output of capload and capload.nullsse (INTERFACES.md 5.4): one summary.json per
worker per tier of the table, and the per-tier capbin record files. Runs after the last tier,
never on the measured path.

Archiving rule (INTERFACES 5.4): per-request rows (requests, streams) by the request's planned
tier; per-event rows (chunks, fionread, lag, conns) by the tier of the event time
(TierTable.tier_seq_at; -1 -> tiers/pre, -2 -> tiers/post). A record file is written for every
tier of the table even when it has no rows, so a missing file is never read as "zero rows".
"""

import array
import os

from capcommon.capbin import write_capbin
from capcommon.inject import tier_report
from capcommon.jsonio import write_json_stamped
from capcommon.runtree import RunTree

from ._util import hist_summary

SUMMARY_SCHEMA = "capkit.tier_summary/1"


def dump_per_tier(rt, ra, kind, name, ident, table, meta):
    """Write ra split by its tier_seq column; empty files for table tiers without rows."""
    role, worker = ident.role, ident.worker
    runs = ra.split_runs("tier_seq")
    seqs = sorted(set(runs) | {t.seq for t in table.tiers})
    for seq in seqs:
        label = table.label_of(seq)
        path = rt.tier_file(seq, label, role, worker, name, "capbin")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cols = []
        for fname, tc in ra.fields:
            a = array.array(tc)
            col = ra.cols[fname]
            for lo, hi in runs.get(seq, ()):
                a.extend(col[lo:hi])
            cols.append((fname, a))
        m = dict(meta)
        m.update({"overflow": ra.overflow, "capacity": ra.capacity, "split_key": "tier_seq",
                  "split_value": seq})
        write_capbin(path, kind, ident.with_tier(seq, label), cols, m)


def _rows(ra):
    cols = [ra.cols[n] for n in ra.names]
    for i in range(ra.n):
        yield tuple(c[i] for c in cols)


def _inject_entries(marks_start, marks_end, seq):
    a = marks_start.get(seq)
    b = marks_end.get(seq)
    if a is None or b is None:
        return [], None
    return tier_report(a[0], b[0]), {"start_rel_ns": a[1], "end_rel_ns": b[1]}


def _lag_by_tier(ra):
    out = {}
    lt, tc = ra.cols["late_ns"], ra.cols["tier_seq"]
    for i in range(ra.n):
        out.setdefault(tc[i], []).append(lt[i])
    return out


def write_loadgen(w, stop_doc):
    from .loadgen import CLASS_HINT, CONNECT_FAILURES, CLOSE_NAMES, ST_OK, status_name  # noqa
    rt = RunTree(w.cfg["run_dir"])
    table = w.table
    used_status = set()
    per = {}
    for (rid, planned, sent, first, done, frames, nbytes, tokens, st, cid, newc,
         seq) in _rows(w.rec_req):
        d = per.get(seq)
        if d is None:
            d = per[seq] = {"ok": 0, "fail": {}, "frames": 0, "bytes": 0, "tokens": 0,
                            "lag": [], "ttft_p": [], "ttft_s": [], "e2e_p": [], "e2e_s": [],
                            "rows": 0, "new_conn": 0}
        d["rows"] += 1
        d["frames"] += frames
        d["bytes"] += nbytes
        d["new_conn"] += newc
        if tokens > 0:
            d["tokens"] += tokens
        used_status.add(st)
        if st == ST_OK:
            d["ok"] += 1
            d["e2e_p"].append(done - planned)
            if sent >= 0:
                d["e2e_s"].append(done - sent)
        else:
            d["fail"][st] = d["fail"].get(st, 0) + 1
        if sent >= 0:
            d["lag"].append(sent - planned)
        if first >= 0:
            d["ttft_p"].append(first - planned)
            if sent >= 0:
                d["ttft_s"].append(first - sent)
    fion = {}
    fb, ft = w.rec_fion.cols["bytes"], w.rec_fion.cols["tier_seq"]
    for i in range(w.rec_fion.n):
        f = fion.setdefault(ft[i], [0, 0, 0])
        b = fb[i]
        f[0] += 1
        if b > 0:
            f[1] += 1
            if b > f[2]:
                f[2] = b
    lag = _lag_by_tier(w.rec_lag)
    status_codes = {str(s): status_name(s) for s in sorted(used_status | {ST_OK})}
    config = {"mode": w.mode, "conn_policy": w.cfg["conn_policy"],
              "pool_pick": w.cfg["pool_pick"], "pool_seed": w.cfg["pool_seed"],
              "pool_reset": w.cfg["pool_reset"], "recv_mode": w.cfg["recv_mode"],
              "chunk_sample_mod": w.sample_mod, "fionread_every_ms": w.cfg["fionread_every_ms"],
              "lag_period_us": w.cfg["lag_period_us"], "max_records": w.cfg["max_records"],
              "slow_read_pause_us": w.cfg["slow_read_pause_us"], "endpoint": w.cfg["endpoint"],
              "model": w.cfg["model"], "target": "%s:%d" % w.target,
              "wave_size": w.cfg["wave_size"], "wave_delays_ns": w.cfg["wave_delays_ns"],
              "wave_gap_ns": w.cfg["wave_gap_ns"], "loop_impl": w.versions["loop_impl"],
              "cpu": w.affinity, "timeout_from": "planned",
              "close_reasons": CLOSE_NAMES}
    for t in table.tiers:
        seq = t.seq
        d = per.get(seq) or {"ok": 0, "fail": {}, "frames": 0, "bytes": 0, "tokens": 0,
                             "lag": [], "ttft_p": [], "ttft_s": [], "e2e_p": [], "e2e_s": [],
                             "rows": 0, "new_conn": 0}
        planned = w.planned.get(seq, 0)
        sent = w.dispatched_by_tier.get(seq, 0)
        failed = sum(d["fail"].values())
        failures = [{"reason": status_name(s), "count": c,
                     "class_hint": CLASS_HINT.get(status_name(s), "unclassified")}
                    for s, c in sorted(d["fail"].items())]
        n_conn_fail = sum(f["count"] for f in failures if f["reason"] in CONNECT_FAILURES)
        ev = w.ev.get(seq, {})
        inj, inj_marks = _inject_entries(w.marks_start, w.marks_end, seq)
        f = fion.get(seq, [0, 0, 0])
        payload = {
            "tool": "capload", "n_workers": w.W, "schedule_sha256": w.sched_sha,
            "tier_table_sha256": table.sha256, "request_timeout_ns": table.request_timeout_ns,
            "counts": {"planned": planned, "sent": sent, "ok": d["ok"], "failed": failed,
                       "not_sent": planned - sent, "recorded": d["rows"],
                       "unfinished": sent - d["rows"]},
            "failures": failures,
            "failures_split": {"connect": n_conn_fail, "request": failed - n_conn_fail},
            "frames": {"frames": d["frames"], "bytes": d["bytes"],
                       "completion_tokens": d["tokens"]},
            "send_lag_ns": hist_summary(d["lag"]),
            "latency_ns": {"first_from_planned": hist_summary(d["ttft_p"]),
                           "first_from_sent": hist_summary(d["ttft_s"]),
                           "done_from_planned_ok": hist_summary(d["e2e_p"]),
                           "done_from_sent_ok": hist_summary(d["e2e_s"])},
            "fionread": {"samples": f[0], "nonzero": f[1], "max_bytes": f[2]},
            "conn_pool": {"size_max": ev.get("size_max", 0), "new_conns": ev.get("new_conns", 0),
                          "reconnects": ev.get("reconnects", 0),
                          "peer_idle_closed": ev.get("peer_idle_closed", 0),
                          "conn_fail": ev.get("conn_fail", 0),
                          "pool_reset_closed": ev.get("pool_reset_closed", 0),
                          "unsolicited": ev.get("unsolicited", 0),
                          "requests_on_new_conn": d["new_conn"]},
            "inject": inj, "inject_marks": inj_marks,
            "loop_lag_ns": hist_summary(lag.get(seq, [])),
            "overflow": {"requests": w.rec_req.overflow, "chunks": w.rec_chunk.overflow,
                         "fionread": w.rec_fion.overflow, "lag": w.rec_lag.overflow,
                         "conns": w.rec_conn.overflow},
            "stop": stop_doc, "stop_reason": w.stop_reason, "stopped_early": w.stopped_early,
            "config": config, "status_codes": status_codes,
        }
        path = rt.tier_file(seq, t.label, w.role, w.k, "summary", "json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_json_stamped(path, SUMMARY_SCHEMA, w.ident.with_tier(seq, t.label), payload)
    common_meta = {"n_workers": w.W, "worker": w.k, "schedule_sha256": w.sched_sha,
                   "tier_table_sha256": table.sha256, "mode": w.mode}
    dump_per_tier(rt, w.rec_req, "loadgen.requests", "requests", w.ident, table,
                  dict(common_meta, status_codes=status_codes))
    dump_per_tier(rt, w.rec_chunk, "loadgen.chunks", "chunks", w.ident, table,
                  dict(common_meta, chunk_sample_mod=w.sample_mod,
                       nbytes_meaning="bytes of the read call that completed the frame",
                       tier_rule="event time"))
    dump_per_tier(rt, w.rec_fion, "loadgen.fionread", "fionread", w.ident, table,
                  dict(common_meta, every_ms=w.cfg["fionread_every_ms"]))
    dump_per_tier(rt, w.rec_lag, "loop.lag", "lag", w.ident, table,
                  dict(common_meta, period_us=w.cfg["lag_period_us"]))
    dump_per_tier(rt, w.rec_conn, "loadgen.conns", "conns", w.ident, table,
                  dict(common_meta, close_reasons={str(k): v for k, v in CLOSE_NAMES.items()},
                       peer_host=w.target[0], tier_rule="connection open time"))


def write_sink(s, stop_doc):
    """nullsse worker summaries and records."""
    rt = RunTree(s.cfg["run_dir"])
    table = s.table
    lag = _lag_by_tier(s.rec_lag)
    config = {"pacing": s.cfg["pacing"], "chunk_bytes": s.cfg["chunk_bytes"],
              "idle_close_s": s.cfg["idle_close_s"], "lag_period_us": s.cfg["lag_period_us"],
              "max_records": s.cfg["max_records"], "loop_impl": s.versions["loop_impl"],
              "cpu": s.affinity, "gate": "product", "transfer": "chunked"}
    for t in table.tiers:
        seq = t.seq
        c = s.counts.get(seq, {})
        inj, inj_marks = _inject_entries(s.marks_start, s.marks_end, seq)
        counts = {k: c.get(k, 0) for k in s.COUNT_KEYS}
        payload = {
            "tool": "nullsse", "n_workers": s.W, "schedule_sha256": None,
            "tier_table_sha256": table.sha256, "counts": counts,
            "inject": inj, "inject_marks": inj_marks,
            "loop_lag_ns": hist_summary(lag.get(seq, [])),
            "overflow": {"streams": s.rec_streams.overflow, "lag": s.rec_lag.overflow},
            "stop": stop_doc, "stop_reason": s.stop_reason, "config": config,
            "count_rules": s.COUNT_RULES,
        }
        path = rt.tier_file(seq, t.label, s.role, s.k, "summary", "json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_json_stamped(path, SUMMARY_SCHEMA, s.ident.with_tier(seq, t.label), payload)
    meta = {"n_workers": s.W, "worker": s.k, "tier_table_sha256": table.sha256,
            "pacing": s.cfg["pacing"], "gate": "product"}
    dump_per_tier(rt, s.rec_streams, "server.streams", "streams", s.ident, table,
                  dict(meta, tier_rule="request header x-capload-tier (planned tier)"))
    dump_per_tier(rt, s.rec_lag, "loop.lag", "lag", s.ident, table,
                  dict(meta, period_us=s.cfg["lag_period_us"]))

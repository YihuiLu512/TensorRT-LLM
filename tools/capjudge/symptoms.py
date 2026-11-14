# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""External symptom X (design 2.6): X2 planned-time lag trend is the one piece of evidence;
X1 delivery ratio (normalised by the same launch's baseline tier) and X3 backlog trend only
reconcile the counters. Overload-type failures make X hold. Internal localisation I and the
per-tier part of the intervention test V enter through the `hooks` of evaluate_x (judge mode
"P+X+I+V": capjudge.localize and capjudge.intervene); without hooks (mode "P+X") they are
NOT_REQUESTED.

Statistic keys ("<T>:<quantity>:<role>") are listed in README.md; each tier's computed values
are written to verdicts.json `stats` so the pilot builder can turn sub-saturation tiers into
noise bands with exactly the same code path.
"""

import bisect

from capcommon import expect
from capcommon.schedule import BeatSchedule

from . import stats
from .thresholds import band_key
from .treedata import NS, per_second

TRUE, FALSE, NOT_RUN = "TRUE", "FALSE", "NOT_RUN"
NOT_REQUESTED = "NOT_REQUESTED"
# Load-generator failure reasons that are connection failures (INTERFACES 5.4.1): never
# overload-type, whatever the hint (design 2.6: a refused connection is non-overload; a failed
# or reset connection is not a registered overload outcome either).
LOADGEN_CONNECTION_REASONS = ("conn_refused", "conn_fail", "conn_reset")
X3_EXCLUDED_GAUGES = ("backlog.unread_bytes",)  # INTERFACES 8.2: record only


def _window(tier):
    return tier.window_start_ns, tier.window_end_ns


def request_lag_series(rows, tier, which):
    """Per-second median lag over the steady window, bucketed by planned time.

    which: 'first' (first data frame - planned, successful requests) or 'send' (sent - planned,
    every sent request)."""
    ws, we = _window(tier)
    times, lags = [], []
    planned = rows.get("planned_rel_ns", [])
    status = rows.get("status", [])
    col = rows.get("first_rel_ns" if which == "first" else "sent_rel_ns", [])
    for i, p in enumerate(planned):
        v = col[i]
        if v < 0 or (which == "first" and status[i] != 0):
            continue
        times.append(p)
        lags.append(v - p)
    return per_second(times, lags, ws, we), lags


def msg_lag_from_chunks(td, ctx, role, beat):
    """Message-level lag for SSE: join the receiver's chunks with the server's streams by req_id;
    content frame j is planned at BeatSchedule.planned_rel_ns(first_step + offsets[j]).
    Returns (series, n_used, reason_if_not_run)."""
    chunks, _ = td.rows("loadgen.chunks", role)
    if not chunks.get("req_id"):
        return None, 0, "no chunks"
    streams, _ = td.rows("server.streams")
    first_step = dict(zip(streams.get("req_id", []), streams.get("first_step", [])))
    t = td.tier
    try:
        offsets = expect.msg_step_offsets(t.max_tokens, t.tokens_per_msg, t.stream_interval,
                                          "product")
    except expect.ExpectError as e:
        return None, 0, str(e)
    ws, we = _window(t)
    times, lags, asap, unjoined = [], [], 0, 0
    for rid, idx, recv in zip(chunks["req_id"], chunks["idx"], chunks["recv_rel_ns"]):
        fs = first_step.get(rid)
        if fs is None:
            unjoined += 1
            continue
        if fs < 0:
            asap += 1
            continue
        if idx >= len(offsets):
            continue  # usage frame: no step
        step = fs + offsets[idx]
        if not 0 <= step < beat.n_steps:
            unjoined += 1
            continue
        planned = beat.planned_rel_ns(step)
        times.append(planned)
        lags.append(recv - planned)
    if asap:
        return None, 0, "asap pacing (no planned step), INTERFACES 3.4"
    if unjoined:
        return None, 0, "%d chunk rows without a server stream / step" % unjoined
    return per_second(times, lags, ws, we), len(lags), None


def msg_lag_from_msgs(td, role):
    rows, _ = td.rows("drain.msgs", role)
    if not rows.get("planned_rel_ns"):
        return None, 0, "no msgs"
    ws, we = _window(td.tier)
    lags = [r - p for p, r in zip(rows["planned_rel_ns"], rows["recv_rel_ns"])]
    return per_second(rows["planned_rel_ns"], lags, ws, we), len(lags), None


def delivery_ratio(rows, tier):
    ws, we = _window(tier)
    offered = sum(1 for p in rows.get("planned_rel_ns", []) if ws <= p < we)
    delivered = sum(1 for d, s in zip(rows.get("done_rel_ns", []), rows.get("status", []))
                    if s == 0 and ws <= d < we)
    if offered == 0:
        return None
    return delivered / offered


def inflight_series(rows, tier):
    """Derived in-flight count (sent <= t < done) sampled at the middle of each window second."""
    ws, we = _window(tier)
    sent = sorted(s for s in rows.get("sent_rel_ns", []) if s >= 0)
    done = sorted(d for s, d in zip(rows.get("sent_rel_ns", []), rows.get("done_rel_ns", []))
                  if s >= 0 and d >= 0)
    k = (we - ws) // NS
    out = []
    for i in range(k):
        t = ws + i * NS + NS // 2
        out.append(bisect.bisect_right(sent, t) - bisect.bisect_right(done, t))
    return out


def gauge_series(ld, td, role):
    """{slot: per-second series} of the role's backlog.* gauges from sidecar/snapshots.jsonl
    (sum over the role's processes of the last value seen in each second). Slot kinds come
    from the window snapshot copies' slot tables (a counter's trend is not a backlog)."""
    sc = ld.sidecar("snapshots")
    if sc is None:
        return {}
    pids = {p["pid"] for p in ld.procs_of_role(role)}
    gauges = set()
    for pid in pids:
        for snap in td.snaps.get(pid, []):
            for s in snap.meta.get("slots", []):
                if s["kind"] == "gauge" and s["name"].startswith("backlog.") \
                        and s["name"] not in X3_EXCLUDED_GAUGES:
                    gauges.add(s["name"])
    ws, we = _window(td.tier)
    k = (we - ws) // NS
    out = {}
    for g in sorted(gauges):
        last = {}  # (second, pid) -> (snap_t, value)
        for r in sc[1]:
            if r["pid"] not in pids:
                continue
            t = r["snap_t_rel_ns"]
            if not ws <= t < ws + k * NS or g not in r["counters"]:
                continue
            key = ((t - ws) // NS, r["pid"])
            if key not in last or last[key][0] <= t:
                last[key] = (t, r["counters"][g])
        series = []
        for i in range(k):
            vals = [v for (sec, _pid), (_t, v) in last.items() if sec == i]
            series.append(sum(vals) if len(vals) == len(pids) else None)
        out[g] = series
    return out


def failures_by_class(td, roles=None):
    """Sum the per-worker `failures` of every summary in the tier, by class and by category
    (reclassify_failure). `roles` is launch.json's {role: {measure_class...}} (None: unknown)."""
    out = {"overload": 0, "non_overload": 0, "by_reason": {}, "by_category": {}}
    for role, per_worker in td.summaries.items():
        mc = ((roles or {}).get(role) or {}).get("measure_class")
        for doc in per_worker.values():
            for f in doc.get("failures") or []:
                cls, cat = reclassify_failure(role, f, mc)
                n = int(f.get("count", 0))
                out["overload" if cls == "overload" else "non_overload"] += n
                key = "%s:%s" % (role, f.get("reason"))
                out["by_reason"][key] = out["by_reason"].get(key, 0) + n
                out["by_category"][cat] = out["by_category"].get(cat, 0) + n
    return out


def reclassify_failure(role, failure, measure_class=None):
    """(class, category) of one failure entry (design 2.6: the class comes from the reason, not
    the status code; unknown -> non-overload).

    * a connection failure of a load generator (reason in LOADGEN_CONNECTION_REASONS, reported
      by a load_source / under_test client role or a role of unknown class) ->
      ("non_overload", "loadgen_connection"), even with an 'overload' hint;
    * class_hint 'overload' (the registered timeout) -> ("overload", "registered_overload");
    * anything else, including 'unclassified' -> ("non_overload", "unclassified" or the hint).
    Server-log reasons (503 at the per-request backlog cap, watchdog aborts) are not read: no
    capkit tool writes them yet, so they stay non-overload."""
    reason, hint = failure.get("reason"), failure.get("class_hint")
    if reason in LOADGEN_CONNECTION_REASONS and measure_class in (None, "load_source",
                                                                  "under_test"):
        return "non_overload", "loadgen_connection"
    if hint == "overload":
        return "overload", "registered_overload"
    return "non_overload", hint if hint == "non_overload" else "unclassified"


def _trend(series, key, bands, record):
    """T1 on a per-second series against the band of `key`. Returns (direction, entry):
    direction 'above' (positive trend outside the band), 'below', 'inside', or None (NOT_RUN)."""
    res = stats.t1_trend(series) if series is not None else None
    band = bands.get(key)
    entry = {"key": key, "D": None if res is None else res[0], "band": band,
             "n_seconds": len(series) if series is not None else 0}
    if res is not None:
        record.setdefault(key, []).append(res[0])
    if res is None:
        entry["status"] = NOT_RUN
        entry["reason"] = "fewer than one value in a half of the window"
        return None, entry
    if band is None:
        entry["status"] = NOT_RUN
        entry["reason"] = "no registered band for %s" % key
        return None, entry
    d = stats.compare_to_band(res[0], band)
    entry["status"] = d
    return d, entry


def evaluate_x(ctx, ld, td, base_td, record, hooks=None):
    """X for one load tier. `record` collects every computed statistic (key -> [values]).

    Returns a dict: X2/X1/X3 entries, overload_failures, holds (True/False/None=NOT_RUN),
    contradiction (list), reasons, and I / V: hooks["I"](x) and hooks["V"]() when given (judge
    mode P+X+I+V), else {"status": NOT_REQUESTED}."""
    tier = td.tier
    beat = BeatSchedule(ctx.tt)
    bands = ctx.bands.view(ctx.tier_regime[tier.seq])
    out = {"X2": [], "X1": [], "X3": [], "reasons": []}
    fails = failures_by_class(td, getattr(ctx, "roles", None))
    out["failures_by_category"] = fails["by_category"]
    out["overload_failures"] = fails["overload"]
    load_roles = td.roles_with("loadgen.requests")
    x2_dirs = []
    for role in load_roles:
        rows, _ = td.rows("loadgen.requests", role)
        series, _ = request_lag_series(rows, tier, "first")
        d, e = _trend(series, band_key("T1", "req_first_lag_p50_ns", role), bands, record)
        e["layer"] = "request"
        out["X2"].append(e)
        x2_dirs.append(d)
        if ctx.roles.get(role, {}).get("measure_class") == "under_test":
            series, _ = request_lag_series(rows, tier, "send")
            d, e = _trend(series, band_key("T1", "send_lag_p50_ns", role), bands, record)
            e["layer"] = "send"
            out["X2"].append(e)
            x2_dirs.append(d)
        if tier.beat_ns > 0 and td.rows("loadgen.chunks", role)[0].get("req_id"):
            series, n, why = msg_lag_from_chunks(td, ctx, role, beat)
            d, e = _trend(series, band_key("T1", "msg_lag_p50_ns", role), bands, record)
            e.update({"layer": "message", "n_msgs": n})
            if why:
                e.update({"status": NOT_RUN, "reason": why})
                d = None
            out["X2"].append(e)
            x2_dirs.append(d)
    for role in td.roles_with("drain.msgs"):
        series, n, why = msg_lag_from_msgs(td, role)
        d, e = _trend(series, band_key("T1", "msg_lag_p50_ns", role), bands, record)
        e.update({"layer": "message", "n_msgs": n})
        out["X2"].append(e)
        x2_dirs.append(d)
    # X1 / X3 (reconciliation quantities)
    x1_dirs, x3_dirs = [], []
    for role in load_roles:
        rows, _ = td.rows("loadgen.requests", role)
        r = delivery_ratio(rows, tier)
        b = delivery_ratio(base_td.rows("loadgen.requests", role)[0], base_td.tier) \
            if base_td is not None else None
        key = band_key("T2", "delivery_ratio_norm", role)
        norm = None if (r is None or not b) else r / b
        if norm is not None and base_td is not td:
            # the baseline tier's own normalised ratio is 1 by construction, not a sample
            record.setdefault(key, []).append(norm)
        band = bands.get(key)
        e = {"key": key, "ratio": r, "baseline_ratio": b, "normalised": norm, "band": band}
        if norm is None or band is None:
            e["status"] = NOT_RUN
            e["reason"] = "no baseline tier ratio" if norm is None else "no registered band"
            x1_dirs.append(None)
        else:
            e["status"] = stats.compare_to_band(norm, band)
            x1_dirs.append(e["status"])
        out["X1"].append(e)
        d, e = _trend(inflight_series(rows, tier), band_key("T1", "inflight_derived", role),
                      bands, record)
        out["X3"].append(e)
        x3_dirs.append(d)
    for role in sorted(ctx.roles):
        for slot, series in gauge_series(ld, td, role).items():
            d, e = _trend(series, band_key("T1", slot, role), bands, record)
            out["X3"].append(e)
            x3_dirs.append(d)
    out = _combine(out, x2_dirs, x1_dirs, x3_dirs)
    hooks = hooks or {}
    out["I"] = hooks["I"](out) if "I" in hooks else {"status": NOT_REQUESTED}
    out["V"] = hooks["V"]() if "V" in hooks else {"status": NOT_REQUESTED}
    return out


def _combine(out, x2_dirs, x1_dirs, x3_dirs):
    """X2 holds if any lag series trends upward outside its band; it is NOT_RUN if none does
    and any series could not be evaluated. At least one reconciliation quantity must be
    computed. Contradiction (VOID), symmetric:
      X2 true  and (X1 significantly above its band, or X3 significantly below its band);
      X2 false and (X1 significantly below its band, or X3 significantly above its band)."""
    if not x2_dirs:
        x2 = NOT_RUN
        out["reasons"].append("X2: no lag series in this tier")
    elif "above" in x2_dirs:
        x2 = TRUE
    elif None in x2_dirs:
        x2 = NOT_RUN
        out["reasons"].append("X2: a lag series is NOT_RUN")
    else:
        x2 = FALSE
    out["X2_status"] = x2
    recon = [d for d in x1_dirs + x3_dirs if d is not None]
    contradiction = []
    if x2 == TRUE:
        contradiction += ["X1 above band (delivery exceeds offer)" for d in x1_dirs
                          if d == "above"]
        contradiction += ["X3 below band (backlog shrinking)" for d in x3_dirs if d == "below"]
    elif x2 == FALSE and out["overload_failures"] == 0:
        # With overload-type failures the lag and backlog are truncated by the cap / watchdog
        # (design 2.6), so a delivery shortfall without a lag trend is expected, not an error.
        contradiction += ["X1 below band (delivery short of offer)" for d in x1_dirs
                          if d == "below"]
        contradiction += ["X3 above band (backlog growing)" for d in x3_dirs if d == "above"]
    out["contradiction"] = contradiction
    if out["overload_failures"] > 0:
        out["holds"] = True
        out["reasons"].append("overload-type failures: %d" % out["overload_failures"])
    elif x2 == NOT_RUN:
        out["holds"] = None
    elif not recon:
        out["holds"] = None
        out["reasons"].append("X1/X3: no reconciliation quantity computed")
    else:
        out["holds"] = x2 == TRUE
    return out

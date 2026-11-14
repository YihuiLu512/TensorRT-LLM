# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common checks (design 2.7; INTERFACES 10.2) and the other preconditions P.

Every check carries expected_count taken from the configuration (schedule meta.per_tier,
launch.json roles[].workers / expected_procs, reconcile, chunk_sample_mod, overlay sites, window
length / sampling period) and the source of that number. Status order RED > NOT_RUN > PASS:
any red evidence is RED even when nothing else could be checked; otherwise expected_count == 0
or checked_count == 0 -> NOT_RUN; never PASS.
"""

import math

from capcommon import expect
from capcommon.cli import parse_cpu_list
from capcommon.exitcodes import ConfigError
from capcommon.gate import check_snapshot, planned_polls
from capcommon.schedule import per_worker_tier_counts
from capcommon.versions import check_versions

from . import calib, stats
from .symptoms import failures_by_class
from .thresholds import band_key
from .treedata import NS, cpu_tick_deltas, first_last_in_window, role_cpus, summary_value

PASS, RED, NOT_RUN = "PASS", "RED", "NOT_RUN"
NOT_IMPLEMENTED = "NOT_IMPLEMENTED_BATCH2"


class Result:
    def __init__(self, check_id, tier_seq, source):
        self.check_id = check_id
        self.tier_seq = tier_seq
        self.source = source
        self.expected = 0
        self.checked = 0
        self.red = 0
        self.not_run = []
        self.detail = []
        self.status = None

    def finish(self, status=None):
        if status is not None:
            self.status = status
        elif self.red > 0:
            self.status = RED
        elif self.expected == 0 or self.checked == 0:
            self.status = NOT_RUN
        elif self.not_run:
            self.status = NOT_RUN
        else:
            self.status = PASS
        return self

    def as_row(self):
        detail = "; ".join(str(d) for d in self.detail + ["NOT_RUN: %s" % n
                                                        for n in self.not_run])
        return {"tier_seq": self.tier_seq, "check_id": self.check_id,
                "expected_count": self.expected, "checked_count": self.checked,
                "red_count": self.red, "status": self.status,
                "source_of_expected": self.source, "detail": detail.replace("\t", " ")
                .replace("\n", " ")[:2000]}


def _roles_of(ctx, measure_class):
    return [r for r in ctx.roles.values() if r["measure_class"] == measure_class]


def client_roles(ctx):
    """Load-generating roles: launch.json roles with kind == 'client' (scenario role kind). None
    when launch.json does not carry `kind` (then the dependent check is NOT_RUN)."""
    if not all("kind" in r for r in ctx.roles.values()):
        return None
    return [r for r in ctx.roles.values() if r["kind"] == "client"]


# ---- C1 load source health, C2 out-of-domain stub health (calibrated shapes) ----------------

def _shape_for(ctx, kind, role):
    """(shape, None) for the shape register.json binds to `role` in the `kind` file, or
    (None, why)."""
    bind = ctx.calib_bind.get(kind)
    if bind is None:
        return None, "no registered %s calibration" % kind
    if role not in bind:
        return None, "the %s entry binds no shape to role %s" % (kind, role)
    return calib.shapes_of(kind, ctx.calib_docs[kind])[bind[role]], None


def _shape_items(r, ctx, kind, shape, tier, role, worker, st, why, split):
    """Compare one worker with its calibrated shape: planned load against the healthy maximum
    (only on tiers the shape's match covers) and every statistic of the kind against the
    shape's band. Above -> RED, except a planned load above a censored maximum (only "at
    least this" is known): NOT_RUN. Returns True when every item was evaluated."""
    names = calib.SHAPE_KINDS[kind][3]
    w = "%s.w%d" % (role, worker)
    ok = True
    mx = calib.max_of(kind, shape)
    if not calib.shape_matches(shape, tier):
        r.not_run.append("%s: tier (tokens_per_msg %s, streams_c %s) outside the calibrated "
                         "match %s" % (w, tier.tokens_per_msg, tier.streams_c, shape["match"]))
        ok = False
    elif mx["value"] is None:
        r.not_run.append("%s: the shape has no healthy maximum" % w)
        ok = False
    else:
        load, lwhy = calib.worker_load(ctx, tier, role, worker, split, mx["load_key"])
        if load is None:
            r.not_run.append("%s: %s" % (w, lwhy))
            ok = False
        elif load > mx["value"] and mx["censored"]:
            r.not_run.append("%s planned %.6g %s beyond the calibrated range: the healthy "
                             "maximum %.6g is censored (only a lower bound)"
                             % (w, load, mx["load_key"], mx["value"]))
            ok = False
        elif load > mx["value"]:
            r.red += 1
            r.detail.append("%s planned %.6g %s above the calibrated healthy maximum %.6g"
                            % (w, load, mx["load_key"], mx["value"]))
    for name in names:
        val, band = st.get(name), shape["bands"].get(name)
        if val is None or band is None:
            r.not_run.append("%s %s: %s" % (w, name, why.get(name, "no value") if val is None
                                            else "no band in the calibrated shape"))
            ok = False
        elif stats.compare_to_band(val, band) == "above":
            r.red += 1
            r.detail.append("%s %s=%s above band max %s" % (w, name, val, band["max"]))
    return ok


def check_c1(ctx, td, record):
    """Per load_source worker against the loadgen_band shape its role is bound to (INTERFACES
    9.6): planned load vs the healthy maximum, and the four health statistics
    (calib.LOADGEN_STATS: send-lag trend T1, send-lag p99, FIONREAD non-zero fraction,
    event-loop lag p99) vs the shape's bands (_shape_items). Above -> RED (planned load
    above a censored maximum: NOT_RUN); no binding, statistic or band -> NOT_RUN. A worker
    without summary or request records -> RED."""
    r = Result("C1", td.tier.seq, "launch.json roles[measure_class=load_source].workers")
    for role in _roles_of(ctx, "load_source"):
        name = role["role"]
        r.expected += role["workers"]
        shape, why = _shape_for(ctx, "loadgen_band", name)
        for k in range(role["workers"]):
            st = calib.loadgen_worker_stats(td, name, k)
            if st is None:
                r.red += 1
                r.detail.append("%s.w%d: summary or requests missing" % (name, k))
                continue
            for stat, val in st.items():
                if val is not None:
                    record.setdefault("%s:%s" % (stat, name), []).append(val)
            if shape is None:
                r.not_run.append("%s.w%d: %s" % (name, k, why))
                continue
            if _shape_items(r, ctx, "loadgen_band", shape, td.tier, name, k, st, {},
                            "schedule"):
                r.checked += 1
    if r.expected == 0:
        r.not_run.append("no load_source role in this launch")
    return r.finish()


def check_c2(ctx, ld, td, record):
    """Out-of-domain stub health (design 2.6 / 2.7 C2; INTERFACES 9.7): per worker of every
    stub_out_of_domain role, against the stub_calib out_of_domain shape its role is bound to:
    planned load (an even split over the workers; NOT_RUN unless the launch has exactly one
    client role, calib.worker_share) vs the healthy maximum, event-loop lag p99
    and thread CPU (cores) vs the shape's bands, as C1 (_shape_items). stub_in_domain roles
    are not a precondition (design 2.6) and are only listed."""
    r = Result("C2", td.tier.seq, "launch.json roles[measure_class=stub_out_of_domain].workers")
    for role in _roles_of(ctx, "stub_out_of_domain"):
        name = role["role"]
        r.expected += role["workers"]
        shape, why = _shape_for(ctx, "stub_calib", name)
        for k in range(role["workers"]):
            st, whys = calib.stub_worker_stats(ld, td, name, k)
            if st is None:
                r.red += 1
                r.detail.append("%s.w%d: %s" % (name, k, whys))
                continue
            for stat, val in st.items():
                if val is not None:
                    record.setdefault("%s:%s" % (stat, name), []).append(val)
            if shape is None:
                r.not_run.append("%s.w%d: %s" % (name, k, why))
                continue
            if _shape_items(r, ctx, "stub_calib", shape, td.tier, name, k, st, whys, "even"):
                r.checked += 1
    if r.expected == 0:
        r.not_run.append("no stub_out_of_domain role in this launch")
    ind = sorted(x["role"] for x in _roles_of(ctx, "stub_in_domain"))
    if ind:
        r.detail.append("stub_in_domain %s: own demand is not a precondition, not checked" % ind)
    return r.finish()


# ---- C3 non-overload failures ----------------------------------------------------------------

def check_c3(ctx, td):
    r = Result("C3", td.tier.seq, "schedule meta.per_tier")
    r.expected = int(ctx.schedule_meta["per_tier"].get(str(td.tier.seq), 0))
    clients = client_roles(ctx)
    if clients is None:
        r.not_run.append("launch.json roles lack `kind`; cannot tell load generators")
        return r.finish(NOT_RUN)
    for role in clients:
        for doc in td.summaries.get(role["role"], {}).values():
            c = doc.get("counts") or {}
            r.checked += int(c.get("ok", 0)) + int(c.get("failed", 0))
    f = failures_by_class(td, ctx.roles)
    r.red = f["non_overload"]
    r.detail.append("non_overload=%d overload=%d by_reason=%s by_category=%s"
                    % (f["non_overload"], f["overload"], f["by_reason"], f["by_category"]))
    if r.checked < r.expected:
        r.not_run.append("%d of %d planned requests have an outcome (ok + failed)"
                         % (r.checked, r.expected))
    return r.finish()


# ---- P-SENT planned == sent per load-generator worker ----------------------------------------

def check_sent(ctx, td):
    r = Result("P-SENT", td.tier.seq, "launch.json roles[kind=client].workers + schedule meta")
    clients = client_roles(ctx)
    if clients is None:
        r.not_run.append("launch.json roles lack `kind`")
        return r.finish(NOT_RUN)
    for role in clients:
        name, w = role["role"], role["workers"]
        r.expected += w
        want = per_worker_tier_counts(ctx.schedule_meta, w)
        for k in range(w):
            doc = td.summaries.get(name, {}).get(k)
            if doc is None:
                r.red += 1
                r.detail.append("%s.w%d: no summary" % (name, k))
                continue
            r.checked += 1
            c = doc.get("counts") or {}
            if doc.get("n_workers") != w:
                r.red += 1
                r.detail.append("%s.w%d: n_workers %r != %d" % (name, k, doc.get("n_workers"), w))
            if c.get("planned") != want.get((k, td.tier.seq), 0):
                r.red += 1
                r.detail.append("%s.w%d: planned %r != schedule %d"
                                % (name, k, c.get("planned"), want.get((k, td.tier.seq), 0)))
            elif c.get("sent") != c.get("planned"):
                r.not_run.append("%s.w%d sent %r != planned %r"
                                 % (name, k, c.get("sent"), c.get("planned")))
            if doc.get("schedule_sha256") != ctx.launch["schedule"]["sha256"] or \
                    doc.get("tier_table_sha256") != ctx.tt.sha256:
                r.red += 1
                r.detail.append("%s.w%d: schedule/tier table sha differ from launch.json"
                                % (name, k))
    return r.finish()


# ---- P-PIDS and RECORDS ----------------------------------------------------------------------

def check_pids(td):
    r = Result("P-PIDS", td.tier.seq, "one tier.json per tier")
    r.expected = 1
    tj = td.tier_json
    if tj is None:
        r.red = 1
        r.detail.append("tier.json missing")
        r.checked = 1
    else:
        r.checked = 1
        if tj.get("pids_stable") is not True:
            r.red = 1
            r.detail.append("pids_stable is %r" % tj.get("pids_stable"))
    return r.finish()


def check_records(td):
    """Record integrity of one tier directory, and its window snapshots: every pid has exactly
    the two copies, and caplaunch's tier.json snapshot_copies (one entry per registered process
    expected to write snapshots, per window edge) has no snapshot_missing entry. A missing copy
    is how a product process whose capprobe stayed off (command line not matching
    CAPPROBE_ENTRY) shows up: RED, not a smaller sum."""
    r = Result("RECORDS", td.tier.seq, "one tier directory per tier")
    r.expected = 1
    if not td.exists:
        r.not_run.append("tier directory missing")
        return r.finish(NOT_RUN)
    r.checked = 1
    r.red = len(td.errors)
    r.detail += td.errors[:20]
    for pid, snaps in td.snaps.items():
        if len(snaps) != 2:
            r.red += 1
            r.detail.append("pid %s: %d window snapshot copies, expected 2" % (pid, len(snaps)))
    copies = (td.tier_json or {}).get("snapshot_copies")
    if not isinstance(copies, list):
        r.red += 1
        r.detail.append("tier.json has no snapshot_copies list")
        return r.finish()
    for c in copies:
        if not isinstance(c, dict):
            r.red += 1
            r.detail.append("tier.json snapshot_copies entry %r is not an object" % (c,))
        elif c.get("snapshot_missing"):
            r.red += 1
            r.detail.append("%s pid %s: no %s window snapshot (%s)" % (
                c.get("role"), c.get("pid"), c.get("which"), c.get("detail")))
        elif c.get("pid") not in td.snaps:
            r.red += 1
            r.detail.append("%s pid %s: snapshot_copies lists %s but no copy was read" % (
                c.get("role"), c.get("pid"), c.get("file")))
    return r.finish()


# ---- C5 / C6 / C7 reconcile chains -----------------------------------------------------------

CHAIN_CHECK = {"requests": "C5", "messages": "C6", "chunks": "C7"}


def slot_deltas(ld, role, slot, t_from, t_to, min_to=None):
    """Counter delta of `slot` for each of the role's registered processes, from
    sidecar/snapshots.jsonl: last row with snap_t_rel_ns <= t_from to last row with
    snap_t_rel_ns <= t_to (that row must also be >= min_to, e.g. the drain gate's pass time).
    Returns ({pid: delta}, None), or (None, reason) when any process lacks a usable pair."""
    sc = ld.sidecar("snapshots")
    if sc is None:
        return None, "sidecar/snapshots.jsonl missing"
    pids = [p["pid"] for p in ld.procs_of_role(role)]
    if not pids:
        return None, "no registered process for role %s" % role
    out = {}
    for pid in pids:
        a = b = None
        for row in sc[1]:
            if row["pid"] != pid or slot not in row["counters"]:
                continue
            t = row["snap_t_rel_ns"]
            if t <= t_from and (a is None or t >= a[0]):
                a = (t, row["counters"][slot])
            if t <= t_to and (b is None or t >= b[0]):
                b = (t, row["counters"][slot])
        if a is None or b is None:
            return None, "pid %d: no snapshot row with %s around the interval" % (pid, slot)
        if min_to is not None and b[0] < min_to:
            return None, "pid %d: last snapshot before the end precedes the drain pass" % pid
        out[pid] = b[1] - a[1]
    return out, None


def _hop_parts(ctx, ld, td, hop, next_drain):
    """One reconcile hop's count in this tier, per unit: ({"w<k>": value} for a summary source,
    {"pid <pid>": delta} for a snapshot slot, from the tier start to the end of the following
    drain tier, after its gate passed), None, False; or (None, why, is_red)."""
    src = hop["source"]
    if src.startswith("summary."):
        role = ctx.roles[hop["role"]]
        docs = td.summaries.get(hop["role"], {})
        missing = [k for k in range(role["workers"]) if k not in docs]
        if missing:
            return None, "workers %s have no summary" % missing, True
        vals = [summary_value(docs[k], src[len("summary."):]) for k in range(role["workers"])]
        if any(not isinstance(v, int) or isinstance(v, bool) for v in vals):
            return None, "%s not an integer in every worker summary" % src, True
        return {"w%d" % k: v for k, v in enumerate(vals)}, None, False
    slot = src[len("slot:"):]
    if next_drain is None or next_drain.drain_gate is None:
        return None, "no following drain gate record", False
    tpass = next_drain.drain_gate.get("t_pass_rel_ns")
    if not next_drain.drain_gate.get("passed") or tpass is None:
        return None, "following drain gate did not pass", False
    per, why = slot_deltas(ld, hop["role"], slot, td.tier.start_ns, next_drain.tier.end_ns,
                           min_to=tpass)
    return (None if per is None else {"pid %d" % p: v for p, v in per.items()}), why, False


def _hop_value(ctx, ld, td, hop, next_drain):
    parts, why, is_red = _hop_parts(ctx, ld, td, hop, next_drain)
    return (None if parts is None else sum(parts.values())), why, is_red


def hop_range(ctx, td, chain, role, ovl=None):
    """(lo, hi) expected count of `role`'s hop on `chain` in this tier: expect.expected_hop_count
    over the schedule's planned requests (hi) and over those without the tier's overload-type
    failures (lo: they may or may not have reached that hop). `ovl`: that failure count when
    the caller has it. Raises ConfigError (capcommon.expect)."""
    t = td.tier
    n = int(ctx.schedule_meta["per_tier"].get(str(t.seq), 0))
    if ovl is None:
        ovl = failures_by_class(td, ctx.roles)["overload"]
    hi = expect.expected_hop_count(chain, n, t.max_tokens, t.tokens_per_msg, t.stream_interval,
                                   role)
    lo = expect.expected_hop_count(chain, n - ovl, t.max_tokens, t.tokens_per_msg,
                                   t.stream_interval, role)
    return lo, hi


def check_chains(ctx, ld, td, next_drain, kind):
    """One Result per chain kind (C5 requests, C6 messages, C7 chunks); every tier-scope hop of
    every chain of that kind is one checked item, expected in hop_range."""
    cid = CHAIN_CHECK[kind]
    r = Result(cid, td.tier.seq, "launch.json reconcile + schedule meta.per_tier")
    ovl = failures_by_class(td, ctx.roles)["overload"]
    for chain in ctx.launch["reconcile"]["chains"]:
        if chain["kind"] != kind:
            continue
        for hop in chain["hops"]:
            if hop["scope"] != "tier":
                continue
            r.expected += 1
            try:
                lo, hi = hop_range(ctx, td, chain, hop["role"], ovl)
            except ConfigError as e:
                r.not_run.append("%s/%s: %s" % (chain["name"], hop["role"], e))
                continue
            v, why, is_red = _hop_value(ctx, ld, td, hop, next_drain)
            if v is None:
                if is_red:
                    r.red += 1
                    r.detail.append("%s/%s: %s" % (chain["name"], hop["role"], why))
                else:
                    r.not_run.append("%s/%s: %s" % (chain["name"], hop["role"], why))
                continue
            r.checked += 1
            if not lo <= v <= hi:
                r.red += 1
                r.detail.append("%s/%s %s=%d expected %s" % (
                    chain["name"], hop["role"], hop["source"], v,
                    hi if lo == hi else "[%d, %d]" % (lo, hi)))
    if kind == "chunks":
        _chunk_rows(ctx, td, r, next_drain)
    return r.finish()


def _chunk_rows(ctx, td, r, next_drain=None):
    """C7 per request: every sampled request (req_id % chunk_sample_mod == 0, planned in this
    tier, completed OK) must have exactly the chain's frames-per-request chunk rows.

    Chunk rows are filed by the tier of their receive time (INTERFACES 5.4), so the rows of a
    request planned near the end of this tier may lie partly in the following tier's files. A
    load tier is always followed by a drain tier (TierTable rule 3, INTERFACES 3.2), so a
    request's rows are counted over this tier's chunk files and those of `next_drain`; rows
    anywhere else are not counted. Stray ids are the rows in this tier's own files whose request
    is not a sampled request planned in this tier."""
    t = td.tier
    chains = [c for c in ctx.launch["reconcile"]["chains"] if c["kind"] == "chunks"]
    for role in td.roles_with("loadgen.chunks"):
        chain = next((c for c in chains if any(h["role"] == role for h in c["hops"])), None)
        if chain is None:
            r.not_run.append("chunks of %s: no chunks chain names this role" % role)
            continue
        chunks, parts = td.rows("loadgen.chunks", role)
        mods = {m.get("chunk_sample_mod") for _r, _w, m in parts}
        if len(mods) != 1 or not isinstance(next(iter(mods)), int) or next(iter(mods)) < 1:
            r.red += 1
            r.detail.append("chunks of %s: chunk_sample_mod missing or inconsistent" % role)
            continue
        mod = mods.pop()
        per = expect.expected_hop_count(chain, 1, t.max_tokens, t.tokens_per_msg,
                                        t.stream_interval, role)
        reqs, _ = td.rows("loadgen.requests", role)
        ok_ids = {rid for rid, st in zip(reqs.get("req_id", []), reqs.get("status", []))
                  if st == 0}
        sampled = [rid for rid in range(len(ctx.schedule_tier))
                   if ctx.schedule_tier[rid] == t.seq and rid % mod == 0]
        r.expected += len(sampled)
        counts = {}
        for rid in chunks.get("req_id", []):
            counts[rid] = counts.get(rid, 0) + 1
        sampled_set = set(sampled)
        stray = [rid for rid in counts if rid not in sampled_set]
        if next_drain is not None:
            later = {}
            for rid in next_drain.rows("loadgen.chunks", role)[0].get("req_id", []):
                if rid in sampled_set:
                    later[rid] = later.get(rid, 0) + 1
            for rid, n in later.items():
                counts[rid] = counts.get(rid, 0) + n
            if later:
                r.detail.append("chunks of %s: %d rows of %d sampled requests counted from tier "
                                "%d (received after this tier's end)"
                                % (role, sum(later.values()), len(later), next_drain.tier.seq))
        bad = 0
        for rid in sampled:
            if rid not in ok_ids:
                continue  # failed requests are C3's business
            r.checked += 1
            if counts.get(rid, 0) != per:
                bad += 1
        if bad or stray:
            r.red += bad + len(stray)
            r.detail.append("chunks of %s: %d sampled requests with != %d rows, %d stray ids"
                            % (role, bad, per, len(stray)))


# ---- C4 output length, C12 doubled processes carry traffic -----------------------------------

def check_c4(ctx, td):
    """Every request planned in this tier that got a response carries exactly the tier's
    max_tokens output tokens (design 2.7 C4). The count of a request is the load generator's
    `tokens` column (>= 0, whatever the request's status: capload counts a short stream too);
    where the generator does not count (-1: capgrpc.load) and the request is OK (status 0),
    the `tokens` of the server.streams row with the same req_id in this tier. The server side
    is read only when the req_id names one stream: with more than one client role in the
    launch (each runs the whole schedule, so req_ids repeat and their streams may reach the
    same server) or more than one server role writing stream records, that part is NOT_RUN.
    A failed request is
    never compared with the server's record (a cancelled or timed-out stream records the
    tokens sent before it ended): without a generator count it is checked only (C3 / X
    account for it). Expected = schedule meta.per_tier x client roles. A count other than
    max_tokens -> RED; an OK request without a count, or a planned request without a record
    -> NOT_RUN."""
    r = Result("C4", td.tier.seq, "schedule meta.per_tier x launch.json roles[kind=client]")
    clients = client_roles(ctx)
    if clients is None:
        r.not_run.append("launch.json roles lack `kind`; cannot tell load generators")
        return r.finish(NOT_RUN)
    seq, want = td.tier.seq, td.tier.max_tokens
    r.expected = int(ctx.schedule_meta["per_tier"].get(str(seq), 0)) * len(clients)
    server = {}
    for ident, _h, cols in td.capbins.get("server.streams", []):
        mine = server.setdefault(ident.get("role"), {})
        for rid, tok, ts in zip(cols["req_id"], cols["tokens"], cols["tier_seq"]):
            if ts == seq:
                mine.setdefault(rid, set()).add(tok)
    names = {x["role"] for x in clients}
    rows = []
    for ident, _h, cols in td.capbins.get("loadgen.requests", []):
        if ident.get("role") not in names:
            continue
        rows += [(ident.get("role"), rid, st, tok) for rid, st, tok, ts in
                 zip(cols["req_id"], cols["status"], cols["tokens"], cols["tier_seq"])
                 if ts == seq]
    need = sorted({role for role, _rid, st, tok in rows if tok < 0 and st == 0})
    ambiguous = None
    if need and (len(server) > 1 or len(clients) > 1):
        ambiguous = ("server-side counts are per req_id but client role(s) %s need them, the "
                     "launch has %d client roles and %d server role(s) %s write stream records"
                     % (need, len(clients), len(server), sorted(server)))
    table = next(iter(server.values())) if len(server) == 1 else {}
    bad, uncounted, unmatched = [], 0, 0
    for role, rid, st, tok in rows:
        if tok >= 0:
            got = {tok}
        elif st != 0:
            r.checked += 1
            continue
        elif ambiguous:
            unmatched += 1
            continue
        else:
            got = table.get(rid, set())
        if got:
            r.checked += 1
            if got != {want}:
                r.red += 1
                bad.append((role, rid, sorted(got)))
        else:
            uncounted += 1
    if bad:
        r.detail.append("%d request(s) with output tokens != max_tokens %d, first: %s"
                        % (len(bad), want, bad[:5]))
    if uncounted:
        r.not_run.append("%d OK request(s) without a token count on either side" % uncounted)
    if unmatched:
        r.not_run.append("%d OK request(s) not compared: %s" % (unmatched, ambiguous))
    if r.checked + uncounted + unmatched < r.expected:
        r.not_run.append("%d of %d planned requests have a record"
                         % (r.checked + uncounted + unmatched, r.expected))
    if r.expected == 0:
        r.not_run.append("no request planned in this tier for %d client role(s)" % len(clients))
    return r.finish()


def _sent_parts(ctx, td, role):
    """Requests each load-generator worker sent in this tier ({"w<k>": counts.sent}), None; or
    (None, why)."""
    docs = td.summaries.get(role, {})
    out, missing = {}, []
    for k in range(ctx.roles[role]["workers"]):
        v = summary_value(docs[k], "counts.sent") if k in docs else None
        if not isinstance(v, int) or isinstance(v, bool):
            missing.append(k)
        else:
            out["w%d" % k] = v
    if missing:
        return None, "workers %s have no integer summary counts.sent" % missing
    return out, None


def check_c12(ctx, ld, td, next_drain):
    """Doubled processes carry traffic (design 2.7 C12). Objects: every role on a tier-scope
    hop of a `requests` reconcile chain that has more than one unit: workers >= 2 for a
    summary source (one unit per worker summary), expected_procs >= 2 for a snapshot slot
    (one unit per registered process). Expected = the units' count from launch.json. Every
    unit's count in this tier must be > 0: a load generator's (kind client, summary source)
    is the requests it sent (summary counts.sent: requests that all timed out still carried
    traffic), any other unit's is the hop's own count. A unit with 0 means the doubling did
    not take effect -> NOT_RUN, never PASS (design 2.7 positive control). A role on several
    request chains is checked once (its first hop)."""
    r = Result("C12", td.tier.seq,
               "launch.json reconcile requests hops x roles[].workers / expected_procs")
    seen = set()
    for chain in ctx.launch["reconcile"]["chains"]:
        if chain["kind"] != "requests":
            continue
        for hop in chain["hops"]:
            role = ctx.roles[hop["role"]]
            summary = hop["source"].startswith("summary.")
            n = role["workers"] if summary else role["expected_procs"]
            if hop["scope"] != "tier" or n < 2 or hop["role"] in seen:
                continue
            seen.add(hop["role"])
            r.expected += n
            if summary and role.get("kind") == "client":
                src = "summary.counts.sent"
                parts, why = _sent_parts(ctx, td, hop["role"])
            else:
                src = hop["source"]
                parts, why, _is_red = _hop_parts(ctx, ld, td, hop, next_drain)
            if parts is None:
                r.not_run.append("%s %s: %s" % (hop["role"], src, why))
                continue
            r.checked += len(parts)
            idle = sorted(u for u, v in parts.items() if v <= 0)
            if idle:
                r.not_run.append("%s %s: %s carried no traffic in this tier"
                                 % (hop["role"], src, idle))
            if len(parts) < n:
                r.not_run.append("%s: %d of %d units have a count" % (hop["role"],
                                                                     len(parts), n))
    if r.expected == 0:
        r.not_run.append("no role on a tier-scope requests hop has two or more units")
    return r.finish()


# ---- C8 probe attached -----------------------------------------------------------------------

def _slot_total(v):
    """Event total of one snapshot slot: a counter's value, or the sum of a histogram's bucket
    counts (lag sites publish loghist/1 slots). None when absent or of another shape."""
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(v, list) and v and all(isinstance(x, int) and not isinstance(x, bool)
                                         for x in v):
        return sum(v)
    return None


def launch_proc_kinds(ctx):
    """Process kinds of this launch: union of launch.json roles[].proc_kinds (product roles
    declare the capprobe process kinds they run, e.g. ["frontend"]). None when no role declares
    them or a declaration is not a list of strings."""
    kinds, declared = set(), False
    for role in ctx.roles.values():
        pk = role.get("proc_kinds")
        if pk is None:
            continue
        if not isinstance(pk, list) or not all(isinstance(k, str) and k for k in pk):
            return None
        declared = True
        kinds.update(pk)
    return kinds if declared else None


def check_c8(ctx, ld, td):
    """Every overlay site whose proc_kind runs in this launch (launch.json roles[].proc_kinds)
    must show a positive window delta on its slots. Sites of process kinds absent from the
    launch are listed as not applicable (neither PASS nor RED); no applicable site -> NOT_RUN."""
    r = Result("C8", td.tier.seq, "overlay_manifest.json sites with proc_kind in "
               "launch.json roles[].proc_kinds")
    ov = ld.overlay
    sites = (ov or {}).get("sites") or {}
    if not sites:
        r.not_run.append("env/overlay_manifest.json missing or lists no sites")
        return r.finish(NOT_RUN)
    kinds = launch_proc_kinds(ctx)
    if kinds is None:
        r.not_run.append("launch.json roles declare no proc_kinds (or a malformed one): cannot "
                         "tell which sites apply")
        return r.finish(NOT_RUN)
    applicable = {k: v for k, v in sites.items() if v.get("proc_kind") is None
                  or v.get("proc_kind") in kinds}
    n_a = sorted("%s(%s)" % (k, v.get("proc_kind")) for k, v in sites.items()
                 if k not in applicable)
    if n_a:
        r.detail.append("not applicable in this topology: %s" % ", ".join(n_a))
    if not applicable:
        r.not_run.append("no site applies to the process kinds %s" % sorted(kinds))
        return r.finish(NOT_RUN)
    r.expected = len(applicable)
    for name, site in sorted(applicable.items()):
        slots = site.get("slots") or []
        if not slots:
            r.red += 1
            r.detail.append("site %s lists no slots" % name)
            continue
        found_all, positive = True, True
        for slot in slots:
            delta, seen = 0, False
            for pid, snaps in td.snaps.items():
                if len(snaps) != 2:
                    continue
                a = _slot_total(snaps[0].values.get(slot))
                b = _slot_total(snaps[1].values.get(slot))
                if a is None or b is None:
                    continue
                seen = True
                delta += b - a
            if not seen:
                found_all = False
            elif delta <= 0:
                positive = False
        r.checked += 1
        if not found_all or not positive:
            r.red += 1
            r.detail.append("site %s: %s" % (name, "slot absent from every window snapshot"
                                             if not found_all else "window delta <= 0"))
    return r.finish()


# ---- C9 pinning readback ---------------------------------------------------------------------

def _inventory_threads(ld):
    """{(pid, tid): inventory entry} over every procs/threads/*.json."""
    out = {}
    for doc, _p in ld.thread_inventories():
        if not isinstance(doc, dict):
            continue
        for th in doc.get("threads") or []:
            out[(doc.get("pid"), th.get("tid"))] = th
    return out


def check_c9(ctx, ld, td):
    """Every sidecar thread row in the window: cpus_allowed within the role's pinned cpus and
    disjoint from every other role's, last_cpu inside the role's cpus. Expected rows = window
    samples (window / sidecar interval) x threads in the thread inventories."""
    r = Result("C9", td.tier.seq, "window_ns / sidecar meta.interval_ns x inventory threads")
    interval = ld.sidecar_interval_ns()
    inv = _inventory_threads(ld)
    th = ld.sidecar("threads")
    cpus = role_cpus(ld.pin_table)
    if interval is None or th is None or not cpus:
        r.not_run.append("sidecar meta.interval_ns / threads.jsonl / pin_table.json missing")
        return r.finish(NOT_RUN)
    ws, we = td.tier.window_start_ns, td.tier.window_end_ns
    r.expected = ((we - ws) // interval) * len(inv)
    by_pid = ld.procs_by_pid()
    for row in th[1]:
        if not ws <= row["t_rel_ns"] < we or row["pid"] not in by_pid:
            continue
        role = by_pid[row["pid"]]["role"]
        mine = cpus.get(role)
        if (row["pid"], row["tid"]) in inv:
            r.checked += 1
        if mine is None:
            r.red += 1
            r.detail.append("role %s has no pin table entry" % role)
            continue
        try:
            allowed = set(parse_cpu_list(row["cpus_allowed"]))
        except (ConfigError, ValueError, TypeError):
            r.red += 1
            r.detail.append("pid %s tid %s: unparsable cpus_allowed %r"
                            % (row["pid"], row["tid"], row["cpus_allowed"]))
            continue
        others = set().union(*[c for k, c in cpus.items() if k != role]) if len(cpus) > 1 \
            else set()
        if not allowed <= mine or allowed & others or row["last_cpu"] not in mine:
            r.red += 1
            if len(r.detail) < 10:
                r.detail.append("t=%d pid %s tid %s role %s: allowed %s last_cpu %s, pinned %s"
                                % (row["t_rel_ns"], row["pid"], row["tid"], role,
                                   sorted(allowed), row["last_cpu"], sorted(mine)))
    if r.checked < r.expected:
        r.red += 1
        r.detail.append("only %d of %d expected thread rows in the window"
                        % (r.checked, r.expected))
    return r.finish()


# ---- C10 version self-report (launch level; failure -> NOT_RUN per D11) ----------------------

def check_c10(ctx, ld):
    r = Result("C10", "launch", "launch.json roles[].expected_procs")
    recs, errors = ld.version_records()
    records, decls = [], {}
    for name, role in sorted(ctx.roles.items()):
        decls[name] = {k: role[k] for k in ("asyncio", "uses_grpc", "probe_tier", "loop")}
        pids = sorted(p["pid"] for p in ld.procs_of_role(name))
        got = [(name, pid, recs.get(pid)) for pid in pids]
        while len(got) < role["expected_procs"]:
            got.append((name, None, None))
        records += got
    res = check_versions(records, decls)
    r.expected, r.checked = res["expected_count"], res["checked_count"]
    r.detail += errors
    if res["status"] != "PASS" or errors:
        r.not_run += res["problems"] or ["version records incomplete"]
        return r.finish(NOT_RUN)
    return r.finish()


# ---- GPU-IDLE (launch level) -----------------------------------------------------------------

def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def launch_pids(ld, tds):
    """{pid: how it belongs} for the processes of this launch: roles.json procs and aux_helpers,
    and every pid the window-boundary scans (tier.json scans: descendants, monitor_descendants,
    aux_helpers, all [pid, starttime, ...]) recorded under them."""
    rj = ld.roles_json or {}
    own = {}
    for p in rj.get("procs") or []:
        if isinstance(p, dict):
            own[p.get("pid")] = "registered process of role %s" % p.get("role")
    for p in rj.get("aux_helpers") or []:
        if isinstance(p, dict):
            own.setdefault(p.get("pid"), "aux helper of role %s" % p.get("role"))
    for td in tds:
        for edge, sc in sorted(((td.tier_json or {}).get("scans") or {}).items()):
            for k in ("descendants", "monitor_descendants", "aux_helpers"):
                for e in (sc or {}).get(k) or []:
                    if isinstance(e, list) and e:
                        own.setdefault(e[0], "%s in the tier %d %s scan" % (k, td.tier.seq, edge))
    own.pop(None, None)
    return own


def check_gpu_idle(ctx, ld, load_tiers, own_pids=None):
    """Returns (Result, void: bool). void=True when any window row shows a busy GPU.

    Expected rows per window = window_ns // sidecar meta.gpu_every_ns (the sidecar samples on
    origin-aligned ticks of that period) x gpu.jsonl gpu_count. A row counts as checked only
    when it is readable: query_rc 0, a numeric util_pct and a compute_pids list. Busy evidence
    is RED even on a row whose other reading is missing: util_pct > 0, or a compute pid that is
    not a process of this launch (own_pids, see launch_pids; a null pid is never this launch's).
    A process of this launch holding a CUDA context at 0 % utilisation (e.g. a null executor's
    startup CUDA call) is not foreign occupancy: it is listed per pid in the detail."""
    r = Result("GPU-IDLE", "launch",
               "sum(window_ns // sidecar meta.gpu_every_ns) x gpu.jsonl gpu_count")
    gi = ld.gpu_idle or {}
    g = ld.sidecar("gpu")
    if not gi.get("positive_control_ok") or g is None:
        r.not_run.append("gpu_idle.json positive control not ok, or gpu.jsonl missing")
        return r.finish(NOT_RUN), False
    if g[0].get("positive_control_ok") is not True:
        r.not_run.append("gpu.jsonl header positive_control_ok %r (sidecar's own GPU listing)"
                         % (g[0].get("positive_control_ok"),))
        return r.finish(NOT_RUN), False
    count = g[0].get("gpu_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        r.not_run.append("gpu_count %r" % (count,))
        return r.finish(NOT_RUN), False
    every = (ld.sidecar_meta or {}).get("gpu_every_ns")
    if not isinstance(every, int) or isinstance(every, bool) or every <= 0:
        r.not_run.append("sidecar meta.json gpu_every_ns %r" % (every,))
        return r.finish(NOT_RUN), False
    r.expected = sum((t.window_ns // every) * count for t in load_tiers)
    own_pids = own_pids or {}
    void = False
    unreadable = 0
    own_seen = {}
    for row in g[1]:
        if not any(t.window_start_ns <= row["t_rel_ns"] < t.window_end_ns for t in load_tiers):
            continue
        util, pids = row.get("util_pct"), row.get("compute_pids")
        foreign = [p for p in pids if p not in own_pids] if isinstance(pids, list) else []
        if (_is_num(util) and util > 0) or foreign:
            r.checked += 1
            r.red += 1
            void = True
            if len(r.detail) < 5:
                r.detail.append("gpu %s busy at t=%s: util %s pids %s (not of this launch: %s)"
                                % (row.get("gpu_index"), row["t_rel_ns"], util, pids, foreign))
            continue
        for p in pids if isinstance(pids, list) else []:
            e = own_seen.setdefault((row.get("gpu_index"), p), [0, row["t_rel_ns"], None])
            e[0] += 1
            e[2] = row["t_rel_ns"]
        if row.get("query_rc") != 0 or not _is_num(util) or not isinstance(pids, list):
            unreadable += 1
            if unreadable <= 5:
                r.not_run.append("unreadable row at t=%s: query_rc %r util %r pids %r" % (
                    row["t_rel_ns"], row.get("query_rc"), util, pids))
            continue
        r.checked += 1
    if unreadable > 5:
        r.not_run.append("%d unreadable rows in the windows" % unreadable)
    for (gi, p), (n, t0, t1) in sorted(own_seen.items(), key=lambda kv: str(kv[0])):
        r.detail.append("gpu %s: pid %s (%s) held a compute context at util 0 in %d window "
                        "row(s), t=%s..%s; not foreign" % (gi, p, own_pids[p], n, t0, t1))
    if r.red == 0 and r.checked < r.expected:
        r.not_run.append("%d of %d expected GPU rows in the windows" % (r.checked, r.expected))
    return r.finish(), void


# ---- LAUNCH-STATE (launch level) -------------------------------------------------------------

def check_launch_state(ctx):
    """The launch-level conclusion of the writers before the judge (inputs._load_upstream:
    STATE lines of caplaunch / capjobs, launch_end.json state and leftovers): VALID -> PASS,
    NOT_RUN -> NOT_RUN, VOID / ENV_INVALID -> RED. The judge only sees the measurement files, so
    it can make that conclusion worse but never better."""
    r = Result("LAUNCH-STATE", "launch", "STATE lines before capjudge + launch_end.json")
    up = ctx.upstream
    r.expected = 1
    r.checked = 1
    for src in up["sources"]:
        if src["status"] != "VALID":
            r.detail.append("%s%s: %s (%s)" % (src["source"], " by %s" % src["by"]
                                               if src.get("by") else "", src["status"],
                                               src["reason"]))
    if up["status"] in ("VOID", "ENV_INVALID"):
        r.red = 1
    elif up["status"] == "NOT_RUN":
        r.not_run.append("launch recorded NOT_RUN before the judge")
    return r.finish()


# ---- PROBE-TIER (launch level) ---------------------------------------------------------------

def check_probe_tier(ctx, ld):
    """The registered probe tier is the one deployed: overlay manifest tier == launch.json
    probe_tier == every product role's declared probe_tier. Only when some role is a product
    process (probe_tier not null)."""
    product = [r for r in ctx.roles.values() if r["probe_tier"] is not None]
    if not product:
        return None
    r = Result("PROBE-TIER", "launch", "launch.json roles with probe_tier")
    r.expected = len(product)
    lt = ctx.launch["probe_tier"]
    lt = lt.get("tier") if isinstance(lt, dict) else lt
    ov = ld.overlay
    if ov is None:
        r.red += 1
        r.checked = 1
        r.detail.append("env/overlay_manifest.json missing")
        return r.finish()
    for role in product:
        r.checked += 1
        if not role["probe_tier"] == lt == ov.get("tier"):
            r.red += 1
            r.detail.append("%s declares %r, launch %r, overlay %r"
                            % (role["role"], role["probe_tier"], lt, ov.get("tier")))
    return r.finish()


# ---- C11 drain gate --------------------------------------------------------------------------

def check_c11(ctx, ld, td, drain_td):
    """Re-check the drain tier before load tier `td` from drain_gate.json (INTERFACES 8.3.5):
    planned poll times, per-poll snapshot usability (gate.check_snapshot on the recorded fields,
    with every registered process of every gate role required), recorded readings == sum of
    their terms, the rule, settle_polls -> passed / t_pass. Expected = number of planned polls
    from the drain start to t_pass (or the deadline)."""
    r = Result("C11", td.tier.seq, "gate.planned_polls(drain start, t_pass|deadline, poll_ns)")
    dt = drain_td.tier
    gate_cfg, _src = ctx.tt.gate_of(dt.seq)
    dg = drain_td.drain_gate
    if dg is None:
        r.expected, r.checked, r.red = 1, 1, 1
        r.detail.append("drain tier %d has no drain_gate.json" % dt.seq)
        return r.finish()
    if dg.get("gate") != gate_cfg:
        r.expected, r.checked, r.red = 1, 1, 1
        r.detail.append("recorded gate config differs from the tier table")
        return r.finish()
    if gate_cfg["rule"] != "abs_max":
        r.not_run.append("rule %s re-check: %s" % (gate_cfg["rule"], NOT_IMPLEMENTED))
        return r.finish(NOT_RUN)
    deadline = dt.end_ns - gate_cfg["deadline_guard_ns"]
    rec_pass, rec_t = dg.get("passed") is True, dg.get("t_pass_rel_ns")
    end = rec_t if rec_pass and isinstance(rec_t, int) else deadline
    planned = planned_polls(dt.start_ns, end, gate_cfg["poll_ns"])
    r.expected = len(planned)
    polls = dg.get("polls") or []
    if dg.get("n_polls") != len(polls) or len(polls) != len(planned) or \
            [p.get("planned_rel_ns") for p in polls] != planned:
        r.red += 1
        r.detail.append("polls %d (n_polls %r) do not match %d planned polls"
                        % (len(polls), dg.get("n_polls"), len(planned)))
    gate_roles = sorted({t["role"] for rd in gate_cfg["readings"] for t in rd["terms"]})
    by_pid = ld.procs_by_pid()
    need = {p: by_pid[p] for p in by_pid if by_pid[p].get("role") in gate_roles}
    n_role = {role: sum(1 for p in need.values() if p["role"] == role) for role in gate_roles}
    key = ctx.ident.launch_key()
    prev, consecutive, my_pass, my_t = {}, 0, False, None
    for poll in polls:
        r.checked += 1
        sat, why = _poll_satisfied(poll, gate_cfg, need, n_role, key, prev)
        consecutive = consecutive + 1 if sat else 0
        if not my_pass and consecutive >= gate_cfg["settle_polls"]:
            my_pass, my_t = True, poll
        if why and len(r.detail) < 8:
            r.detail.append("poll@%s: %s" % (poll.get("planned_rel_ns"), why))
    if my_pass != rec_pass or (my_pass and rec_t not in (my_t.get("planned_rel_ns"),
                                                         my_t.get("actual_rel_ns"))):
        r.red += 1
        r.detail.append("recorded passed=%r t=%r, re-check passed=%r t=%r" % (
            rec_pass, rec_t, my_pass, None if my_t is None else my_t.get("planned_rel_ns")))
    elif not my_pass:
        if gate_cfg["mode"] == "record_only":
            r.red += 1
            r.detail.append("record_only gate did not pass before this tier")
        else:
            r.not_run.append("enforced gate did not pass: launch stopped")
    return r.finish()


def _poll_satisfied(poll, gate_cfg, need, n_role, key, prev):
    snaps = {int(k): v for k, v in (poll.get("snapshots") or {}).items()}
    usable = set()
    reasons = []
    for pid, proc in need.items():
        f = snaps.get(pid)
        if f is None:
            reasons.append("pid %d no snapshot" % pid)
            continue
        ok, why = check_snapshot(f, proc, poll.get("now_ns"), gate_cfg["max_snapshot_age_ns"],
                                 prev.get(pid), key)
        prev[pid] = f.get("seq")
        if ok:
            usable.add(pid)
        else:
            reasons.append("pid %d %s" % (pid, why))
    sat = not reasons
    values = poll.get("values") or {}
    details = poll.get("details") or {}
    for rd in gate_cfg["readings"]:
        v = values.get(rd["name"])
        terms = details.get(rd["name"])
        if v is None or not isinstance(terms, list):
            sat = False
            reasons.append("reading %s not computed" % rd["name"])
            continue
        if sum(t["sign"] * t["value"] for t in terms) != v:
            sat = False
            reasons.append("reading %s != sum of its terms" % rd["name"])
        for t in terms:
            if t.get("n_procs") != n_role.get(t["role"]):
                sat = False
                reasons.append("reading %s used %s of %s %s processes" % (
                    rd["name"], t.get("n_procs"), n_role.get(t["role"]), t["role"]))
        if v > rd["abs_max"]:
            sat = False
    return sat, "; ".join(reasons)


# ---- C13 thread inventory and foreign busy time ----------------------------------------------

def check_c13_launch(ctx, ld):
    """Launch-level first half: an inventory for every under_test process (expected =
    expected_procs of under_test roles), every thread identified, no unattributed process."""
    r = Result("C13", "launch", "launch.json roles[measure_class=under_test].expected_procs")
    ut = [x["role"] for x in _roles_of(ctx, "under_test")]
    r.expected = sum(ctx.roles[x]["expected_procs"] for x in ut)
    by_pid = ld.procs_by_pid()
    seen = set()
    for doc, p in ld.thread_inventories():
        pid = (doc or {}).get("pid")
        if by_pid.get(pid, {}).get("role") not in ut:
            continue
        seen.add(pid)
        r.checked += 1
        unknown = [t.get("tid") for t in doc.get("threads") or [] if t.get("identified_as")
                   is None]
        if unknown:
            r.red += 1
            r.detail.append("pid %s: unidentified threads %s" % (pid, unknown[:10]))
    if r.checked < r.expected:
        r.red += 1
        r.detail.append("%d of %d under_test processes have a thread inventory"
                        % (r.checked, r.expected))
    un = (ld.roles_json or {}).get("unattributed") or []
    if un:
        r.red += 1
        r.detail.append("%d unattributed process(es) in roles.json" % len(un))
    # roles.json aux_helpers (e.g. multiprocessing.resource_tracker of a capkit tool role) are
    # attributed to their parent role but are not processes of it: never counted here, in procs
    # or in any per-role process set; only listed for the record.
    aux = (ld.roles_json or {}).get("aux_helpers") or []
    if aux:
        r.detail.append("%d aux helper(s) attributed to %s (not under test)" % (
            len(aux), sorted({str(a.get("role")) for a in aux if isinstance(a, dict)})))
    return r.finish()


C13_STAT = "T3:foreign_busy_frac"


def _c13_d14(ctx):
    """irq_softirq_is_experiment of the registered d14_result (capjudge.nullz), or None."""
    d14 = ctx.thresholds.d14()
    v = d14.get("irq_softirq_is_experiment") if isinstance(d14, dict) else None
    return v if isinstance(v, bool) else None


def check_c13_tier(ctx, ld, td, record):
    """Per tier: window-boundary thread lists all identified; second half: busy ticks on each
    under_test role's cores not explained by registered threads, as a fraction of the cores'
    ticks (T3:foreign_busy_frac:<role>). Only softirq ticks are further subtracted, and only
    when the registered D14 result (d14_result, `irq_softirq_is_experiment`) is true: the
    registered D14 rule decides on softirq ticks alone (capjudge.d14), so hard-irq ticks stay
    foreign. Calibration v2 (DESIGN-r7a 2.11.1, P2-05): a reading above the per-window cap of
    the pilot/2 `c13_cap` (nullz.cap) is RED; every reading is written to `record`, also
    without a cap (then the second half is NOT_RUN); without the D14 result no reading is
    computed and the second half is NOT_RUN. The launch mean is check_c13_mean."""
    r = Result("C13", td.tier.seq, "launch.json roles[measure_class=under_test] + pin_table")
    ut = [x["role"] for x in _roles_of(ctx, "under_test")]
    r.expected = len(ut)
    tj = td.tier_json or {}
    for edge in ("wstart", "wend"):
        for pid, ths in ((tj.get("threads") or {}).get(edge) or {}).items():
            unknown = [t.get("tid") for t in ths if t.get("identified_as") is None]
            if unknown:
                r.red += 1
                r.detail.append("%s pid %s unidentified %s" % (edge, pid, unknown[:10]))
    if not tj.get("threads"):
        r.red += 1
        r.detail.append("tier.json has no window-boundary thread lists")
    softirq_ours = _c13_d14(ctx)
    if softirq_ours is None:
        r.not_run.append("second half: no registered D14 result (d14_result)")
        r.checked = len(ut)
        return r.finish()
    cpus = role_cpus(ld.pin_table)
    cpu_rows, th_rows = ld.sidecar("cpus"), ld.sidecar("threads")
    clk = (ld.sidecar_meta or {}).get("clk_tck")
    if cpu_rows is None or th_rows is None or not isinstance(clk, int):
        r.not_run.append("second half: sidecar cpus/threads/clk_tck missing")
        r.checked = len(ut)
        return r.finish()
    cap = ctx.thresholds.cap(C13_STAT)
    if cap is None:
        r.not_run.append("second half: no c13_cap %s in a registered pilot/2 (readings "
                         "recorded)" % C13_STAT)
    ws, we = td.tier.window_start_ns, td.tier.window_end_ns
    by_pid = ld.procs_by_pid()
    for role in ut:
        r.checked += 1
        cores = cpus.get(role) or set()
        frac, why = _foreign_frac(cpu_rows[1], th_rows[1], by_pid, role, cores, ws, we, clk,
                                  softirq_ours)
        key = band_key("T3", "foreign_busy_frac", role)
        if frac is None:
            r.not_run.append("%s: %s" % (role, why))
            continue
        record.setdefault(key, []).append(frac)
        if cap is not None and frac > cap["window"]:
            r.red += 1
            r.detail.append("%s foreign busy fraction %.4f above the window cap %s"
                            % (role, frac, cap["window"]))
    return r.finish()


def check_c13_mean(ctx, tiers):
    """C13-MEAN (DESIGN-r7a 2.11.1, P2-05), launch level: for each under_test role the mean of
    its T3:foreign_busy_frac readings over all judged load tiers (verdicts.json `tiers[].stats`,
    written by check_c13_tier) must not exceed the launch-mean cap of the pilot/2 `c13_cap`;
    above -> RED (the launch is VOID). Without the cap or the D14 result -> NOT_RUN (the means
    are still reported); a role without any reading -> NOT_RUN. Returns (Result, {role: {mean,
    n, cap}})."""
    r = Result("C13-MEAN", "launch", "launch.json roles[measure_class=under_test]")
    ut = [x["role"] for x in _roles_of(ctx, "under_test")]
    r.expected = len(ut)
    cap = ctx.thresholds.cap(C13_STAT)
    if _c13_d14(ctx) is None:
        r.not_run.append("no registered D14 result (d14_result)")
    if cap is None:
        r.not_run.append("no c13_cap %s in a registered pilot/2" % C13_STAT)
    means = {}
    for role in ut:
        key = band_key("T3", "foreign_busy_frac", role)
        vals = [v for e in tiers if e.get("judged")
                for v in (e.get("stats") or {}).get(key, [])]
        if not vals:
            r.not_run.append("%s: no %s reading on any judged load tier" % (role, key))
            continue
        r.checked += 1
        m = math.fsum(vals) / len(vals)
        means[role] = {"mean": m, "n": len(vals),
                       "cap": None if cap is None else cap["launch_mean"]}
        if cap is not None and m > cap["launch_mean"]:
            r.red += 1
            r.detail.append("%s mean foreign busy fraction %.4f over %d tiers above the "
                            "launch-mean cap %s" % (role, m, len(vals), cap["launch_mean"]))
    return r.finish(), means


BUSY_COLS = ("user", "nice", "system", "irq", "softirq", "steal")


def _foreign_frac(cpu_rows, th_rows, by_pid, role, cores, ws, we, clk, softirq_is_ours):
    """(fraction, None) or (None, why): busy ticks of `cores` minus the registered threads of
    `role` (minus softirq when softirq_is_ours), over the cores' ticks in the window."""
    d, why = cpu_tick_deltas(cpu_rows, cores, ws, we)
    if why:
        return None, why
    busy = sum(d[c][k] for c in cores for k in BUSY_COLS)
    own_softirq = sum(d[c]["softirq"] for c in cores) if softirq_is_ours else 0
    span_ns = max(d[c]["t_last_rel_ns"] - d[c]["t_first_rel_ns"] for c in cores)
    if span_ns <= 0:
        return None, "cpu rows of cores %s span no time in the window" % sorted(cores)
    tf, tl = first_last_in_window(th_rows, lambda x: (x["pid"], x["tid"])
                                  if by_pid.get(x["pid"], {}).get("role") == role else None,
                                  ws, we)
    threads = sum((tl[k]["utime_ticks"] + tl[k]["stime_ticks"])
                  - (tf[k]["utime_ticks"] + tf[k]["stime_ticks"]) for k in tf)
    foreign = busy - threads - own_softirq
    total = len(cores) * span_ns * clk / NS
    return foreign / total, None

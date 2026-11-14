# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal localisation I (design 2.6; INTERFACES errata B2-2E): which process is the limit.

Computed per load tier in judge mode "P+X+I+V" from /proc readings and backlog readings, which
are independent of X's sources (load generator and stub counts).

I1 CPU, per registered process of every non-observer role, over the tier's steady window:
  * the process's cores (roles.json procs[].cpus) must be exclusive (no other registered process
    on any of them); per-core /proc/stat (sidecar cpus.jsonl) is the authority, thread
    utime + stime (sidecar threads.jsonl) is the breakdown;
  * one core ("single_core"): holds iff the core's busy fraction >= near_one_core_min AND its
    offset from the same launch's baseline tier is above its band. Softirq ticks count as busy
    only when the pilot's D14 result says they are the experiment's work: the statistic is
    `T2:core_busy_frac_softirq:<role>` then, `T2:core_busy_frac:<role>` (softirq left out)
    otherwise. Both are recorded on every tier whatever the pilot says, so a pilot can be built
    from trees judged before it existed;
  * several cores ("multi_core"): the GIL domain (threads of class python, summed) holds iff
    its cores >= near_one_core_min, its offset is above `T2:gil_domain_cores:<role>` and the
    run-queue wait of those threads did not rise (`T2:gil_run_delay_frac:<role>` not above);
    or one thread outside the GIL (grpc_core / zmq_io / other, each alone) holds iff its cores
    >= near_one_core_min and its offset is above `T2:nongil_thread_cores:<role>`.
  near_one_core_min comes from pilot.json `i1_calibration`; missing -> I1 NOT_RUN.
I2 event-loop lag: side evidence only (its criterion needs E0's calibration curve).
I3 backlog side: register.json judge.localize.readings, each a sum of backlog slot terms with
  a `receiver` and a `sender` role (in-flight between two roles points to the receiver; a
  role's own queue has receiver = that role, sender null). A reading is "growing" (T1 above
  `T1:i3_backlog:<name>`), "elevated" (no growth, T2 offset from the baseline tier above
  `T2:i3_backlog:<name>`: held at a plateau), "flat", or NOT_RUN. I3 holds for role R iff an
  entry reading (receiver R) is growing or elevated and every exit reading (sender R) is flat.

Combination: the roles the holding signals point to. One role -> LOCALIZED; more than one ->
UNDETERMINED (signals disagree: no boundary from this tier); none -> NOT_LOCALIZED, or NOT_RUN
when some signal could not be evaluated. Only a tier where X holds gets a conclusion; every
tier records its statistics so the pilot can build their bands from sub-saturation tiers.
"""

from capcommon import hist

from . import stats
from .thresholds import band_key
from .treedata import NS, cpu_tick_deltas, first_last_in_window

TRUE, FALSE, NOT_RUN = "TRUE", "FALSE", "NOT_RUN"
LOCALIZED, UNDETERMINED, NOT_LOCALIZED = "LOCALIZED", "UNDETERMINED", "NOT_LOCALIZED"
NOT_APPLICABLE = "NOT_APPLICABLE"  # X does not hold: nothing to localise
CANDIDATE_CLASSES = ("under_test", "stub_in_domain", "load_source", "stub_out_of_domain")
GIL_CLASS = "python"
IDLE_COLS = ("idle", "iowait")
ALL_COLS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")


# ---- I1 readings -----------------------------------------------------------------------------

def thread_classes(ld, td, pid):
    """{tid: (class, identified_as)} for `pid`: the launch thread inventory, overridden by the
    tier's window-boundary thread lists (tier.json, written by the orchestrator for every
    registered process)."""
    out = {}
    for doc, _p in ld.thread_inventories():
        if isinstance(doc, dict) and doc.get("pid") == pid:
            for t in doc.get("threads") or []:
                if t.get("class"):
                    out[t["tid"]] = (t["class"], t.get("identified_as"))
    tj = td.tier_json or {}
    for edge in ("wstart", "wend"):
        for t in ((tj.get("threads") or {}).get(edge) or {}).get(str(pid)) or []:
            if t.get("class"):
                out[t["tid"]] = (t["class"], t.get("identified_as"))
    return out


def proc_cores(ld, pid):
    """(cores of `pid` from roles.json, exclusive?) - exclusive when no other registered process
    is pinned to any of them."""
    procs = (ld.roles_json or {}).get("procs") or []
    mine = None
    for p in procs:
        if p.get("pid") == pid:
            mine = set(p.get("cpus") or [])
    if not mine:
        return set(), False
    others = set()
    for p in procs:
        if p.get("pid") != pid:
            others |= set(p.get("cpus") or [])
    return mine, not (mine & others)


def proc_readings(ld, td, pid):
    """CPU readings of one process over the tier's window: ({...}, None) or (None, why).

    Keys: cores (sorted list), n_cores, core_busy_frac / core_busy_cores (non-idle ticks without
    softirq), core_busy_frac_softirq (with it), threads [{tid, class, identified_as, cores,
    run_delay_frac}], gil_cores, gil_run_delay_frac (None when a GIL thread lacks
    run_delay_ns), unattributed_cores (core busy without softirq - all threads)."""
    clk = (ld.sidecar_meta or {}).get("clk_tck")
    cpu_rows, th_rows = ld.sidecar("cpus"), ld.sidecar("threads")
    if cpu_rows is None or th_rows is None or not isinstance(clk, int) or clk <= 0:
        return None, "sidecar cpus/threads/clk_tck missing"
    cores, excl = proc_cores(ld, pid)
    if not cores:
        return None, "pid %d has no cpus in roles.json" % pid
    if not excl:
        return None, "pid %d cores %s are shared with another registered process" \
            % (pid, sorted(cores))
    ws, we = td.tier.window_start_ns, td.tier.window_end_ns
    d, why = cpu_tick_deltas(cpu_rows[1], cores, ws, we)
    if why:
        return None, why
    span = max(d[c]["t_last_rel_ns"] - d[c]["t_first_rel_ns"] for c in cores)
    if span <= 0:
        return None, "cpu rows of cores %s span no time in the window" % sorted(cores)
    total = sum(d[c][k] for c in cores for k in ALL_COLS)
    idle = sum(d[c][k] for c in cores for k in IDLE_COLS)
    softirq = sum(d[c]["softirq"] for c in cores)
    busy = total - idle - softirq
    tf, tl = first_last_in_window(th_rows[1], lambda x: x["tid"] if x.get("pid") == pid
                                  else None, ws, we)
    classes = thread_classes(ld, td, pid)
    threads, gil, gil_rd, rd_ok = [], 0.0, 0.0, True
    for tid in sorted(tf):
        a, b = tf[tid], tl[tid]
        tspan = b["t_rel_ns"] - a["t_rel_ns"]
        if tspan <= 0:
            continue
        cores_t = ((b["utime_ticks"] + b["stime_ticks"]) - (a["utime_ticks"]
                                                          + a["stime_ticks"])) / clk / (tspan / NS)
        rd = None
        if isinstance(b.get("run_delay_ns"), int) and isinstance(a.get("run_delay_ns"), int):
            rd = (b["run_delay_ns"] - a["run_delay_ns"]) / tspan
        cls, ident = classes.get(tid, ("unknown", None))
        threads.append({"tid": tid, "comm": b.get("comm"), "class": cls,
                        "identified_as": ident, "cores": cores_t, "run_delay_frac": rd})
        if cls == GIL_CLASS:
            gil += cores_t
            if rd is None:
                rd_ok = False
            else:
                gil_rd += rd
    if not threads:
        return None, "pid %d: fewer than two sidecar thread rows in the window" % pid
    core_busy_cores = busy / clk / (span / NS)
    return {"pid": pid, "cores": sorted(cores), "n_cores": len(cores),
            "core_busy_cores": core_busy_cores,
            "core_busy_frac": busy / total if total > 0 else None,
            "core_busy_frac_softirq": (busy + softirq) / total if total > 0 else None,
            "threads": threads, "gil_cores": gil,
            "gil_run_delay_frac": gil_rd if rd_ok else None,
            "unattributed_cores": core_busy_cores - sum(t["cores"] for t in threads)}, None


def _offset(cur, base, key):
    if cur is None or base is None or cur.get(key) is None or base.get(key) is None:
        return None
    return cur[key] - base[key]


def i1_process(cur, base, bands, role, near, d14, record, is_baseline):
    """I1 for one process from its readings in this tier (`cur`) and in the baseline tier
    (`base`; None when missing). d14: the pilot's softirq verdict (None when absent). Records
    every T2 offset first (not on the baseline tier, where they are 0 by construction), then
    evaluates. Returns {status, via, branch, ...}."""
    out = {"status": NOT_RUN, "via": None, "reasons": []}
    single = cur["n_cores"] == 1
    out["branch"] = "single_core" if single else "multi_core"
    if base is None:
        out["reasons"].append("no baseline-tier reading of the same process")
        return out
    offs = {}
    qtys = (("core_busy_frac", "core_busy_frac"),
            ("core_busy_frac_softirq", "core_busy_frac_softirq")) if single else \
        (("gil_cores", "gil_domain_cores"), ("gil_run_delay_frac", "gil_run_delay_frac"))
    for key, qty in qtys:
        v = _offset(cur, base, key)
        k = band_key("T2", qty, role)
        if v is not None and not is_baseline:
            record.setdefault(k, []).append(v)
        offs[qty] = (v, k)
    base_t = {t["tid"]: t["cores"] for t in base["threads"]}
    nongil_offs = {t["tid"]: t["cores"] - base_t.get(t["tid"], 0.0) for t in cur["threads"]
                   if t["class"] != GIL_CLASS}
    kn = band_key("T2", "nongil_thread_cores", role)
    if nongil_offs and not single and not is_baseline:
        record.setdefault(kn, []).append(max(nongil_offs.values()))
    if near is None:
        out["reasons"].append("pilot.json has no i1_calibration.near_one_core_min")
        return out

    def cmp(qty):
        v, k = offs[qty]
        b = bands.get(k)
        return v, (stats.compare_to_band(v, b) if b is not None else None), k

    if single:
        if not isinstance(d14, bool):
            out["reasons"].append("pilot.json has no D14 result (softirq: ours or not)")
            return out
        qty = "core_busy_frac_softirq" if d14 else "core_busy_frac"
        frac = cur[qty]
        v, cmp_, k = cmp(qty)
        out.update(core_busy_frac=frac, offset=v, statistic=k)
        if cmp_ is None:
            out["reasons"].append("no band %s or no value" % k)
            return out
        hold = frac is not None and frac >= near and cmp_ == "above"
        out.update(status=TRUE if hold else FALSE, via="core_idle" if hold else None)
        return out
    v, cmp_g, kg = cmp("gil_domain_cores")
    rd, cmp_r, kr = cmp("gil_run_delay_frac")
    out.update(gil_cores=cur["gil_cores"], gil_offset=v, run_delay_offset=rd)
    gil = NOT_RUN
    if cmp_g is None or cmp_r is None:
        out["reasons"].append("GIL domain: no band %s / %s or no value" % (kg, kr))
    else:
        gil = TRUE if (cur["gil_cores"] >= near and cmp_g == "above"
                       and cmp_r != "above") else FALSE
    bn = bands.get(kn)
    hot = [t["tid"] for t in cur["threads"] if t["class"] != GIL_CLASS and bn is not None
           and t["cores"] >= near
           and stats.compare_to_band(nongil_offs[t["tid"]], bn) == "above"]
    nongil = FALSE if not nongil_offs else (NOT_RUN if bn is None else
                                            (TRUE if hot else FALSE))
    if nongil == NOT_RUN:
        out["reasons"].append("non-GIL threads: no band %s" % kn)
    if gil == TRUE:
        out.update(status=TRUE, via="gil_domain")
    elif nongil == TRUE:
        out.update(status=TRUE, via="non_gil_thread", threads_hot=hot)
    elif NOT_RUN in (gil, nongil):
        out["status"] = NOT_RUN
    else:
        out["status"] = FALSE
    return out


# ---- I2 (side evidence) ----------------------------------------------------------------------

def loop_lag_p50(td, role):
    """Median-bucket upper bound of the role's summed loop_lag_ns histograms; None if absent."""
    acc = None
    for doc in (td.summaries.get(role) or {}).values():
        h = ((doc.get("loop_lag_ns") or {}).get("hist") or {}).get("counts")
        if isinstance(h, list) and len(h) == hist.NBINS:
            acc = list(h) if acc is None else [a + b for a, b in zip(acc, h)]
    return None if acc is None else hist.quantile(acc, 0.5)


# ---- I3 backlog readings ---------------------------------------------------------------------

def reading_series(ld, tier, reading):
    """Per-second series of one registered reading over the tier's window: per term, the last
    sidecar snapshot value in each second of every registered process of the term's role
    (agg sum / max, times sign); a second without a value for every process is None."""
    sc = ld.sidecar("snapshots")
    if sc is None:
        return None, "sidecar/snapshots.jsonl missing"
    ws, we = tier.window_start_ns, tier.window_end_ns
    k = (we - ws) // NS
    total = [0] * k
    for term in reading["terms"]:
        pids = [p["pid"] for p in ld.procs_of_role(term["role"])]
        if not pids:
            return None, "no registered process for role %s" % term["role"]
        last = {}
        for r in sc[1]:
            if r["pid"] not in pids or term["slot"] not in (r.get("counters") or {}):
                continue
            t = r["snap_t_rel_ns"]
            if not ws <= t < ws + k * NS:
                continue
            key = ((t - ws) // NS, r["pid"])
            if key not in last or last[key][0] <= t:
                last[key] = (t, r["counters"][term["slot"]])
        for i in range(k):
            vals = [last[(i, p)][1] for p in pids if (i, p) in last]
            if total[i] is None or len(vals) != len(pids):
                total[i] = None
                continue
            v = sum(vals) if term["agg"] == "sum" else max(vals)
            total[i] += term["sign"] * v
    return total, None


def i3_reading(ld, td, base_td, reading, bands, record, is_baseline):
    """State of one reading: growing / elevated / flat / NOT_RUN, with T1 and T2."""
    name = reading["name"]
    out = {"name": name, "receiver": reading.get("receiver"), "sender": reading.get("sender"),
           "state": NOT_RUN}
    series, why = reading_series(ld, td.tier, reading)
    bseries, bwhy = (None, "no baseline tier") if base_td is None else \
        reading_series(ld, base_td.tier, reading)
    k1, k2 = band_key("T1", "i3_backlog", name), band_key("T2", "i3_backlog", name)
    t1 = stats.t1_trend(series) if series is not None else None
    t2 = stats.t2_offset(series, bseries) if series is not None and bseries is not None \
        else None
    if t1 is not None:
        record.setdefault(k1, []).append(t1[0])
    if t2 is not None and not is_baseline:
        record.setdefault(k2, []).append(t2)
    out.update(T1=None if t1 is None else t1[0], T2=t2)
    if series is None or t1 is None:
        out["reason"] = why or "fewer than one value in a half of the window"
        return out
    c1 = stats.compare_to_band(t1[0], bands.get(k1))
    if c1 is None:
        out["reason"] = "no band %s" % k1
        return out
    if c1 == "above":
        out["state"] = "growing"
        return out
    if t2 is None:
        out["reason"] = bwhy or "no baseline value"
        return out
    c2 = stats.compare_to_band(t2, bands.get(k2))
    if c2 is None:
        out["reason"] = "no band %s" % k2
        return out
    out["state"] = "elevated" if c2 == "above" else "flat"
    return out


def i3_roles(readings, roles):
    """{role: {status, entry_backed, exit_blocked, not_run}} for every role with at least one
    entry reading (receiver = role)."""
    out = {}
    for role in roles:
        entry = [r for r in readings if r["receiver"] == role]
        exit_ = [r for r in readings if r["sender"] == role]
        if not entry:
            continue
        backed = [r["name"] for r in entry if r["state"] in ("growing", "elevated")]
        blocked = [r["name"] for r in exit_ if r["state"] in ("growing", "elevated")]
        unknown_in = [r["name"] for r in entry if r["state"] == NOT_RUN]
        unknown_out = [r["name"] for r in exit_ if r["state"] == NOT_RUN]
        if blocked:
            st = FALSE
        elif backed and not unknown_out:
            st = TRUE
        elif backed or unknown_in:
            st = NOT_RUN
        else:
            st = FALSE
        out[role] = {"status": st, "entry_backed": backed, "exit_blocked": blocked,
                     "not_run": unknown_in + unknown_out}
    return out


# ---- per tier --------------------------------------------------------------------------------

def localize_tier(ctx, ld, td, base_td, x, record, procs_cache):
    """I for one load tier (called through the symptoms hook with the tier's X result).

    procs_cache: {(tier seq, pid): readings or (None, why)} shared across tiers so the
    baseline tier's readings are computed once."""
    bands = ctx.bands.view(ctx.tier_regime[td.tier.seq])
    pilot = ctx.bands.extras.get("pilot") or {}
    near = (pilot.get("i1_calibration") or {}).get("near_one_core_min")
    d14 = (pilot.get("d14_softirq") or {}).get("irq_softirq_is_experiment")
    d14 = d14 if isinstance(d14, bool) else None
    is_base = base_td is td
    out = {"I1": {}, "I2": {}, "I3": {"readings": [], "roles": {}}, "reasons": []}
    roles = sorted(r for r, d in ctx.roles.items() if d["measure_class"] in CANDIDATE_CLASSES)

    def readings(tdx, pid):
        key = (tdx.tier.seq, pid)
        if key not in procs_cache:
            procs_cache[key] = proc_readings(ld, tdx, pid)
        return procs_cache[key]

    for role in roles:
        per = []
        for p in ld.procs_of_role(role):
            pid = p["pid"]
            cur, why = readings(td, pid)
            if cur is None:
                per.append({"pid": pid, "status": NOT_RUN, "reasons": [why]})
                continue
            base = readings(base_td, pid)[0] if base_td is not None else None
            r = i1_process(cur, base, bands, role, near, d14, record, is_base)
            r.update(pid=pid, readings=cur)
            per.append(r)
        sts = [r["status"] for r in per]
        st = TRUE if TRUE in sts else (NOT_RUN if (NOT_RUN in sts or not per) else FALSE)
        out["I1"][role] = {"status": st, "processes": per}
        lag = loop_lag_p50(td, role)
        blag = loop_lag_p50(base_td, role) if base_td is not None else None
        k = band_key("T2", "loop_lag_p50_ns", role)
        off = None if lag is None or blag is None else lag - blag
        if off is not None and not is_base:
            record.setdefault(k, []).append(off)
        out["I2"][role] = {"status": "SIDE_EVIDENCE", "offset_ns": off,
                           "vs_band": stats.compare_to_band(off, bands.get(k))}
    spec = ((ctx.register.get("judge") or {}).get("localize") or {}).get("readings") or []
    rds = [i3_reading(ld, td, base_td, r, bands, record, is_base) for r in spec]
    out["I3"] = {"readings": rds, "roles": i3_roles(rds, roles)}
    if not spec:
        out["reasons"].append("I3: register.json judge.localize lists no backlog reading")
    return _conclude(out, x, ctx)


def _conclude(out, x, ctx):
    holding = {}
    not_run = []
    for role, v in out["I1"].items():
        if v["status"] == TRUE:
            holding.setdefault(role, []).append("I1")
        elif v["status"] == NOT_RUN:
            not_run.append("I1:%s" % role)
    for role, v in out["I3"]["roles"].items():
        if v["status"] == TRUE:
            holding.setdefault(role, []).append("I3")
        elif v["status"] == NOT_RUN:
            not_run.append("I3:%s" % role)
    out["holding"] = {r: sorted(s) for r, s in sorted(holding.items())}
    out["not_run"] = not_run
    if x.get("holds") is not True:
        out["status"] = NOT_APPLICABLE
        out["reasons"].append("X does not hold in this tier")
        return out
    if len(holding) > 1:
        out["status"] = UNDETERMINED
        out["reasons"].append("signals point to different roles %s" % sorted(holding))
        return out
    if not holding:
        out["status"] = NOT_RUN if not_run else NOT_LOCALIZED
        return out
    role, via = next(iter(holding.items()))
    i1 = out["I1"].get(role) or {}
    hot = [p for p in i1.get("processes", []) if p.get("status") == TRUE]
    if hot and any(p.get("via") in ("core_idle", "gil_domain") for p in hot):
        finding = "domain_saturated"
    elif hot:
        finding = "thread_saturated"
    else:
        finding = "backlog_at_entry"
    out.update(status=LOCALIZED, role=role, via=via, finding=finding,
               measure_class=ctx.roles[role]["measure_class"], granularity="process",
               pids=[p["pid"] for p in hot],
               thread_shares={str(p["pid"]): [{k: t[k] for k in ("tid", "class",
                                                                  "identified_as", "cores")}
                                              for t in p["readings"]["threads"]]
                              for p in hot})
    if not_run:
        out["reasons"].append("localised with signals not evaluated: %s" % not_run)
    return out

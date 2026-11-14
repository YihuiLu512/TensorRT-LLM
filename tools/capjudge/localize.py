# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal localisation I (design 2.6; INTERFACES errata B2-2E): which process is the limit.

Computed per load tier in judge mode "P+X+I+V" from /proc readings and backlog readings, which
are independent of X's sources (load generator and stub counts).

I1 CPU, per core group of every non-observer role, over the tier's steady window:
  * a core group is the role's registered processes whose roles.json procs[].cpus intersect (a
    parent and the workers pinned to the role's core set are one group); its cores must not be
    shared with a registered process of another role (else NOT_RUN). What the criterion
    reads depends on the group's core count: a one-core group, its core's /proc/stat (sidecar
    cpus.jsonl; the D14 softirq verdict applies only here); a several-core group, each
    process's GIL domain from its threads' utime + stime (sidecar threads.jsonl; one GIL per
    process), with the per-core readings and `unattributed_cores` recorded alongside but not
    judged (per-core busy time cannot tell which process's GIL is busy when the group holds
    several processes);
  * one core ("single_core", one unit per group): holds iff the core's busy fraction >=
    near_one_core_min AND its offset from the same launch's baseline tier is above its band.
    Softirq ticks count as busy only when the pilot's D14 result says they are the
    experiment's work: the statistic is `T2:core_busy_frac_softirq:<role>` then,
    `T2:core_busy_frac:<role>` (softirq left out) otherwise. Both are recorded on every tier
    whatever the pilot says, so a pilot can be built from trees judged before it existed;
  * several cores ("multi_core", one unit per process of the group): the process's GIL domain
    (threads of class python, summed) holds iff its cores >= near_one_core_min, its offset is
    above `T2:gil_domain_cores:<role>` and the run-queue wait of those threads did not rise
    (`T2:gil_run_delay_frac:<role>` not above); or one thread outside the GIL (grpc_core /
    zmq_io / other, each alone) holds iff its cores >= near_one_core_min and its offset is
    above `T2:nongil_thread_cores:<role>`. A thread of class unknown (classified by neither
    the thread inventory nor tier.json) that used CPU in this or the baseline window, or a
    thread of class python or unknown with a single sidecar row there (its CPU unknown),
    leaves the GIL domain incomplete: that branch is NOT_RUN (its offsets are not recorded)
    and an unknown thread is no non-GIL candidate.
  near_one_core_min comes from pilot.json `i1_calibration`; missing -> I1 NOT_RUN.
I2 event-loop lag: side evidence only (its criterion needs E0's calibration curve).
I3 backlog side: register.json judge.localize.readings, each a sum of backlog slot terms with
  a `receiver` and a `sender` role (in-flight between two roles points to the receiver; a
  role's own queue has receiver = that role, sender null). A reading is "growing" (T1 above
  `T1:i3_backlog:<name>`), "elevated" (no growth, T2 offset from the baseline tier above
  `T2:i3_backlog:<name>`: held at a plateau), "flat", or NOT_RUN. I3 holds for role R iff an
  entry reading (receiver R) is growing or elevated and every exit reading (sender R) is flat.

Every unit lists the group's pids. Combination: the roles the holding signals point to. One
role -> LOCALIZED, unless a signal of another candidate role is NOT_RUN (that role is not
ruled out) -> NOT_RUN with `holding` listed; more than one -> UNDETERMINED (signals disagree:
no boundary from this tier); none -> NOT_LOCALIZED, or NOT_RUN when some signal could not be
evaluated. Only a tier where X holds gets a conclusion; every tier records its statistics so
the pilot can build their bands from sub-saturation tiers.
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
UNKNOWN_CLASS = "unknown"  # a thread neither the inventory nor tier.json classifies
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


def core_groups(ld, role):
    """The role's registered processes grouped by the cores they are pinned to: processes whose
    roles.json cpus intersect (directly or through another process of the role) form one
    group - a parent and the workers pinned to the role's core set are one group.

    Returns ([{pids, cores, shared_with}], [pids without cpus]); shared_with lists the pids of
    other roles' registered processes pinned to any of the group's cores (exclusive when
    empty). Processes of the same role never make a group non-exclusive."""
    groups, nocpu = [], []
    for p in ld.procs_of_role(role):
        cpus = set(p.get("cpus") or [])
        if not cpus:
            nocpu.append(p["pid"])
            continue
        merged, rest = {"pids": [p["pid"]], "cores": cpus}, []
        for g in groups:
            if g["cores"] & merged["cores"]:
                merged = {"pids": g["pids"] + merged["pids"], "cores": g["cores"] | merged["cores"]}
            else:
                rest.append(g)
        groups = rest + [merged]
    procs = (ld.roles_json or {}).get("procs") or []
    for g in groups:
        g["pids"] = sorted(g["pids"])
        g["shared_with"] = sorted(p["pid"] for p in procs if p.get("role") != role
                                  and set(p.get("cpus") or []) & g["cores"])
    return sorted(groups, key=lambda g: g["pids"]), sorted(nocpu)


def _thread_readings(th_rows, classes, pid, clk, ws, we):
    """Per-thread CPU of one process over [ws, we): (threads, gil_cores, gil_run_delay_frac,
    partial); gil_run_delay_frac is None when a GIL thread lacks run_delay_ns; partial lists
    the (tid, class) of threads with a single row in the window (CPU there unknown)."""
    tf, tl = first_last_in_window(th_rows, lambda x: x["tid"] if x.get("pid") == pid
                                  else None, ws, we)
    threads, gil, gil_rd, rd_ok, partial = [], 0.0, 0.0, True, []
    for tid in sorted(tf):
        a, b = tf[tid], tl[tid]
        tspan = b["t_rel_ns"] - a["t_rel_ns"]
        if tspan <= 0:
            partial.append((tid, classes.get(tid, (UNKNOWN_CLASS, None))[0]))
            continue
        cores_t = ((b["utime_ticks"] + b["stime_ticks"]) - (a["utime_ticks"]
                                                          + a["stime_ticks"])) / clk / (tspan / NS)
        rd = None
        if isinstance(b.get("run_delay_ns"), int) and isinstance(a.get("run_delay_ns"), int):
            rd = (b["run_delay_ns"] - a["run_delay_ns"]) / tspan
        cls, ident = classes.get(tid, (UNKNOWN_CLASS, None))
        threads.append({"tid": tid, "pid": pid, "comm": b.get("comm"), "class": cls,
                        "identified_as": ident, "cores": cores_t, "run_delay_frac": rd})
        if cls == GIL_CLASS:
            gil += cores_t
            if rd is None:
                rd_ok = False
            else:
                gil_rd += rd
    return threads, gil, (gil_rd if rd_ok else None), partial


def group_readings(ld, td, group):
    """CPU readings of one core group over the tier's window: ({...}, None) or (None, why).

    Core-level keys (the group's cores, per-core /proc/stat): pids, cores (sorted list),
    n_cores, core_busy_frac / core_busy_cores (non-idle ticks without softirq),
    core_busy_frac_softirq (with it), unattributed_cores (core busy without softirq - every
    thread of the group). Per process (one GIL each): procs {pid: {threads [{tid, pid, class,
    identified_as, cores, run_delay_frac}], gil_cores, gil_run_delay_frac, partial_threads
    [(tid, class) with a single row in the window]}}."""
    clk = (ld.sidecar_meta or {}).get("clk_tck")
    cpu_rows, th_rows = ld.sidecar("cpus"), ld.sidecar("threads")
    if cpu_rows is None or th_rows is None or not isinstance(clk, int) or clk <= 0:
        return None, "sidecar cpus/threads/clk_tck missing"
    cores = group["cores"]
    if group["shared_with"]:
        return None, "cores %s of pids %s are shared with process(es) %s of another role" \
            % (sorted(cores), group["pids"], group["shared_with"])
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
    procs, missing = {}, []
    for pid in group["pids"]:
        th, gil, gil_rd, partial = _thread_readings(th_rows[1], thread_classes(ld, td, pid),
                                                    pid, clk, ws, we)
        if not th:
            missing.append(pid)
        procs[pid] = {"threads": th, "gil_cores": gil, "gil_run_delay_frac": gil_rd,
                      "partial_threads": partial}
    if missing:
        return None, "pid(s) %s: fewer than two sidecar thread rows in the window" % missing
    core_busy_cores = busy / clk / (span / NS)
    return {"pids": list(group["pids"]), "cores": sorted(cores), "n_cores": len(cores),
            "core_busy_cores": core_busy_cores,
            "core_busy_frac": busy / total if total > 0 else None,
            "core_busy_frac_softirq": (busy + softirq) / total if total > 0 else None,
            "procs": procs,
            "unattributed_cores": core_busy_cores - sum(t["cores"] for p in procs.values()
                                                        for t in p["threads"])}, None


CORE_KEYS = ("pids", "cores", "n_cores", "core_busy_cores", "core_busy_frac",
             "core_busy_frac_softirq", "unattributed_cores")


def unit_view(g, pid=None):
    """What i1_process reads for one unit of a group's readings `g`: the group's core-level
    keys with the threads, GIL domain and run-queue wait of process `pid` (several cores: one
    unit per process), or of every process of the group (pid None: a single-core group is one
    unit; its GIL keys are not read there)."""
    if g is None:
        return None
    v = {k: g[k] for k in CORE_KEYS}
    ps = [g["procs"][pid]] if pid is not None else [g["procs"][p] for p in g["pids"]]
    v["threads"] = [t for p in ps for t in p["threads"]]
    if pid is not None:
        v.update(pid=pid, gil_cores=ps[0]["gil_cores"],
                 gil_run_delay_frac=ps[0]["gil_run_delay_frac"],
                 partial_threads=ps[0].get("partial_threads") or [])
    return v


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
    # a thread of unknown class that used CPU in either window may belong to the GIL domain:
    # the domain's reading is incomplete (never recorded or evaluated as if complete), and the
    # thread is no non-GIL candidate either
    unknown = [] if single else sorted(
        {t["tid"] for v in (cur, base) for t in v["threads"]
         if t["class"] == UNKNOWN_CLASS and t["cores"] > 0}
        | {tid for v in (cur, base) for tid, cls in v.get("partial_threads") or []
           if cls in (GIL_CLASS, UNKNOWN_CLASS)})
    for key, qty in qtys:
        v = None if unknown else _offset(cur, base, key)
        k = band_key("T2", qty, role)
        if v is not None and not is_baseline:
            record.setdefault(k, []).append(v)
        offs[qty] = (v, k)
    base_t = {t["tid"]: t["cores"] for t in base["threads"]}
    nongil_offs = {t["tid"]: t["cores"] - base_t.get(t["tid"], 0.0) for t in cur["threads"]
                   if t["class"] not in (GIL_CLASS, UNKNOWN_CLASS)}
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
    if unknown:
        out["unknown_threads"] = unknown
        out["reasons"].append("GIL domain: thread readings incomplete: tids %s (class unknown "
                              "with CPU in the window, or a python / unknown thread with one "
                              "sidecar row there)" % unknown)
    elif cmp_g is None or cmp_r is None:
        out["reasons"].append("GIL domain: no band %s / %s or no value" % (kg, kr))
    else:
        gil = TRUE if (cur["gil_cores"] >= near and cmp_g == "above"
                       and cmp_r != "above") else FALSE
    bn = bands.get(kn)
    hot = [t["tid"] for t in cur["threads"] if t["tid"] in nongil_offs and bn is not None
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

def i1_role(ld, td, base_td, role, readings, eval_unit):
    """I1 units of one role: a single-core group is one unit (its core is the reading), a
    group of several cores has one unit per process (one GIL each); every unit lists the
    group's pids. Processes without cpus and groups sharing a core with another role are
    NOT_RUN."""
    groups, nocpu = core_groups(ld, role)
    per = [{"pid": pid, "pids": [pid], "status": NOT_RUN,
            "reasons": ["pid %d has no cpus in roles.json" % pid]} for pid in nocpu]
    for g in groups:
        cur, why = readings(td, g)
        if cur is None:
            per.append({"pid": None, "pids": g["pids"], "cores": sorted(g["cores"]),
                        "status": NOT_RUN, "reasons": [why]})
            continue
        base = readings(base_td, g)[0] if base_td is not None else None
        units = [None] if cur["n_cores"] == 1 else list(g["pids"])
        for pid in units:
            v = unit_view(cur, pid)
            r = eval_unit(v, unit_view(base, pid))
            r.update(pid=pid, pids=g["pids"], unit="core_group" if pid is None else "process",
                     readings=v)
            per.append(r)
    return per


def localize_tier(ctx, ld, td, base_td, x, record, procs_cache):
    """I for one load tier (called through the symptoms hook with the tier's X result).

    procs_cache: {(tier seq, group pids): group readings or (None, why)} shared across tiers
    so the baseline tier's readings are computed once."""
    bands = ctx.bands.view(ctx.tier_regime[td.tier.seq])
    pilot = ctx.bands.extras.get("pilot") or {}
    near = (pilot.get("i1_calibration") or {}).get("near_one_core_min")
    d14 = (pilot.get("d14_softirq") or {}).get("irq_softirq_is_experiment")
    d14 = d14 if isinstance(d14, bool) else None
    is_base = base_td is td
    out = {"I1": {}, "I2": {}, "I3": {"readings": [], "roles": {}}, "reasons": []}
    roles = sorted(r for r, d in ctx.roles.items() if d["measure_class"] in CANDIDATE_CLASSES)

    def readings(tdx, group):
        key = (tdx.tier.seq, tuple(group["pids"]))
        if key not in procs_cache:
            procs_cache[key] = group_readings(ld, tdx, group)
        return procs_cache[key]

    for role in roles:
        per = i1_role(ld, td, base_td, role, readings,
                      lambda cur, base: i1_process(cur, base, bands, role, near, d14, record,
                                                   is_base))
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
            not_run.append(("I1", role))
    for role, v in out["I3"]["roles"].items():
        if v["status"] == TRUE:
            holding.setdefault(role, []).append("I3")
        elif v["status"] == NOT_RUN:
            not_run.append(("I3", role))
    out["holding"] = {r: sorted(s) for r, s in sorted(holding.items())}
    out["not_run"] = ["%s:%s" % sr for sr in not_run]
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
    competing = sorted({"%s:%s" % (s, r) for s, r in not_run if r != role})
    if competing:
        # an unevaluated signal of another role may point elsewhere: not "not that role"
        out["status"] = NOT_RUN
        out["reasons"].append("only %s holds; competing roles not evaluated: %s"
                              % (role, competing))
        return out
    units = (out["I1"].get(role) or {}).get("processes", [])
    hot = [p for p in units if p.get("status") == TRUE]
    if hot and any(p.get("via") in ("core_idle", "gil_domain") for p in hot):
        finding = "domain_saturated"
    elif hot:
        finding = "thread_saturated"
    else:
        finding = "backlog_at_entry"
    pids = sorted({q for p in hot for q in p.get("pids") or [p.get("pid")]})
    shares = {}
    for p in units:
        if set(p.get("pids") or []) & set(pids):
            for t in (p.get("readings") or {}).get("threads", []):
                shares.setdefault(str(t["pid"]), []).append(
                    {k: t[k] for k in ("tid", "class", "identified_as", "cores")})
    kinds = {p.get("unit") for p in hot}
    out.update(status=LOCALIZED, role=role, via=via, finding=finding,
               measure_class=ctx.roles[role]["measure_class"],
               granularity="process" if "process" in kinds else
               ("core_group" if kinds else "role"),
               pids=pids, thread_shares=shares)
    if not_run:
        out["reasons"].append("localised with the role's own signals not evaluated: %s"
                              % out["not_run"])
    return out

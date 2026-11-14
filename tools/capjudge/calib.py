# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibration files of stage 2 (INTERFACES 9.6-9.8): schema validation, the per-worker /
per-process statistics the judge and the builders share, and the builders.

    python3 -m capjudge.calib near-boundary --inputs <list> --workdir <W> --regime R --out F
    python3 -m capjudge.calib loadgen-band  --inputs <list> --workdir <W> --out F
    python3 -m capjudge.calib stub-calib    --inputs <list> --workdir <W> --out F

(`analysis/calib_build.py` is the same entry point.) Every builder reads only judged launch
trees named explicitly in a list file and re-reads raw data through the verified inputs
(capjudge.inputs.verify), so a builder and the judge compute each statistic with the same code.
"""

from capcommon import hist
from capcommon.schedule import per_worker_tier_counts

from . import stats
from .symptoms import request_lag_series
from .thresholds import REGIMES, BandError, check_noise_bands
from .treedata import NS, first_last_in_window, summary_value

NEAR_BOUNDARY_SCHEMA = "capkit.near_boundary_noise/1"
LOADGEN_SCHEMA = "capkit.loadgen_band/1"
STUB_SCHEMA = "capkit.stub_calib/1"
PILOT_SCHEMA = "capkit.pilot/1"
ENV_KEYS = ("hw_profile_sha", "image_sha", "capkit_sha", "baseline_sha")
# Load keys a shape's healthy maximum can be written in (computable from one tier's parameters).
LOAD_KEYS = ("rate_rps", "nominal_msgs_per_s")
# Shape statistics ("<T>:<quantity>"; the object is the bound role). C1 per load-source worker,
# C2 per out-of-domain stub worker.
LOADGEN_STATS = ("T1:send_lag_p50_ns", "T3:send_lag_p99_ns", "T3:fionread_nonzero_frac",
                 "T3:loop_lag_p99_ns")
STUB_STATS = ("T3:loop_lag_p99_ns", "T3:thread_cpu_cores")
# kind -> (accepted schemas, shapes key, healthy-maximum key, statistic names); None = no shapes
SHAPE_KINDS = {"loadgen_band": ((LOADGEN_SCHEMA,), "shapes", "per_worker_max", LOADGEN_STATS),
               "stub_calib": ((STUB_SCHEMA,), "out_of_domain", "per_process_max", STUB_STATS)}


class CalibError(Exception):
    pass


# ---- schema validation (the judge calls validate() for every registered calibration file) ---

def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_env(doc, where):
    env = doc.get("environment")
    if not isinstance(env, dict) or any(k not in env for k in ENV_KEYS):
        raise CalibError("%s: environment must have %s" % (where, ENV_KEYS))


def _check_match(m, where):
    if not isinstance(m, dict) or set(m) != {"tokens_per_msg", "streams_c"}:
        raise CalibError("%s: match must be {tokens_per_msg: [...], streams_c: [...]}" % where)
    k, c = m["tokens_per_msg"], m["streams_c"]
    if not isinstance(k, list) or not k or not all(isinstance(x, int) and not isinstance(
            x, bool) and x >= 1 for x in k):
        raise CalibError("%s: match.tokens_per_msg must be a non-empty list of ints >= 1"
                         % where)
    if not isinstance(c, list) or not c or not all(x is None or (isinstance(x, int) and not
                                                                  isinstance(x, bool) and x >= 1)
                                                    for x in c):
        raise CalibError("%s: match.streams_c must be a non-empty list of null / ints >= 1"
                         % where)


def _check_max(mx, where):
    if not isinstance(mx, dict) or set(mx) != {"load_key", "value", "censored", "n_launches"}:
        raise CalibError("%s must be {load_key, value, censored, n_launches}" % where)
    if mx["load_key"] not in LOAD_KEYS:
        raise CalibError("%s.load_key must be one of %s" % (where, LOAD_KEYS))
    if mx["value"] is not None and (not _is_num(mx["value"]) or mx["value"] <= 0):
        raise CalibError("%s.value must be null or a number > 0" % where)
    if not isinstance(mx["censored"], bool):
        raise CalibError("%s.censored must be a bool" % where)
    n = mx["n_launches"]
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise CalibError("%s.n_launches must be an int >= 0" % where)


def _check_shapes(doc, kind, where):
    _schemas, skey, mkey, names = SHAPE_KINDS[kind]
    shapes = doc.get(skey)
    if not isinstance(shapes, dict) or not shapes:
        raise CalibError("%s: %s must be a non-empty object" % (where, skey))
    for name, sh in shapes.items():
        w = "%s %s[%r]" % (where, skey, name)
        if not isinstance(name, str) or not name or not isinstance(sh, dict) \
                or set(sh) != {"match", mkey, "bands"}:
            raise CalibError("%s must be {match, %s, bands}" % (w, mkey))
        _check_match(sh["match"], w)
        _check_max(sh[mkey], "%s.%s" % (w, mkey))
        b = sh["bands"]
        if not isinstance(b, dict) or any(k not in names for k in b):
            raise CalibError("%s.bands keys must be among %s" % (w, names))
        for k, band in b.items():
            if not stats.band_is_valid(band):
                raise CalibError("%s.bands[%r] must have numeric min <= max and int n >= 1"
                                 % (w, k))


def validate(kind, doc, where):
    """Raise CalibError unless `doc` is a well-formed calibration file of `kind` (INTERFACES
    9.4, 9.6-9.8). Returns the regime of a near-boundary file (None for other kinds)."""
    if not isinstance(doc, dict):
        raise CalibError("%s: not a JSON object" % where)
    schema = doc.get("schema")
    if kind == "pilot":
        if schema != PILOT_SCHEMA:
            raise CalibError("%s: pilot schema must be %s" % (where, PILOT_SCHEMA))
        _check_env(doc, where)
        return None
    if kind == "near_boundary_noise":
        if schema == PILOT_SCHEMA:
            # A pilot-format file registered as this kind (RUNBOOK stage 2 D5: E1-0 bands built
            # by capjudge.pilot): only its noise_bands are read, each band's own regime counts.
            _check_env(doc, where)
            return None
        if schema != NEAR_BOUNDARY_SCHEMA:
            raise CalibError("%s: schema must be %s (or %s)" % (where, NEAR_BOUNDARY_SCHEMA,
                                                                 PILOT_SCHEMA))
        _check_env(doc, where)
        if doc.get("regime") not in REGIMES:
            raise CalibError("%s: regime must be one of %s" % (where, REGIMES))
        try:
            nb = check_noise_bands(doc.get("noise_bands"), where)
        except BandError as e:
            raise CalibError(str(e))
        off = sorted(k for k, (_b, reg) in nb.items() if reg != doc["regime"]
                     or "regime" not in _b)
        if off:
            raise CalibError("%s: bands %s must carry regime %r" % (where, off[:5],
                                                                     doc["regime"]))
        return doc["regime"]
    if kind in SHAPE_KINDS:
        if schema not in SHAPE_KINDS[kind][0]:
            raise CalibError("%s: schema must be %s" % (where, SHAPE_KINDS[kind][0][0]))
        _check_env(doc, where)
        if "noise_bands" in doc:
            raise CalibError("%s: a %s file has no noise_bands (its bands are per shape)"
                             % (where, kind))
        _check_shapes(doc, kind, where)
        if kind == "stub_calib" and doc.get("in_domain") is not None:
            raise CalibError("%s: in_domain is not defined in %s (must be null or absent)"
                             % (where, STUB_SCHEMA))
        return None
    raise CalibError("%s: unknown calibration kind %r" % (where, kind))


def shapes_of(kind, doc):
    """{shape name: shape} of a validated loadgen_band / stub_calib document."""
    return doc[SHAPE_KINDS[kind][1]]


def max_of(kind, shape):
    return shape[SHAPE_KINDS[kind][2]]


# ---- shared statistics -----------------------------------------------------------------------

def p99(values):
    """Nearest-rank 99th percentile; None for no values."""
    if not values:
        return None
    v = sorted(values)
    return v[max(0, -(-99 * len(v) // 100) - 1)]


def worker_capbin(td, kind, role, worker):
    for ident, header, cols in td.capbins.get(kind, []):
        if ident.get("role") == role and ident.get("worker") == worker:
            return header, cols
    return None, None


def tier_load(tier, load_key):
    """The tier's total offered load in `load_key` units, from its own parameters; None when
    the key cannot be computed for this tier (nominal msgs/s needs streams_c and a beat)."""
    if load_key == "rate_rps":
        return float(tier.rate_rps)
    if load_key == "nominal_msgs_per_s":
        if tier.streams_c is None or tier.beat_ns <= 0:
            return None
        return tier.streams_c * NS / (tier.beat_ns * tier.stream_interval)
    return None


def worker_share(ctx, tier, role, worker, split):
    """Planned share of the tier's load carried by one worker of `role`. split 'schedule': the
    worker's rows of the shared schedule over the role's rows (load generators); 'even':
    1 / workers (a stub's workers share one listening socket; nothing plans a split)."""
    w = ctx.roles[role]["workers"]
    if split == "even":
        return 1.0 / w
    counts = per_worker_tier_counts(ctx.schedule_meta, w)
    tot = sum(counts.get((j, tier.seq), 0) for j in range(w))
    return counts.get((worker, tier.seq), 0) / tot if tot else None


def worker_load(ctx, tier, role, worker, split, load_key):
    tl = tier_load(tier, load_key)
    sh = worker_share(ctx, tier, role, worker, split)
    return None if tl is None or sh is None else tl * sh


def shape_matches(shape, tier):
    m = shape["match"]
    return tier.tokens_per_msg in m["tokens_per_msg"] and tier.streams_c in m["streams_c"]


def _loop_p99(doc):
    h = summary_value(doc, "loop_lag_ns.hist.counts")
    if isinstance(h, list) and len(h) == hist.NBINS:
        return hist.quantile(h, 0.99)
    return None


def loadgen_worker_stats(td, role, worker):
    """C1 statistics of one load-generator worker in one tier: {name: value or None}, or None
    when the worker's summary or request records are missing. Names are LOADGEN_STATS."""
    doc = td.summaries.get(role, {}).get(worker)
    _h, cols = worker_capbin(td, "loadgen.requests", role, worker)
    if doc is None or cols is None:
        return None
    ws, we = td.tier.window_start_ns, td.tier.window_end_ns
    series, _ = request_lag_series(cols, td.tier, "send")
    t1 = stats.t1_trend(series)
    in_win = [s - p for p, s in zip(cols["planned_rel_ns"], cols["sent_rel_ns"])
              if ws <= p < we and s >= 0]
    fr = doc.get("fionread") or {}
    frac = (fr["nonzero"] / fr["samples"]) if fr.get("samples") else None
    return {"T1:send_lag_p50_ns": None if t1 is None else t1[0],
            "T3:send_lag_p99_ns": p99(in_win), "T3:fionread_nonzero_frac": frac,
            "T3:loop_lag_p99_ns": _loop_p99(doc)}


def pid_thread_cpu_cores(th_rows, pid, ws, we, clk):
    """(cores, None) or (None, why): utime + stime of every thread of `pid` between its first
    and last sidecar row inside [ws, we), over the longest such span, in cores."""
    tf, tl = first_last_in_window(th_rows, lambda x: (x["pid"], x["tid"])
                                  if x.get("pid") == pid else None, ws, we)
    keys = [k for k in tf if tl[k] is not tf[k]]
    if not keys:
        return None, "pid %s: fewer than two sidecar thread rows in the window" % pid
    span = max(tl[k]["t_rel_ns"] - tf[k]["t_rel_ns"] for k in keys)
    if span <= 0:
        return None, "pid %s: thread rows span no time in the window" % pid
    ticks = sum((tl[k]["utime_ticks"] + tl[k]["stime_ticks"])
                - (tf[k]["utime_ticks"] + tf[k]["stime_ticks"]) for k in keys)
    return ticks / clk / (span / NS), None


def stub_worker_stats(ld, td, role, worker):
    """C2 statistics of one stub worker (one process: the pid in its summary identity) in one
    tier: ({name: value or None}, {name: why}) or (None, why) without a summary."""
    doc = td.summaries.get(role, {}).get(worker)
    if doc is None:
        return None, "no summary"
    out, why = {"T3:loop_lag_p99_ns": _loop_p99(doc), "T3:thread_cpu_cores": None}, {}
    if out["T3:loop_lag_p99_ns"] is None:
        why["T3:loop_lag_p99_ns"] = "summary has no loop_lag_ns histogram"
    pid = (doc.get("identity") or {}).get("pid")
    th = ld.sidecar("threads")
    clk = (ld.sidecar_meta or {}).get("clk_tck")
    if not isinstance(pid, int) or th is None or not isinstance(clk, int) or clk <= 0:
        why["T3:thread_cpu_cores"] = "summary pid, sidecar threads.jsonl or clk_tck missing"
    else:
        v, w = pid_thread_cpu_cores(th[1], pid, td.tier.window_start_ns,
                                   td.tier.window_end_ns, clk)
        out["T3:thread_cpu_cores"] = v
        if w:
            why["T3:thread_cpu_cores"] = w
    return out, why


# ---- builders --------------------------------------------------------------------------------

def read_shape_list(path):
    """Builder list file for loadgen_band / stub_calib: one line per launch tree,
    `<tree>\\t<seq>[,<seq>...]\\t<role>\\t<shape>` (literal paths, no globs, no tree twice)."""
    import os
    out = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 4 or any(c in parts[0] for c in "*?[") or not parts[2] \
                    or not parts[3]:
                raise CalibError("%s:%d: need '<tree>\\t<seq,...>\\t<role>\\t<shape>'"
                                 % (path, n))
            try:
                seqs = sorted({int(x) for x in parts[1].split(",")})
            except ValueError:
                raise CalibError("%s:%d: tier seqs must be integers" % (path, n))
            out.append((parts[0], seqs, parts[2], parts[3]))
    if not out:
        raise CalibError("%s lists no tree" % path)
    if len({os.path.abspath(t) for t, *_ in out}) != len(out):
        raise CalibError("a tree is listed twice")
    return out


def _same_env(env, ctx, tree):
    from . import pilot as P
    e = P._env_of(ctx)
    if env is not None and e != env:
        raise CalibError("%s: environment %s differs from %s (one file per environment)"
                         % (tree, e, env))
    return e


def build_near_boundary(list_path, W, regime):
    """Noise bands (T3 [min, max] of the judge's per-tier statistics) over the listed tiers,
    every band tagged `regime`. Tiers with a RED check or a VOID / UNSTABLE verdict are left
    out: a noise band is measured where nothing has reached its limit."""
    import os
    from . import pilot as P
    if regime not in REGIMES:
        raise CalibError("--regime must be one of %s" % (REGIMES,))
    try:
        entries = P.read_list(list_path)
    except P.PilotError as e:
        raise CalibError(str(e))
    W = os.path.abspath(W)
    values, sources, inputs, env, experiment = {}, {}, [], None, None
    for tree, seqs in entries:
        sel = _select(tree, seqs, W, ("VOID", "UNSTABLE"))
        ctx, rec = sel["ctx"], sel["rec"]
        env = _same_env(env, ctx, tree)
        experiment = experiment or ctx.ident.experiment
        inputs.append(rec)
        for seq in rec["tiers_used"]:
            for key, vals in (sel["by_seq"][seq].get("stats") or {}).items():
                for v in vals:
                    values.setdefault(key, []).append(v)
                    sources.setdefault(key, []).append({"tree": rec["tree"], "tier_seq": seq})
    bands = {}
    for key, vals in sorted(values.items()):
        b = stats.t3_band(vals)
        if b is not None:
            bands[key] = dict(b, regime=regime, unit=P._unit(key), source_tiers=sources[key])
    if not bands:
        raise CalibError("no tier survived; nothing to build a noise band from")
    return {"schema": NEAR_BOUNDARY_SCHEMA, "experiment": experiment, "regime": regime,
            "environment": env, "inputs": inputs, "noise_bands": bands}


def _select(tree, seqs, W, drop):
    from . import pilot as P
    try:
        return P.select_tiers(tree, seqs, W, drop_verdicts=drop)
    except P.PilotError as e:
        raise CalibError(str(e))


SPLIT = {"loadgen_band": "schedule", "stub_calib": "even"}


def build_shapes(kind, list_path, W):
    """loadgen_band (E1) or the out-of-domain part of stub_calib (E2a). For each shape: the
    health bands are T3 [min, max] of the per-worker statistics (loadgen_worker_stats /
    stub_worker_stats) over the listed STABLE tiers; the healthy maximum is, per launch, the
    busiest worker's planned load in the highest STABLE tier (verdicts.json boundary_T4
    lower end), and over launches the smallest of those (the lower end of the T4 envelope);
    `censored` when some launch never became UNSTABLE. `match` lists the (tokens_per_msg,
    streams_c) of every tier that contributed; the judge applies the maximum only to matching
    tiers."""
    import os
    from .treedata import LaunchData, TierData
    entries = read_shape_list(list_path)
    _s, skey, mkey, names = SHAPE_KINDS[kind]
    split = SPLIT[kind]
    W = os.path.abspath(W)
    acc, inputs, env, experiment = {}, [], None, None
    for tree, seqs, role, shape in entries:
        sel = _select(tree, seqs, W, ("VOID", "UNSTABLE", "NOT_RUN"))
        ctx, rec = sel["ctx"], sel["rec"]
        env = _same_env(env, ctx, tree)
        experiment = experiment or ctx.ident.experiment
        rec.update(role=role, shape=shape)
        inputs.append(rec)
        if role not in ctx.roles:
            raise CalibError("%s: no role %r in launch.json" % (tree, role))
        lk = ctx.tt.axis["load_key"]
        if lk not in LOAD_KEYS:
            raise CalibError("%s: tier table load_key %r is not one of %s" % (tree, lk,
                                                                              LOAD_KEYS))
        S = acc.setdefault(shape, {"vals": {}, "src": {}, "tk": set(), "sc": set(),
                                   "maxes": [], "censored": False, "load_key": lk})
        if S["load_key"] != lk:
            raise CalibError("shape %s: inputs on load keys %s and %s" % (shape,
                                                                          S["load_key"], lk))
        if "tree_excluded" in rec:
            continue
        ld = LaunchData(ctx)
        workers = ctx.roles[role]["workers"]
        for seq in rec["tiers_used"]:
            td = TierData(ctx, ctx.tt.tiers[seq])
            S["tk"].add(td.tier.tokens_per_msg)
            S["sc"].add(td.tier.streams_c)
            for k in range(workers):
                st = loadgen_worker_stats(td, role, k) if kind == "loadgen_band" \
                    else stub_worker_stats(ld, td, role, k)[0]
                for name in names:
                    v = None if st is None else st.get(name)
                    if v is not None:
                        S["vals"].setdefault(name, []).append(v)
                        S["src"].setdefault(name, []).append(
                            {"tree": rec["tree"], "tier_seq": seq, "worker": k})
        b = sel["verdicts"].get("boundary_T4") or {}
        rec["boundary_T4"] = b
        if b.get("lower_seq") is not None:
            t = ctx.tt.tiers[b["lower_seq"]]
            loads = [worker_load(ctx, t, role, k, split, lk) for k in range(workers)]
            if None not in loads:
                S["maxes"].append(max(loads))
                S["tk"].add(t.tokens_per_msg)
                S["sc"].add(t.streams_c)
            if b.get("kind") == "lower_bound_only":
                S["censored"] = True
    shapes = {}
    for shape, S in sorted(acc.items()):
        if not S["tk"]:
            raise CalibError("shape %s: no tier survived" % shape)
        bands = {}
        for name in names:
            band = stats.t3_band(S["vals"].get(name, []))
            if band is not None:
                bands[name] = dict(band, unit=name.rsplit("_", 1)[1],
                                   source_tiers=S["src"][name])
        shapes[shape] = {
            "match": {"tokens_per_msg": sorted(S["tk"]),
                      "streams_c": sorted(S["sc"], key=lambda c: (c is not None, c or 0))},
            mkey: {"load_key": S["load_key"], "value": min(S["maxes"]) if S["maxes"] else None,
                   "censored": S["censored"], "n_launches": len(S["maxes"])},
            "bands": bands}
    doc = {"schema": SHAPE_KINDS[kind][0][0], "experiment": experiment, "environment": env,
           "inputs": inputs, skey: shapes}
    if kind == "stub_calib":
        doc["in_domain"] = None
    validate(kind, doc, "built %s" % kind)
    return doc


def main(argv=None):
    import argparse
    import os
    import sys
    from capcommon.exitcodes import ConfigError
    from capcommon.jsonio import atomic_write_bytes, dumps
    ap = argparse.ArgumentParser(prog="capjudge.calib", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="what", required=True)
    for name in ("near-boundary", "loadgen-band", "stub-calib"):
        p = sub.add_parser(name)
        p.add_argument("--inputs", required=True)
        p.add_argument("--workdir", required=True)
        p.add_argument("--out", required=True)
        if name == "near-boundary":
            p.add_argument("--regime", required=True, choices=REGIMES)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        sys.stderr.write("capjudge.calib: refusing to overwrite %s\n" % a.out)
        return 3
    try:
        if a.what == "near-boundary":
            doc = build_near_boundary(a.inputs, a.workdir, a.regime)
            what = "%d noise bands (regime %s)" % (len(doc["noise_bands"]), a.regime)
        else:
            kind = a.what.replace("-", "_")
            doc = build_shapes(kind, a.inputs, a.workdir)
            sh = shapes_of(kind, doc)
            what = "%d shape(s): %s" % (len(sh), ", ".join(
                "%s max=%s" % (n, max_of(kind, s)["value"]) for n, s in sorted(sh.items())))
    except (CalibError, ConfigError, OSError, ValueError) as e:
        sys.stderr.write("capjudge.calib: %s\n" % e)
        return 3
    atomic_write_bytes(a.out, (dumps(doc) + "\n").encode("utf-8"))
    print("wrote %s: %s" % (a.out, what))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

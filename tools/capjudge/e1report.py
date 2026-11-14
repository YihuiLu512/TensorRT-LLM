# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E1 report: slope reconciliation and count reconciliation (design E1 "对照与自检";
INTERFACES 14.3).

    python3 -m capjudge.e1report --boundary <boundary.json> --workdir <W> --out <new file>

Input is a capjudge.boundary product. Two parts:
  * slope: the known-answer check of E1 (A): a fixed busy delta on the load generator's
    per-pass path, three or more deltas; 1 / boundary against delta must have slope 1 once both
    are in the same unit. Launches judged in mode P+X+I+V: control launches (no dose) are
    delta = 0; dosed launches must dose one role, the same in all, with a heartbeat PASS (others
    are excluded and listed). Per delta level the envelope of the launches' intervals; for every
    pair of levels a < b the normalised slope lies in
        [(1/b.upper - 1/a.lower), (1/b.lower - 1/a.upper)] / (delta_b - delta_a) * W / p
    with W the dosed role's workers and p its passes per unit of load_value by construction
    (capcommon.expect on the launch's reconcile chain of the site's unit and the tier
    parameters; load_key rate_rps or nominal_msgs_per_s). Per pair: FAIL when a known end
    excludes 1; PASS when the interval is closed, contains 1 and its lower end is > 0 (an
    interval that also holds 0 cannot tell slope 1 from "the dose did not move the boundary");
    INCONCLUSIVE otherwise. Overall: FAIL if any pair fails, else INCONCLUSIVE if any pair is,
    else PASS; fewer than 3 levels or no usable p -> NOT_RUN.
  * counts per launch (B, "计数真实性"): bytes the sink sent == bytes the load generator
    received, and connections the sink accepted == connections the load generator opened, at
    launch scope (sums over every tier directory; the tools attribute these to tiers by planned
    or by event time, which differ). HTTP: the sink's bytes include one response head per
    stream, capload's do not, so the heads are rebuilt from the sink's per-stream records with
    capload.sink.response_head. gRPC: payload bytes on both sides; nullgrpc does not count
    accepts (NOT_RUN). The sink counts bytes handed to its transport: more than the load
    generator received is expected when requests failed, timed out or were unfinished there
    (bytes in flight when the stream was dropped), so that case is NOT_RUN with the
    difference; without such requests, and whenever the load generator received more, FAIL.
    A counter missing from any summary leaves its item NOT_RUN (never summed as 0).
  * sink health (with --sink-bands; E1 "隔离": the sink's thread CPU and event-loop lag stay
    in their sub-saturation band): per load tier and sink worker, the C2 statistics against the
    bands `T3:loop_lag_p99_ns:<sink>` / `T3:thread_cpu_cores:<sink>` of a registered
    calibration file (regime subsat). Above the band -> OUTSIDE; no value or no band ->
    NOT_RUN. The tree is re-verified (capjudge.inputs.verify) to read the raw data.
"""

import argparse
import os
import sys

from capcommon import expect
from capcommon.capbin import CapbinError, read_capbin
from capcommon.exitcodes import ConfigError
from capcommon.hashing import sha256_file
from capcommon.jsonio import atomic_write_bytes, dumps, read_json
from capcommon.tiers import TierTable

from . import boundary as B
from . import cmdrec
from . import intervene
from . import stats

SCHEMA = "capkit.e1_report/1"
PASS, FAIL, NOT_RUN, INCONCLUSIVE = "PASS", "FAIL", "NOT_RUN", "INCONCLUSIVE"
# byte counters per tool module: (summary path, ...); the HTTP sink's include response heads
CLIENT_BYTES = {"capload": ("frames", "bytes"), "capgrpc.load": ("msgs", "bytes")}
SERVER_BYTES = {"capload.nullsse": ("counts", "bytes_sent"),
                "capgrpc.nullgrpc": ("counts", "bytes_sent")}
HTTP_SINKS = ("capload.nullsse",)


class E1Error(ConfigError):
    pass


# ---- slope -------------------------------------------------------------------------------------

def passes_per_unit(tree, W, role, site):
    """(p, None) passes of `role` at `site` per unit of load_value, the same on every load tier;
    or (None, why)."""
    unit = intervene.SITE_UNIT.get(site)
    if unit is None:
        return None, "site %s has no per-pass unit" % site
    launch = read_json(os.path.join(tree, "raw", "launch.json"))
    tt_ref = launch.get("tier_table") or {}
    path = os.path.join(W, tt_ref.get("relpath", ""))
    if not os.path.isfile(path) or sha256_file(path) != tt_ref.get("sha256"):
        return None, "tier table %s missing or not the registered sha" % tt_ref.get("relpath")
    tt = TierTable.load(path)
    chains = [c for c in launch["reconcile"]["chains"] if c["kind"] == unit
              and any(h["role"] == role for h in c["hops"])]
    if len(chains) != 1:
        return None, "%d reconcile chains of kind %s count %s" % (len(chains), unit, role)
    key = tt.axis["load_key"]
    ps = set()
    for t in tt.tiers:
        if t.kind not in ("up", "down"):
            continue
        per_req = expect.expected_hop_count(chains[0], 1, t.max_tokens, t.tokens_per_msg,
                                            t.stream_interval, role)
        if key == "rate_rps":
            ps.add(round(per_req, 9))
        elif key == "nominal_msgs_per_s":
            ps.add(round(per_req * t.rate_rps / t.load_value, 9))
        else:
            return None, "load_key %s: no per-pass conversion" % key
    if len(ps) != 1:
        return None, "passes per unit differ between load tiers %s" % sorted(ps)
    return ps.pop(), None


def slope(bdoc, W):
    out = {"status": NOT_RUN, "levels": [], "pairs": [], "excluded": [], "reasons": []}
    control, dosed = [], []
    for x in bdoc["launches"]:
        if x.get("mode") != "P+X+I+V":
            out["excluded"].append({"tree": x["tree"], "why": "mode %s" % x.get("mode")})
        elif x["doses"] == {}:
            control.append(x)
        elif x["doses"] == "mixed" or len(x["doses"] or {}) != 1:
            out["excluded"].append({"tree": x["tree"], "why": "doses %s" % x["doses"]})
        elif x["heartbeat"] != intervene.PASS:
            out["excluded"].append({"tree": x["tree"], "why": "heartbeat %s" % x["heartbeat"]})
        else:
            dosed.append(x)
    if not dosed:
        out["reasons"].append("no usable dosed launch")
        return out
    targets = {(next(iter(x["doses"])), tuple(d["site"] for d in next(iter(
        x["doses"].values())))) for x in dosed}
    if len(targets) != 1:
        out["reasons"].append("dosed launches must dose one role at one site: %s"
                              % sorted(targets))
        return out
    (role, sites), = targets
    if len(sites) != 1:
        out["reasons"].append("dose on several sites %s" % list(sites))
        return out
    out.update(role=role, site=sites[0])
    ps, workers = set(), set()
    for x in dosed:
        p, why = passes_per_unit(x["tree"], W, role, sites[0])
        if p is None:
            out["reasons"].append("%s: %s" % (x["tree"], why))
            return out
        ps.add(p)
        workers.add(x["roles"][role]["workers"])
    if len(ps) != 1 or len(workers) != 1:
        out["reasons"].append("passes per unit %s or workers %s differ" % (ps, workers))
        return out
    p, w = ps.pop(), workers.pop()
    out.update(passes_per_unit=p, workers=w)
    levels = {0.0: control} if control else {}
    for x in dosed:
        d = intervene._scalar_dose(x["doses"][role]) / 1e9
        levels.setdefault(d, []).append(x)
    order = bdoc["axis"]["order"]
    for d in sorted(levels):
        env = stats.t4_envelope([(x["interval"]["lower"], x["interval"]["upper"])
                                 for x in levels[d]], order)
        out["levels"].append({"delta_s": d, "trees": [x["tree"] for x in levels[d]],
                              "envelope": env})
    if len(out["levels"]) < 3:
        out["reasons"].append("%d delta level(s); the slope needs at least 3"
                              % len(out["levels"]))
        return out
    if order != "asc":
        out["reasons"].append("axis order %s: 1 / boundary needs an ascending axis" % order)
        return out
    verdicts = []
    for i, a in enumerate(out["levels"]):
        for b in out["levels"][i + 1:]:
            ea, eb, dd = a["envelope"], b["envelope"], b["delta_s"] - a["delta_s"]
            k = w / p / dd
            lo = (None if None in (stats.ratio(1, eb["upper"]), stats.ratio(1, ea["lower"]))
                  else (1 / eb["upper"] - 1 / ea["lower"]) * k)
            hi = (None if None in (stats.ratio(1, eb["lower"]), stats.ratio(1, ea["upper"]))
                  else (1 / eb["lower"] - 1 / ea["upper"]) * k)
            if (lo is not None and lo > 1) or (hi is not None and hi < 1):
                v = FAIL
            elif lo is not None and hi is not None and lo > 0:
                v = PASS
            else:  # an open end, or the interval also holds 0: the dose may not have moved it
                v = INCONCLUSIVE
            verdicts.append(v)
            out["pairs"].append({"from_delta_s": a["delta_s"], "to_delta_s": b["delta_s"],
                                 "slope_min": lo, "slope_max": hi, "status": v})
    out["status"] = (FAIL if FAIL in verdicts else INCONCLUSIVE if INCONCLUSIVE in verdicts
                     else PASS)
    return out


# ---- counts ------------------------------------------------------------------------------------

def tier_dirs(tree):
    """Names of the tree's tier directories (`NNN_<label>`), sorted; other entries of
    raw/tiers (pre/) are not tiers."""
    base = os.path.join(tree, "raw", "tiers")
    return sorted(d for d in os.listdir(base) if d[:3].isdigit() and d[3:4] == "_"
                  and os.path.isdir(os.path.join(base, d)))


def _summaries(tree, role):
    """[(tier dir, worker, doc)] of every per-tier summary of `role` in the tree."""
    out = []
    pre = role + ".w"
    for d in tier_dirs(tree):
        dp = os.path.join(tree, "raw", "tiers", d)
        for f in sorted(os.listdir(dp)):
            if f.startswith(pre) and f.endswith(".summary.json"):
                out.append((d, int(f[len(pre):-len(".summary.json")]),
                            read_json(os.path.join(dp, f))))
    return out


def _get(doc, path):
    v = doc
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
    return v


def _sum(docs, path):
    """(sum of the integer at `path` over the summaries, None) or (None, why) when one lacks
    it (never counted as 0)."""
    vals = [_get(d, path) for _t, _w, d in docs]
    bad = [(t, w) for (t, w, _d), v in zip(docs, vals) if not isinstance(v, int)]
    if bad:
        return None, "%d summaries lack %s (first: %s w%s)" % (len(bad), ".".join(path),
                                                                bad[0][0], bad[0][1])
    return sum(vals), None


def _head_bytes(tree, role):
    """(sum of response-head bytes rebuilt from the sink's stream records, None) or (None, why)."""
    from capload.sink import response_head  # noqa: PLC0415
    total = rows = 0
    for d in tier_dirs(tree):
        dp = os.path.join(tree, "raw", "tiers", d)
        for f in sorted(os.listdir(dp)):
            if f.startswith(role + ".w") and f.endswith(".streams.capbin"):
                try:
                    _h, cols = read_capbin(os.path.join(dp, f), ["req_id"])
                except CapbinError as e:
                    return None, "%s: %s" % (f, e)
                rows += len(cols["req_id"])
                total += sum(len(response_head(int(r), False)) for r in cols["req_id"])
    return (total, rows), None


def counts(tree):
    """Bytes and accepts of the one client (load generator) and one server (sink) role."""
    launch = read_json(os.path.join(tree, "raw", "launch.json"))
    roles = launch.get("roles") or []
    cl = [r for r in roles if r.get("kind") == "client"]
    sv = [r for r in roles if r.get("kind") == "server"]
    out = {"tree": tree, "bytes": {"status": NOT_RUN}, "accepts": {"status": NOT_RUN}}
    if len(cl) != 1 or len(sv) != 1 or sv[0].get("measure_class") != "stub_out_of_domain":
        why = ("need exactly one client and one server role, the server an out-of-domain "
               "sink (E1 topology), have %s / %s" % ([r["role"] for r in cl],
                                                     [r["role"] for r in sv]))
        out["bytes"]["why"] = out["accepts"]["why"] = why
        return out
    c, s = cl[0], sv[0]
    out.update(client=c["role"], server=s["role"])
    cs, ss = _summaries(tree, c["role"]), _summaries(tree, s["role"])
    n_tiers = len(tier_dirs(tree))
    short = ["%s: %d of %d per-tier summaries" % (r["role"], len(x), r["workers"] * n_tiers)
             for r, x in ((c, cs), (s, ss)) if len(x) != r["workers"] * n_tiers]
    if short:  # a missing summary would drop from both sums alike: never a PASS
        out["bytes"]["why"] = out["accepts"]["why"] = "; ".join(short)
        return out
    out["bytes"] = _bytes(tree, c, s, cs, ss)
    if s["module"] in HTTP_SINKS and c["module"] == "capload":
        ca, wc = _sum(cs, ("conn_pool", "new_conns"))
        sa, wa = _sum(ss, ("counts", "accepts"))
        out["accepts"] = {"client_new_conns": ca, "server_accepts": sa}
        if wc or wa:
            out["accepts"].update(status=NOT_RUN, why="; ".join(w for w in (wc, wa) if w))
        else:
            out["accepts"]["status"] = PASS if ca == sa else FAIL
    else:
        out["accepts"]["why"] = "%s does not count accepted connections" % s["module"]
    return out


def _bytes(tree, c, s, cs, ss):
    cp, sp = CLIENT_BYTES.get(c["module"]), SERVER_BYTES.get(s["module"])
    if cp is None or sp is None:
        return {"status": NOT_RUN, "why": "no byte counter for %s / %s" % (c["module"],
                                                                           s["module"])}
    cb, wc = _sum(cs, cp)
    sb, ws = _sum(ss, sp)
    if wc or ws:
        return {"status": NOT_RUN, "why": "; ".join(w for w in (wc, ws) if w)}
    res = {"client_bytes": cb, "server_bytes": sb, "head_bytes": 0}
    if s["module"] in HTTP_SINKS:
        got = {}
        for name, path in (("foreign", ("counts", "foreign_requests")),
                           ("requests", ("counts", "requests")),
                           ("overflow", ("overflow", "streams"))):
            got[name], why = _sum(ss, path)
            if why:
                res.update(status=NOT_RUN, why="heads cannot be rebuilt: " + why)
                return res
        hb, why = _head_bytes(tree, s["role"])
        if got["foreign"] or got["overflow"] or hb is None or hb[1] != got["requests"]:
            res.update(status=NOT_RUN, why="heads cannot be rebuilt: foreign %d, stream overflow "
                       "%d, %s" % (got["foreign"], got["overflow"], why or "%s stream records for "
                                   "%d requests" % (hb[1], got["requests"])))
            return res
        res["head_bytes"] = hb[0]
    diff = res["server_bytes"] - res["head_bytes"] - res["client_bytes"]
    res["difference"] = diff
    if diff == 0:
        res["status"] = PASS
    elif diff < 0:
        res.update(status=FAIL, why="the load generator received %d bytes more than the sink "
                   "sent" % -diff)
    else:
        res.update(_in_flight(cs, diff, c["module"]))
    return res


# load-generator counters of requests that may leave sink bytes in flight (capgrpc.load
# counts a request it gave up on as failed; it has no unfinished counter)
FAILED_COUNTERS = {"capload": (("counts", "failed"), ("counts", "unfinished")),
                   "capgrpc.load": (("counts", "failed"),)}


def _in_flight(cs, diff, module):
    """The sink sent `diff` > 0 bytes more than the load generator received: NOT_RUN when the
    load generator had failed or unfinished requests (bytes in flight when they were dropped),
    else FAIL."""
    n = {}
    for path in FAILED_COUNTERS.get(module, (("counts", "failed"),)):
        n[path[-1]], why = _sum(cs, path)
        if why:
            return {"status": NOT_RUN, "why": "sink sent %d bytes more; %s" % (diff, why)}
    failed, unfinished = n.get("failed", 0), n.get("unfinished", 0)
    if failed or unfinished:
        return {"status": NOT_RUN, "why": "sink sent %d bytes more; the load generator had %d "
                "failed and %d unfinished requests (bytes in flight when dropped)"
                % (diff, failed, unfinished)}
    return {"status": FAIL, "why": "sink sent %d bytes more and no request failed" % diff}


# ---- null-sink health --------------------------------------------------------------------------

SINK_STATS = ("T3:loop_lag_p99_ns", "T3:thread_cpu_cores")


def load_sink_bands(path, W):
    """noise_bands of a registered calibration file (any kind whose `noise_bands` carry the
    sink keys; subsat regime only). Raises E1Error when the file is not registered."""
    from capcommon import ledger  # noqa: PLC0415
    if ledger.registry_lookup(os.path.join(W, "REGISTRY.tsv"), sha256_file(path)) is None:
        raise E1Error("%s is not registered in <W>/REGISTRY.tsv" % path)
    doc = read_json(path)
    nb = doc.get("noise_bands") if isinstance(doc, dict) else None
    if not isinstance(nb, dict):
        raise E1Error("%s has no noise_bands" % path)
    return {k: b for k, b in nb.items() if stats.band_is_valid(b)
            and b.get("regime", "subsat") == "subsat"}


def sink_health(tree, W, bands):
    """Per load tier and sink worker: the C2 statistics (capjudge.calib.stub_worker_stats)
    against the sink's sub-saturation bands; a reading above the band is OUTSIDE."""
    from . import calib  # noqa: PLC0415
    from . import inputs as I  # noqa: PLC0415
    from .treedata import LaunchData, TierData  # noqa: PLC0415
    try:
        ctx = I.verify(W, tree)
    except I.InputInvalid as e:
        return {"status": NOT_RUN, "why": "input invalid: %s" % e}
    sinks = [r for r, d in sorted(ctx.roles.items())
             if d["measure_class"] == "stub_out_of_domain"]
    ld = LaunchData(ctx)
    out = {"status": NOT_RUN, "tiers": []}
    sts = []
    for t in ctx.tt.tiers:
        if t.kind not in ("up", "down"):
            continue
        td = TierData(ctx, t)
        for role in sinks:
            for k in range(ctx.roles[role]["workers"]):
                st, why = calib.stub_worker_stats(ld, td, role, k)
                row = {"tier_seq": t.seq, "role": role, "worker": k, "items": {}}
                for name in SINK_STATS:
                    val = None if st is None else st.get(name)
                    band = bands.get("%s:%s" % (name, role))
                    cmp_ = stats.compare_to_band(val, band)
                    row["items"][name] = {"value": val, "band": band,
                                          "status": NOT_RUN if cmp_ is None else
                                          "OUTSIDE" if cmp_ == "above" else "INSIDE"}
                    sts.append(row["items"][name]["status"])
                if st is None:
                    row["why"] = why
                out["tiers"].append(row)
    if sts:
        out["status"] = ("OUTSIDE" if "OUTSIDE" in sts else NOT_RUN if NOT_RUN in sts
                         else "INSIDE")
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="capjudge.e1report", description=__doc__.split("\n")[0])
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sink-bands", help="registered calibration file with the sink's "
                    "T3:loop_lag_p99_ns / T3:thread_cpu_cores sub-saturation bands")
    a = ap.parse_args(argv)
    try:
        bad = cmdrec.refuse_existing(a.out)
        if bad:
            raise E1Error(bad)
        bdoc = read_json(a.boundary)
        if not isinstance(bdoc, dict) or bdoc.get("schema") != B.SCHEMA:
            raise E1Error("%s is not a %s file" % (a.boundary, B.SCHEMA))
        W = os.path.abspath(a.workdir)
        doc = {"schema": SCHEMA, "boundary": os.path.abspath(a.boundary),
               "boundary_sha256": sha256_file(a.boundary), "axis": bdoc["axis"],
               "slope": slope(bdoc, W),
               "counts": [counts(x["tree"]) for x in bdoc["launches"]]}
        if a.sink_bands:
            bands = load_sink_bands(a.sink_bands, W)
            doc["sink_bands"] = {"path": os.path.abspath(a.sink_bands),
                                 "sha256": sha256_file(a.sink_bands)}
            doc["sink_health"] = [dict(sink_health(x["tree"], W, bands), tree=x["tree"])
                                  for x in bdoc["launches"]]
        else:
            doc["sink_health"] = {"status": NOT_RUN, "why": "no --sink-bands"}
    except (E1Error, ConfigError, OSError, ValueError, KeyError) as e:
        sys.stderr.write("capjudge.e1report: %s\n" % e)
        return 3
    atomic_write_bytes(a.out, (dumps(doc) + "\n").encode("utf-8"))
    cmdrec.write(a.out, "capjudge.e1report", argv,
                 [a.boundary] + ([a.sink_bands] if a.sink_bands else []))
    st = {}
    for c in doc["counts"]:
        for k in ("bytes", "accepts"):
            st.setdefault(k, []).append(c[k]["status"])
    print("wrote %s: slope %s; bytes %s; accepts %s" % (
        a.out, doc["slope"]["status"], sorted(set(st.get("bytes", []))),
        sorted(set(st.get("accepts", [])))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

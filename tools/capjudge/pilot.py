# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capjudge.pilot: build E0's pilot.json (INTERFACES 9.4) from sub-saturation tiers.

    python3 -m capjudge.pilot --inputs <list file> --workdir <W> --d14-rule <rule.json>
        --out <new pilot.json>

The list file names every input explicitly, one per line: `<launch tree>\t<seq>[,<seq>...]`
(the sub-saturation tiers of that tree to use; no globs, no default "all tiers"). Every tree
must already be judged: noise bands (T3) are the empirical [min, max] of the statistics the
judge wrote into verdicts.json `stats` for those tiers, so judge and pilot share one code path.
Tiers with any RED check or a VOID verdict are excluded and listed. A whole tree is excluded
(listed with the reason, all its tiers unused) unless the launch-level conclusion recorded
before the judge is VALID, both as the judge recorded it (verdicts.json launch_checks
LAUNCH-STATE PASS) and as STATE / launch_end.json read now: a launch that caplaunch or capjobs
recorded VOID, ENV_INVALID or NOT_RUN never enters a noise band. The judge's own launch verdict
is not required to be VALID (E0 trees are judged without noise bands, so it is NOT_RUN there).
Raw data are re-read (via the same verified inputs) only for the steady-window and warm-up
lengths (searched per arm) and for the D14 readings (capjudge.d14: idle window vs the used load
tiers, decided by the registered rule --d14-rule).
"""

import argparse
import bisect
import os
import sys

from capcommon.exitcodes import ConfigError
from capcommon.hashing import sha256_file
from capcommon.jsonio import atomic_write_bytes, dumps, read_json

from . import d14 as D14
from . import inputs as I
from . import stats
from .symptoms import msg_lag_from_chunks, request_lag_series
from .treedata import NS, LaunchData, TierData, role_cpus

SCHEMA = "capkit.pilot/1"
NOT_COMPUTED = "NOT_IMPLEMENTED_BATCH2"


class PilotError(ConfigError):
    pass


def read_list(path):
    out = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2 or any(c in parts[0] for c in "*?["):
                raise PilotError("%s:%d: need '<tree>\\t<seq,...>' with a literal path"
                                 % (path, n))
            try:
                seqs = sorted({int(x) for x in parts[1].split(",")})
            except ValueError:
                raise PilotError("%s:%d: tier seqs must be integers" % (path, n))
            out.append((parts[0], seqs))
    if not out:
        raise PilotError("%s lists no tree" % path)
    if len({os.path.abspath(t) for t, _ in out}) != len(out):
        raise PilotError("a tree is listed twice")
    return out


def _unit(key):
    q = key.split(":")[1]
    if q.endswith("_ns"):
        return "ns"
    if q.endswith(("_frac", "_norm", "_ratio")):
        return "ratio"
    return "count"


def _env_of(ctx):
    return {"hw_profile_sha": ctx.launch.get("hw_profile_sha"),
            "image_sha": (ctx.launch.get("image") or {}).get("sha256"),
            "capkit_sha": (ctx.launch.get("code") or {}).get("capkit_sha"),
            "baseline_sha": (ctx.launch.get("code") or {}).get("baseline_sha")}


def _lag_series(ctx, td):
    """{key: per-second series over the full steady window} for the X2 lag statistics."""
    out = {}
    from capcommon.schedule import BeatSchedule
    beat = BeatSchedule(ctx.tt)
    for role in td.roles_with("loadgen.requests"):
        rows, _ = td.rows("loadgen.requests", role)
        out["T1:req_first_lag_p50_ns:%s" % role] = request_lag_series(rows, td.tier,
                                                                      "first")[0]
        if td.tier.beat_ns > 0 and td.rows("loadgen.chunks", role)[0].get("req_id"):
            s, _n, why = msg_lag_from_chunks(td, ctx, role, beat)
            if why is None:
                out["T1:msg_lag_p50_ns:%s" % role] = s
    return out


def _inflight_from(rows, start, k):
    sent = sorted(s for s in rows.get("sent_rel_ns", []) if s >= 0)
    done = sorted(d for s, d in zip(rows.get("sent_rel_ns", []), rows.get("done_rel_ns", []))
                  if s >= 0 and d >= 0)
    return [bisect.bisect_right(sent, start + i * NS + NS // 2)
            - bisect.bisect_right(done, start + i * NS + NS // 2) for i in range(k)]


def select_tiers(tree, seqs, W, drop_verdicts=("VOID",)):
    """Verify one judged tree and choose which of the listed tiers enter a calibration file
    (shared by the pilot and capjudge.calib). Returns {"ctx", "rec", "by_seq", "verdicts",
    "vpath"}; rec = {"tree", "judge_verdicts_sha256", "tiers_used", "tiers_excluded"} plus
    "tree_excluded" when the launch-level conclusion recorded before the judge is not VALID.
    A listed tier with any RED check or a verdict in `drop_verdicts` is excluded and listed."""
    tree = os.path.abspath(tree)
    vpath = os.path.join(tree, "judge", "verdicts.json")
    if not os.path.isfile(vpath):
        raise PilotError("%s has not been judged (no judge/verdicts.json)" % tree)
    try:
        ctx = I.verify(W, tree)
    except I.InputInvalid as e:
        raise PilotError("%s: input invalid: %s" % (tree, e))
    verdicts = read_json(vpath)
    by_seq = {t["tier_seq"]: t for t in verdicts["tiers"]}
    rec = {"tree": os.path.relpath(tree, W), "judge_verdicts_sha256": sha256_file(vpath),
           "tiers_used": [], "tiers_excluded": {}}
    out = {"ctx": ctx, "rec": rec, "by_seq": by_seq, "verdicts": verdicts, "vpath": vpath}
    launch_state = (verdicts.get("launch_checks") or {}).get("LAUNCH-STATE")
    if launch_state is None:
        raise PilotError("%s: verdicts.json has no LAUNCH-STATE check (judged by an older "
                         "capjudge): cannot tell the launch-level conclusion" % tree)
    if launch_state != "PASS" or ctx.upstream["status"] != "VALID":
        rec["tree_excluded"] = "launch-level conclusion before the judge: %s (judged %s)" % (
            "; ".join("%s %s: %s" % (x["source"], x["status"], x["reason"])
                      for x in ctx.upstream["sources"] if x["status"] != "VALID")
            or ctx.upstream["status"], launch_state)
        rec["tiers_excluded"] = {str(q): "tree excluded" for q in seqs}
        return out
    for seq in seqs:
        t = by_seq.get(seq)
        if t is None or not t.get("judged"):
            raise PilotError("%s: tier %d is not a judged load tier" % (tree, seq))
        bad = sorted(c for c, st in (t.get("checks") or {}).items() if st == "RED")
        if bad or t["verdict"] in drop_verdicts:
            rec["tiers_excluded"][str(seq)] = "%s / RED checks %s" % (t["verdict"], bad)
            continue
        rec["tiers_used"].append(seq)
    return out


def build(list_path, W, d14_rule_path):
    W = os.path.abspath(W)
    entries = read_list(list_path)
    try:
        d14_rule, d14_rule_rec = D14.load_rule(d14_rule_path, W)
    except D14.D14RuleError as e:
        raise PilotError(str(e))
    values, sources, inputs, env, experiment = {}, {}, [], None, None
    used = []  # (ctx, td) of every included tier
    drain_waits, switch = [], set()
    d14_trees, d14_submitted = [], {}
    for tree, seqs in entries:
        # v1 builder: a CONTRA tier stays out as it did when a contradiction voided the tier
        sel = select_tiers(tree, seqs, W, drop_verdicts=("VOID", "CONTRA"))
        ctx, rec, by_seq = sel["ctx"], sel["rec"], sel["by_seq"]
        e = _env_of(ctx)
        if env is None:
            env, experiment = e, ctx.ident.experiment
        elif e != env:
            raise PilotError("%s: environment %s differs from %s (one pilot per environment)"
                             % (tree, e, env))
        if "tree_excluded" in rec:
            inputs.append(rec)
            continue
        for seq in rec["tiers_used"]:
            t = by_seq[seq]
            for key, vals in (t.get("stats") or {}).items():
                for v in vals:
                    values.setdefault(key, []).append(v)
                    sources.setdefault(key, []).append({"tree": rec["tree"], "tier_seq": seq})
            used.append((ctx, TierData(ctx, ctx.tt.tiers[seq])))
        inputs.append(rec)
        ld = LaunchData(ctx)
        for t in ctx.tt.tiers:
            if t.kind == "drain":
                dg = TierData(ctx, t).drain_gate or {}
                if dg.get("passed") and isinstance(dg.get("t_pass_rel_ns"), int):
                    drain_waits.append(dg["t_pass_rel_ns"] - t.start_ns)
        recs, _errs = ld.version_records()
        switch |= {r.get("switchinterval_s") for r in recs.values()}
        d14_trees.append((rec["tree"], D14.tree_readings(ctx, ld, role_cpus(ld.pin_table),
                                                          rec["tiers_used"])))
        d14_submitted[rec["tree"]] = (ctx.launch.get("submission") or {}).get("submitted_at")
    if not used:
        raise PilotError("no tier survived; nothing to build a noise band from")
    bands = {}
    for key, vals in sorted(values.items()):
        b = stats.t3_band(vals)
        if b is not None:
            bands[key] = dict(b, regime="subsat", unit=_unit(key), source_tiers=sources[key])
    search = search_by_arm(used, bands)
    return {"schema": SCHEMA, "experiment": experiment, "environment": env, "inputs": inputs,
            "noise_bands": bands, **search,
            "drain_max_wait_ns": max(drain_waits) if drain_waits else None,
            "d13_switchinterval_s": {"values": sorted(v for v in switch if v is not None)}
            if switch else None,
            "d14_softirq": D14.conclude(d14_rule, d14_rule_rec, d14_trees, d14_submitted),
            "tier_resolution": None, "launches_per_arm": None, "min_detectable_diff": None,
            "i2_calibration": {"enters_criteria": False, "per_loop_impl": {}},
            "core_budget": None, "canary_band": None,
            "not_computed": {k: NOT_COMPUTED for k in (
                "tier_resolution", "launches_per_arm", "min_detectable_diff",
                "i2_calibration", "core_budget", "canary_band")}}


def search_by_arm(used, bands):
    """Steady-window and warm-up search done separately for each arm (the launch identity's
    `arm`), so one arm's series are never cut to another arm's shorter window. The top-level
    window_ns / warmup_ns are the largest per-arm value when every arm has one, else None."""
    by_arm = {}
    for ctx, td in used:
        by_arm.setdefault(ctx.ident.arm, []).append((ctx, td))
    out = {"window_ns_by_arm": {}, "warmup_ns_by_arm": {}, "window_search_by_arm": {},
           "warmup_search_by_arm": {}}
    for arm in sorted(by_arm):
        win, win_table = _window_length(by_arm[arm], bands)
        warm, warm_table = _warmup_length(by_arm[arm], bands, win)
        out["window_ns_by_arm"][arm] = win
        out["window_search_by_arm"][arm] = win_table
        out["warmup_ns_by_arm"][arm] = warm
        out["warmup_search_by_arm"][arm] = warm_table
    for key in ("window_ns", "warmup_ns"):
        vals = list(out[key + "_by_arm"].values())
        out[key] = max(vals) if vals and None not in vals else None
    return out


def _window_length(used, bands):
    """Shortest whole-second steady window L (even, >= 2 s) for which the half-split difference
    (T1) of every lag statistic of every given tier (one arm), computed on the first L seconds,
    lies in that statistic's full-window noise band.

    The series must all have the same length (the arm's full window); otherwise there is no
    common L and the result is None with `unequal_series_lengths` in the table. Reaching zero
    only at L = the full window is no finding (each full-window value is itself a band sample):
    None, with `saturated_at_full_window` in the table."""
    series = [(key, s) for ctx, td in used for key, s in _lag_series(ctx, td).items()
              if key in bands]
    if not series:
        return None, {}
    lengths = sorted({len(s) for _k, s in series})
    if len(lengths) != 1:
        return None, {"unequal_series_lengths": lengths}
    full = lengths[0]
    table = {}
    for L in range(2, full + 1, 2):
        outside = 0
        for key, s in series:
            res = stats.t1_trend(s[:L])
            if res is None or stats.compare_to_band(res[0], bands[key]) != "inside":
                outside += 1
        table[str(L)] = {"n_series": len(series), "n_outside": outside}
        if outside == 0:
            if L == full:
                table["saturated_at_full_window"] = True
                return None, table
            return L * NS, table
    return None, table


def _warmup_length(used, bands, window_ns):
    """Smallest whole-second offset w from the tier start after which the derived in-flight
    count of every given tier (one arm) shows no trend (T1 over [start + w, start + w + window)
    inside its band), window being that arm's steady window."""
    if window_ns is None:
        return None, {}
    k = window_ns // NS
    table = {}
    max_w = min((td.tier.end_ns - td.tier.start_ns) // NS - k for _c, td in used)
    for w in range(0, max_w + 1):
        outside, n = 0, 0
        for _ctx, td in used:
            for role in td.roles_with("loadgen.requests"):
                key = "T1:inflight_derived:%s" % role
                if key not in bands:
                    continue
                n += 1
                rows, _ = td.rows("loadgen.requests", role)
                res = stats.t1_trend(_inflight_from(rows, td.tier.start_ns + w * NS, k))
                if res is None or stats.compare_to_band(res[0], bands[key]) != "inside":
                    outside += 1
        table[str(w)] = {"n_series": n, "n_outside": outside}
        if n and outside == 0:
            return w * NS, table
    return None, table


def main(argv=None):
    ap = argparse.ArgumentParser(prog="capjudge.pilot", description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--d14-rule", required=True,
                    help="capkit.d14_rule/1 file registered in <W>/REGISTRY.tsv (kind=d14_rule)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        sys.stderr.write("capjudge.pilot: refusing to overwrite %s\n" % a.out)
        return 3
    try:
        doc = build(a.inputs, a.workdir, a.d14_rule)
    except (PilotError, ConfigError, OSError, ValueError) as e:
        sys.stderr.write("capjudge.pilot: %s\n" % e)
        return 3
    atomic_write_bytes(a.out, (dumps(doc) + "\n").encode("utf-8"))
    d14 = doc["d14_softirq"]
    print("wrote %s: %d noise bands from %d tiers; window_ns=%s warmup_ns=%s; d14=%s (%s)" % (
        a.out, len(doc["noise_bands"]), sum(len(i["tiers_used"]) for i in doc["inputs"]),
        doc["window_ns"], doc["warmup_ns"], d14.get("irq_softirq_is_experiment"),
        d14["conclusion_reason"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

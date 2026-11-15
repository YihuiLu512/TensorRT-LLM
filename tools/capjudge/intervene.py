# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Intervention test V (design 2.6; INTERFACES 7.3; errata B2-2E).

Two parts:
  * inside one launch (the judge, mode "P+X+I+V"): the heartbeat of every busy / sleep
    injection the tier table declares for a load tier, from the tools' per-tier `inject`
    reports (INTERFACES 5.4.1): the observed spec equals the declared one and the activation
    did not change, in the load tier and in the drain tier right after it; and, per worker
    over those two tiers (the window of the reconcile check C5-C7: the passes of the tier's
    last requests fall in the drain tier, and an aux thread applies passes after the tier
    that counted them), applied == expected_applied, the injected time is at least applied x
    delta_ns, nothing is pending on an aux thread at the end of the drain tier, and the
    number of passes lies in the reconcile check's expected count of that role's hop
    (checks.hop_range; per-request sites: the requests chain, per-message sites: the
    messages / chunks chain). No following drain tier, its gate not passed, or its summaries
    not reporting the site -> those counts are NOT_RUN. Plus the launch's dose: the busy /
    sleep specs per role, identical on every load tier, else "mixed" (unusable for V).
  * across launches (`python3 -m capjudge.intervene`): the dose-response. A launch whose
    launch-level conclusion is not VALID now is excluded first (capjudge.boundary.upstream_now,
    as capjudge.boundary). Hop counts next: a dosed launch is used only when every judged load
    tier's heartbeat is PASS. One topology only: the control launches (no dose) must share
    their roles (name, kind, measure_class, workers; capjudge.boundary.topology), and a dosed
    launch of another topology is excluded. The boundary of a launch is the interval
    capjudge.boundary uses across launches (T4 below the first tool-limited tier, INTERFACES
    14.1); the envelope per delta spans those. Forward direction: delta only on the role the
    control launches localised, at least two delta > 0 besides delta = 0; the boundary must
    move to less load at every step and every step must exceed the noise band (T5
    "different", in that direction). Reverse: the same kind of delta on a load source or an
    out-of-domain stub must leave the boundary where it is (T5 "indistinguishable" from
    delta = 0). A reverse launch is evidence only when it stayed healthy (no VOID or CONTRA
    tier, no tool limit cut its interval) and both envelopes are closed (else excluded); such a launch T5
    tells apart from the control is a FAIL. An indistinguishable one counts only when the
    comparison could have seen the move: its min_resolvable_diff is below the forward shift of
    the same delta (distance of the envelope midpoints of that forward level and the control,
    which T5 must tell apart); otherwise it is excluded with both numbers. The boundary at
    delta = 0 is the result; the intercept of the dose-response is never used.

    python3 -m capjudge.intervene --inputs <list> --workdir <W> --out <new json>

The list names one judged launch tree per line (literal paths, no globs; the format of
capjudge.boundary). The output and its recompute record (`<name>.cmd.txt`, capjudge.cmdrec) are
never overwritten; no launch left after the exclusions -> exit 3. Output: schema
capkit.intervention/1 with every launch used or excluded (and why), the forward and reverse
comparisons, V's status (PASS / FAIL / INCONCLUSIVE / NOT_RUN) and the final grade
(capjudge.grade.final_grade over the control launches).
"""

import argparse
import os
import sys

from capcommon.exitcodes import ConfigError
from capcommon.hashing import sha256_file
from capcommon.jsonio import atomic_write_bytes, dumps

from . import boundary as B
from . import checks as C
from . import cmdrec
from . import grade as G
from . import stats

SCHEMA = "capkit.intervention/1"
DOSE_MODES = ("busy", "sleep")
PASS, FAIL, NOT_RUN, NONE = "PASS", "FAIL", "NOT_RUN", "NONE"
INCONCLUSIVE = "INCONCLUSIVE"
# Unit of one pass of each capkit tool site (the tools' slot declarations): the reconcile chain
# kind whose hop of the injected role counts those passes. beat.step (per step) and product
# probe sites have no such hop here: their pass count is NOT_RUN.
SITE_UNIT = {"send.req": "requests", "short.req": "requests", "recv.chunk": "chunks",
             "send.chunk": "chunks", "send.msg": "messages", "recv.msg": "messages",
             "relay.msg": "messages"}
SPEC_KEYS = ("site", "mode", "delta_ns", "every_n", "phase", "where")
# Counter keys of a tool's per-tier `inject` report the heartbeat reads (the load tier's; the
# following drain tier's also aux_pending_at_end, the end of the counting window); a missing one
# is NOT_RUN
LOAD_KEYS = ("hits_delta", "applied_delta", "applied_ns_delta")
COUNT_KEYS = LOAD_KEYS + ("aux_pending_at_end",)


class InterveneError(ConfigError):
    pass


def spec_sig(spec):
    d = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
    return {k: d.get(k, {"every_n": 1, "phase": 0, "where": "inline"}.get(k)) for k in SPEC_KEYS}


def declared_doses(tt, role, seq):
    return sorted((spec_sig(s) for s in tt.inject_for(role, seq) if s.mode in DOSE_MODES),
                  key=lambda d: d["site"])


def launch_doses(ctx):
    """{role: [spec sig]} of the busy / sleep injections, the same on every load tier; or
    "mixed" when some role's set differs between load tiers."""
    tt = ctx.tt
    out = {}
    for role in sorted(ctx.roles):
        sets = [declared_doses(tt, role, t.seq) for t in tt.tiers if t.kind in ("up", "down")]
        if not sets:
            continue
        if any(s != sets[0] for s in sets):
            return "mixed"
        if sets[0]:
            out[role] = sets[0]
    return out


def _hop_range(ctx, td, role, unit):
    """(lo, hi) expected passes of `role` for one tier: checks.hop_range (the reconcile check's
    own expectation) of the role's tier-scope hop on the first chain of kind `unit`; None when
    no chain has one; raises ConfigError as checks.hop_range."""
    for chain in ctx.launch["reconcile"]["chains"]:
        if chain["kind"] != unit:
            continue
        if any(h["role"] == role and h["scope"] == "tier" for h in chain["hops"]):
            return C.hop_range(ctx, td, chain, role)
    return None


def _observed(td, role):
    """{site: [(worker, inject entry)]} of the busy / sleep entries in the role's summaries of
    one tier."""
    out = {}
    docs = (td.summaries.get(role) or {}) if td is not None else {}
    for w, doc in sorted(docs.items(), key=lambda kv: str(kv[0])):
        for e in doc.get("inject") or []:
            sp = e.get("spec")
            if sp and sp.get("mode") in DOSE_MODES:
                out.setdefault(e["site"], []).append((w, e))
    return out


def tier_heartbeat(ctx, td, next_drain=None):
    """V heartbeat of one load tier: {status PASS/FAIL/NOT_RUN/NONE, injections: [...]}.
    `next_drain`: the drain tier right after it (the passes of the tier's last requests are
    counted there, by event time), as the reconcile check gets it."""
    tt = ctx.tt
    items = []
    for role in sorted(ctx.roles):
        declared = {d["site"]: d for d in declared_doses(tt, role, td.tier.seq)}
        observed = _observed(td, role)
        tail = _observed(next_drain, role)
        for site in sorted(set(declared) | set(observed)):
            items.append(_one_site(ctx, td, next_drain, role, site, declared.get(site),
                                   observed.get(site, []), tail.get(site, [])))
    if not items:
        return {"status": NONE, "injections": []}
    sts = [i["status"] for i in items]
    st = FAIL if FAIL in sts else (NOT_RUN if NOT_RUN in sts else PASS)
    return {"status": st, "injections": items}


def _one_site(ctx, td, next_drain, role, site, decl, obs, tail):
    """Heartbeat of one (role, site). Every count is read over the load tier plus the following
    drain tier, per worker (the window of the reconcile check, checks._hop_parts: the passes
    of the tier's last requests fall in the drain tier, and an aux thread applies passes
    after they were counted): passes, applied == expected_applied, injected time >= applied x
    delta_ns, and nothing pending on an aux thread at the end of the window. No data (a
    worker's summary missing in either tier, a counter key missing, no following drain tier or
    its gate not passed, the drain tier not reporting the site) is NOT_RUN; a load-tier
    summary that is there but does not report a declared site, a wrong spec in either tier,
    or a wrong count over the window is FAIL."""
    it = {"role": role, "site": site, "declared": decl, "status": PASS, "why": []}

    def bad(st, why):
        it["why"].append(why)
        if st == FAIL or it["status"] != FAIL:
            it["status"] = st
    if decl is None:
        bad(FAIL, "injection observed but not declared in the tier table")
        return it
    workers = ctx.roles[role]["workers"]
    have = {str(w) for w in (td.summaries.get(role) or {})}
    missing = [w for w in range(workers) if str(w) not in have]
    if missing:
        bad(NOT_RUN, "no summary of worker(s) %s in this tier (not run to the end?)" % missing)
    reported = {str(w) for w, _e in obs}
    silent = sorted(have - reported)
    if silent:
        bad(FAIL, "summaries of worker(s) %s do not report the site" % silent)
    load = {}
    for w, e in obs:
        if _entry_ok(bad, decl, e, LOAD_KEYS, "w%s" % w):
            load[str(w)] = e
    complete = not missing and len(load) == len(obs)
    hits = sum(e["hits_delta"] for e in load.values())
    applied = sum(e["applied_delta"] for e in load.values())
    drain, why = _drain_entries(bad, next_drain, workers, decl, tail)
    hd = None if drain is None else sum(e["hits_delta"] for e in drain.values())
    it.update(hits_tier=hits, hits_drain=hd, hits=None if hd is None else hits + hd)
    if complete and applied + (0 if drain is None else sum(
            e["applied_delta"] for e in drain.values())) <= 0:
        bad(FAIL, "no pass was injected (site not on the path?)")
    if why:
        bad(NOT_RUN, "counts over the tier and its drain tier: %s" % why)
    if not complete or drain is None:
        it["hits_expected"] = None
        return it
    for w in sorted(load):
        _window_counts(bad, decl, w, load[w], drain[w], next_drain.tier.seq)
    unit = SITE_UNIT.get(site)
    try:
        rng = _hop_range(ctx, td, role, unit) if unit else None
    except ConfigError as e:
        it["hits_expected"] = None
        bad(NOT_RUN, "expected passes of %s/%s: %s" % (role, site, e))
        return it
    it["hits_expected"] = list(rng) if rng else None
    if rng is None:
        bad(NOT_RUN, "no reconcile hop counts the passes of %s/%s" % (role, site))
    elif not rng[0] <= it["hits"] <= rng[1]:
        bad(FAIL, "passes %d (tier %d + drain %d) outside the expected %s"
            % (it["hits"], hits, hd, list(rng)))
    return it


def _entry_ok(bad, decl, e, keys, who):
    """One summary entry: the declared spec (else FAIL), an unchanged activation and every
    counter in `keys` plus expected_applied_delta present (else NOT_RUN). Returns whether its
    counts can be used."""
    if spec_sig(e["spec"]) != decl:
        bad(FAIL, "%s spec %s != declared" % (who, spec_sig(e["spec"])))
    if not e.get("activation_unchanged"):
        bad(NOT_RUN, "%s activation changed inside the tier" % who)
    absent = [k for k in keys + ("expected_applied_delta",) if not isinstance(e.get(k), int)]
    if absent:
        bad(NOT_RUN, "%s reports no %s" % (who, absent))
        return False
    return True


def _drain_entries(bad, next_drain, workers, decl, tail):
    """({worker: entry} of the site in the following drain tier, None), or (None, why)."""
    if next_drain is None or next_drain.drain_gate is None:
        return None, "no following drain tier with a gate record"
    if not next_drain.drain_gate.get("passed"):
        return None, "the following drain gate did not pass"
    have = {str(w) for w, _e in tail}
    missing = [w for w in range(workers) if str(w) not in have]
    if missing:
        return None, ("worker(s) %s have no summary reporting the site in drain tier %d"
                      % (missing, next_drain.tier.seq))
    out = {}
    for w, e in tail:
        if _entry_ok(bad, decl, e, COUNT_KEYS, "drain tier %d w%s" % (next_drain.tier.seq, w)):
            out[str(w)] = e
    if len(out) != len(tail):
        return None, "a drain-tier entry cannot be used"
    return out, None


def _window_counts(bad, decl, w, a, b, drain_seq):
    """One worker's counts over the load tier (`a`) and the following drain tier (`b`)."""
    ap = a["applied_delta"] + b["applied_delta"]
    ex = a["expected_applied_delta"] + b["expected_applied_delta"]
    ns = a["applied_ns_delta"] + b["applied_ns_delta"]
    if ap != ex:
        bad(FAIL, "w%s applied %d != expected %d (tier + drain tier %d)" % (w, ap, ex, drain_seq))
    if ns < ap * decl["delta_ns"]:
        bad(FAIL, "w%s injected %d ns < applied %d x delta %d (tier + drain tier %d)"
            % (w, ns, ap, decl["delta_ns"], drain_seq))
    if b["aux_pending_at_end"] != 0:
        bad(FAIL, "w%s aux_pending_at_end %s at the end of drain tier %d"
            % (w, b["aux_pending_at_end"], drain_seq))


def launch_heartbeat(tier_entries):
    """Launch-level heartbeat over the judged load tiers' X.V entries."""
    sts = [((e.get("X") or {}).get("V") or {}).get("status") for e in tier_entries
           if e.get("judged")]
    if not sts or all(s == NONE for s in sts):
        return NONE
    if FAIL in sts:
        return FAIL
    if any(s != PASS for s in sts):
        return NOT_RUN
    return PASS


# ---- across launches -------------------------------------------------------------------------

def load_launch(tree, W):
    """The judged launch record of one tree (judge/verdicts.json, committed: its JUDGED.tsv row
    carries the same sha), with the interval capjudge.boundary uses across launches: T4 below
    the first tool-limited tier (`boundary`, `censored_by` set when cut there) next to the
    judge's T4 as recorded (`boundary_t4`), and its roles (raw/launch.json; the topology).
    A launch whose launch-level conclusion is not VALID now (capjudge.boundary.upstream_now: a
    STATE line appended after the judgement counts) carries `excluded_why` and nothing else is
    read. Raises InterveneError."""
    try:
        v, _vp, sha = B.load_judged(tree, W)
        st, why = B.upstream_now(tree, v)
        if st != "VALID":
            return {"tree": tree, "verdicts_sha256": sha,
                    "excluded_why": "launch-level conclusion %s: %s" % (st, why)}
        if v.get("mode") != "P+X+I+V":
            raise InterveneError("%s was judged in mode %r, not P+X+I+V" % (tree, v.get("mode")))
        t4, used = B.launch_interval(v)
        roles = B._roles(tree)
    except B.BoundaryError as e:
        raise InterveneError(str(e))
    ident = v.get("identity") or {}
    iv = v.get("intervention_V") or {}
    return {"tree": tree, "verdicts_sha256": sha, "identity": ident, "axis": v.get("axis"),
            "boundary": used, "boundary_t4": t4, "grade": v.get("grade") or {},
            "doses": iv.get("doses"), "heartbeat": iv.get("heartbeat"),
            "measure_class": iv.get("measure_class") or {}, "roles": roles,
            "void_tiers": _tiers_of(v, "VOID"), "contra_tiers": _tiers_of(v, "CONTRA")}


def _tiers_of(v, verdict):
    return [t["tier_seq"] for t in v.get("tiers", []) if t.get("judged")
            and t.get("verdict") == verdict]


def _scalar_dose(specs):
    """Added time per pass: sum of delta_ns / every_n over the role's specs."""
    return sum(s["delta_ns"] / s["every_n"] for s in specs)


def _env(launches, order):
    sign = 1 if order == "asc" else -1
    ivs = [(sign * x["boundary"]["lower"] if x["boundary"].get("lower") is not None else None,
            sign * x["boundary"]["upper"] if x["boundary"].get("upper") is not None else None)
           for x in launches]
    return stats.t4_envelope(ivs, "asc")


def _reverse_unusable(x, base_env, env):
    """Why a reverse-dose launch cannot show "the boundary did not move", or None. Usable only
    when no tier is VOID or CONTRA (a CONTRA tier passed the deterministic checks, but X2
    contradicts X1 / X3 there: the judge voided such a tier before the CONTRA verdict existed),
    no tool limit cut its interval (the dosed tool or another one reached its limit first: the
    interval is not the boundary of the localised role) and both envelopes are closed (an open
    end cannot be shown to overlap or not)."""
    if x["void_tiers"]:
        return "VOID tiers %s (a tool left its healthy band?)" % x["void_tiers"]
    if x.get("contra_tiers"):
        return "CONTRA tiers %s (X2 contradicts X1 / X3; the deterministic checks passed)" \
            % x["contra_tiers"]
    cut = x["boundary"].get("censored_by")
    if cut:
        return "interval cut at tier %s (%s)" % (cut.get("tier_seq"), cut.get("why"))
    for name, e in (("control", base_env), ("reverse", env)):
        if None in (e.get("lower"), e.get("upper")):
            return "the %s envelope has an open end" % name
    return None


def _mid(env):
    return (env["lower"] + env["upper"]) / 2


def _forward_shifts(levels, base_env, order):
    """{delta: (shift, None) or (None, why)}: per forward dose level, the distance between the
    midpoints of its envelope and the control's when T5 tells them apart (the size of the
    move a reverse dose of the same delta must be able to resolve)."""
    out = {}
    for d, ls in levels.items():
        env = _env(ls, order)
        cmp_ = stats.t5_compare([{"block": "all", "a": base_env, "b": env}])
        if cmp_["result"] == "different":
            out[d] = (abs(_mid(env) - _mid(base_env)), None)
        else:
            out[d] = (None, "the forward level of this delta is T5 %s against the control"
                      % cmp_["result"])
    return out


def _reverse(x, role, base_env, order, shifts):
    """(reverse record, None) or (None, excluded record) of one reverse-dose launch."""
    env = _env([x], order)
    why = _reverse_unusable(x, base_env, env)
    if why:
        return None, {"tree": x["tree"], "why": "reverse dose: " + why}
    cmp_ = stats.t5_compare([{"block": "all", "a": base_env, "b": env}])
    if cmp_["result"] not in ("different", "indistinguishable"):
        return None, {"tree": x["tree"], "why": "reverse dose: T5 %s" % cmp_["result"]}
    rec = {"tree": x["tree"], "role": role, "t5": cmp_, "moved": cmp_["result"] == "different"}
    if rec["moved"]:
        return rec, None
    d = _scalar_dose(x["doses"][role])
    shift, why = shifts.get(d, (None, "no forward level of this delta"))
    res = cmp_["min_resolvable_diff"]
    rec.update(delta_ns=d, forward_shift=shift, min_resolvable_diff=res)
    if shift is None:
        return None, {"tree": x["tree"], "delta_ns": d, "min_resolvable_diff": res,
                      "why": "reverse dose: no forward shift of this delta to compare (%s)"
                      % why}
    if not res < shift:
        return None, {"tree": x["tree"], "delta_ns": d, "min_resolvable_diff": res,
                      "forward_shift": shift,
                      "why": "reverse dose: cannot resolve a shift of the forward size "
                      "(min_resolvable_diff %s >= forward shift %s)" % (res, shift)}
    return rec, None


def dose_response(launches):
    """V across launches (module docstring). Returns the V document."""
    out = {"status": NOT_RUN, "used": [], "excluded": [], "forward": [], "reverse": [],
           "reasons": []}
    if not launches:
        out["reasons"].append("no launch")
        return out
    axes = {repr(x["axis"]) for x in launches}
    exps = {x["identity"].get("experiment") for x in launches}
    if len(axes) != 1 or len(exps) != 1:
        raise InterveneError("launches of different experiments %s or axes %s"
                             % (sorted(map(str, exps)), sorted(axes)))
    order = launches[0]["axis"]["order"]
    control, dosed = [], []
    for x in launches:
        if x["doses"] == "mixed":
            out["excluded"].append({"tree": x["tree"], "why": "dose differs between tiers"})
        elif not x["doses"]:
            control.append(x)
        elif x["heartbeat"] != PASS:
            out["excluded"].append({"tree": x["tree"], "why": "heartbeat %s (hop counts "
                                    "first)" % x["heartbeat"]})
        elif len(x["doses"]) != 1:
            out["excluded"].append({"tree": x["tree"], "why": "doses on several roles %s"
                                    % sorted(x["doses"])})
        else:
            dosed.append(x)
    out["control"] = [x["tree"] for x in control]
    if not control:
        out["reasons"].append("no launch without a dose (delta = 0)")
        return out
    topos = sorted({B.topology(x["roles"]) for x in control})
    out["control_topologies"] = len(topos)
    if len(topos) != 1:
        out["reasons"].append("control launches have %d topologies (roles, kind, "
                              "measure_class, workers), one is needed: %s"
                              % (len(topos), [list(t) for t in topos]))
        return out
    topo = topos[0]
    loc = {(x["grade"].get("role"), x["grade"].get("grade")) for x in control}
    roles = {r for r, g in loc if g == "LOCALIZED"}
    if len(roles) != 1 or any(g != "LOCALIZED" for _r, g in loc):
        out["reasons"].append("control launches are not all localised to one role: %s"
                              % sorted(map(str, loc)))
        return out
    target = roles.pop()
    out["role"] = target
    base_env = _env(control, order)
    levels, rev = {}, []
    for x in dosed:
        role = next(iter(x["doses"]))
        cls = x["measure_class"].get(role)
        if B.topology(x["roles"]) != topo:
            out["excluded"].append({"tree": x["tree"], "why": "topology differs from the "
                                    "control launches' (roles, kind, measure_class, workers)"})
        elif role == target:
            levels.setdefault(_scalar_dose(x["doses"][role]), []).append(x)
        elif cls in ("load_source", "stub_out_of_domain"):
            rev.append((x, role))
        else:
            out["excluded"].append({"tree": x["tree"], "why": "dose on %s (%s) is neither the "
                                    "localised role nor a tool" % (role, cls)})
    shifts = _forward_shifts(levels, base_env, order)
    for x, role in rev:
        rec, excl = _reverse(x, role, base_env, order, shifts)
        if rec:
            out["reverse"].append(rec)
        else:
            out["excluded"].append(excl)
    steps = [(0.0, control)] + sorted(levels.items())
    prev_d, prev_l = steps[0]
    for d, ls in steps[1:]:
        cmp_ = stats.t5_compare([{"block": "all", "a": _env(prev_l, order),
                                  "b": _env(ls, order)}])
        ok = cmp_["result"] == "different" and cmp_["direction"] == "a_higher"
        wrong = cmp_["result"] == "different" and not ok
        out["forward"].append({"from_delta_ns": prev_d, "to_delta_ns": d, "t5": cmp_,
                               "moved_down": ok, "moved_up": wrong})
        prev_d, prev_l = d, ls
    used = {r["tree"] for r in out["reverse"]}
    out["used"] = [x["tree"] for x in control + [y for ls in levels.values() for y in ls]] \
        + [x["tree"] for x, _r in rev if x["tree"] in used]
    n_pos = len(levels)
    if any(s["moved_up"] for s in out["forward"]) or any(r["moved"] for r in out["reverse"]):
        out["status"] = FAIL
        out["reasons"].append("the boundary moved up with delta, or moved under a reverse dose")
    elif n_pos < 2:
        out["reasons"].append("%d delta > 0 on %s (need at least 2)" % (n_pos, target))
    elif not out["reverse"]:
        out["reasons"].append("no usable reverse dose")
    elif all(s["moved_down"] for s in out["forward"]):
        out["status"] = PASS
    else:
        out["status"] = INCONCLUSIVE
        out["reasons"].append("a step did not move the boundary beyond the noise band")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="capjudge.intervene", description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    argv = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(argv)
    try:
        bad = cmdrec.refuse_existing(a.out)
        if bad:
            raise InterveneError(bad)
        try:
            trees = B.read_list(a.inputs)
        except B.BoundaryError as e:
            raise InterveneError(str(e))
        read = [load_launch(os.path.abspath(t), os.path.abspath(a.workdir)) for t in trees]
        gone = [{"tree": x["tree"], "why": x["excluded_why"]} for x in read
                if "excluded_why" in x]
        launches = [x for x in read if "excluded_why" not in x]
        if not launches:
            raise InterveneError("no launch left after the exclusions %s" % gone)
        v = dose_response(launches)
        v["excluded"] = gone + v["excluded"]
        doc = {"schema": SCHEMA, "inputs": [{"tree": x["tree"], "verdicts_sha256":
                                             x["verdicts_sha256"]} for x in read],
               "inputs_list_sha256": sha256_file(a.inputs), "V": v,
               "grade": G.final_grade([x["grade"] for x in launches
                                       if x["tree"] in v.get("control", [])], v)}
    except (InterveneError, ConfigError, OSError, ValueError, KeyError) as e:
        sys.stderr.write("capjudge.intervene: %s\n" % e)
        return 3
    atomic_write_bytes(a.out, (dumps(doc) + "\n").encode("utf-8"))
    cmdrec.write(a.out, "capjudge.intervene", argv,
                 [a.inputs] + [os.path.join(x["tree"], "judge", "verdicts.json")
                               for x in read])
    print("V %s; grade %s -> %s" % (v["status"], doc["grade"]["grade"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Intervention test V (design 2.6; INTERFACES 7.3; errata B2-2E).

Two parts:
  * inside one launch (the judge, mode "P+X+I+V"): the heartbeat of every busy / sleep
    injection the tier table declares for a load tier, from the tools' per-tier `inject`
    reports (INTERFACES 5.4.1): the observed spec equals the declared one, the activation did not
    change inside the tier, applied == expected_applied, nothing left pending on an aux thread,
    the injected time is at least applied x delta_ns, and the number of passes equals the
    reconcile chain's expected count of that role's hop (per-request sites: the requests
    chain, per-message sites: the messages / chunks chain). Plus the launch's dose: the busy /
    sleep specs per role, identical on every load tier, else "mixed" (unusable for V).
  * across launches (`python3 -m capjudge.intervene`): the dose-response. Hop counts first:
    a dosed launch is used only when every judged load tier's heartbeat is PASS. Then the
    forward direction: delta only on the role the control launches (no dose) localised, at
    least two delta > 0 besides delta = 0; the boundary (T4 interval per launch, envelope per
    delta) must move to less load at every step and every step must exceed the noise band (T5
    "different", in that direction). Reverse: the same kind of delta on a load source or an
    out-of-domain stub that stayed healthy (no VOID tier) must leave the boundary where it is
    (T5 "indistinguishable" from delta = 0). The boundary at delta = 0 is the result; the
    intercept of the dose-response is never used.

    python3 -m capjudge.intervene --inputs <list> --workdir <W> --out <new json>

The list names one judged launch tree per line (literal paths, no globs). Output: schema
capkit.intervention/1 with every launch used or excluded (and why), the forward and reverse
comparisons, V's status (PASS / FAIL / INCONCLUSIVE / NOT_RUN) and the final grade
(capjudge.grade.final_grade).
"""

import argparse
import os
import sys

from capcommon import expect
from capcommon.exitcodes import ConfigError
from capcommon.hashing import sha256_file
from capcommon.jsonio import atomic_write_bytes, dumps, read_json

from . import grade as G
from . import stats
from .symptoms import failures_by_class

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
    """[lo, hi] expected passes of `role` for one tier from the reconcile chain of `unit`."""
    t = td.tier
    n = int(ctx.schedule_meta["per_tier"].get(str(t.seq), 0))
    ovl = failures_by_class(td, ctx.roles)["overload"]
    for chain in ctx.launch["reconcile"]["chains"]:
        if chain["kind"] != unit:
            continue
        for hop in chain["hops"]:
            if hop["role"] == role and hop["scope"] == "tier":
                hi = expect.expected_hop_count(chain, n, t.max_tokens, t.tokens_per_msg,
                                               t.stream_interval, role)
                lo = expect.expected_hop_count(chain, n - ovl, t.max_tokens,
                                               t.tokens_per_msg, t.stream_interval, role)
                return lo, hi
    return None


def tier_heartbeat(ctx, td):
    """V heartbeat of one load tier: {status PASS/FAIL/NOT_RUN/NONE, injections: [...]}."""
    tt = ctx.tt
    items = []
    for role in sorted(ctx.roles):
        declared = {d["site"]: d for d in declared_doses(tt, role, td.tier.seq)}
        docs = td.summaries.get(role) or {}
        observed = {}
        for w, doc in sorted(docs.items(), key=lambda kv: str(kv[0])):
            for e in doc.get("inject") or []:
                sp = e.get("spec")
                if sp and sp.get("mode") in DOSE_MODES:
                    observed.setdefault(e["site"], []).append((w, e))
        for site in sorted(set(declared) | set(observed)):
            items.append(_one_site(ctx, td, role, site, declared.get(site),
                                   observed.get(site, [])))
    if not items:
        return {"status": NONE, "injections": []}
    sts = [i["status"] for i in items]
    st = FAIL if FAIL in sts else (NOT_RUN if NOT_RUN in sts else PASS)
    return {"status": st, "injections": items}


def _one_site(ctx, td, role, site, decl, obs):
    it = {"role": role, "site": site, "declared": decl, "status": PASS, "why": []}

    def bad(st, why):
        it["why"].append(why)
        if st == FAIL or it["status"] != FAIL:
            it["status"] = st
    if decl is None:
        bad(FAIL, "injection observed but not declared in the tier table")
        return it
    workers = ctx.roles[role]["workers"]
    if len(obs) != workers:
        bad(FAIL, "%d of %d worker summaries report the site" % (len(obs), workers))
    hits = applied = 0
    for w, e in obs:
        if spec_sig(e["spec"]) != decl:
            bad(FAIL, "w%s spec %s != declared" % (w, spec_sig(e["spec"])))
        if not e.get("activation_unchanged"):
            bad(NOT_RUN, "w%s activation changed inside the tier" % w)
        exp = e.get("expected_applied_delta")
        if not isinstance(exp, int):
            bad(NOT_RUN, "w%s expected_applied_delta is null" % w)
        elif e.get("applied_delta") != exp:
            bad(FAIL, "w%s applied %s != expected %s" % (w, e.get("applied_delta"), exp))
        if e.get("aux_pending_at_end", 0) != 0:
            bad(FAIL, "w%s aux_pending_at_end %s" % (w, e.get("aux_pending_at_end")))
        ad, ns = e.get("applied_delta") or 0, e.get("applied_ns_delta") or 0
        if ns < ad * decl["delta_ns"]:
            bad(FAIL, "w%s injected %d ns < applied %d x delta %d" % (w, ns, ad,
                                                                     decl["delta_ns"]))
        hits += e.get("hits_delta") or 0
        applied += ad
    it.update(hits=hits, applied=applied)
    if applied <= 0:
        bad(FAIL, "no pass was injected (site not on the path?)")
    unit = SITE_UNIT.get(site)
    rng = _hop_range(ctx, td, role, unit) if unit else None
    it["hits_expected"] = list(rng) if rng else None
    if rng is None:
        bad(NOT_RUN, "no reconcile hop counts the passes of %s/%s" % (role, site))
    elif not rng[0] <= hits <= rng[1]:
        bad(FAIL, "passes %d outside the expected %s" % (hits, list(rng)))
    return it


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

def read_list(path):
    out = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if any(c in line for c in "*?[") or "\t" in line:
                raise InterveneError("%s:%d: one literal tree path per line" % (path, n))
            out.append(line)
    if not out:
        raise InterveneError("%s lists no tree" % path)
    if len({os.path.abspath(t) for t in out}) != len(out):
        raise InterveneError("a tree is listed twice")
    return out


def load_launch(tree, W):
    """The judged launch record of one tree (judge/verdicts.json, committed: its JUDGED.tsv row
    carries the same sha). Raises InterveneError."""
    from capcommon import ledger  # noqa: PLC0415
    vp = os.path.join(tree, "judge", "verdicts.json")
    if not os.path.isfile(vp):
        raise InterveneError("%s is not judged (no judge/verdicts.json)" % tree)
    v = read_json(vp)
    ident = v.get("identity") or {}
    key = tuple(ident.get(k) for k in ("job_id", "restart_count", "launch_seq", "arm"))
    hit = ledger.judged_lookup(os.path.join(W, "JUDGED.tsv"), *key)
    sha = sha256_file(vp)
    if hit is None or hit["verdicts_sha256"] != sha:
        raise InterveneError("%s: JUDGED.tsv has no row with this verdicts.json sha" % tree)
    if v.get("mode") != "P+X+I+V":
        raise InterveneError("%s was judged in mode %r, not P+X+I+V" % (tree, v.get("mode")))
    iv = v.get("intervention_V") or {}
    return {"tree": tree, "verdicts_sha256": sha, "identity": ident, "axis": v.get("axis"),
            "boundary": v.get("boundary_T4") or {}, "grade": v.get("grade") or {},
            "doses": iv.get("doses"), "heartbeat": iv.get("heartbeat"),
            "measure_class": iv.get("measure_class") or {},
            "void_tiers": [t["tier_seq"] for t in v.get("tiers", []) if t.get("judged")
                           and t.get("verdict") == "VOID"]}


def _scalar_dose(specs):
    """Added time per pass: sum of delta_ns / every_n over the role's specs."""
    return sum(s["delta_ns"] / s["every_n"] for s in specs)


def _env(launches, order):
    sign = 1 if order == "asc" else -1
    ivs = [(sign * x["boundary"]["lower"] if x["boundary"].get("lower") is not None else None,
            sign * x["boundary"]["upper"] if x["boundary"].get("upper") is not None else None)
           for x in launches]
    return stats.t4_envelope(ivs, "asc")


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
    loc = {(x["grade"].get("role"), x["grade"].get("grade")) for x in control}
    roles = {r for r, g in loc if g == "LOCALIZED"}
    out["control"] = [x["tree"] for x in control]
    if not control:
        out["reasons"].append("no launch without a dose (delta = 0)")
        return out
    if len(roles) != 1 or any(g != "LOCALIZED" for _r, g in loc):
        out["reasons"].append("control launches are not all localised to one role: %s"
                              % sorted(map(str, loc)))
        return out
    target = roles.pop()
    out["role"] = target
    base_env = _env(control, order)
    levels = {}
    for x in dosed:
        role = next(iter(x["doses"]))
        cls = x["measure_class"].get(role)
        if role == target:
            levels.setdefault(_scalar_dose(x["doses"][role]), []).append(x)
        elif cls in ("load_source", "stub_out_of_domain"):
            if x["void_tiers"]:
                out["excluded"].append({"tree": x["tree"], "why": "reverse dose: VOID tiers %s "
                                        "(a tool left its healthy band?)" % x["void_tiers"]})
                continue
            cmp_ = stats.t5_compare([{"block": "all", "a": base_env, "b": _env([x], order)}])
            out["reverse"].append({"tree": x["tree"], "role": role, "t5": cmp_,
                                   "moved": cmp_["result"] == "different"})
        else:
            out["excluded"].append({"tree": x["tree"], "why": "dose on %s (%s) is neither the "
                                    "localised role nor a tool" % (role, cls)})
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
    out["used"] = [x["tree"] for x in control + dosed]
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
    a = ap.parse_args(argv)
    try:
        if os.path.lexists(a.out):
            raise InterveneError("%s exists (never overwritten)" % a.out)
        trees = read_list(a.inputs)
        launches = [load_launch(os.path.abspath(t), os.path.abspath(a.workdir))
                    for t in trees]
        v = dose_response(launches)
        doc = {"schema": SCHEMA, "inputs": [{"tree": x["tree"], "verdicts_sha256":
                                             x["verdicts_sha256"]} for x in launches],
               "inputs_list_sha256": sha256_file(a.inputs), "V": v,
               "grade": G.final_grade([x["grade"] for x in launches
                                       if x["tree"] in v.get("control", [])], v)}
    except (InterveneError, ConfigError, OSError, ValueError) as e:
        sys.stderr.write("capjudge.intervene: %s\n" % e)
        return 3
    atomic_write_bytes(a.out, (dumps(doc) + "\n").encode("utf-8"))
    print("V %s; grade %s -> %s" % (v["status"], doc["grade"]["grade"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

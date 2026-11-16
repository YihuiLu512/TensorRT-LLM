# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conclusion grade P + X + I + V (design 2.6 "结论等级"; DATA-AND-ANALYSIS 6 step 5; errata
B2-2E).

Grades:
  CONFIRMED      (已确认)        P + X + I localised to a measured role + V PASS on that role
  LOCALIZED      (已定位，未干预)  P + X + I localised to a measured role (under_test /
                                  stub_in_domain); V not PASS (not run, or inconclusive)
  LOWER_BOUND    (下界)          the load source or an out-of-domain stub left its healthy band
                                  first (C1 / C2 RED at the first non-stable tier, or I points
                                  at it), or X never held within the tool's range: the limit is
                                  >= the highest stable tier
  NOT_LOCALIZED  (已到顶，未定位)  P + X at the boundary, I did not localise: no signal held
                                  (`I_status` NOT_LOCALIZED), or some signal could not be
                                  evaluated (`I_status` NOT_RUN: not computed, which a reader
                                  must not take for "nothing reached its limit"). Not a design
                                  grade: the design lists no grade for P + X without I; it is
                                  reported, never upgraded
  UNDETERMINED   (原因未定)        I signals point to different roles at the boundary, or V FAIL
  NOT_RUN        (未运行)        no judged load tier passed P

One launch (the judge): the first tier in load order that is not STABLE decides, skipping
VOID / NOT_RUN tiers whose RED checks are not only C1 / C2, and CONTRA tiers (they widen the
interval, they are listed); `I_status` is I's status on that tier when it is UNSTABLE or
UNDETERMINED, else null. V is a cross-launch result, so the judge's grade never reaches CONFIRMED;
capjudge.intervene combines the control launches' grades with V (final_grade). Granularity is
the process (GIL domain), or the core group when the localised processes share one core (the
core is the reading; I's `granularity`); thread shares are reported, never a thread's limit.
"""

MEASURED = ("under_test", "stub_in_domain")
TOOLS = ("load_source", "stub_out_of_domain")
TOOL_CHECKS = ("C1", "C2")
# verdicts the walk passes over: VOID / NOT_RUN (unless only C1 / C2 are RED) and CONTRA (a
# counter contradiction, formerly a VOID without RED checks; calib-v2 DESIGN-r7a P2-06)
SKIPPED = ("VOID", "NOT_RUN", "CONTRA")


def _red_checks(entry):
    return sorted(c for c, s in (entry.get("checks") or {}).items() if s == "RED")


def _tool_void(entry):
    red = _red_checks(entry)
    return entry["verdict"] == "VOID" and bool(red) and set(red) <= set(TOOL_CHECKS)


def first_non_stable(tiers):
    """The walk launch_grade makes over verdicts.json `tiers`: the judged load tiers in load
    order; VOID / NOT_RUN tiers whose RED checks are not only C1 / C2, and CONTRA tiers, are
    passed over (listed in `skipped`). Returns {"judged", "first", "last_stable", "skipped"}:
    `first` is the first tier that decides the grade (UNSTABLE, UNDETERMINED, or VOID with
    only C1 / C2 RED; None when there is none) and `last_stable` the highest STABLE tier below
    it (the highest STABLE tier overall when `first` is None) -- the grade's interval lower
    end, which the shape builders (capjudge.calib) also take as a launch's healthy maximum."""
    judged = [e for e in tiers if e.get("judged") and e.get("order_key") is not None]
    judged.sort(key=lambda e: (e["order_key"], e["tier_seq"]))
    last_stable, skipped = None, []
    for e in judged:
        v = e["verdict"]
        if v == "STABLE":
            last_stable = e
            continue
        if v in SKIPPED and not _tool_void(e):
            skipped.append({"tier_seq": e["tier_seq"], "verdict": v, "red": _red_checks(e)})
            continue
        return {"judged": judged, "first": e, "last_stable": last_stable, "skipped": skipped}
    return {"judged": judged, "first": None, "last_stable": last_stable, "skipped": skipped}


def launch_grade(tiers, boundary):
    """Grade of one judged launch from verdicts.json `tiers` (load tiers carry verdict, checks,
    order_key, X with X.I) and its T4 boundary (a non-monotone boundary is noted)."""
    out = {"grade": "NOT_RUN", "reasons": [], "skipped_tiers": [], "V": "NOT_RUN: across "
           "launches, capjudge.intervene", "granularity": "process", "I_status": None}
    if (boundary or {}).get("non_monotone"):
        out["reasons"].append("T4: a stable tier lies above the lowest unstable or "
                              "undetermined tier")
    w = first_non_stable(tiers)
    judged = w["judged"]
    if not any(e["verdict"] in ("STABLE", "UNSTABLE", "UNDETERMINED") for e in judged):
        out["reasons"].append("no load tier passed P")
        return out
    last_stable, e = w["last_stable"], w["first"]
    out["skipped_tiers"] = w["skipped"]
    if e is not None:
        v = e["verdict"]
        if v == "VOID":
            return _lower(out, last_stable, "tier %d: %s RED (a tool left its healthy "
                          "band first)" % (e["tier_seq"], _red_checks(e)))
        if v == "UNDETERMINED":
            out.update(grade="UNDETERMINED", tier_seq=e["tier_seq"], I_status="UNDETERMINED")
            out["reasons"].append("tier %d: I signals disagree" % e["tier_seq"])
            return out
        i = (e.get("X") or {}).get("I") or {}
        out.update(tier_seq=e["tier_seq"], I_status=i.get("status"))
        if i.get("status") == "LOCALIZED":
            cls = i.get("measure_class")
            out.update(role=i.get("role"), finding=i.get("finding"), via=i.get("via"),
                       measure_class=cls, pids=i.get("pids"),
                       granularity=i.get("granularity") or "process")
            if cls in TOOLS:
                return _lower(out, last_stable, "tier %d: I localised to %s (%s): the tool "
                              "saturated first" % (e["tier_seq"], i.get("role"), cls))
            out["grade"] = "LOCALIZED"
        else:
            out["grade"] = "NOT_LOCALIZED"
            out["reasons"].append("tier %d: I %s" % (e["tier_seq"], i.get("status")))
        _interval(out, last_stable, e)
        return out
    above = [x["tier_seq"] for x in out["skipped_tiers"] if last_stable is None
             or (x["tier_seq"] != last_stable["tier_seq"] and _key(judged, x["tier_seq"])
                 > last_stable["order_key"])]
    if above:
        _lower(out, last_stable, "no unstable tier, but tiers %s above the highest stable tier "
               "have no stability conclusion (VOID / NOT_RUN / CONTRA)" % above)
        out["grade"] = "NOT_RUN"
        return out
    return _lower(out, last_stable, "X never held within the tool's range")


def _key(judged, seq):
    return next(e["order_key"] for e in judged if e["tier_seq"] == seq)


def _interval(out, last_stable, first_unstable):
    out["interval"] = {"lower_seq": last_stable["tier_seq"] if last_stable else None,
                       "lower": last_stable["load_value"] if last_stable else None,
                       "upper_seq": first_unstable["tier_seq"],
                       "upper": first_unstable["load_value"]}


def _lower(out, last_stable, why):
    out["grade"] = "LOWER_BOUND" if last_stable is not None else "NOT_RUN"
    out["reasons"].append(why)
    out["interval"] = {"lower_seq": last_stable["tier_seq"] if last_stable else None,
                       "lower": last_stable["load_value"] if last_stable else None,
                       "upper_seq": None, "upper": None}
    if last_stable is None:
        out["reasons"].append("no stable tier below it")
    return out


def final_grade(control_grades, v):
    """Grade across launches: the control launches' (delta = 0) grades with V; NOT_RUN when V
    found the control launches in more than one topology. `I_status`
    counts the control launches' I_status ("null": no UNSTABLE / UNDETERMINED tier set that
    launch's grade), so a NOT_LOCALIZED from I not run is visible."""
    out = {"grade": "NOT_RUN", "reasons": [], "V": v.get("status"), "granularity": "process"}
    ist = [g.get("I_status") or "null" for g in control_grades]
    out["I_status"] = {k: ist.count(k) for k in sorted(set(ist))}
    if v.get("control_topologies", 1) != 1:  # grades of different topologies do not combine
        out["reasons"].append("control launches of %s topologies" % v["control_topologies"])
        return out
    gs = [g for g in control_grades if g.get("grade") != "NOT_RUN"]
    out["n_launches"] = len(gs)
    if not gs:
        out["reasons"].append("no control launch with a grade")
        return out
    kinds = {g["grade"] for g in gs}
    roles = {g.get("role") for g in gs if g["grade"] == "LOCALIZED"}
    if "UNDETERMINED" in kinds or len(roles) > 1:
        out["grade"] = "UNDETERMINED"
        out["reasons"].append("launches disagree or I undetermined: %s / roles %s"
                              % (sorted(kinds), sorted(map(str, roles))))
    elif "LOWER_BOUND" in kinds:
        out["grade"] = "LOWER_BOUND"
    elif "NOT_LOCALIZED" in kinds:
        out["grade"] = "NOT_LOCALIZED"
    else:
        role = roles.pop()
        out.update(role=role, finding=sorted({str(g.get("finding")) for g in gs}))
        if v.get("status") == "PASS" and v.get("role") == role:
            out["grade"] = "CONFIRMED"
        elif v.get("status") == "FAIL":
            out["grade"] = "UNDETERMINED"
            out["reasons"].append("V FAIL: the intervention contradicts the localisation")
        else:
            out["grade"] = "LOCALIZED"
            out["reasons"].append("V %s" % v.get("status"))
    return out

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boundary analysis across launches (DATA-AND-ANALYSIS 6 steps 2-4; design 2.6 T4, T5;
INTERFACES 14.1).

    python3 -m capjudge.boundary --inputs <list> --workdir <W> --out <new boundary.json>
        [--compare <armA>,<armB> ...]

The list names one judged launch tree per line (literal paths, no globs; the same format as
capjudge.intervene). Each tree must be committed in <W>/JUDGED.tsv with the sha of its
verdicts.json. A tree whose launch-level conclusion from the writers before the judge
(caplaunch, capjobs: every STATE line not written by capjudge, and the judge's own record of
them) is not VALID is excluded and listed with the reason. Per launch:
  * T4 as the judge wrote it (verdicts.json boundary_T4; re-derived from the tiers and compared,
    a mismatch is an error);
  * the interval used across launches: T4 over the load tiers below the first tool-limited tier
    (a VOID tier whose RED checks are only C1 / C2, or an UNSTABLE tier whose I localised to a
    load_source / stub_out_of_domain role). When that cut comes before any unstable tier the
    upper end is dropped: the result is a lower bound ("censored", design 2.6).
Per arm: the envelope [min of lower ends, max of upper ends] in load order, over all its
launches and per block (block = job id: one job carries one launch of each arm), of the
interval used across launches (`envelope`, `by_block`) and of the judge's T4 as recorded
(`envelope_t4`, `by_block_t4`: for capjudge.scaling, which asks which side reached its limit
first, so a tool-limited tier is evidence there, not a cut). --compare gives T5 of the
intervals for two arms over the blocks both have. All inputs must share one axis.
"""

import argparse
import os
import sys

from capcommon.exitcodes import ConfigError
from capcommon.hashing import sha256_file
from capcommon.jsonio import atomic_write_bytes, dumps, read_json
from capcommon.runtree import read_state, worst_state

from . import cmdrec
from . import grade as G
from . import stats

SCHEMA = "capkit.boundary/1"
JUDGE_WRITER = "capjudge"
ID_KEYS = ("experiment", "arm", "job_id", "restart_count", "launch_seq")


class BoundaryError(ConfigError):
    pass


def read_list(path):
    out = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if any(c in line for c in "*?[") or "\t" in line:
                raise BoundaryError("%s:%d: one literal tree path per line" % (path, n))
            out.append(line)
    if not out:
        raise BoundaryError("%s lists no tree" % path)
    if len({os.path.abspath(t) for t in out}) != len(out):
        raise BoundaryError("a tree is listed twice")
    return out


def load_judged(tree, W):
    """(verdicts doc, verdicts path, sha) of a committed judgement (its JUDGED.tsv row carries
    the same verdicts.json sha). Raises BoundaryError."""
    from capcommon import ledger  # noqa: PLC0415
    vp = os.path.join(tree, "judge", "verdicts.json")
    if not os.path.isfile(vp):
        raise BoundaryError("%s is not judged (no judge/verdicts.json)" % tree)
    v = read_json(vp)
    ident = v.get("identity") or {}
    key = tuple(ident.get(k) for k in ("job_id", "restart_count", "launch_seq", "arm"))
    hit = ledger.judged_lookup(os.path.join(W, "JUDGED.tsv"), *key)
    sha = sha256_file(vp)
    if hit is None or hit["verdicts_sha256"] != sha:
        raise BoundaryError("%s: JUDGED.tsv has no row with this verdicts.json sha" % tree)
    return v, vp, sha


def upstream_now(tree, verdicts):
    """Launch-level conclusion before the judge: the judge's record of it (verdicts.json
    `upstream`, LAUNCH-STATE) and every STATE line not written by capjudge, read now (a line
    appended after the judgement counts). Returns (status, [reasons])."""
    try:
        lines = read_state(tree)
    except (ConfigError, OSError, UnicodeDecodeError) as e:
        return "NOT_RUN", ["STATE unreadable: %s" % e]
    sts = [(st, "STATE %s by %s: %s" % (st, by, reason)) for _at, st, by, reason in lines
           if by != JUDGE_WRITER]
    up = verdicts.get("upstream") or {}
    sts.append((up.get("status") or "NOT_RUN", "judge's upstream record %s"
                % up.get("status")))
    ls = (verdicts.get("launch_checks") or {}).get("LAUNCH-STATE")
    sts.append(("VALID" if ls == "PASS" else "NOT_RUN", "LAUNCH-STATE %s" % ls))
    status = worst_state(s for s, _ in sts)
    return status, [why for s, why in sts if s != "VALID"]


def _judged_tiers(verdicts):
    return [t for t in verdicts.get("tiers", []) if t.get("judged")
            and t.get("order_key") is not None]


def _t4_of(tiers):
    return stats.t4_boundary([{"seq": e["tier_seq"], "order_key": e["order_key"],
                               "load_value": e["load_value"], "verdict": e["verdict"]}
                              for e in tiers])


def tool_limited(entry):
    """Why this load tier marks a tool limit, or None (design 2.6: the load source or an
    out-of-domain stub left its healthy band first)."""
    red = G._red_checks(entry)
    if entry.get("verdict") == "VOID" and red and set(red) <= set(G.TOOL_CHECKS):
        return "%s RED" % red
    i = (entry.get("X") or {}).get("I") or {}
    if (entry.get("verdict") == "UNSTABLE" and i.get("status") == "LOCALIZED"
            and i.get("measure_class") in G.TOOLS):
        return "I localised to %s (%s)" % (i.get("role"), i.get("measure_class"))
    return None


def launch_interval(verdicts):
    """(judge T4 as recorded, interval used across launches). Raises BoundaryError when the
    recorded T4 does not follow from the recorded tiers."""
    tiers = _judged_tiers(verdicts)
    rec = verdicts.get("boundary_T4") or {}
    t4 = _t4_of(tiers)
    for k in ("lower", "upper", "lower_seq", "upper_seq", "kind"):
        if rec.get(k) != t4[k]:
            raise BoundaryError("verdicts.json boundary_T4.%s=%r but its tiers give %r"
                                % (k, rec.get(k), t4[k]))
    cuts = sorted(((e["order_key"], e["tier_seq"], tool_limited(e)) for e in tiers
                   if tool_limited(e)), key=lambda c: (c[0], c[1]))
    if not cuts:
        used = dict(t4, censored_by=None)
    else:
        key, seq, why = cuts[0]
        used = _t4_of([e for e in tiers if e["order_key"] < key])
        used["censored_by"] = {"tier_seq": seq, "why": why}
        if used["upper"] is None:
            used["kind"] = "lower_bound_only" if used["lower"] is not None else "none"
    used["skipped"] = [{"tier_seq": e["tier_seq"], "verdict": e["verdict"]} for e in tiers
                       if e["verdict"] not in ("STABLE", "UNSTABLE")]
    return t4, used


def _roles(tree):
    try:
        launch = read_json(os.path.join(tree, "raw", "launch.json"))
    except (OSError, ValueError) as e:
        raise BoundaryError("%s: raw/launch.json unreadable: %s" % (tree, e))
    return {r["role"]: {"kind": r.get("kind"), "measure_class": r.get("measure_class"),
                        "workers": r.get("workers")} for r in launch.get("roles") or []}


def build(trees, W, compares):
    W = os.path.abspath(W)
    out = {"schema": SCHEMA, "workdir": W, "inputs": [], "excluded": [], "launches": [],
           "arms": {}, "t5": []}
    axes = set()
    for tree in trees:
        tree = os.path.abspath(tree)
        v, vp, sha = load_judged(tree, W)
        ident = {k: (v.get("identity") or {}).get(k) for k in ID_KEYS}
        out["inputs"].append({"tree": tree, "verdicts": vp, "verdicts_sha256": sha})
        st, why = upstream_now(tree, v)
        if st != "VALID":
            out["excluded"].append({"tree": tree, "identity": ident,
                                    "why": "launch-level conclusion %s: %s" % (st, why)})
            continue
        axes.add(repr(v.get("axis")))
        if len(axes) > 1:
            raise BoundaryError("inputs scan different axes %s" % sorted(axes))
        out["axis"] = v.get("axis")
        t4, used = launch_interval(v)
        iv = v.get("intervention_V")
        out["launches"].append({
            "tree": tree, "identity": ident, "block": str(ident["job_id"]),
            "mode": v.get("mode"), "launch_verdict": v.get("launch_verdict"),
            "t4": t4, "interval": used, "roles": _roles(tree),
            "grade": v.get("grade") if isinstance(v.get("grade"), dict) else None,
            "doses": iv.get("doses") if isinstance(iv, dict) else None,
            "heartbeat": iv.get("heartbeat") if isinstance(iv, dict) else None})
    if not out["launches"]:
        raise BoundaryError("no launch left after the exclusions %s" % out["excluded"])
    order = out["axis"]["order"]
    by_arm = {}
    for x in out["launches"]:
        by_arm.setdefault(x["identity"]["arm"], []).append(x)
    for arm, ls in sorted(by_arm.items()):
        blocks = {}
        for x in ls:
            blocks.setdefault(x["block"], []).append(x)
        out["arms"][arm] = {
            "n_launches": len(ls), "envelope": _env(ls, order),
            "by_block": {b: _env(xs, order) for b, xs in sorted(blocks.items())},
            "envelope_t4": _env(ls, order, "t4"),
            "by_block_t4": {b: _env(xs, order, "t4") for b, xs in sorted(blocks.items())}}
    for a, b in compares:
        out["t5"].append(compare(out["arms"], a, b))
    return out


def _env(launches, order, key="interval"):
    return stats.t4_envelope([(x[key]["lower"], x[key]["upper"]) for x in launches], order)


def compare(arms, a, b):
    """T5 of arms a and b over the blocks both have (design 2.6)."""
    for arm in (a, b):
        if arm not in arms:
            raise BoundaryError("--compare names arm %r, not among the analysed launches" % arm)
    ba, bb = arms[a]["by_block"], arms[b]["by_block"]
    common = sorted(set(ba) & set(bb))
    res = {"a": a, "b": b, "blocks": common,
           "blocks_missing": sorted(set(ba) ^ set(bb))}
    if not common:
        res.update(result="NOT_RUN", why="no block has launches of both arms")
        return res
    res["t5"] = stats.t5_compare([{"block": k, "a": ba[k], "b": bb[k]} for k in common])
    res["result"] = res["t5"]["result"]
    return res


def parse_compare(vals):
    out = []
    for s in vals or []:
        parts = s.split(",")
        if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
            raise BoundaryError("--compare wants '<armA>,<armB>' (two different arms): %r" % s)
        out.append(tuple(parts))
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="capjudge.boundary", description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", action="append")
    a = ap.parse_args(argv)
    try:
        bad = cmdrec.refuse_existing(a.out)
        if bad:
            raise BoundaryError(bad)
        doc = build(read_list(a.inputs), a.workdir, parse_compare(a.compare))
        doc["inputs_list"] = {"path": os.path.abspath(a.inputs),
                              "sha256": sha256_file(a.inputs)}
    except (BoundaryError, ConfigError, OSError, ValueError, KeyError) as e:
        sys.stderr.write("capjudge.boundary: %s\n" % e)
        return 3
    atomic_write_bytes(a.out, (dumps(doc) + "\n").encode("utf-8"))
    cmdrec.write(a.out, "capjudge.boundary", argv,
                 [a.inputs] + [i["verdicts"] for i in doc["inputs"]])
    print("wrote %s: %d launch(es), %d excluded, %d arm(s), %d T5" % (
        a.out, len(doc["launches"]), len(doc["excluded"]), len(doc["arms"]), len(doc["t5"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

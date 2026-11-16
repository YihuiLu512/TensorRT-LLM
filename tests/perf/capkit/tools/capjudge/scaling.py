# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scaling ratio T6 and opposite-side doubling attribution (design 2.6 T6; E1 "隔离";
INTERFACES 14.2).

    python3 -m capjudge.scaling --boundary <boundary.json> --pair <base>,<scaled> [--pair ...]
        --grows-min <ratio> --flat-max <ratio> --out <new scaling.json>

Input is a capjudge.boundary product (the T4 intervals are computed once, there). A pair names
two arms of it; the scaled arm must differ from the base only in the worker count of the roles
on ONE side (kind client = the load generators, kind server = the sinks / stubs), all by the
same factor f > 1, the same in every launch of the arm. The envelopes are those of the judge's
T4 as recorded (`by_block_t4`), not the intervals cut at the first tool-limited tier: which
side reached its limit first is what the doubling decides, so a tier where a tool (an
out-of-domain sink, the load generator) saturated is evidence here, not a cut. Per block both
arms have, with the per-block envelopes E_b (base) and E_s (scaled): the ratio of the boundary
lies in
[E_s.lower / E_b.upper, E_s.upper / E_b.lower] (None when an end is missing). T6 = the range
of the per-block ratios (also of the lower ends and of the upper ends alone). The claims use
the unfavourable end over every common block (design 2.6 T6): "grows" when the smallest ratio
>= --grows-min, "flat" when the largest ratio <= --flat-max, else "unresolved". A block whose
lower (upper) ratio is unknown leaves the smallest (largest) ratio unknown (None, the blocks
listed), so no "grows" ("flat") is claimed; no common block with its lower ratio known (base
interval closed, scaled lower end present) -> NOT_RUN.
The bounds are the experiment's registered values; they are passed explicitly and recorded.

Attribution (E1): growth when side S is scaled says S was limiting; no growth says the limit is
on the other side. Only when a client-side and a server-side pair agree is a side named
("两种情形各出现一次，才知道这一对里谁是谁"): client when the server-side pair is flat and the
client-side pair grows; server for the mirror image; any other combination of resolved pairs
is UNDETERMINED; missing one side is NOT_RUN.
"""

import argparse
import os
import sys

from capcommon.exitcodes import ConfigError
from capcommon.jsonio import atomic_write_bytes, dumps, read_json

from . import boundary as B
from . import cmdrec
from . import stats

SCHEMA = "capkit.scaling/1"
SIDES = ("client", "server")


class ScalingError(ConfigError):
    pass


def _arm_roles(bdoc, arm):
    sets = {B.topology(x["roles"]) for x in bdoc["launches"] if x["identity"]["arm"] == arm}
    if not sets:
        raise ScalingError("arm %r has no launch in the boundary file" % arm)
    if len(sets) != 1:
        raise ScalingError("arm %r: roles differ between its launches" % arm)
    x = next(x for x in bdoc["launches"] if x["identity"]["arm"] == arm)
    return x["roles"]


def scaled_side(base, scaled):
    """(side, factor) when only one side's worker counts changed, all by one factor > 1;
    raises ScalingError otherwise."""
    if set(base) != set(scaled):
        raise ScalingError("the arms have different roles %s / %s" % (sorted(base),
                                                                      sorted(scaled)))
    changed = {}
    for r in base:
        b, s = base[r], scaled[r]
        if (b["kind"], b["measure_class"]) != (s["kind"], s["measure_class"]):
            raise ScalingError("role %s differs in kind / measure_class" % r)
        if b["workers"] != s["workers"]:
            if not (isinstance(b["workers"], int) and isinstance(s["workers"], int)
                    and b["workers"] > 0):
                raise ScalingError("role %s: worker counts %r / %r" % (r, b["workers"],
                                                                      s["workers"]))
            changed[r] = (b["kind"], s["workers"] / b["workers"])
    kinds = {k for k, _ in changed.values()}
    factors = {f for _, f in changed.values()}
    if len(kinds) != 1 or kinds - set(SIDES) or len(factors) != 1:
        raise ScalingError("not a one-sided scaling: changed roles %s" % changed)
    side, f = kinds.pop(), factors.pop()
    if f <= 1:
        raise ScalingError("the scaled arm has fewer workers (factor %s)" % f)
    unchanged_same_side = [r for r in base if base[r]["kind"] == side and r not in changed]
    if unchanged_same_side:
        raise ScalingError("side %s: roles %s were not scaled with %s"
                           % (side, unchanged_same_side, sorted(changed)))
    return side, f, sorted(changed)


def _t4_blocks(bdoc, arm):
    blocks = bdoc["arms"][arm].get("by_block_t4")
    if not isinstance(blocks, dict):
        raise ScalingError("boundary file has no arms.%s.by_block_t4 (written by an older "
                           "capjudge.boundary?)" % arm)
    return blocks


def pair(bdoc, base, scaled, grows_min, flat_max):
    if bdoc["axis"]["order"] != "asc":
        raise ScalingError("T6 of boundaries needs an ascending axis (load grows with "
                           "load_value); axis is %s" % bdoc["axis"])
    side, f, roles = scaled_side(_arm_roles(bdoc, base), _arm_roles(bdoc, scaled))
    eb, es = (_t4_blocks(bdoc, base), _t4_blocks(bdoc, scaled))
    common = sorted(set(eb) & set(es))
    res = {"base": base, "scaled": scaled, "side": side, "factor": f, "roles": roles,
           "blocks": common, "blocks_missing": sorted(set(eb) ^ set(es)), "per_block": []}
    lo, hi = {}, {}
    for k in common:
        b, s = eb[k], es[k]
        r_lo, r_hi = stats.ratio(s["lower"], b["upper"]), stats.ratio(s["upper"], b["lower"])
        res["per_block"].append({"block": k, "ratio_min": r_lo, "ratio_max": r_hi,
                                 "efficiency_min": stats.ratio(r_lo, f),
                                 "efficiency_max": stats.ratio(r_hi, f)})
        lo[k], hi[k] = r_lo, r_hi
    res["t6_lower_ends"] = stats.t6_ratio_range([(k, es[k]["lower"], eb[k]["lower"])
                                                 for k in common])
    res["t6_upper_ends"] = stats.t6_ratio_range([(k, es[k]["upper"], eb[k]["upper"])
                                                 for k in common])
    if all(lo[k] is None for k in common):
        res.update(ratio=None, claim="NOT_RUN", why="no common block with the base interval "
                   "closed and the scaled lower end present")
        return res
    # every common block counts: a block whose end is unknown leaves the claim that needs that
    # end unresolved, never decided on the other blocks
    open_lo = [k for k in common if lo[k] is None]
    open_hi = [k for k in common if hi[k] is None]
    res["ratio"] = {"min": None if open_lo else min(lo.values()),
                    "max": None if open_hi else max(hi.values()),
                    "min_unknown_blocks": open_lo, "max_unknown_blocks": open_hi}
    if res["ratio"]["min"] is not None and res["ratio"]["min"] >= grows_min:
        res["claim"] = "grows"
    elif res["ratio"]["max"] is not None and res["ratio"]["max"] <= flat_max:
        res["claim"] = "flat"
    else:
        res["claim"] = "unresolved"
    res["limit_side"] = (side if res["claim"] == "grows" else
                         _other(side) if res["claim"] == "flat" else None)
    return res


def _other(side):
    return SIDES[1 - SIDES.index(side)]


def attribute(pairs):
    """Combine resolved pairs (module docstring)."""
    by = {s: [p for p in pairs if p["side"] == s and p.get("claim") in ("grows", "flat")]
          for s in SIDES}
    out = {"result": "NOT_RUN", "evidence": {s: [(p["base"], p["scaled"], p["claim"])
                                                 for p in by[s]] for s in SIDES}}
    missing = [s for s in SIDES if not by[s]]
    if missing:
        out["why"] = "no resolved pair scaling the %s side" % " / ".join(missing)
        return out
    claims = {s: {p["claim"] for p in by[s]} for s in SIDES}
    if any(len(c) != 1 for c in claims.values()):
        out.update(result="UNDETERMINED", why="pairs on one side disagree %s" % claims)
        return out
    c = {s: claims[s].pop() for s in SIDES}
    if c["server"] == "flat" and c["client"] == "grows":
        out["result"] = "client"
    elif c["client"] == "flat" and c["server"] == "grows":
        out["result"] = "server"
    else:
        out.update(result="UNDETERMINED", why="client side %s, server side %s"
                   % (c["client"], c["server"]))
    return out


def parse_pairs(vals):
    out = []
    for s in vals:
        parts = s.split(",")
        if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
            raise ScalingError("--pair wants '<base arm>,<scaled arm>': %r" % s)
        out.append(tuple(parts))
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="capjudge.scaling", description=__doc__.split("\n")[0])
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--pair", action="append", required=True)
    ap.add_argument("--grows-min", type=float, required=True)
    ap.add_argument("--flat-max", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        bad = cmdrec.refuse_existing(a.out)
        if bad:
            raise ScalingError(bad)
        if not 0 < a.flat_max < a.grows_min:
            raise ScalingError("need 0 < --flat-max < --grows-min")
        bdoc = read_json(a.boundary)
        if not isinstance(bdoc, dict) or bdoc.get("schema") != B.SCHEMA:
            raise ScalingError("%s is not a %s file" % (a.boundary, B.SCHEMA))
        pairs = [pair(bdoc, b, s, a.grows_min, a.flat_max) for b, s in parse_pairs(a.pair)]
        doc = {"schema": SCHEMA, "boundary": os.path.abspath(a.boundary),
               "bounds": {"grows_min": a.grows_min, "flat_max": a.flat_max},
               "axis": bdoc["axis"], "pairs": pairs, "attribution": attribute(pairs)}
    except (ScalingError, ConfigError, OSError, ValueError, KeyError) as e:
        sys.stderr.write("capjudge.scaling: %s\n" % e)
        return 3
    atomic_write_bytes(a.out, (dumps(doc) + "\n").encode("utf-8"))
    cmdrec.write(a.out, "capjudge.scaling", argv, [a.boundary])
    print("wrote %s: %s; attribution %s" % (a.out, ", ".join(
        "%s->%s %s" % (p["base"], p["scaled"], p["claim"]) for p in pairs),
        doc["attribution"]["result"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

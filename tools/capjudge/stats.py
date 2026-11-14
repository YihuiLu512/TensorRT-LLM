# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Statistics T1-T6 with the fixed forms of design section 2.6 (stdlib only).

The forms are fixed by the design; the pilot only supplies thresholds (noise bands). Every
function returns None (never 0) when its input is insufficient, so callers can record NOT_RUN.
"""

import statistics

# ---- helpers ---------------------------------------------------------------------------------


def median(values):
    """Median of a non-empty sequence; None for an empty one."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return statistics.median(vals)


# ---- T1 trend --------------------------------------------------------------------------------


def t1_trend(series):
    """T1: split the steady-window series in half; D = median(second half) - median(first half).

    series: per-second values in time order (None entries are empty seconds and are skipped
    after the split, so the split is by time, not by sample count). With an odd number of
    seconds the middle second belongs to neither half. Returns (D, n_first, n_second) or None
    when either half has no value.
    """
    n = len(series)
    if n < 2:
        return None
    half = n // 2
    first = [v for v in series[:half] if v is not None]
    second = [v for v in series[n - half:] if v is not None]
    if not first or not second:
        return None
    return (statistics.median(second) - statistics.median(first), len(first), len(second))


# ---- T2 location offset ----------------------------------------------------------------------


def t2_offset(values, baseline_values):
    """T2: median(this tier's window readings) - median(the same launch's baseline tier)."""
    a = median(values)
    b = median(baseline_values)
    if a is None or b is None:
        return None
    return a - b


# ---- T3 noise band ---------------------------------------------------------------------------


def t3_band(samples):
    """T3: empirical range [min, max] of the statistic over known-unsaturated tiers, with N.

    The nominal false-alarm rate of "new reading outside the range" is 1 / (N + 1) for N
    exchangeable samples (computed by the reader, not stored)."""
    vals = [v for v in samples if v is not None]
    if not vals:
        return None
    return {"min": min(vals), "max": max(vals), "n": len(vals)}


def band_is_valid(band):
    return (isinstance(band, dict) and all(k in band for k in ("min", "max", "n"))
            and isinstance(band["n"], int) and not isinstance(band["n"], bool) and band["n"] >= 1
            and _num(band["min"]) and _num(band["max"]) and band["min"] <= band["max"])


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare_to_band(value, band):
    """'above' / 'below' / 'inside' (inside includes the end points); None if either is None."""
    if value is None or band is None:
        return None
    if value > band["max"]:
        return "above"
    if value < band["min"]:
        return "below"
    return "inside"


def nominal_false_alarm(band):
    return 1.0 / (band["n"] + 1)


# ---- T4 boundary interval --------------------------------------------------------------------


def t4_boundary(tiers):
    """T4 for one launch: [highest stable tier, lowest unstable tier] on the load axis.

    tiers: list of dicts {"seq", "order_key", "load_value", "verdict"}; order_key increases with
    load (TierTable.load_order_key). Only STABLE / UNSTABLE tiers count. Returns a dict with
    lower (load_value of the highest stable tier or None), upper (lowest unstable or None),
    kind: "interval" | "lower_bound_only" (no unstable tier: the result is a lower bound) |
    "upper_only" (no stable tier) | "none", and non_monotone=True when some stable tier lies
    above the lowest unstable tier (reported, never hidden).
    """
    stable = [t for t in tiers if t["verdict"] == "STABLE" and t["order_key"] is not None]
    unstable = [t for t in tiers if t["verdict"] == "UNSTABLE" and t["order_key"] is not None]
    hi_stable = max(stable, key=lambda t: t["order_key"]) if stable else None
    lo_unstable = min(unstable, key=lambda t: t["order_key"]) if unstable else None
    if hi_stable and lo_unstable:
        kind = "interval"
    elif hi_stable:
        kind = "lower_bound_only"
    elif lo_unstable:
        kind = "upper_only"
    else:
        kind = "none"
    non_mono = bool(hi_stable and lo_unstable
                    and hi_stable["order_key"] > lo_unstable["order_key"])
    return {"lower": hi_stable["load_value"] if hi_stable else None,
            "lower_seq": hi_stable["seq"] if hi_stable else None,
            "upper": lo_unstable["load_value"] if lo_unstable else None,
            "upper_seq": lo_unstable["seq"] if lo_unstable else None,
            "kind": kind, "non_monotone": non_mono,
            "n_stable": len(stable), "n_unstable": len(unstable)}


def t4_envelope(intervals, order="asc"):
    """T4 across launches: list every interval, envelope = [min of lower ends, max of upper ends]
    (in load order; for order 'desc' the numerically smaller value is the heavier load).

    intervals: list of (lower, upper) where either may be None. The envelope end is None when any
    launch lacks that end (a missing upper end means "no unstable tier reached": lower bound)."""
    if not intervals:
        return None
    sign = 1 if order == "asc" else -1
    lows = [lo for lo, _ in intervals]
    highs = [hi for _, hi in intervals]
    low = None if any(v is None for v in lows) else sign * min(sign * v for v in lows)
    high = None if any(v is None for v in highs) else sign * max(sign * v for v in highs)
    return {"lower": low, "upper": high, "n_launches": len(intervals),
            "launches": [list(i) for i in intervals]}


# ---- T5 two-arm difference -------------------------------------------------------------------


def _span(env):
    return (min(env["lower"], env["upper"]), max(env["lower"], env["upper"]))


def t5_compare(blocks):
    """T5: blocks = list of {"block": name, "a": envelope, "b": envelope} (same-block T4
    envelopes of the two arms). Report a difference only if in EVERY block the two envelopes are
    disjoint and the direction is the same; otherwise "indistinguishable" with the width of the
    union of both envelopes as the smallest difference this run could resolve.

    An envelope with a missing end cannot be shown disjoint, so that block counts against
    "different"; when no block has both envelopes closed the comparison cannot tell a
    difference from none, and the result is NOT_RUN (not "indistinguishable"). With 3
    launches per arm, complete separation has probability 1/20 (direction specified) or 1/10
    (unspecified) under no difference; it is the only conclusion 3 launches support.
    """
    if not blocks:
        return None
    per_block = []
    directions = set()
    min_resolvable = []
    all_disjoint = True
    for b in blocks:
        a, c = b["a"], b["b"]
        if None in (a.get("lower"), a.get("upper"), c.get("lower"), c.get("upper")):
            per_block.append({"block": b["block"], "disjoint": False,
                              "reason": "envelope has an open end"})
            all_disjoint = False
            continue
        sa, sc = _span(a), _span(c)
        union = (min(sa[0], sc[0]), max(sa[1], sc[1]))
        min_resolvable.append(union[1] - union[0])
        if sa[1] < sc[0]:
            d = "b_higher"
        elif sc[1] < sa[0]:
            d = "a_higher"
        else:
            d = None
        if d is None:
            all_disjoint = False
        else:
            directions.add(d)
        per_block.append({"block": b["block"], "disjoint": d is not None, "direction": d,
                          "union_width": union[1] - union[0]})
    if not min_resolvable:
        return {"result": "NOT_RUN", "min_resolvable_diff": None, "blocks": per_block,
                "why": "no block has both envelopes closed"}
    if all_disjoint and len(directions) == 1:
        return {"result": "different", "direction": directions.pop(), "blocks": per_block}
    return {"result": "indistinguishable",
            "min_resolvable_diff": max(min_resolvable) if min_resolvable else None,
            "blocks": per_block}


def ratio(a, b):
    """a / b, or None when a is None or b is None / 0."""
    return None if a is None or b in (None, 0) else a / b


# ---- T6 ratio --------------------------------------------------------------------------------


def t6_ratio_range(pairs):
    """T6: pairs = list of (block, numerator, denominator) from the same block. Ratio per block;
    interval = [min, max] of the per-block ratios. Blocks with a zero/None denominator are
    reported and excluded; no valid block -> None."""
    ratios, skipped = [], []
    for block, num, den in pairs:
        if num is None or den in (None, 0):
            skipped.append(block)
            continue
        ratios.append((block, num / den))
    if not ratios:
        return None
    vals = [r for _, r in ratios]
    return {"min": min(vals), "max": max(vals), "per_block": ratios, "skipped": skipped}


def t6_meets_bound(rng, bound, better):
    """Compare a T6 interval with a pre-registered bound using the unfavourable end.

    better='higher': passes iff rng.min >= bound; better='lower': passes iff rng.max <= bound."""
    if rng is None:
        return None
    if better == "higher":
        return rng["min"] >= bound
    if better == "lower":
        return rng["max"] <= bound
    raise ValueError("better must be 'higher' or 'lower'")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Statistics T1-T6 with the fixed forms of design section 2.6, and the calibration v2
primitives (stdlib only).

The forms are fixed by the design; the pilot only supplies thresholds (noise bands). The T1-T6
functions return None (never 0) when their input is insufficient, so callers can record
NOT_RUN. The calibration v2 primitives (section "calibration v2" below; calib-v2 DESIGN-r7a
2.2, 2.3, 2.6, 2.7, 2.10, 4.5) return a NotRun(reason) instead, and raise ValueError on input
that breaks their contract.
"""

import bisect
import math
import random
import statistics
from fractions import Fraction

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
    load (TierTable.load_order_key). Only STABLE / UNSTABLE tiers make the interval. Returns a
    dict with lower (load_value of the highest stable tier or None), upper (lowest unstable or
    None), kind: "interval" | "lower_bound_only" (no unstable tier: the result is a lower
    bound) | "upper_only" (no stable tier) | "none", and non_monotone=True when some stable
    tier lies above the lowest tier where X held: the lowest UNSTABLE or UNDETERMINED tier
    (UNDETERMINED: X held, I did not agree on a cause; reported, never hidden).
    """
    stable = [t for t in tiers if t["verdict"] == "STABLE" and t["order_key"] is not None]
    unstable = [t for t in tiers if t["verdict"] == "UNSTABLE" and t["order_key"] is not None]
    x_held = [t["order_key"] for t in tiers
              if t["verdict"] in ("UNSTABLE", "UNDETERMINED") and t["order_key"] is not None]
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
    non_mono = bool(hi_stable and x_held and hi_stable["order_key"] > min(x_held))
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


# ---- calibration v2 (calib-v2 DESIGN-r7a; pure functions) ------------------------------------


class NotRun:
    """A statistic that could not be computed; `reason` names why (the caller records
    NOT_RUN with it). Compares equal to another NotRun with the same reason."""

    __slots__ = ("reason",)

    def __init__(self, reason):
        self.reason = reason

    def __eq__(self, other):
        return isinstance(other, NotRun) and other.reason == self.reason

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(("NotRun", self.reason))

    def __repr__(self):
        return "NotRun(%r)" % (self.reason,)


ZD_MAD_SCALE = 1.4826     # MAD -> SD under normality
ZD_MEDIAN_SE = 1.2533     # sqrt(pi / 2): SE of a median over SE of a mean


def _mad(xs):
    m = statistics.median(xs)
    return statistics.median([abs(x - m) for x in xs])


def zd(series, quantum=None):
    """Trend statistic z_d of one steady window (DESIGN-r7a 2.2).

    series: per-second values s[0..L-1] in time order, None for an empty second. Steps:
    1. D, n1, n2 = t1_trend(series) (half split by time; the middle second of an odd length
       belongs to neither half);
    2. first differences d_i of adjacent seconds that are both non-None; fewer than L / 2 of
       them -> NotRun("too_few_diffs");
    3. sigma = 1.4826 * MAD(d) / sqrt(2); 0 -> population SD(d) / sqrt(2); with a `quantum` q
       (count family) sigma = max(sigma, q / sqrt(12)); still 0 -> z = 0 when D == 0, else
       NotRun("degenerate_scale");
    4. z = D / (1.2533 * sigma * sqrt(1 / n1 + 1 / n2)).
    Returns {"z", "D", "sigma", "n1", "n2"}. A half without any value (possible only when step
    2 already fails, checked first) -> NotRun("empty_half")."""
    L = len(series)
    d = [b - a for a, b in zip(series, series[1:]) if a is not None and b is not None]
    if len(d) < L / 2.0:
        return NotRun("too_few_diffs")
    t1 = t1_trend(series)
    if t1 is None:
        return NotRun("empty_half")
    D, n1, n2 = t1
    sigma = ZD_MAD_SCALE * _mad(d) / math.sqrt(2)
    if sigma == 0:
        sigma = statistics.pstdev(d) / math.sqrt(2)
    if quantum is not None:
        sigma = max(sigma, quantum / math.sqrt(12))
    if sigma == 0:
        if D == 0:
            return {"z": 0.0, "D": D, "sigma": 0.0, "n1": n1, "n2": n2}
        return NotRun("degenerate_scale")
    z = D / (ZD_MEDIAN_SE * sigma * math.sqrt(1.0 / n1 + 1.0 / n2))
    return {"z": z, "D": D, "sigma": sigma, "n1": n1, "n2": n2}


# Student t tail and quantile from the regularized incomplete beta function (continued
# fraction, Numerical Recipes betacf); a port of calib-v2 r4/tdist.py.

def _betacf(a, b, x, itmax=400, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x)
           + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_sf(x, df):
    """P(T > x) for Student t with df > 0 degrees of freedom."""
    if df <= 0:
        raise ValueError("t_sf: df must be > 0")
    p = 0.5 * _betai(df / 2.0, 0.5, df / (df + x * x))
    return p if x >= 0 else 1.0 - p


T_ISF_MAX = 1e12


def t_isf(p, df):
    """x with P(T > x) = p, 0 < p < 1 (p > 0.5 by symmetry): the upper end doubles from 1
    until t_sf(hi) < p (past T_ISF_MAX: ValueError, never a clipped value), then 200
    bisection steps on t_sf."""
    if not 0 < p < 1:
        raise ValueError("t_isf: p must be in (0, 1)")
    if p > 0.5:
        return -t_isf(1.0 - p, df)
    if p == 0.5:
        return 0.0
    lo, hi = 0.0, 1.0
    while t_sf(hi, df) >= p:
        lo, hi = hi, hi * 2
        if hi > T_ISF_MAX:
            raise ValueError("t_isf(%r, %r) is above %g" % (p, df, T_ISF_MAX))
    for _ in range(200):
        m = (lo + hi) / 2
        if t_sf(m, df) > p:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


# ---- empirical-quantile thresholds (DESIGN-r7a 2.3) ------------------------------------------

def _binom_sf(n, p, k):
    """P(Binom(n, p) >= k) (pmf summed in log space)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    below = math.fsum(math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
                               + j * lp + (n - j) * lq) for j in range(k))
    return max(0.0, 1.0 - below)


def kmax(n, p, gamma=0.95):
    """Largest integer k with P(Binom(n, p) >= k) >= gamma (k = 0 always qualifies; 0 means the
    key is `uncalibrated`). With p = 0.01: n = 298 -> 0, 299 -> 1, 600 -> 2, 720 -> 3."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("kmax: n must be an int >= 0")
    if not 0 < p < 1:
        raise ValueError("kmax: p must be in (0, 1)")
    k = 0
    while k < n and _binom_sf(n, p, k + 1) >= gamma:
        k += 1
    return k


def n_eff(n_windows, m_mean, icc_ub):
    """Effective number of windows floor(n_windows / (1 + (m - 1) * ICC_ub)), m = mean windows
    per launch on the key (>= 1), ICC_ub floored at 0. The quotient is floored after adding
    1e-9 so that an exact integer computed as x.999999... is not lost to rounding."""
    if not isinstance(n_windows, int) or isinstance(n_windows, bool) or n_windows < 0:
        raise ValueError("n_eff: n_windows must be an int >= 0")
    if m_mean < 1:
        raise ValueError("n_eff: m_mean must be >= 1")
    deff = 1.0 + (m_mean - 1.0) * max(0.0, icc_ub)
    return int(math.floor(n_windows / deff + 1e-9))


def _ceil_div(a, b):
    return -(-a // b)


def rank_for_k(k, n_windows, n_eff):
    """Rank r = ceil(k * (n_windows + 1) / (n_eff + 1)) on the full sample (2.3): the k-th
    largest of n_eff independent values mapped by quantile position onto all n_windows values
    (exact integer arithmetic). 449 / 299, k = 1 -> 2; 748 / 299 -> 3; 1500 / 600, k = 2 -> 5."""
    for name, v in (("k", k), ("n_windows", n_windows), ("n_eff", n_eff)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError("rank_for_k: %s must be an int >= 0" % name)
    return _ceil_div(k * (n_windows + 1), n_eff + 1)


def fold_rank(r, n, n_fold):
    """Rank on a leave-one-out fold (2.3, CR3-S3): r' = max(1, ceil(r * (n_fold + 1) /
    (n + 1))), r the full-sample rank, n the full-sample window count; k is not recomputed."""
    for name, v in (("r", r), ("n", n), ("n_fold", n_fold)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError("fold_rank: %s must be an int >= 0" % name)
    return max(1, _ceil_div(r * (n_fold + 1), n + 1))


def order_stat_thresholds(values, n_eff, p):
    """Two-sided empirical thresholds of one key (2.3): k = kmax(n_eff, p), r = rank_for_k(k,
    len(values), n_eff); thr_hi = the r-th largest value, thr_lo = the r-th smallest. k = 0 ->
    `uncalibrated` (thresholds None; the checks that use the key are NOT_RUN). Returns
    {"k", "r", "n_windows", "n_eff", "p", "thr_hi", "thr_lo", "uncalibrated"}."""
    vals = list(values)
    if any(v is None or not isinstance(v, (int, float)) or isinstance(v, bool)
           or not math.isfinite(v) for v in vals):
        raise ValueError("order_stat_thresholds: values must be finite numbers")
    n = len(vals)
    if not isinstance(n_eff, int) or isinstance(n_eff, bool) or not 0 <= n_eff <= n:
        raise ValueError("order_stat_thresholds: n_eff must be an int in [0, len(values)]")
    k = kmax(n_eff, p)
    out = {"k": k, "r": 0, "n_windows": n, "n_eff": n_eff, "p": p, "thr_hi": None,
           "thr_lo": None, "uncalibrated": k == 0}
    if k == 0:
        return out
    r = rank_for_k(k, n, n_eff)
    if r > n:
        raise ValueError("order_stat_thresholds: rank %d exceeds %d values" % (r, n))
    s = sorted(vals)
    out.update(r=r, thr_hi=s[n - r], thr_lo=s[r - 1])
    return out


# ---- intra-class correlation (2.3 n_eff, 2.8 ICC_ub) -----------------------------------------

def icc_oneway(groups):
    """One-way random-effects ICC(1) by ANOVA with unequal group sizes (n0 correction; port of
    calib-v2 r2/icc_zd.py). groups: lists of numbers (one list per launch); empty lists are
    ignored. Fewer than 2 groups, no within-group degree of freedom, or a zero total variance
    -> NotRun. The estimate can be negative (no floor here)."""
    gs = [list(g) for g in groups if len(g) >= 1]
    N = sum(len(g) for g in gs)
    k = len(gs)
    if k < 2 or N <= k:
        return NotRun("too_few_groups_or_values")
    grand = math.fsum(math.fsum(g) for g in gs) / N
    means = [math.fsum(g) / len(g) for g in gs]
    ssb = math.fsum(len(g) * (m - grand) ** 2 for g, m in zip(gs, means))
    ssw = math.fsum(math.fsum((x - m) ** 2 for x in g) for g, m in zip(gs, means))
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - math.fsum(len(g) ** 2 for g in gs) / N) / (k - 1)
    s2b = (msb - msw) / n0
    if s2b + msw <= 0:
        return NotRun("zero_variance")
    return s2b / (s2b + msw)


def icc_cluster_boot(groups, B, seed):
    """ICC 95 % upper bound by the cluster bootstrap (2.3: resample whole launches with
    replacement, B times, random.Random(seed)). ub = the ceil(0.95 * n_defined)-th smallest of
    the replicates where icc_oneway is defined, floored at 0. Returns {"icc" (point estimate),
    "ub", "B", "n_defined", "seed"}; the point estimate or every replicate undefined ->
    NotRun."""
    gs = [list(g) for g in groups if len(g) >= 1]
    point = icc_oneway(gs)
    if isinstance(point, NotRun):
        return point
    if not isinstance(B, int) or isinstance(B, bool) or B < 1:
        raise ValueError("icc_cluster_boot: B must be an int >= 1")
    rng = random.Random(seed)
    reps = []
    for _ in range(B):
        v = icc_oneway([gs[rng.randrange(len(gs))] for _ in gs])
        if not isinstance(v, NotRun):
            reps.append(v)
    if not reps:
        return NotRun("no_defined_replicate")
    reps.sort()
    q95 = reps[int(math.ceil(0.95 * len(reps))) - 1]
    return {"icc": point, "ub": max(0.0, q95), "B": B, "n_defined": len(reps), "seed": seed}


# ---- T5 v2: exact sign-flip test on block differences (2.10) --------------------------------

SIGNFLIP_FULL_MAX = 20     # full 2^n enumeration up to here, meet-in-the-middle above


def _exact_ints(xs):
    """The numbers as integers on one common scale (Fraction is exact for floats), so sums
    compare without rounding: ties and zeros count exactly."""
    fr = []
    for x in xs:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError("sign-flip: differences must be finite numbers")
        fr.append(Fraction(x))
    den = 1
    for f in fr:
        den = den * f.denominator // math.gcd(den, f.denominator)
    return [int(f * den) for f in fr]


def _subset_sums(ints):
    sums = [0]
    for v in ints:
        sums = [s + v for s in sums] + [s - v for s in sums]
    return sums


def signflip_exact(diffs):
    """Two-sided exact sign-flip p-value of sum(d) (2.10): the share of the 2^n sign vectors s
    with |sum(s_i d_i)| >= |sum(d_i)| (zeros and ties included; exact integer arithmetic).
    n <= 20: full enumeration; larger: meet-in-the-middle (two halves of subset sums, one
    sorted), the same count. n = 8, all one sign -> 2 / 256 = 0.0078125."""
    ints = _exact_ints(diffs)
    n = len(ints)
    if n == 0:
        raise ValueError("signflip_exact: no differences")
    T = abs(sum(ints))
    if n <= SIGNFLIP_FULL_MAX:
        count = sum(1 for s in _subset_sums(ints) if abs(s) >= T)
    elif T == 0:
        count = 1 << n
    else:
        a, b = _subset_sums(ints[:n // 2]), sorted(_subset_sums(ints[n // 2:]))
        nb = len(b)
        count = 0
        for sa in a:   # sa + sb >= T or sa + sb <= -T (disjoint since T > 0)
            count += nb - bisect.bisect_left(b, T - sa)
            count += bisect.bisect_right(b, -T - sa)
    return count / float(1 << n)


def signflip_shift_ci(diffs, alpha):
    """Shift interval by inverting the sign-flip test: {delta : signflip_exact(d - delta) >
    alpha}, returned as (lo, hi) (closed; (-inf, inf) when even the most extreme sign vector
    has p > alpha, i.e. 2 / 2^n > alpha).

    Exact for n <= 20: a sign vector that flips a proper non-empty subset F counts at delta
    iff mean(d over F) and mean(d over the rest) lie on both sides of delta (inclusive), and
    every such pair of means brackets the overall mean; so p(delta) = (2 + 2 * #{pairs
    bracketing delta}) / 2^n is non-increasing away from the mean, and the bounds are order
    statistics of the pair end points. n > 20: bisection on delta with signflip_exact (the
    same monotone p), 60 steps per side."""
    _exact_ints(diffs)   # validates
    d = [float(x) for x in diffs]
    n = len(d)
    if not 0 < alpha < 1:
        raise ValueError("signflip_shift_ci: alpha must be in (0, 1)")
    if 2.0 / (1 << n) > alpha:
        return (-math.inf, math.inf)
    c = (alpha * (1 << n) - 2) / 2.0
    j = int(math.floor(c)) + 1      # p > alpha <=> number of bracketing pairs >= j
    if n <= SIGNFLIP_FULL_MAX:
        los, his = [], []
        total = math.fsum(d)
        for mask in range(1, 1 << (n - 1)):   # subsets holding index n-1 are the complements
            f = [d[i] for i in range(n - 1) if mask >> i & 1]
            sf = math.fsum(f)
            mf, mc = sf / len(f), (total - sf) / (n - len(f))
            los.append(min(mf, mc))
            his.append(max(mf, mc))
        los.sort()
        his.sort()
        return (los[j - 1], his[len(his) - j])
    mean = math.fsum(d) / n

    def p_at(delta):
        return signflip_exact([x - delta for x in d])

    def bound(far):
        inner, outer = mean, far
        for _ in range(60):
            mid = (inner + outer) / 2
            if p_at(mid) > alpha:
                inner = mid
            else:
                outer = mid
        return inner
    return (bound(min(d)), bound(max(d)))


def t5_block(diffs, alpha=0.05, min_blocks=8):
    """T5 v2 on the uncensored blocks' differences d_i = b_A,i - b_B,i (2.10): fewer than
    `min_blocks` -> NotRun("blocks_lt_<min_blocks>"); else the exact two-sided sign-flip
    p; p <= alpha -> "different" with the direction of sum(d) ("a_higher" when > 0), else
    "indistinguishable". Returns {"result", "p", "n", "alpha", "sum", "direction"}."""
    n = len(diffs)
    if n < min_blocks:
        return NotRun("blocks_lt_%d" % min_blocks)
    p = signflip_exact(diffs)
    s = math.fsum(diffs)
    out = {"result": "indistinguishable", "p": p, "n": n, "alpha": alpha, "sum": s,
           "direction": None}
    if p <= alpha:
        out.update(result="different", direction="a_higher" if s > 0 else "b_higher")
    return out


# ---- T7 slope reconciliation (2.7; P2-11 single tier) ----------------------------------------

def t7_fit(points):
    """Least-squares fit g = lam / mu + beta over the slope tiers (2.7): points are (lam, g)
    with g = 2 * T1 / L of `req_first_lag` (D and L in the same unit, so g is a pure number).
    Returns {"mu", "beta", "n"}; fewer than 2 distinct loads -> NotRun("too_few_loads"), a
    slope <= 0 -> NotRun("nonpositive_slope")."""
    pts = [(float(lam), float(g)) for lam, g in points]
    n = len(pts)
    if len({lam for lam, _g in pts}) < 2:
        return NotRun("too_few_loads")
    mx = math.fsum(lam for lam, _g in pts) / n
    my = math.fsum(g for _l, g in pts) / n
    sxx = math.fsum((lam - mx) ** 2 for lam, _g in pts)
    sxy = math.fsum((lam - mx) * (g - my) for lam, g in pts)
    a = sxy / sxx
    if a <= 0:
        return NotRun("nonpositive_slope")
    return {"mu": 1.0 / a, "beta": my - a * mx, "n": n}


def t7_single(lam, g):
    """One overloaded tier with beta fixed at -1 (2.7, P2-11 `bracket`): mu = lam / (g + 1),
    assuming deterministic arrivals and an unchanged service rate in overload (the lag grows
    as lam / mu - 1). g + 1 <= 0 -> NotRun("g_le_minus_one")."""
    if lam <= 0:
        raise ValueError("t7_single: lam must be > 0")
    if g + 1 <= 0:
        return NotRun("g_le_minus_one")
    return lam / (g + 1.0)


def count_floor_z(L, quantum=1):
    """z_1q of the count family (DESIGN-r7a 2.9): the z_d of a one-quantum difference of the
    half medians at the scale floor quantum / sqrt(12) on an L-second window,
    1 / (1.2533 * (1 / sqrt(12)) * sqrt(2 / (L // 2))); 10.705 (rounded) at L = 60. It does
    not depend on the quantum."""
    n = L // 2
    if n < 1 or quantum <= 0:
        raise ValueError("count_floor_z: L >= 2 and quantum > 0 required")
    return quantum / (ZD_MEDIAN_SE * (quantum / math.sqrt(12)) * math.sqrt(2.0 / n))


def floor_count_thresholds(thr, L, quantum=1):
    """A count-family key's thresholds (order_stat_thresholds output) with the floor of 2.9:
    thr_hi below z_1q takes z_1q and thr_lo above -z_1q takes -z_1q; `floor_bound` is true when
    either did. An uncalibrated key is returned unchanged (floor_bound false)."""
    out = dict(thr, floor_bound=False)
    if thr.get("uncalibrated"):
        return out
    z1 = count_floor_z(L, quantum)
    if thr["thr_hi"] < z1:
        out.update(thr_hi=z1, floor_bound=True)
    if thr["thr_lo"] > -z1:
        out.update(thr_lo=-z1, floor_bound=True)
    return out


def level_switch_ratio(buckets):
    """Two-level switching diagnostic of one steady window (DESIGN-r7a 2.13; the definition of
    calib-v2 drift/excess.py): SD over the seconds of the per-second medians divided by the RMS
    of each second's sampling SE of its median. buckets: per second, the list of raw values (or
    None / empty for an empty second). A second's SE = (x[hi] - x[lo]) / 3.92 over its sorted
    values, ranks lo = floor(n/2 - 1.96 sqrt(n)/2), hi = ceil(n/2 + 1.96 sqrt(n)/2) clipped to
    [0, n - 1] (distribution-free 95% interval of the median). About 1 for pure sampling noise;
    > 3 is reported, never judged. Fewer than 2 non-empty seconds -> NotRun("too_few_seconds"),
    zero RMS SE -> NotRun("zero_sampling_se")."""
    meds, ses = [], []
    for b in buckets:
        if not b:
            continue
        xs = sorted(b)
        n = len(xs)
        h = 1.96 * math.sqrt(n) / 2
        lo, hi = max(0, int(math.floor(n / 2 - h))), min(n - 1, int(math.ceil(n / 2 + h)))
        meds.append(statistics.median(xs))
        ses.append((xs[hi] - xs[lo]) / 3.92)
    if len(meds) < 2:
        return NotRun("too_few_seconds")
    rms = math.sqrt(math.fsum(s * s for s in ses) / len(ses))
    if rms == 0:
        return NotRun("zero_sampling_se")
    return statistics.pstdev(meds) / rms


# ---- settle segment S (2.6 steps 3-5) --------------------------------------------------------

SETTLE_CANDS = (0, 30, 60, 90, 120, 150, 180, 240, 300, 400)   # candidate S, seconds
SETTLE_N_MIN = 60          # backstop (2.6 step 4): a cell with fewer holds is `insufficient`
SETTLE_GRID = 120          # Simpson intervals per dimension (121 points), +-9 SD
_SETTLE_GRIDS = {}
_TOL_CACHE = {}


def _chi_grid(k, ngrid):
    """Simpson nodes and weights x density of R = sqrt(chi2_k / k) on +-9 SD."""
    key = (k, ngrid)
    if key not in _SETTLE_GRIDS:
        sd = 1 / math.sqrt(2 * k)
        lo, hi = max(1e-9, 1 - 9 * sd), 1 + 9 * sd
        h = (hi - lo) / ngrid
        c = math.log(2) + (k / 2) * math.log(k / 2) - math.lgamma(k / 2)
        out = []
        for i in range(ngrid + 1):
            r = lo + i * h
            w = 1 if i in (0, ngrid) else (4 if i % 2 else 2)
            out.append((r, w * h / 3 * math.exp(c + (k - 1) * math.log(r) - k * r * r / 2)))
        _SETTLE_GRIDS[key] = out
    return _SETTLE_GRIDS[key]


def _settle_pfail(tau, n, ngrid=SETTLE_GRID):
    """P(Z + t_{0.95, n-1} * Q > tau * R) with Z ~ N(0, 1) and Q, R iid sqrt(chi2_{n-1} /
    (n - 1)), all independent: the null probability that one candidate fails at TOL = tau /
    sqrt(n) (2.6 step 3; port of calib-v2 r5/settle_r5.py p_fail)."""
    g = _chi_grid(n - 1, ngrid)
    t = t_isf(0.05, n - 1)
    tot = 0.0
    for r, wr in g:
        inner = 0.0
        for q, wq in g:
            inner += wq * 0.5 * math.erfc((tau * r - t * q) / math.sqrt(2))
        tot += wr * inner
    return tot


def tol_cell(n, F, n_cand=10, floor=0.6):
    """TOL_cell(n, F) (2.6 step 3, r5): the smallest TOL >= `floor` with
    _settle_pfail(TOL * sqrt(n), n) <= 0.01 / (n_cand * F), by 32 bisection steps on
    TOL * sqrt(n) over [floor * sqrt(n), 40]. Cached per argument tuple. F = 60: n = 20 / 28 /
    40 / 66 / 80 / 100 -> 1.912 / 1.433 / 1.101 / 0.795 / 0.708 / 0.622, n >= 107 -> 0.6."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("tol_cell: n must be an int >= 2")
    key = (n, F, n_cand, floor)
    if key in _TOL_CACHE:
        return _TOL_CACHE[key]
    a = 0.01 / (n_cand * F)
    if _settle_pfail(floor * math.sqrt(n), n) <= a:
        _TOL_CACHE[key] = floor
        return floor
    lo, hi = floor * math.sqrt(n), 40.0
    for _ in range(32):
        m = (lo + hi) / 2
        if _settle_pfail(m, n) > a:
            lo = m
        else:
            hi = m
    _TOL_CACHE[key] = hi / math.sqrt(n)
    return _TOL_CACHE[key]


def excess_kurtosis(values):
    """Sample excess kurtosis m4 / m2^2 - 3 (moment estimator; descriptive only, 2.6 step 3
    `kurt_ref` / `kurt_400`). Fewer than 4 values -> NotRun("too_few_values"); zero variance
    -> NotRun("zero_variance")."""
    xs = [float(v) for v in values]
    n = len(xs)
    if n < 4:
        return NotRun("too_few_values")
    m = math.fsum(xs) / n
    m2 = math.fsum((x - m) ** 2 for x in xs) / n
    if m2 == 0:
        return NotRun("zero_variance")
    m4 = math.fsum((x - m) ** 4 for x in xs) / n
    return m4 / (m2 * m2) - 3.0


def _settle_chain(z_by_cand, ref, cands, tol):
    """2.6 steps 3 and 5 for one cell at a given TOL: per candidate mean m_S, SD s_S, one-sided
    95 % upper bound UB_S = m_S + t_{0.95, n-1} * s_S / sqrt(n), pass <=> UB_S <= TOL * s_ref
    (s_ref = SD of z(ref)); tested in fixed order from the largest candidate down, the first
    failure stops; S_cell = the earliest candidate of the unbroken passing run, "unconfirmed"
    when the largest one fails. Untested candidates keep pass = None."""
    n = len(ref)
    t = t_isf(0.05, n - 1)
    s_ref = statistics.stdev(ref)
    per = {}
    for S in cands:
        zs = z_by_cand[S]
        m, s = math.fsum(zs) / n, statistics.stdev(zs)
        per[S] = {"m": m, "s": s, "ub": m + t * s / math.sqrt(n), "pass": None}
    s_cell = "unconfirmed"
    for S in sorted(cands, reverse=True):
        ok = per[S]["ub"] <= tol * s_ref
        per[S]["pass"] = ok
        if not ok:
            break
        s_cell = S
    return {"s_ref": s_ref, "candidates": per, "s_cell": s_cell}


def settle_rule(cells, cands, F):
    """2.6 steps 3-5 over the cells of one family. cells: {cell_id: {"z": {S: [z_i(S)]},
    "ref": [z_i(ref)]}} with one value per remaining hold tier (the same n in every list; the
    exclusions of step 2 are done by the caller); cands: the candidate S values (every one a
    key of "z"); F: family size (step 3). Per cell: n, tol = tol_cell(n, F, len(cands)),
    s_ref, per-candidate m / s / ub / pass, s_cell (int or "unconfirmed"), `insufficient`
    (n < SETTLE_N_MIN: step 4's backstop decides; s_cell is still computed and is not final),
    kurt_ref and kurt_<largest candidate> (excess kurtosis, descriptive; None when not
    computable). n < 2 -> s_cell None with not_run "n_lt_2". S_reg (step 6) is the caller's."""
    cands = sorted(cands)
    if not cands or len(set(cands)) != len(cands):
        raise ValueError("settle_rule: cands must be distinct and non-empty")
    out = {"F": F, "cands": cands, "n_min": SETTLE_N_MIN, "cells": {}}
    top = cands[-1]
    for cid, cell in sorted(cells.items()):
        ref = list(cell["ref"])
        zc = {S: list(cell["z"][S]) for S in cands}
        n = len(ref)
        if any(len(v) != n for v in zc.values()):
            raise ValueError("settle_rule: cell %r has lists of different lengths" % (cid,))
        rec = {"n": n, "insufficient": n < SETTLE_N_MIN}
        if n < 2:
            rec.update(s_cell=None, not_run="n_lt_2")
            out["cells"][cid] = rec
            continue
        rec["tol"] = tol_cell(n, F, len(cands))
        rec.update(_settle_chain(zc, ref, cands, rec["tol"]))
        for name, vals in (("kurt_ref", ref), ("kurt_%s" % top, zc[top])):
            k = excess_kurtosis(vals)
            rec[name] = None if isinstance(k, NotRun) else k
        out["cells"][cid] = rec
    return out


# ---- shortfall top-up (4.5) ------------------------------------------------------------------

def shortfall_topup(d, lost, ended, mult=1.25):
    """Launches (blocks) to register for a shortfall of d (4.5): ceil(mult * d / (1 - q)) + 1,
    q = min(0.5, max(0.05, lost / ended)), q = 0.05 when ended = 0; exact rational arithmetic.
    d <= 0 -> 0. (3, 2, 40) -> 5; (10, 40, 40) -> 26; (4, 0, 0) -> 7."""
    for name, v in (("d", d), ("lost", lost), ("ended", ended)):
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("shortfall_topup: %s must be an int" % name)
    if lost < 0 or ended < 0 or lost > ended:
        raise ValueError("shortfall_topup: need 0 <= lost <= ended")
    if d <= 0:
        return 0
    q = Fraction(lost, ended) if ended else Fraction(1, 20)
    q = min(Fraction(1, 2), max(Fraction(1, 20), q))
    m = Fraction(str(mult)) if isinstance(mult, float) else Fraction(mult)
    return int(math.ceil(m * d / (1 - q))) + 1

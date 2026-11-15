# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibration v2 statistics primitives (calib-v2 DESIGN-r7a 3.5 tests 1, 2, 3, the stats
part of 8, and 11). Fixed seeds; expected numbers come from the design text and the
read-only reference outputs it cites (CV/r4/tdist.py, CV/r5/out/settle_r5.json,
CV/r3/out/settle_r3.txt)."""

import itertools
import math
import random
import statistics
import zlib
from fractions import Fraction

import pytest
from conftest import judge_mod

S = judge_mod("stats")
NotRun = S.NotRun
CANDS = S.SETTLE_CANDS


# ---- test 1: z_d calibration on iid N(0, 1) -------------------------------------------------

def _zd_sd(L, p_empty, seed, reps=20000):
    rng = random.Random(seed)
    zs, uneven = [], 0
    for _ in range(reps):
        s = [None if rng.random() < p_empty else rng.gauss(0, 1) for _ in range(L)]
        o = S.zd(s)
        if isinstance(o, NotRun):
            continue
        zs.append(o["z"])
        uneven += o["n1"] != o["n2"]
    return statistics.pstdev(zs), statistics.fmean(zs), uneven, len(zs)


def test_zd_is_calibrated_on_white_noise_at_the_registered_window():
    # L = 60 (the registered window), with and without random empty seconds (n1 != n2)
    sd, mean, uneven, n = _zd_sd(60, 0.0, 601)
    assert abs(sd - 1.0) <= 0.05 and abs(mean) < 0.03 and n == 20000
    sd, mean, uneven, n = _zd_sd(60, 0.1, 602)
    assert abs(sd - 1.0) <= 0.05 and abs(mean) < 0.03 and uneven > 10000


def test_zd_short_window_sd_follows_the_design_property():
    # DESIGN-r7a 2.2 "L >= 14: sd 1.085 - 1.175" (the median-of-halves form at small L); the
    # 3.5 test-1 line "sd 1.00 +- 0.05" holds at L = 60 only (errata calib-v2-A1)
    sd, _m, _u, _n = _zd_sd(20, 0.0, 201)
    assert 1.05 <= sd <= 1.20
    sd, _m, uneven, _n = _zd_sd(20, 0.1, 202)
    assert 1.05 <= sd <= 1.20 and uneven > 0


# ---- test 2: degenerate inputs ---------------------------------------------------------------

def test_zd_constant_series_is_zero():
    o = S.zd([5.0] * 60)
    assert o == {"z": 0.0, "D": 0.0, "sigma": 0.0, "n1": 30, "n2": 30}


def test_zd_mad_zero_falls_back_to_the_sd():
    s = [0.0] * 59 + [6.0]          # 58 zero differences and one 6: MAD 0, SD > 0
    o = S.zd(s)
    d = [b - a for a, b in zip(s, s[1:])]
    sig = statistics.pstdev(d) / math.sqrt(2)
    assert o["sigma"] == pytest.approx(sig) and sig > 0
    assert o["D"] == 0 and o["z"] == 0
    s2 = [0.0] * 40 + [1.0] * 20   # D = 1 with MAD 0
    o2 = S.zd(s2)
    sig2 = statistics.pstdev([b - a for a, b in zip(s2, s2[1:])]) / math.sqrt(2)
    assert o2["z"] == pytest.approx(1.0 / (1.2533 * sig2 * math.sqrt(2.0 / 30)))


def test_zd_too_few_differences_and_degenerate_scale():
    assert S.zd([1.0, None] * 30) == NotRun("too_few_diffs")
    assert S.zd([1.0]) == NotRun("too_few_diffs")
    # 3 differences of 6 seconds (>= L / 2), all 0, but the halves differ: no scale
    assert S.zd([1.0, 1.0, 1.0, None, 5.0, 5.0]) == NotRun("degenerate_scale")
    # the count quantum gives it a scale
    o = S.zd([1.0, 1.0, 1.0, None, 5.0, 5.0], quantum=1)
    assert o["sigma"] == pytest.approx(1 / math.sqrt(12)) and o["z"] > 0


def test_zd_count_floor_one_quantum_is_10_705():
    o = S.zd([0] * 30 + [1] * 30, quantum=1)
    assert o["D"] == 1 and round(o["z"], 3) == 10.705
    assert S.zd([0] * 60, quantum=1)["z"] == 0.0
    assert round(S.zd([0] * 30 + [1] * 30)["z"], 3) != 10.705   # without the floor


# ---- test 3: numeric expectations ------------------------------------------------------------

def test_kmax_neff_ranks():
    assert [S.kmax(n, 0.01) for n in (298, 299, 600, 720)] == [0, 1, 2, 3]
    assert S.kmax(0, 0.01) == 0
    assert S.n_eff(450, 6, 0.1) == 300 and S.n_eff(449, 6, 0.1) == 299   # deff 1.5
    assert S.n_eff(450, 6, -0.2) == 450          # ICC_ub floored at 0
    assert S.rank_for_k(1, 449, 299) == 2 and S.rank_for_k(1, 748, 299) == 3
    assert S.kmax(600, 0.01) == 2 and S.rank_for_k(2, 1500, 600) == 5
    assert S.rank_for_k(0, 449, 299) == 0
    assert S.fold_rank(2, 450, 444) == 2 and S.fold_rank(2, 450, 10) == 1
    with pytest.raises(ValueError):
        S.n_eff(10, 0.5, 0.1)


def test_order_stat_thresholds_two_sided_by_rank():
    vals = list(range(1, 450))                  # 449 values 1..449
    r = S.order_stat_thresholds(vals, 299, 0.01)
    assert (r["k"], r["r"], r["thr_hi"], r["thr_lo"]) == (1, 2, 448, 2)
    assert r["uncalibrated"] is False
    u = S.order_stat_thresholds(vals, 298, 0.01)
    assert u["uncalibrated"] is True and u["thr_hi"] is None and u["k"] == 0
    with pytest.raises(ValueError):
        S.order_stat_thresholds(vals, 450, 0.01)    # n_eff > n_windows
    with pytest.raises(ValueError):
        S.order_stat_thresholds([1.0, None], 1, 0.01)


def test_icc_oneway_by_hand():
    # means 1.5 / 3.5, MSB 4, MSW 0.5, n0 2: s2b 1.75, ICC 1.75 / 2.25
    assert S.icc_oneway([[1, 2], [3, 4]]) == pytest.approx(1.75 / 2.25)
    assert S.icc_oneway([[1, 3], [1, 3]]) == pytest.approx(-1.0)     # no floor here
    # unequal sizes: n0 = (N - sum n_i^2 / N) / (k - 1)
    g = [[1.0, 2.0, 3.0], [5.0], [2.0, 4.0]]
    N, k = 6, 3
    grand = sum(map(sum, g)) / N
    msb = sum(len(x) * (statistics.mean(x) - grand) ** 2 for x in g) / (k - 1)
    msw = sum(sum((v - statistics.mean(x)) ** 2 for v in x) for x in g) / (N - k)
    n0 = (N - sum(len(x) ** 2 for x in g) / N) / (k - 1)
    s2b = (msb - msw) / n0
    assert S.icc_oneway(g + [[]]) == pytest.approx(s2b / (s2b + msw))
    assert S.icc_oneway([[1, 2, 3]]) == NotRun("too_few_groups_or_values")
    assert S.icc_oneway([[1], [2]]) == NotRun("too_few_groups_or_values")
    assert S.icc_oneway([[2, 2], [2, 2]]) == NotRun("zero_variance")


def test_icc_cluster_boot_upper_bound():
    rng = random.Random(15)
    groups = []
    for _ in range(60):                       # true ICC 0.3: launch effect var 0.3, noise 0.7
        u = rng.gauss(0, math.sqrt(0.3))
        groups.append([u + rng.gauss(0, math.sqrt(0.7)) for _ in range(6)])
    r = S.icc_cluster_boot(groups, 400, 5)
    assert 0.1 < r["icc"] < 0.5 and r["ub"] > r["icc"] and r["n_defined"] == 400
    assert S.icc_cluster_boot(groups, 400, 5) == r            # same seed, same result
    assert S.icc_cluster_boot(groups, 400, 6)["ub"] != r["ub"]
    # replicates by hand: the ceil(0.95 * B)-th smallest
    rng2, reps = random.Random(5), []
    for _ in range(400):
        reps.append(S.icc_oneway([groups[rng2.randrange(60)] for _ in groups]))
    assert r["ub"] == sorted(reps)[379]
    # negative ICC: the upper bound is floored at 0
    flat = [[0.0, 2.0], [0.1, 1.9], [0.2, 1.8], [-0.1, 2.1], [0.0, 2.0]]
    f = S.icc_cluster_boot(flat, 200, 1)
    assert f["icc"] < 0 and f["ub"] == 0.0
    assert S.icc_cluster_boot([[1, 2]], 10, 1) == NotRun("too_few_groups_or_values")


def test_t_quantiles_match_the_reference():
    assert round(S.t_isf(0.05, 19), 4) == 1.7291
    assert round(S.t_isf(0.01 / 60, 19), 4) == 4.3649
    assert S.t_sf(S.t_isf(0.05, 19), 19) == pytest.approx(0.05, rel=1e-9)
    assert S.t_isf(0.95, 19) == pytest.approx(-S.t_isf(0.05, 19))
    # past the old fixed bracket [0, 1000]: the bracket grows instead of clipping at 1000
    assert S.t_isf(1e-4, 1) == pytest.approx(1 / math.tan(math.pi * 1e-4), rel=1e-9)  # 3183.1
    assert round(S.t_isf(1e-4, 1), 1) == 3183.1
    with pytest.raises(ValueError, match="above"):
        S.t_isf(1e-300, 1)


# CV/r5/out/settle_r5.json TOL_table (TOL_new; the design quotes 3 decimals, +-0.001)
TOL_60 = {20: 1.912, 28: 1.433, 40: 1.101, 66: 0.795, 80: 0.708, 100: 0.622}


def test_tol_cell_table_and_floor():
    for n, tol in TOL_60.items():
        assert abs(S.tol_cell(n, 60) - tol) <= 0.001, n
    assert S.tol_cell(107, 60) == 0.6 and S.tol_cell(140, 60) == 0.6
    assert S.tol_cell(106, 60) > 0.6          # the floor binds from n = 107 (F = 60)
    assert abs(S.tol_cell(60, 90) - 0.860) <= 0.001
    assert S.tol_cell(110, 90) == 0.6 and S.tol_cell(109, 90) > 0.6
    # the old r4 formula (sample SDs taken as equal) gives 1.098 at n = 28: a different rule
    old = max(0.6, (S.t_isf(0.05, 27) + S.t_isf(0.01 / 60, 27)) / math.sqrt(28))
    assert round(old, 3) == 1.098 and S.tol_cell(28, 60) > old + 0.3
    with pytest.raises(ValueError):
        S.tol_cell(1, 60)


@pytest.mark.parametrize("n, F", [(20, 60), (66, 60), (40, 90)])
def test_tol_cell_meets_the_per_candidate_bound_and_the_grid_is_fine_enough(n, F):
    target = 0.01 / (10 * F)
    tau = S.tol_cell(n, F) * math.sqrt(n)
    p120 = S._settle_pfail(tau, n)
    assert abs(p120 / target - 1) <= 0.01
    assert abs(S._settle_pfail(tau, n, ngrid=240) / p120 - 1) <= 1e-6   # grid doubled
    # one bisection step below TOL the bound is broken: the bisection found the edge
    assert S._settle_pfail(tau * 0.999, n) > target


def test_tol_cell_is_cached_per_arguments():
    a = S.tol_cell(33, 60)
    assert S._TOL_CACHE[(33, 60, 10, 0.6)] == a
    assert S.tol_cell(33, 60, n_cand=1) < a       # fewer candidates, looser bound


def test_excess_kurtosis_normal_and_heavy_tail():
    rng = random.Random(7)
    normal = [rng.gauss(0, 1) for _ in range(2000)]
    assert abs(S.excess_kurtosis(normal)) <= 0.25
    # t5: normal / sqrt(chi2_5 / 5); theoretical excess kurtosis 6
    t5 = [rng.gauss(0, 1) / math.sqrt(sum(rng.gauss(0, 1) ** 2 for _ in range(5)) / 5)
          for _ in range(2000)]
    assert S.excess_kurtosis(t5) > 1.0
    assert S.excess_kurtosis([1, 2, 3]) == NotRun("too_few_values")
    assert S.excess_kurtosis([2, 2, 2, 2]) == NotRun("zero_variance")


def test_shortfall_topup():
    assert S.shortfall_topup(3, 2, 40) == 5          # q = 0.05 (2 / 40 = 0.05)
    assert S.shortfall_topup(10, 40, 40) == 26       # q clipped to 0.5
    assert S.shortfall_topup(4, 0, 0) == 7           # ended = 0 -> q = 0.05
    assert S.shortfall_topup(4, 1, 100) == 7         # 0.01 -> 0.05
    assert S.shortfall_topup(4, 20, 100) == 8        # q = 0.2: ceil(5 / 0.8) + 1
    assert S.shortfall_topup(2, 1, 4) == 5           # q = 0.25: 2.5 / 0.75 = 3.33 -> 4 + 1
    assert S.shortfall_topup(0, 3, 9) == 0
    assert S.shortfall_topup(4, 0, 0, mult=1) == 6   # ceil(4 / 0.95) + 1
    for bad in ((1, 5, 4), (1, -1, 3), (1.0, 0, 0)):
        with pytest.raises(ValueError):
            S.shortfall_topup(*bad)


def test_shortfall_q_is_the_q_hat_of_topup_and_the_product():
    """RS15: one definition of q_hat; pilot2 shortfall records float(shortfall_q)."""
    assert S.shortfall_q(2, 40) == Fraction(1, 20) and S.shortfall_q(3, 8) == Fraction(3, 8)
    assert S.shortfall_q(40, 40) == Fraction(1, 2) and S.shortfall_q(0, 0) == Fraction(1, 20)
    assert S.shortfall_q(1, 100) == Fraction(1, 20)
    for bad in ((5, 4), (-1, 3), (1.0, 2), (True, 2)):
        with pytest.raises(ValueError):
            S.shortfall_q(*bad)


# ---- test 11: sign-flip test, shift interval, T5 v2 ------------------------------------------

def _brute_p(diffs):
    fr = [Fraction(x) for x in diffs]
    T = abs(sum(fr))
    hits = sum(1 for signs in itertools.product((1, -1), repeat=len(fr))
               if abs(sum(s * x for s, x in zip(signs, fr))) >= T)
    return hits / 2 ** len(fr)


@pytest.mark.parametrize("d", [[1, -1, 0, 2, 2], [0, 0, 0], [3, 3, 3, -3], [0.1, 0.2, -0.3],
                               [1.5, -0.5, 0.5, 2.5, -2.5, 0], [4], [2, 2, -2, -2, 1, 1]])
def test_signflip_exact_matches_hand_enumeration(d):
    assert S.signflip_exact(d) == _brute_p(d)


def test_signflip_small_cases_by_hand():
    assert S.signflip_exact([0, 0, 0]) == 1.0                 # every vector ties
    assert S.signflip_exact([1, 1]) == 0.5                    # (+,+), (-,-) of 4
    assert S.signflip_exact([1, -1]) == 1.0                   # sum 0
    assert S.signflip_exact([2, 1, 0]) == 0.5                 # |+-3| x 2 zeros of 8


def test_signflip_meet_in_the_middle_equals_full_enumeration(monkeypatch):
    rng = random.Random(11)
    cases = [[rng.choice((-3, -1, 0, 1, 2, 2, 5)) for _ in range(12)] for _ in range(5)]
    cases.append([rng.gauss(0.5, 1) for _ in range(13)])
    full = [S.signflip_exact(d) for d in cases]
    monkeypatch.setattr(S, "SIGNFLIP_FULL_MAX", 4)
    assert [S.signflip_exact(d) for d in cases] == full
    assert S.signflip_exact([1, -1] * 6) == 1.0


def test_signflip_p_is_near_uniform_under_no_difference():
    rng = random.Random(12)
    ps = [S.signflip_exact([rng.gauss(0, 1) for _ in range(10)]) for _ in range(2000)]
    # exact level at alpha 0.05 for n = 10 is 50 / 1024 = 0.0488; binomial 3-SD band
    assert abs(sum(p <= 0.05 for p in ps) / 2000 - 0.0488) <= 3 * math.sqrt(0.05 * 0.95 / 2000)
    assert abs(sum(p <= 0.5 for p in ps) / 2000 - 0.5) <= 3 * math.sqrt(0.25 / 2000)


def test_shift_interval_edges_are_where_the_test_flips():
    rng = random.Random(13)
    for _ in range(5):
        d = [rng.gauss(1, 2) for _ in range(7)]
        lo, hi = S.signflip_shift_ci(d, 0.05)
        eps = 1e-9 * (1 + abs(lo) + abs(hi))
        assert S.signflip_exact([x - lo for x in d]) > 0.05
        assert S.signflip_exact([x - (lo - eps) for x in d]) <= 0.05
        assert S.signflip_exact([x - hi for x in d]) > 0.05
        assert S.signflip_exact([x - (hi + eps) for x in d]) <= 0.05
        assert lo <= statistics.fmean(d) <= hi
    # 2 / 2^n > alpha: no shift can be rejected
    assert S.signflip_shift_ci([1, 2, 3, 4], 0.05) == (-math.inf, math.inf)


def test_shift_interval_bisection_path_agrees(monkeypatch):
    d = [0.3, -1.2, 2.5, 0.7, 1.1, -0.4, 3.0, 0.9]
    exact = S.signflip_shift_ci(d, 0.05)
    monkeypatch.setattr(S, "SIGNFLIP_FULL_MAX", 4)
    approx = S.signflip_shift_ci(d, 0.05)
    assert approx[0] == pytest.approx(exact[0], abs=1e-9)
    assert approx[1] == pytest.approx(exact[1], abs=1e-9)


def test_shift_interval_covers_95_percent():
    rng = random.Random(14)
    reps, hit = 4000, 0
    for _ in range(reps):
        d = [rng.gauss(0.3, 1) for _ in range(10)]
        lo, hi = S.signflip_shift_ci(d, 0.05)
        hit += lo <= 0.3 <= hi
    assert abs(hit / reps - 0.95) <= 0.01


def test_t5_block_needs_eight_blocks():
    assert S.t5_block([1.0] * 7) == NotRun("blocks_lt_8")
    assert S.t5_block([1.0] * 3, min_blocks=4) == NotRun("blocks_lt_4")
    r = S.t5_block([1.0] * 8)
    assert r["p"] == pytest.approx(0.0078, abs=5e-5) and r["p"] == 2 / 256
    assert (r["result"], r["direction"], r["n"]) == ("different", "a_higher", 8)
    r = S.t5_block([-0.5] * 8)
    assert (r["result"], r["direction"]) == ("different", "b_higher")
    r = S.t5_block([1, -1, 2, -2, 0.5, -0.5, 1, -1])
    assert (r["result"], r["direction"], r["p"]) == ("indistinguishable", None, 1.0)


# ---- T7 --------------------------------------------------------------------------------------

def test_t7_fit_recovers_mu_and_beta_from_linear_lag_growth():
    mu = 1000.0
    pts = []
    for rho in (1.05, 1.10, 1.20):
        lam = rho * mu
        lag = [(lam / mu - 1) * t for t in range(60)]      # deterministic arrivals, overload
        D, _n1, _n2 = S.t1_trend(lag)
        pts.append((lam, 2 * D / 60))
    r = S.t7_fit(pts)
    assert abs(r["mu"] / mu - 1) < 0.005 and abs(r["beta"] + 1) < 0.01 and r["n"] == 3
    assert S.t7_fit([(1.0, 0.1), (1.0, 0.2)]) == NotRun("too_few_loads")
    assert S.t7_fit([(1.0, 0.3), (2.0, 0.1)]) == NotRun("nonpositive_slope")


def test_t7_single_beta_fixed():
    assert S.t7_single(1050.0, 0.05) == pytest.approx(1000.0)
    assert S.t7_single(10.0, -1.0) == NotRun("g_le_minus_one")
    with pytest.raises(ValueError):
        S.t7_single(0.0, 0.1)


# ---- test 8 (stats part): the settle rule on synthetic hold tiers ---------------------------
# Series generator and seeds as CV/r3/settle_r3.py (600 s hold, white / AR(1) / exponential
# approach); 50 cells of n = 140 per scenario (tol_cell(140, 60) = 0.6, the r3 TOL).

H, L60 = 600, 60
SCEN = [("white", "white", {}), ("ar0.5", "white", {"phi": 0.5}),
        ("ar0.9", "white", {"phi": 0.9}),
        ("exp_tau40_A5", "exp", {"A": 5.0, "tau": 40.0}),
        ("exp_tau60_A10", "exp", {"A": 10.0, "tau": 60.0}),
        ("exp_tau100_A20", "exp", {"A": 20.0, "tau": 100.0})]


def _series(rng, kind, p):
    x, out = 0.0, []
    phi = p.get("phi", 0.0)
    innov = math.sqrt(1 - phi * phi)
    for t in range(H):
        x = phi * x + innov * rng.gauss(0, 1) if phi else rng.gauss(0, 1)
        f = p["A"] * (1 - math.exp(-t / p["tau"])) if kind == "exp" else 0.0
        out.append(f + x)
    return out


def _r3_cell(z, ref, n, tol=0.6):
    """CV/r3/settle_r3.py `cell` on the same z values (Cornish-Fisher t95)."""
    df, q = n - 1, 1.6448536
    t95 = q + (q ** 3 + q) / (4 * df) + (5 * q ** 5 + 16 * q ** 3 + 3 * q) / (96 * df ** 2)
    sref, ok_from = statistics.stdev(ref), None
    for c in reversed(CANDS):
        ub = statistics.mean(z[c]) + t95 * statistics.stdev(z[c]) / math.sqrt(n)
        if ub <= tol * sref:
            ok_from = c
        else:
            break
    return "unconfirmed" if ok_from is None else ok_from


def _true_s(A, tau, tol=0.6):
    """CV/r3 analytic truth: the earliest candidate from which the expected shift <= TOL."""
    sd_ref = 1.2533 * math.sqrt(2.0 / 30) * 1.085
    true = None
    for c in reversed(CANDS):
        w = 4 + c
        D = A * math.exp(-w / tau) * (math.exp(-15 / tau) - math.exp(-45 / tau))
        if D / sd_ref > tol:
            break
        true = c
    return true


def _settle_scenario(name, kind, p, n=140, reps=50):
    rng = random.Random(zlib.crc32(("%s|%d" % (name, n)).encode()))
    rows = []
    for _ in range(reps):
        z, ref = {c: [] for c in CANDS}, []
        for _ in range(n):
            s = _series(rng, kind, p)
            for c in CANDS:
                z[c].append(S.zd(s[4 + c:4 + c + L60])["z"])
            ref.append(S.zd(s[H - L60:H])["z"])
        cell = S.settle_rule({"c": {"z": z, "ref": ref}}, CANDS, 60)["cells"]["c"]
        rows.append((cell, _r3_cell(z, ref, n), S._settle_chain(z, ref, CANDS, 0.1)["s_cell"]))
    return rows


@pytest.fixture(scope="module")
def settle_runs():
    return {name: _settle_scenario(name, kind, p) for name, kind, p in SCEN}


def test_settle_negative_samples_give_zero(settle_runs):
    for name in ("white", "ar0.5", "ar0.9"):
        cells = [r[0] for r in settle_runs[name]]
        assert all(c["tol"] == 0.6 and c["n"] == 140 and not c["insufficient"] for c in cells)
        assert sum(c["s_cell"] == 0 for c in cells) / len(cells) >= 0.98, name
        assert not any(c["s_cell"] == "unconfirmed" for c in cells), name


def test_settle_positive_samples_reach_the_analytic_truth(settle_runs):
    for name, _k, p in SCEN[3:]:
        truth = _true_s(p["A"], p["tau"])
        got = [r[0]["s_cell"] for r in settle_runs[name]]
        num = sorted(400 if s == "unconfirmed" else s for s in got)
        assert num[len(num) // 2] >= truth, name           # median at or after the truth
        # (100 s, 20 sigma): the truth 400 sits on the edge (expected shift at 300 is 1.01
        # TOL), so one cell in 50 may stop at 300; unconfirmed is allowed there
        assert sum(s >= truth for s in num) / len(num) >= 0.95, name
        if name != "exp_tau100_A20":
            assert "unconfirmed" not in got


def test_settle_matches_the_r3_rule_on_the_same_seeds(settle_runs):
    # regression baseline: on the same series the rule gives r3's S_cell cell by cell
    for name in settle_runs:
        assert [r[0]["s_cell"] for r in settle_runs[name]] == [r[1] for r in settle_runs[name]]


def test_settle_with_tol_0_1_never_accepts_zero_on_negative_samples(settle_runs):
    # positive control: an over-strict TOL stops every negative cell early (most at 400)
    for name in ("white", "ar0.5", "ar0.9"):
        tight = [r[2] for r in settle_runs[name]]
        assert 0 not in tight and sum(s == "unconfirmed" for s in tight) >= len(tight) // 2


def _chol_cands():
    """CV/r5/settle_r5.py chol_cands: the 10 candidate windows' trend proxies (second-half
    mean minus first-half mean of white noise) are correlated by overlap; Cholesky factor."""
    w = []
    for c in CANDS:
        a, v = 4 + c, [0.0] * 600
        for s in range(a, a + 30):
            v[s] = -1 / 30
        for s in range(a + 30, a + 60):
            v[s] = 1 / 30
        w.append(v)
    var = 60 / 900
    cov = [[sum(x * y for x, y in zip(wi, wj)) / var for wj in w] for wi in w]
    m = len(cov)
    low = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1):
            s = cov[i][j] - sum(low[i][k] * low[j][k] for k in range(j))
            low[i][j] = math.sqrt(s) if i == j else s / low[j][j]
    return low


def test_settle_small_n_negative_meets_the_bound_and_the_old_formula_does_not():
    # n = 28, F = 60, 400 null cells: new rule (tol_cell 1.433) <= 1 cell with S_cell > 0;
    # the r4 formula 1.098 on the same cells is the positive control (about 2.6 expected)
    low, m, n = _chol_cands(), len(CANDS), 28
    rng = random.Random(20261115)
    new = old = 0
    for _ in range(400):
        zs, ref = [[0.0] * n for _ in range(m)], [rng.gauss(0, 1) for _ in range(n)]
        for i in range(n):
            e = [rng.gauss(0, 1) for _ in range(m)]
            for a in range(m):
                zs[a][i] = sum(low[a][b] * e[b] for b in range(a + 1))
        z = {CANDS[a]: zs[a] for a in range(m)}
        cell = S.settle_rule({"c": {"z": z, "ref": ref}}, CANDS, 60)["cells"]["c"]
        assert cell["tol"] == pytest.approx(1.433, abs=0.001) and cell["insufficient"]
        new += cell["s_cell"] != 0
        old += S._settle_chain(z, ref, CANDS, 1.098)["s_cell"] != 0
    assert new <= 1 and old >= 1


def _cell(means, sd=1.0, n=140, ref_sd=1.0):
    """A cell whose candidate z lists have the given means (exactly) and SD `sd`."""
    base = [(-1) ** i * sd * math.sqrt((n - 1) / n) for i in range(n)]   # mean 0, stdev sd
    ref = [(-1) ** i * ref_sd * math.sqrt((n - 1) / n) for i in range(n)]
    return {"z": {c: [b + means.get(c, 0.0) for b in base] for c in CANDS}, "ref": ref}


def test_settle_rule_fixed_order_upper_bound_and_flags():
    # 300 fails (mean 1.0 > 0.6), 400 passes: fixed order stops at 300 -> S_cell 400 even
    # though every earlier candidate passes ("any candidate passes" would give 0)
    r = S.settle_rule({"a": _cell({300: 1.0})}, CANDS, 60)["cells"]["a"]
    assert r["s_cell"] == 400 and r["candidates"][300]["pass"] is False
    assert r["candidates"][240]["pass"] is None and r["candidates"][400]["pass"] is True
    # the upper bound, not the point estimate: mean 0.5 < 0.6 but UB = 0.5 + t * 1 / sqrt(140)
    # = 0.639 > 0.6 fails at 400 -> unconfirmed
    r = S.settle_rule({"b": _cell({c: 0.5 for c in CANDS})}, CANDS, 60)["cells"]["b"]
    assert r["s_cell"] == "unconfirmed"
    assert r["candidates"][400]["ub"] == pytest.approx(0.5 + S.t_isf(0.05, 139) / math.sqrt(140))
    # earliest of the unbroken run
    r = S.settle_rule({"c": _cell({0: 2.0, 30: 2.0})}, CANDS, 60)["cells"]["c"]
    assert r["s_cell"] == 60 and r["tol"] == 0.6 and r["kurt_ref"] is not None
    assert "kurt_400" in r and r["insufficient"] is False


def test_settle_rule_small_cells_and_bad_input():
    r = S.settle_rule({"s": _cell({}, n=59), "t": _cell({}, n=60)}, CANDS, 60)["cells"]
    assert r["s"]["insufficient"] is True and r["t"]["insufficient"] is False
    assert r["s"]["tol"] == pytest.approx(S.tol_cell(59, 60))
    one = {"z": {c: [0.1] for c in CANDS}, "ref": [0.2]}
    assert S.settle_rule({"u": one}, CANDS, 60)["cells"]["u"] == {
        "n": 1, "insufficient": True, "s_cell": None, "not_run": "n_lt_2"}
    bad = _cell({})
    bad["z"][30] = bad["z"][30][:-1]
    with pytest.raises(ValueError):
        S.settle_rule({"x": bad}, CANDS, 60)
    with pytest.raises(KeyError):
        S.settle_rule({"y": {"z": {}, "ref": [1, 2]}}, CANDS, 60)

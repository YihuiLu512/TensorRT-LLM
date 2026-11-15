# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T1-T6 against hand-computed values (design 2.6 table)."""

from conftest import judge_mod

S = judge_mod("stats")


def test_t1_half_split_median_difference():
    assert S.t1_trend([1, 2, 3, 10, 20, 30]) == (20 - 2, 3, 3)
    # odd length: the middle second belongs to neither half
    assert S.t1_trend([1, 1, 100, 5, 5]) == (4, 2, 2)
    # empty seconds are skipped inside each half, the split stays by time
    assert S.t1_trend([None, 2, 4, None]) == (2, 1, 1)
    assert S.t1_trend([None, None, 3, 4]) is None
    assert S.t1_trend([7]) is None


def test_t2_offset_vs_baseline():
    assert S.t2_offset([5, 6, 7], [1, 2, 3]) == 4
    assert S.t2_offset([], [1]) is None


def test_t3_band_and_compare():
    b = S.t3_band([3, 1, 2, None])
    assert b == {"min": 1, "max": 3, "n": 3}
    assert S.compare_to_band(3, b) == "inside" and S.compare_to_band(1, b) == "inside"
    assert S.compare_to_band(3.0001, b) == "above"
    assert S.compare_to_band(0, b) == "below"
    assert S.compare_to_band(None, b) is None
    assert abs(S.nominal_false_alarm(b) - 0.25) < 1e-12
    assert S.t3_band([]) is None
    assert not S.band_is_valid({"min": 2, "max": 1, "n": 1})
    assert not S.band_is_valid({"min": 0, "max": 1, "n": True})


def _t(seq, key, verdict):
    return {"seq": seq, "order_key": key, "load_value": key, "verdict": verdict}


def test_t4_boundary_interval_and_lower_bound():
    r = S.t4_boundary([_t(1, 10, "STABLE"), _t(2, 20, "STABLE"), _t(3, 40, "UNSTABLE"),
                       _t(4, 80, "UNSTABLE"), _t(5, 30, "VOID")])
    assert (r["lower"], r["upper"], r["kind"], r["non_monotone"]) == (20, 40, "interval", False)
    r = S.t4_boundary([_t(1, 10, "STABLE"), _t(2, 20, "STABLE")])
    assert r["kind"] == "lower_bound_only" and r["upper"] is None
    r = S.t4_boundary([_t(1, 10, "UNSTABLE"), _t(2, 20, "STABLE")])
    assert r["non_monotone"] is True
    assert S.t4_boundary([_t(1, 10, "NOT_RUN")])["kind"] == "none"


def test_t4_non_monotone_counts_undetermined_tiers_where_x_held():
    # b2-judge FU-4: an UNDETERMINED tier (X held) below the highest stable tier
    r = S.t4_boundary([_t(1, 10, "STABLE"), _t(2, 20, "UNDETERMINED"), _t(3, 30, "STABLE")])
    assert r["non_monotone"] is True
    assert (r["lower_seq"], r["upper_seq"], r["kind"]) == (3, None, "lower_bound_only")
    # VOID / NOT_RUN tiers below a stable tier: X was not established there
    for v in ("VOID", "NOT_RUN"):
        r = S.t4_boundary([_t(1, 10, "STABLE"), _t(2, 20, v), _t(3, 30, "STABLE")])
        assert r["non_monotone"] is False
    r = S.t4_boundary([_t(1, 10, "STABLE"), _t(2, 20, "UNDETERMINED"),
                       _t(3, 30, "UNSTABLE")])
    assert r["non_monotone"] is False and r["upper_seq"] == 3


def test_t4_envelope_asc_desc_and_open_end():
    e = S.t4_envelope([(20, 40), (30, 50), (10, 45)])
    assert (e["lower"], e["upper"], e["n_launches"]) == (10, 50, 3)
    # desc axis (e.g. beat_ns): numerically larger = lighter load
    e = S.t4_envelope([(40, 20), (50, 30)], order="desc")
    assert (e["lower"], e["upper"]) == (50, 20)
    assert S.t4_envelope([(20, None), (30, 50)])["upper"] is None


def test_t5_disjoint_same_direction_else_indistinguishable():
    a = {"lower": 10, "upper": 20}
    b = {"lower": 30, "upper": 40}
    r = S.t5_compare([{"block": "x", "a": a, "b": b}, {"block": "y", "a": a, "b": b}])
    assert r["result"] == "different" and r["direction"] == "b_higher"
    r = S.t5_compare([{"block": "x", "a": a, "b": b}, {"block": "y", "a": b, "b": a}])
    assert r["result"] == "indistinguishable" and r["min_resolvable_diff"] == 30
    r = S.t5_compare([{"block": "x", "a": {"lower": 10, "upper": 35}, "b": b}])
    assert r["result"] == "indistinguishable" and r["min_resolvable_diff"] == 30
    # no block with both envelopes closed: no comparison at all
    r = S.t5_compare([{"block": "x", "a": {"lower": 10, "upper": None}, "b": b}])
    assert r["result"] == "NOT_RUN" and r["min_resolvable_diff"] is None
    # one closed block and one open: the open one counts against "different"
    r = S.t5_compare([{"block": "x", "a": {"lower": 10, "upper": None}, "b": b},
                      {"block": "y", "a": a, "b": b}])
    assert r["result"] == "indistinguishable"


def test_t6_ratio_range_and_unfavourable_end():
    r = S.t6_ratio_range([("b1", 3, 2), ("b2", 4, 2), ("b3", 1, 0)])
    assert (r["min"], r["max"], r["skipped"]) == (1.5, 2.0, ["b3"])
    assert S.t6_meets_bound(r, 1.5, "higher") is True
    assert S.t6_meets_bound(r, 1.6, "higher") is False
    assert S.t6_meets_bound(r, 2.0, "lower") is True
    assert S.t6_ratio_range([("b", 1, 0)]) is None

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""X combination rules (design 2.6): X2 is the one piece of evidence, X1/X3 reconcile it."""

from conftest import judge_mod

X = judge_mod("symptoms")


def combine(x2, x1, x3, overload=0):
    out = {"reasons": [], "overload_failures": overload}
    return X._combine(out, x2, x1, x3)


def test_x2_true_consistent_is_unstable():
    r = combine(["above", "inside"], ["below"], ["above"])
    assert r["X2_status"] == "TRUE" and r["holds"] is True and not r["contradiction"]


def test_x2_false_consistent_is_stable():
    r = combine(["inside"], ["inside"], ["inside"])
    assert r["holds"] is False and not r["contradiction"]


def test_contradictions_both_directions():
    assert combine(["above"], ["above"], [])["contradiction"]
    assert combine(["above"], [], ["below"])["contradiction"]
    assert combine(["inside"], ["below"], [])["contradiction"]
    assert combine(["inside"], [], ["above"])["contradiction"]


def test_overload_failures_hold_and_suppress_truncation_contradiction():
    r = combine(["inside"], ["below"], ["inside"], overload=3)
    assert r["holds"] is True and not r["contradiction"]
    # an X2 rise contradicted by over-delivery is still a counter error
    assert combine(["above"], ["above"], [], overload=3)["contradiction"]


def test_not_run_paths():
    assert combine([], ["inside"], [])["holds"] is None  # no lag series
    assert combine(["inside", None], ["inside"], [])["holds"] is None  # a series NOT_RUN
    assert combine(["inside"], [None], [None])["holds"] is None  # no reconciliation
    # a positive trend is evidence even if another series could not be computed
    assert combine(["above", None], ["inside"], [])["holds"] is True

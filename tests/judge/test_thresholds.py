# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bands come only from calibration documents; duplicates and malformed keys are rejected."""

import pytest
from conftest import judge_mod

T = judge_mod("thresholds")


def test_lookup_records_found_and_missing():
    b = T.Bands()
    b.add_source("pilot", "p.json", "a" * 64,
                 {"noise_bands": {"T1:x_ns:capload": {"min": -1, "max": 1, "n": 3}}})
    assert b.get("T1:x_ns:capload")["max"] == 1
    assert b.get("T1:y_ns:capload") is None
    assert b.lookups == {"T1:x_ns:capload": "found", "T1:y_ns:capload": "missing"}


def test_duplicate_key_across_sources_is_an_error():
    b = T.Bands()
    doc = {"noise_bands": {"T1:x_ns:capload": {"min": -1, "max": 1, "n": 3}}}
    b.add_source("pilot", "p.json", "a" * 64, doc)
    with pytest.raises(T.BandError):
        b.add_source("loadgen_band", "l.json", "b" * 64, doc)


@pytest.mark.parametrize("key,band", [("T9:x:capload", {"min": 0, "max": 1, "n": 1}),
                                      ("T1:x", {"min": 0, "max": 1, "n": 1}),
                                      ("T1:x:capload", {"min": 2, "max": 1, "n": 1}),
                                      ("T1:x:capload", {"min": 0, "max": 1})])
def test_malformed_band_rejected(key, band):
    with pytest.raises(T.BandError):
        T.Bands().add_source("pilot", "p.json", "a" * 64, {"noise_bands": {key: band}})

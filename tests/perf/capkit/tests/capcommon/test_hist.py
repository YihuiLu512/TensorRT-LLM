# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import random

import pytest

from capcommon import hist


def test_bucket_bounds_contain_values():
    rng = random.Random(1)
    vals = [0, 1, 1023, 1024, 1025, 2047, 2048, 4095, 10 ** 6, 2 ** 34 - 1, 2 ** 34, 10 ** 12]
    vals += [rng.randrange(0, 2 ** 35) for _ in range(5000)]
    for v in vals:
        i = hist.bucket_index(v)
        lo, hi = hist.bucket_bounds(i)
        assert lo <= v and (hi is None or v < hi), (v, i, lo, hi)
    assert hist.NBINS == 194
    assert hist.bucket_index(2 ** 34) == hist.OVERFLOW
    assert hist.bucket_index(2 ** 34 - 1) == hist.OVERFLOW - 1


def test_buckets_are_contiguous_and_resolution():
    prev_hi = 0
    for i in range(hist.NBINS - 1):
        lo, hi = hist.bucket_bounds(i)
        assert lo == prev_hi
        if i > 0:
            assert (hi - lo) * 8 == 1 << (lo.bit_length() - 1)
        prev_hi = hi


def test_quantile_and_diff():
    c = hist.new_counts()
    for v in [500] * 50 + [10_000] * 50:
        hist.add(c, v)
    assert hist.quantile(c, 0.5) == 1024
    q99 = hist.quantile(c, 0.99)
    lo, hi = hist.bucket_bounds(hist.bucket_index(10_000))
    assert q99 == hi
    assert hist.quantile(hist.new_counts(), 0.5) is None
    later = list(c)
    later[0] += 3
    assert sum(hist.diff(later, c)) == 3
    with pytest.raises(ValueError):
        hist.diff(c, later)

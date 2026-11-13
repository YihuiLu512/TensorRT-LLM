# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""loghist/1 bucketing (INTERFACES.md section 6.3).

v < 1024 -> bucket 0; otherwise e = bit_length(v) - 1, o = e - 10, sub = (v >> (e - 3)) & 7,
bucket = 1 + 8*o + sub; o >= 24 (v >= 2**34 ns) -> overflow bucket 193. 194 buckets, relative
resolution 1/8. Negative values go to bucket 0 (callers count them separately).
"""

import math

NAME = "loghist/1"
SUB = 8
OCTAVES = 24
NBINS = 1 + SUB * OCTAVES + 1  # 194
OVERFLOW = NBINS - 1
_MIN_EXP = 10
_MAX_V = 1 << (_MIN_EXP + OCTAVES)  # 2**34


def bucket_index(v):
    if v < 1024:
        return 0
    if v >= _MAX_V:
        return OVERFLOW
    e = v.bit_length() - 1
    return 1 + SUB * (e - _MIN_EXP) + ((v >> (e - 3)) & 7)


def bucket_bounds(i):
    """Return (lo, hi) with lo <= v < hi for every v in bucket i (hi is None for overflow)."""
    if i == 0:
        return (0, 1024)
    if i == OVERFLOW:
        return (_MAX_V, None)
    if not 0 < i < OVERFLOW:
        raise IndexError(i)
    o, sub = divmod(i - 1, SUB)
    e = o + _MIN_EXP
    step = 1 << (e - 3)
    lo = (1 << e) + sub * step
    return (lo, lo + step)


def new_counts():
    return [0] * NBINS


def add(counts, v):
    counts[bucket_index(v)] += 1


def diff(later, earlier):
    """Element-wise later - earlier (interval distribution between two cumulative snapshots)."""
    if len(later) != NBINS or len(earlier) != NBINS:
        raise ValueError("histogram length must be %d" % NBINS)
    out = [a - b for a, b in zip(later, earlier)]
    if any(x < 0 for x in out):
        raise ValueError("cumulative histogram decreased")
    return out


def total(counts):
    return sum(counts)


def quantile(counts, q):
    """Upper bound (exclusive hi, conservative) of the bucket holding the q-quantile.

    Returns None when the histogram is empty. For the overflow bucket returns its lower bound.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    n = sum(counts)
    if n == 0:
        return None
    # rank of the q-quantile, 1-based, nearest-rank definition
    rank = max(1, math.ceil(q * n))
    acc = 0
    for i, c in enumerate(counts):
        acc += c
        if acc >= rank:
            lo, hi = bucket_bounds(i)
            return lo if hi is None else hi
    lo, hi = bucket_bounds(NBINS - 1)
    return lo

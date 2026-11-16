# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monotonic clock helpers. All capkit time arithmetic uses integer time.monotonic_ns()."""

import datetime
import time

now_ns = time.monotonic_ns
wall_ns = time.time_ns

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000
NS_PER_US = 1_000


def clock_pair():
    """Return (monotonic_ns, wall_ns) read back to back, for recording their correspondence."""
    m = time.monotonic_ns()
    w = time.time_ns()
    return m, w


def wall_iso(ts_ns=None):
    """ISO8601 UTC with millisecond precision and a trailing Z (ledger timestamps).

    Millisecond precision so that "registered before submitted" is decidable for events in the
    same second (INTERFACES.md 9.3); equal timestamps still count as "not before".
    """
    if ts_ns is None:
        ts_ns = time.time_ns()
    ms = ts_ns // NS_PER_MS
    dt = datetime.datetime.fromtimestamp(ms // 1000, tz=datetime.timezone.utc)
    return "%s.%03dZ" % (dt.strftime("%Y-%m-%dT%H:%M:%S"), ms % 1000)


def parse_wall_iso(text):
    """Inverse of wall_iso; returns epoch milliseconds (int). Raises ValueError on other formats
    (including the second-precision form, so a ledger never mixes precisions silently)."""
    if not isinstance(text, str) or len(text) != 24 or text[19] != "." or text[-1] != "Z":
        raise ValueError("not a millisecond ISO8601 UTC timestamp: %r" % (text,))
    dt = datetime.datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")
    frac = text[20:23]
    if not frac.isdigit():
        raise ValueError("bad millisecond field in %r" % (text,))
    secs = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
    return secs * 1000 + int(frac)


def busy_wait_ns(delta_ns):
    """Spin (holding the GIL) until delta_ns has elapsed. Returns the elapsed ns actually spent."""
    t0 = time.monotonic_ns()
    end = t0 + delta_ns
    t = t0
    while t < end:
        t = time.monotonic_ns()
    return t - t0


def sleep_until_ns(target_ns, spin_ns=0):
    """Block the calling thread until monotonic time >= target_ns.

    Sleeps (releasing the GIL) until target_ns - spin_ns, then spins the remainder. spin_ns=0 means
    pure sleep. Returns the lateness (now - target) in ns, >= 0 unless the clock misbehaves.
    """
    now = time.monotonic_ns()
    remaining = target_ns - now - spin_ns
    if remaining > 0:
        time.sleep(remaining / NS_PER_S)
    now = time.monotonic_ns()
    while now < target_ns:
        now = time.monotonic_ns()
    return now - target_ns


class RelClock:
    """Converts between absolute monotonic ns and ns relative to a launch origin."""

    __slots__ = ("origin_ns",)

    def __init__(self, origin_ns):
        if not isinstance(origin_ns, int) or isinstance(origin_ns, bool):
            raise TypeError("origin_ns must be int")
        self.origin_ns = origin_ns

    def rel(self, abs_ns):
        return abs_ns - self.origin_ns

    def abs(self, rel_ns):
        return self.origin_ns + rel_ns

    def now_rel(self):
        return time.monotonic_ns() - self.origin_ns

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arrival schedules, beat schedules, prompt assignment (INTERFACES.md sections 3.4, 4.2, 4.3, 7.4).

The generation algorithm is fixed; changing it requires bumping GENERATOR_VERSION.
"""

import array
import bisect
import fractions
import math
import random
import struct

from .capbin import read_capbin, write_capbin
from .exitcodes import ConfigError

KIND = "schedule"
GENERATOR_VERSION = "capkit.schedule/1"
STEP_STAMP = struct.Struct("<qq")  # (step, planned_rel_ns) at the head of every beat message
NS = 10 ** 9


class ScheduleError(ConfigError):
    pass


def _fixed_tier_times(tier):
    # i-th arrival at start + floor(i * NS / rate), rate taken exactly as a Fraction of the float
    rate = fractions.Fraction(tier.rate_rps)
    num, den = rate.numerator, rate.denominator
    step_num = NS * den
    start, end = tier.start_ns, tier.end_ns
    i = 0
    while True:
        t = start + (i * step_num) // num
        if t >= end:
            return
        yield t
        i += 1


def _fixed_tier_count(tier):
    # number of i >= 0 with floor(i * NS / rate) < duration  <=>  i * NS / rate < duration
    rate = fractions.Fraction(tier.rate_rps)
    dur = tier.duration_ns
    if dur <= 0:
        return 0
    return math.ceil(fractions.Fraction(dur) * rate / NS)


def iter_arrivals(tier_table):
    """Yield (planned_rel_ns, tier_seq) for the whole table, in time order."""
    rng = random.Random(tier_table.seed)
    for tier in tier_table.tiers:
        if not tier.is_load:
            continue
        if tier_table.dist == "fixed":
            for t in _fixed_tier_times(tier):
                yield t, tier.seq
        else:
            t = float(tier.start_ns)
            rate = tier.rate_rps
            while True:
                u = rng.random()
                t += -math.log(1.0 - u) / rate * NS
                ti = int(t)
                if ti >= tier.end_ns:
                    break
                yield ti, tier.seq


def count_schedule_rows(tier_table):
    """Exact row count for fixed arrivals; for poisson, generates once to count."""
    if tier_table.dist == "fixed":
        return sum(_fixed_tier_count(t) for t in tier_table.tiers if t.is_load)
    return sum(1 for _ in iter_arrivals(tier_table))


def generate_schedule(tier_table, max_rows):
    """Return (planned_rel_ns array('q'), tier_seq array('h'), per_tier {seq: count})."""
    if not isinstance(max_rows, int) or max_rows <= 0:
        raise ScheduleError("max_rows must be an int > 0")
    planned = array.array("q")
    tseq = array.array("h")
    per_tier = {t.seq: 0 for t in tier_table.tiers if t.is_load}
    for t, s in iter_arrivals(tier_table):
        if len(planned) >= max_rows:
            raise ScheduleError("schedule exceeds max_rows=%d" % max_rows)
        planned.append(t)
        tseq.append(s)
        per_tier[s] += 1
    return planned, tseq, per_tier


def write_schedule(path, tier_table, max_rows, identity=None):
    """Generate and write the schedule capbin. tier_table must have been loaded from a file
    (its sha256 is recorded in the header). Returns the header meta."""
    if not tier_table.sha256:
        raise ScheduleError("tier table sha256 unknown (load it with TierTable.load)")
    planned, tseq, per_tier = generate_schedule(tier_table, max_rows)
    meta = {"generator": GENERATOR_VERSION, "tier_table_sha256": tier_table.sha256,
            "dist": tier_table.dist, "seed": tier_table.seed,
            "per_tier": {str(k): v for k, v in sorted(per_tier.items())},
            "prompt_assign": "mod", "n_requests": len(planned)}
    write_capbin(path, KIND, identity, [("planned_rel_ns", planned), ("tier_seq", tseq)], meta)
    return meta


def load_schedule(path, worker, n_workers, tier_table_sha=None):
    """Rows with req_id % n_workers == worker: (req_ids, planned_rel_ns, tier_seq, meta)."""
    if not (isinstance(n_workers, int) and n_workers >= 1 and isinstance(worker, int)
            and 0 <= worker < n_workers):
        raise ScheduleError("need 0 <= worker < n_workers")
    header, cols = read_capbin(path, stride=(worker, n_workers))
    if header.get("kind") != KIND:
        raise ScheduleError("%s is not a schedule capbin" % path)
    meta = header["meta"]
    if meta.get("generator") != GENERATOR_VERSION:
        raise ScheduleError("%s: generator %r unsupported" % (path, meta.get("generator")))
    if tier_table_sha is not None and meta.get("tier_table_sha256") != tier_table_sha:
        raise ScheduleError("%s was generated from tier table %s, not %s"
                            % (path, meta.get("tier_table_sha256"), tier_table_sha))
    n = len(cols["planned_rel_ns"])
    req_ids = array.array("q", range(worker, worker + n * n_workers, n_workers))
    return req_ids, cols["planned_rel_ns"], cols["tier_seq"], meta


def planned_per_worker(meta_per_tier_counts, n_rows_before, worker, n_workers):
    """How many of a contiguous block of rows [n_rows_before, n_rows_before + count) fall on
    `worker` (req_id % n_workers == worker). Used to derive expected per-worker counts."""
    lo = n_rows_before
    hi = n_rows_before + meta_per_tier_counts

    def upto(x):  # count of r in [0, x) with r % W == worker
        return (x - worker + n_workers - 1) // n_workers if x > worker else 0

    return upto(hi) - upto(lo)


def per_worker_tier_counts(meta, n_workers):
    """{(worker, tier_seq): planned count} derived from the schedule header alone."""
    out = {}
    before = 0
    for seq_s, count in sorted(meta["per_tier"].items(), key=lambda kv: int(kv[0])):
        for w in range(n_workers):
            out[(w, int(seq_s))] = planned_per_worker(count, before, w, n_workers)
        before += count
    return out


def prompt_index(req_id, set_size):
    """prompt_assign "mod": item index of request req_id within its tier's prompt set."""
    if set_size <= 0:
        raise ScheduleError("prompt set is empty")
    return req_id % set_size


class BeatSchedule:
    """Planned step times from a tier table (INTERFACES.md section 3.4).

    Step j of tier i (beat_ns > 0) is planned at start_i + j * beat_ns; global step numbers are
    cumulative over tiers in order.
    """

    def __init__(self, tier_table):
        self._segments = []  # (first_step, start_ns, beat_ns, n_steps, seq)
        step = 0
        for t in tier_table.tiers:
            if t.beat_ns <= 0:
                continue
            n = t.duration_ns // t.beat_ns
            self._segments.append((step, t.start_ns, t.beat_ns, n, t.seq))
            step += n
        self.n_steps = step
        self._firsts = [s[0] for s in self._segments]
        self._starts = [s[1] for s in self._segments]

    def first_step(self, seq):
        for first, _start, _beat, _n, s in self._segments:
            if s == seq:
                return first
        raise KeyError("tier %d has no beat" % seq)

    def planned_rel_ns(self, step):
        if not 0 <= step < self.n_steps:
            raise IndexError("step %d outside [0, %d)" % (step, self.n_steps))
        k = bisect.bisect_right(self._firsts, step) - 1
        first, start, beat, _n, _seq = self._segments[k]
        return start + (step - first) * beat

    def tier_of_step(self, step):
        if not 0 <= step < self.n_steps:
            raise IndexError(step)
        return self._segments[bisect.bisect_right(self._firsts, step) - 1][4]

    def step_at_or_after(self, rel_ns):
        """Smallest step whose planned time >= rel_ns, or None when there is none."""
        if not self._segments:
            return None
        k = max(0, bisect.bisect_right(self._starts, rel_ns) - 1)
        while k < len(self._segments):
            first, start, beat, n, _seq = self._segments[k]
            if rel_ns <= start:
                return first
            j = -(-(rel_ns - start) // beat)
            if j < n:
                return first + j
            k += 1
        return None

    def segments(self):
        return list(self._segments)


def stamp(step, planned_rel_ns):
    return STEP_STAMP.pack(step, planned_rel_ns)


def unstamp(buf):
    return STEP_STAMP.unpack_from(buf, 0)

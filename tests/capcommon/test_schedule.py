# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from capcommon import schedule
from capcommon.capbin import read_capbin
from capcommon.tiers import TierTable

from conftest import good_table_doc

S = 1_000_000_000


def test_fixed_schedule_counts_and_bounds(tmp_path, table_path):
    tt = TierTable.load(table_path())
    planned, tseq, per_tier = schedule.generate_schedule(tt, 10 ** 6)
    # warmup 1 s @100 + up 2 s @200 + up 1 s @400
    assert per_tier == {0: 100, 2: 400, 4: 400}
    assert schedule.count_schedule_rows(tt) == len(planned) == 900
    assert list(planned) == sorted(planned)
    for t, s in zip(planned, tseq):
        tier = tt[s]
        assert tier.start_ns <= t < tier.end_ns and tier.is_load
    assert planned[0] == 0 and planned[1] == 10_000_000
    with pytest.raises(schedule.ScheduleError):
        schedule.generate_schedule(tt, 899)


def test_odd_rate_exact_count(table_path):
    doc = good_table_doc()
    doc["tiers"][2]["rate_rps"] = 333.3333
    doc["tiers"][2]["load_value"] = 333.3333
    tt = TierTable.load(table_path(doc))
    _p, _t, per_tier = schedule.generate_schedule(tt, 10 ** 6)
    assert per_tier[2] == schedule._fixed_tier_count(tt[2]) == 667


def test_poisson_deterministic(table_path):
    doc = good_table_doc("poisson")
    tt1 = TierTable.load(table_path(doc, "a.json"))
    a = schedule.generate_schedule(tt1, 10 ** 6)
    b = schedule.generate_schedule(TierTable.load(table_path(doc, "b.json")), 10 ** 6)
    assert list(a[0]) == list(b[0])
    doc["arrival"]["seed"] = 8
    c = schedule.generate_schedule(TierTable.load(table_path(doc, "c.json")), 10 ** 6)
    assert list(a[0]) != list(c[0])
    assert schedule.count_schedule_rows(tt1) == len(a[0])
    for t, s in zip(*a[:2]):
        assert tt1[s].start_ns <= t < tt1[s].end_ns


def test_write_load_partition(tmp_path, table_path, ident):
    tt = TierTable.load(table_path())
    p = str(tmp_path / "sched.capbin")
    meta = schedule.write_schedule(p, tt, 10 ** 6, identity=ident)
    hdr, cols = read_capbin(p)
    assert hdr["kind"] == "schedule" and hdr["meta"]["tier_table_sha256"] == tt.sha256
    W = 3
    seen = {}
    for w in range(W):
        ids, planned, tseq, m = schedule.load_schedule(p, w, W, tier_table_sha=tt.sha256)
        assert all(r % W == w for r in ids)
        for r, t, s in zip(ids, planned, tseq):
            assert cols["planned_rel_ns"][r] == t and cols["tier_seq"][r] == s
            seen[r] = (w, s)
    assert sorted(seen) == list(range(meta["n_requests"]))
    expect = schedule.per_worker_tier_counts(meta, W)
    actual = {}
    for r, (w, s) in seen.items():
        actual[(w, s)] = actual.get((w, s), 0) + 1
    assert all(expect[k] == actual.get(k, 0) for k in expect)
    assert sum(expect.values()) == meta["n_requests"]
    with pytest.raises(schedule.ScheduleError):
        schedule.load_schedule(p, 0, W, tier_table_sha="b" * 64)
    with pytest.raises(schedule.ScheduleError):
        schedule.load_schedule(p, 3, W)


def test_beat_schedule(table_path):
    tt = TierTable.load(table_path())
    bs = schedule.BeatSchedule(tt)
    # every tier has beat 10 ms; total 5.5 s -> 550 steps
    assert bs.n_steps == 550
    assert bs.first_step(2) == 150 and bs.planned_rel_ns(150) == tt[2].start_ns
    for step in (0, 1, 99, 100, 149, 150, 549):
        t = bs.planned_rel_ns(step)
        assert bs.step_at_or_after(t) == step
        assert bs.step_at_or_after(t - 1) == step
        assert bs.tier_of_step(step) == tt.tier_seq_at(t)
    assert bs.step_at_or_after(tt.end_ns) is None
    with pytest.raises(IndexError):
        bs.planned_rel_ns(550)
    assert schedule.unstamp(schedule.stamp(7, 123) + b"rest") == (7, 123)


def test_prompt_index():
    assert schedule.prompt_index(10, 3) == 1
    with pytest.raises(schedule.ScheduleError):
        schedule.prompt_index(1, 0)

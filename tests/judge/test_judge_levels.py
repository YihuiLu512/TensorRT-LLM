# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibration v2 judging (calib-v2 DESIGN-r7a 3.1 P2-04, P2-05, P2-06; 3.5 tests 6, 9, 10,
14, 15, 16, 18, 19).

P2-04 capjudge.symptoms: X2 / X3 by z_d against the nullz threshold, X1 by |norm - 1|, every
statistic recorded with or without a threshold. P2-05 capjudge.checks: C13 window cap and the
launch-level C13-MEAN. P2-06 capjudge.judge: CONTRA tiers, levels[], boundary, T7, batch."""

import hashlib
import json
import os
import random
import sys

import pytest
from conftest import SAMPLES, judge_mod

S = judge_mod("stats")
X = judge_mod("symptoms")
Z = judge_mod("nullz")
I = judge_mod("inputs")
J = judge_mod("judge")
C = judge_mod("checks")
G = judge_mod("grade")

NS = 1_000_000_000
MS = 1_000_000
ARM = "synthetic-a"
LAG = "T1:req_first_lag_p50_ns:capload"
COUNT = "T1:inflight_derived:capload"
RATIO = "T2:delivery_ratio_norm:capload"
FOREIGN = "T3:foreign_busy_frac:nullsse"
ENV = {"hw_profile_sha": "d" * 64, "image_sha": "e" * 64, "capkit_sha": "c" * 40,
       "baseline_sha": "f" * 10}
REAL_W = "/home/scratch.yihlu_coreai/workfile/feat_dev-J17503/data/stage-01/test-02"
REAL_TREE = "02_e0/E0/runs/w60/J5940628-r0-L2_e0b-s1-single-w60"
TWO_LEVEL = os.path.join(SAMPLES, "calibv2_real", "j5940628_l1_t4_requests.json")
TWO_LEVEL_SHA = "92dbf6ec983246fc687f3daf7e97f934d0df23e53a8803ce4d2f3c9beef79f36"


def _boot_doc(**kw):
    d = {"schema": "capkit.pilot/2", "experiment": "E0", "stage": "bootstrap",
         "environment": dict(ENV, judge_sha="b" * 40),
         "d14": {"relpath": "e0_d14.json", "sha256": "1" * 64},
         "x_const": {"lag": {"z": 10.0, "min_D_ns": 10 * MS}, "count": {"z": 10.705}},
         "ratio_tol": {"T2:delivery_ratio_norm": {"min_per_request": 2, "tol": 0.02}},
         "c13_cap": {"T3:foreign_busy_frac": {"window": 0.10, "launch_mean": 0.05}}}
    d.update(kw)
    return d


def _thr_doc(null_z):
    return {"schema": "capkit.pilot/2", "experiment": "E0", "stage": "thresholds",
            "environment": dict(ENV, judge_sha="b" * 40),
            "d14": {"relpath": "e0_d14.json", "sha256": "1" * 64},
            "window_ns": 60 * NS, "inputs": [], "null_z": null_z, "ratio": {},
            "c13_cap": {"T3:foreign_busy_frac": {"window": 0.10, "launch_mean": 0.05}}}


def _th(doc):
    t = Z.Thresholds()
    t.add_source("pilot", "p.json", "a" * 64, doc)
    return t


def _make_samples():
    sys.path.insert(0, SAMPLES)
    try:
        import make_samples
    finally:
        sys.path.remove(SAMPLES)
    return make_samples


def _v2(ladder=None, **calib):
    """Options of a v2 sample (bootstrap pilot/2 + d14_result; arm_config; one ladder)."""
    return {"calib_v2": dict({"stage": "bootstrap"}, **calib),
            "v2": {"configs": {ARM: "s1@1ms"}, "ladders": {ARM: ladder or {}}}}


def _judge(tmp_path, opts, name="good"):
    """Build one sample work dir and judge its tree in process. Returns (verdicts, ctx)."""
    M = _make_samples()
    n = len([x for x in os.listdir(tmp_path) if x.startswith("w")])
    out = os.path.join(str(tmp_path), "w%02d" % n)
    W = M.build(out, name, opts, {})
    with open(os.path.join(W, "expect.json")) as f:
        tree = os.path.join(W, json.load(f)["tree"])
    ctx = I.verify(W, tree)
    return J.judge_tree(ctx, lambda _m: None)["verdicts"], ctx


def _tier(v, seq):
    return next(t for t in v["tiers"] if t["tier_seq"] == seq)


# ---- P2-04: X2 / X3 by z_d, X1 by |norm - 1| -------------------------------------------------

def _linear(L, per_s, noise, seed=1, offset=0.0):
    rng = random.Random(seed)
    return [offset + per_s * i + rng.gauss(0.0, noise) for i in range(L)]


def test_trend_z_records_D_and_z_without_a_threshold():
    rec = X.Record()
    s = _linear(60, 1000.0, 50.0)
    d, e = X._trend_z(s, LAG, None, "lag", rec)
    assert d is None and e["status"] == "NOT_RUN" and "no threshold source" in e["reason"]
    assert rec[LAG] == [S.t1_trend(s)[0]] and rec.z[LAG] == [S.zd(s)["z"]]
    assert e["D"] == rec[LAG][0] and e["z"] == rec.z[LAG][0]
    # a plain dict has nowhere to put z_d: evaluate_x refuses it
    with pytest.raises(TypeError):
        X.evaluate_x(None, None, None, None, {})


def test_bootstrap_lag_needs_both_z_and_D():
    lag = _th(_boot_doc()).get(LAG, "s1@1ms", "near_boundary")
    small_D = _linear(60, 5 * MS / 30, 10_000.0)  # D ~ 5 ms, z ~ 1e3
    big_D = _linear(60, 20 * MS / 30, 10_000.0)  # D ~ 20 ms, z ~ 4e3
    noisy = _linear(60, 20 * MS / 30, 20 * MS)  # D ~ 20 ms, z ~ 3
    for s, want in ((small_D, "inside"), (big_D, "above"), (noisy, "inside")):
        rec = X.Record()
        d, e = X._trend_z(s, LAG, lag, "lag", rec)
        assert d == want == e["status"], (e["D"], e["z"])
        assert rec[LAG] == [e["D"]] and rec.z[LAG] == [e["z"]]
    assert X._trend_z(small_D, LAG, lag, "lag", X.Record())[1]["z"] > 100
    # a falling lag is never "below" against the bootstrap lag constant
    assert X._trend_z([-v for v in big_D], LAG, lag, "lag", X.Record())[0] == "inside"


def test_thresholds_stage_key_is_two_sided_and_uncalibrated_is_not_run():
    key = LAG + "|s1@1ms|near|L=60"
    th = _th(_thr_doc({key: {"family": "lag", "thr_hi": 3.5, "thr_lo": -3.1,
                             "uncalibrated": False},
                       COUNT + "|s1@1ms|near|L=60": {"family": "count", "thr_hi": None,
                                                     "thr_lo": None, "uncalibrated": True}}))
    e = th.get(LAG, "s1@1ms", "near_boundary")
    rng = random.Random(3)
    base = [rng.gauss(0.0, 1.0) for _ in range(60)]
    up = [v + (2.0 if i >= 30 else 0.0) for i, v in enumerate(base)]
    down = [v - (2.0 if i >= 30 else 0.0) for i, v in enumerate(base)]
    zs = [S.zd(s)["z"] for s in (base, up, down)]
    assert -3.1 <= zs[0] <= 3.5 and zs[1] > 3.5 and zs[2] < -3.1
    got = [X._trend_z(s, LAG, e, "lag", X.Record())[0] for s in (base, up, down)]
    assert got == ["inside", "above", "below"]
    rec = X.Record()
    d, ent = X._trend_z([0, 1] * 30, COUNT, th.get(COUNT, "s1@1ms", "near_boundary"), "count",
                        rec)
    assert d is None and ent["status"] == "NOT_RUN" and "uncalibrated" in ent["reason"]
    assert rec.z[COUNT] == [ent["z"]]  # recorded all the same
    # another configuration has no key in a thresholds-stage file; a v1 register (no
    # configuration) reads no null_z key at all
    assert th.get(LAG, "s4", "near_boundary") is None
    assert th.get(LAG, None, "near_boundary") is None
    assert th.lookups[LAG + "|-|near_boundary"] == "missing"


def _count_series(second_half):
    return [0] * 30 + list(second_half)


def test_count_floor_and_strict_comparison():
    # test 15: z_1q = 10.705 at L = 60; an all-zero calibration gives z_d 0 everywhere, so its
    # empirical thresholds take the floor (floor_bound)
    z1 = S.count_floor_z(60)
    assert round(z1, 3) == 10.705
    zeros = [S.zd([0] * 60, quantum=1)["z"] for _ in range(600)]
    assert set(zeros) == {0.0}
    thr = S.floor_count_thresholds(S.order_stat_thresholds(zeros, 600, 0.01), 60)
    assert thr["thr_hi"] == z1 and thr["thr_lo"] == -z1 and thr["floor_bound"] is True
    wide = S.floor_count_thresholds({"thr_hi": 12.0, "thr_lo": -11.0, "uncalibrated": False},
                                    60)
    assert (wide["thr_hi"], wide["thr_lo"], wide["floor_bound"]) == (12.0, -11.0, False)
    nz = COUNT + "|s1@1ms|near|L=60"
    floor_doc = _th(_thr_doc({nz: {"family": "count", "thr_hi": thr["thr_hi"],
                                   "thr_lo": thr["thr_lo"], "floor_bound": True,
                                   "uncalibrated": False}}))
    halves = {0.5: [0] * 15 + [1] * 15, 1.0: [1] * 30, 1.5: [1] * 15 + [2] * 15}
    for th in (_th(_boot_doc()).get(COUNT, "s1@1ms", "near_boundary"),
               floor_doc.get(COUNT, "s1@1ms", "near_boundary")):
        got = {}
        for D, half in halves.items():
            rec = X.Record()
            d, e = X._trend_z(_count_series(half), COUNT, th, "count", rec)
            assert e["D"] == D and abs(e["sigma"] - 1 / 12 ** 0.5) < 1e-12  # the floor
            got[D] = (d, round(e["z"], 2))
        assert got == {0.5: ("inside", 5.35), 1.0: ("inside", 10.7), 1.5: ("above", 16.06)}
    # without the quantum the same series has a much smaller scale and D = 1 would cross
    assert S.zd(_count_series(halves[1.0]))["z"] > 30


def test_truncated_overload_series_is_not_run_not_no_trend():
    # test 10: an overloaded tier whose late messages were cut after the tail: the second half
    # of the window has two seconds with a value. t1 still gives a D (recorded), but z_d has
    # too few first differences -> the key is NOT_RUN, never "inside"
    s = [1000.0 * i for i in range(28)] + [None, None, 31000.0, None, 33000.0] + [None] * 27
    assert len(s) == 60 and S.t1_trend(s) is not None
    lag = _th(_boot_doc()).get(LAG, "s1@1ms", "near_boundary")
    rec = X.Record()
    d, e = X._trend_z(s, LAG, lag, "lag", rec)
    assert d is None and e["status"] == "NOT_RUN" and "too_few_diffs" in e["reason"]
    assert rec[LAG] == [S.t1_trend(s)[0]] and LAG not in rec.z
    out = X._combine({"reasons": [], "overload_failures": 0}, [d], ["inside"], ["inside"])
    assert out["X2_status"] == "NOT_RUN" and out["holds"] is None


def _two_level():
    with open(TWO_LEVEL, "rb") as f:
        raw = f.read()
    assert hashlib.sha256(raw).hexdigest() == TWO_LEVEL_SHA
    return json.loads(raw)


class _Tier:
    def __init__(self, ws, we):
        self.window_start_ns, self.window_end_ns = ws, we


def test_two_level_switching_is_reported_not_judged():
    # test 6: J5940628:L1 tier 4 (e0a-nullsse-w60, 100 rps), capload requests of the steady
    # window copied from the run tree (samples/calibv2_real/SOURCES.md)
    d = _two_level()
    t = _Tier(d["window_start_ns"], d["window_end_ns"])
    times, lags = X.request_lag_pairs(d["rows"], t, "first")
    series, lsr = X.lag_series(times, lags, t)
    assert len(series) == 60 and None not in series
    r = S.zd(series)
    assert abs(r["D"] + 36293.75) < 1e-6 and abs(r["z"] + 1.58) < 0.01
    assert lsr > 3 and abs(lsr - 5.13) < 0.01
    lag = _th(_boot_doc()).get(LAG, "s1@1ms", "near_boundary")
    assert X._trend_z(series, LAG, lag, "lag", X.Record())[0] == "inside"
    # negative control: the same seconds with the values shuffled across seconds (one level)
    rng = random.Random(7)
    flat = list(lags)
    rng.shuffle(flat)
    _s, lsr0 = X.lag_series(times, flat, t)
    assert lsr0 < 1.5


def test_level_switch_ratio_edge_cases():
    assert S.level_switch_ratio([[1, 2, 3]]) == S.NotRun("too_few_seconds")
    assert S.level_switch_ratio([[5, 5], [5, 5], None]) == S.NotRun("zero_sampling_se")
    rng = random.Random(11)
    iid = [[rng.gauss(0, 1) for _ in range(100)] for _ in range(60)]
    assert 0.5 < S.level_switch_ratio(iid) < 1.5
    two = [[rng.gauss(10 if i % 2 else 0, 1) for _ in range(100)] for i in range(60)]
    assert S.level_switch_ratio(two) > 3


def test_x1_tolerance_floor_per_planned_requests():
    # test 19, X1 side: bootstrap tolerance max(2 / planned requests, tol)
    r = _th(_boot_doc()).get(RATIO, "s1@1ms", "near_boundary")
    assert Z.compare(0.019, r, planned_requests=1145) == "inside"
    assert Z.compare(0.021, r, planned_requests=1145) == "above"
    assert Z.compare(-0.021, r, planned_requests=1145) == "below"
    assert Z.compare(0.039, r, planned_requests=50) == "inside"  # 2 / 50 = 0.04
    assert Z.compare(-0.041, r, planned_requests=50) == "below"


@pytest.mark.parametrize("offered,tol,want", [(100, 0.0, "inside"), (1000, 0.0, "below"),
                                               (1000, 0.02, "inside")])
def test_x1_tolerance_uses_the_tiers_planned_requests(tmp_path, monkeypatch, offered, tol,
                                                      want):
    # tier 4 delivers 1 % short of its offer (the baseline tier in full): with the bootstrap
    # tol 0 the floor 2 / planned requests decides (0.02 at 100 requests, 0.002 at 1000)
    real = X.delivery_counts

    def counts(rows, tier):
        return (offered, offered - offered // 100) if tier.seq == 4 else real(rows, tier)

    monkeypatch.setattr(X, "delivery_counts", counts)
    opts = {"calib_v2": {"bootstrap_set": {"ratio_tol.T2:delivery_ratio_norm.tol": tol}}}
    v, _ctx = _judge(tmp_path, opts)
    x1 = _tier(v, 4)["X"]["X1"][0]
    assert x1["planned_requests"] == offered and abs(x1["abs_dev"] - 0.01) < 1e-12
    assert x1["status"] == want
    assert _tier(v, 4)["verdict"] == ("CONTRA" if want == "below" else "STABLE")


def test_sample_tiers_record_abs_deviation_z_and_D(tmp_path):
    v, _ctx = _judge(tmp_path, _v2())
    assert v["batch"] == "calib-v2"
    t2, t4 = _tier(v, 2), _tier(v, 4)
    # the baseline tier's own ratio is 1 by construction: not recorded
    assert RATIO not in t2["stats"] and t4["stats"][RATIO] == [0.0]
    x1 = t4["X"]["X1"][0]
    assert x1["abs_dev"] == 0.0 and x1["status"] == "inside" and x1["planned_requests"] > 0
    for t in (t2, t4):
        for e in t["X"]["X2"] + t["X"]["X3"]:
            assert t["stats"][e["key"]] == [e["D"]] and t["zstats"][e["key"]] == [e["z"]]
            assert e["status"] == "inside"
        assert t["verdict"] == "STABLE"
    assert set(t4["zstats"]) == {e["key"] for e in t4["X"]["X2"] + t4["X"]["X3"]}


def test_v0_cal_register_without_thresholds_records_everything(tmp_path):
    # V0-cal: only e0_d14.json is registered -> X and the C13 second half are NOT_RUN, every
    # reading is recorded (stats, zstats, |norm - 1|, C13)
    v, ctx = _judge(tmp_path, {"calib_v2": {"pilot": False}})
    assert ctx.thresholds.stage is None and ctx.thresholds.d14() is not None
    t4 = _tier(v, 4)
    assert t4["verdict"] == "NOT_RUN" and t4["checks"]["C13"] == "NOT_RUN"
    assert all(e["status"] == "NOT_RUN" and "no threshold source" in e["reason"]
               for e in t4["X"]["X2"] + t4["X"]["X1"] + t4["X"]["X3"])
    assert t4["stats"][RATIO] == [0.0] and len(t4["stats"][FOREIGN]) == 1
    assert set(t4["zstats"]) == {e["key"] for e in t4["X"]["X2"] + t4["X"]["X3"]}
    assert v["launch_checks"]["C13-MEAN"] == "NOT_RUN" and v["launch_verdict"] == "NOT_RUN"
    assert v["c13_mean"]["nullsse"]["cap"] is None


# ---- P2-05: C13 window cap and C13-MEAN ----------------------------------------------------------

@pytest.mark.parametrize("rate,t4_c13,t4_verdict,mean_status,launch", [
    ({"4": 0.11}, "RED", "VOID", "RED", "VOID"),        # 0.11 > 0.10 window cap
    ({"4": 0.09}, "PASS", "STABLE", "PASS", "VALID"),   # mean (0 + 0.09) / 2 <= 0.05
    ({"all": 0.06}, "PASS", "STABLE", "RED", "VOID"),   # every window 0.06: mean > 0.05
])
def test_c13_caps_from_the_bootstrap_pilot(tmp_path, rate, t4_c13, t4_verdict, mean_status,
                                           launch):
    # test 18 (synthetic sidecar): caps 0.10 / 0.05 from the bootstrap pilot/2 sample
    v, ctx = _judge(tmp_path, dict(_v2(), c13_foreign_rate=rate))
    assert ctx.thresholds.cap("T3:foreign_busy_frac") == {"window": 0.10, "launch_mean": 0.05}
    t2, t4 = _tier(v, 2), _tier(v, 4)
    want = {"4": (0.0, rate.get("4")), "all": (0.06, 0.06)}["all" if "all" in rate else "4"]
    assert abs(t2["stats"][FOREIGN][0] - want[0]) < 1e-9
    assert abs(t4["stats"][FOREIGN][0] - want[1]) < 1e-9
    assert t4["checks"]["C13"] == t4_c13 and t4["verdict"] == t4_verdict
    assert t2["checks"]["C13"] == "PASS" and t2["verdict"] == "STABLE"
    assert v["launch_checks"]["C13-MEAN"] == mean_status and v["launch_verdict"] == launch
    m = v["c13_mean"]["nullsse"]
    assert m["n"] == 2 and m["cap"] == 0.05
    assert abs(m["mean"] - (want[0] + want[1]) / 2) < 1e-9
    # the level of a one-pass ladder is its tier
    assert v["levels"][0]["verdict"] == t4_verdict


def test_c13_without_d14_result_is_not_run_and_reads_nothing(tmp_path):
    v, ctx = _judge(tmp_path, dict(_v2(d14=False), c13_foreign_rate={"4": 0.11}))
    assert ctx.thresholds.d14() is None and ctx.thresholds.stage == "bootstrap"
    t4 = _tier(v, 4)
    assert t4["checks"]["C13"] == "NOT_RUN" and FOREIGN not in t4["stats"]
    assert t4["verdict"] == "NOT_RUN"
    assert v["launch_checks"]["C13-MEAN"] == "NOT_RUN" and v["launch_verdict"] == "NOT_RUN"


def test_c13_with_d14_but_without_a_cap_records_the_readings(tmp_path):
    v, _ctx = _judge(tmp_path, {"calib_v2": {"pilot": False}, "c13_foreign_rate": {"4": 0.11}})
    t4 = _tier(v, 4)
    assert t4["checks"]["C13"] == "NOT_RUN"  # 0.11, but no cap: no RED
    assert abs(t4["stats"][FOREIGN][0] - 0.11) < 1e-9
    assert abs(v["c13_mean"]["nullsse"]["mean"] - 0.055) < 1e-9


def _real_ctx():
    tree = os.path.join(REAL_W, REAL_TREE)
    if not os.path.isdir(tree):
        pytest.skip("read-only run tree %s not present on this machine" % tree)
    return I.verify(REAL_W, tree)


def test_c13_passes_on_a_real_e0w60_e0b_tree():
    # test 18, real tree: J5940628:L2 (e0b-s1-single-w60) is read in place (read only, 58 MB,
    # not copied); its register has no calibration, so the v2 sources are added in memory: an
    # e0_d14.json sample whose environment is the tree's profile and image, and the bootstrap
    # pilot/2 sample. Every load tier's C13 is PASS and the launch mean is under its cap.
    ctx = _real_ctx()
    env = dict(ENV, hw_profile_sha=ctx.launch["hw_profile_sha"],
               image_sha=ctx.launch["image"]["sha256"])
    d14 = {"schema": "capkit.d14_result/1", "rule": {"relpath": "d14_rule.json",
                                                     "sha256": "2" * 64},
           "select": "capjudge.pilot.select_tiers", "environment": env, "inputs": [],
           "d14_softirq": {"irq_softirq_is_experiment": False}}
    boot = _boot_doc(environment=dict(env, judge_sha="b" * 40))
    I._check_env(ctx, d14, "e0_d14.json")
    I._check_env(ctx, boot, "pilot2.json")
    with pytest.raises(I.InputInvalid, match="calibration environment"):
        I._check_env(ctx, dict(d14, environment=ENV), "e0_d14.json")
    th = Z.Thresholds()
    th.add_source("d14_result", "e0_d14.json", "1" * 64, d14)
    th.add_source("pilot", "pilot2.json", "a" * 64, boot)
    ctx.thresholds = th
    TD = judge_mod("treedata")
    ld = TD.LaunchData(ctx)
    tiers = []
    for t in ctx.tt.tiers:
        if t.kind not in ("up", "down"):
            continue
        rec = X.Record()
        r = C.check_c13_tier(ctx, ld, TD.TierData(ctx, t), rec)
        assert r.status == "PASS", (t.seq, r.detail, r.not_run)
        assert len(rec["T3:foreign_busy_frac:relay1"]) == 1
        tiers.append({"judged": True, "stats": dict(rec)})
    assert len(tiers) == 5
    r, means = C.check_c13_mean(ctx, tiers)
    assert r.status == "PASS" and means["relay1"]["n"] == 5
    assert abs(means["relay1"]["mean"]) < 0.05


# ---- P2-06: tier CONTRA, levels[], boundary, launch verdict --------------------------------------

KA, KB = "T1:req_first_lag_p50_ns:a", "T1:msg_lag_p50_ns:b"
X1_BELOW = "X1 below band (delivery short of offer)"
X3_BELOW = "X3 below band (backlog shrinking)"


def _p(verdict, keys=(), overload=0, contra=()):
    return {"verdict": verdict, "X": {"X2": [{"key": k, "status": "above"} for k in keys]
                                      + [{"key": "T1:send_lag_p50_ns:a", "status": "inside"}],
                                      "overload_failures": overload,
                                      "contradiction": list(contra)}}


@pytest.mark.parametrize("up,down,want,flags", [
    (_p("UNSTABLE", [KA]), _p("STABLE"), "AMBIGUOUS", []),                 # one pass crosses
    (_p("UNSTABLE", [KA]), _p("UNSTABLE", [KA, KB]), "UNSTABLE", []),      # same key twice
    (_p("UNSTABLE", [KA]), _p("UNSTABLE", [KB]), "AMBIGUOUS", ["keys_differ"]),
    (_p("STABLE"), _p("STABLE"), "STABLE", []),
    (_p("VOID"), _p("UNSTABLE", [KA]), "VOID", []),                       # e.g. C13 RED
    (_p("STABLE"), _p("VOID", [KA]), "VOID", []),
    (_p("NOT_RUN"), _p("UNSTABLE", [KA]), "NOT_RUN", []),
    (_p("UNSTABLE", overload=3), _p("UNSTABLE", overload=1), "UNSTABLE", []),
    (_p("UNSTABLE", overload=3), _p("UNSTABLE", [KA]), "AMBIGUOUS", ["keys_differ"]),
    (_p("UNDETERMINED", [KA]), _p("UNSTABLE", [KA]), "UNDETERMINED", []),
    (_p("UNDETERMINED", [KA]), _p("STABLE"), "AMBIGUOUS", []),
    # contradictions (test 19): CONTRA only when both passes show the same one
    (_p("CONTRA", contra=[X1_BELOW]), _p("CONTRA", contra=[X1_BELOW]), "CONTRA",
     ["contradiction:" + X1_BELOW]),
    (_p("CONTRA", [KA], contra=[X3_BELOW]), _p("CONTRA", [KA], contra=[X3_BELOW, X1_BELOW]),
     "CONTRA", ["contradiction:" + X3_BELOW]),
    (_p("CONTRA", contra=[X1_BELOW]), _p("STABLE"), "STABLE", ["contradiction_single:up"]),
    (_p("STABLE"), _p("CONTRA", [KA], contra=[X3_BELOW]), "STABLE",
     ["contradiction_single:down"]),
    (_p("CONTRA", [KA], contra=[X3_BELOW]), _p("UNSTABLE", [KA]), "AMBIGUOUS",
     ["contradiction_single:up"]),
    (_p("CONTRA", contra=[X1_BELOW]), _p("CONTRA", contra=[X3_BELOW]), "STABLE",
     ["contradiction_single:up", "contradiction_single:down"]),
    (_p("CONTRA", contra=[X1_BELOW]), _p("NOT_RUN"), "NOT_RUN", ["contradiction_single:up"]),
])
def test_level_composition(up, down, want, flags):
    # test 14
    v, ku, kd, fl = J.compose_level(up, down)
    assert v == want and fl == flags
    assert ku == sorted(k["key"] for k in up["X"]["X2"] if k["status"] == "above")
    assert kd == sorted(k["key"] for k in down["X"]["X2"] if k["status"] == "above")


@pytest.mark.parametrize("verdict", ["STABLE", "UNSTABLE", "UNDETERMINED", "VOID", "NOT_RUN",
                                     "CONTRA"])
def test_one_pass_level_is_its_tier(verdict):
    assert J.compose_level(_p(verdict, [KA]), None) == (verdict, [KA], None, [])


def _e(seq, verdict):
    return {"tier_seq": seq, "verdict": verdict}


def test_launch_verdict_never_voided_by_contra():
    P, N, R = "PASS", "NOT_RUN", "RED"
    lv = J.launch_verdict_of
    assert lv([_e(2, "STABLE"), _e(4, "CONTRA")], P, False, P)[0] == "VALID"
    assert lv([_e(2, "STABLE"), _e(4, "CONTRA")], P, False, P, {4})[0] == "VALID"
    assert lv([_e(2, "STABLE"), _e(4, "VOID")], P, False, P, {4})[0] == "VOID"  # C13 RED etc.
    assert lv([_e(2, "STABLE"), _e(4, "STABLE")], P, False, R)[0] == "VOID"  # C13-MEAN
    assert lv([_e(2, "STABLE"), _e(4, "STABLE")], R, False, P)[0] == "VOID"
    assert lv([_e(2, "STABLE"), _e(4, "STABLE")], P, True, P)[0] == "VOID"
    # a ladder level tier NOT_RUN leaves the launch verdict alone; any other tier does not
    assert lv([_e(2, "STABLE"), _e(4, "NOT_RUN")], P, False, P, {4})[0] == "VALID"
    assert lv([_e(2, "NOT_RUN"), _e(4, "STABLE")], P, False, P, {4})[0] == "NOT_RUN"
    assert lv([_e(2, "STABLE"), _e(4, "NOT_RUN")], P, False, P)[0] == "NOT_RUN"
    assert lv([_e(2, "STABLE")], N, False, P, {4})[0] == "NOT_RUN"
    assert lv([_e(2, "STABLE")], P, False, N)[0] == "NOT_RUN"
    assert lv([], P, False, P)[0] == "NOT_RUN"


def _levels(code, base=100.0):
    names = {"S": "STABLE", "U": "UNSTABLE", "A": "AMBIGUOUS", "C": "CONTRA", "N": "NOT_RUN",
             "D": "UNDETERMINED", "V": "VOID"}
    return [{"level": k, "rho": 1 + k / 100, "load_value": base + k, "verdict": names[c]}
            for k, c in enumerate(code)]


def _b(code):
    return J.ladder_boundary(_levels(code))


def test_boundary_definitions():
    # test 16
    b = _b("SSSUU")
    assert (b["s_star"]["level"], b["u_star"]["level"], b["b"]) == (2, 3, 102.5)
    assert not b["censored"] and not b["non_monotone"] and b["b_robust"] == 102.5
    amb, con = _b("SSAU"), _b("SSCU")
    assert amb["ambiguous_inside"] and not amb["censored"] and amb["b"] == 102.0
    # contra_inside: b as with the level ambiguous, not censored
    assert con["contra_inside"] and not con["ambiguous_inside"] and not con["censored"]
    assert con["b"] == amb["b"] and con["n_contra_levels"] == 1
    assert _b("SCSUU")["contra_inside"] is False  # below s*: not inside
    for code, why in (("SSS", "no UNSTABLE level"), ("UUU", "no STABLE level"),
                      ("AAC", "no STABLE level")):
        c = _b(code)
        assert c["censored"] and c["b"] is None and c["censored_reasons"] == [why]
    for code, flag in (("SNU", "notrun_inside"), ("SDU", "undetermined_inside"),
                       ("SVU", "void_inside")):
        c = _b(code)
        assert c[flag] and c["censored"] and c["censored_reasons"] == [flag]
        assert c["b"] == 101.0  # reported, but the launch is censored (left out of T5)
    assert not _b("NSU")["censored"]  # NOT_RUN below s*: not inside
    # s* > u*: non_monotone, inside is (u*, s*), b = (s* + u*) / 2
    nm = _b("SUASUU")
    assert nm["non_monotone"] and nm["s_star"]["level"] == 3 and nm["u_star"]["level"] == 1
    assert nm["ambiguous_inside"] and nm["b"] == (103.0 + 101.0) / 2 and not nm["censored"]
    assert _b("SUNSU")["notrun_inside"] and _b("SUNSU")["censored"]
    # b_robust: a single UNSTABLE level at u* with a STABLE level above -> the higher run
    r = _b("SSUSSUU")
    assert r["u_star"]["level"] == 2 and r["b"] == (104.0 + 102.0) / 2
    assert r["b_robust"] == (104.0 + 105.0) / 2
    assert _b("SSUS")["b_robust"] is None
    assert _b("ASASUU")["n_ambiguous_below_b"] == 2 and _b("SSUA")["n_ambiguous_below_b"] == 0


def test_contra_level_leaves_b_and_the_launch_alone():
    # test 14: the same contradiction in both passes -> level CONTRA, launch not VOID, b as if
    # the level were AMBIGUOUS
    contra = J.compose_level(_p("CONTRA", contra=[X1_BELOW]), _p("CONTRA", contra=[X1_BELOW]))
    lv = _levels("SSAU")
    lv[2]["verdict"] = contra[0]
    assert contra[0] == "CONTRA" and J.ladder_boundary(lv)["b"] == _b("SSAU")["b"]
    judged = [_e(10, "CONTRA"), _e(11, "CONTRA"), _e(12, "STABLE")]
    assert J.launch_verdict_of(judged, "PASS", False, "PASS", {10, 11, 12})[0] == "VALID"


def test_grade_walk_passes_over_contra_tiers():
    tiers = [{"judged": True, "order_key": 1, "tier_seq": 2, "verdict": "STABLE",
              "load_value": 20.0, "checks": {}},
             {"judged": True, "order_key": 2, "tier_seq": 4, "verdict": "CONTRA",
              "load_value": 40.0, "checks": {}}]
    w = G.first_non_stable(tiers)
    assert w["first"] is None and w["last_stable"]["tier_seq"] == 2
    assert w["skipped"] == [{"tier_seq": 4, "verdict": "CONTRA", "red": []}]


# ---- P2-06: T7 on the slope tiers, ladder roles in a judged sample -------------------------------

class _TT:
    def __init__(self, seq, load):
        self.seq, self.load_value = seq, load
        self.window_start_ns, self.window_end_ns = 0, 60 * NS


def test_t7_on_linear_growth_slope_tiers():
    # test 9: rho = 1.05 / 1.10 / 1.20 of mu = 100 rps, per-second lag growing at
    # (lam / mu - 1) s per s plus 10 ms noise: z_d far above the bootstrap threshold, T7 mu
    # within 0.5 %, beta about -1
    mu = 100.0
    lag = _th(_boot_doc()).get(LAG, "s1@1ms", "near_boundary")
    by_seq, tiers, slope = {}, {}, []
    for k, rho in enumerate((1.05, 1.10, 1.20)):
        lam, seq = rho * mu, 10 + k
        s = _linear(60, (lam / mu - 1) * NS, 10 * MS, seed=k, offset=0.5 * (lam / mu - 1) * NS)
        rec = X.Record()
        d, e = X._trend_z(s, LAG, lag, "lag", rec)
        assert d == "above" and e["z"] > 100  # bootstrap z 10
        by_seq[seq] = {"tier_seq": seq, "stats": dict(rec), "zstats": dict(rec.z)}
        tiers[seq] = _TT(seq, lam)
        slope.append({"rho": rho, "seq": seq})
    t7 = J.ladder_t7({"slope": slope}, by_seq, tiers)
    assert t7["role"] == "capload" and t7["n"] == 3
    assert abs(t7["mu"] / mu - 1) < 0.005 and abs(t7["beta"] + 1) < 0.01
    assert [p["tier_seq"] for p in t7["points"]] == [10, 11, 12]
    # one slope tier -> NOT_RUN; two request roles -> NOT_RUN with each role's fit
    one = J.ladder_t7({"slope": slope[:1]}, by_seq, tiers)
    assert one == {"status": "NOT_RUN", "reason": "too_few_loads"}
    for e in by_seq.values():
        e["stats"]["T1:req_first_lag_p50_ns:other"] = e["stats"][LAG]
    two = J.ladder_t7({"slope": slope}, by_seq, tiers)
    assert two["status"] == "NOT_RUN" and sorted(two["by_role"]) == ["capload", "other"]
    assert J.ladder_t7({"slope": []}, by_seq, tiers)["status"] == "NOT_RUN"


def test_slope_tier_keeps_its_x_statistics_out_of_levels(tmp_path):
    # test 9 (r7a, CR6-S3 3): the overloaded tier 4 in the slope role still has its
    # req_first_lag D in stats and z in zstats, T7 reads it, and the tier is in no level and
    # not in the boundary
    ladder = {"baseline_seq": None, "levels": [],
              "slope": [{"rho": 0.5, "seq": 2}, {"rho": 1.0, "seq": 4}]}
    v, ctx = _judge(tmp_path, dict(_v2(ladder), overload=True), name="sat_overloaded_tier")
    assert ctx.tier_role == {2: "slope", 4: "slope"}
    t4 = _tier(v, 4)
    assert t4["ladder_role"] == "slope" and t4["X"]["X2"]
    assert len(t4["stats"][LAG]) == 1 and len(t4["zstats"][LAG]) == 1
    assert t4["verdict"] == "UNSTABLE" and t4["X"]["overload_failures"] > 0
    assert v["levels"] == [] and v["boundary"]["s_star"] is None
    assert v["boundary"]["u_star"] is None and v["boundary"]["censored"]
    t7 = v["t7"]
    assert t7["role"] == "capload" and [p["tier_seq"] for p in t7["points"]] == [2, 4]
    assert [p["D"] for p in t7["points"]] == [_tier(v, 2)["stats"][LAG][0],
                                              t4["stats"][LAG][0]]
    assert t7["mu"] > 0 and v["criterion_bias"] is None  # no b without levels


def test_contradiction_gives_contra_not_void(tmp_path):
    # test 14 (non-ladder tier and one-pass level): the backlog grows without a lag trend
    v, _ctx = _judge(tmp_path, {"calib_v2": {"stage": "bootstrap"}, "x_contradiction": True},
                     name="x_contradiction")
    t4 = _tier(v, 4)
    assert t4["verdict"] == "CONTRA" and t4["X"]["contradiction"]
    assert v["launch_verdict"] == "VALID" and "levels" not in v and "boundary" not in v
    assert "t7" not in v and v["batch"] == "calib-v2"
    v, _ctx = _judge(tmp_path, dict(_v2(), x_contradiction=True), name="x_contradiction")
    assert v["levels"][0]["verdict"] == "CONTRA" and v["launch_verdict"] == "VALID"
    assert v["boundary"]["n_contra_levels"] == 1 and v["boundary"]["censored"]


def test_v1_register_reads_bootstrap_values_without_a_configuration(tmp_path):
    v, ctx = _judge(tmp_path, {"calib_v2": {"stage": "bootstrap"}})
    assert ctx.config is None and v["launch_verdict"] == "VALID"
    assert _tier(v, 4)["X"]["X2"][0]["threshold"]["form"] == "lag_const"
    assert ctx.thresholds.lookups[LAG + "|-|subsat"] == "found"


@pytest.mark.parametrize("levels,match", [
    ([{"level": 1, "load_value": 40.0, "up_seq": 4, "down_seq": None}], "numeric rho"),
    ([{"level": "1", "rho": 1.0, "load_value": 40.0, "up_seq": 4, "down_seq": None}],
     "int level"),
    ([{"level": 1, "rho": 1.0, "load_value": 41.0, "up_seq": 4, "down_seq": None}],
     r"level 1 load_value 41.0 != tier 4 load_value 40.0"),
    ([{"level": 1, "rho": 0.5, "load_value": 20.0, "up_seq": 2, "down_seq": None},
      {"level": 1, "rho": 1.0, "load_value": 40.0, "up_seq": 4, "down_seq": None}],
     "level numbers must be distinct"),
])
def test_ladder_levels_need_level_rho_and_the_tier_load(tmp_path, levels, match):
    M = _make_samples()
    W = M.build(os.path.join(str(tmp_path), "w"), "good",
                _v2({"levels": levels, "baseline_seq": None if len(levels) > 1 else 2}), {})
    with open(os.path.join(W, "expect.json")) as f:
        tree = os.path.join(W, json.load(f)["tree"])
    with pytest.raises(I.InputInvalid, match=match):
        I.verify(W, tree)

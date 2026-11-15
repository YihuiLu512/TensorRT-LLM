# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibration v2 threshold sources and per-arm registers (calib-v2 DESIGN-r7a 3.1 P2-02,
P2-03; 3.5 tests 17, 25, the judge side of 34 and the inline-thresholds item of 22).

capjudge.nullz: pilot/2 (thresholds and bootstrap stages) and d14_result, v1 files rejected.
capjudge.inputs: arm_config / arm_threshold_config, the per-arm ladder.json and the tier regime
from its roles, v1 registers unchanged, inline thresholds refused in a v2 register."""

import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest
from conftest import SAMPLES, judge_mod

Z = judge_mod("nullz")
I = judge_mod("inputs")
K = judge_mod("calib")

REAL = os.path.join(SAMPLES, "calibv2_real")
REAL_SHA = {"e1_0_http.register.json":
            "242f52af1ba44c2b271e4dc795054b26c986375e51f9af9ceea8c4fc927131b4",
            "e0_round1.pilot.json":
            "8e823f128a3c6104c3c6f4fdd8931c148f695ed9f2f8126bdc38b2b4464ea39a",
            "e0_round1.register.json":
            "7c0f268cc8431bbcc7358f3f081a3c2d0803594355ca74a22ba3ce9bc60f4fa8"}
ENV = {"hw_profile_sha": "d" * 64, "image_sha": "e" * 64, "capkit_sha": "c" * 40,
       "baseline_sha": "f" * 10}
D14_REF = {"relpath": "e0_d14.json", "sha256": "1" * 64}
LAG = "T1:req_first_lag_p50_ns:relay1"
COUNT = "T1:backlog.inflight:relay1"
RATIO = "T2:delivery_ratio_norm:relay1"
MS = 1_000_000


def _real(name):
    p = os.path.join(REAL, name)
    with open(p, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == REAL_SHA[name], name
    return p


def _make_samples():
    sys.path.insert(0, SAMPLES)
    try:
        import make_samples
    finally:
        sys.path.remove(SAMPLES)
    return make_samples


# ---- capjudge.nullz: documents ------------------------------------------------------------------

def _boot(**kw):
    d = {"schema": "capkit.pilot/2", "experiment": "E0", "stage": "bootstrap",
         "environment": dict(ENV, judge_sha="b" * 40), "d14": dict(D14_REF),
         "select": "capjudge.pilot.select_tiers", "rule": "r", "inputs": [],
         "x_const": {"lag": {"z": 10.0, "min_D_ns": 10 * MS}, "count": {"z": 10.705}},
         "ratio_tol": {"T2:delivery_ratio_norm": {"min_per_request": 2, "tol": 0.02}},
         "c13_cap": {"T3:foreign_busy_frac": {"window": 0.10, "launch_mean": 0.05}}}
    d.update(kw)
    return d


def _thr(**kw):
    d = {"schema": "capkit.pilot/2", "experiment": "E0", "stage": "thresholds",
         "environment": dict(ENV, judge_sha="b" * 40), "d14": dict(D14_REF),
         "window_ns": 60_000_000_000, "inputs": [],
         "null_z": {LAG + "|s1@1ms|near|L=60": {"family": "lag", "thr_hi": 3.5, "thr_lo": -3.1,
                                                "uncalibrated": False},
                    LAG + "|s1@1ms|subsat|L=60": {"family": "lag", "thr_hi": 2.5,
                                                  "thr_lo": -2.4, "uncalibrated": False},
                    COUNT + "|s1@1ms|near|L=60": {"family": "count", "thr_hi": None,
                                                  "thr_lo": None, "uncalibrated": True}},
         "ratio": {RATIO + "|*|*": {"abs_tol": 0.01, "k": 1, "n_eff": 300}},
         "c13_cap": {"T3:foreign_busy_frac": {"window": 0.10, "launch_mean": 0.06}},
         "i1_calibration": {"near_one_core_min": 0.93, "source_tiers": [], "n": 4}}
    d.update(kw)
    return d


def _d14(**kw):
    d = {"schema": "capkit.d14_result/1", "rule": {"relpath": "d14_rule.json",
                                                   "sha256": "2" * 64},
         "select": "capjudge.pilot.select_tiers", "environment": dict(ENV), "inputs": [],
         "d14_softirq": {"irq_softirq_is_experiment": False, "readings": {}}}
    d.update(kw)
    return d


def test_bootstrap_returns_the_family_value_for_any_config_and_regime():
    th = Z.Thresholds()
    th.add_source("pilot", "boot.json", "a" * 64, _boot())
    assert th.stage == "bootstrap"
    for cfg in ("s1@1ms", "s4", "anything-else"):
        for rg in ("subsat", "near_boundary"):
            lag = th.get(LAG, cfg, rg)
            assert (lag["form"], lag["z"], lag["min_D_ns"]) == ("lag_const", 10.0, 10 * MS)
            assert th.get("T1:send_lag_p50_ns:ld", cfg, rg)["form"] == "lag_const"
            assert th.get("T1:msg_lag_p50_ns:ld", cfg, rg)["form"] == "lag_const"
            cnt = th.get(COUNT, cfg, rg)
            assert (cnt["form"], cnt["z"]) == ("count_const", 10.705)
            assert th.get("T1:inflight_derived:ld", cfg, rg)["z"] == 10.705
            r = th.get(RATIO, cfg, rg)
            assert (r["form"], r["tol"], r["min_per_request"]) == ("ratio_tol", 0.02, 2)
    # no family -> no threshold (I offsets, T3 readings)
    assert th.get("T2:core_busy_frac_softirq:relay1", "s1@1ms", "near_boundary") is None
    assert th.lookups["T2:core_busy_frac_softirq:relay1|s1@1ms|near_boundary"] == "missing"
    assert th.lookups[LAG + "|s4|subsat"] == "found"
    assert th.cap("T3:foreign_busy_frac") == {"window": 0.10, "launch_mean": 0.05}
    assert th.cap("T3:other") is None
    assert th.i1_near() is None and th.d14() is None
    assert th.sources == [{"kind": "pilot", "relpath": "boot.json", "sha256": "a" * 64,
                           "schema": "capkit.pilot/2", "stage": "bootstrap"}]


def test_compare_bootstrap_forms():
    th = Z.Thresholds()
    th.add_source("pilot", "boot.json", "a" * 64, _boot())
    lag = th.get(LAG, "c", "near_boundary")
    # lag: above iff z >= z_lag AND D >= min_D_ns (both inclusive), never below
    assert Z.compare(10.0, lag, D=10 * MS) == "above"
    assert Z.compare(9.999, lag, D=50 * MS) == "inside"
    assert Z.compare(500.0, lag, D=10 * MS - 1) == "inside"
    assert Z.compare(-500.0, lag, D=-50 * MS) == "inside"
    with pytest.raises(ValueError):
        Z.compare(12.0, lag)
    cnt = th.get(COUNT, "c", "subsat")
    assert Z.compare(10.705, cnt) == "inside" and Z.compare(-10.705, cnt) == "inside"
    assert Z.compare(10.706, cnt) == "above" and Z.compare(-10.706, cnt) == "below"
    r = th.get(RATIO, "c", "subsat")
    # tolerance max(2 / planned_requests, tol): 0.04 at 50 requests, 0.02 at 1145
    assert Z.compare(0.03, r, planned_requests=50) == "inside"
    assert Z.compare(0.03, r, planned_requests=1145) == "above"
    assert Z.compare(-0.03, r, planned_requests=1145) == "below"
    assert Z.compare(0.041, r, planned_requests=50) == "above"
    with pytest.raises(ValueError):
        Z.compare(0.03, r)
    for missing in (None, Z.stats.NotRun("too_few_diffs")):
        assert Z.compare(missing, cnt) is None
    assert Z.compare(3.0, None) is None


def test_thresholds_stage_reads_keys_per_config_and_regime():
    th = Z.Thresholds()
    th.add_source("pilot", "p2.json", "a" * 64, _thr())
    near = th.get(LAG, "s1@1ms", "near_boundary")
    assert near["form"] == "null_z" and near["key"] == LAG + "|s1@1ms|near|L=60"
    assert (near["thr_hi"], near["thr_lo"]) == (3.5, -3.1)
    assert th.get(LAG, "s1@1ms", "subsat")["thr_hi"] == 2.5
    assert th.get(LAG, "s1@1.003ms", "near_boundary") is None  # no key of that config
    assert th.lookups[LAG + "|s1@1.003ms|near_boundary"] == "missing"
    assert Z.compare(3.5, near) == "inside" and Z.compare(3.51, near) == "above"
    assert Z.compare(-3.11, near) == "below"
    unc = th.get(COUNT, "s1@1ms", "near_boundary")
    assert unc["uncalibrated"] and Z.compare(1e9, unc) is None
    assert th.lookups[COUNT + "|s1@1ms|near_boundary"] == "uncalibrated"
    r = th.get(RATIO, "any", "subsat")
    assert r["form"] == "ratio_abs" and Z.compare(0.011, r, planned_requests=1000) == "above"
    assert Z.compare(-0.01, r, planned_requests=1000) == "inside"
    # DESIGN-r7a 2.9: the V2 tolerance has the floor 1 / planned requests (0.02 at 50)
    assert Z.compare(0.015, r, planned_requests=50) == "inside"
    assert Z.compare(-0.021, r, planned_requests=50) == "below"
    for bad in (None, 0, True, 2.0):
        with pytest.raises(ValueError, match="ratio_abs comparison needs planned_requests"):
            Z.compare(0.011, r, planned_requests=bad)
    assert th.i1_near() == 0.93
    assert th.cap("T3:foreign_busy_frac")["launch_mean"] == 0.06
    with pytest.raises(ValueError):
        th.get(LAG, "s1@1ms", "near")  # the judge's regime words only
    with pytest.raises(ValueError):
        th.get("T1:req_first_lag_p50_ns", "s1@1ms", "subsat")
    with pytest.raises(ValueError):
        th.get(LAG, "a|b", "subsat")
    assert Z.Thresholds().get(LAG, "s1@1ms", "subsat") is None  # no pilot/2 at all


def test_d14_result_and_its_link_to_the_pilot():
    th = Z.Thresholds()
    th.add_source("d14_result", "e0_d14.json", "1" * 64, _d14())
    assert th.d14() == {"irq_softirq_is_experiment": False, "readings": {}}
    th.add_source("pilot", "boot.json", "a" * 64, _boot())  # d14 ref sha "1" * 64: agrees
    for first, second in ((("d14_result", "x.json", "9" * 64, _d14()),
                           ("pilot", "boot.json", "a" * 64, _boot())),
                          (("pilot", "boot.json", "a" * 64, _boot()),
                           ("d14_result", "x.json", "9" * 64, _d14()))):
        th = Z.Thresholds()
        th.add_source(*first)
        with pytest.raises(Z.ThresholdError, match="names D14 file sha"):
            th.add_source(*second)
    th = Z.Thresholds()
    th.add_source("pilot", "boot.json", "a" * 64, _boot())
    with pytest.raises(Z.ThresholdError, match="two calibration files of kind pilot"):
        th.add_source("pilot", "thr.json", "b" * 64, _thr())


def _broken():
    def boot(f):
        d = _boot()
        f(d)
        return ("pilot", d)

    def thr(f):
        d = _thr()
        f(d)
        return ("pilot", d)

    def d14(f):
        d = _d14()
        f(d)
        return ("d14_result", d)
    nz = lambda d: d["null_z"]  # noqa: E731
    return {
        "stage": boot(lambda d: d.update(stage="pilot")),
        "schema": boot(lambda d: d.update(schema="capkit.pilot/3")),
        "env judge_sha": boot(lambda d: d["environment"].pop("judge_sha")),
        "d14 ref": boot(lambda d: d.update(d14={"relpath": "x"})),
        "c13 missing": boot(lambda d: d.pop("c13_cap")),
        "c13 zero": boot(lambda d: d["c13_cap"]["T3:foreign_busy_frac"].update(window=0)),
        "x_const count": boot(lambda d: d["x_const"].pop("count")),
        "lag z bool": boot(lambda d: d["x_const"]["lag"].update(z=True)),
        "lag D negative": boot(lambda d: d["x_const"]["lag"].update(min_D_ns=-1)),
        "ratio_tol mpr": boot(lambda d: d["ratio_tol"]["T2:delivery_ratio_norm"].update(
            min_per_request=0)),
        "ratio_tol key": boot(lambda d: d.update(ratio_tol={"T2:x": {"min_per_request": 2,
                                                                     "tol": 0.0}})),
        "window": thr(lambda d: d.update(window_ns=60_500_000_000)),
        "null_z L": thr(lambda d: nz(d).update({LAG + "|c|near|L=20": dict(
            nz(d)[LAG + "|s1@1ms|near|L=60"])})),
        "null_z regime": thr(lambda d: nz(d).update({LAG + "|c|near_boundary|L=60": dict(
            nz(d)[LAG + "|s1@1ms|near|L=60"])})),
        "null_z family": thr(lambda d: nz(d)[LAG + "|s1@1ms|near|L=60"].update(family="x")),
        "null_z lo>hi": thr(lambda d: nz(d)[LAG + "|s1@1ms|near|L=60"].update(thr_lo=9)),
        "null_z no thr": thr(lambda d: nz(d)[LAG + "|s1@1ms|near|L=60"].update(thr_hi=None)),
        "ratio key": thr(lambda d: d.update(ratio={RATIO + "|s1@1ms|near": {"abs_tol": 0.1}})),
        # a key with a trailing newline is not the key (anchors are \Z, not $)
        "null_z newline": thr(lambda d: nz(d).update({LAG + "|c|near|L=60\n": dict(
            nz(d)[LAG + "|s1@1ms|near|L=60"])})),
        "ratio newline": thr(lambda d: d.update(ratio={RATIO + "|*|*\n": {"abs_tol": 0.1}})),
        "i1": thr(lambda d: d.update(i1_calibration={"near_one_core_min": 1.5,
                                                     "source_tiers": [], "n": 1})),
        "d14 schema": d14(lambda d: d.update(schema="capkit.d14_result/2")),
        "d14 bool": d14(lambda d: d["d14_softirq"].update(irq_softirq_is_experiment=None)),
        "d14 env": d14(lambda d: d["environment"].pop("capkit_sha")),
        "d14 rule": d14(lambda d: d.pop("rule")),
        "kind": ("loadgen_band", _boot()),
    }


@pytest.mark.parametrize("case", sorted(_broken()))
def test_malformed_v2_sources_rejected(case):
    kind, doc = _broken()[case]
    with pytest.raises(Z.ThresholdError):
        Z.Thresholds().add_source(kind, "f.json", "a" * 64, doc)


def test_v1_files_are_not_threshold_sources():
    pilot1 = {"schema": "capkit.pilot/1", "environment": dict(ENV), "noise_bands": {}}
    for kind, doc in (("pilot", pilot1), ("near_boundary_noise", pilot1),
                      ("near_boundary_noise", _boot())):
        with pytest.raises(Z.ThresholdError) as e:
            Z.Thresholds().add_source(kind, "f.json", "a" * 64, doc)
        assert str(e.value) == "v1 calibration is not a threshold source (calib-v2 erratum)"
    with open(_real("e0_round1.pilot.json")) as f:
        real = json.load(f)
    assert real["schema"] == "capkit.pilot/1"
    with pytest.raises(Z.ThresholdError, match="v1 calibration is not a threshold source"):
        Z.Thresholds().add_source("pilot", "02_e0/E0/analysis/pilot.json", REAL_SHA[
            "e0_round1.pilot.json"], real)
    # the v1 builders still validate their own output
    assert K.validate("pilot", real, "built") is None


def test_family_of():
    assert Z.family_of(LAG) == "lag" and Z.family_of(COUNT) == "count"
    assert Z.family_of(RATIO) == "ratio"
    assert Z.family_of("T3:foreign_busy_frac:relay1") is None
    assert Z.family_of("T2:req_first_lag_p50_ns:relay1") is None  # lag quantities are T1
    with pytest.raises(ValueError):
        Z.family_of("T9:x:y")


# ---- capjudge.inputs on generated samples -------------------------------------------------------

ARM = "synthetic-a"


def _v2(configs=None, ladders=None, **kw):
    v2 = {"configs": configs or {ARM: "s1@1ms"},
          "ladders": {ARM: {}} if ladders is None else ladders}
    v2.update(kw)
    return {"calib_v2": {"stage": "bootstrap"}, "v2": v2}


def _build(tmp_path, opts):
    M = _make_samples()
    n = len([x for x in os.listdir(tmp_path) if x.startswith("w")]) if os.path.isdir(
        tmp_path) else 0
    out = os.path.join(str(tmp_path), "w%02d" % n)
    W = M.build(out, "good", opts, M.SAMPLES["good"][1])
    with open(os.path.join(W, "expect.json")) as f:
        return W, os.path.join(W, json.load(f)["tree"])


def _verify(tmp_path, opts):
    return I.verify(*_build(tmp_path, opts))


def _invalid(tmp_path, opts, match):
    with pytest.raises(I.InputInvalid, match=match) as e:
        _verify(tmp_path, opts)
    return e.value


def _register(W):
    with open(os.path.join(W, "register", "register.json")) as f:
        return json.load(f)


def test_e1_0_register_naming_the_round_1_pilot_is_input_invalid(tmp_path):
    # test 17: the E1-0 register's calibration entry, verbatim, with the round-1 pilot.json
    # (sha 8e823f12...) at its relpath
    with open(_real("e1_0_http.register.json")) as f:
        entries = json.load(f)["calibration"]
    assert entries == [{"kind": "pilot", "relpath": "02_e0/E0/analysis/pilot.json",
                        "sha256": REAL_SHA["e0_round1.pilot.json"]}]
    e = _invalid(tmp_path, {"calibration_entries": entries, "calibration_files": {
        entries[0]["relpath"]: _real("e0_round1.pilot.json")}},
        r"v1 calibration is not a threshold source \(calib-v2 erratum\)")
    assert e.item == "register.calibration[0]"


@pytest.mark.parametrize("n", [1, 2])
def test_near_boundary_noise_registrations_are_input_invalid(tmp_path, n):
    nb = {"schema": "capkit.near_boundary_noise/1", "regime": "near_boundary",
          "environment": dict(_make_samples().ENV), "inputs": [], "noise_bands": {}}
    files, entries = {}, []
    for i in range(n):
        src = tmp_path / ("nb%d.json" % i)
        src.write_text(json.dumps(dict(nb, note=i)))
        rel = "register/nb%d.json" % i
        files[rel] = str(src)
        entries.append({"kind": "near_boundary_noise", "relpath": rel,
                        "sha256": hashlib.sha256(src.read_bytes()).hexdigest()})
    _invalid(tmp_path, {"calibration_entries": entries, "calibration_files": files},
             r"v1 calibration is not a threshold source \(calib-v2 erratum\)")


def test_v2_sample_with_a_bootstrap_pilot_and_d14_result(tmp_path):
    ctx = _verify(tmp_path, _v2())
    assert ctx.config == ctx.threshold_config == "s1@1ms"
    assert [s["kind"] for s in ctx.thresholds.sources] == ["pilot", "d14_result"]
    assert ctx.thresholds.stage == "bootstrap"
    assert ctx.thresholds.d14()["irq_softirq_is_experiment"] is True
    # one family value whatever the configuration and regime
    vals = {json.dumps(ctx.thresholds.get("T1:req_first_lag_p50_ns:capload", c, r),
                       sort_keys=True)
            for c in ("s1@1ms", "other") for r in ("subsat", "near_boundary")}
    assert len(vals) == 1
    assert ctx.bands.sources == [{"kind": "loadgen_band", "relpath": "register/loadgen_band.json",
                                  "sha256": ctx.bands.sources[0]["sha256"], "n_keys": 0}]
    assert {"W:register/pilot2.json", "W:register/e0_d14.json"} <= set(ctx.files_read)


@pytest.mark.parametrize("field", ["hw_profile_sha", "image_sha"])
def test_d14_result_from_another_environment_is_input_invalid(tmp_path, field):
    # test 34, the judge side
    opts = _v2()
    opts["calib_v2"]["d14_env"] = {field: "1" * 64}
    e = _invalid(tmp_path, opts, "calibration environment")
    assert e.item == "register.calibration[1]"


def test_inline_thresholds_in_a_v2_register_are_input_invalid(tmp_path):
    # test 22 item; a v1 register (no arm_config) still reads them (E0)
    inline = {"noise_bands": {"T1:req_first_lag_p50_ns:capload": {"min": -1, "max": 1,
                                                                   "n": 3}}}
    e = _invalid(tmp_path, dict(_v2(), inline=inline), "inline thresholds are not a "
                 r"calibration file \(CONSTRAINTS 4.2; calib-v2 r7\)")
    assert e.item == "register.inline_thresholds"
    ctx = _verify(tmp_path, dict(_v2(), inline={}))  # empty: allowed
    assert ctx.bands.get("T1:req_first_lag_p50_ns:capload", "subsat") is None
    ctx = _verify(tmp_path, {"calib_v2": {"stage": "bootstrap"}, "inline": inline})
    assert ctx.config is None
    assert ctx.bands.get("T1:req_first_lag_p50_ns:capload", "subsat")["max"] == 1


# ---- test 25: per-arm registers and the regime from ladder roles -----------------------------

def test_two_arms_each_with_its_own_ladder(tmp_path):
    configs = {"s1-n01": "s1@1ms", "s4-n01": "s4"}
    ladders = {"s1-n01": {}, "s4-n01": {"levels": [], "near": [{"rho": 1.0, "seq": 4,
                                                                "pass": 1}]}}
    seen = {}
    for arm in configs:
        opts = dict(_v2(configs, ladders), arm=arm, extra_arms=[a for a in configs if a != arm])
        ctx = _verify(tmp_path, opts)
        seen[arm] = (ctx.config, ctx.ladder["config"], ctx.tier_role, ctx.tier_regime)
    assert seen["s1-n01"] == ("s1@1ms", "s1@1ms", {2: "baseline", 4: "levels"},
                              {2: "subsat", 4: "near_boundary"})
    assert seen["s4-n01"] == ("s4", "s4", {2: "baseline", 4: "near"},
                              {2: "subsat", 4: "near_boundary"})


def test_arm_names_are_never_parsed(tmp_path):
    # identity.ARM_RE allows [A-Za-z0-9._-] only, so the design's "<config>@<block>.<slot>"
    # names are written with '-' here; any two valid names give the same reading
    out = []
    for arm in ("s1-1ms.b07.1", "zz-arbitrary-name"):
        opts = _v2({arm: "s1@1.003ms"}, {arm: {}}, threshold_configs={arm: "s1@1ms"})
        ctx = _verify(tmp_path, dict(opts, arm=arm))
        out.append((ctx.config, ctx.threshold_config, ctx.tier_role, ctx.tier_regime,
                    {k: v for k, v in ctx.ladder.items() if k != "arm"}))
    assert out[0] == out[1]


@pytest.mark.parametrize("over,roles,regimes", [
    ({}, {2: "baseline", 4: "levels"}, {2: "subsat", 4: "near_boundary"}),
    ({"baseline_seq": None, "levels": [], "cal": [{"rho": 0.3, "seq": 2, "pass": 1},
                                                  {"rho": 0.4, "seq": 4, "pass": 1}]},
     {2: "cal", 4: "cal"}, {2: "subsat", 4: "subsat"}),
    ({"levels": [], "near": [{"rho": 0.95, "seq": 4, "pass": 1}]}, {2: "baseline", 4: "near"},
     {2: "subsat", 4: "near_boundary"}),
    ({"levels": [], "holds": [{"rho": 1.0, "seq": 4, "hold_ns": 600}]},
     {2: "baseline", 4: "holds"}, {2: "subsat", 4: "near_boundary"}),
    ({"levels": [], "slope": [{"rho": 1.05, "seq": 4}]}, {2: "baseline", 4: "slope"},
     {2: "subsat", 4: "near_boundary"}),
])
def test_regime_from_ladder_roles(tmp_path, over, roles, regimes):
    ctx = _verify(tmp_path, _v2(ladders={ARM: over}))
    assert ctx.tier_role == roles and ctx.tier_regime == regimes


def test_v2_arm_without_a_ladder_judges_every_load_tier_subsat(tmp_path):
    ctx = _verify(tmp_path, _v2(ladders={}))
    assert ctx.ladder is None and ctx.tier_role == {}
    assert ctx.tier_regime == {2: "subsat", 4: "subsat"}
    assert "ladder" not in _register(ctx.W)["files_by_arm"][ARM]


@pytest.mark.parametrize("over,match", [
    ({"levels": []}, r"load tiers \[4\] are named by no ladder role"),
    ({"cal": [{"rho": 0.5, "seq": 4, "pass": 1}]}, "seq 4 named by both levels and cal"),
    ({"slope": [{"rho": 1.0, "seq": 3}]}, "slope seq 3 must be a load"),
    ({"levels": [{"level": 1, "rho": 1.0, "load_value": 40.0, "up_seq": 4, "down_seq": 2}]},
     "levels seq 2 must be a down"),
    ({"levels": [{"level": 1, "up_seq": "4", "down_seq": None}]}, "int up_seq"),
    ({"holds": None}, "holds must be a list"),
    ({"config": "s4"}, r"config 's4' != register.json arm_config"),
    ({"schema": "capkit.ladder/2"}, "schema must be capkit.ladder/1"),
    ({"baseline_seq": 9}, "baseline seq 9 must be a load"),
    # a ladder of another tier table (same layout) or of another arm (R2 F22)
    ({"tier_table_sha256": "0" * 64}, "tier_table_sha256 '0+' != the registered tier table"),
    ({"tier_table_sha256": None}, "tier_table_sha256 None != the registered tier table"),
    ({"arm": "other-arm"}, "arm 'other-arm' != this launch's arm"),
])
def test_bad_ladder_is_input_invalid(tmp_path, over, match):
    e = _invalid(tmp_path, _v2(ladders={ARM: over}), match)
    assert e.item == "ladder.json"


def test_ladder_registered_as_another_kind_is_input_invalid(tmp_path):
    e = _invalid(tmp_path, _v2(ladder_kind="scenario"), "registered as kind 'scenario'")
    assert e.item == "register.files_by_arm.%s.ladder" % ARM


def test_ladder_sha_mismatch_is_input_invalid(tmp_path):
    W, tree = _build(tmp_path, _v2())
    rel = _register(W)["files_by_arm"][ARM]["ladder"]["relpath"]
    with open(os.path.join(W, rel), "a") as f:
        f.write("\n")
    with pytest.raises(I.InputInvalid, match="sha256 .* != listed") as e:
        I.verify(W, tree)
    assert e.value.item == "register.files_by_arm.%s.ladder" % ARM


@pytest.mark.parametrize("v2,match", [
    ({"configs": {"other-arm": "s1@1ms"}}, "arm_config must map exactly the arms"),
    ({"configs": {ARM: "s1@1ms", "other-arm": "s4"}}, "arm_config must map exactly the arms"),
    ({"configs": {ARM: "a|b"}}, "arm_config must map exactly the arms"),
    ({"threshold_configs": {ARM: "s1@1ms", "x": "y"}},
     "arm_threshold_config must map exactly"),
    ({"threshold_configs": {}}, "arm_threshold_config must map exactly"),
    ({"nbt": True}, "judge.near_boundary_tiers is not read for a register with arm_config"),
])
def test_bad_v2_register_is_input_invalid(tmp_path, v2, match):
    opts = _v2()
    opts["v2"].update(v2)
    e = _invalid(tmp_path, opts, match)
    assert e.item == "register.json"


def test_resolution_arm_reads_the_base_configs_thresholds(tmp_path):
    # arm_config = its own configuration, arm_threshold_config = s1@1ms; the thresholds pilot/2
    # has null_z keys of s1@1ms only
    opts = _v2({ARM: "s1@1.003ms"}, threshold_configs={ARM: "s1@1ms"})
    opts["calib_v2"] = {"stage": "thresholds", "configs": ["s1@1ms"], "window_s": 4}
    W, tree = _build(tmp_path, opts)
    ctx = I.verify(W, tree)
    assert (ctx.config, ctx.threshold_config) == ("s1@1.003ms", "s1@1ms")
    key = "T1:req_first_lag_p50_ns:capload"
    hit = ctx.thresholds.get(key, ctx.threshold_config, ctx.tier_regime[4])
    assert hit["key"] == key + "|s1@1ms|near|L=4"
    assert ctx.thresholds.get(key, ctx.config, ctx.tier_regime[4]) is None
    # without arm_threshold_config the same arm reads its own configuration: no key
    opts2 = copy.deepcopy(opts)
    del opts2["v2"]["threshold_configs"]
    ctx2 = _verify(tmp_path, opts2)
    assert ctx2.threshold_config == "s1@1.003ms"
    assert ctx2.thresholds.get(key, ctx2.threshold_config, ctx2.tier_regime[4]) is None
    # verdicts.json records both
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(Z.__file__)),
               CAPKIT_CODE_SHA="b" * 40)
    p = subprocess.run([sys.executable, "-m", Z.__name__.split(".")[0], "--tree", tree,
                        "--workdir", W], env=env, capture_output=True, text=True, timeout=300,
                       check=False)
    assert p.returncode in (0, 10, 11), p.stderr
    with open(os.path.join(tree, "judge", "verdicts.json")) as f:
        v = json.load(f)
    assert (v["arm_config"], v["arm_threshold_config"]) == ("s1@1.003ms", "s1@1ms")
    assert {t["tier_seq"]: t["regime"] for t in v["tiers"] if t["judged"]} == {
        2: "subsat", 4: "near_boundary"}
    with open(os.path.join(tree, "judge", "inputs.json")) as f:
        inp = json.load(f)
    assert [s["kind"] for s in inp["thresholds"]] == ["pilot", "d14_result"]


def _load_register(monkeypatch, tmp_path, doc, arm):
    """inputs._load_register on `doc` for `arm`, with every registered file resolved to the
    register itself (the file checks are not under test here)."""
    p = tmp_path / "register.json"
    p.write_text(json.dumps(doc))
    monkeypatch.setattr(I, "_registered_file", lambda ctx, ref, item, kinds=None: str(p))
    files = doc["files"] if "files" in doc else doc["files_by_arm"][arm]
    ctx = I.Ctx(str(tmp_path), str(tmp_path))
    ctx.launch = dict({k: files[k] for k in I.REGISTER_FILE_KEYS}, register={})
    ctx.ident = type("Ident", (), {"experiment": doc["experiment"], "arm": arm})()
    I._load_register(ctx)
    return ctx


def test_round_1_e0_register_without_arm_config_keeps_v1_semantics(monkeypatch, tmp_path):
    with open(_real("e0_round1.register.json")) as f:
        doc = json.load(f)
    assert "arm_config" not in doc
    for arm in doc["arms"]:
        ctx = _load_register(monkeypatch, tmp_path, doc, arm)
        assert (ctx.config, ctx.threshold_config, ctx.ladder_path) == (None, None, None)
        assert ctx.mode == doc["judge"]["mode"]
    # a ladder in a v1 register's files is not read: the file set is checked as before
    bad = copy.deepcopy(doc)
    bad["files_by_arm"][doc["arms"][0]]["ladder"] = {"relpath": "l.json", "sha256": "0" * 64}
    with pytest.raises(I.InputInvalid, match="files must have exactly"):
        _load_register(monkeypatch, tmp_path, bad, doc["arms"][0])
    bad = dict(doc, arm_threshold_config={a: "x" for a in doc["arms"]})
    with pytest.raises(I.InputInvalid, match="arm_threshold_config needs arm_config"):
        _load_register(monkeypatch, tmp_path, bad, doc["arms"][0])


def test_v1_sample_with_v2_sources_judges_per_tier(tmp_path):
    ctx = _verify(tmp_path, {"calib_v2": {"stage": "bootstrap"}})
    assert ctx.config is None and ctx.tier_role == {} and ctx.ladder is None
    assert ctx.tier_regime == {2: "subsat", 4: "subsat"}
    ctx = _verify(tmp_path, {"calib_v2": {"stage": "bootstrap"},
                                "near_boundary_tiers": True})
    assert ctx.tier_regime == {2: "subsat", 4: "near_boundary"}  # near_boundary_tiers, v1


def test_thresholds_window_must_match_the_judged_tiers(tmp_path):
    """A thresholds-stage pilot/2 is for one window length: load tiers of another window are
    INPUT_INVALID (calib-v2 erratum A5 item 6); a bootstrap pilot/2 has no window."""
    opts = _v2({ARM: "s1@1ms"})
    opts["calib_v2"] = {"stage": "thresholds", "configs": ["s1@1ms"], "window_s": 60}
    e = _invalid(tmp_path, opts, "load tiers .* have other windows")
    assert e.item == "register.calibration"
    opts["calib_v2"]["window_s"] = 4
    ctx = _verify(tmp_path, opts)
    assert ctx.calib_docs["pilot"]["window_ns"] == 4 * 10 ** 9
    # a hold tier of another window is not checked (its X conclusion is not used)
    opts["calib_v2"]["window_s"] = 60
    opts["v2"]["ladders"] = {ARM: {"levels": [], "holds": [{"rho": 0.95, "seq": 4,
                                                            "hold_ns": 600 * 10 ** 9}],
                                   "baseline_seq": None, "cal": [{"rho": 0.3, "seq": 2,
                                                                  "pass": 1}]}}
    with pytest.raises(I.InputInvalid, match=r"load tiers \{2: 4000000000\}"):
        _verify(tmp_path, opts)
    assert _verify(tmp_path, dict(opts, calib_v2={"stage": "bootstrap"})).ladder is not None

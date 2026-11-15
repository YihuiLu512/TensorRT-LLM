# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""caplaunch.laddergen (calib-v2 DESIGN-r7a P2-07; tests 12 and 33 of its 3.5): tier tables,
ladder.json, drain slots, packing against the phase's e_c, K_fit, job plans (no processes)."""

import hashlib
import json
import math
import os
import types

import pytest

import conftest as cf
from capcommon import jobplan, ledger
from capcommon.hashing import sha256_file
from capcommon.tiers import TierTable
from caplaunch import laddergen as L
from caplaunch import tiergen

TEMPLATE = os.path.join(cf.C, "configs", "e0", "e0b-s1-single.tiers.template.json")
BASE_VALUES = {"gate_poll_ns": 50 * cf.MS, "gate_settle_polls": 3,
               "gate_deadline_guard_ns": 100 * cf.MS, "gate_max_snapshot_age_ns": 200 * cf.MS,
               "arrival_seed": 7, "request_timeout_ns": 60_000 * cf.MS, "beat_ns": 10 * cf.MS,
               "n_tokens": 32, "delta_ns": cf.MS}
R4_SFIT = os.path.join(cf.C, "samples", "calibv2_real", "plan_r4_S_fit_by_K.json")
C0 = 19.09


@pytest.fixture
def w(tmp_path):
    d = tmp_path / "W"
    d.mkdir()
    bv = tmp_path / "base.values.json"
    bv.write_text(json.dumps({"schema": tiergen.VALUES_SCHEMA, "values": BASE_VALUES}))
    return types.SimpleNamespace(root=str(tmp_path), w=str(d), base_values=str(bv))


def argv(w, kind, out, *extra, phase="V0", center=("--center-literal", str(C0)), **kw):
    a = ["--kind", kind, "--phase", phase, "--config", kw.get("config", "s1@1ms"),
         "--arm", kw.get("arm", "s1"), "--name", kw.get("name", "t-" + kind),
         "--base-template", TEMPLATE, "--base-values", w.base_values, "--workdir", w.w,
         "--out-dir", out, "--launches-per-job", str(kw.get("per_job", 1))] + list(center)
    return a + list(extra)


def ns(w, kind, out, *extra, **kw):
    return L.build_parser().parse_args(argv(w, kind, out, *extra, **kw))


LADDER = ("--settle-s", "90", "--K", "24", "--span-half", "0.03", "--seed-base", "E0v2-t",
          "--block-id", "b01", "--slot", "1")


def gen(w, kind, *extra, **kw):
    files, summary = L.generate(ns(w, kind, w.root, *extra, **kw))
    name = kw.get("name", "t-" + kind)
    tpl = json.loads(files[name + ".tiers.template.json"])
    vals = json.loads(files[name + ".values.json"])["values"]
    lad = json.loads(files[name + ".ladder.json"])
    return files, summary, tpl, vals, lad, tiergen.expand(tpl, vals)


def register(w, rel, doc, kind="e0_params"):
    p = os.path.join(w.w, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(doc, f)
    ledger.registry_append(os.path.join(w.w, "REGISTRY.tsv"), "E0", rel, sha256_file(p),
                           "kind=%s;" % kind)
    return p


def judge_accepts(lad_doc, doc, tmp, config="s1@1ms", arm="s1"):
    """capjudge.inputs._load_ladder (the judge's own check) on the generated pair."""
    from capjudge import inputs  # noqa: PLC0415
    tp, lp = os.path.join(tmp, "tt.json"), os.path.join(tmp, "lad.json")
    with open(tp, "w") as f:
        json.dump(doc, f)
    with open(lp, "w") as f:
        json.dump(lad_doc, f)
    ctx = types.SimpleNamespace(ladder_path=lp, config=config, tt=TierTable.load(tp),
                                ident=types.SimpleNamespace(arm=arm))
    return inputs._load_ladder(ctx)


# ---- test 12: determinism, loads, seed, tiergen, ramp, drain slots ------------------------------

def test_same_seed_same_bytes_and_tiergen_writes_the_recorded_table(w, tmp_path):
    a1, _ = L.generate(ns(w, "ladder", w.root, *LADDER))
    a2, _ = L.generate(ns(w, "ladder", w.root, *LADDER))
    assert a1 == a2
    out = tmp_path / "o"
    out.mkdir()
    assert L.main(argv(w, "ladder", str(out), *LADDER)) == 0
    assert L.main(argv(w, "ladder", str(out), *LADDER)) == 3  # never overwrites
    tt = str(out / "t-ladder.tiers.json")
    assert tiergen.main(["--template", str(out / "t-ladder.tiers.template.json"),
                         "--values", str(out / "t-ladder.values.json"), "--out", tt]) == 0
    lad = json.load(open(out / "t-ladder.ladder.json"))
    assert sha256_file(tt) == lad["tier_table_sha256"]
    TierTable.load(tt)
    other, _ = L.generate(ns(w, "ladder", w.root, *LADDER[:-1], "2"))
    assert other != a1  # another slot, another u


def test_ladder_levels_loads_ramp_and_roles(w, tmp_path):
    _f, _s, _t, _v, lad, doc = gen(w, "ladder", *LADDER)
    tiers = doc["tiers"]
    u = lad["u"]
    sha = hashlib.sha256(b"E0v2-t|s1@1ms|b01|1").hexdigest()
    assert lad["seed"]["sha"] == sha and u == int(sha[:13], 16) / 16 ** 13
    assert len(lad["levels"]) == 24 and lad["passes"] == 2
    for k, lv in enumerate(lad["levels"]):
        rho = 1 - 0.03 + 0.0025 * (k + u)
        assert lv["level"] == k and abs(lv["rho"] - rho) <= 1e-12
        for q in (lv["up_seq"], lv["down_seq"]):
            assert abs(tiers[q]["load_value"] - rho * C0) <= 1e-9 * rho * C0
            assert tiers[q]["ramp_ns"] == (4 + 90) * 10 ** 9
        assert tiers[lv["up_seq"]]["load_value"] == tiers[lv["down_seq"]]["load_value"]
        assert tiers[lv["up_seq"]]["kind"] == "up" and tiers[lv["down_seq"]]["kind"] == "down"
    assert [s["rho"] for s in lad["slope"]] == [1.05, 1.10, 1.20]
    assert all(tiers[s["seq"]]["ramp_ns"] == 8 * 10 ** 9 for s in lad["slope"])
    assert tiers[lad["baseline_seq"]]["baseline"] is True
    assert tiers[lad["baseline_seq"]]["load_value"] == pytest.approx(0.3 * C0, rel=1e-12)
    assert doc["axis"]["load_key"] == "rate_rps"
    assert all(t["streams_c"] is None for t in tiers)
    judge_accepts(lad, doc, str(tmp_path))


def test_u_is_close_to_uniform_across_slots():
    us = [L.seed_u("E0v2-V2b", "s1@1ms", "AA-b%02d" % b, s)[0] for b in range(500)
          for s in (1, 2)]
    assert abs(sum(us) / len(us) - 0.5) < 0.03
    bins = [sum(1 for u in us if k / 10 <= u < (k + 1) / 10) for k in range(10)]
    assert max(bins) < 140 and min(bins) > 60  # 100 expected per bin


def test_drain_slots_follow_the_formula_with_the_phase_e_c(w):
    _f, _s, _t, _v, lad, doc = gen(w, "ladder", *LADDER)
    tiers = doc["tiers"]
    assert lad["drain_rule"]["e_c"] == 0.30
    for t in tiers[1:]:
        if t["kind"] != "drain":
            continue
        prev = tiers[t["seq"] - 1]
        rho = 0.3 if prev["kind"] == "warmup" else prev["load_value"] / C0
        length = (prev["ramp_ns"] + prev["window_ns"] + prev["tail_ns"]) / 1e9
        want = max(60, math.ceil((1.5 * max(0.0, rho * 1.3 - 1) * length + 30) / 10) * 10)
        assert t["window_ns"] == want * 10 ** 9, t["label"]
    top = tiers[lad["levels"][-1]["up_seq"] + 1]["window_ns"] // 10 ** 9
    assert top == 110  # rho 1.0275 + u h: B = 0.336 x 156 s ~ 52 s -> ceil10(108.7)
    assert L.drain_slot_s(1.5, 0.3, 70, bracket=True) == 180
    assert L.drain_slot_s(0.6, 0.3, 156) == 60


def test_the_centre_comes_only_from_a_registered_e0_params(w, tmp_path):
    doc = {"schema": L.E0_PARAMS_SCHEMA, "stage": "bracket", "by_config": {"s1@1ms": {"c1": 20.0}}}
    p = register(w, "02_e0/E0/register/v2/V1a1/e0_params.bracket.json", doc)
    extra = ("--settle-s", "30")
    _f, _s, _t, _v, lad, d = gen(w, "refine", *extra, phase="V1a2", center=(
        "--center", p, "--center-key", "c1", "--e-c", "0.02"))
    assert lad["c_lad"] == 20.0 and lad["center_source"] == {
        "relpath": "02_e0/E0/register/v2/V1a1/e0_params.bracket.json",
        "sha256": sha256_file(p), "key": "c1"}
    assert [round(lv["rho"], 6) for lv in lad["levels"]][::10] == [0.95, 1.0, 1.05]
    assert all(lv["down_seq"] is None for lv in lad["levels"])
    judge_accepts(lad, d, str(tmp_path))
    bad = dict(doc, by_config={"s1@1ms": {"c1": 21.0}})
    unreg = os.path.join(w.w, "unregistered.json")
    with open(unreg, "w") as f:
        json.dump(bad, f)
    for center, msg in (((unreg, "c1"), "not in REGISTRY"), ((p, "c_hat"), "stage")):
        with pytest.raises(L.LadderError, match=msg):
            L.generate(ns(w, "refine", w.root, *extra, phase="V1a2", center=(
                "--center", center[0], "--center-key", center[1], "--e-c", "0.02")))
    with open(p, "w") as f:  # changed after registration: the sha no longer matches
        json.dump(bad, f)
    with pytest.raises(L.LadderError, match="not in REGISTRY"):
        L.generate(ns(w, "refine", w.root, *extra, phase="V1a2", center=(
            "--center", p, "--center-key", "c1", "--e-c", "0.02")))
    for kind in ("refine", "hold", "null"):
        with pytest.raises(L.LadderError, match="center-literal"):
            L.generate(ns(w, kind, w.root, *extra, "--rho-top", "0.98", phase="V2n"))


# ---- test 12 (r7 / r7a): packing at the phase's e_c, K_fit, V0-cal ------------------------------

def test_v0_full_length_timing_row_fits_at_k24_and_not_at_k26(w, tmp_path):
    """DESIGN-r7a 4.2: K = 24, S = 90 s, e_c = 0.30 fits; K = 26 does not (only fits / does not
    fit is asserted, never the table length)."""
    _f, summary, *_ = gen(w, "ladder", *LADDER)
    assert summary["packing"]["time_budget_s"] <= 14400
    out = tmp_path / "o26"
    out.mkdir()
    k26 = ("--settle-s", "90", "--K", "26", "--span-half", "0.0325", "--seed-base", "E0v2-t",
           "--block-id", "b01", "--slot", "1")
    with pytest.raises(L.LadderError, match="does not fit"):
        L.generate(ns(w, "ladder", str(out), *k26))
    assert L.main(argv(w, "ladder", str(out), *k26)) == 3 and os.listdir(out) == []
    # the same K = 26 table fits at V2-b's e_c (0.01): the refusal comes from the V0 e_c
    L.generate(ns(w, "ladder", w.root, *k26, phase="V2b"))
    with pytest.raises(L.LadderError, match="registers e_c"):
        L.generate(ns(w, "ladder", w.root, *k26, "--e-c", "0.01"))
    with pytest.raises(L.LadderError, match="no registered e_c"):
        L.generate(ns(w, "ladder", w.root, *LADDER, phase="V1a2"))


def test_job_condition_binds_where_the_time_budget_does_not(w):
    """DESIGN 4.1: n (T + 6.5) + job_overhead_s_max <= 14340 is checked on its own: with a
    registered overhead of 1100 s the K = 24 row fails it while capjobs' time budget passes."""
    _f, summary, *_ = gen(w, "ladder", *LADDER)
    assert summary["packing"]["time_budget_s"] <= 14400
    with pytest.raises(L.LadderError, match="job condition"):
        L.generate(ns(w, "ladder", w.root, *LADDER, "--job-overhead-s-max", "1100"))
    with pytest.raises(L.LadderError, match="job-overhead-s-max must be"):
        L.generate(ns(w, "ladder", w.root, *LADDER, "--job-overhead-s-max", "700"))


def test_capjobs_time_budget_is_checked_too(w):
    with pytest.raises(L.LadderError, match="time budget"):
        L.generate(ns(w, "cal", w.root, "--settle-s", "30", "--time-limit-s", "7600",
                      phase="V0cal", per_job=4))
    _f, summary, *_ = gen(w, "cal", "--settle-s", "30", phase="V0cal", per_job=4)
    assert summary["packing"]["time_budget_s"] == 7610  # r7/out/plan_r7.json v0_cal


def test_plan_k_fit_matches_plan_r4(w, capsys):
    want = json.load(open(R4_SFIT))["S_fit_by_K"]
    assert L.s_fit_by_k(L.JOB_OVERHEAD_FLOOR_S) == want
    assert L.main(["--plan", "--settle-s", "90", "--K", "28"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["S_fit_by_K"] == want and out["K_fit"] == 28 and out["fits"] is True
    assert L.k_fit(390, L.JOB_OVERHEAD_FLOOR_S) == 12 and L.k_fit(400, 763.5) is None
    assert L.main(["--plan", "--job-overhead-s-max", "500"]) == 3


def test_cal_kind_tier_order_as_v0_cal(w, tmp_path):
    _f, summary, _t, _v, lad, doc = gen(w, "cal", "--settle-s", "30", phase="V0cal", per_job=4)
    tiers = doc["tiers"]
    loads = [t for t in tiers if t["kind"] in ("up", "down")]
    assert [round(t["load_value"] / C0, 6) for t in loads] == [0.3, 0.3, 0.4, 0.5, 0.6, 0.6, 0.5,
                                                               0.4, 0.3]
    assert [t["kind"] for t in loads] == ["up"] * 5 + ["down"] * 4
    assert loads[0]["seq"] == lad["baseline_seq"]
    assert [(c["rho"], c["pass"]) for c in lad["cal"]] == [
        (0.3, 1), (0.4, 1), (0.5, 1), (0.6, 1), (0.6, 2), (0.5, 2), (0.4, 2), (0.3, 2)]
    assert [c["seq"] for c in lad["cal"]] == [t["seq"] for t in loads[1:]]
    assert lad["levels"] == [] and lad["slope"] == [] and lad["u"] is None
    assert all(t["ramp_ns"] == 34 * 10 ** 9 for t in loads)
    assert summary["table_s"] == 1474  # r7/out/plan_r7.json v0_cal launch_table_s
    judge_accepts(lad, doc, str(tmp_path))


@pytest.mark.parametrize("kind,extra,phase", [
    ("refine", ("--settle-s", "30"), "V1a2"),
    ("bracket", ("--settle-s", "30", "--rho-first", "0.56", "--ratio", "1.25", "--n-levels", "6"),
     "V1a1"),
    ("hold", ("--settle-s", "30", "--seed-base", "E0v2-V1b", "--block-id", "A0001", "--slot", "1"),
     "V1b"),
    ("null", ("--settle-s", "90", "--rho-top", "0.98"), "V2n"),
    ("pc", ("--settle-s", "4"), "V0"),
    ("e2e", ("--settle-s", "30", "--seed-base", "E0v2-V0", "--block-id", "e1", "--slot", "1"),
     "V0")])
def test_every_kind_generates_a_table_the_judge_reads(w, tmp_path, kind, extra, phase):
    center = ("--center-literal", str(C0))
    if kind in ("refine", "hold", "null"):
        stage, key = {"refine": ("bracket", "c1"), "hold": ("capacity", "c_hat"),
                      "null": ("settle", "c_n")}[kind]
        p = register(w, "e0p.%s.json" % kind, {"schema": L.E0_PARAMS_SCHEMA, "stage": stage,
                                               "by_config": {"s1@1ms": {key: C0}}})
        center = ("--center", p, "--center-key", key)
    if phase in ("V1a1", "V1a2"):
        center = center + ("--e-c", "0.02")
    _f, _s, _t, _v, lad, doc = gen(w, kind, *extra, phase=phase, center=center)
    judge_accepts(lad, doc, str(tmp_path))
    tiers = doc["tiers"]
    if kind == "hold":
        assert sorted(h["rho"] for h in lad["holds"]) == [0.95, 0.98, 0.99]
        for h in lad["holds"]:
            t = tiers[h["seq"]]
            assert (t["ramp_ns"], t["window_ns"], h["hold_ns"]) == (0, 600 * 10 ** 9,
                                                                    600 * 10 ** 9)
            assert tiers[h["seq"] - 1]["window_ns"] == 60 * 10 ** 9  # 60 s slot before a hold
    if kind == "null":
        assert [(n["rho"], n["pass"]) for n in lad["near"]] == [
            (0.9, 1), (0.95, 1), (0.98, 1), (0.98, 1), (0.98, 1), (0.98, 2), (0.98, 2),
            (0.98, 2), (0.95, 2), (0.9, 2)]
        assert [(c["rho"], c["pass"]) for c in lad["cal"]] == [(0.3, 2)]
    if kind == "bracket":
        over = [lv for lv in lad["levels"] if lv["rho"] * 1.02 > 1]
        assert over and all(tiers[lv["up_seq"] + 1]["window_ns"] == 180 * 10 ** 9 for lv in over)
        assert round(lad["levels"][-1]["rho"], 3) == 1.709
    if kind == "pc":
        assert [s["rho"] for s in lad["slope"]] == [1.5] and len(lad["cal"]) == 2
    if kind == "e2e":
        assert [round(lv["rho"] - lad["h"] * lad["u"], 9) for lv in lad["levels"]] == [
            0.5, 1.0, 1.5]


def test_hold_order_follows_the_seed(w):
    p = register(w, "e0p.capacity.json", {"schema": L.E0_PARAMS_SCHEMA, "stage": "capacity",
                                          "by_config": {"s1@1ms": {"c_hat": C0}}})
    center = ("--center", p, "--center-key", "c_hat")
    seen = set()
    for slot in range(1, 13):
        args = ("--settle-s", "30", "--seed-base", "E0v2-V1b", "--block-id", "A0001", "--slot",
                str(slot))
        lad = gen(w, "hold", *args, phase="V1b", center=center)[4]
        lad2 = gen(w, "hold", *args, phase="V1b", center=center)[4]
        assert lad == lad2
        seen.add(tuple(h["rho"] for h in lad["holds"]))
    assert len(seen) >= 3


# ---- job plans (DESIGN 3.2, 4.5) and test 33: no replace path -----------------------------------

SPEC = {"schema": L.SPEC_SCHEMA, "phase_id": "V1b-A", "seed_base": "E0v2-V1b-20261115",
        "launches_per_job": 4, "pin": False,
        "configs": [{"config": "s1@1ms", "arm": "s1", "planned": 6, "surplus": 2},
                    {"config": "s2", "arm": "s2", "planned": 6, "surplus": 2}],
        "blocks": [{"comparison": "AA", "configs": [["s1@1ms", "aa1"], ["s1@1ms", "aa2"]],
                    "planned": 3, "surplus": 1}]}


def test_job_plan_mixes_planned_then_surplus_by_the_seed(w, tmp_path):
    items = L.job_plan_items(json.loads(json.dumps(SPEC)))
    assert items == L.job_plan_items(json.loads(json.dumps(SPEC)))
    classes = [it["item_class"] for it in items]
    assert classes == sorted(classes, key=["planned", "surplus"].index)
    orders = [it["order"] for it in items]
    assert sorted(set(orders)) == list(range(1, max(orders) + 1))
    single = [it for it in items if it["block_id"] is None]
    assert sum(it["launches"] for it in single if it["item_class"] == "planned") == 12
    assert sum(it["launches"] for it in single if it["item_class"] == "surplus") == 4
    assert any(len(it["config_counts"]) == 2 for it in single)  # configs mixed in a job
    assert all(not it["pin"] for it in single)
    blocks = [it for it in items if it["block_id"] is not None]
    assert len(blocks) == 8 and all(it["pin"] and it["launches"] == 1 for it in blocks)
    for o in {it["order"] for it in blocks}:
        pair = [it for it in blocks if it["order"] == o]
        assert sorted(it["slot"] for it in pair) == [1, 2] and len({it["block_id"]
                                                                    for it in pair}) == 1
    other = L.job_plan_items(dict(json.loads(json.dumps(SPEC)), seed_base="other"))
    assert [it["config_counts"] for it in other] != [it["config_counts"] for it in items]
    out = tmp_path / "plan.tsv"
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    assert L.main(["--job-plan", str(spec), "--out", str(out)]) == 0
    assert jobplan.read(str(out)) == items
    assert L.main(["--job-plan", str(spec), "--out", str(out)]) == 3  # never overwrites


def _gate_product(w, cls, rel="02_e0/E0/register/v2/V0/e0_params.shortfall.A.r1.json"):
    return register(w, rel, {"schema": L.E0_PARAMS_SCHEMA, "stage": "shortfall", "items": [
        {"item_class": cls, "config": "s1@1ms", "arm": "s1", "launches": 3, "pin": False},
        {"item_class": cls, "config": "s2", "arm": "s2", "launches": 2, "pin": False}]})


def _plan_from(w, product, out):
    return L.main(["--plan-from", product, "--workdir", w.w, "--phase-id", "V0", "--seed-base",
                   "E0v2-V0", "--launches-per-job", "2", "--out", out])


def test_plan_from_only_takes_a_registered_gate_product(w, tmp_path, capsys):
    p = _gate_product(w, "post_gate:shortfall")
    out = str(tmp_path / "sf.tsv")
    assert _plan_from(w, p, out) == 0
    items = jobplan.read(out)
    assert {it["item_class"] for it in items} == {"post_gate:shortfall"}
    assert sum(it["launches"] for it in items) == 5 and [it["order"] for it in items] == [1, 2, 3]
    assert json.loads(capsys.readouterr().out)["gate_product"]["sha256"] == sha256_file(p)
    # not registered, or changed after registration -> fail, nothing written
    unreg = tmp_path / "unreg.json"
    unreg.write_text(open(p).read())
    assert _plan_from(w, str(unreg), str(tmp_path / "x1.tsv")) == 3
    with open(p, "a") as f:
        f.write(" ")
    assert _plan_from(w, p, str(tmp_path / "x2.tsv")) == 3
    assert not os.path.exists(tmp_path / "x1.tsv") and not os.path.exists(tmp_path / "x2.tsv")


@pytest.mark.parametrize("cls", ["replace", "post_gate:replace", "planned", "surplus"])
def test_plan_from_refuses_every_class_but_post_gate(w, tmp_path, cls):
    p = _gate_product(w, cls, rel="gp.%s.json" % cls.replace(":", "_"))
    assert _plan_from(w, p, str(tmp_path / "x.tsv")) == 3
    assert not os.path.exists(tmp_path / "x.tsv")


def test_job_plan_format_has_no_replace_class():
    assert not any("replace" in c or "resubmit" in c for c in jobplan.ITEM_CLASSES)
    it = {"order": 1, "item_id": "V0-0001", "item_class": "replace", "arms": ["s1"],
          "launches": 1, "block_id": None, "slot": None, "config_counts": {"s1": 1}, "pin": False}
    with pytest.raises(jobplan.JobPlanError, match="no replace"):
        jobplan.format_tsv([it])
    with pytest.raises(jobplan.JobPlanError, match="order"):
        jobplan.validate([dict(it, item_class="planned", order=2)])
    with pytest.raises(jobplan.JobPlanError, match="surplus"):
        jobplan.validate([dict(it, item_class="surplus"),
                          dict(it, item_class="planned", order=2, item_id="V0-0002")])
    with pytest.raises(jobplan.JobPlanError, match="share an order"):
        jobplan.validate([dict(it, item_class="planned"),
                          dict(it, item_class="planned", item_id="V0-0002")])

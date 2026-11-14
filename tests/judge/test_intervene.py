# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Intervention test V and the conclusion grade (design 2.6; errata B2-2E): the per-tier
heartbeat (hop counts first), the launch dose, the cross-launch dose-response with its reverse
injection, the per-launch grade and the final grade; plus the judge's mode / localize inputs and
the capjudge.intervene CLI on judged samples."""

import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
from conftest import SAMPLES, TOOLS, judge_mod

from capcommon.inject import InjectSpec  # noqa: E402

V = judge_mod("intervene")
G = judge_mod("grade")
I = judge_mod("inputs")
K = judge_mod("calib")
JENV = dict(os.environ, PYTHONPATH=TOOLS, CAPKIT_CODE_SHA="b" * 40)
SPEC = {"site": "relay.msg", "mode": "busy", "delta_ns": 20000, "every_n": 1, "phase": 0,
        "where": "inline"}


# ---- per-tier heartbeat ----------------------------------------------------------------------

def _hb_ctx(declared=True, workers=1):
    spec = InjectSpec(role="relay1", **SPEC)
    tier = NS(seq=4, max_tokens=3, tokens_per_msg=1, stream_interval=1, inject=[])
    tt = NS(inject_for=lambda role, seq: [spec] if declared and role == "relay1" else [])
    chain = {"name": "m", "kind": "messages", "gate": "product", "include_usage": None,
             "corrections": [], "hops": [{"role": "relay1", "source": "summary.counts.msgs",
                                          "scope": "tier"}]}
    ctx = NS(tt=tt, roles={"relay1": {"workers": workers, "measure_class": "under_test"}},
             schedule_meta={"per_tier": {"4": 10}}, launch={"reconcile": {"chains": [chain]}})
    return ctx, tier


def _entry(**kw):
    e = {"site": "relay.msg", "spec": dict(SPEC), "activation_unchanged": True,
         "hits_delta": 30, "applied_delta": 30, "applied_ns_delta": 30 * 20000 + 5,
         "expected_applied_delta": 30, "aux_pending_at_end": 0}
    e.update(kw)
    return e


def _hb(entries, **kw):
    ctx, tier = _hb_ctx(**kw)
    td = NS(tier=tier, summaries={"relay1": {w: {"inject": [e]} for w, e in
                                             enumerate(entries)}})
    return V.tier_heartbeat(ctx, td)


def test_heartbeat_pass_when_every_count_matches():
    r = _hb([_entry()])
    assert r["status"] == "PASS"
    assert r["injections"][0]["hits_expected"] == [30, 30]  # 10 requests x 3 messages


@pytest.mark.parametrize("kw,why", [
    ({"applied_delta": 29}, "applied 29 != expected 30"),
    ({"hits_delta": 20, "applied_delta": 20, "expected_applied_delta": 20,
      "applied_ns_delta": 20 * 20000}, "passes 20 outside"),
    ({"applied_ns_delta": 1000}, "injected 1000 ns"),
    ({"aux_pending_at_end": 2}, "aux_pending_at_end 2"),
    ({"spec": dict(SPEC, delta_ns=40000)}, "!= declared"),
    ({"hits_delta": 0, "applied_delta": 0, "expected_applied_delta": 0, "applied_ns_delta": 0},
     "no pass was injected"),
])
def test_heartbeat_fail(kw, why):
    r = _hb([_entry(**kw)])
    assert r["status"] == "FAIL" and any(why in w for w in r["injections"][0]["why"])


def test_heartbeat_undeclared_or_unchecked():
    assert _hb([_entry()], declared=False)["status"] == "FAIL"  # observed, not declared
    # declared, the worker's summary is there but does not report it
    ctx, tier = _hb_ctx()
    r = V.tier_heartbeat(ctx, NS(tier=tier, summaries={"relay1": {0: {"inject": []}}}))
    assert r["status"] == "FAIL" and "do not report the site" in r["injections"][0]["why"][0]
    assert _hb([_entry(expected_applied_delta=None)])["status"] == "NOT_RUN"
    assert _hb([_entry(activation_unchanged=False)])["status"] == "NOT_RUN"
    ctx, tier = _hb_ctx()
    ctx.launch["reconcile"]["chains"] = []
    td = NS(tier=tier, summaries={"relay1": {0: {"inject": [_entry()]}}})
    r = V.tier_heartbeat(ctx, td)
    assert r["status"] == "NOT_RUN" and "no reconcile hop" in r["injections"][0]["why"][0]
    ctx.tt = NS(inject_for=lambda role, seq: [])
    assert V.tier_heartbeat(ctx, NS(tier=tier, summaries={}))["status"] == "NONE"


def test_heartbeat_without_data_is_not_run():
    # no summary of the worker in this tier (the tier did not run to the end): no data, and
    # "nothing injected" is not concluded from it
    r = _hb([], declared=True)
    why = r["injections"][0]["why"]
    assert r["status"] == "NOT_RUN" and "no summary of worker(s) [0]" in why[0]
    assert not any("no pass was injected" in w for w in why)
    # two workers, one summary: the other's passes are unknown
    ctx, tier = _hb_ctx(workers=2)
    r = V.tier_heartbeat(ctx, NS(tier=tier, summaries={"relay1": {0: {"inject": [_entry(
        hits_delta=0, applied_delta=0, expected_applied_delta=0, applied_ns_delta=0)]}}}))
    assert r["status"] == "NOT_RUN" and r["injections"][0]["hits_expected"] is None
    # a counter key missing is NOT_RUN, never read as 0
    for k in ("hits_delta", "applied_delta", "applied_ns_delta", "aux_pending_at_end"):
        e = _entry()
        del e[k]
        r = _hb([e])
        assert r["status"] == "NOT_RUN" and k in r["injections"][0]["why"][0], k


def test_launch_doses_constant_or_mixed():
    s = InjectSpec(role="relay1", **SPEC)
    tiers = [NS(seq=1, kind="drain"), NS(seq=2, kind="up"), NS(seq=4, kind="up")]
    ctx = NS(roles={"relay1": {}, "drain": {}},
             tt=NS(tiers=tiers, inject_for=lambda role, seq: [s] if role == "relay1" else []))
    assert V.launch_doses(ctx) == {"relay1": [SPEC]}
    ctx.tt = NS(tiers=tiers, inject_for=lambda role, seq: [s] if seq == 4 else [])
    assert V.launch_doses(ctx) == "mixed"
    ctx.tt = NS(tiers=tiers, inject_for=lambda role, seq: [])
    assert V.launch_doses(ctx) == {}


# ---- dose-response across launches -----------------------------------------------------------

AXIS = {"name": "R", "load_key": "rate_rps", "unit": "req/s", "order": "asc"}
MC = {"relay1": "under_test", "drain": "stub_out_of_domain", "beat": "stub_out_of_domain"}


def _launch(lo, hi, doses=None, role_dose="relay1", delta=20000, hb="PASS", grade=None,
            void=(), axis=AXIS, n=[0]):
    n[0] += 1
    d = {} if doses == "none" else (doses or {role_dose: [dict(SPEC, delta_ns=delta)]})
    return {"tree": "t%d" % n[0], "identity": {"experiment": "E0"}, "axis": axis,
            "boundary": {"lower": lo, "upper": hi}, "doses": d, "heartbeat": hb,
            "measure_class": MC, "void_tiers": list(void),
            "grade": grade or {"grade": "LOCALIZED", "role": "relay1",
                               "finding": "domain_saturated"}}


def _ctl(lo=1000, hi=1100):
    return _launch(lo, hi, doses="none", hb="NONE")


def test_dose_response_pass_confirms():
    ls = [_ctl(), _ctl(1000, 1100), _launch(800, 900, delta=20000),
          _launch(500, 600, delta=40000), _launch(1000, 1100, role_dose="drain", delta=20000)]
    v = V.dose_response(ls)
    assert v["status"] == "PASS" and v["role"] == "relay1"
    assert [s["moved_down"] for s in v["forward"]] == [True, True]
    assert G.final_grade([x["grade"] for x in ls[:2]], v)["grade"] == "CONFIRMED"


def test_dose_response_fail_when_the_reverse_dose_moves_the_boundary():
    ls = [_ctl(), _launch(800, 900), _launch(500, 600, delta=40000),
          _launch(700, 750, role_dose="drain")]
    v = V.dose_response(ls)
    assert v["status"] == "FAIL" and v["reverse"][0]["moved"]
    assert G.final_grade([ls[0]["grade"]], v)["grade"] == "UNDETERMINED"


def test_dose_response_fail_when_the_boundary_moves_up():
    v = V.dose_response([_ctl(), _launch(1200, 1300), _launch(500, 600, delta=40000),
                         _launch(1000, 1100, role_dose="drain")])
    assert v["status"] == "FAIL"


def test_dose_response_inconclusive_and_not_run():
    # the second step stays inside the first's envelope
    v = V.dose_response([_ctl(), _launch(800, 900), _launch(850, 950, delta=40000),
                         _launch(1000, 1100, role_dose="drain")])
    assert v["status"] == "INCONCLUSIVE"
    g = G.final_grade([_ctl()["grade"]], v)
    assert g["grade"] == "LOCALIZED" and "V INCONCLUSIVE" in g["reasons"]
    v = V.dose_response([_ctl(), _launch(800, 900), _launch(1000, 1100, role_dose="drain")])
    assert v["status"] == "NOT_RUN" and "need at least 2" in v["reasons"][0]
    v = V.dose_response([_ctl(), _launch(800, 900), _launch(500, 600, delta=40000)])
    assert v["status"] == "NOT_RUN" and "no usable reverse dose" in v["reasons"][0]


def test_reverse_dose_needs_a_healthy_closed_interval():
    fwd = [_launch(800, 900), _launch(500, 600, delta=40000)]
    # the reverse launch's interval was cut at a tool limit (the dosed stub saturated first):
    # not evidence that the boundary stayed, and not a FAIL either
    cut = _launch(None, None, role_dose="drain")
    cut["boundary"] = {"lower": 700, "upper": None, "censored_by": {
        "tier_seq": 6, "why": "I localised to drain (stub_out_of_domain)"}}
    v = V.dose_response([_ctl()] + fwd + [cut])
    assert v["status"] == "NOT_RUN" and v["reverse"] == []
    assert "interval cut at tier 6" in v["excluded"][0]["why"]
    # an open end on either side cannot show "did not move"
    for ctl, rev in ((_ctl(), _launch(1000, None, role_dose="drain")),
                     (_ctl(1000, None), _launch(1000, 1100, role_dose="drain"))):
        v = V.dose_response([ctl] + fwd + [rev])
        assert v["reverse"] == [] and "open end" in v["excluded"][-1]["why"], v
        assert v["status"] != "PASS"


def test_hop_counts_first_and_unusable_launches_are_listed():
    ls = [_ctl(), _launch(800, 900, hb="FAIL"), _launch(500, 600, delta=40000, hb="NOT_RUN"),
          _launch(1000, 1100, doses="mixed"),
          _launch(1000, 1100, role_dose="drain", void=(6,))]
    v = V.dose_response(ls)
    whys = " ".join(e["why"] for e in v["excluded"])
    assert "heartbeat FAIL" in whys and "heartbeat NOT_RUN" in whys and "dose differs" in whys
    assert "VOID tiers [6]" in whys and v["status"] == "NOT_RUN"


def test_dose_response_needs_localised_controls():
    v = V.dose_response([_launch(1000, 1100, doses="none", grade={"grade": "LOWER_BOUND"}),
                         _launch(800, 900)])
    assert v["status"] == "NOT_RUN" and "not all localised" in v["reasons"][0]
    with pytest.raises(V.InterveneError):
        V.dose_response([_ctl(), _launch(800, 900, axis=dict(AXIS, order="desc"))])


def test_descending_axis_moves_the_other_way():
    ax = dict(AXIS, order="desc", load_key="beat_ns")
    ls = [_launch(10, 9, doses="none", axis=ax), _launch(12, 11, axis=ax),
          _launch(15, 14, delta=40000, axis=ax), _launch(10, 9, role_dose="drain", axis=ax)]
    assert V.dose_response(ls)["status"] == "PASS"


# ---- per-launch grade ------------------------------------------------------------------------

def _t(seq, verdict, I=None, red=()):
    return {"tier_seq": seq, "judged": True, "verdict": verdict, "order_key": float(seq),
            "load_value": float(seq * 10), "checks": {c: "RED" for c in red},
            "X": {"I": I or {"status": "NOT_APPLICABLE"}}}


LOC = {"status": "LOCALIZED", "role": "sink", "measure_class": "under_test",
       "finding": "domain_saturated", "via": ["I1"], "pids": [5]}


def test_launch_grade_localized_and_interval():
    g = G.launch_grade([_t(2, "STABLE"), _t(4, "UNSTABLE", LOC), _t(6, "UNSTABLE", LOC)], {})
    assert (g["grade"], g["role"], g["tier_seq"]) == ("LOCALIZED", "sink", 4)
    assert g["interval"] == {"lower_seq": 2, "lower": 20.0, "upper_seq": 4, "upper": 40.0}
    assert g["V"].startswith("NOT_RUN")


def test_launch_grade_lower_bound_paths():
    tool = dict(LOC, role="ld", measure_class="load_source")
    assert G.launch_grade([_t(2, "STABLE"), _t(4, "UNSTABLE", tool)], {})["grade"] \
        == "LOWER_BOUND"
    g = G.launch_grade([_t(2, "STABLE"), _t(4, "VOID", red=("C1",)), _t(6, "UNSTABLE", LOC)],
                       {})
    assert g["grade"] == "LOWER_BOUND" and g["interval"]["lower_seq"] == 2
    g = G.launch_grade([_t(2, "STABLE"), _t(4, "STABLE")], {})
    assert g["grade"] == "LOWER_BOUND" and "X never held" in g["reasons"][0]


def test_launch_grade_skips_other_void_tiers_but_never_claims_x_unreached():
    g = G.launch_grade([_t(2, "STABLE"), _t(4, "VOID", red=("C5",)), _t(6, "UNSTABLE", LOC)],
                       {})
    assert g["grade"] == "LOCALIZED" and g["skipped_tiers"][0]["tier_seq"] == 4
    g = G.launch_grade([_t(2, "STABLE"), _t(4, "NOT_RUN")], {})
    assert g["grade"] == "NOT_RUN" and "not evaluated" in g["reasons"][0]


def test_launch_grade_undetermined_not_localized_not_run():
    und = {"status": "UNDETERMINED", "reasons": ["x"]}
    assert G.launch_grade([_t(2, "STABLE"), _t(4, "UNDETERMINED", und)], {})["grade"] \
        == "UNDETERMINED"
    g = G.launch_grade([_t(2, "STABLE"), _t(4, "UNSTABLE", {"status": "NOT_RUN"})],
                       {"non_monotone": True})
    assert g["grade"] == "NOT_LOCALIZED" and "non" not in g["grade"]
    assert any("T4" in r for r in g["reasons"])
    assert G.launch_grade([_t(2, "VOID", red=("C5",))], {})["grade"] == "NOT_RUN"


def test_final_grade_combines_launches():
    loc = {"grade": "LOCALIZED", "role": "sink", "finding": "domain_saturated"}
    v_pass = {"status": "PASS", "role": "sink"}
    assert G.final_grade([loc, loc], v_pass)["grade"] == "CONFIRMED"
    assert G.final_grade([loc, dict(loc, role="other")], v_pass)["grade"] == "UNDETERMINED"
    assert G.final_grade([loc, {"grade": "LOWER_BOUND"}], v_pass)["grade"] == "LOWER_BOUND"
    assert G.final_grade([loc], {"status": "PASS", "role": "x"})["grade"] == "LOCALIZED"
    assert G.final_grade([{"grade": "NOT_RUN"}], v_pass)["grade"] == "NOT_RUN"


# ---- inputs: mode, localize, i1_calibration -------------------------------------------------

def _reg_ctx(judge):
    ctx = NS(roles={"a": {}, "b": {}}, register={"judge": judge})
    return ctx


GOOD_LOC = {"readings": [{"name": "q", "receiver": "a", "sender": None,
                          "terms": [{"role": "a", "slot": "backlog.q", "sign": 1,
                                     "agg": "sum"}]}]}


def test_localize_spec_valid():
    I._load_localize(_reg_ctx({"mode": "P+X+I+V", "localize": GOOD_LOC}))
    I._load_localize(_reg_ctx({"mode": "P+X+I+V"}))


@pytest.mark.parametrize("mut", [
    lambda r: r.update(receiver=None),
    lambda r: r.update(sender="a"),
    lambda r: r.update(receiver="zz"),
    lambda r: r.update(name="Bad Name"),
    lambda r: r["terms"][0].update(slot="count.x"),
    lambda r: r["terms"][0].update(sign=True),
    lambda r: r["terms"][0].update(agg="avg"),
    lambda r: r["terms"][0].update(role="zz"),
    lambda r: r.update(terms=[]),
    lambda r: r.update(extra=1),
])
def test_localize_spec_rejected(mut):
    loc = json.loads(json.dumps(GOOD_LOC))
    mut(loc["readings"][0])
    with pytest.raises(I.InputInvalid):
        I._load_localize(_reg_ctx({"mode": "P+X+I+V", "localize": loc}))


def test_duplicate_reading_names_rejected():
    loc = {"readings": GOOD_LOC["readings"] * 2}
    with pytest.raises(I.InputInvalid):
        I._load_localize(_reg_ctx({"mode": "P+X+I+V", "localize": loc}))


ENV = {"hw_profile_sha": "d" * 64, "image_sha": "e" * 64, "capkit_sha": "c" * 40,
       "baseline_sha": "x"}


@pytest.mark.parametrize("i1,ok", [
    (None, True), ({"near_one_core_min": 0.9, "source_tiers": [], "n": 2}, True),
    ({"near_one_core_min": 1.2, "source_tiers": [], "n": 2}, False),
    ({"near_one_core_min": 0, "source_tiers": [], "n": 2}, False),
    ({"near_one_core_min": 0.9, "source_tiers": [], "n": 0}, False),
    ({"near_one_core_min": 0.9, "n": 2}, False),
])
def test_pilot_i1_calibration(i1, ok):
    doc = {"schema": K.PILOT_SCHEMA, "environment": ENV, "noise_bands": {}}
    if i1 is not None:
        doc["i1_calibration"] = i1
    if ok:
        K.validate("pilot", doc, "f")
    else:
        with pytest.raises(K.CalibError):
            K.validate("pilot", doc, "f")


# ---- judge mode and the CLI on judged samples ------------------------------------------------

def _judge_sample(tmp_path, name, mutate=None):
    W = str(tmp_path / name)
    shutil.copytree(os.path.join(SAMPLES, name), W)
    exp = json.load(open(os.path.join(W, "expect.json")))
    if mutate:
        mutate(W)
    tree = os.path.join(W, exp["tree"])
    p = subprocess.run([sys.executable, "-m", "capjudge", "--tree", tree, "--workdir", W],
                       env=JENV, capture_output=True, text=True, timeout=300, check=False)
    return W, tree, p


def test_mode_p_x_writes_not_requested(tmp_path):
    _W, tree, p = _judge_sample(tmp_path, "good")
    assert p.returncode == 0, p.stderr
    v = json.load(open(os.path.join(tree, "judge", "verdicts.json")))
    assert v["mode"] == "P+X" and v["grade"] == "NOT_REQUESTED"
    assert all(t["X"]["I"] == {"status": "NOT_REQUESTED"} for t in v["tiers"] if t["judged"])


def _register(monkeypatch, tmp_path, judge):
    files = {k: {"relpath": k, "sha256": "0" * 64} for k in I.REGISTER_FILE_KEYS}
    doc = {"schema": I.REGISTER_SCHEMA, "experiment": "E0", "arms": ["a"], "files": files,
           "judge": judge}
    p = tmp_path / "register.json"
    p.write_text(json.dumps(doc))
    monkeypatch.setattr(I, "_registered_file", lambda ctx, ref, item, kinds=None: str(p))
    ctx = NS(launch=dict(files, register={}), ident=NS(experiment="E0", arm="a"),
             roles={"a": {}})
    I._load_register(ctx)
    return ctx


@pytest.mark.parametrize("mode", ["P+X", "P+X+I+V"])
def test_judge_modes_accepted(monkeypatch, tmp_path, mode):
    assert _register(monkeypatch, tmp_path, {"mode": mode, "checks": []}).mode == mode


@pytest.mark.parametrize("judge", [{"mode": "P+X+I", "checks": []}, {"checks": []},
                                   {"mode": "P+X", "checks": [], "localize": GOOD_LOC}])
def test_bad_mode_or_localize_without_its_mode(monkeypatch, tmp_path, judge):
    with pytest.raises(I.InputInvalid) as e:
        _register(monkeypatch, tmp_path, judge)
    assert "mode" in str(e.value)


def _intervene(tmp_path, W, trees, out="v.json"):
    lst = tmp_path / ("list-%s.txt" % out)
    lst.write_text("".join(t + "\n" for t in trees))
    o = str(tmp_path / out)
    p = subprocess.run([sys.executable, "-m", "capjudge.intervene", "--inputs", str(lst),
                        "--workdir", W, "--out", o], env=JENV, capture_output=True, text=True,
                       timeout=120, check=False)
    return p, o


def test_intervene_cli_on_a_judged_control_launch(tmp_path):
    W, tree, p = _judge_sample(tmp_path, "i_server_slow")
    assert p.returncode == 0, p.stderr
    v = json.load(open(os.path.join(tree, "judge", "verdicts.json")))
    assert v["intervention_V"]["doses"] == {} and v["intervention_V"]["heartbeat"] == "NONE"
    p, o = _intervene(tmp_path, W, [tree])
    assert p.returncode == 0, p.stderr
    doc = json.load(open(o))
    assert doc["V"]["status"] == "NOT_RUN" and doc["grade"]["grade"] == "LOCALIZED"
    assert doc["grade"]["role"] == "nullsse"
    p2, _ = _intervene(tmp_path, W, [tree])  # never overwrites
    assert p2.returncode == 3 and "exists" in p2.stderr


def test_load_launch_uses_the_interval_cut_at_the_tool_limit(tmp_path):
    W, tree, p = _judge_sample(tmp_path, "i_reader_slow")
    assert p.returncode == 0, p.stderr
    x = V.load_launch(tree, W)
    v = json.load(open(os.path.join(tree, "judge", "verdicts.json")))
    assert x["boundary_t4"]["upper"] == v["boundary_T4"]["upper"] is not None
    # tier 4 localised to capload (the load source): the interval used stops below it
    assert x["boundary"]["upper"] is None and x["boundary"]["censored_by"]["tier_seq"] == 4
    p, o = _intervene(tmp_path, W, [tree])
    assert p.returncode == 0, p.stderr
    rec = open(o[:-len(".json")] + ".cmd.txt").read()
    assert "module\tcapjudge.intervene" in rec
    assert "input\t%s\t" % os.path.join(tree, "judge", "verdicts.json") in rec


def test_intervene_cli_rejects_p_x_trees_and_globs(tmp_path):
    W, tree, p = _judge_sample(tmp_path, "good")
    assert p.returncode == 0
    p, _o = _intervene(tmp_path, W, [tree], out="a.json")
    assert p.returncode == 3 and "not P+X+I+V" in p.stderr
    p, _o = _intervene(tmp_path, W, [tree + "*"], out="b.json")
    assert p.returncode == 3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-2 calibration files and checks (INTERFACES errata B2-2A): schema validation of
loadgen_band / stub_calib / near_boundary_noise, regime-selected noise bands, the register
binding and tier regimes, C1 against loadgen_band, C2 (out-of-domain stubs), C4, C12, and the
builders (capjudge.calib, analysis/calib_build.py)."""

import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
from conftest import C, PKG, SAMPLES, TOOLS

if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
from capjudge import calib as K  # noqa: E402
from capjudge import checks as CH  # noqa: E402
from capjudge import inputs as I  # noqa: E402
from capjudge import thresholds as T  # noqa: E402

ENV = {"hw_profile_sha": "d" * 64, "image_sha": "e" * 64, "capkit_sha": "c" * 40,
       "baseline_sha": "x"}
BAND = {"min": 0, "max": 10, "n": 3}
JENV = dict(os.environ, PYTHONPATH=TOOLS, CAPKIT_CODE_SHA="b" * 40)


def _shape(names, mkey, value=30.0, **match):
    m = {"tokens_per_msg": [1], "streams_c": [None]}
    m.update(match)
    return {"match": m, mkey: {"load_key": "rate_rps", "value": value, "censored": False,
                               "n_launches": 1},
            "bands": {n: dict(BAND) for n in names}}


def _loadgen(**kw):
    d = {"schema": K.LOADGEN_SCHEMA, "environment": dict(ENV), "inputs": [],
         "shapes": {"s": _shape(K.LOADGEN_STATS, "per_worker_max")}}
    d.update(kw)
    return d


def _stub(**kw):
    d = {"schema": K.STUB_SCHEMA, "environment": dict(ENV), "inputs": [], "in_domain": None,
         "out_of_domain": {"s": _shape(K.STUB_STATS, "per_process_max")}}
    d.update(kw)
    return d


def _nb(regime="near_boundary", **kw):
    d = {"schema": K.NEAR_BOUNDARY_SCHEMA, "regime": regime, "environment": dict(ENV),
         "inputs": [], "noise_bands": {"T1:x_ns:r": dict(BAND, regime=regime)}}
    d.update(kw)
    return d


# ---- schema validation -----------------------------------------------------------------------

def test_valid_files_pass_and_near_boundary_returns_its_regime():
    assert K.validate("loadgen_band", _loadgen(), "f") is None
    assert K.validate("stub_calib", _stub(), "f") is None
    assert K.validate("near_boundary_noise", _nb(), "f") == "near_boundary"
    assert K.validate("near_boundary_noise", _nb("subsat"), "f") == "subsat"


def test_pilot_format_file_is_accepted_as_near_boundary_noise():
    # RUNBOOK stage 2 D5: E1-0 bands built by capjudge.pilot, registered as this kind
    doc = {"schema": K.PILOT_SCHEMA, "environment": dict(ENV), "noise_bands": {}}
    assert K.validate("near_boundary_noise", doc, "f") is None


def _broken():
    def lg(f):
        d = _loadgen()
        f(d)
        return ("loadgen_band", d)

    def nb(f):
        d = _nb()
        f(d)
        return ("near_boundary_noise", d)
    s = lambda d: d["shapes"]["s"]  # noqa: E731
    return {
        "lg schema": lg(lambda d: d.update(schema=K.STUB_SCHEMA)),
        "lg env": lg(lambda d: d["environment"].pop("image_sha")),
        "lg no shapes": lg(lambda d: d.update(shapes={})),
        "lg noise_bands": lg(lambda d: d.update(noise_bands={})),
        "lg unknown stat": lg(lambda d: s(d)["bands"].update({"T3:x_ns": dict(BAND)})),
        "lg band min>max": lg(lambda d: s(d)["bands"]["T3:send_lag_p99_ns"].update(min=11)),
        "lg load_key": lg(lambda d: s(d)["per_worker_max"].update(load_key="streams_c")),
        "lg value 0": lg(lambda d: s(d)["per_worker_max"].update(value=0)),
        "lg censored": lg(lambda d: s(d)["per_worker_max"].update(censored=None)),
        "lg match empty": lg(lambda d: s(d)["match"].update(tokens_per_msg=[])),
        "lg match bool": lg(lambda d: s(d)["match"].update(streams_c=[True])),
        "lg extra key": lg(lambda d: s(d).update(note="x")),
        "stub in_domain": ("stub_calib", _stub(in_domain={})),
        "stub max key": ("stub_calib", _stub(out_of_domain={"s": _shape(
            K.STUB_STATS, "per_worker_max")})),
        "nb regime": nb(lambda d: d.update(regime="sat")),
        "nb band regime": nb(lambda d: d["noise_bands"]["T1:x_ns:r"].update(regime="subsat")),
        "nb band no regime": nb(lambda d: d["noise_bands"]["T1:x_ns:r"].pop("regime")),
        "nb schema": nb(lambda d: d.update(schema=K.LOADGEN_SCHEMA)),
        "nb bad key": nb(lambda d: d["noise_bands"].update({"T1:x": dict(BAND)})),
        "pilot schema": ("pilot", _nb()),
    }


@pytest.mark.parametrize("case", sorted(_broken()))
def test_malformed_files_rejected(case):
    kind, doc = _broken()[case]
    with pytest.raises(K.CalibError):
        K.validate(kind, doc, "f")


# ---- regime-selected bands -------------------------------------------------------------------

def test_same_key_from_two_sources_each_owning_a_regime():
    b = T.Bands()
    b.add_source("pilot", "p", "a" * 64, {"noise_bands": {"T1:x_ns:r": dict(BAND)}})
    b.add_source("near_boundary_noise", "n", "b" * 64, _nb())
    assert b.get("T1:x_ns:r", "subsat") == BAND
    assert b.get("T1:x_ns:r", "near_boundary")["regime"] == "near_boundary"
    assert b.source_of("T1:x_ns:r", "subsat") == "p"
    assert b.source_of("T1:x_ns:r", "near_boundary") == "n"


def test_same_key_and_regime_twice_is_an_error():
    b = T.Bands()
    b.add_source("pilot", "p", "a" * 64, _nb())
    with pytest.raises(T.BandError):
        b.add_source("near_boundary_noise", "n", "b" * 64, _nb())


def test_near_boundary_tier_never_falls_back_to_the_subsat_band():
    b = T.Bands()
    b.add_source("pilot", "p", "a" * 64, {"noise_bands": {"T1:x_ns:r": dict(BAND)}})
    assert b.view("near_boundary").get("T1:x_ns:r") is None
    assert b.view("subsat").get("T1:x_ns:r") == BAND
    assert b.lookups == {"near_boundary|T1:x_ns:r": "missing", "subsat|T1:x_ns:r": "found"}


def test_unknown_regime_rejected():
    with pytest.raises(T.BandError):
        T.Bands().add_source("pilot", "p", "a" * 64,
                             {"noise_bands": {"T1:x_ns:r": dict(BAND, regime="sat")}})
    with pytest.raises(ValueError):
        T.Bands().view("sat")


# ---- register: binding, calibration entries, tier regimes ------------------------------------

def _ctx(**kw):
    c = NS(launch={"hw_profile_sha": ENV["hw_profile_sha"], "image": {"sha256": ENV["image_sha"]},
                   "register": {"relpath": "r.json"}},
           ident=NS(experiment="E2a", arm="a", register_sha="f" * 64), bands=T.Bands(),
           calib_docs={}, calib_bind={}, tier_regime={})
    c.__dict__.update(kw)
    return c


def _load_cal(monkeypatch, tmp_path, entries, docs):
    paths = {}
    for i, d in enumerate(docs):
        p = tmp_path / ("c%d.json" % i)
        p.write_text(json.dumps(d))
        paths[entries[i]["relpath"]] = str(p)
    monkeypatch.setattr(I, "_registered_file", lambda ctx, c, item, kinds=None:
                        paths[c["relpath"]])
    ctx = _ctx(register={"calibration": entries})
    I._load_calibration(ctx)
    return ctx


def _entry(kind, rel, **kw):
    return dict({"kind": kind, "relpath": rel, "sha256": "a" * 64}, **kw)


def test_bind_loads_and_shapes_resolve(monkeypatch, tmp_path):
    ctx = _load_cal(monkeypatch, tmp_path, [_entry("loadgen_band", "l", bind={"ld": "s"})],
                    [_loadgen()])
    assert ctx.calib_bind == {"loadgen_band": {"ld": "s"}}
    assert CH._shape_for(ctx, "loadgen_band", "ld")[0]["per_worker_max"]["value"] == 30.0
    assert CH._shape_for(ctx, "loadgen_band", "other")[0] is None
    assert CH._shape_for(ctx, "stub_calib", "ld")[0] is None


@pytest.mark.parametrize("entry,doc", [
    (_entry("loadgen_band", "l", bind={"ld": "nope"}), _loadgen()),  # unknown shape
    (_entry("loadgen_band", "l"), _loadgen()),  # bind required
    (_entry("loadgen_band", "l", bind={}), _loadgen()),
    (_entry("pilot", "p", bind={"ld": "s"}), {"schema": K.PILOT_SCHEMA,
                                              "environment": dict(ENV)}),  # bind not allowed
    (_entry("loadgen_band", "l", bind={"ld": "s"}, note="x"), _loadgen()),  # unknown key
    (_entry("loadgen_band", "l", bind={"ld": "s"}),
     _loadgen(environment=dict(ENV, image_sha="0" * 64))),  # other environment
    (_entry("near_boundary_noise", "n"), _nb(environment=dict(ENV, hw_profile_sha="1" * 64))),
    (_entry("stub_calib", "s", bind={"sink": "s"}), _stub(in_domain={"x": 1})),
])
def test_bad_calibration_entries_are_input_invalid(monkeypatch, tmp_path, entry, doc):
    with pytest.raises(I.InputInvalid):
        _load_cal(monkeypatch, tmp_path, [entry], [doc])


def _tt(*tiers):
    return NS(tiers=[NS(seq=i, label=lab, kind=k) for i, (lab, k) in enumerate(tiers)])


TT = _tt(("warmup", "warmup"), ("d1", "drain"), ("lo", "up"), ("d2", "drain"), ("hi", "up"),
         ("d3", "drain"))


def test_tier_regimes_from_the_register():
    ctx = _ctx(tt=TT, register={"judge": {"near_boundary_tiers": {"a": ["hi"], "b": ["lo"]}}})
    I._load_tier_regimes(ctx)
    assert ctx.tier_regime == {2: "subsat", 4: "near_boundary"}
    ctx = _ctx(tt=TT, register={"judge": {}})
    I._load_tier_regimes(ctx)
    assert ctx.tier_regime == {2: "subsat", 4: "subsat"}


@pytest.mark.parametrize("labels", [["d2"], ["warmup"], ["nope"]])
def test_near_boundary_label_must_be_a_load_tier(labels):
    ctx = _ctx(tt=TT, register={"judge": {"near_boundary_tiers": {"a": labels}}})
    with pytest.raises(I.InputInvalid):
        I._load_tier_regimes(ctx)


# ---- C1 / C2 against calibrated shapes -------------------------------------------------------

def _tier(seq=4, rate=40.0, k=1, c=None, beat=20_000_000, max_tokens=3):
    return NS(seq=seq, rate_rps=rate, tokens_per_msg=k, streams_c=c, beat_ns=beat,
              stream_interval=1, max_tokens=max_tokens, window_start_ns=0,
              window_end_ns=4 * 10 ** 9)


def _items(shape, st, tier=None, kind="loadgen_band", split="even", workers=2):
    r = CH.Result("Cx", 4, "t")
    ctx = NS(roles={"ld": {"workers": workers}})
    ok = CH._shape_items(r, ctx, kind, shape, tier or _tier(), "ld", 0, st, {}, split)
    return ok, r.finish() if ok else r


GOOD_ST = {n: 1 for n in K.LOADGEN_STATS}


def test_shape_items_pass_and_each_way_to_fail():
    sh = _shape(K.LOADGEN_STATS, "per_worker_max", value=30.0)
    ok, r = _items(sh, GOOD_ST)  # 40 req/s over 2 workers = 20 <= 30
    assert ok and r.red == 0 and not r.not_run
    ok, r = _items(_shape(K.LOADGEN_STATS, "per_worker_max", value=15.0), GOOD_ST)
    assert ok and r.red == 1 and "above the calibrated healthy maximum" in r.detail[0]
    ok, r = _items(sh, dict(GOOD_ST, **{"T3:send_lag_p99_ns": 11}))
    assert ok and r.red == 1
    ok, r = _items(sh, dict(GOOD_ST, **{"T3:send_lag_p99_ns": None}))
    assert not ok and r.red == 0 and r.not_run
    ok, r = _items(sh, GOOD_ST, tier=_tier(k=2))  # outside the calibrated match
    assert not ok and "outside the calibrated match" in r.not_run[0]
    ok, r = _items(_shape(K.LOADGEN_STATS, "per_worker_max", value=None), GOOD_ST)
    assert not ok and "no healthy maximum" in r.not_run[0]
    sh2 = copy.deepcopy(sh)
    del sh2["bands"]["T3:loop_lag_p99_ns"]
    ok, r = _items(sh2, GOOD_ST)
    assert not ok and "no band in the calibrated shape" in r.not_run[0]


def test_c1_uses_each_workers_schedule_share(monkeypatch):
    # 3 + 3 requests round-robin over 2 workers: in tier 4 worker 1 has 2 of 3 rows
    meta = {"per_tier": {"2": 3, "4": 3}}
    ctx = NS(roles={"ld": {"role": "ld", "workers": 2, "measure_class": "load_source"}},
             schedule_meta=meta, calib_bind={"loadgen_band": {"ld": "s"}},
             calib_docs={"loadgen_band": _loadgen(shapes={"s": _shape(
                 K.LOADGEN_STATS, "per_worker_max", value=26.0)})})
    assert K.worker_share(ctx, _tier(), "ld", 1, "schedule") == 2 / 3
    assert K.worker_share(ctx, _tier(), "ld", 1, "even") == 1 / 2
    monkeypatch.setattr(K, "loadgen_worker_stats", lambda td, role, k: dict(GOOD_ST))
    r = CH.check_c1(ctx, NS(tier=_tier()), {})
    assert (r.status, r.red) == ("RED", 1) and "ld.w1 planned 26.6667" in r.detail[0]
    ctx.calib_bind = {}
    r = CH.check_c1(ctx, NS(tier=_tier()), {})
    assert r.status == "NOT_RUN" and "no registered loadgen_band" in r.not_run[0]


def test_nominal_msgs_load_needs_streams_and_a_beat():
    assert K.tier_load(_tier(c=8, beat=4_000_000), "nominal_msgs_per_s") == 2000.0
    assert K.tier_load(_tier(c=None), "nominal_msgs_per_s") is None
    assert K.tier_load(_tier(), "rate_rps") == 40.0


def test_thread_cpu_cores_from_sidecar_rows():
    rows = [{"t_rel_ns": t, "pid": 7, "tid": tid, "utime_ticks": u, "stime_ticks": 0}
            for t, tid, u in ((0, 7, 0), (10 ** 9, 7, 40), (0, 8, 0), (10 ** 9, 8, 10),
                              (0, 9, 0), (5 * 10 ** 9, 9, 99))]  # tid 9's 2nd row is late
    v, why = K.pid_thread_cpu_cores(rows, 7, 0, 4 * 10 ** 9, 100)
    assert why is None and abs(v - 0.5) < 1e-9
    assert K.pid_thread_cpu_cores(rows, 6, 0, 4 * 10 ** 9, 100)[0] is None


def _c2_ctx_td(lag_ns, ticks, bind=True):
    from capcommon import hist
    h = hist.new_counts()
    h[hist.bucket_index(lag_ns)] = 10
    doc = {"identity": {"pid": 7}, "loop_lag_ns": {"hist": {"counts": h}}}
    rows = [{"t_rel_ns": 0, "pid": 7, "tid": 7, "utime_ticks": 0, "stime_ticks": 0},
            {"t_rel_ns": 10 ** 9, "pid": 7, "tid": 7, "utime_ticks": ticks, "stime_ticks": 0}]
    ld = NS(sidecar=lambda name: ({}, rows), sidecar_meta={"clk_tck": 100})
    sh = _shape(K.STUB_STATS, "per_process_max", value=100.0)
    sh["bands"] = {"T3:loop_lag_p99_ns": {"min": 0, "max": 2_000_000, "n": 3},
                   "T3:thread_cpu_cores": {"min": 0, "max": 0.5, "n": 3}}
    ctx = NS(roles={"sink": {"role": "sink", "workers": 1, "measure_class": "stub_out_of_domain"},
                    "x": {"role": "x", "workers": 1, "measure_class": "stub_in_domain"}},
             calib_bind={"stub_calib": {"sink": "s"}} if bind else {},
             calib_docs={"stub_calib": _stub(out_of_domain={"s": sh})})
    td = NS(tier=_tier(), summaries={"sink": {0: doc}})
    return ctx, ld, td


def test_c2_pass_red_and_not_run():
    rec = {}
    r = CH.check_c2(*_c2_ctx_td(50_000, 20), rec)
    assert (r.status, r.expected, r.checked) == ("PASS", 1, 1)
    assert "stub_in_domain ['x']" in r.detail[0]
    assert abs(rec["T3:thread_cpu_cores:sink"][0] - 0.2) < 1e-9
    assert CH.check_c2(*_c2_ctx_td(50_000_000, 20), {}).status == "RED"
    assert CH.check_c2(*_c2_ctx_td(50_000, 90), {}).status == "RED"
    r = CH.check_c2(*_c2_ctx_td(50_000, 20, bind=False), {})
    assert r.status == "NOT_RUN" and "no registered stub_calib" in r.not_run[0]
    ctx, ld, td = _c2_ctx_td(50_000, 20)
    td.summaries = {}
    assert CH.check_c2(ctx, ld, td, {}).status == "RED"  # a stub worker without its summary
    ctx.roles = {}
    assert CH.check_c2(ctx, ld, td, {}).status == "NOT_RUN"  # no out-of-domain stub


# ---- C4 output length ------------------------------------------------------------------------

def _c4(client_rows, server_rows=(), planned=3):
    def cb(rows, cols):
        return {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    caps = {"loadgen.requests": [({"role": "ld"}, {}, cb(client_rows, (
        "req_id", "status", "tokens", "tier_seq")))]}
    if server_rows:
        caps["server.streams"] = [({"role": "sink"}, {}, cb(server_rows, (
            "req_id", "tokens", "tier_seq")))]
    ctx = NS(roles={"ld": {"role": "ld", "kind": "client"}, "sink": {"role": "sink",
                                                                     "kind": "server"}},
             schedule_meta={"per_tier": {"4": planned}})
    return CH.check_c4(ctx, NS(tier=_tier(), capbins=caps))


def test_c4_pass_when_every_response_has_max_tokens():
    r = _c4([(1, 0, 3, 4), (2, 0, 3, 4), (3, 1, -1, 4), (9, 0, 2, 2)])  # 9: other tier
    assert (r.status, r.expected, r.checked, r.red) == ("PASS", 3, 3, 0)


def test_c4_short_output_is_red_whatever_the_status():
    assert _c4([(1, 0, 3, 4), (2, 0, 2, 4), (3, 0, 3, 4)]).status == "RED"
    assert _c4([(1, 0, 3, 4), (2, 6, 2, 4), (3, 0, 3, 4)]).status == "RED"


def test_c4_uses_the_server_count_when_the_loader_does_not_count():
    rows = [(1, 0, -1, 4), (2, 0, -1, 4), (3, 0, -1, 4)]
    assert _c4(rows, [(1, 3, 4), (2, 3, 4), (3, 3, 4)]).status == "PASS"
    assert _c4(rows, [(1, 3, 4), (2, 1, 4), (3, 3, 4)]).status == "RED"
    r = _c4(rows, [(1, 3, 4), (2, 3, 4), (3, 3, 2)])  # req 3's stream row is another tier's
    assert r.status == "NOT_RUN" and "without a token count" in r.not_run[0]


def test_c4_missing_records_not_run():
    r = _c4([(1, 0, 3, 4), (2, 0, 3, 4)])
    assert r.status == "NOT_RUN" and "2 of 3 planned" in r.not_run[0]


# ---- C12 doubled processes carry traffic -----------------------------------------------------

def _c12(workers=2, vals=(5, 7), slot=False, procs=(), slot_rows=()):
    hop = {"role": "sink", "source": "slot:count.req" if slot else "summary.counts.requests",
           "scope": "tier"}
    ctx = NS(roles={"ld": {"workers": 1, "expected_procs": 1},
                    "sink": {"workers": workers, "expected_procs": len(procs) or workers}},
             launch={"reconcile": {"chains": [
                 {"kind": "requests", "hops": [{"role": "ld", "source": "summary.counts.ok",
                                                "scope": "tier"}, hop]},
                 {"kind": "chunks", "hops": [dict(hop, source="summary.counts.frames")]}]}})
    td = NS(tier=NS(seq=4, start_ns=0),
            summaries={"ld": {0: {"counts": {"ok": 3}}},
                       "sink": {k: {"counts": {"requests": v}} for k, v in enumerate(vals)}})
    ld = NS(procs_of_role=lambda role: [{"pid": p} for p in procs],
            sidecar=lambda name: ({}, list(slot_rows)))
    nxt = NS(drain_gate={"passed": True, "t_pass_rel_ns": 5}, tier=NS(end_ns=10))
    return CH.check_c12(ctx, ld, td, nxt)


def test_c12_every_doubled_worker_received_traffic():
    r = _c12()
    assert (r.status, r.expected, r.checked) == ("PASS", 2, 2)


def test_c12_idle_worker_is_not_run_never_pass():
    r = _c12(vals=(9, 0))
    assert r.status == "NOT_RUN" and "['w1'] carried no traffic" in r.not_run[0]


def test_c12_without_doubled_roles_has_nothing_to_check():
    assert _c12(workers=1, vals=(9,)).status == "NOT_RUN"


def _snap(pid, t, v):
    return {"pid": pid, "snap_t_rel_ns": t, "counters": {"count.req": v}}


def test_c12_per_process_slot_deltas():
    rows = [_snap(1, 0, 0), _snap(1, 8, 4), _snap(2, 0, 3), _snap(2, 8, 9)]
    assert _c12(slot=True, procs=(1, 2), slot_rows=rows).status == "PASS"
    rows = [_snap(1, 0, 0), _snap(1, 8, 4), _snap(2, 0, 3), _snap(2, 8, 3)]
    r = _c12(slot=True, procs=(1, 2), slot_rows=rows)
    assert r.status == "NOT_RUN" and "pid 2" in r.not_run[0]


# ---- builders (judged synthetic samples) -----------------------------------------------------

def _judged(tmp_path, name):
    out = tmp_path / ("s-" + name)
    subprocess.run([sys.executable, os.path.join(SAMPLES, "make_samples.py"), "--out",
                    str(out), "--only", name], check=True, capture_output=True, timeout=300)
    W = out / name
    tree = W / json.loads((W / "expect.json").read_text())["tree"]
    subprocess.run([sys.executable, "-m", PKG, "--tree", str(tree), "--workdir", str(W)],
                   env=JENV, capture_output=True, timeout=300, check=False)
    return W, tree


def _build(tmp_path, W, what, lines, *extra, out="out.json"):
    lst = tmp_path / ("%s.tsv" % what)
    lst.write_text("".join(lines))
    o = tmp_path / out
    p = subprocess.run([sys.executable, os.path.join(C, "analysis", "calib_build.py"), what,
                        "--inputs", str(lst), "--workdir", str(W), "--out", str(o)] + list(extra),
                       capture_output=True, text=True, timeout=300, check=False)
    return p, o


def test_near_boundary_builder_tags_every_band_with_the_regime(tmp_path):
    W, tree = _judged(tmp_path, "good")
    p, o = _build(tmp_path, W, "near-boundary", ["%s\t2,4\n" % tree], "--regime",
                  "near_boundary")
    assert p.returncode == 0, p.stderr
    d = json.loads(o.read_text())
    assert K.validate("near_boundary_noise", d, "built") == "near_boundary"
    assert d["noise_bands"]["T1:req_first_lag_p50_ns:capload"]["n"] == 2
    assert d["inputs"][0]["tiers_used"] == [2, 4]
    p2, _ = _build(tmp_path, W, "near-boundary", ["%s\t2\n" % tree], "--regime", "subsat")
    assert p2.returncode == 3 and "refusing to overwrite" in p2.stderr


def test_near_boundary_builder_leaves_out_unstable_tiers(tmp_path):
    W, tree = _judged(tmp_path, "sat_overloaded_tier")
    p, o = _build(tmp_path, W, "near-boundary", ["%s\t2,4\n" % tree], "--regime", "subsat")
    assert p.returncode == 0, p.stderr
    d = json.loads(o.read_text())
    assert d["inputs"][0]["tiers_used"] == [2] and "4" in d["inputs"][0]["tiers_excluded"]


def test_loadgen_band_builder(tmp_path):
    W, tree = _judged(tmp_path, "good")
    p, o = _build(tmp_path, W, "loadgen-band", ["%s\t2,4\tcapload\tld-r\n" % tree])
    assert p.returncode == 0, p.stderr
    d = json.loads(o.read_text())
    K.validate("loadgen_band", d, "built")
    sh = d["shapes"]["ld-r"]
    # both tiers STABLE: T4 lower = tier 4 (40 req/s), busiest of two workers = 20; never
    # UNSTABLE -> censored
    assert sh["per_worker_max"] == {"load_key": "rate_rps", "value": 20.0, "censored": True,
                                    "n_launches": 1}
    assert sh["match"] == {"tokens_per_msg": [1], "streams_c": [None]}
    assert sorted(sh["bands"]) == sorted(K.LOADGEN_STATS)
    assert all(b["n"] == 4 for b in sh["bands"].values())  # 2 tiers x 2 workers


def test_loadgen_band_builder_uses_stable_tiers_only(tmp_path):
    W, tree = _judged(tmp_path, "sat_overloaded_tier")
    p, o = _build(tmp_path, W, "loadgen-band", ["%s\t2,4\tcapload\tld-r\n" % tree])
    assert p.returncode == 0, p.stderr
    d = json.loads(o.read_text())
    sh = d["shapes"]["ld-r"]
    assert sh["per_worker_max"]["value"] == 10.0 and sh["per_worker_max"]["censored"] is False
    assert all(b["n"] == 2 for b in sh["bands"].values())
    assert "4" in d["inputs"][0]["tiers_excluded"]


def test_stub_calib_builder(tmp_path):
    W, tree = _judged(tmp_path, "c2_stub_good")
    p, o = _build(tmp_path, W, "stub-calib", ["%s\t2,4\tnullsse\tsink-r\n" % tree])
    assert p.returncode == 0, p.stderr
    d = json.loads(o.read_text())
    K.validate("stub_calib", d, "built")
    sh = d["out_of_domain"]["sink-r"]
    assert sh["per_process_max"]["value"] == 40.0  # one worker carries all of tier 4
    cpu = sh["bands"]["T3:thread_cpu_cores"]
    assert cpu["n"] == 2 and 0.15 < cpu["min"] <= cpu["max"] < 0.3


@pytest.mark.parametrize("line", ["{t}\t2\tnope\ts\n", "{t}\t2\tcapload\n",
                                  "{t}*\t2\tcapload\ts\n"])
def test_shape_builders_reject_bad_lists(tmp_path, line):
    W, tree = _judged(tmp_path, "good")
    p, o = _build(tmp_path, W, "loadgen-band", [line.format(t=tree)])
    assert p.returncode == 3 and not o.exists()

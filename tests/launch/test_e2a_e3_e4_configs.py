# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""configs/e2a, configs/e3, configs/e4 (fast-track scenario templates, INTERFACES errata FT1-scen):
the generator reproduces the shipped files; every arm validates, expands and cross-checks; the
arm families keep the contracts the experiment cards rely on (no processes)."""

import json
import os
import subprocess
import sys

import pytest

import conftest as cf
from capcommon import jobinputs as JI
from capcommon.templates import placeholders
from capcommon.tiers import TierTable
from caplaunch import scenario as sc
from caplaunch import tiergen

GEN = os.path.join(cf.C, "configs", "gen_e2a_e3_e4_configs.py")
SUBS = ("e2a", "e3", "e4")
DIRS = {s: os.path.join(cf.C, "configs", s) for s in SUBS}
EXP = {"e2a": "E2a", "e3": "E3", "e4": "E4"}
K_CONN, W_HTTP = 4, 4
SNAP_MS = 50


def _load(d, n):
    with open(os.path.join(d, n)) as f:
        return json.load(f)


def _arms(sub):
    return sorted(n[:-len(".scenario.json")] for n in os.listdir(DIRS[sub])
                  if n.endswith(".scenario.json"))


ALL = [(s, a) for s in SUBS for a in _arms(s)]


def _scen(sub, arm):
    return _load(DIRS[sub], arm + ".scenario.json")


def _tpl(sub, arm):
    return _load(DIRS[sub], arm + ".tiers.template.json")


def _roles(doc):
    return {r["role"]: r for r in doc["roles"]}


def _strs(items):
    return [a for a in items if isinstance(a, str)]


def _opt(args, flag):
    a = _strs(args)
    return a[a.index(flag) + 1] if flag in a else None


def _values(tpl):
    """Numbers that satisfy every template shape: beats are powers of two ms and every duration
    a multiple of 128 ms, so the fixed-C path (beat halving) keeps durations whole."""
    v = {}
    for k in tiergen.tbd_names(tpl):
        if k.startswith("beat_ns_") and k != "beat_ns_warmup":
            v[k] = (128 >> int(k.split("_")[-1])) * cf.MS
        elif k in ("beat_ns", "beat_ns_warmup"):
            v[k] = 128 * cf.MS if k == "beat_ns_warmup" else 8 * cf.MS
        elif k.startswith("c_") and k[2:].isdigit():
            v[k] = 2 << int(k[2:])
        elif k.startswith("rate_") and k[5:].isdigit():
            v[k] = 4.0 * (2 ** int(k[5:]))
        else:
            v[k] = {"arrival_seed": 1, "c_warmup": 1, "c_fixed": 4, "rate_warmup": 2.0,
                    "n_tokens": 8, "delta_ns": 50_000, "gate_poll_ns": 100 * cf.MS,
                    "gate_settle_polls": 2, "gate_deadline_guard_ns": 100 * cf.MS,
                    "gate_max_snapshot_age_ns": 200 * cf.MS, "ramp_ns": 0, "tail_ns": 0,
                    "window_ns": 1280 * cf.MS, "warmup_ns": 1280 * cf.MS,
                    "drain_max_wait_ns": 1280 * cf.MS,
                    "request_timeout_ns": 5000 * cf.MS}[k]
    return v


def _pin_roles(roles):
    out, nxt = {}, 0
    for r in roles.values():
        out[r["pin_role"]] = {"cpus": list(range(nxt, nxt + r["cpus_needed"])), "numa_node": 0}
        nxt += r["cpus_needed"]
    return out


def test_generator_reproduces_the_shipped_files(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-B", GEN, str(out)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    assert sorted(os.listdir(str(out))) == list(SUBS)
    for s in SUBS:
        made = sorted(os.listdir(str(out / s)))
        assert made == sorted(n for n in os.listdir(DIRS[s]) if n != "README.md"), s
        for n in made:
            with open(str(out / s / n), "rb") as a, open(os.path.join(DIRS[s], n), "rb") as b:
                assert a.read() == b.read(), (s, n)


def test_every_arm_has_scenario_template_and_values():
    for s in SUBS:
        names = os.listdir(DIRS[s])
        arms = _arms(s)
        assert arms, s
        for a in arms:
            for suffix in (".tiers.template.json", ".values.todo.json"):
                assert a + suffix in names, (s, a)
        assert len(names) == 3 * len(arms) + 1, s  # + README.md


@pytest.mark.parametrize("sub,arm", ALL)
def test_arm_validates_expands_and_cross_checks(sub, arm):
    doc = _scen(sub, arm)
    assert doc["experiment"] == EXP[sub] and doc["name"] == arm
    roles = sc.validate_scenario(doc)
    if JI.declares(doc):
        JI.check_scenario(doc)
    tpl = _tpl(sub, arm)
    todo = _load(DIRS[sub], arm + ".values.todo.json")
    assert set(todo["values"]) == tiergen.tbd_names(tpl)
    assert all(v is None for v in todo["values"].values())
    assert tpl["experiment"] == EXP[sub] and tpl["name"] == arm
    tt = TierTable.from_doc(tiergen.expand(tpl, _values(tpl)))
    cpus = sc.cross_check(doc, roles, tt, _pin_roles(roles), SNAP_MS * cf.MS)
    assert sorted(cpus) == sorted(roles)
    # one message shape for all load tiers (S-PB / S-NL take k, s from the table)
    assert {(t.tokens_per_msg, t.stream_interval) for t in tt.load_tiers()} == {(1, 1)}
    assert len([t for t in tt.tiers if t.kind == "up"]) >= 1


@pytest.mark.parametrize("sub,arm", ALL)
def test_params_required_is_exactly_what_the_arm_uses(sub, arm):
    """The plan must pass exactly params_required (the orchestrator refuses any other set), so
    an unused name would be a param nobody reads and a missing one a refused launch."""
    doc = _scen(sub, arm)
    used = set()

    def walk(o):
        if isinstance(o, str):
            used.update(a for n, a in placeholders(o) if n == "param")
        elif isinstance(o, dict):
            [walk(v) for v in o.values()]
        elif isinstance(o, list):
            [walk(v) for v in o]
    walk(doc["roles"])
    walk(doc.get("job_inputs", {}))
    assert sorted(used) == doc["params_required"]


def _shape_procs(arm):
    tag = [t for t in arm.split("-") if t in ("1c", "kc", "k2c")]
    return {"1c": 1, "kc": K_CONN, "k2c": 2 * K_CONN}[tag[0]] if tag else None


def test_reader_connection_shapes():
    """1c = one process, one channel, in-process (the shape E3-3b / E4 readers see); kc = K worker
    processes (parent + K) with one channel each; k2c = 2K (loader doubling, C12). Readers named
    without a shape tag (e3b-*, e4*-weak-*) are single-connection."""
    seen = set()
    for s, a in ALL:
        r = _roles(_scen(s, a)).get("reader")
        if r is None:
            continue
        procs = _shape_procs(a) or 1
        seen.add(procs)
        args = _strs(r["args"])
        assert _opt(args, "--channels-per-worker") == "1", a
        assert r["workers"] == procs == r["cpus_needed"], a
        if procs == 1:
            assert "--in-process" in args and r["expected_procs"] == 1, a
        else:
            assert "--in-process" not in args and r["expected_procs"] == procs + 1, a
            assert r["module"] == "capgrpc.load", a
        assert _opt(args, "--request-format") == "openengine", a
        assert _opt(args, "--request-bytes") == "{input:request_bytes}", a
        assert _opt(args, "--method") == "/openengine.v1.Inference/Generate", a
        assert "--planned-msgs" in args and "--max-msg-records-per-worker" in args, a
        assert "request_bytes" in JI.kinds_of(_scen(s, a)), a
    assert seen == {1, K_CONN, 2 * K_CONN}


def test_stubs_are_one_process_on_one_core_and_schedule_only_when_read_directly():
    """A stub read by a capgrpc reader takes --schedule (request_id = req_id, streams filed by
    planned tier); a stub behind the frontend does not (the frontend's request ids are not
    schedule ids). E4 weak-reader arms keep the frontend arms' stub argv."""
    n = 0
    for s, a in ALL:
        doc = _scen(s, a)
        rs = _roles(doc)
        for name in ("spb", "snl"):
            if name not in rs:
                continue
            n += 1
            r = rs[name]
            assert r["cpus_needed"] == r["workers"] == r["expected_procs"] == 1, a
            behind_fe = s == "e4"
            assert ("--schedule" in _strs(r["args"])) == (not behind_fe), a
            if name == "snl":
                mode = _opt(r["args"], "--mode")
                assert mode == ("metronome-only" if "-mono-" in a else "full"), a
    assert n == len(ALL)


def test_weak_reader_swaps_only_the_module_and_the_oe_sink_flags():
    """Design 2.3 'weak-reader check' (errata FT1-10): the oe_sink arm and the weak arm of each
    E3-3b shape differ in the reader's module, --tier / --connect and measure_class only; the
    stub and the template are the same, so one values file registers both."""
    for shape in ("mc", "md", "r"):
        weak = _scen("e3", "e3b-weak-" + shape)
        for kind, tier in (("gs", "generate-stream"), ("ees", "engine-event-stream")):
            sink = _scen("e3", "e3b-%s-%s" % (kind, shape))
            rw, rs = _roles(weak), _roles(sink)
            assert rw["spb"]["args"] == rs["spb"]["args"]
            assert rs["reader"]["module"] == "capgrpc.oe_sink"
            assert rw["reader"]["module"] == "capgrpc.load"
            assert rs["reader"]["args"] == rw["reader"]["args"] + ["--tier", tier, "--connect",
                                                                   "channel"]
            assert rs["reader"]["measure_class"] == "under_test"
            assert rw["reader"]["measure_class"] == "load_source"
            assert sink["job_inputs"] == weak["job_inputs"]
            assert sink["reconcile"] == weak["reconcile"]
            tw, ts = _tpl("e3", "e3b-weak-" + shape), _tpl("e3", "e3b-%s-%s" % (kind, shape))
            assert {k: v for k, v in tw.items() if k != "name"} == \
                {k: v for k, v in ts.items() if k != "name"}
    for stub, tag in (("spb", "e4b"), ("snl", "e4bp")):
        for shape in ("mc", "md", "r"):
            fe, weak = _scen("e4", "%s-%s" % (tag, shape)), _scen("e4", "%s-weak-%s" % (tag, shape))
            assert _roles(fe)[stub] == _roles(weak)[stub]
            assert _roles(weak)["reader"]["module"] == "capgrpc.load"
            t1, t2 = _tpl("e4", "%s-%s" % (tag, shape)), _tpl("e4", "%s-weak-%s" % (tag, shape))
            assert t1["tiers"] == t2["tiers"] and t1["common"] == t2["common"]


def _sites(role):
    """Injection sites (and their modes) of the tool a role runs."""
    import importlib.util
    from capgrpc import SITES as GRPC_SITES
    from capstubs import null_llm_server, prebaked_server
    # tests/launch/fakepkgs shadows the capload package: read the real one by path
    spec = importlib.util.spec_from_file_location(
        "_real_capload", os.path.join(cf.TOOLS, "capload", "__init__.py"))
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)
    LOADGEN_SITES = real.LOADGEN_SITES
    m = role["module"]
    if m in ("capgrpc.load", "capgrpc.oe_sink"):
        return {s: ("busy", "sleep") for s in GRPC_SITES if s != "drop.msg"}
    return {"capload": LOADGEN_SITES, "capstubs.prebaked_server": prebaked_server.SITE_MODES,
            "capstubs.null_llm_server": null_llm_server.SITE_MODES}[m]


V_SITE = {"vrdr": ("reader", "recv.msg"), "vstub": (None, "send.msg"),
          "vsrv": ("snl", "send.msg"), "vload": ("capload", "recv.chunk")}


def test_intervention_arms_are_their_base_arm_plus_one_injection():
    """V arms (card E2 / E3 / E4 'V'): the base arm's scenario and template with exactly one
    launch-level busy delta at the named end; the site exists in that tool (an unknown site
    exits 3 at start-up). Doubling arms (k2c, w2) change only the load role's process count."""
    n = 0
    for s, a in ALL:
        v = a.rsplit("-", 1)[-1]
        if v not in V_SITE:
            continue
        n += 1
        base = a[:-(len(v) + 1)]
        doc, bdoc = _scen(s, a), _scen(s, base)
        t, bt = _tpl(s, a), _tpl(s, base)
        assert doc["roles"] == bdoc["roles"], a
        assert doc.get("job_inputs") == bdoc.get("job_inputs"), a
        assert {k: x for k, x in t.items() if k not in ("name", "note", "launch_inject")} == \
            {k: x for k, x in bt.items() if k not in ("name", "note", "launch_inject")}, a
        assert bt["launch_inject"] == []
        [spec] = t["launch_inject"]
        role_, site = V_SITE[v]
        stub = "spb" if "spb" in _roles(doc) else "snl"
        assert spec == {"role": role_ or stub, "site": site, "mode": "busy",
                        "delta_ns": "TBD:delta_ns"}, a
        assert "busy" in _sites(_roles(doc)[spec["role"]])[site], a
    assert n == 12
    for s, a, base, load in (("e2a", "e2a-spb-k2c-mc", "e2a-spb-kc-mc", "reader"),
                             ("e2a", "e2a-snl-k2c-mc", "e2a-snl-kc-mc", "reader"),
                             ("e3", "e3a-snl-k2c-mc", "e3a-snl-kc-mc", "reader"),
                             ("e4", "e4b-w2-mc", "e4b-mc", "capload")):
        r, br = _roles(_scen(s, a)), _roles(_scen(s, base))
        assert r[load]["workers"] == 2 * br[load]["workers"]
        assert {k: x for k, x in r.items() if k != load} == \
            {k: x for k, x in br.items() if k != load}
        assert _tpl(s, a)["tiers"] == _tpl(s, base)["tiers"]


def test_metronome_only_arms_share_the_full_arms_tier_shape():
    """E3-3a 'with / without metronome': e3a-mono-X registers with the values of
    e3a-snl-kc-X, so their templates differ only in the gate (no reader in mono)."""
    for shape in ("mc", "md", "r"):
        mono, full = _scen("e3", "e3a-mono-" + shape), _scen("e3", "e3a-snl-kc-" + shape)
        assert sorted(_roles(mono)) == ["canary", "sidecar", "snl"]
        assert "job_inputs" not in mono
        sm, sf = _roles(mono)["snl"]["args"], _roles(full)["snl"]["args"]
        assert [x if x != "metronome-only" else "full" for x in sm] == sf
        tm, tf = _tpl("e3", "e3a-mono-" + shape), _tpl("e3", "e3a-snl-kc-" + shape)
        for k in ("tiers", "common", "axis", "arrival", "launch_inject"):
            assert tm[k] == tf[k], (shape, k)
        assert tiergen.tbd_names(tm) == tiergen.tbd_names(tf)
        assert {t["role"] for rd in tm["default_gate"]["readings"] for t in rd["terms"]} == \
            {"snl"}


def test_axis_shapes():
    """mc: M axis, C doubles at a fixed beat; md: M axis, beat varies at a fixed C; r: R axis,
    shortest output (max_tokens 1). Prompt sets: short token ids (E2a, E3); token-id L = 128
    (E4)."""
    for s, a in ALL:
        t = _tpl(s, a)
        shape = [x for x in ("mc", "md", "r") if x in a.split("-")][0]
        ups = [x for x in t["tiers"] if x["kind"] == "up"]
        assert len(ups) == (1 if "-nb-" in a else 8), a
        assert t["common"]["prompt_set"] == ("tid128" if s == "e4" else "short"), a
        if shape == "r":
            assert t["axis"]["name"] == "R" and t["common"]["max_tokens"] == 1, a
            assert all(isinstance(u["rate_rps"], str) and "streams_c" not in u for u in ups)
        else:
            assert t["axis"]["load_key"] == "nominal_msgs_per_s", a
            assert t["common"]["max_tokens"] == "TBD:n_tokens", a
            key = "streams_c" if shape == "mc" else "beat_ns"
            assert [u[key] for u in ups] == ["TBD:%s_%d" % ("c" if shape == "mc" else "beat_ns",
                                                            i) for i in range(len(ups))], a
            assert all(u["rate_rps"] == "from_streams_c" for u in ups), a


FP_FIXED = ["--tokenizer", "{param:tokenizer_dir}", "--trust-remote-code", "false",
            "--tokenizer-mode", "auto"]


def test_e4_frontend_main_tier_and_stub_tokenizer_settings():
    """Card E4: one new-arch frontend, postproc pool off, the input-processor pool size written
    explicitly, and no fewer cores than loop + pool + grpcio threads; the stub's tokenizer
    settings equal what the frontend's argv implies (S-PB: the fingerprint job input; S-NL: its
    own --tokenizer load), and the served model name is the tokenizer dir everywhere."""
    n = 0
    for a in _arms("e4"):
        doc = _scen("e4", a)
        rs = _roles(doc)
        stub = "spb" if "spb" in rs else "snl"
        if "frontend" in rs:
            n += 1
            fe = rs["frontend"]
            args = fe["args"]
            assert args[0] == "{param:tokenizer_dir}"
            assert _opt(args, "--openengine-endpoint") == "{addr:%s}" % stub, a
            assert _opt(args, "--num_serve_frontends") == "1", a
            assert _opt(args, "--num_postprocess_workers") == "0", a
            pool = int(_opt(args, "--num_input_processor_workers"))
            assert fe["cpus_needed"] >= 1 + pool + 1, a
            assert fe["measure_class"] == "under_test" and fe["probe_tier"] == "P0", a
            cl = rs["capload"]
            assert _opt(cl["args"], "--model") == "{param:tokenizer_dir}", a
            assert cl["workers"] == (2 * W_HTTP if "-w2-" in a else W_HTTP), a
            assert doc["reconcile"]["chains"][0]["hops"][0]["role"] == "capload"
        if stub == "spb":
            assert doc["job_inputs"]["fingerprint"]["args"] == FP_FIXED, a
            assert _opt(rs["spb"]["args"], "--model") == "{param:tokenizer_dir}", a
            assert "--sfe-module" in _strs(rs["spb"]["args"]), a
        else:
            sa = _strs(rs["snl"]["args"])
            assert sa[sa.index("--tokenizer"):sa.index("--tokenizer") + 6] == FP_FIXED, a
            assert _opt(sa, "--model") == "{param:tokenizer_dir}", a
            assert "--server-module" not in sa, a  # the product's server module
        if "reader" in rs:
            assert doc["job_inputs"]["request_bytes"]["args"][-1] == "{param:tokenizer_dir}"
    assert n == 11


def test_direct_arms_name_one_model_for_stub_and_request_bytes():
    for s in ("e2a", "e3"):
        for a in _arms(s):
            doc = _scen(s, a)
            rs = _roles(doc)
            if "reader" not in rs:
                continue
            rb = doc["job_inputs"]["request_bytes"]["args"]
            assert rb[rb.index("--model") + 1] == "{param:model_name}", a
            if "snl" in rs:
                assert _opt(rs["snl"]["args"], "--model") == "{param:model_name}", a
            if "spb" in rs:
                assert doc["job_inputs"]["prebaked"]["args"][3] == "{input:tier_table}", a

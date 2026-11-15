# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job inputs (errata B2B-9) in the scenario, the orchestrator (--job-input) and the judge's
input check, plus the new TOOL_TABLE rows (S-PB, caplaunch.smoke). Launches use the fake tools."""

import copy
import json
import os

import pytest

import conftest as cf
from capcommon import jobinputs as JI
from capcommon.exitcodes import ConfigError
from capcommon.hashing import sha256_file
from capcommon.runtree import read_state
from capjudge import inputs as JIN
from caplaunch import scenario as sc


def _scen(**top):
    doc = {"schema": sc.SCHEMA, "experiment": "E0", "name": "t", "tier_table": "a/t.json",
           "schedule": "a/s.capbin", "prompts": "a/p.json", "probe_tier": None,
           "probe_groups": [], "stop_grace_s": 1.0, "stop_order": ["spb"],
           "roles": [cf.role("spb", "capkit", "server", module="capstubs.prebaked_server",
                             args=["--prebaked", "{input:prebaked}"])],
           "params_required": ["ids"],
           "dims": {k: None for k in sc.DIMS_KEYS},
           "job_inputs": {"prebaked": {"args": ["--tier-table", "{input:tier_table}",
                                                "--token-ids", "{param:ids}"]}}}
    doc.update(top)
    doc["reconcile"] = {"schema": "capkit.reconcile/1", "chains": [
        {"name": "requests", "kind": "requests", "gate": None, "include_usage": None,
         "hops": [{"role": doc["roles"][0]["role"], "source": "summary.counts.ok",
                   "scope": "tier"}], "corrections": []}]}
    return doc


# ---- scenario ----------------------------------------------------------------------------------

def test_scenario_with_job_inputs_validates():
    roles = sc.validate_scenario(_scen())
    assert list(roles) == ["spb"]
    assert JI.kinds_of(_scen()) == ["prebaked"]
    assert JI.kinds_of({k: v for k, v in _scen().items() if k != "job_inputs"}) == []


@pytest.mark.parametrize("ji,msg", [
    ({}, "non-empty object"),
    ([], "non-empty object"),
    ({"prebaked": {"args": []}, "nope": {"args": []}}, "unknown kinds"),
    ({"prebaked": {"args": [], "module": "x"}}, "exactly {args"),
    ({"prebaked": {"args": ["ok", 3]}}, "non-empty strings"),
    ({"prebaked": {"args": ["--out", "x"]}}, "--out"),
    ({"prebaked": {"args": ["--out=x"]}}, "--out"),
    ({"prebaked": {"args": ["{param:other}"]}}, "placeholder"),
    ({"prebaked": {"args": ["{input:prebaked}"]}}, "placeholder"),
    ({"prebaked": {"args": ["{addr:spb}"]}}, "placeholder"),
])
def test_job_inputs_section_refusals(ji, msg):
    with pytest.raises(ConfigError, match=msg):
        sc.validate_scenario(_scen(job_inputs=ji))


def test_input_placeholder_of_a_job_input_needs_the_declaration():
    doc = _scen()
    doc["roles"][0]["args"] = ["--prebaked", "{input:request_bytes}"]
    with pytest.raises(ConfigError, match=r"\{input:request_bytes\} needs request_bytes"):
        sc.validate_scenario(doc)
    doc = _scen()
    del doc["job_inputs"]
    with pytest.raises(ConfigError, match="needs prebaked in the scenario's job_inputs"):
        sc.validate_scenario(doc)
    doc = _scen()
    doc["roles"][0]["args"] = ["--x", "{input:nope}"]
    with pytest.raises(ConfigError, match="unknown placeholder"):
        sc.validate_scenario(doc)
    doc = _scen()  # a splice_json item is checked the same way
    doc["roles"][0]["args"] = [{"splice_json": "{input:fingerprint}", "key": "sfe_args"}]
    with pytest.raises(ConfigError, match="needs fingerprint"):
        sc.validate_scenario(doc)
    doc["job_inputs"]["fingerprint"] = {"args": []}
    sc.validate_scenario(doc)


@pytest.mark.parametrize("workers,expected,ok", [(1, 1, True), (2, 2, False), (1, 2, False)])
def test_spb_is_single_process(workers, expected, ok):
    doc = _scen()
    doc["roles"][0].update(workers=workers, expected_procs=expected, cpus_needed=workers)
    if ok:
        sc.validate_scenario(doc)
    else:
        with pytest.raises(ConfigError, match="runs one process"):
            sc.validate_scenario(doc)


@pytest.mark.parametrize("workers,expected,ok", [(1, 1, True), (2, 2, False), (1, 2, False)])
def test_snl_is_single_process_and_gets_the_managed_flags(workers, expected, ok):
    """S-NL's TOOL_TABLE row (errata FT1-1): one process, port file, ready file, --workers."""
    doc = _scen(roles=[cf.role("snl", "capkit", "server", module="capstubs.null_llm_server",
                               args=["--mode", "full"])], stop_order=["snl"])
    del doc["job_inputs"]
    doc["roles"][0].update(workers=workers, expected_procs=expected, cpus_needed=workers)
    if not ok:
        with pytest.raises(ConfigError, match="runs one process"):
            sc.validate_scenario(doc)
        return
    snl = sc.validate_scenario(doc)["snl"]
    ctx = _ctx({})
    ctx.own_files["snl"] = {"ready_file": "/T/ready/snl.json"}
    argv = sc.build_argv(snl, ctx, [5], "py")
    assert argv[:3] == ["py", "-m", "capstubs.null_llm_server"]
    for pair in (["--loop", "asyncio"], ["--tier-table", "/W/t.json"], ["--origin-ns", "7"],
                 ["--ready-file", "/T/ready/snl.json"], ["--port-file", "/T/ports/snl.json"],
                 ["--workers", "1"], ["--cpus", "5"], ["--mode", "full"]):
        assert any(argv[i:i + 2] == pair for i in range(len(argv))), pair


@pytest.mark.parametrize("workers,expected,ok", [(1, 1, True), (2, 2, False), (1, 2, False)])
def test_oe_sink_is_single_process_without_port_file(workers, expected, ok):
    """oe_sink's TOOL_TABLE row (errata FT1-8): one process, ready file, --workers, no port."""
    doc = _scen(roles=[cf.role("sink", "capkit", "client", module="capgrpc.oe_sink",
                               args=["--tier", "generate-stream"])], stop_order=["sink"])
    del doc["job_inputs"]
    doc["roles"][0].update(workers=workers, expected_procs=expected, cpus_needed=workers)
    if not ok:
        with pytest.raises(ConfigError, match="runs one process"):
            sc.validate_scenario(doc)
        return
    sink = sc.validate_scenario(doc)["sink"]
    ctx = _ctx({})
    ctx.own_files["sink"] = {"ready_file": "/T/ready/sink.json"}
    argv = sc.build_argv(sink, ctx, [5], "py")
    assert argv[:3] == ["py", "-m", "capgrpc.oe_sink"]
    for pair in (["--loop", "asyncio"], ["--tier-table", "/W/t.json"], ["--origin-ns", "7"],
                 ["--ready-file", "/T/ready/sink.json"], ["--workers", "1"], ["--cpus", "5"],
                 ["--tier", "generate-stream"]):
        assert any(argv[i:i + 2] == pair for i in range(len(argv))), pair
    assert "--port-file" not in argv


class _Id:
    experiment, arm, launch_seq, job_id, restart_count, register_sha = \
        "E0", "a", 0, "local-t", 0, "f" * 64


def _ctx(inputs):
    return sc.ArgvContext(identity=_Id(), run_dir="/T", raw_dir="/T/raw", params={"ids": "1,2"},
                          stop_grace_s=1.0, shm_dir="/shm", snapshot_interval_ms=50,
                          tier_table_path="/W/t.json", origin_ns=7, own_files={
                              "spb": {"ready_file": "/T/ready/spb.json"},
                              "smoke": {"ready_file": None}},
                          port_file_of=lambda r: "/T/ports/%s.json" % r, addr_of=None,
                          inputs=inputs)


def test_argv_of_spb_and_smoke_roles():
    spb = sc.validate_scenario(_scen())["spb"]
    argv = sc.build_argv(spb, _ctx({"prebaked": "/T/raw/env/job_inputs/prebaked.json"}), [3],
                         "py")
    assert argv[:3] == ["py", "-m", "capstubs.prebaked_server"]
    for pair in (["--loop", "asyncio"], ["--tier-table", "/W/t.json"], ["--origin-ns", "7"],
                 ["--shm-dir", "/shm"], ["--ready-file", "/T/ready/spb.json"],
                 ["--port-file", "/T/ports/spb.json"], ["--workers", "1"],
                 ["--prebaked", "/T/raw/env/job_inputs/prebaked.json"]):
        assert any(argv[i:i + 2] == pair for i in range(len(argv))), pair
    smoke = cf.role("smoke", "capkit", "aux", asyncio=False, mclass="observer",
                    module="caplaunch.smoke", snapshots="none",
                    readiness=[{"kind": "file_exists", "path": "{raw}/smoke/smoke.json"}],
                    args=["--result-dir", "{raw}/smoke", "--items", "env", "--hold"])
    doc = _scen(roles=[smoke], stop_order=["smoke"])
    del doc["job_inputs"]
    smoke = sc.validate_scenario(doc)["smoke"]
    argv = sc.build_argv(smoke, _ctx({}), [4], "py")
    assert argv[3:] == ["--experiment", "E0", "--arm", "a", "--launch-seq", "0", "--job-id",
                        "local-t", "--restart-count", "0", "--register-sha", "f" * 64, "--role",
                        "smoke", "--run-dir", "/T", "--stop-grace-s", "1.0", "--cpus", "4",
                        "--result-dir", "/T/raw/smoke", "--items", "env", "--hold"]
    bad = copy.deepcopy(doc)
    bad["roles"][0]["args"] = ["--out", "x"]  # --out is orchestrator-managed
    with pytest.raises(ConfigError, match="managed"):
        sc.validate_scenario(bad)


# ---- orchestrator --job-input and the judge's input check --------------------------------------

def _gate():
    return cf.gate([("nullsse", "backlog.active_streams"), ("capload", "backlog.inflight")])


def _workdir_with_job_input(tmp_path):
    """The default fake-tool launch, its scenario declaring one job input (fingerprint), and a
    generated file + record as capjobs.jobinputs writes them."""
    wd = cf.make_workdir(tmp_path, cf.tier_doc(_gate()), cf.default_roles())
    with open(wd["scenario"]) as f:
        doc = json.load(f)
    doc["job_inputs"] = {"fingerprint": {"args": ["--tier", "{input:tier_table}"]}}
    with open(wd["scenario"], "w") as f:
        json.dump(doc, f)
    with open(wd["register"]) as f:
        reg = json.load(f)
    reg["files"]["scenario"]["sha256"] = sha256_file(wd["scenario"])
    with open(wd["register"], "w") as f:
        json.dump(reg, f)
    wd["register_sha"] = sha256_file(wd["register"])
    gen = os.path.join(str(tmp_path), "gen")
    os.makedirs(gen)
    path = os.path.join(gen, "fingerprint.json")
    write_input(path, {"schema": "capkit.fingerprint/1", "sfe_args": ["--vocab-size", "9"]},
                {"tier_table": reg["files"]["tier_table"]["sha256"]})
    return wd, path


def write_input(path, doc, inputs, **rec_over):
    with open(path, "w") as f:
        json.dump(doc, f)
    rec = {"schema": JI.RECORD_SCHEMA, "kind": "fingerprint", "module": "capstubs.fingerprint",
           "argv": [], "inputs": inputs, "rc": 0, "out_sha256": sha256_file(path)}
    rec.update(rec_over)
    with open(path + ".record.json", "w") as f:
        json.dump(rec, f)


def _run(tmp_path, wd, extra, name="T"):
    T = os.path.join(str(tmp_path), name)
    return cf.run_launch(cf.launch_argv(wd, T, os.path.join(str(tmp_path), "shm-" + name))
                         + extra), T


def test_launch_copies_and_records_the_job_input(tmp_path):
    wd, path = _workdir_with_job_input(tmp_path)
    r, T = _run(tmp_path, wd, ["--job-input", "fingerprint=%s" % path])
    assert r.returncode == 0, r.stderr
    assert [s[1] for s in read_state(T)] == ["VALID"]
    with open(os.path.join(T, "raw", "launch.json")) as f:
        L = json.load(f)
    ent = L["job_inputs"]["fingerprint"]
    copy_ = os.path.join(T, ent["relpath"])
    assert ent["relpath"] == "raw/env/job_inputs/fingerprint.json"
    assert open(copy_, "rb").read() == open(path, "rb").read()
    assert ent["sha256"] == sha256_file(path) and ent["source"] == path
    assert ent["record_sha256"] == sha256_file(copy_[:-5] + ".record.json")
    assert ent["module"] == "capstubs.fingerprint"
    # the judge's input check accepts the tree, and refuses it once a copy is changed
    ctx = _judge_ctx(wd, T, L)
    JIN._load_job_inputs(ctx)
    assert "raw/env/job_inputs/fingerprint.json" in ctx.files_read
    with open(copy_, "a") as f:
        f.write(" ")
    with pytest.raises(JIN.InputInvalid, match="job_inputs.fingerprint: sha256"):
        JIN._load_job_inputs(_judge_ctx(wd, T, L))


def _judge_ctx(wd, T, launch_doc):
    ctx = JIN.Ctx(wd["W"], T)
    ctx.launch = launch_doc
    with open(wd["register"]) as f:
        ctx.register_files = json.load(f)["files"]
    return ctx


def test_judge_job_input_refusals(tmp_path):
    wd, path = _workdir_with_job_input(tmp_path)
    r, T = _run(tmp_path, wd, ["--job-input", "fingerprint=%s" % path])
    assert r.returncode == 0, r.stderr
    with open(os.path.join(T, "raw", "launch.json")) as f:
        L = json.load(f)
    ok = copy.deepcopy(L)
    for change, msg in (
            (lambda d: d.pop("job_inputs"), r"kinds \[\] != scenario"),
            (lambda d: d["job_inputs"].update(prebaked={}), "kinds"),
            (lambda d: d["job_inputs"]["fingerprint"].update(relpath="raw/x.json"), "relpath"),
            (lambda d: d["job_inputs"]["fingerprint"].update(record_sha256="0" * 64), "record"),
            (lambda d: d["tier_table"].update(sha256="1" * 64), "built from tier_table")):
        d = copy.deepcopy(ok)
        change(d)
        with pytest.raises(JIN.InputInvalid, match=msg):
            JIN._load_job_inputs(_judge_ctx(wd, T, d))


@pytest.mark.parametrize("case,msg", [
    ("none", r"--job-input kinds \[\] != scenario job_inputs \['fingerprint'\]"),
    ("extra", "kinds"),
    ("twice", "each once"),
    ("noeq", "KIND=PATH"),
    ("relative", "absolute"),
    ("schema", "is not a capkit.fingerprint/1 file"),
    ("rc", "record rc 1"),
    ("outsha", "out_sha256"),
    ("inputsha", "built from tier_table"),
    ("kind", "record kind/module"),
    ("norecord", "record.json"),
])
def test_orchestrator_job_input_refusals(tmp_path, case, msg):
    wd, path = _workdir_with_job_input(tmp_path)
    tt_sha = json.load(open(wd["register"]))["files"]["tier_table"]["sha256"]
    arg = ["--job-input", "fingerprint=%s" % path]
    if case == "none":
        arg = []
    elif case == "extra":
        arg += ["--job-input", "prebaked=%s" % path]
    elif case == "twice":
        arg += arg
    elif case == "noeq":
        arg = ["--job-input", "fingerprint"]
    elif case == "relative":
        arg = ["--job-input", "fingerprint=gen/fingerprint.json"]
    elif case == "schema":
        write_input(path, {"schema": "capkit.prebaked/1"}, {"tier_table": tt_sha})
    elif case == "rc":
        write_input(path, {"schema": "capkit.fingerprint/1"}, {"tier_table": tt_sha}, rc=1)
    elif case == "outsha":
        write_input(path, {"schema": "capkit.fingerprint/1"}, {"tier_table": tt_sha},
                    out_sha256="0" * 64)
    elif case == "inputsha":
        write_input(path, {"schema": "capkit.fingerprint/1"}, {"tier_table": "2" * 64})
    elif case == "kind":
        write_input(path, {"schema": "capkit.fingerprint/1"}, {"tier_table": tt_sha},
                    module="capstubs.prebaked")
    elif case == "norecord":
        os.rename(path + ".record.json", path + ".record.json.moved")
    r, T = _run(tmp_path, wd, arg)
    assert r.returncode == 3, (r.returncode, r.stderr)
    assert msg in r.stderr or __import__("re").search(msg, r.stderr), r.stderr
    assert not os.path.exists(os.path.join(T, "raw", "launch.json"))

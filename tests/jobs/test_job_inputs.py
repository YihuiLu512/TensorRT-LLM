# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job inputs (errata B2B-9) on the capjobs side: capjobs.jobinputs.generate, capjobs.launches
running the generators before a launch, the time bounds, and the overlay manifest's stub_modules
in the snapshot and the in-job overlay step (errata B2B-11). The generators here are a fake
module behind the {PY} wrapper; the real generators run in tests/capstubs/test_spb_job_path.py."""

import json
import os
import sys

import pytest

from capcommon import jobinputs as JI
from capcommon.exitcodes import ConfigError
from capcommon.hashing import sha256_bytes, sha256_file
from conftest import CODE_ROOT

from capjobs import jobinputs, launches, launchplan, overlay, snapshot, submit
from test_container import launch_env

FAKEGEN = r'''
import json, os, sys
mod = sys.argv[2]
schema = {"capstubs.fingerprint": "capkit.fingerprint/1", "capgrpc.reqbytes": "capkit.reqbytes/1",
          "capstubs.prebaked": "capkit.prebaked/1"}[mod]
mode = os.environ.get("FAKEGEN_MODE", "") if os.environ.get("FAKEGEN_MOD") == mod else ""
assert sys.argv[-2] == "--out"
out = sys.argv[-1]
if mode == "fail":
    sys.exit(7)
if mode == "noout":
    sys.exit(0)
with open(out, "w") as f:
    json.dump({"schema": "capkit.other/1" if mode == "schema" else schema, "argv": sys.argv[3:]},
              f)
'''


def fake_python(tmp_path):
    """{PY}: the generator modules run FAKEGEN; anything else (capenv.pins) the real python."""
    gen = tmp_path / "fakegen.py"
    gen.write_text(FAKEGEN)
    py = tmp_path / "pyj"
    py.write_text("#!/bin/sh\ncase \"$2\" in\n  capstubs.fingerprint|capgrpc.reqbytes|"
                  "capstubs.prebaked) exec %s %s \"$@\";;\nesac\nPYTHONPATH=%s exec %s \"$@\"\n"
                  % (sys.executable, gen, os.path.join(CODE_ROOT, "tools"), sys.executable))
    py.chmod(0o755)
    return str(py)


SECTION = {"fingerprint": {"args": ["--tokenizer", "{param:tok}", "--trust-remote-code",
                                    "false", "--tokenizer-mode", "auto"]},
           "prebaked": {"args": ["--pb2-module", "m", "--tier-table", "{input:tier_table}",
                                 "--token-ids", "{param:ids}", "--prompt-tokens", "4"]}}


def scen_doc(w, section=SECTION):
    for n, data in (("t.json", b"tiers\n"), ("s.capbin", b"sched\n"), ("p.json", b"prompts\n")):
        with open(os.path.join(w, n), "wb") as f:
            f.write(data)
    return {"tier_table": "t.json", "schedule": "s.capbin", "prompts": "p.json",
            "params_required": ["tok", "ids"], "job_inputs": section}


# ---- capjobs.jobinputs ---------------------------------------------------------------------------

def test_params_from_argv():
    assert jobinputs.params_from_argv(["x", "--param", "a=1", "--param=b=c=d", "--paramx"]) == \
        {"a": "1", "b": "c=d"}
    for bad in (["--param", "a=1", "--param", "a=2"], ["--param", "noeq"], ["--param"]):
        with pytest.raises(ConfigError):
            jobinputs.params_from_argv(bad)


def test_generate_writes_files_and_records(tmp_path):
    w = str(tmp_path)
    doc = scen_doc(w)
    paths, recs = jobinputs.generate(doc, w, {"tok": "/tok", "ids": "1,2"}, fake_python(tmp_path),
                                     str(tmp_path / "out"), dict(os.environ), 60)
    assert list(paths) == ["fingerprint", "prebaked"] and [r["kind"] for r in recs] == list(paths)
    for kind, p in paths.items():
        assert p == str(tmp_path / "out" / ("%s.json" % kind))
        rec = json.load(open(p + ".record.json"))
        assert rec["rc"] == 0 and rec["out_sha256"] == sha256_file(p)
        assert rec["module"] == JI.KINDS[kind][0] and rec["argv"][-2:] == ["--out", p]
        JI.check_record(rec, kind, sha256_file(p), {"tier_table": sha256_bytes(b"tiers\n")})
    pb = json.load(open(paths["prebaked"]))
    assert pb["argv"][:4] == ["--pb2-module", "m", "--tier-table", os.path.join(w, "t.json")]
    assert pb["argv"][5] == "1,2"
    # only the registered inputs an args template names are recorded
    assert json.load(open(paths["prebaked"] + ".record.json"))["inputs"] == \
        {"tier_table": sha256_bytes(b"tiers\n")}
    assert json.load(open(paths["fingerprint"] + ".record.json"))["inputs"] == {}
    # nothing declared: nothing generated, no directory
    assert jobinputs.generate({}, w, {}, "py", str(tmp_path / "none"), {}, 60) == ({}, [])
    assert not os.path.exists(str(tmp_path / "none"))


@pytest.mark.parametrize("mode,msg", [("fail", "exit 7"), ("noout", "no "),
                                      ("schema", "capkit.other/1")])
def test_generate_failures(tmp_path, monkeypatch, mode, msg):
    w = str(tmp_path)
    monkeypatch.setenv("FAKEGEN_MOD", "capstubs.prebaked")
    monkeypatch.setenv("FAKEGEN_MODE", mode)
    with pytest.raises(ConfigError, match="job input prebaked: .*%s" % msg):
        jobinputs.generate(scen_doc(w), w, {"tok": "/t", "ids": "1"}, fake_python(tmp_path),
                           str(tmp_path / "out"), dict(os.environ), 60)
    rec = json.load(open(str(tmp_path / "out" / "prebaked.json.record.json")))
    assert rec["out_sha256"] is None  # a failed record never vouches for a file
    assert rec["rc"] == (7 if mode == "fail" else 0)


def test_generate_refuses_existing_output_and_missing_param(tmp_path):
    w = str(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "fingerprint.json").write_text("{}")
    with pytest.raises(ConfigError, match="already exists"):
        jobinputs.generate(scen_doc(w), w, {"tok": "/t", "ids": "1"}, fake_python(tmp_path),
                           str(out), dict(os.environ), 60)
    with pytest.raises(ConfigError, match="no --param ids"):
        jobinputs.generate(scen_doc(w), w, {"tok": "/t"}, fake_python(tmp_path),
                           str(tmp_path / "o2"), dict(os.environ), 60)


def test_cli_prints_caplaunch_arguments(tmp_path, capsys):
    w = str(tmp_path)
    with open(os.path.join(w, "scen.json"), "w") as f:
        json.dump(scen_doc(w), f)
    rc = jobinputs.main(["--workdir", w, "--scenario", "scen.json", "--python",
                         fake_python(tmp_path), "--out-dir", str(tmp_path / "o"),
                         "--param", "tok=/t", "--param", "ids=3"])
    assert rc == 0
    assert capsys.readouterr().out.split() == [
        "--job-input", "fingerprint=%s" % (tmp_path / "o" / "fingerprint.json"),
        "--job-input", "prebaked=%s" % (tmp_path / "o" / "prebaked.json")]
    with open(os.path.join(w, "none.json"), "w") as f:
        json.dump({}, f)
    assert jobinputs.main(["--workdir", w, "--scenario", "none.json", "--python", "py",
                           "--out-dir", str(tmp_path / "o3")]) == 3


# ---- capjobs.launches ----------------------------------------------------------------------------

def jobinput_env(tmp_path, modes=("ok",)):
    a, plan, manifest = launch_env(tmp_path, list(modes))
    a.python = fake_python(tmp_path)
    w = os.path.join(str(tmp_path), "W")
    doc = dict(json.load(open(os.path.join(w, "scen.json"))), **scen_doc(w))
    with open(os.path.join(w, "scen.json"), "w") as f:
        json.dump(doc, f)
    for ln in plan["launches"]:
        ln["argv"] += ["--param", "tok={SNAP}/tok", "--param", "ids=5,6"]
    return a, plan, manifest


def test_launches_generate_job_inputs_before_the_launch(tmp_path):
    a, plan, manifest = jobinput_env(tmp_path)
    res = launches.run_all(a, plan, manifest)
    assert res[0]["status"] == "VALID", res[0]["reason"]
    name = res[0]["tree"]
    d = os.path.join(a.job_dir, "job_inputs", name)
    want = ["fingerprint=%s" % os.path.join(d, "fingerprint.json"),
            "prebaked=%s" % os.path.join(d, "prebaked.json")]
    seen = json.load(open(os.path.join(a.run_root, name, "raw", "launch.json")))["job_inputs"]
    assert seen == want
    assert [x for x in res[0]["argv"] if x.startswith(("fingerprint=", "prebaked="))] == want
    assert [(j["kind"], j["rc"]) for j in res[0]["job_inputs"]] == [("fingerprint", 0),
                                                                    ("prebaked", 0)]
    fp = json.load(open(os.path.join(d, "fingerprint.json")))
    assert fp["argv"][1] == os.path.join(a.snap, "tok")  # {SNAP} substituted before --param


def test_launches_generator_failure_is_not_run(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKEGEN_MOD", "capstubs.fingerprint")
    monkeypatch.setenv("FAKEGEN_MODE", "fail")
    a, plan, manifest = jobinput_env(tmp_path, ("ok", "ok"))
    res = launches.run_all(a, plan, manifest)
    for r in res:  # each launch generates its own inputs; both fail the same way here
        assert r["status"] == "NOT_RUN" and r["ran"] is False and r["job_input_failed"]
        assert "job inputs" in r["reason"] and "exit 7" in r["reason"]
        assert r["job_inputs"][0]["rc"] == 7
        assert not os.path.exists(os.path.join(a.run_root, r["tree"]))


def test_in_job_bound_counts_each_declared_kind(tmp_path):
    a, plan, manifest = jobinput_env(tmp_path)
    ln = plan["launches"][0]
    assert launchplan.in_job_bound_s(ln) == launchplan.PINS_TIMEOUT_S + \
        2 * launchplan.JOB_INPUT_TIMEOUT_S
    renv = {"idle_window_s": 3, "slurm_view_wait_s": 4}
    b = submit.time_budget(plan, renv)
    assert b["parts"]["pin_tables_s"] == launchplan.PINS_TIMEOUT_S
    assert b["parts"]["job_inputs_s"] == 2 * launchplan.JOB_INPUT_TIMEOUT_S
    assert b["total_s"] == sum(b["parts"].values())
    # the remaining-time check before the launch includes the generators
    need = ln["budget_s"] + plan["kill_grace_s"] + plan["margin_s"] + \
        launchplan.in_job_bound_s(ln)
    res = launches.run_all(a, plan, manifest, now_fn=lambda: a.deadline_epoch - need + 1)
    assert res[0]["status"] == "NOT_RUN" and "pin table and job inputs %d s" % (
        launchplan.PINS_TIMEOUT_S + 2 * launchplan.JOB_INPUT_TIMEOUT_S) in res[0]["reason"]


def test_reserved_job_input_flag_in_plan_argv_is_refused():
    plan = {"schema": launchplan.SCHEMA, "experiment": "E0", "register_sha": "b" * 64,
            "job_arm": "job", "margin_s": 1, "kill_grace_s": 1,
            "launches": [{"launch_seq": 0, "arm": "a", "budget_s": 1,
                          "argv": ["{PY}", "-m", "caplaunch", "--job-input", "prebaked=/x"]}]}
    with pytest.raises(launchplan.PlanError, match="--job-input"):
        launchplan.validate(plan)


# ---- overlay manifest stub_modules ---------------------------------------------------------------

def stub_entry(src="exp/tests/fake_openengine.py", data=b"SFE = 1\n", key="sfe"):
    return {"key": key, "repo_path": "tests/fake_openengine.py", "commit": "c" * 40, "src": src,
            "sha256": sha256_bytes(data), "size": len(data)}


def test_snapshot_lists_stub_modules():
    doc = {"files": [], "capkit_packages": [], "stub_modules": [stub_entry()]}
    assert snapshot.overlay_entries(doc) == [("exp/tests/fake_openengine.py",
                                              sha256_bytes(b"SFE = 1\n"), 8)]
    bad = [{"stub_modules": {"sfe": stub_entry()}},
           {"stub_modules": [stub_entry(), stub_entry()]},
           {"stub_modules": [dict(stub_entry(), dst="x.py")]},
           {"stub_modules": [{k: v for k, v in stub_entry().items() if k != "commit"}]},
           {"stub_modules": [stub_entry(src="/abs/fake_openengine.py")]}]
    for d in bad:
        with pytest.raises(snapshot.SnapshotError):
            snapshot.overlay_entries(d)


def test_overlay_step_verifies_stub_modules_in_place(tmp_path):
    src = tmp_path / "snap"
    (src / "exp" / "tests").mkdir(parents=True)
    (src / "exp" / "tests" / "fake_openengine.py").write_bytes(b"SFE = 1\n")
    dest = tmp_path / "site"
    dest.mkdir()
    doc = {"files": [], "capkit_packages": [], "stub_modules": [stub_entry()]}
    results, passed, expected = overlay.apply(doc, str(src), str(dest))
    assert passed and expected == 1 and results[0]["list"] == "stub_modules"
    assert results[0]["path"] == str(src / "exp" / "tests" / "fake_openengine.py")
    assert os.listdir(str(dest)) == []  # never installed into site-packages
    (src / "exp" / "tests" / "fake_openengine.py").write_bytes(b"SFE = 2\n")
    results, passed, _ = overlay.apply(doc, str(src), str(dest))
    assert not passed and results[0]["problem"] == "source differs from manifest"

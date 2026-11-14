# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job inputs (errata B2B-9) on the capjobs side: capjobs.jobinputs.generate, capjobs.launches
running the generators before a launch, the time bounds, and the overlay manifest's stub_modules
in the snapshot and the in-job overlay step (errata B2B-11). The generators here are a fake
module behind the {PY} wrapper; the real generators run in tests/capstubs/test_spb_job_path.py."""

import json
import os
import signal
import subprocess
import sys
import time

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
if mode == "sleep":  # a slow generator with a child of its own; both pids for the test
    import subprocess, time
    kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    with open(os.environ["FAKEGEN_PIDS"] + ".tmp", "w") as f:
        f.write("%d %d" % (os.getpid(), kid.pid))
    os.rename(os.environ["FAKEGEN_PIDS"] + ".tmp", os.environ["FAKEGEN_PIDS"])
    time.sleep(300)
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


def _gone(pid):
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


def test_sigterm_during_a_generator_stops_it_and_writes_the_record(tmp_path, monkeypatch):
    """RS4/RS7: the job's SIGTERM while a job-input generator runs (up to 600 s each) stops the
    generator's whole process group within the poll period; the launch is NOT_RUN (not
    job_input_failed), the rest are skipped, and launches.json is written before KillWait."""
    pids = tmp_path / "gen.pids"
    monkeypatch.setenv("FAKEGEN_MOD", "capstubs.fingerprint")
    monkeypatch.setenv("FAKEGEN_MODE", "sleep")
    monkeypatch.setenv("FAKEGEN_PIDS", str(pids))
    a, plan, manifest = jobinput_env(tmp_path, ("ok", "ok"))
    pp, mp = tmp_path / "plan.json", tmp_path / "manifest.json"
    pp.write_text(json.dumps(plan))
    mp.write_text(json.dumps(manifest))
    out = os.path.join(a.job_dir, "launches.json")
    env = dict(os.environ, PYTHONPATH=os.path.join(CODE_ROOT, "tools"))
    p = subprocess.Popen([sys.executable, "-B", "-m", "capjobs.launches", "--plan", str(pp),
                          "--overlay-manifest", str(mp), "--nvidia-smi", a.nvidia_smi, "--snap",
                          a.snap, "--job-dir", a.job_dir, "--local-root", a.local_root,
                          "--shm-root", a.shm_root, "--run-root", a.run_root, "--job-id", "9",
                          "--hw-profile", a.hw_profile, "--python", a.python, "--out", out,
                          "--restart-count", "0", "--deadline-epoch", "1e12"], env=env)
    try:
        end = time.monotonic() + 60
        while not pids.exists() and time.monotonic() < end and p.poll() is None:
            time.sleep(0.1)
        assert pids.exists(), "the generator never started"
        gen, kid = map(int, pids.read_text().split())
        t0 = time.monotonic()
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=30) == 0
        assert time.monotonic() - t0 < launches.TERM_GRACE_S
    finally:
        if p.poll() is None:
            p.kill()
        if pids.exists():  # never leave the stand-in generator behind, even on failure
            for pid in map(int, pids.read_text().split()):
                if not _gone(pid):
                    os.kill(pid, signal.SIGKILL)
    doc = json.load(open(out))
    r0, r1 = doc["launches"]
    assert r0["status"] == "NOT_RUN" and r0["ran"] is False, r0
    assert "SIGTERM" in r0["reason"] and "job inputs" in r0["reason"], r0["reason"]
    assert not r0.get("job_input_failed") and doc["job_input_failed"] == 0
    assert r1["status"] == "NOT_RUN" and "SIGTERM" in r1["reason"]
    end = time.monotonic() + 10
    while not (_gone(gen) and _gone(kid)) and time.monotonic() < end:
        time.sleep(0.1)
    assert _gone(gen) and _gone(kid)  # the generator's whole process group


def test_pin_table_and_job_inputs_run_through_the_polled_helper(tmp_path, monkeypatch):
    """Both in-job helper commands go through launches.run_helper (SIGTERM-aware): a stop in
    either is NOT_RUN for the job's SIGTERM, not a pin-table or job-input failure."""
    real = launches.run_helper
    for stop_at, where in (("capenv.pins", "pin table"), ("capstubs.fingerprint", "job inputs")):
        d = tmp_path / stop_at
        d.mkdir()
        a, plan, manifest = jobinput_env(d, ("ok",))
        calls = []

        def fake(argv, stop_at=stop_at, calls=calls, **kw):
            calls.append(argv[2])
            if argv[2] == stop_at:
                raise launches.JobTerminated("the job received SIGTERM")
            return real(argv, **kw)
        monkeypatch.setattr(launches, "run_helper", fake)
        res = launches.run_all(a, plan, manifest)
        assert calls[-1] == stop_at and res[0]["status"] == "NOT_RUN", res[0]
        assert where in res[0]["reason"] and "SIGTERM" in res[0]["reason"], res[0]["reason"]
        assert not res[0].get("pin_table_failed") and not res[0].get("job_input_failed")


def test_run_helper_timeout_kills_the_group(tmp_path):
    """The timeout path keeps subprocess.run's contract (TimeoutExpired, nothing left) and also
    takes the helper's children."""
    pidf = tmp_path / "kid.pid"
    code = ("import subprocess, sys, time; k = subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(300)']); open(%r, 'w').write(str(k.pid)); time.sleep(300)"
            % str(pidf))
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        launches.run_helper([sys.executable, "-c", code], timeout=2.0)
    assert time.monotonic() - t0 < 2.0 + 2 * launches.POLL_S + 1.0
    kid = int(pidf.read_text())
    end = time.monotonic() + 10
    while not _gone(kid) and time.monotonic() < end:
        time.sleep(0.1)
    assert _gone(kid)
    assert launches.run_helper([sys.executable, "-c", "raise SystemExit(4)"],
                               timeout=30).returncode == 4


@pytest.mark.parametrize("case", ["missing_param", "output_exists", "input_unreadable"])
def test_failure_before_a_generator_runs_keeps_the_earlier_records(tmp_path, case):
    """RS22: a kind that fails before its generator runs (no --param, output already there,
    registered input unreadable) is a job-input failure like any other: the records of the
    kinds already generated stay in launches.json, the failing kind is listed with rc null."""
    a, plan, manifest = jobinput_env(tmp_path, ("ok",))
    ln = plan["launches"][0]
    name = launches.tree_dir_name(a.job_id, a.restart_count, ln["launch_seq"], ln["arm"])
    if case == "missing_param":
        ln["argv"] = ln["argv"][:-2]  # drops --param ids=5,6 (only prebaked uses it)
    elif case == "output_exists":
        d = os.path.join(a.job_dir, "job_inputs", name)
        os.makedirs(d)
        open(os.path.join(d, "prebaked.json"), "w").close()
    else:
        os.rename(os.path.join(str(tmp_path), "W", "t.json"),
                  os.path.join(str(tmp_path), "W", "t.json.gone"))
    res = launches.run_all(a, plan, manifest)
    r = res[0]
    assert r["status"] == "NOT_RUN" and r["job_input_failed"], r
    assert [(j["kind"], j["rc"]) for j in r["job_inputs"]] == [("fingerprint", 0),
                                                               ("prebaked", None)], r
    assert r["reason"].startswith("skipped: job inputs: job input prebaked: "), r["reason"]
    assert r["reason"].count("job input prebaked") == 1, r["reason"]


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


def _bad_scenario(doc, bad):
    """A scenario with a job_inputs key that capjobs reads before caplaunch's validate_scenario
    does: a key missing or of the wrong type, a misspelt or no kind."""
    if bad == "missing_prompts":
        del doc["prompts"]
    elif bad == "prompts_int":
        doc["prompts"] = 5
    elif bad == "params_required_int":
        doc["params_required"] = 5
    elif bad == "typo_kind":
        doc["job_inputs"] = {"prebake": {"args": []}}
    else:
        doc["job_inputs"] = {}
    return doc


BAD_SCENARIOS = {"missing_prompts": "prompts must be", "prompts_int": "prompts must be",
                 "params_required_int": "params_required must be",
                 "typo_kind": "unknown kinds", "empty_section": "non-empty object"}


@pytest.mark.parametrize("bad", sorted(BAD_SCENARIOS))
def test_generate_checks_what_it_reads_of_the_scenario(tmp_path, bad):
    """RS32: with a job_inputs key, generate checks the section and the scenario keys it reads
    (params_required, tier_table, schedule, prompts) as caplaunch's validate_scenario does: a
    ConfigError before anything is generated, never a KeyError or TypeError."""
    w = str(tmp_path)
    doc = _bad_scenario(scen_doc(w), bad)
    with pytest.raises(ConfigError, match="scenario .*%s" % BAD_SCENARIOS[bad]):
        jobinputs.generate(doc, w, {"tok": "/t", "ids": "1"}, fake_python(tmp_path),
                           str(tmp_path / "out"), dict(os.environ), 60)
    assert not os.path.exists(str(tmp_path / "out"))


@pytest.mark.parametrize("bad", sorted(BAD_SCENARIOS))
def test_bad_job_inputs_scenario_fails_only_its_launch(tmp_path, monkeypatch, bad):
    """RS32/RS22: such a scenario fails its own launch (NOT_RUN, job_input_failed);
    launches.json is written, the next launch runs, and capjobs.launches exits 3."""
    monkeypatch.setattr(launches, "become_subreaper", lambda: False)  # not on pytest itself
    a, plan, manifest = jobinput_env(tmp_path, ("ok", "ok"))
    w = os.path.join(str(tmp_path), "W")
    doc = _bad_scenario(json.load(open(os.path.join(w, "scen.json"))), bad)
    with open(os.path.join(w, "bad.json"), "w") as f:
        json.dump(doc, f)
    argv = plan["launches"][0]["argv"]
    argv[argv.index("--scenario") + 1] = "bad.json"
    pp, mp = tmp_path / "plan.json", tmp_path / "manifest.json"
    pp.write_text(json.dumps(plan))
    mp.write_text(json.dumps(manifest))
    out = os.path.join(a.job_dir, "launches.json")
    assert launches.main(["--plan", str(pp), "--overlay-manifest", str(mp), "--nvidia-smi",
                          a.nvidia_smi, "--snap", a.snap, "--hw-profile", a.hw_profile,
                          "--job-dir", a.job_dir, "--local-root", a.local_root, "--shm-root",
                          a.shm_root, "--run-root", a.run_root, "--job-id", "9", "--python",
                          a.python, "--out", out, "--restart-count", "0",
                          "--deadline-epoch", "1e12"]) == 3
    res = json.load(open(out))
    r0, r1 = res["launches"]
    assert r0["status"] == "NOT_RUN" and r0["ran"] is False and r0["job_input_failed"], r0
    assert r0["reason"].startswith("skipped: job inputs: ConfigError: scenario"), r0["reason"]
    assert r1["status"] == "VALID" and not r1.get("job_input_failed"), r1
    assert res["job_input_failed"] == 1


@pytest.mark.parametrize("exc", [KeyError, TypeError])
def test_unexpected_generate_error_fails_only_its_launch(tmp_path, monkeypatch, capsys, exc):
    """RS32 backstop: a KeyError or TypeError left in generation still fails only its launch
    (job_input_failed, the exception type in the reason); the command line exits 3."""
    def boom(*_a, **_k):
        raise exc("leftover")
    a, plan, manifest = jobinput_env(tmp_path, ("ok", "ok"))
    monkeypatch.setattr(jobinputs, "generate", boom)
    res = launches.run_all(a, plan, manifest)
    for r in res:
        assert r["status"] == "NOT_RUN" and r["job_input_failed"], r
        assert r["reason"].startswith("skipped: job inputs: %s" % exc.__name__), r["reason"]
    w = os.path.join(str(tmp_path), "W")
    assert jobinputs.main(["--workdir", w, "--scenario", "scen.json", "--python", "py",
                           "--out-dir", str(tmp_path / "o")]) == 3
    assert "leftover" in capsys.readouterr().err


def test_scenario_without_job_inputs_does_not_parse_params(tmp_path):
    """RS24: a scenario with no job_inputs key needs no job inputs, so its --param values are
    not parsed here; a repeated --param is caplaunch's own refusal, not a job-input failure."""
    a, plan, _manifest = launch_env(tmp_path, ["ok"])
    ln = plan["launches"][0]
    ln["argv"] += ["--param", "a=1", "--param", "a=2"]
    values = {"PY": a.python, "SNAP": a.snap, "JOB_DIR": a.job_dir, "SHM_ROOT": a.shm_root,
              "LOCAL_ROOT": a.local_root, "JOB_ID": a.job_id, "RESTART_COUNT": a.restart_count}
    assert launches.job_inputs(a, ln, values, {}, str(tmp_path / "ji")) == ({}, [], None)
    assert not os.path.exists(str(tmp_path / "ji"))


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

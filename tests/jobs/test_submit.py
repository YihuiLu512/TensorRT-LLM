# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests of capjobs.submit with a stub runner (no ssh is ever executed)."""

import json
import os
import subprocess

import pytest
from conftest import CODE_ROOT, DATA, SAVED_CDB, external_input

from capcommon import ledger, policy
from capcommon.hashing import sha256_bytes, sha256_file
from capjobs import snapshot, submit

HEAD = "c" * 40
REG_ENV = {"gpus": 1, "idle_window_s": 30, "idle_busy_max_frac": 0.02, "slurm_view_wait_s": 300,
           "chassis_source": "slurm_view"}
BASIS = "2026-11-13 18:00 user approved E0 smoke (DECISIONS 1, item 7) -> submit E0 job"
PARTS = ("b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,"
         "b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb")


class Env:
    def __init__(self, root):
        self.root = str(root)
        self.w = os.path.join(self.root, "W")
        os.makedirs(self.w)
        with open(os.path.join(self.w, "STATUS.md"), "w") as f:
            f.write("# STATUS\n\n- earlier line\n%s\n" % BASIS)
        self.reg_sha = self.register("register.json", REG_ENV)
        self.image = os.path.join(self.root, "img.sqsh")
        with open(self.image, "wb") as f:
            f.write(b"x" * 1000)
        self.image_sha = sha256_file(self.image)
        self.tok = os.path.join(self.root, "tok")
        os.makedirs(self.tok)
        with open(os.path.join(self.tok, "tokenizer.json"), "w") as f:
            f.write("{}")
        entries = []
        for rel in ("tools/capcommon/__init__.py", "tools/capjobs/__init__.py"):
            p = os.path.join(CODE_ROOT, rel)
            dst = rel.split("/", 1)[1]
            entries.append({"src": rel, "dst": dst, "sha256": sha256_file(p),
                            "size": os.path.getsize(p)})
        self.manifest = {"tier": "P0", "image_commit": "9" * 40, "base_commit": "2" * 40,
                         "target_commit": "2" * 40, "groups": [], "files": [],
                         "capkit_packages": entries, "capkit_code_sha": HEAD,
                         "non_python_changed": [], "sites": {}}
        self.manifest_path = self.write_json("overlay_manifest.json", self.manifest)
        self.plan = {"schema": "capkit.launch_plan/1", "experiment": "E0",
                     "register_sha": self.reg_sha, "job_arm": "job", "margin_s": 60,
                     "kill_grace_s": 60,
                     "launches": [{"launch_seq": i, "arm": arm, "budget_s": 600,
                                   "argv": ["{PY}", "-m", "caplaunch", "--scenario",
                                            "{SNAP}/configs/e0/e0a-nullsse.scenario.json",
                                            "--kill-after-s", "2"]}
                                  for i, arm in enumerate(["nullsse-a", "sfe-a"])]}
        self.plan_path = self.write_json("plan.json", self.plan)

    def register(self, name, environment, experiment="E0"):
        """A real register.json under <W> with an "environment" section, entered in REGISTRY."""
        rel = "02_e0/E0/register/" + name
        doc = {"schema": "capkit.register/1", "experiment": experiment, "arms": ["nullsse-a"]}
        if environment is not None:
            doc["environment"] = environment
        p = os.path.join(self.w, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(doc, f)
        sha = sha256_file(p)
        ledger.registry_append(os.path.join(self.w, "REGISTRY.tsv"), experiment, rel, sha,
                               "kind=register;")
        return sha

    def plan_for(self, name, reg_sha):
        return self.write_json(name, dict(self.plan, register_sha=reg_sha))

    def write_json(self, name, doc):
        p = os.path.join(self.root, name)
        with open(p, "w") as f:
            json.dump(doc, f)
        return p

    def submit_args(self, **over):
        a = {"--account": "acct", "--partition": PARTS, "--qos": "batch", "--exclude": "none",
             "--time": "01:00:00", "--mem": "1900G", "--cpus-per-task": "224", "--gpus": "1",
             "--job-name": "capkit-E0", "--login-host": "fe", "--ssh-timeout-s": "60",
             "--workdir": self.w, "--code-root": CODE_ROOT, "--experiment": "E0",
             "--register-sha": self.reg_sha, "--arms": "nullsse-a,sfe-a", "--launches": "2",
             "--basis": BASIS, "--launch-plan": self.plan_path,
             "--overlay-manifest": self.manifest_path, "--image": self.image,
             "--image-sha256": self.image_sha, "--image-size": "1000",
             "--mount": "%s:%s" % (self.root, self.root), "--tokenizer-dir": self.tok,
             "--snapshot-root": os.path.join(self.w, "snapshots"),
             "--run-root": os.path.join(self.w, "runs"),
             "--local-base": os.path.join(self.root, "local"),
             "--shm-base": os.path.join(self.root, "shm"), "--min-local-free-bytes": "1000",
             "--srun-mpi": "none", "--container-python": "python3",
             "--nvidia-smi": "/usr/bin/nvidia-smi", "--gate-module": "capenv.gate",
             "--registered-hw-sha": "e" * 64, "--no-comparable": None,
             "--include": ["configs"]}
        a.update(over)
        argv = ["submit"]
        for k, v in a.items():
            if v is False:
                continue
            if isinstance(v, list):
                for x in v:
                    argv += [k, x]
            elif v is None:
                argv.append(k)
            else:
                argv += [k, v]
        return argv

    def ledger_rows(self):
        return ledger.read_tsv(os.path.join(self.w, "submissions.tsv"), ledger.SUBMISSION_COLUMNS)


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def clean_state(_root, _paths):
    return HEAD, []


def run(argv, runner=None, code_state_fn=clean_state):
    return submit.main(argv, runner=runner, code_state_fn=code_state_fn)


def test_dry_run_writes_nothing(env, stub_runner, capsys):
    r = stub_runner({})
    assert run(env.submit_args(**{"--dry-run": None}), r) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and out["remote_argv"][:2] == ["sbatch", "--parsable"]
    assert out["assertions"] == {k: True for k in ("experiment_no_model", "gpus_le_1",
                                                   "no_model_mounts", "tokenizer_no_weights",
                                                   "basis_present")}
    assert r.calls == [] and not os.path.exists(os.path.join(env.w, "submissions.tsv"))
    assert not os.path.exists(os.path.join(env.w, "snapshots"))


def test_submit_end_to_end(env, stub_runner, capsys):
    r = stub_runner({"sbatch": (0, "5601234\n", "")})
    assert run(env.submit_args(), r) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["job_id"] == "5601234" and len(r.calls) == 1
    remote_cmd = r.calls[0][-1]
    assert remote_cmd.startswith("sbatch --parsable ") and "--test-only" not in remote_cmd
    assert "--exclusive" in remote_cmd and "--cpus-per-task=224" in remote_cmd
    rows = env.ledger_rows()
    assert [x["event"] for x in rows] == ["PLANNED", "SUBMITTED"]
    planned = rows[0]
    assert planned["basis"] == BASIS and planned["gpus"] == "1" and planned["exclude"] == "none"
    assert planned["reason"] == "purpose=measurement"
    assert json.loads(planned["assertions_json"])["basis_present"] is True
    assert planned["tokenizer_dir_sha256"] != "none"
    cur = ledger.submission_for_job(os.path.join(env.w, "submissions.tsv"), "5601234")
    assert cur["plan_id"] == planned["plan_id"] and cur["submitted_at"]
    assert ledger.parse_wall_iso(cur["submitted_at"]) >= ledger.parse_wall_iso(
        planned["event_at"])
    snap = planned["snapshot_dir"]
    ok, problems = snapshot.verify(snap)
    assert ok, problems
    with open(os.path.join(snap, snapshot.MANIFEST), "rb") as f:
        assert sha256_bytes(f.read()) == planned["snapshot_sha256"]
    assert not os.access(os.path.join(snap, "jobs", "capkit_job.sbatch"), os.W_OK)
    assert os.path.isfile(os.path.join(snap, "tools", "capjobs", "submit.py"))
    assert not os.path.exists(os.path.join(snap, "jobs", "prep_tokenizer.sbatch"))
    assert json.loads(planned["sbatch_argv"])[-1] == snap
    sh = subprocess.run(["bash", "-c", 'source "$1/job.env"; printf "%s|%s|%s" '
                         '"$CAPJOBS_SNAP" "$CAPJOBS_MOUNTS" "$CAPJOBS_TIME_LIMIT_S"', "x", snap],
                        capture_output=True, text=True, check=True)
    assert sh.stdout == "%s|%s:%s|3600" % (snap, env.root, env.root)
    # K1/K6/K23: the job's hard limits come from register.json "environment"; the image sha256
    # is the one computed here, and the job re-checks size and mtime_ns only.
    sh = subprocess.run(["bash", "-c", 'source "$1/job.env"; printf "%s|%s|%s|%s|%s|%s|%s" '
                         '"$CAPJOBS_MAX_ALLOC_GPUS" "$CAPJOBS_IDLE_WINDOW_S" '
                         '"$CAPJOBS_IDLE_BUSY_MAX_FRAC" "$CAPJOBS_SLURM_VIEW_WAIT_S" '
                         '"$CAPJOBS_CHASSIS_SOURCE" "$CAPJOBS_IMAGE_SHA256" '
                         '"$CAPJOBS_IMAGE_MTIME_NS"', "x", snap],
                        capture_output=True, text=True, check=True)
    assert sh.stdout.split("|") == ["1", "30", "0.02", "300", "slurm_view", env.image_sha,
                                    str(os.stat(env.image).st_mtime_ns)]
    assert out["time_budget"]["total_s"] == 2 * (600 + 60 + 60) + 30 + 300


def test_submit_failure_records_ended(env, stub_runner):
    r = stub_runner({"sbatch": (1, "", "sbatch: error: Batch job submission failed")})
    assert run(env.submit_args(), r) == 4
    rows = env.ledger_rows()
    assert [x["event"] for x in rows] == ["PLANNED", "ENDED"]
    assert rows[1]["status"] == "SUBMIT_FAILED" and "submission failed" in rows[1]["reason"]


@pytest.mark.parametrize("answer", [(0, "Submitted batch job 12\n", ""),
                                    (None, "", "TimeoutExpired: ssh timed out"),
                                    (255, "", "ssh: connection reset")])
def test_submit_outcome_unknown_is_not_ended(env, stub_runner, answer):
    """K16: sbatch may have queued the job; the plan stays open as UNKNOWN_SUBMIT (no ENDED row,
    no resubmission) until reconcile-submit closes it from evidence."""
    r = stub_runner({"sbatch": answer})
    assert run(env.submit_args(), r) == 4
    assert len(r.calls) == 1  # never retried
    rows = env.ledger_rows()
    assert [x["event"] for x in rows] == ["PLANNED", "NOTE"]
    assert rows[1]["status"] == "UNKNOWN_SUBMIT" and rows[1]["submitted_at"]
    assert rows[1]["reason"].startswith("purpose=measurement; ")


def _unknown_plan(env, stub_runner):
    assert run(env.submit_args(), stub_runner({"sbatch": (None, "", "timeout")})) == 4
    rows = env.ledger_rows()
    return rows[0]["plan_id"], rows[0]["snapshot_dir"]


def test_reconcile_submit_by_scontrol_then_record_end(env, stub_runner, capsys):
    plan_id, snap = _unknown_plan(env, stub_runner)
    end = ["record-end", "--workdir", env.w, "--plan-id", plan_id, "--login-host", "fe",
           "--ssh-timeout-s", "60", "--status", "VALID", "--reason", "judged"]
    done = stub_runner({"squeue": (0, "", ""), "sacct": (0, "88|COMPLETED|0:0\n", "")})
    assert run(end, done) == 3 and "reconcile-submit" in capsys.readouterr().err
    rec = ["reconcile-submit", "--workdir", env.w, "--plan-id", plan_id, "--login-host", "fe",
           "--ssh-timeout-s", "60", "--reason", "squeue showed job 88"]
    other = stub_runner({"scontrol": (0, "JobId=88 Command=/elsewhere/capkit_job.sbatch\n", "")})
    assert run(rec + ["--job-id", "88"], other) == 3  # evidence names another script
    mine = stub_runner({"scontrol": (0, "JobId=88 JobName=x\n   Command=%s/jobs/capkit_job.sbatch"
                                        "\n" % snap, "")})
    assert run(rec + ["--job-id", "88"], mine) == 0
    rows = env.ledger_rows()
    assert rows[-1]["event"] == "SUBMITTED" and rows[-1]["job_id"] == "88"
    assert rows[-1]["submitted_at"] == rows[1]["submitted_at"]  # the sbatch attempt time
    assert run(rec + ["--job-id", "88"], mine) == 3  # no longer UNKNOWN_SUBMIT
    cur = ledger.submissions_current(os.path.join(env.w, "submissions.tsv"))[plan_id]
    assert cur["event"] == "SUBMITTED" and cur["status"] == ""
    assert run(end, done) == 0
    assert env.ledger_rows()[-1]["status"] == "VALID"


def test_reconcile_submit_by_batch_tsv_and_not_submitted(env, stub_runner):
    plan_id, _snap = _unknown_plan(env, stub_runner)
    rec = ["reconcile-submit", "--workdir", env.w, "--plan-id", plan_id, "--login-host", "fe",
           "--ssh-timeout-s", "60", "--reason", "checked"]
    jd = os.path.join(env.w, "runs", "J91-r0")
    os.makedirs(jd)
    with open(os.path.join(jd, "batch.tsv"), "w") as f:
        f.write("plan_id\t%s\njob_id\t91\n" % plan_id)
    gone = stub_runner({"scontrol": (1, "", "Invalid job id specified"), "squeue": (0, "", "")})
    assert run(rec + ["--not-submitted"], gone) == 3  # a batch.tsv names the plan
    assert run(rec + ["--job-id", "91"], gone) == 0
    assert env.ledger_rows()[-1]["job_id"] == "91"


def test_reconcile_not_submitted(env, stub_runner):
    plan_id, snap = _unknown_plan(env, stub_runner)
    rec = ["reconcile-submit", "--workdir", env.w, "--plan-id", plan_id, "--login-host", "fe",
           "--ssh-timeout-s", "60", "--reason", "checked squeue and sacct", "--not-submitted"]
    assert run(rec, stub_runner({"squeue": (0, "5 %s/jobs/capkit_job.sbatch\n" % snap, "")})) == 3
    assert run(rec, stub_runner({"squeue": (None, "", "timeout")})) == 3
    assert run(rec, stub_runner({"squeue": (0, "5 /other/capkit_job.sbatch\n", "")})) == 0
    last = env.ledger_rows()[-1]
    assert last["event"] == "ENDED" and last["status"] == "SUBMIT_FAILED"


def test_unregistered_hw_profile_is_submittable(env, stub_runner):
    r = stub_runner({"sbatch": (0, "31\n", "")})
    assert run(env.submit_args(**{"--registered-hw-sha": "UNREGISTERED"}), r) == 0
    rows = env.ledger_rows()
    snap = rows[0]["snapshot_dir"]
    assert "CAPJOBS_REGISTERED_HW_SHA=UNREGISTERED" in open(os.path.join(snap, "job.env")).read()
    # a profile-capture job ends ENV_INVALID by design: its rows say so
    assert rows[0]["reason"] == "purpose=hw_profile_capture"
    done = stub_runner({"squeue": (0, "", ""), "sacct": (0, "31|FAILED|6:0\n", "")})
    assert run(["record-end", "--workdir", env.w, "--plan-id", rows[0]["plan_id"],
                "--login-host", "fe", "--ssh-timeout-s", "60", "--status", "ENV_INVALID",
                "--reason", "gate: hardware profile unregistered"], done) == 0
    cur = ledger.submissions_current(os.path.join(env.w, "submissions.tsv"))[rows[0]["plan_id"]]
    assert cur["status"] == "ENV_INVALID" and cur["reason"].startswith(
        "purpose=hw_profile_capture; ")


def test_zero_gpus_warns(env, stub_runner, capsys):
    reg0 = env.register("gpu0.json", dict(REG_ENV, gpus=0))
    r = stub_runner({"sbatch": (0, "32\n", "")})
    assert run(env.submit_args(**{"--gpus": "0", "--register-sha": reg0,
                                  "--launch-plan": env.plan_for("p0.json", reg0)}), r) == 0
    assert "GPU-IDLE check will be NOT_RUN" in capsys.readouterr().err


def _forbidden_cases(env):
    weights = os.path.join(env.root, "tokw")
    os.makedirs(weights)
    open(os.path.join(weights, "tokenizer.json"), "w").close()
    open(os.path.join(weights, "model-00001.safetensors"), "w").close()
    linked = os.path.join(env.root, "tokl")
    os.makedirs(linked)
    os.symlink(os.path.join(env.tok, "tokenizer.json"), os.path.join(linked, "tokenizer.json"))
    bad_plan = dict(env.plan, launches=[dict(env.plan["launches"][0],
                                             argv=["{PY}", "--arm", "x"])])
    other_arms = dict(env.plan, launches=[dict(env.plan["launches"][0], arm="other")]
                      + env.plan["launches"][1:])
    bad_manifest = json.loads(json.dumps(env.manifest))
    bad_manifest["capkit_packages"][0]["sha256"] = "0" * 64
    other_head = dict(env.manifest, capkit_code_sha="b" * 40)
    np_changed = dict(env.manifest, non_python_changed=["cpp/x.cpp"])
    mount = "%s:%s" % (env.root, env.root)
    def exp(e):  # plan and --experiment agree, so only the no-model allowlist can refuse
        return {"--experiment": e, "--launch-plan": env.write_json(
            "plan-%s.json" % e, dict(env.plan, experiment=e))}
    unknown_reg = "f" * 64
    no_env = env.register("no-env.json", None)
    env_extra = env.register("env-extra.json", dict(REG_ENV, cpu=1))
    env_frac = env.register("env-frac.json", dict(REG_ENV, idle_busy_max_frac=0))
    env_chassis = env.register("env-chassis.json", dict(REG_ENV, chassis_source="guess"))
    lib_link = os.path.join(env.root, "lib-link")
    os.symlink("/home/scratch.trt_llm_data_ci/llm-models", lib_link)
    return {
        "experiment_E10": exp("E10"),
        "experiment_E10_proxy": exp("E10-proxy"),
        "experiment_E13_main": exp("E13-main"),
        "experiment_E11": exp("E11"),
        "gpus_2": {"--gpus": "2"},
        "mount_model_lib": {"--mount": [mount, "/home/scratch.trt_llm_data_ci:"
                                               "/home/scratch.trt_llm_data_ci:ro"]},
        "mount_model_lib_2": {"--mount": [mount, "/home/scratch.trt_llm_data:/data:ro"]},
        "mount_ancestor": {"--mount": [mount, "/home:/home"]},
        "mount_child": {"--mount": [mount, "/home/scratch.trt_llm_data_ci/llm-models:/m"]},
        "tokenizer_weights": {"--tokenizer-dir": weights},
        "tokenizer_symlink": {"--tokenizer-dir": linked},
        "basis_missing_line": {"--basis": "approved by nobody"},
        "basis_partial_line": {"--basis": BASIS[:-3]},
        "basis_empty": {"--basis": " "},
        "register_unknown": {"--register-sha": unknown_reg, "--launch-plan": env.write_json(
            "plan-reg.json", dict(env.plan, register_sha=unknown_reg))},
        "plan_reserved_flag": {"--launch-plan": env.write_json("p1.json", bad_plan),
                               "--arms": "nullsse-a", "--launches": "1"},
        "arms_mismatch": {"--arms": "sfe-a,nullsse-a"},
        "arms_vs_plan": {"--launch-plan": env.write_json("p2.json", other_arms)},
        "launches_mismatch": {"--launches": "3"},
        "time_too_short": {"--time": "00:20:00"},
        "mount_not_covering": {"--mount": "%s:%s" % (env.w, env.w)},
        "mount_not_identity": {"--mount": "%s:/x" % env.root},
        "mount_readonly_for_writes": {"--mount": "%s:%s:ro" % (env.root, env.root)},
        "overlay_sha_mismatch": {"--overlay-manifest": env.write_json("m1.json",
                                                                       bad_manifest)},
        "overlay_head_mismatch": {"--overlay-manifest": env.write_json("m2.json", other_head)},
        "overlay_non_python": {"--overlay-manifest": env.write_json("m3.json", np_changed)},
        "image_size": {"--image-size": "999"},
        "image_sha_format": {"--image-sha256": "xyz"},
        "gate_module": {"--gate-module": "capenv.gate; rm"},
        "relative_path_chars": {"--run-root": env.root + "/runs dir"},
        "exclude_bad": {"--exclude": "a b"},
        "mount_symlink_to_lib": {"--mount": [mount, "%s:/data:ro" % lib_link]},
        "register_no_environment": {"--register-sha": no_env,
                                    "--launch-plan": env.plan_for("pn.json", no_env)},
        "register_env_extra_key": {"--register-sha": env_extra,
                                   "--launch-plan": env.plan_for("pe.json", env_extra)},
        "register_env_bad_frac": {"--register-sha": env_frac,
                                  "--launch-plan": env.plan_for("pf.json", env_frac)},
        "register_env_bad_chassis": {"--register-sha": env_chassis,
                                     "--launch-plan": env.plan_for("pc.json", env_chassis)},
        "gpus_vs_register": {"--gpus": "0"},
        "image_sha_wrong": {"--image-sha256": "d" * 64},
        "hw_sha_format": {"--registered-hw-sha": "abc"},
        "nvidia_smi_from_path": {"--nvidia-smi": "nvidia-smi"},
        "kill_grace_below_worst_stop": {"--launch-plan": env.write_json(
            "pk.json", dict(env.plan, kill_grace_s=51))},
        "scenario_not_in_snapshot": {"--include": False},
    }


EXPECTED_REASON = {
    "experiment_E10": "no-model list", "experiment_E10_proxy": "no-model list",
    "experiment_E13_main": "no-model list", "experiment_E11": "no-model list",
    "gpus_2": "gpus must be", "mount_model_lib": "model library",
    "mount_model_lib_2": "model library", "mount_ancestor": "model library",
    "mount_child": "model library", "tokenizer_weights": "weight file",
    "tokenizer_symlink": "symlink", "basis_missing_line": "whole line",
    "basis_partial_line": "whole line", "basis_empty": "non-empty line",
    "register_unknown": "REGISTRY.tsv", "overlay_sha_mismatch": "differs from the overlay",
    "mount_symlink_to_lib": "model library", "register_no_environment": "'environment'",
    "register_env_extra_key": "'environment'", "register_env_bad_frac": "idle_busy_max_frac",
    "register_env_bad_chassis": "chassis_source", "gpus_vs_register": "environment.gpus",
    "image_sha_wrong": "!= --image-sha256", "hw_sha_format": "UNREGISTERED",
    "nvidia_smi_from_path": "--nvidia-smi must be an absolute path",
    "kill_grace_below_worst_stop": "worst-case orchestrator stop 52.0",
    "scenario_not_in_snapshot": "not in the job snapshot",
}


def test_forbidden_inputs_refused(env, stub_runner, capsys):
    cases = _forbidden_cases(env)
    assert len(cases) >= 25
    for name, over in cases.items():
        r = stub_runner({"sbatch": (0, "1\n", "")})
        rc = run(env.submit_args(**over), r)
        err = capsys.readouterr().err
        assert rc == 3, name
        assert EXPECTED_REASON.get(name, "") in err, (name, err)
        assert r.calls == [], name
        assert not os.path.exists(os.path.join(env.w, "submissions.tsv")), name
        assert not os.path.exists(os.path.join(env.w, "snapshots")), name
    # positive control: the same environment accepts the unmodified arguments
    r = stub_runner({"sbatch": (0, "1\n", "")})
    assert run(env.submit_args(), r) == 0


def test_dirty_tree_refused_only_for_real_submit(env, stub_runner, capsys):
    dirty = lambda _r, _p: (HEAD, [" M tools/capcommon/policy.py"])  # noqa: E731
    r = stub_runner({"sbatch": (0, "1\n", "")})
    assert run(env.submit_args(**{"--dry-run": None}), r, dirty) == 0
    assert json.loads(capsys.readouterr().out)["would_refuse_dirty"] is True
    assert run(env.submit_args(), r, dirty) == 3 and r.calls == []


def test_missing_required_args(env):
    argv = env.submit_args(**{"--exclude": False})
    with pytest.raises(SystemExit) as e:
        run(argv)
    assert e.value.code == 2
    for flag in ("--account", "--qos", "--mem", "--time", "--cpus-per-task", "--partition"):
        with pytest.raises(SystemExit):
            run(env.submit_args(**{flag: False}))


def test_exclude_subcommand_and_file(env, stub_runner, capsys):
    out = os.path.join(env.root, "exclude.json")
    assert run(["exclude", "--cdb-file", external_input(SAVED_CDB), "--main-profile-spec",
                os.path.join(DATA, "g0_main_profile.json"), "--partition", PARTS,
                "--out", out]) == 0
    doc = json.load(open(out))
    assert len(doc["exclude"]) == 12 and doc["source"]["cdb_file"] == SAVED_CDB
    capsys.readouterr()
    r = stub_runner({"sbatch": (0, "9\n", "")})
    assert run(env.submit_args(**{"--exclude": False, "--exclude-file": out}), r) == 0
    assert "--exclude=" + ",".join(doc["exclude"]) in r.calls[0][-1]
    assert env.ledger_rows()[0]["exclude"] == ",".join(doc["exclude"])
    # partitions of the file must equal --partition
    r = stub_runner({"sbatch": (0, "9\n", "")})
    assert run(env.submit_args(**{"--exclude": False, "--exclude-file": out,
                                  "--partition": PARTS.split(",")[0]}), r) == 3


def test_probe(env, stub_runner, capsys):
    log = os.path.join(env.root, "probe.jsonl")
    base = ["probe", "--account", "acct", "--partition", PARTS, "--qos", "batch-short",
            "--exclude", "none", "--time", "00:30:00", "--mem", "1900G", "--cpus-per-task",
            "224", "--gpus", "0", "--job-name", "capkit-probe", "--login-host", "fe",
            "--ssh-timeout-s", "60", "--probe-log", log]
    r = stub_runner({})
    assert run(base + ["--dry-run"], r) == 0 and r.calls == [] and not os.path.exists(log)
    capsys.readouterr()
    r = stub_runner({"sbatch": (0, "", "sbatch: Job 1 to start at 2026-11-13T20:00:00\n")})
    assert run(base, r) == 0
    cmd = r.calls[0][-1]
    assert cmd.startswith("sbatch --test-only ") and "--wrap=true" in cmd
    lines = [json.loads(x) for x in open(log)]
    assert [x["phase"] for x in lines] == ["start", "result"]
    assert "Job 1 to start" in lines[1]["stderr"]
    r = stub_runner({"sbatch": (1, "", "error")})
    assert run(base, r) == 4


def test_local_record(env, capsys):
    base = ["local-record", "--workdir", env.w, "--experiment", "E0", "--register-sha",
            env.reg_sha, "--arms", "nullsse-a", "--launches", "1", "--basis", BASIS,
            "--no-tokenizer"]
    assert run(base + ["--job-id", "local-smoke1"]) == 0
    rows = env.ledger_rows()
    assert [x["event"] for x in rows] == ["PLANNED", "SUBMITTED"]
    assert rows[0]["gpus"] == "0" and rows[0]["sbatch_argv"] == ""
    assert rows[0]["tokenizer_dir_sha256"] == "none" and rows[1]["job_id"] == "local-smoke1"
    assert run(base + ["--job-id", "local-smoke1"]) == 3  # same job id twice
    assert run(base + ["--job-id", "12345"]) == 3
    assert run(base[:-1] + ["--tokenizer-dir", env.tok, "--job-id", "local-x",
                            "--mount", "/home:/home"]) == 3
    assert run([x if x != "E0" else "E11" for x in base] + ["--job-id", "local-y"]) == 3


def test_record_end(env, stub_runner, capsys):
    r = stub_runner({"sbatch": (0, "77\n", "")})
    assert run(env.submit_args(), r) == 0
    plan_id = env.ledger_rows()[0]["plan_id"]
    common = ["record-end", "--workdir", env.w, "--plan-id", plan_id, "--login-host", "fe",
              "--ssh-timeout-s", "60", "--reason", "judged"]
    running = stub_runner({"squeue": (0, "77 RUNNING\n", ""), "sacct": (0, "77|RUNNING|0:0", "")})
    assert run(common + ["--status", "VALID"], running) == 3
    broken = stub_runner({"squeue": (1, "", "slurm down"), "sacct": (0, "77|COMPLETED|0:0", "")})
    assert run(common + ["--status", "VALID"], broken) == 3
    done = stub_runner({"squeue": (0, "", ""), "sacct": (0, "77|COMPLETED|0:0\n", "")})
    assert run(common + ["--status", "BOGUS"], done) == 3
    assert run(common + ["--status", "VALID"], done) == 0
    rows = env.ledger_rows()
    assert rows[-1]["event"] == "ENDED" and rows[-1]["status"] == "VALID"
    assert "sacct state COMPLETED" in rows[-1]["reason"]
    assert run(common + ["--status", "VALID"], done) == 3  # only one ENDED row


def test_jobstate_and_slurm_view_cmds(env, stub_runner, capsys):
    jd = os.path.join(env.root, "J5-r0")
    os.makedirs(jd)
    base = ["--login-host", "fe", "--ssh-timeout-s", "60", "--job-id", "5"]
    assert run(["jobstate"] + base, stub_runner(
        {"squeue": (0, "", ""), "sacct": (0, "5|FAILED|1:0\n", "")})) == 0
    assert run(["jobstate"] + base, stub_runner(
        {"squeue": (0, "5 PENDING", ""), "sacct": (0, "", "")})) == 20
    assert run(["jobstate"] + base, stub_runner(
        {"squeue": (None, "", "timeout"), "sacct": (0, "", "")})) == 21
    r = stub_runner({"scontrol": (0, "JobId=5 NodeList=n1\n", ""), "squeue": (0, "5 u R\n", "")})
    assert run(["slurm-view", "--job-dir", jd] + base, r) == 0
    assert json.load(open(os.path.join(jd, "env", "slurm_view.json")))["node"] == "n1"


MODEL_DIR = "/home/scratch.trt_llm_data_ci/llm-models/GLM-5.3-Flash"


def prep_args(env, *extra, partition=None):
    a = ["prep-tokenizer", "--account", "acct", "--qos", "batch", "--exclude", "none",
         "--time", "00:30:00", "--mem", "4G", "--cpus-per-task", "2", "--job-name", "capkit-tok",
         "--login-host", "fe", "--ssh-timeout-s", "60", "--workdir", env.w,
         "--code-root", env.root, "--snapshot-root", os.path.join(env.root, "snaps"),
         "--model-dir", MODEL_DIR, "--name", "glm53", "--image", env.image, "--basis", BASIS]
    if partition:
        a += ["--partition", partition]
    return a + list(extra)


def _prep_code_root(env):
    os.makedirs(os.path.join(env.root, "jobs"), exist_ok=True)
    with open(os.path.join(CODE_ROOT, "jobs", "prep_tokenizer.sbatch"), "rb") as f, \
            open(os.path.join(env.root, "jobs", "prep_tokenizer.sbatch"), "wb") as g:
        g.write(f.read())


def test_prep_tokenizer_submits_a_minimal_ro_job(env, stub_runner, capsys):
    """R1b: the formal tokenizer-copy entry; no GPU, no --exclusive, no partition unless given;
    the registered model dir read-only is its only model-library mount."""
    _prep_code_root(env)
    r = stub_runner({"sbatch": (0, "777\n", "")})
    assert run(prep_args(env), r) == 0, capsys.readouterr().err
    out = json.loads(capsys.readouterr().out)
    ra = out["remote_argv"]
    assert ra[:2] == ["sbatch", "--parsable"] and ra[-1] == out["plan_dir"]
    assert ra[-2] == os.path.join(out["plan_dir"], "jobs", "prep_tokenizer.sbatch")
    assert not [x for x in ra if x.startswith(("--gres", "--gpus", "--exclusive", "--partition"))]
    assert "--nodes=1" in ra and "--ntasks=1" in ra and "--cpus-per-task=2" in ra
    assert out["mounts"][0] == "%s:%s:ro" % (MODEL_DIR, MODEL_DIR)
    assert out["dest"] == os.path.join(env.root, "configs", "tokenizers", "glm53")
    rows = ledger.read_tsv(os.path.join(env.w, "submissions.tsv"), ledger.SUBMISSION_COLUMNS)
    assert [x["event"] for x in rows] == ["PLANNED", "SUBMITTED"]
    pl = rows[0]
    assert pl["reason"] == "purpose=prep_tokenizer" and pl["experiment"] == "PREP-TOKENIZER"
    assert pl["gpus"] == "0" and pl["exclusive"] == "false" and pl["partition"] == ""
    assert json.loads(pl["mounts_json"]) == out["mounts"] and pl["tokenizer_dir_sha256"] == "none"
    assert rows[1]["job_id"] == "777"
    plan = out["plan_dir"]
    for rel in ("jobs/prep_tokenizer.sbatch", "job.env", "mounts.json"):
        assert not os.access(os.path.join(plan, rel), os.W_OK), rel
    sh = subprocess.run(["bash", "-c", 'source "$1/job.env"; printf "%s|%s|%s|%s" '
                         '"$CAPJOBS_PREP_MODEL_DIR" "$CAPJOBS_PREP_MAX_FILE_BYTES" '
                         '"$CAPJOBS_PREP_WEIGHT_SUFFIXES" "$CAPJOBS_RUN_ROOT"', "x", plan],
                        capture_output=True, text=True, check=True)
    assert sh.stdout == "%s|%d|%s|%s" % (MODEL_DIR, policy.PREP_TOKENIZER_MAX_FILE_BYTES,
                                         " ".join(policy.WEIGHT_SUFFIXES), plan)
    # RS1: the batch step reads every value from the job.env written here (fake srun records)
    bindir = os.path.join(env.root, "bin")
    os.makedirs(bindir)
    rec = os.path.join(env.root, "srun_calls.txt")
    with open(os.path.join(bindir, "srun"), "w") as f:
        f.write('#!/bin/bash\nprintf "%%s\\n" "$*" >> "%s"\nexit 0\n' % rec)
    os.chmod(os.path.join(bindir, "srun"), 0o755)
    b = subprocess.run(["bash", ra[-2], plan], capture_output=True, text=True,
                       env=dict(os.environ, PATH=bindir + ":" + os.environ["PATH"],
                                SLURM_JOB_ID="777"))
    assert b.returncode == 0, b.stdout + b.stderr
    call = open(rec).read()
    assert "--container-mounts=%s" % ",".join(out["mounts"]) in call
    assert "--container-image=%s" % env.image in call and " inner %s " % plan in call
    # a partition is passed through only when given
    assert run(prep_args(env, "--dry-run", partition="cpu_a,cpu_b"), r) == 0
    ra = json.loads(capsys.readouterr().out)["remote_argv"]
    assert "--partition=cpu_a,cpu_b" in ra


@pytest.mark.parametrize("change,msg", [
    ({"--model-dir": "/home/scratch.trt_llm_data_ci/llm-models/Other"}, "not a registered"),
    ({"--model-dir": "/home/scratch.trt_llm_data_ci/llm-models"}, "not a registered"),
    ({"--name": "../x"}, "bad tokenizer --name"),
    ({"--basis": "not in STATUS.md"}, "--basis"),
])
def test_prep_tokenizer_refusals(env, stub_runner, capsys, change, msg):
    _prep_code_root(env)
    a = prep_args(env)
    for k, v in change.items():
        a[a.index(k) + 1] = v
    r = stub_runner({})
    assert run(a, r) == 3 and r.calls == []
    assert msg in capsys.readouterr().err
    assert not os.path.exists(os.path.join(env.w, "submissions.tsv"))


def test_prep_tokenizer_destination_must_be_new(env, stub_runner, capsys):
    _prep_code_root(env)
    os.makedirs(os.path.join(env.root, "configs", "tokenizers", "glm53"))
    r = stub_runner({})
    assert run(prep_args(env), r) == 3 and r.calls == []
    assert "already exists" in capsys.readouterr().err


def test_model_library_mounts_only_for_prep_tokenizer(env):
    """The ledger refuses a model-library mount on every other plan, and any prep plan that is
    not exactly the registered dir read-only without a GPU."""
    ok = ["%s:%s:ro" % (MODEL_DIR, MODEL_DIR), "/x/configs/tokenizers:/x/configs/tokenizers:rw"]
    path = os.path.join(env.w, "submissions.tsv")
    good = submit.ledger_row(
        plan_id="P1", event="PLANNED", reason="purpose=prep_tokenizer",
        experiment="PREP-TOKENIZER", basis=BASIS, gpus="0", mounts_json=json.dumps(ok),
        tokenizer_dir_sha256="none",
        assertions_json=json.dumps({k: True for k in policy.PREP_ASSERTION_KEYS}))
    ledger.submissions_append(path, good)
    bad = [
        dict(good, reason="purpose=measurement"),  # a measurement plan with the model mount
        dict(good, experiment="E0"),
        dict(good, gpus="1"),
        dict(good, mounts_json=json.dumps(["%s:%s:rw" % (MODEL_DIR, MODEL_DIR)])),
        dict(good, mounts_json=json.dumps(["%s:%s" % (MODEL_DIR, MODEL_DIR)])),  # rw default
        dict(good, mounts_json=json.dumps(["%s:/mnt/m:ro" % MODEL_DIR])),
        dict(good, mounts_json=json.dumps(ok + ["/home/scratch.trt_llm_data:/d:ro"])),
        dict(good, mounts_json=json.dumps(["/home/scratch.trt_llm_data_ci:/home/"
                                           "scratch.trt_llm_data_ci:ro"])),
        dict(good, mounts_json=json.dumps(["/home:/home:ro"])),
        dict(good, assertions_json=json.dumps({k: True for k in policy.ASSERTION_KEYS})),
    ]
    for row in bad:
        with pytest.raises(ledger.LedgerError):
            ledger.submissions_append(path, row)
    with pytest.raises(policy.PolicyError):
        policy.check_mounts(ok)  # every other job: the model dir mount is refused


def test_exclude_live_raw_text(env, stub_runner, capsys):
    """K26: cdb output without a final newline keeps the exit marker on its own line; an ssh
    failure is recorded as such (not 'exit: None') and refused."""
    with open(external_input(SAVED_CDB)) as f:
        csv_text = "".join(ln for ln in f if not ln.startswith("#")).rstrip("\n")
    base = ["exclude", "--live", "--cdb-query", "q", "--login-host", "fe", "--ssh-timeout-s",
            "60", "--main-profile-spec", os.path.join(DATA, "g0_main_profile.json"),
            "--partition", PARTS]
    raw, out = os.path.join(env.root, "raw1.txt"), os.path.join(env.root, "ex1.json")
    assert run(base + ["--raw-out", raw, "--out", out],
               stub_runner({"cdb": (0, csv_text, "")})) == 0
    lines = open(raw).read().splitlines()
    assert lines[-1] == "# exit: 0" and not lines[-2].endswith("# exit: 0")
    assert len(json.load(open(out))["exclude"]) == 12
    raw2 = os.path.join(env.root, "raw2.txt")
    assert run(base + ["--raw-out", raw2, "--out", os.path.join(env.root, "ex2.json")],
               stub_runner({"cdb": (None, "", "timeout")})) == 3
    assert open(raw2).read().splitlines()[-1].startswith("# exit: none (ssh")

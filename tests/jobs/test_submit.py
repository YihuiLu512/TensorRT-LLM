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
from capjobs import launchplan, snapshot, submit

HEAD = "c" * 40
REG_ENV = {"gpus": 1, "idle_window_s": 30, "idle_busy_max_frac": 0.02, "slurm_view_wait_s": 300,
           "chassis_source": "slurm_view"}
BASIS = "2026-11-13 18:00 user approved E0 smoke (DECISIONS 1, item 7) -> submit E0 job"
PARTS = ("b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,"
         "b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb")
SEC = "%s:%s" % ((submit.REQUIRED_IDENTITY_MOUNTS[0],) * 2)  # the secondary scratch disk


class Env:
    def __init__(self, root):
        self.root = str(root)
        self.w = os.path.join(self.root, "W")
        os.makedirs(self.w)
        with open(os.path.join(self.w, "STATUS.md"), "w") as f:
            f.write("# STATUS\n\n- earlier line\n%s\n" % BASIS)
        # the registered scenario: in <W>, listed in register.json files.scenario
        self.scen_rel = "02_e0/E0/register/e0a-nullsse.scenario.json"
        os.makedirs(os.path.join(self.w, os.path.dirname(self.scen_rel)))
        with open(os.path.join(CODE_ROOT, "configs", "e0", "e0a-nullsse.scenario.json"),
                  "rb") as f, open(os.path.join(self.w, self.scen_rel), "wb") as g:
            g.write(f.read())
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
                                   "argv": ["{PY}", "-m", "caplaunch", "--workdir", self.w,
                                            "--register", "02_e0/E0/register/register.json",
                                            "--scenario", self.scen_rel,
                                            "--shm-dir", "{SHM_ROOT}/snap",
                                            "--snapshot-interval-ms", "50",
                                            "--origin-lead-ms", "6000", "--housekeeping-ms", "50",
                                            "--kill-after-s", "2", "--capkit-sha", HEAD,
                                            "--probe-tree-sha", "none", "--baseline-sha", "none",
                                            "--snapshot-dir", "{SNAP}",
                                            "--image-path", "/img.sqsh", "--image-sha256", "none",
                                            "--hw-profile-sha", "none"]}
                                  for i, arm in enumerate(["nullsse-a", "sfe-a"])]}
        self.plan_path = self.write_json("plan.json", self.plan)

    def register(self, name, environment, experiment="E0"):
        """A real register.json under <W> with an "environment" section, entered in REGISTRY."""
        rel = "02_e0/E0/register/" + name
        doc = {"schema": "capkit.register/1", "experiment": experiment, "arms": ["nullsse-a"],
               "files": {"scenario": {"relpath": self.scen_rel, "sha256": sha256_file(
                   os.path.join(self.w, self.scen_rel))}}}
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
             "--mount": ["%s:%s" % (self.root, self.root), SEC], "--tokenizer-dir": self.tok,
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
    assert sh.stdout == "%s|%s:%s,%s|3600" % (snap, env.root, env.root, SEC)
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
    assert out["time_budget"]["total_s"] == 2 * (600 + 60 + 60 + 120) + 30 + 300
    assert out["time_budget"]["parts"]["pin_tables_s"] == 2 * launchplan.PINS_TIMEOUT_S


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
    other_scen = env.scen_rel.replace("e0a-nullsse", "copy")
    with open(os.path.join(env.w, env.scen_rel), "rb") as f, \
            open(os.path.join(env.w, other_scen), "wb") as g:
        g.write(f.read())
    def plan_argv(**kw):  # the plan with each launch's argv changed by kw (flag -> value)
        lns = []
        for ln in env.plan["launches"]:
            argv = list(ln["argv"])
            for k, v in kw.items():
                argv[argv.index(k) + 1] = v
            lns.append(dict(ln, argv=argv))
        return dict(env.plan, launches=lns)
    other_w = os.path.join(env.root, "W2")
    os.makedirs(os.path.dirname(os.path.join(other_w, env.scen_rel)))
    with open(os.path.join(env.w, env.scen_rel), "rb") as f, \
            open(os.path.join(other_w, env.scen_rel), "wb") as g:
        g.write(f.read())
    lib_link = os.path.join(env.root, "lib-link")
    os.symlink("/home/scratch.trt_llm_data_ci/llm-models", lib_link)
    return {
        "experiment_E10": exp("E10"),
        "experiment_E10_proxy": exp("E10-proxy"),
        "experiment_E13_main": exp("E13-main"),
        "experiment_E11": exp("E11"),
        "gpus_2": {"--gpus": "2"},
        "mount_model_lib": {"--mount": [mount, SEC, "/home/scratch.trt_llm_data_ci:"
                                               "/home/scratch.trt_llm_data_ci:ro"]},
        "mount_model_lib_2": {"--mount": [mount, SEC, "/home/scratch.trt_llm_data:/data:ro"]},
        "mount_ancestor": {"--mount": [mount, SEC, "/home:/home"]},
        "mount_child": {"--mount": [mount, SEC,
                                    "/home/scratch.trt_llm_data_ci/llm-models:/m"]},
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
        "mount_not_covering": {"--mount": ["%s:%s" % (env.w, env.w), SEC]},
        "mount_not_identity": {"--mount": ["%s:/x" % env.root, SEC]},
        "mount_readonly_for_writes": {"--mount": ["%s:%s:ro" % (env.root, env.root), SEC]},
        "mount_no_secondary_disk": {"--mount": mount},
        "mount_secondary_not_identity": {"--mount": [mount, "%s:/sec" % SEC.split(":")[0]]},
        "overlay_sha_mismatch": {"--overlay-manifest": env.write_json("m1.json",
                                                                       bad_manifest)},
        "overlay_head_mismatch": {"--overlay-manifest": env.write_json("m2.json", other_head)},
        "overlay_non_python": {"--overlay-manifest": env.write_json("m3.json", np_changed)},
        "image_size": {"--image-size": "999"},
        "image_sha_format": {"--image-sha256": "xyz"},
        "gate_module": {"--gate-module": "capenv.gate; rm"},
        "relative_path_chars": {"--run-root": env.root + "/runs dir"},
        "exclude_bad": {"--exclude": "a b"},
        "mount_symlink_to_lib": {"--mount": [mount, SEC, "%s:/data:ro" % lib_link]},
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
        "scenario_not_registered": {"--launch-plan": env.write_json(
            "ps.json", plan_argv(**{"--scenario": other_scen}))},
        "scenario_snap_path": {"--launch-plan": env.write_json(
            "pq.json", plan_argv(**{"--scenario": "{SNAP}/configs/e0/e0a-nullsse.scenario.json"}))},
        "scenario_other_workdir": {"--launch-plan": env.write_json(
            "pw.json", plan_argv(**{"--workdir": other_w}))},
        "capkit_sha_not_head": {"--launch-plan": env.write_json(
            "pc2.json", plan_argv(**{"--capkit-sha": "b" * 40}))},
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
    "mount_not_covering": "is not under an identity rw mount",
    "mount_not_identity": "is not under an identity rw mount",
    "mount_readonly_for_writes": "is not under an identity rw mount",
    "mount_no_secondary_disk": "secondary scratch disk",
    "mount_secondary_not_identity": "secondary scratch disk",
    "scenario_not_registered": "is not the register's files.scenario",
    "scenario_snap_path": "must be literal paths",
    "scenario_other_workdir": "is not --workdir",
    "capkit_sha_not_head": "--capkit-sha must be the code HEAD",
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


def _edit_argv(argv, drop=(), rename=None, add=()):
    """argv without each flag in drop (and its value), flag rename {old: new}, add appended."""
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
        elif x in drop:
            skip = True
        else:
            out.append((rename or {}).get(x, x))
    return out + list(add)


@pytest.mark.parametrize("edit,reason", [
    ({"drop": ["--housekeeping-ms"]},
     "the following arguments are required: --housekeeping-ms"),
    ({"rename": {"--housekeeping-ms": "--housekeeping"}}, "--housekeeping-ms"),
    ({"add": ["--par", "a=1"]}, "unrecognized arguments: --par a=1"),
    ({"add": ["--bogus", "1"]}, "unrecognized arguments: --bogus 1"),
    ({"add": ["--snapshot-interval-ms", "{JOB_DIR}"]}, "invalid int value"),
    ({"add": ["-h"]}, "exited while parsing"),
])
def test_plan_argv_is_parsed_with_caplaunchs_parser(env, stub_runner, capsys, edit, reason):
    """RS22/RS15: each launch's argv, with placeholders replaced and the arguments
    capjobs.launches appends, goes through caplaunch's own parser (build_parser) at submit: a
    missing required flag, an abbreviation or an unknown flag is refused before anything is
    written or submitted, instead of caplaunch exiting 2 in the job."""
    plan = dict(env.plan, launches=[dict(ln, argv=_edit_argv(ln["argv"], **edit))
                                    for ln in env.plan["launches"]])
    r = stub_runner({"sbatch": (0, "1\n", "")})
    assert run(env.submit_args(**{"--launch-plan": env.write_json("pa.json", plan)}), r) == 3
    out = capsys.readouterr()
    assert "launch 0: caplaunch would refuse its argv" in out.err and reason in out.err, out.err
    assert "usage:" not in out.out and "usage:" not in out.err
    assert r.calls == [] and not os.path.exists(os.path.join(env.w, "submissions.tsv"))
    assert not os.path.exists(os.path.join(env.w, "snapshots"))


def test_plan_argv_check_parses_what_the_job_runs(monkeypatch, tmp_path):
    """The pre-parse covers the argv capjobs.launches builds: every placeholder substituted,
    the appended identity/--run-dir/--pin-table/... arguments present (launches.launch_argv),
    with one --job-input per kind the launch's scenario declares."""
    scen = {"tier_table": "t.json", "schedule": "s.capbin", "prompts": "p.json",
            "params_required": [],
            "job_inputs": {"fingerprint": {"args": []}, "prebaked": {"args": []}}}
    (tmp_path / "s.json").write_text(json.dumps(scen))
    seen = []
    from caplaunch import __main__ as caplaunch_main
    real = caplaunch_main.build_parser

    def spy():
        p = real()
        orig = p.parse_args
        p.parse_args = lambda args: seen.append(list(args)) or orig(args)
        return p
    monkeypatch.setattr(caplaunch_main, "build_parser", spy)
    argv = ["{PY}", "-m", "caplaunch", "--workdir", str(tmp_path), "--register", "r.json",
            "--scenario", "s.json", "--shm-dir", "{SHM_ROOT}/snap", "--snapshot-interval-ms",
            "50", "--origin-lead-ms", "6000", "--housekeeping-ms", "50", "--kill-after-s", "2",
            "--capkit-sha", HEAD, "--probe-tree-sha", "none", "--baseline-sha", "none",
            "--snapshot-dir", "{SNAP}", "--image-path", "/i", "--image-sha256", "none",
            "--hw-profile-sha", "none"]
    plan = {"experiment": "E0", "register_sha": "b" * 64,
            "launches": [{"launch_seq": 0, "arm": "nullsse-a", "argv": argv}]}
    submit.check_plan_argv(plan)
    assert len(seen) == 1 and not any("{" in x for x in seen[0])
    for flag in launchplan.RESERVED_FLAGS:
        assert flag in seen[0], flag
    assert [seen[0][i + 1].split("=")[0] for i, x in enumerate(seen[0])
            if x == "--job-input"] == ["fingerprint", "prebaked"]
    with pytest.raises(submit.SubmitError, match="must run .* -m caplaunch"):
        submit.check_plan_argv(dict(plan, launches=[dict(plan["launches"][0],
                                                         argv=["{PY}", "x.py"] + argv[3:])]))
    # RS32: a job_inputs section caplaunch would refuse is refused here, before the allocation
    for bad in ({"prebake": {"args": []}}, {}):
        (tmp_path / "s.json").write_text(json.dumps(dict(scen, job_inputs=bad)))
        with pytest.raises(submit.SubmitError, match="launch 0: .*job_inputs"):
            submit.check_plan_argv(plan)
    (tmp_path / "s.json").write_text(json.dumps(dict(scen, prompts=5)))
    with pytest.raises(submit.SubmitError, match="launch 0: .*prompts"):
        submit.check_plan_argv(plan)


def _mount_list(env, *paths, extra=()):
    return ["%s:%s" % (q, q) for q in paths] + list(extra) + [SEC]


def test_workdir_must_be_mounted(env, stub_runner, capsys):
    """<W> is read in the job (register, scenario, tier table, schedule, prompts): mounts that
    cover the run, snapshot, local, shm and tokenizer paths but not <W> are refused at submit
    (errata R5-2); a ro mount of <W> is enough."""
    for d in ("runs", "snapshots"):
        os.makedirs(os.path.join(env.w, d), exist_ok=True)
    each = [os.path.join(env.w, "runs"), os.path.join(env.w, "snapshots"),
            os.path.join(env.root, "local"), os.path.join(env.root, "shm"), env.tok]
    for q in each:
        os.makedirs(q, exist_ok=True)
    r = stub_runner({"sbatch": (0, "1\n", "")})
    assert run(env.submit_args(**{"--mount": _mount_list(env, *each)}), r) == 3
    err = capsys.readouterr().err
    assert "path %s is not under an identity ro/rw mount" % env.w in err, err
    assert r.calls == [] and not os.path.exists(os.path.join(env.w, "submissions.tsv"))
    r = stub_runner({"sbatch": (0, "1\n", "")})
    mounts = _mount_list(env, *each, extra=["%s:%s:ro" % (env.w, env.w)])
    assert run(env.submit_args(**{"--mount": mounts}), r) == 0, capsys.readouterr().err
    job_env = os.path.join(env.ledger_rows()[0]["snapshot_dir"], "job.env")
    sh = subprocess.run(["bash", "-c", 'source "$1"; printf %s "$CAPJOBS_WORKDIR"', "x",
                         job_env], capture_output=True, text=True, check=True)
    assert sh.stdout == env.w


def test_registered_input_symlinked_out_of_the_mounts_is_refused(env, stub_runner, capsys):
    """Each registered file the job reads in <W> is checked by its realpath too: a tier table
    that is a symlink to an unmounted directory is refused even though <W> is mounted."""
    outside = os.path.join(env.root, "outside")
    os.makedirs(outside)
    with open(os.path.join(outside, "tiers.json"), "w") as f:
        f.write("{}")
    tt_rel = "02_e0/E0/register/tiers.json"
    os.symlink(os.path.join(outside, "tiers.json"), os.path.join(env.w, tt_rel))
    rel = "02_e0/E0/register/with-tiers.json"
    doc = {"schema": "capkit.register/1", "experiment": "E0", "arms": ["nullsse-a"],
           "environment": REG_ENV,
           "files": {"scenario": {"relpath": env.scen_rel,
                                  "sha256": sha256_file(os.path.join(env.w, env.scen_rel))},
                     "tier_table": {"relpath": tt_rel,
                                    "sha256": sha256_file(os.path.join(env.w, tt_rel))}}}
    with open(os.path.join(env.w, rel), "w") as f:
        json.dump(doc, f)
    sha = sha256_file(os.path.join(env.w, rel))
    ledger.registry_append(os.path.join(env.w, "REGISTRY.tsv"), "E0", rel, sha, "kind=register;")
    each = [env.w, os.path.join(env.root, "local"), os.path.join(env.root, "shm"), env.tok]
    for q in each:
        os.makedirs(q, exist_ok=True)
    over = {"--register-sha": sha, "--launch-plan": env.plan_for("plan-tiers.json", sha)}
    r = stub_runner({"sbatch": (0, "1\n", "")})
    assert run(env.submit_args(**{"--mount": _mount_list(env, *each)}, **over), r) == 3
    err = capsys.readouterr().err
    assert "path %s is not under an identity ro/rw mount" % os.path.join(
        os.path.realpath(outside), "tiers.json") in err, err
    r = stub_runner({"sbatch": (0, "1\n", "")})
    mounts = _mount_list(env, *each, extra=["%s:%s:ro" % (outside, outside)])
    assert run(env.submit_args(**{"--mount": mounts}, **over), r) == 0, capsys.readouterr().err


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


@pytest.mark.parametrize("jid", ["\u00b2", "\u0661"])
def test_job_ids_are_ascii_digits(env, stub_runner, capsys, jid):
    """Cluster job ids are ASCII digits only: in sbatch --parsable output and in --job-id."""
    assert submit.parse_parsable("5601234\n") == "5601234"
    assert submit.parse_parsable("%s\n" % jid) is None
    plan_id, _snap = _unknown_plan(env, stub_runner)
    capsys.readouterr()
    r = stub_runner({"scontrol": (0, "JobId=5 NodeList=n1\n", "")})
    rec = ["reconcile-submit", "--workdir", env.w, "--plan-id", plan_id, "--login-host", "fe",
           "--ssh-timeout-s", "60", "--reason", "x", "--job-id", jid]
    assert run(rec, r) == 3
    assert "numeric cluster job id" in capsys.readouterr().err and r.calls == []


MODEL_DIR = "/home/scratch.trt_llm_data_ci/llm-models/GLM-5.3-Flash"


def prep_args(env, *extra, partition=None):
    a = ["prep-tokenizer", "--account", "acct", "--qos", "batch", "--exclude", "none",
         "--time", "00:30:00", "--mem", "4G", "--cpus-per-task", "2", "--job-name", "capkit-tok",
         "--login-host", "fe", "--ssh-timeout-s", "60", "--workdir", env.w,
         "--code-root", env.root, "--snapshot-root", os.path.join(env.root, "snaps"),
         "--model-dir", MODEL_DIR, "--name", "glm53", "--image", env.image, "--basis", BASIS,
         "--mount", "%s:%s:rw" % (env.root, env.root), "--mount", SEC]
    if partition:
        a += ["--partition", partition]
    return a + list(extra)


def _git(root, *args):
    return subprocess.run(["git", "-C", root, "-c", "user.name=t", "-c", "user.email=t@t",
                           "-c", "commit.gpgsign=false"] + list(args),
                          capture_output=True, check=True).stdout


def _prep_code_root(env, commit=True):
    """env.root as the code repo: a git repo whose HEAD holds jobs/prep_tokenizer.sbatch."""
    os.makedirs(os.path.join(env.root, "jobs"), exist_ok=True)
    with open(os.path.join(CODE_ROOT, "jobs", "prep_tokenizer.sbatch"), "rb") as f, \
            open(os.path.join(env.root, "jobs", "prep_tokenizer.sbatch"), "wb") as g:
        g.write(f.read())
    with open(os.path.join(env.root, "README"), "w") as f:
        f.write("x\n")
    _git(env.root, "init", "-q")
    _git(env.root, "add", "README", *(["jobs/prep_tokenizer.sbatch"] if commit else []))
    _git(env.root, "commit", "-q", "-m", "c")
    return _git(env.root, "rev-parse", "HEAD").decode().strip()


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


def test_prep_tokenizer_records_the_committed_script(env, stub_runner, capsys):
    """Q8: the plan holds the job script as committed at HEAD; MANIFEST.json names HEAD and the
    sha256 of every plan file, and the PLANNED row's snapshot_sha256 is MANIFEST.json's."""
    head = _prep_code_root(env)
    committed = _git(env.root, "cat-file", "blob", "HEAD:jobs/prep_tokenizer.sbatch")
    r = stub_runner({"sbatch": (0, "778\n", "")})
    assert run(prep_args(env), r) == 0, capsys.readouterr().err
    out = json.loads(capsys.readouterr().out)
    plan = out["plan_dir"]
    pl = env.ledger_rows()[0]
    assert pl["snapshot_sha256"] == sha256_file(os.path.join(plan, "MANIFEST.json"))
    man = json.load(open(os.path.join(plan, "MANIFEST.json")))
    assert man["code_sha"] == head == out["code_head"]
    files = {x["path"]: x for x in man["files"]}
    assert sorted(files) == ["job.env", "jobs/prep_tokenizer.sbatch", "mounts.json"]
    for rel, x in files.items():
        assert sha256_file(os.path.join(plan, rel)) == x["sha256"], rel
    assert files["jobs/prep_tokenizer.sbatch"]["sha256"] == sha256_bytes(committed)
    assert out["job_env"]["CAPJOBS_CODE_SHA"] == head
    assert out["job_env"]["CAPJOBS_PREP_SCRIPT_SHA256"] == sha256_bytes(committed)
    assert not os.access(os.path.join(plan, "MANIFEST.json"), os.W_OK)


def test_prep_tokenizer_library_roots_include_realpaths(env, stub_runner, capsys, monkeypatch,
                                                         tmp_path_factory):
    """Q19 (C-O5): the job's "nothing else of the library is visible" check gets each library's
    realpath too, so a library seen in the container under its resolved path is checked."""
    _prep_code_root(env)
    elsewhere = str(tmp_path_factory.mktemp("libs"))  # not under the mounted env.root
    real = os.path.join(elsewhere, "real-lib")
    os.makedirs(real)
    link = os.path.join(elsewhere, "lib-link")
    os.symlink(real, link)
    monkeypatch.setattr(policy, "FORBIDDEN_MOUNT_PATHS", policy.FORBIDDEN_MOUNT_PATHS + (link,))
    assert run(prep_args(env, "--dry-run"), stub_runner({})) == 0, capsys.readouterr().err
    roots = json.loads(capsys.readouterr().out)["job_env"]["CAPJOBS_PREP_LIBRARY_ROOTS"].split()
    assert link in roots and real in roots


@pytest.mark.parametrize("state", ["modified", "untracked"])
def test_prep_tokenizer_refuses_an_uncommitted_script(env, stub_runner, capsys, state):
    """Q8: a job script that is not at HEAD, or differs from it in the working tree, is refused
    before any write; --dry-run only reports it."""
    _prep_code_root(env, commit=state == "modified")
    if state == "modified":
        with open(os.path.join(env.root, "jobs", "prep_tokenizer.sbatch"), "a") as f:
            f.write("# local edit\n")
    r = stub_runner({})
    assert run(prep_args(env), r) == 3 and r.calls == []
    err = capsys.readouterr().err
    assert ("differs from HEAD" if state == "modified" else "not committed at HEAD") in err, err
    assert not os.path.exists(os.path.join(env.w, "submissions.tsv"))
    assert not os.path.exists(os.path.join(env.root, "snaps"))
    assert run(prep_args(env, "--dry-run"), r) == 0
    assert json.loads(capsys.readouterr().out)["would_refuse_uncommitted"]


@pytest.mark.parametrize("mounts,msg", [
    (["{root}:{root}:rw"], "secondary scratch disk"),
    (["{w}:{w}:rw", SEC], "is not under an identity rw mount"),
    (["{root}:{root}:ro", SEC], "is not under an identity rw mount"),
    (["{root}:{root}:rw", SEC, "/home/scratch.trt_llm_data:/d:ro"], "exactly one model-library"),
    (["{root}:{root}:rw", SEC, MODEL_DIR + ":" + MODEL_DIR + ":ro"], "exactly one model-library"),
])
def test_prep_tokenizer_mounts_go_through_the_job_checks(env, stub_runner, capsys, mounts, msg):
    """Q15: --mount is parsed and checked like `submit`'s (secondary disk, coverage of the
    destination parent and the plan dir); the model dir is added by the entry, never by --mount."""
    _prep_code_root(env)
    a = prep_args(env)
    i = a.index("--mount")
    a = a[:i] + [x for m in mounts for x in ("--mount", m.format(root=env.root, w=env.w))]
    r = stub_runner({})
    assert run(a, r) == 3 and r.calls == []
    assert msg in capsys.readouterr().err
    assert not os.path.exists(os.path.join(env.w, "submissions.tsv"))


def test_tokenizer_dir_refuses_weight_content(tmp_path):
    """RS16 (sweep of Q9): the measurement submit's tokenizer check also reads each file's
    header, so weights under a tokenizer-like name never reach a no-model job."""
    tok = tmp_path / "tok"
    tok.mkdir()
    (tok / "tokenizer.json").write_text('        {"a": 1}\n')  # '{"' at byte 8, not safetensors
    (tok / "empty.txt").write_bytes(b"")
    assert policy.tokenizer_dir_sha256(str(tok)) != "none"
    hdr = b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    for data in (len(hdr).to_bytes(8, "little") + hdr + b"\0" * 4, b"\x80\x02}q\x00.",
                 b"PK\x03\x04rest", b"GGUF\x03\x00\x00\x00"):
        (tok / "weights.json").write_bytes(data)
        with pytest.raises(policy.PolicyError, match="weight content"):
            policy.tokenizer_dir_sha256(str(tok))


def test_policy_refuses_mount_separators(tmp_path, monkeypatch):
    """Q19: pyxis splits --container-mounts at ',' and ':'; the policy layer refuses both on its
    own (the entries' parse_mount also does), and compares library realpaths."""
    sneak = "/a:/a,/home/scratch.trt_llm_data:/lib"
    with pytest.raises(policy.PolicyError):
        policy.check_mounts([sneak])
    with pytest.raises(policy.PolicyError):
        policy.check_mounts([{"src": "/a", "dst": "/b,/home/scratch.trt_llm_data:/lib"}])
    with pytest.raises(policy.PolicyError):
        policy.check_mounts(["/a:/b:ro:x"])
    ok = "%s:%s:ro" % (MODEL_DIR, MODEL_DIR)
    for bad in (["%s,rbind" % ok], [ok, "/x:/x:ro,rbind"], [ok, sneak], [ok, "/x:/x:rx"]):
        with pytest.raises(policy.PolicyError):
            policy.check_prep_tokenizer_mounts(bad)
    policy.check_prep_tokenizer_mounts([ok, "/x:/x:rw"])  # positive control
    real = tmp_path / "real-lib"
    real.mkdir()
    link = tmp_path / "lib-link"
    os.symlink(str(real), str(link))
    monkeypatch.setattr(policy, "FORBIDDEN_MOUNT_PATHS", (str(link),))
    assert policy.library_roots() == sorted([str(link), str(real)])
    with pytest.raises(policy.PolicyError):
        policy.check_mounts(["%s:/m:ro" % real])


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

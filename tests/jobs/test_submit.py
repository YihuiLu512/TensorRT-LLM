# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests of capjobs.submit with a stub runner (no ssh is ever executed)."""

import json
import os
import subprocess

import pytest
from conftest import CODE_ROOT, DATA, SAVED_CDB

from capcommon import ledger
from capcommon.hashing import sha256_bytes, sha256_file
from capjobs import snapshot, submit

HEAD = "c" * 40
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
        self.reg_sha = sha256_bytes(b"register")
        ledger.registry_append(os.path.join(self.w, "REGISTRY.tsv"), "E0",
                               "02_e0/E0/register/register.json", self.reg_sha, "kind=register;")
        self.image = os.path.join(self.root, "img.sqsh")
        with open(self.image, "wb") as f:
            f.write(b"x" * 1000)
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
                     "launches": [{"launch_seq": i, "arm": arm, "budget_s": 600,
                                   "argv": ["{PY}", "-m", "caplaunch", "--scenario",
                                            "{SNAP}/configs/e0.json"]}
                                  for i, arm in enumerate(["nullsse-a", "sfe-a"])]}
        self.plan_path = self.write_json("plan.json", self.plan)

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
             "--image-sha256": "d" * 64, "--image-size": "1000",
             "--mount": "%s:%s" % (self.root, self.root), "--tokenizer-dir": self.tok,
             "--snapshot-root": os.path.join(self.w, "snapshots"),
             "--run-root": os.path.join(self.w, "runs"),
             "--local-base": os.path.join(self.root, "local"),
             "--shm-base": os.path.join(self.root, "shm"), "--min-local-free-bytes": "1000",
             "--srun-mpi": "none", "--container-python": "python3",
             "--nvidia-smi": "nvidia-smi", "--gate-module": "capenv.gate",
             "--registered-hw-sha": "e" * 64, "--no-comparable": None,
             "--slurm-view-wait-s": "300"}
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
    assert not os.path.exists(os.path.join(snap, "jobs", "prep_tokenizer.sbatch.draft"))
    assert json.loads(planned["sbatch_argv"])[-1] == snap
    sh = subprocess.run(["bash", "-c", 'source "$1/job.env"; printf "%s|%s|%s" '
                         '"$CAPJOBS_SNAP" "$CAPJOBS_MOUNTS" "$CAPJOBS_TIME_LIMIT_S"', "x", snap],
                        capture_output=True, text=True, check=True)
    assert sh.stdout == "%s|%s:%s|3600" % (snap, env.root, env.root)


def test_submit_failure_records_ended(env, stub_runner):
    r = stub_runner({"sbatch": (1, "", "sbatch: error: Batch job submission failed")})
    assert run(env.submit_args(), r) == 4
    rows = env.ledger_rows()
    assert [x["event"] for x in rows] == ["PLANNED", "ENDED"]
    assert rows[1]["status"] == "SUBMIT_FAILED" and "submission failed" in rows[1]["reason"]


def test_submit_unparsable_output(env, stub_runner):
    r = stub_runner({"sbatch": (0, "Submitted batch job 12\n", "")})
    assert run(env.submit_args(), r) == 4
    assert env.ledger_rows()[-1]["status"] == "SUBMIT_FAILED"


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
    assert run(["exclude", "--cdb-file", SAVED_CDB, "--main-profile-spec",
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


def test_prep_tokenizer_is_draft_only(env, stub_runner, capsys):
    base = ["prep-tokenizer", "--account", "acct", "--partition", PARTS, "--qos", "batch",
            "--exclude", "none", "--time", "00:30:00", "--mem", "8G", "--cpus-per-task", "2",
            "--gpus", "0", "--job-name", "capkit-tok", "--login-host", "fe", "--code-root",
            env.root, "--model-dir", "/home/scratch.trt_llm_data_ci/llm-models/GLM-5.3-Flash",
            "--name", "glm53"]
    r = stub_runner({})
    assert run(base, r) == 3 and r.calls == []
    assert run(base + ["--dry-run"], r) == 0 and r.calls == []
    out = json.loads(capsys.readouterr().out)
    assert out["draft"] is True and not any(x.startswith("--gres") for x in out["remote_argv"])
    assert run([x if x != "0" else "1" for x in base] + ["--dry-run"], r) == 3

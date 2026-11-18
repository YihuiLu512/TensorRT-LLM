# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The script job body (capjobs.launchplan body "script"; main session decision D-37a): plan
validation, the experiment whitelist for it (capcommon.policy.SCRIPT_BODY_EXPERIMENTS), the run
in capjobs.launches (no pin table, no appended flags, identity in the environment, exit-code
classification, copy-back) and capjobs.submit end to end. Positive controls: an experiment
outside the whitelist, a GPU count above the design ceiling, a model mount the experiment does
not own and a script outside the code snapshot are all refused before anything is submitted."""

import json
import os
import sys
import types

import pytest
from conftest import CODE_ROOT
from test_model_whitelist import GLM, QWEN, ro
from test_submit import REG_ENV, SEC, Env, run

from capcommon import policy
from capcommon.runtree import read_state
from capjobs import launches, launchplan

SCRIPT_REL = "jobs/capkit_inner.sh"  # any file the code snapshot ships


def script_plan(exp="E12-L2a", reg="b" * 64, launches_=None, **over):
    lns = launches_ or [{"launch_seq": 0, "arm": "a-def", "budget_s": 60, "body": "script",
                         "argv": ["bash", "{SNAP}/" + SCRIPT_REL, "--rows", "1"]}]
    doc = {"schema": "capkit.launch_plan/1", "experiment": exp, "register_sha": reg,
           "job_arm": "job", "margin_s": 1, "kill_grace_s": 1, "launches": lns}
    doc.update(over)
    return doc


def test_plan_accepts_a_script_launch_and_skips_scenario_bounds():
    doc = launchplan.validate(script_plan())
    ln = doc["launches"][0]
    assert launchplan.body_of(ln) == "script" and launchplan.script_relpath(ln) == SCRIPT_REL
    assert launchplan.in_job_bound_s(ln) == 0
    assert launchplan.check_kill_grace(doc) == {}


@pytest.mark.parametrize("argv, why", [
    (["python3", "{SNAP}/x.sh"], "must start with"),
    (["bash", "/abs/x.sh"], "must start with"),
    (["bash", "{SNAP}/../x.sh"], "must start with"),
    (["bash", "{SNAP}/"], "must start with"),
    (["bash"], "must start with"),
    (["bash", "{SNAP}/x.sh", "--experiment", "E0"], "appended by capjobs"),
])
def test_plan_refuses_bad_script_argv(argv, why):
    doc = script_plan(launches_=[{"launch_seq": 0, "arm": "a", "budget_s": 60,
                                  "body": "script", "argv": argv}])
    with pytest.raises(launchplan.PlanError, match=why):
        launchplan.validate(doc)


def test_plan_refuses_unknown_body_and_mixing():
    with pytest.raises(launchplan.PlanError, match="body must be"):
        launchplan.validate(script_plan(launches_=[{"launch_seq": 0, "arm": "a", "budget_s": 9,
                                                     "body": "shell", "argv": ["bash"]}]))
    mixed = [dict(script_plan()["launches"][0]),
             {"launch_seq": 1, "arm": "b", "budget_s": 9, "argv": ["{PY}", "-m", "caplaunch"]}]
    with pytest.raises(launchplan.PlanError, match="mix"):
        launchplan.validate(script_plan(launches_=mixed))


@pytest.mark.parametrize("exp", sorted(set(policy.NO_MODEL_EXPERIMENTS)
                                       | (set(policy.MODEL_EXPERIMENTS)
                                          - set(policy.SCRIPT_BODY_EXPERIMENTS))))
def test_script_body_refused_outside_its_whitelist(exp):
    """Positive control: every other id is refused the script body (caplaunch stays allowed)."""
    with pytest.raises(policy.PolicyError, match="script job body"):
        policy.check_job_body(exp, ["script"])
    assert policy.check_job_body(exp, ["caplaunch"]) == ["caplaunch"]


def test_script_body_whitelist_is_the_e12_l2_set():
    assert set(policy.SCRIPT_BODY_EXPERIMENTS) == {"E12-L2a", "E12-L2b", "E14"}
    assert set(policy.SCRIPT_BODY_EXPERIMENTS) <= set(policy.MODEL_EXPERIMENTS)
    for exp in policy.SCRIPT_BODY_EXPERIMENTS:
        assert policy.check_job_body(exp, ["script"]) == ["script"]
    with pytest.raises(policy.PolicyError, match="unknown job body"):
        policy.check_job_body("E12-L2a", ["shell"])


SCRIPT = r"""#!/bin/bash
# test script launch: dumps its argv and CAPKIT_* environment into the launch tree, exits $1
set -u
mkdir -p "$CAPKIT_RUN_DIR/raw"
{ printf '%s\n' "$@"; env | grep '^CAPKIT_' | sort; } > "$CAPKIT_RUN_DIR/raw/script.txt"
exit "$1"
"""


def run_env(tmp_path, codes):
    snap = tmp_path / "snap"
    (snap / "jobs").mkdir(parents=True)
    (snap / "jobs" / "e12_test.sh").write_text(SCRIPT)
    lns = [{"launch_seq": i, "arm": "arm%d" % i, "budget_s": 30, "body": "script",
            "argv": ["bash", "{SNAP}/jobs/e12_test.sh", str(c), "{JOB_ID}"]}
           for i, c in enumerate(codes)]
    plan = launchplan.validate(script_plan(launches_=lns))
    a = types.SimpleNamespace(
        python=sys.executable, snap=str(snap), job_dir=str(tmp_path / "runs" / "J9-r0"),
        shm_root=str(tmp_path / "shm"), local_root=str(tmp_path / "local"),
        run_root=str(tmp_path / "runs"), job_id="9", restart_count=0, deadline_epoch=1e12,
        overlay_manifest=str(snap / "overlay_manifest.json"), nvidia_smi="/opt/x/nvidia-smi",
        hw_profile=str(tmp_path / "runs" / "J9-r0" / "env" / "hw_profile.json"))
    os.makedirs(os.path.dirname(a.hw_profile))
    os.mkdir(a.local_root, 0o700)
    manifest = {"tier": "P0", "groups": [], "capkit_code_sha": "c" * 40}
    return a, plan, manifest


def test_script_launches_run_classify_and_copy_back(tmp_path):
    a, plan, manifest = run_env(tmp_path, [0, 1, 6, 0])
    res = launches.run_all(a, plan, manifest)
    assert [r["status"] for r in res] == ["VALID", "VOID", "ENV_INVALID", "NOT_RUN"]
    assert [r["ran"] for r in res] == [True, True, True, False]
    assert not res[1]["config_refused"]  # a failing script is not a pre-start refusal
    assert res[0]["body"] == "script" and res[0]["argv"] == [
        "bash", os.path.join(a.snap, "jobs", "e12_test.sh"), "0", "9"]  # nothing appended
    assert not os.path.exists(os.path.join(a.job_dir, "pins", "J9-r0-L0_arm0.pin_table.json"))
    t0 = os.path.join(a.run_root, "J9-r0-L0_arm0")
    lines = open(os.path.join(t0, "raw", "script.txt")).read().splitlines()
    env = dict(x.split("=", 1) for x in lines if x.startswith("CAPKIT_"))
    assert lines[:2] == ["0", "9"]
    assert env["CAPKIT_EXPERIMENT"] == "E12-L2a" and env["CAPKIT_ARM"] == "arm0"
    assert env["CAPKIT_LAUNCH_SEQ"] == "0" and env["CAPKIT_JOB_ID"] == "9"
    assert env["CAPKIT_REGISTER_SHA"] == "b" * 64 and env["CAPKIT_LAUNCH_BUDGET_S"] == "30"
    assert env["CAPKIT_RUN_DIR"] == os.path.join(a.local_root, "runs", "J9-r0-L0_arm0")
    assert env["CAPKIT_JOB_ENV_DIR"] == os.path.join(a.job_dir, "env")
    assert env["CAPKIT_CODE_SHA"] == "c" * 40 and env["CAPKIT_NVIDIA_SMI"] == "/opt/x/nvidia-smi"
    st = read_state(t0)
    assert st[-1][1] == "VALID" and "copy-back verified" in st[-1][3]
    assert read_state(os.path.join(a.run_root, "J9-r0-L1_arm1"))[-1][1] == "VOID"


def test_script_launch_over_budget_is_killed_void(tmp_path):
    a, plan, manifest = run_env(tmp_path, [0])
    (tmp_path / "snap" / "jobs" / "e12_test.sh").write_text("#!/bin/bash\nsleep 30\n")
    plan["launches"][0]["budget_s"] = 1
    res = launches.run_all(a, plan, manifest)
    assert res[0]["status"] == "VOID" and res[0]["timed_out"] is True


def _e12_env(tmp_path, exp="E12-L2a", reg_gpus=4, script=SCRIPT_REL):
    env = Env(tmp_path)
    reg = env.register("reg-%s.json" % exp, dict(REG_ENV, gpus=reg_gpus), experiment=exp)
    plan = env.write_json("plan-%s.json" % exp, script_plan(
        exp=exp, reg=reg, launches_=[
            {"launch_seq": i, "arm": arm, "budget_s": 600, "body": "script",
             "argv": ["bash", "{SNAP}/" + script, "--which", arm]}
            for i, arm in enumerate(["nullsse-a", "sfe-a"])]))
    return env, {"--experiment": exp, "--register-sha": reg, "--launch-plan": plan,
                 "--gpus": str(reg_gpus), "--no-tokenizer": None, "--tokenizer-dir": False,
                 "--mount": ["%s:%s" % (env.root, env.root), SEC, ro(GLM)]}


def test_submit_e12_script_body_dry_run(tmp_path, stub_runner, capsys):
    env, over = _e12_env(tmp_path)
    over["--dry-run"] = None
    assert run(env.submit_args(**over), stub_runner({})) == 0
    out = json.loads(capsys.readouterr().out)
    assert "--gres=gpu:4" in " ".join(out["remote_argv"]) and out["model_dirs"] == [GLM]
    assert out["assertions"] == {k: True for k in policy.MODEL_ASSERTION_KEYS}
    assert out["time_budget"]["parts"]["pin_tables_s"] == 0
    assert out["time_budget"]["parts"]["job_inputs_s"] == 0


@pytest.mark.parametrize("exp, gpus, reg_gpus, mount, script, why", [
    ("E11", 1, 1, QWEN, SCRIPT_REL, "script job body"),        # whitelisted model id, no script
    ("E4", 1, 1, None, SCRIPT_REL, "script job body"),         # no-model id
    ("E12-L2a", 5, 5, GLM, SCRIPT_REL, "gpus must be"),        # above the design ceiling
    ("E12-L2a", 4, 4, QWEN, SCRIPT_REL, "registered model dirs"),
    ("E12-L2a", 4, 4, GLM, "jobs/not_there.sh", "not in the job's code snapshot"),
])
def test_submit_script_body_refusals(tmp_path, stub_runner, capsys, exp, gpus, reg_gpus, mount,
                                     script, why):
    env, over = _e12_env(tmp_path, exp, reg_gpus, script)
    over["--gpus"] = str(gpus)
    over["--mount"] = ["%s:%s" % (env.root, env.root), SEC] + ([ro(mount)] if mount else [])
    r = stub_runner({"sbatch": (0, "1\n", "")})
    assert run(env.submit_args(**over), r) == 3
    assert why in capsys.readouterr().err
    assert r.calls == [] and not os.path.exists(os.path.join(env.w, "submissions.tsv"))


def test_script_file_of_the_tests_exists_in_the_code_root():
    assert os.path.isfile(os.path.join(CODE_ROOT, SCRIPT_REL))

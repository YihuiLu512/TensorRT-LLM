# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model-experiment whitelist (capcommon.policy.MODEL_EXPERIMENTS): only the whitelisted ids
may mount a model library, only their registered model dirs, identity and read-only, with at most
their design GPU count; every other id is still refused a model mount; the job-side mount check
sees nothing of the library but the allowed dirs; the ledger accepts only such rows."""

import json
import os
import subprocess

import pytest
from conftest import CODE_ROOT
from test_submit import BASIS, HEAD, REG_ENV, SEC, Env, run

from capcommon import ledger, policy
from capjobs import envfiles, submit

QWEN = "/home/scratch.trt_llm_data_ci/llm-models/Qwen3/Qwen3-0.6B"
GPTOSS = "/home/scratch.trt_llm_data_ci/llm-models/gpt_oss/gpt-oss-120b"
GLM = "/home/scratch.trt_llm_data_ci/llm-models/GLM-5.3-Flash"
OTHER = ("/x:/x",)


def ro(d):
    return "%s:%s:ro" % (d, d)


def test_whitelist_is_exactly_the_designed_set():
    assert sorted(policy.MODEL_EXPERIMENTS) == ["E11", "E12-L2a", "E12-L2b", "E13-S", "E14",
                                                "E15-B"]
    assert {k: v["max_gpus"] for k, v in policy.MODEL_EXPERIMENTS.items()} == {
        "E11": 1, "E13-S": 1, "E15-B": 1, "E12-L2a": 4, "E12-L2b": 4, "E14": 4}
    assert not set(policy.MODEL_EXPERIMENTS) & set(policy.NO_MODEL_EXPERIMENTS)
    for e, v in policy.MODEL_EXPERIMENTS.items():
        assert all(d.startswith(policy.MODEL_LIBRARY + "/") for d in v["model_dirs"]), e


@pytest.mark.parametrize("exp", list(policy.NO_MODEL_EXPERIMENTS) + [
    "E10", "E10-proxy", "E12", "E12-L1", "E13", "E13-main", "E15", "E11x", "E15-A", ""])
def test_every_other_id_is_refused_a_model_mount(exp):
    with pytest.raises(policy.PolicyError):
        policy.check_any_submission(exp, 1, [ro(QWEN), *OTHER], None, BASIS)
    with pytest.raises(policy.PolicyError):
        policy.check_model_submission(exp, 1, [ro(QWEN), *OTHER], None, BASIS)


@pytest.mark.parametrize("exp", sorted(policy.MODEL_EXPERIMENTS))
def test_no_model_entry_still_refuses_model_ids(exp):
    with pytest.raises(policy.PolicyError, match="no-model list"):
        policy.check_no_model_submission(exp, 1, list(OTHER), None, BASIS)


def test_model_submission_accepts_the_registered_dirs():
    a, tok = policy.check_any_submission("E11", 1, [ro(QWEN), *OTHER], None, BASIS)
    assert a == {k: True for k in policy.MODEL_ASSERTION_KEYS} and tok == "none"
    assert policy.check_model_mounts("E11", [ro(QWEN)]) == [QWEN]
    assert policy.check_model_submission("E15-B", 1, [ro(GPTOSS)], None, BASIS)[0]
    for e in ("E12-L2a", "E12-L2b", "E14"):
        assert policy.check_model_submission(e, 4, [ro(GLM)], None, BASIS)[0]
    assert policy.max_gpus_for("E12-L2a") == 4 and policy.max_gpus_for("E4") == 1


@pytest.mark.parametrize("exp,gpus,mounts,why", [
    ("E11", 2, [ro(QWEN)], "gpus must be"),
    ("E11", 0, [ro(QWEN)], "gpus must be"),
    ("E12-L2a", 5, [ro(GLM)], "gpus must be"),
    ("E11", 1, ["%s:%s" % (QWEN, QWEN)], "read-only"),
    ("E11", 1, ["%s:%s:rw" % (QWEN, QWEN)], "read-only"),
    ("E11", 1, ["%s:/model:ro" % QWEN], "read-only"),
    ("E11", 1, [ro(GPTOSS)], "registered model dirs"),
    ("E11", 1, [ro(QWEN), ro(GPTOSS)], "registered model dirs"),
    ("E11", 1, [ro(policy.MODEL_LIBRARY)], "registered model dirs"),
    ("E11", 1, [ro("/home/scratch.trt_llm_data_ci")], "registered model dirs"),
    ("E11", 1, [ro("/home/scratch.trt_llm_data")], "registered model dirs"),
    ("E11", 1, [ro("/home")], "registered model dirs"),
    ("E11", 1, [ro(QWEN + "/sub")], "registered model dirs"),
    ("E11", 1, list(OTHER), "mounts none of its model dirs"),
    ("E11", 1, [ro(QWEN), ro(QWEN)], "mounted twice"),
    ("E15-B", 1, [ro(QWEN)], "registered model dirs"),
])
def test_model_submission_refusals(exp, gpus, mounts, why):
    with pytest.raises(policy.PolicyError, match=why):
        policy.check_model_submission(exp, gpus, mounts, None, BASIS)


def test_model_mount_through_a_symlink_is_refused(tmp_path):
    link = str(tmp_path / "q")
    os.symlink(QWEN, link)
    with pytest.raises(policy.PolicyError, match="registered model dirs"):
        policy.check_model_submission("E11", 1, [ro(link)], None, BASIS)


def _row(exp, gpus, mounts, keys):
    return submit.ledger_row(
        plan_id="P1-%s-abc" % exp, event="PLANNED", experiment=exp,
        reason="purpose=measurement", gpus=str(gpus), basis=BASIS,
        mounts_json=json.dumps(mounts), tokenizer_dir_sha256="none",
        assertions_json=json.dumps({k: True for k in keys}))


def test_ledger_accepts_only_whitelisted_model_rows(tmp_path):
    p = str(tmp_path / "submissions.tsv")
    ledger.submissions_append(p, _row("E11", 1, [ro(QWEN), SEC], policy.MODEL_ASSERTION_KEYS))
    ledger.submissions_append(p, _row("E12-L2a", 4, [ro(GLM)], policy.MODEL_ASSERTION_KEYS))
    bad = [_row("E11", 2, [ro(QWEN)], policy.MODEL_ASSERTION_KEYS),
           _row("E11", 1, [ro(GLM)], policy.MODEL_ASSERTION_KEYS),
           _row("E11", 1, [ro(QWEN)], policy.ASSERTION_KEYS),
           _row("E4", 1, [ro(QWEN)], policy.ASSERTION_KEYS),
           _row("E4", 1, [ro(QWEN)], policy.MODEL_ASSERTION_KEYS)]
    for r in bad:
        with pytest.raises(ledger.LedgerError):
            ledger.submissions_append(p, r)
    assert len(ledger.read_tsv(p, ledger.SUBMISSION_COLUMNS)) == 2


def _lib(tmp_path):
    root = tmp_path / "lib"
    model = root / "llm-models" / "Qwen3" / "Qwen3-0.6B"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    other = tmp_path / "w"
    other.mkdir()
    (other / "f").write_text("x")
    return str(root), str(model), str(other)


def test_job_mount_check_allows_only_the_model_dir(tmp_path):
    root, model, other = _lib(tmp_path)
    mounts = ["%s:%s" % (other, other), ro(model)]
    doc = envfiles.check_mounts(mounts, [other], forbidden=(root,), allow_model_dirs=[model])
    assert doc["passed"], doc
    # negative control: without the allowance the same container fails
    assert not envfiles.check_mounts(mounts, [other], forbidden=(root,))["passed"]
    # anything else visible in the library fails
    os.makedirs(os.path.join(root, "llm-models", "Qwen3", "Qwen3-8B"))
    doc = envfiles.check_mounts(mounts, [other], forbidden=(root,), allow_model_dirs=[model])
    assert not doc["passed"] and "more than the allowed" in json.dumps(doc)


def test_job_mount_check_refusals(tmp_path):
    root, model, other = _lib(tmp_path)
    # an allowed dir outside every library root
    doc = envfiles.check_mounts(["%s:%s" % (other, other)], [other], forbidden=(root,),
                                allow_model_dirs=[other])
    assert not doc["passed"]
    # an empty allowed dir (the mount did not happen)
    os.remove(os.path.join(model, "config.json"))
    doc = envfiles.check_mounts([ro(model)], [other], forbidden=(root,), allow_model_dirs=[model])
    assert not doc["passed"] and "missing or empty" in json.dumps(doc)
    # a mount of the library root itself is still refused
    (tmp_path / "lib" / "llm-models" / "Qwen3" / "Qwen3-0.6B" / "c").write_text("{}")
    doc = envfiles.check_mounts([ro(root)], [other], forbidden=(root,), allow_model_dirs=[model])
    assert not doc["passed"]


def test_inner_script_passes_the_model_dirs():
    with open(os.path.join(CODE_ROOT, "jobs", "capkit_inner.sh")) as f:
        s = f.read()
    assert 'for d in ${CAPJOBS_MODEL_DIRS:-}; do MODEL_ARGS+=(--allow-model-dir "$d"); done' in s
    assert '"${MODEL_ARGS[@]}" --out "$JOB_DIR/env/mounts.json"' in s


def _model_env(tmp_path, exp, gpus):
    env = Env(tmp_path)
    reg = env.register("reg-%s.json" % exp, dict(REG_ENV, gpus=gpus), experiment=exp)
    plan = env.write_json("plan-%s.json" % exp, dict(env.plan, experiment=exp, register_sha=reg))
    return env, {"--experiment": exp, "--register-sha": reg, "--launch-plan": plan,
                 "--gpus": str(gpus)}


def test_submit_e11_end_to_end(tmp_path, stub_runner, capsys):
    env, over = _model_env(tmp_path, "E11", 1)
    over["--mount"] = ["%s:%s" % (env.root, env.root), SEC, ro(QWEN)]
    r = stub_runner({"sbatch": (0, "77\n", "")})
    assert run(env.submit_args(**over), r) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["job_id"] == "77" and out["model_dirs"] == [QWEN]
    assert out["assertions"] == {k: True for k in policy.MODEL_ASSERTION_KEYS}
    planned = env.ledger_rows()[0]
    assert planned["experiment"] == "E11" and planned["gpus"] == "1"
    sh = subprocess.run(["bash", "-c", 'source "$1/job.env"; printf "%s|%s" '
                         '"$CAPJOBS_MODEL_DIRS" "$CAPJOBS_MAX_ALLOC_GPUS"', "x",
                         planned["snapshot_dir"]], capture_output=True, text=True, check=True)
    assert sh.stdout == "%s|1" % QWEN


def test_submit_e12_four_gpus(tmp_path, stub_runner, capsys):
    env, over = _model_env(tmp_path, "E12-L2a", 4)
    over["--mount"] = ["%s:%s" % (env.root, env.root), SEC, ro(GLM)]
    over["--dry-run"] = None
    assert run(env.submit_args(**over), stub_runner({})) == 0
    out = json.loads(capsys.readouterr().out)
    assert "--gres=gpu:4" in " ".join(out["remote_argv"]) and out["model_dirs"] == [GLM]


@pytest.mark.parametrize("exp,gpus,mount,reg_gpus,why", [
    ("E11", 2, QWEN, 1, "gpus must be"),
    ("E11", 1, GPTOSS, 1, "registered model dirs"),
    ("E11", 1, None, 1, "mounts none of its model dirs"),
    ("E12-L2a", 4, GLM, 5, "environment.gpus"),
    ("E4", 1, QWEN, 1, "model library"),
])
def test_submit_model_refusals(tmp_path, stub_runner, capsys, exp, gpus, mount, reg_gpus, why):
    env, over = _model_env(tmp_path, exp, reg_gpus)
    over["--gpus"] = str(gpus)
    over["--mount"] = ["%s:%s" % (env.root, env.root), SEC] + ([ro(mount)] if mount else [])
    r = stub_runner({"sbatch": (0, "1\n", "")})
    assert run(env.submit_args(**over), r) == 3
    assert why in capsys.readouterr().err
    assert r.calls == [] and not os.path.exists(os.path.join(env.w, "submissions.tsv"))


def test_head_constant_is_shared():
    assert HEAD == "c" * 40

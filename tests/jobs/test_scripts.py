# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bash scripts: syntax, static mechanism checks, and local runs that never reach a real Slurm
command (a fake `srun` on PATH records its arguments) and never delete a directory."""

import json
import os
import re
import subprocess
import sys

import pytest
from conftest import CODE_ROOT

from capcommon import policy
from capcommon.runtree import read_state

JOBS = os.path.join(CODE_ROOT, "jobs")
SBATCH = os.path.join(JOBS, "capkit_job.sbatch")
INNER = os.path.join(JOBS, "capkit_inner.sh")
DRAFT = os.path.join(JOBS, "prep_tokenizer.sbatch.draft")
WT21_SBATCH = ("/home/scratch.yihlu_coreai/dev/feat-dev-J17728/tmp/data/stage-00/test-00/"
               "00_plan/scripts/wt21_round_v4.sbatch")


def read(p):
    with open(p) as f:
        return f.read()


@pytest.mark.parametrize("path", [SBATCH, INNER, DRAFT])
def test_bash_syntax(path):
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def hardcoded(text):
    """Lines (ignoring comments) with resource or site literals that must come from params."""
    pat = re.compile(r"--gres|b200|/home/|scratch\.|--account|--partition=|computelab|umbriel")
    return [ln for ln in text.splitlines()
            if not ln.lstrip().startswith("#") and pat.search(ln)]


def test_no_hardcoded_resources_or_paths():
    assert hardcoded(read(WT21_SBATCH)), "positive control: the WT-21 script must be flagged"
    for p in (SBATCH, INNER, DRAFT):
        assert hardcoded(read(p)) == [], p
    assert not [ln for ln in read(SBATCH).splitlines() if ln.startswith("#SBATCH")]
    for p in (SBATCH, INNER):
        assert "--test-only" not in read(p)


def test_required_mechanisms_present():
    s = read(SBATCH)
    for tok in ("${SLURM_RESTART_COUNT:-0}", "--cpu-bind=none", "--no-container-mount-home",
                "--container-remap-root", "--container-writable", '--mpi="$CAPJOBS_SRUN_MPI"',
                '--container-mounts="$CAPJOBS_MOUNTS"', '--cpus-per-task="$CPT"',
                "sha256sum", "stat -c %s"):
        assert tok in s, tok
    i = read(INNER)
    for tok in ('rm -rf -- "${p:?}"', "capcommon.scratchpath", "unset PYTHONPATH",
                '"$base/$JOB_USER-$JOB_ID"', "capjobs.overlay", "capjobs.launches",
                '"$CAPJOBS_GATE_MODULE"', "env_invalid", "trap finalize EXIT"):
        assert tok in i, tok
    rm_lines = [ln.strip() for ln in i.splitlines() if re.search(r"\brm\b", ln)
                and not ln.lstrip().startswith("#")]
    assert rm_lines == ['log "rm -rf -- ${p:?}"', 'rm -rf -- "${p:?}"', 'log "rm rc=$rc"'], \
        rm_lines
    assert [ln for ln in read(SBATCH).splitlines()
            if re.search(r"\brm\b", ln) and not ln.lstrip().startswith("#")] == []
    d = read(DRAFT)
    assert "DRAFT - NOT SUBMITTABLE" in d and "python" not in d.lower().replace(
        "no python", "")


# ---- prep_tokenizer draft run locally on a fake model dir ---------------------------------------

def fake_model(root, with_vocab=True):
    os.makedirs(root)
    files = {"tokenizer_config.json": "{}", "config.json": "{}", "generation_config.json": "{}",
             "special_tokens_map.json": "{}", "chat_template.jinja": "{{x}}",
             "model-00001-of-00002.safetensors": "W", "pytorch_model.bin": "W",
             "consolidated.pth": "W", "README.md": "readme", "o200k.tiktoken": "t"}
    if with_vocab:
        files["tokenizer.json"] = '{"v":1}'
    for n, d in files.items():
        with open(os.path.join(root, n), "w") as f:
            f.write(d)
    blob = os.path.join(os.path.dirname(root), "blob-tokmodel")
    with open(blob, "w") as f:
        f.write("spm")
    os.symlink(blob, os.path.join(root, "tokenizer.model"))
    os.makedirs(os.path.join(root, "subdir"))
    with open(os.path.join(root, "subdir", "tokenizer.json"), "w") as f:
        f.write("nested is not copied")


def test_prep_tokenizer_draft(tmp_path):
    src = str(tmp_path / "model")
    fake_model(src)
    dest = str(tmp_path / "configs" / "tokenizers" / "glm")
    r = subprocess.run(["bash", DRAFT, src, dest], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = sorted(os.listdir(dest))
    assert got == sorted(["tokenizer.json", "tokenizer_config.json", "config.json",
                          "generation_config.json", "special_tokens_map.json",
                          "chat_template.jinja", "o200k.tiktoken", "tokenizer.model",
                          "SHA256SUMS.txt", "PROVENANCE.txt"])
    assert not os.path.islink(os.path.join(dest, "tokenizer.model"))
    c = subprocess.run(["sha256sum", "-c", "SHA256SUMS.txt"], cwd=dest, capture_output=True,
                       text=True)
    assert c.returncode == 0 and c.stdout.count(": OK") == 8
    prov = read(os.path.join(dest, "PROVENANCE.txt"))
    assert "model-00001-of-00002.safetensors" in prov and "pytorch_model.bin" in prov
    assert policy.tokenizer_dir_sha256(dest) != "none"
    assert not os.access(dest, os.W_OK)
    r = subprocess.run(["bash", DRAFT, src, dest], capture_output=True, text=True)
    assert r.returncode == 3
    src2 = str(tmp_path / "model2")
    fake_model(src2, with_vocab=False)
    os.unlink(os.path.join(src2, "tokenizer.model"))
    os.unlink(os.path.join(src2, "o200k.tiktoken"))
    r = subprocess.run(["bash", DRAFT, src2, str(tmp_path / "t2")], capture_output=True,
                       text=True)
    assert r.returncode == 4 and "vocabulary" in r.stderr
    assert not os.path.exists(str(tmp_path / "t2"))
    r = subprocess.run(["bash", DRAFT, "rel", dest], capture_output=True, text=True)
    assert r.returncode == 2


# ---- inner script, finalize-only, with nothing to delete ----------------------------------------

def fake_snap(tmp_path, extra=None):
    snap = tmp_path / "snap"
    snap.mkdir()
    os.symlink(os.path.join(CODE_ROOT, "tools"), str(snap / "tools"))
    os.symlink(os.path.join(CODE_ROOT, "jobs"), str(snap / "jobs"))
    env = {"CAPJOBS_SNAP": str(snap), "CAPJOBS_PLAN_ID": "P1", "CAPJOBS_EXPERIMENT": "E0",
           "CAPJOBS_REGISTER_SHA": "a" * 64, "CAPJOBS_JOB_ARM": "job",
           "CAPJOBS_RUN_ROOT": str(tmp_path / "runs"), "CAPJOBS_LOCAL_BASE":
           str(tmp_path / "local"), "CAPJOBS_SHM_BASE": str(tmp_path / "shm"),
           "CAPJOBS_PYTHON": sys.executable, "CAPJOBS_IMAGE": str(tmp_path / "img.sqsh"),
           "CAPJOBS_IMAGE_SHA256": "0" * 64, "CAPJOBS_IMAGE_SIZE": "3",
           "CAPJOBS_MOUNTS": "%s:%s" % (tmp_path, tmp_path), "CAPJOBS_SRUN_MPI": "none",
           "CAPJOBS_TIME_LIMIT_S": "3600"}
    env.update(extra or {})
    (snap / "job.env").write_text("".join("export %s='%s'\n" % kv for kv in env.items()))
    for d in ("runs", "local", "shm"):
        (tmp_path / d).mkdir()
    return str(snap)


def slurm_env(tmp_path, **kw):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "CAPJOBS_"))}
    e.update({"SLURM_JOB_ID": "4242", "SLURM_JOB_USER": "tuser", "SLURM_CPUS_PER_TASK": "8",
              "SLURMD_NODENAME": "node1"})
    e.update(kw)
    return e


def test_inner_finalize_only_empty(tmp_path):
    snap = fake_snap(tmp_path)
    jd = tmp_path / "runs" / "J4242-r0"
    jd.mkdir()
    r = subprocess.run(["bash", INNER, snap, str(jd), "finalize-only"], capture_output=True,
                       text=True, env=slurm_env(tmp_path))
    assert r.returncode == 0, r.stderr
    assert (jd / "finalize.done").exists()
    v = json.load(open(jd / "copyback" / "final.verify.json"))
    assert v["outcome"] == "EMPTY"
    log = (jd / "inner.finalize-only.log").read_text()
    assert log.count("nothing to delete") == 2 and "rm -rf" not in log


def test_inner_finalize_keeps_unverified_data(tmp_path):
    snap = fake_snap(tmp_path)
    jd = tmp_path / "runs" / "J4242-r0"
    jd.mkdir()
    local_root = tmp_path / "local" / "tuser-4242"
    local_root.mkdir()
    (local_root / "stray.txt").write_text("not inside a tree")
    r = subprocess.run(["bash", INNER, snap, str(jd), "finalize-only"], capture_output=True,
                       text=True, env=slurm_env(tmp_path))
    assert r.returncode == 0
    assert (local_root / "stray.txt").exists() and not (jd / "finalize.done").exists()
    assert "did not verify" in (jd / "inner.finalize-only.log").read_text()


def test_inner_bad_mode(tmp_path):
    snap = fake_snap(tmp_path)
    r = subprocess.run(["bash", INNER, snap, str(tmp_path), "bogus"], capture_output=True,
                       text=True, env=slurm_env(tmp_path))
    assert r.returncode == 2


# ---- batch script with a fake srun ------------------------------------------------------------

def fake_srun_path(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    rec = tmp_path / "srun_calls.txt"
    (b / "srun").write_text('#!/bin/bash\nprintf "%%s\\n" "$*" >> "%s"\nexit 0\n' % rec)
    (b / "srun").chmod(0o755)
    return str(b), rec


def test_batch_image_mismatch_stops_before_srun(tmp_path):
    snap = fake_snap(tmp_path)
    (tmp_path / "img.sqsh").write_bytes(b"abc")
    b, rec = fake_srun_path(tmp_path)
    env = slurm_env(tmp_path, PATH=b + ":" + os.environ["PATH"])
    r = subprocess.run(["bash", SBATCH, snap], capture_output=True, text=True, env=env)
    assert r.returncode == 6, r.stdout + r.stderr
    assert not rec.exists()
    jd = tmp_path / "runs" / "J4242-r0"
    assert read_state(str(jd))[-1][1] == "ENV_INVALID"
    assert "measured_size\t3" in (jd / "env" / "image_check.tsv").read_text()
    r = subprocess.run(["bash", SBATCH, snap], capture_output=True, text=True, env=env)
    assert r.returncode == 3  # the job dir exists: never reused


def test_batch_runs_main_and_fallback_steps(tmp_path):
    sha = subprocess.run(["sha256sum"], input=b"abc", capture_output=True).stdout.split()[0]
    snap = fake_snap(tmp_path, {"CAPJOBS_IMAGE_SHA256": sha.decode()})
    (tmp_path / "img.sqsh").write_bytes(b"abc")
    b, rec = fake_srun_path(tmp_path)
    env = slurm_env(tmp_path, PATH=b + ":" + os.environ["PATH"], SLURM_RESTART_COUNT="2")
    r = subprocess.run(["bash", SBATCH, snap], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = rec.read_text().splitlines()
    assert len(calls) == 2  # main step, then fallback finalize (no finalize.done)
    jd = str(tmp_path / "runs" / "J4242-r2")
    assert calls[0].endswith("%s/jobs/capkit_inner.sh %s %s run" % (snap, snap, jd))
    assert calls[1].endswith("finalize-only")
    for tok in ("--cpu-bind=none", "--mpi=none", "--cpus-per-task=8",
                "--no-container-mount-home",
                "--container-mounts=%s:%s" % (tmp_path, tmp_path), "--container-writable",
                "--container-remap-root", "--container-image=%s" % (tmp_path / "img.sqsh")):
        assert tok in calls[0], tok
    assert "deadline_epoch" in open(os.path.join(jd, "batch.tsv")).read()

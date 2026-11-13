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
from conftest import CODE_ROOT, WT21_SBATCH, external_input

from capcommon import policy
from capcommon.runtree import read_state

JOBS = os.path.join(CODE_ROOT, "jobs")
SBATCH = os.path.join(JOBS, "capkit_job.sbatch")
INNER = os.path.join(JOBS, "capkit_inner.sh")
DRAFT = os.path.join(JOBS, "prep_tokenizer.sbatch.draft")


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
    assert hardcoded(read(external_input(WT21_SBATCH))), \
        "positive control: the WT-21 script must be flagged"
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
                "stat -c %s", "date -r", "CAPJOBS_IMAGE_MTIME_NS"):
        assert tok in s, tok
    i = read(INNER)
    assert "sha256sum" not in s  # the image is hashed once at submission, not in the allocation
    for tok in ('rm -rf -- "${p:?}"', "capcommon.scratchpath", "unset PYTHONPATH",
                '"$base/$JOB_USER-$SCRATCH_ID"', 'SCRATCH_ID="$JOB_ID-r$RESTART"',
                "capjobs.overlay", "capjobs.launches",
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
           "CAPJOBS_IMAGE_MTIME_NS": "1",
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
    assert "no node-local dir owned by this run" in log and "rm -rf" not in log


def test_inner_finalize_keeps_unverified_data(tmp_path):
    snap = fake_snap(tmp_path)
    jd = tmp_path / "runs" / "J4242-r0"
    jd.mkdir()
    local_root = tmp_path / "local" / "tuser-4242-r0"
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
    (tmp_path / "img.sqsh").write_bytes(b"abc")  # size matches; mtime differs from the record
    b, rec = fake_srun_path(tmp_path)
    env = slurm_env(tmp_path, PATH=b + ":" + os.environ["PATH"])
    r = subprocess.run(["bash", SBATCH, snap], capture_output=True, text=True, env=env)
    assert r.returncode == 6, r.stdout + r.stderr
    assert not rec.exists()
    jd = tmp_path / "runs" / "J4242-r0"
    assert read_state(str(jd))[-1][1] == "ENV_INVALID"
    tsv = (jd / "env" / "image_check.tsv").read_text()
    assert "measured_size\t3" in tsv and "expected_mtime_ns\t1\n" in tsv
    r = subprocess.run(["bash", SBATCH, snap], capture_output=True, text=True, env=env)
    assert r.returncode == 3  # the job dir exists: never reused


def test_batch_runs_main_and_fallback_steps(tmp_path):
    (tmp_path / "img.sqsh").write_bytes(b"abc")
    mt = os.stat(tmp_path / "img.sqsh").st_mtime_ns
    snap = fake_snap(tmp_path, {"CAPJOBS_IMAGE_MTIME_NS": str(mt)})
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


# ---- inner script, run mode, with a fake container python (wiring of K1 / K6 / K18 / K22) ------

FAKE_PY = r'''#!/bin/bash
printf '%%s\n' "$*" >> "%(calls)s"
[[ $1 == -c ]] && { echo "%(purelib)s"; exit 0; }
[[ $1 == -m && $2 == capjobs.copyback ]] && exit 1   # finalize keeps the node-local dirs
[[ $1 == -m && $2 == capenv.gate ]] && exit "${FAKE_GATE_RC:-0}"
exit 0
'''
NVSMI_ROW = "0, GPU-0, 00000000:1B:00.0, NVIDIA B200, 183359, 0, 0, 580.1"


def run_snap(tmp_path, **over):
    calls, purelib = tmp_path / "py_calls.txt", tmp_path / "purelib"
    purelib.mkdir()
    py = tmp_path / "fakepy"
    py.write_text(FAKE_PY % {"calls": calls, "purelib": purelib})
    py.chmod(0o755)
    smi = tmp_path / "nv" / "nvidia-smi"
    smi.parent.mkdir()
    smi.write_text('#!/bin/bash\n[[ $1 == --query-gpu=* ]] && echo "%s"\nexit 0\n' % NVSMI_ROW)
    smi.chmod(0o755)
    extra = {"CAPJOBS_PYTHON": str(py), "CAPJOBS_GATE_MODULE": "capenv.gate",
             "CAPJOBS_REGISTERED_HW_SHA": "e" * 64, "CAPJOBS_NVIDIA_SMI": str(smi),
             "CAPJOBS_MAX_ALLOC_GPUS": "1", "CAPJOBS_IDLE_WINDOW_S": "30",
             "CAPJOBS_IDLE_BUSY_MAX_FRAC": "0.02", "CAPJOBS_SLURM_VIEW_WAIT_S": "1",
             "CAPJOBS_CHASSIS_SOURCE": "slurm_view", "CAPJOBS_CODE_SHA": "c" * 40,
             "CAPJOBS_MIN_LOCAL_FREE_BYTES": "1", "CAPJOBS_DEADLINE_EPOCH": "9999999999",
             "CAPJOBS_IMAGE_SHA256": "f" * 64, "CAPJOBS_SRUN_MPI": "pmix"}
    extra.update(over)
    snap = fake_snap(tmp_path, extra)
    return snap, calls, smi


def slurm_view_doc(jd):
    items = [{"id": 1, "cmd": ["scontrol", "show", "job", "4242"], "rc": 0,
              "stdout": "JobId=4242 Partition=p1 NodeList=node1\n", "stdout_sha256": None,
              "stderr": "", "taken_wall": "2026-11-13T00:00:00.000Z"},
             {"id": 2, "cmd": ["scontrol", "show", "node", "node1"], "rc": 0,
              "stdout": "NodeName=node1 SystemProductName=DGXB200\n", "stdout_sha256": None,
              "stderr": "", "taken_wall": "2026-11-13T00:00:00.000Z"}]
    os.makedirs(os.path.join(str(jd), "env"), exist_ok=True)
    with open(os.path.join(str(jd), "env", "slurm_view.json"), "w") as f:
        json.dump({"schema": "capkit.slurm_view/1", "job_id": "4242", "node": "node1",
                   "items": items}, f)


def call_of(calls, module):
    found = [ln.split(" ") for ln in calls.read_text().splitlines()
             if ln.startswith("-m %s " % module)]
    assert len(found) == 1, (module, calls.read_text())
    return found[0][2:]


def flag(argv, name):
    return argv[argv.index(name) + 1]


def test_inner_run_wires_registered_limits_into_profile_gate_and_launches(tmp_path):
    snap, calls, smi = run_snap(tmp_path)
    jd = tmp_path / "runs" / "J4242-r0"
    jd.mkdir()
    slurm_view_doc(jd)
    r = subprocess.run(["bash", INNER, snap, str(jd), "run"], capture_output=True, text=True,
                       env=slurm_env(tmp_path))
    assert r.returncode == 0, (jd / "inner.run.log").read_text()
    local_root = str(tmp_path / "local" / "tuser-4242-r0")
    prof = call_of(calls, "capenv.hwprofile")
    assert flag(prof, "--nvidia-smi") == str(smi) and flag(prof, "--role") == "hwprofile"
    assert flag(prof, "--chassis-source") == "slurm_view"
    assert flag(prof, "--slurm-view-file") == str(jd / "env" / "slurm_view.json")
    assert flag(prof, "--local-dir") == local_root
    assert flag(prof, "--out") == str(jd / "env" / "hw_profile.json")
    gate = call_of(calls, "capenv.gate")
    for k, v in (("--max-alloc-gpus", "1"), ("--idle-window-s", "30"),
                 ("--idle-busy-max-frac", "0.02"), ("--nvidia-smi", str(smi)),
                 ("--hw-profile", str(jd / "env" / "hw_profile.json")),
                 ("--registered-hw-sha", "e" * 64), ("--slurm-view-wait-s", "1")):
        assert flag(gate, k) == v, k
    gpu = [ln.split(" ") for ln in calls.read_text().splitlines()
           if ln.startswith("-m capjobs.envfiles gpu-idle")][0]
    assert flag(gpu, "--max-alloc-gpus") == "1" and flag(gpu, "--nvidia-smi") == str(smi)
    lau = call_of(calls, "capjobs.launches")
    assert flag(lau, "--nvidia-smi") == str(smi)
    assert flag(lau, "--overlay-manifest") == snap + "/overlay_manifest.json"
    # The recorded argv must be accepted by the real capenv components (cross-component names).
    from capenv import gate as real_gate
    from capenv import hwprofile as real_hwprofile
    g = real_gate.build_parser().parse_args(gate[2:] if gate[0] == "-m" else gate)
    assert (g.max_alloc_gpus, g.idle_window_s, g.idle_busy_max_frac, g.nvidia_smi) == (
        1, 30.0, 0.02, str(smi))
    assert real_hwprofile.main(prof) == 0
    doc = json.load(open(jd / "env" / "hw_profile.json"))
    assert doc["profile"]["gpu_count"] == 1 and doc["profile"]["chassis_model"] == "DGXB200"


def test_inner_unregistered_profile_capture_stops_at_gate(tmp_path):
    snap, calls, _ = run_snap(tmp_path, CAPJOBS_REGISTERED_HW_SHA="UNREGISTERED")
    jd = tmp_path / "runs" / "J4242-r0"
    jd.mkdir()
    slurm_view_doc(jd)
    r = subprocess.run(["bash", INNER, snap, str(jd), "run"], capture_output=True, text=True,
                       env=slurm_env(tmp_path, FAKE_GATE_RC="6"))
    assert r.returncode == 6
    st = read_state(str(jd))[-1]
    assert st[1] == "ENV_INVALID" and "profile-only" in st[3]
    assert "capjobs.launches" not in calls.read_text()
    assert "--registered-hw-sha UNREGISTERED" in calls.read_text()


def test_inner_waits_for_slurm_view_then_env_invalid(tmp_path):
    snap, calls, _ = run_snap(tmp_path)
    jd = tmp_path / "runs" / "J4242-r0"
    jd.mkdir()
    r = subprocess.run(["bash", INNER, snap, str(jd), "run"], capture_output=True, text=True,
                       env=slurm_env(tmp_path))
    assert r.returncode == 6
    assert "slurm_view.json not written" in read_state(str(jd))[-1][3]
    assert "capenv.hwprofile" not in calls.read_text()


def test_inner_requeue_and_foreign_scratch_dir(tmp_path):
    """K22: a requeued run (restart 1) is not blocked by restart 0's leftover dir; a dir this run
    did not create is ENV_INVALID and is never handed to the copy-back sweep."""
    snap, calls, _ = run_snap(tmp_path)
    (tmp_path / "local" / "tuser-4242-r0").mkdir()  # left by an unclean restart 0
    jd = tmp_path / "runs" / "J4242-r1"
    jd.mkdir()
    slurm_view_doc(jd)
    r = subprocess.run(["bash", INNER, snap, str(jd), "run"], capture_output=True, text=True,
                       env=slurm_env(tmp_path, SLURM_RESTART_COUNT="1"))
    assert r.returncode == 0, (jd / "inner.run.log").read_text()
    assert (tmp_path / "local" / "tuser-4242-r1").is_dir()
    sweep = call_of(calls, "capjobs.copyback")
    assert flag(sweep, "--local-root") == str(tmp_path / "local" / "tuser-4242-r1")
    calls.unlink()
    foreign = tmp_path / "local" / "tuser-4242-r2"
    foreign.mkdir()
    (foreign / "theirs.txt").write_text("x")
    jd2 = tmp_path / "runs" / "J4242-r2"
    jd2.mkdir()
    r = subprocess.run(["bash", INNER, snap, str(jd2), "run"], capture_output=True, text=True,
                       env=slurm_env(tmp_path, SLURM_RESTART_COUNT="2"))
    assert r.returncode == 6
    sweep = call_of(calls, "capjobs.copyback")
    assert "--local-root" not in sweep and "--shm-root" not in sweep
    assert (foreign / "theirs.txt").exists()

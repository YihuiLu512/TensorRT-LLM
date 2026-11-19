# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""jobs/e12_inner.sh (E12-L2a script launch body) without a GPU: a fake interpreter stands in for
the CUDA probe and the runner (E12_PYTHON); checks the per-step environment (arm, reqlog dir,
marker, PATH shim first), step records, the deadline / preflight refusals and the exit codes."""

import os
import subprocess

import pytest
from conftest import CODE_ROOT

INNER = os.path.join(CODE_ROOT, "jobs", "e12_inner.sh")
FAKE_PY = r"""#!/bin/bash
# probe: `PY - N` with the program on stdin; runner: `PY <runner.py> --chain C --out O ...`
if [[ $1 == - ]]; then cat > /dev/null; [[ -n ${FAKE_PROBE_FAIL:-} ]] && exit 1
    echo "CUDA_DEVICE_COUNT $2"; exit 0; fi
out=""; prev=""
for a in "$@"; do [[ $prev == --out ]] && out=$a; prev=$a; done
mkdir -p "$out"
{ echo "argv=$*"; echo "arm=$TRTLLM_PERF_ARM"; echo "reqlog=$E12_REQLOG_DIR";
  echo "mark=$E12_MARK"; echo "path0=${PATH%%:*}"; echo "agg=$AGG_CONFIG_FOLDER";
  echo "python=$(command -v python)"; } > "$out/fake.txt"
exit ${FAKE_RUNNER_RC:-0}
"""


@pytest.fixture
def setup(tmp_path):
    src = tmp_path / "src"
    runner = src / "tests" / "integration" / "defs" / "openengine_chain" / "runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# fake\n")
    g = ["git", "-C", str(src), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(g[:3] + ["init", "-q"], check=True)
    subprocess.run(g + ["add", "-A"], check=True)
    subprocess.run(g + ["commit", "-q", "-m", "x"], check=True)
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    py = tmp_path / "fakepy"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    env = {"PATH": "/usr/bin:/bin", "CAPKIT_RUN_DIR": str(run_dir), "CAPKIT_JOB_ID": "77",
           "CAPKIT_LAUNCH_BUDGET_S": "3600", "CAPKIT_NVIDIA_SMI": "/bin/true",
           "CAPKIT_EXPERIMENT": "E12-L2a", "CAPKIT_ARM": "job", "E12_PYTHON": str(py),
           "TMPDIR": str(tmp_path)}
    args = ["--src", str(src), "--src-head", head, "--config-dir", str(cfg), "--gpus", "4",
            "--step-timeout", "600", "--step-expect", "500"]
    return env, args, run_dir


def _run(env, args, steps="A-def=u4m-r1 B-full=u4m-r1", **extra):
    return subprocess.run(["bash", INNER] + args + ["--steps", steps], env=dict(env, **extra),
                          capture_output=True, text=True, timeout=120)


def test_steps_run_in_order_with_their_environment(setup):
    env, args, run_dir = setup
    p = _run(env, args)
    assert p.returncode == 0, p.stdout + p.stderr
    raw = run_dir / "raw"
    steps = sorted(d.name for d in raw.iterdir() if d.is_dir())
    assert steps == ["01_A-def__u4m-r1", "02_B-full__u4m-r1"]
    for name, arm, chain in ((steps[0], "A-def", "none"), (steps[1], "B-full", "trtllm")):
        fake = dict(x.split("=", 1) for x in (raw / name / "run" / "fake.txt").read_text()
                    .splitlines())
        assert fake["arm"] == arm and fake["reqlog"] == str(raw / name / "reqlog")
        assert ("--chain %s" % chain) in fake["argv"] and fake["mark"].startswith("77-")
        assert fake["path0"] == str(run_dir / "e12shim" / "bin") and fake["agg"] == args[5]
        assert fake["python"] == str(run_dir / "e12shim" / "bin" / "python")
        assert (raw / name / "list.txt").read_text().strip() == \
            "perf/test_perf_sanity.py::test_e2e[aggr-u4m-r1]"
        assert "rc=0" in (raw / name / "step.txt").read_text()
    assert "INNER DONE" in (raw / "inner.txt").read_text()


def test_a_failing_step_is_recorded_and_the_rest_still_run(setup):
    env, args, run_dir = setup
    p = _run(env, args, FAKE_RUNNER_RC="1")
    assert p.returncode == 0
    for d in ("01_A-def__u4m-r1", "02_B-full__u4m-r1"):
        assert "rc=1" in (run_dir / "raw" / d / "step.txt").read_text()


@pytest.mark.parametrize("change, why", [
    ({"budget": "1200"}, "80% of the launch budget"),       # 2 x 500 > 0.8 x 1200
    ({"steps": "A-def=x C-s1=y"}, "bad arm"),
    ({"steps": "A-def="}, "bad step"),
    ({"head": "0" * 40}, "source HEAD differs"),
])
def test_refusals_before_any_step(setup, change, why):
    env, args, run_dir = setup
    if "head" in change:
        args = list(args)
        args[3] = change["head"]
    extra = {"CAPKIT_LAUNCH_BUDGET_S": change["budget"]} if "budget" in change else {}
    p = _run(env, args, steps=change.get("steps", "A-def=u4m-r1 B-full=u4m-r1"), **extra)
    assert p.returncode == 3 and why in (run_dir / "raw" / "REFUSED").read_text()
    assert not any(d.is_dir() for d in (run_dir / "raw").iterdir())


def test_cuda_probe_failure_is_env_invalid(setup):
    env, args, run_dir = setup
    p = _run(env, args, FAKE_PROBE_FAIL="1")
    assert p.returncode == 6
    assert not any(d.is_dir() for d in (run_dir / "raw").iterdir())


def test_shim_works_from_a_read_only_snapshot_without_exec_bits(setup, tmp_path):
    """Positive control for E12 smoke 6148825: capjobs' snapshot drops exec bits; the inner must
    still put an executable shim first on PATH (a run-local copy)."""
    import shutil
    env, args, run_dir = setup
    snap = tmp_path / "snap"
    (snap / "jobs").mkdir(parents=True)
    shutil.copyfile(INNER, snap / "jobs" / "e12_inner.sh")
    shutil.copytree(os.path.join(CODE_ROOT, "tools", "cape12"), snap / "tools" / "cape12",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for root, dirs, files in os.walk(snap):
        for f in files:
            os.chmod(os.path.join(root, f), 0o444)
    assert not os.access(snap / "tools" / "cape12" / "bin" / "python", os.X_OK)
    p = subprocess.run(["bash", str(snap / "jobs" / "e12_inner.sh")] + args
                       + ["--steps", "A-def=u4m-r1"], env=env, capture_output=True, text=True,
                       timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    fake = dict(x.split("=", 1) for x in (run_dir / "raw" / "01_A-def__u4m-r1" / "run" /
                                          "fake.txt").read_text().splitlines())
    shim = run_dir / "e12shim"
    assert fake["python"] == str(shim / "bin" / "python") and os.access(fake["python"], os.X_OK)
    assert (shim / "reqlog_l2a.py").is_file()
    assert "reqlog_files=0" in (run_dir / "raw" / "01_A-def__u4m-r1" / "step.txt").read_text()

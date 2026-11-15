# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the sample self-test (good, one broken sample per check, empty dir, two satisfiability
samples, input-invalid samples, judge-twice, glob, sabotage) and show that it can go red:
a tampered copy of the judge must make the runner fail (pitfalls 1)."""

import json
import os
import shutil
import subprocess
import sys

import pytest
from conftest import PKG, SAMPLES, TOOLS

N_SAMPLES = 52  # entries of make_samples.SAMPLES + empty_tree (test_sample_table_size)
RUNNER = os.path.join(SAMPLES, "run_selftest.py")


def _expected_count():
    sys.path.insert(0, SAMPLES)
    try:
        import make_samples
    finally:
        sys.path.remove(SAMPLES)
    return len(make_samples.SAMPLES) + 1


def _gen(tmp_path):
    out = tmp_path / "samples"
    p = subprocess.run([sys.executable, os.path.join(SAMPLES, "make_samples.py"), "--out",
                        str(out)], capture_output=True, text=True, timeout=300, check=False)
    assert p.returncode == 0, p.stderr
    return str(out)


def _runner(samples, tools, tmp_path, n):
    tmp = tmp_path / "rt"
    tmp.mkdir(exist_ok=True)
    return subprocess.run([sys.executable, RUNNER, "--samples", samples, "--tools", tools,
                           "--package", PKG, "--tmp", str(tmp), "--expect-samples", str(n)],
                          capture_output=True, text=True, timeout=600, check=False)


def test_sample_table_size():
    assert _expected_count() == N_SAMPLES


def test_generated_samples_pass(tmp_path):
    samples = _gen(tmp_path)
    p = _runner(samples, TOOLS, tmp_path, N_SAMPLES)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]


def test_runner_requires_expect_samples(tmp_path):
    tmp = tmp_path / "rt"
    tmp.mkdir()
    base = [sys.executable, RUNNER, "--samples", str(tmp_path / "none"), "--tools", TOOLS,
            "--package", PKG, "--tmp", str(tmp)]
    for extra in ([], ["--expect-samples", "0"]):
        p = subprocess.run(base + extra, capture_output=True, text=True, timeout=60,
                           check=False)
        assert p.returncode == 2, (extra, p.stdout, p.stderr)
        assert "expect-samples" in p.stderr


def test_runner_red_on_wrong_sample_count(tmp_path):
    samples = _gen(tmp_path)
    p = _runner(samples, TOOLS, tmp_path, N_SAMPLES + 1)
    assert p.returncode == 1 and "expected %d" % (N_SAMPLES + 1) in p.stdout


@pytest.mark.skipif(not os.path.isdir(os.path.join(SAMPLES, "good")),
                    reason="committed samples not generated yet")
def test_committed_samples_pass(tmp_path):
    p = _runner(SAMPLES, TOOLS, tmp_path, N_SAMPLES)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]


TAMPERS = [
    # (file, old, new): each makes a class of samples pass that must fail
    ("checks.py", "        elif self.red > 0:\n            self.status = RED\n",
     "        elif self.red > 0:\n            self.status = PASS\n"),
    ("symptoms.py", "    elif \"above\" in x2_dirs:\n        x2 = TRUE\n",
     "    elif \"above\" in x2_dirs:\n        x2 = FALSE\n"),
    ("judge.py", "        elif x[\"contradiction\"]:\n", "        elif False:\n"),
]


@pytest.mark.parametrize("fname,old,new", TAMPERS)
def test_tampered_judge_turns_selftest_red(tmp_path, fname, old, new):
    samples = _gen(tmp_path)
    tdir = tmp_path / "tampered_tools_TAMPERED_DO_NOT_USE"
    shutil.copytree(os.path.join(TOOLS, PKG), tdir / PKG)
    target = tdir / PKG / fname
    src = target.read_text()
    assert src.count(old) == 1  # positive control: the edit hits its target exactly once
    target.write_text(src.replace(old, new))
    assert target.read_text() != src
    p = _runner(samples, str(tdir) + os.pathsep + TOOLS, tmp_path, N_SAMPLES)
    assert p.returncode == 1, p.stdout[-2000:]
    report = json.loads(p.stdout)  # the runner itself completed: red, not a crash
    assert len(report["ran"]) == N_SAMPLES and report["errors"], report
    print(fname, report["errors"])


LEVEL_SAMPLES = ["v2_near_stable_level_2p", "v2_sat_overloaded_level_2p"]


def test_level_expectations_are_read(tmp_path):
    """expect.json `levels` (calib-v2 test 22, the two-pass satisfiability samples): with the
    two-pass composition tampered so that a level inside in both passes composes UNSTABLE, the
    tier verdicts do not change and only the level expectation of v2_near_stable_level_2p goes
    red; untampered, both samples pass."""
    out = tmp_path / "samples"
    p = subprocess.run([sys.executable, os.path.join(SAMPLES, "make_samples.py"), "--out",
                        str(out)] + [x for n in LEVEL_SAMPLES for x in ("--only", n)],
                       capture_output=True, text=True, timeout=300, check=False)
    assert p.returncode == 0, p.stderr
    tdir = tmp_path / "tampered_tools_TAMPERED_DO_NOT_USE"
    shutil.copytree(os.path.join(TOOLS, PKG), tdir / PKG)
    target = tdir / PKG / "judge.py"
    old = '    if not a["holds"] and not b["holds"]:\n        return "STABLE", a["keys"]'
    src = target.read_text()
    assert src.count(old) == 1  # positive control: the edit hits its target exactly once
    target.write_text(src.replace(old, old.replace('"STABLE"', '"UNSTABLE"')))
    reports = {}
    for tag, tools in (("clean", TOOLS), ("tampered", str(tdir) + os.pathsep + TOOLS)):
        rt = tmp_path / ("rt-" + tag)
        rt.mkdir()
        q = subprocess.run([sys.executable, RUNNER, "--samples", str(out), "--tools", tools,
                            "--package", PKG, "--tmp", str(rt), "--expect-samples",
                            str(len(LEVEL_SAMPLES))],
                           capture_output=True, text=True, timeout=600, check=False)
        reports[tag] = (q.returncode, json.loads(q.stdout))
    assert reports["clean"][0] == 0 and not reports["clean"][1]["errors"], reports["clean"]
    rc, rep = reports["tampered"]
    assert rc == 1 and sorted(r[0] for r in rep["ran"]) == LEVEL_SAMPLES, rep
    assert rep["errors"] == ["v2_near_stable_level_2p: level 1 verdict 'UNSTABLE' != "
                             "expected 'STABLE'"], rep

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


# ---- samples deferred to P2-13 (INTERFACES erratum calib-v2-A7) ---------------------------------

def _deferred_list():
    sys.path.insert(0, SAMPLES)
    try:
        import run_selftest
    finally:
        sys.path.remove(SAMPLES)
    return run_selftest.DEFERRED_P2_13


def _gen_only(tmp_path, *names):
    out = tmp_path / "samples"
    p = subprocess.run([sys.executable, os.path.join(SAMPLES, "make_samples.py"), "--out",
                        str(out)] + [x for n in names for x in ("--only", n)],
                       capture_output=True, text=True, timeout=300, check=False)
    assert p.returncode == 0, p.stderr
    return out


def test_deferred_list_is_the_five_i_samples_of_the_table():
    sys.path.insert(0, SAMPLES)
    try:
        import make_samples
    finally:
        sys.path.remove(SAMPLES)
    assert sorted(_deferred_list()) == ["i_conflict", "i_reader_slow", "i_reader_slow_real_pins",
                                        "i_server_slow", "sat_spin_thread_same_process"]
    assert set(_deferred_list()) <= set(make_samples.SAMPLES)
    for name, parts in _deferred_list().items():  # i_conflict: UNDETERMINED at tier 4 needs I
        assert sorted(parts) == sorted(["tier_localisation", "grade"] + (
            ["tier_verdicts.4"] if name == "i_conflict" else [])), name


def test_deferred_sample_is_reported_apart_and_does_not_fail_the_runner(tmp_path):
    out = _gen_only(tmp_path, "i_server_slow")
    p = _runner(str(out), TOOLS, tmp_path, 1)
    rep = json.loads(p.stdout)
    assert p.returncode == 0 and rep["errors"] == [], rep
    assert rep["ran"] == [["i_server_slow", 0]]
    assert any("I status='NOT_RUN' != expected 'LOCALIZED'" in e
               for e in rep["deferred"]["i_server_slow"]), rep


def test_deferred_sample_with_a_wrong_exit_code_is_still_an_error(tmp_path):
    out = _gen_only(tmp_path, "i_server_slow")
    ep = out / "i_server_slow" / "expect.json"
    exp = json.loads(ep.read_text())
    exp["exit_code"] = 3
    ep.write_text(json.dumps(exp))
    p = _runner(str(out), TOOLS, tmp_path, 1)
    rep = json.loads(p.stdout)
    assert p.returncode == 1 and rep["errors"] == ["i_server_slow: exit 0 != expected 3"], rep


def test_deferred_sample_keeps_its_other_expectations(tmp_path):
    """Only the listed parts are deferred: a wrong tier verdict outside them is an error (this
    keeps e.g. the X2 tamper red on the I samples)."""
    out = _gen_only(tmp_path, "i_server_slow")
    ep = out / "i_server_slow" / "expect.json"
    exp = json.loads(ep.read_text())
    exp["tier_verdicts"]["2"] = "UNSTABLE"
    ep.write_text(json.dumps(exp))
    p = _runner(str(out), TOOLS, tmp_path, 1)
    rep = json.loads(p.stdout)
    assert p.returncode == 1, rep
    assert rep["errors"] == ["i_server_slow: tier 2 verdict 'STABLE' != expected 'UNSTABLE'"], rep
    assert rep["deferred"]["i_server_slow"], rep


def test_deferred_sample_that_matches_its_expectation_fails_the_runner(tmp_path):
    """The list cannot rot: a listed sample whose whole expectation holds is an error (here the
    good sample under a deferred name)."""
    out = _gen_only(tmp_path, "good")
    os.rename(str(out / "good"), str(out / "i_conflict"))
    p = _runner(str(out), TOOLS, tmp_path, 1)
    rep = json.loads(p.stdout)
    assert p.returncode == 1, rep
    assert rep["errors"] == ["i_conflict: listed in DEFERRED_P2_13 but matches its expectation; "
                             "remove it from the list"], rep
    assert rep["deferred"] == {"i_conflict": []}

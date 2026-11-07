# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The runner's test lists, environment, pytest commands and summary."""

import json
import os
import subprocess
import sys

import pytest
from openengine_chain import conventions as cv
from openengine_chain import runner

pytestmark = pytest.mark.cpu_only


def test_isolation_entries_run_alone_without_the_marker():
    entries = [
        "test_e2e.py::test_a",
        "test_e2e.py::test_b TIMEOUT (90) ISOLATION",
        "unittest/x.py -k y",
        "test_e2e.py::test_c ISOLATION",
    ]
    assert runner.batches(entries) == [
        ("", ["test_e2e.py::test_a", "unittest/x.py -k y"]),
        ("-iso1", ["test_e2e.py::test_b TIMEOUT (90)"]),
        ("-iso2", ["test_e2e.py::test_c"]),
    ]


def test_read_entries_drops_comments_and_duplicates(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("# header\ntest_e2e.py::test_a  # why\n\ntest_e2e.py::test_a\n")
    assert runner.read_entries([str(path)]) == ["test_e2e.py::test_a"]


def test_pytest_command_writes_one_list_and_junit_per_batch(tmp_path):
    cmd = runner.pytest_command(
        ["test_e2e.py::test_b", "unittest/x.py -k y"],
        str(tmp_path),
        timeout=60,
        waives=None,
        extra=[],
        name="-iso1",
    )
    assert (
        tmp_path / "test-list-iso1.txt"
    ).read_text() == "test_e2e.py::test_b\nunittest/x.py -k y\n"
    assert f"--junitxml={tmp_path / 'outer-junit-iso1.xml'}" in cmd
    assert cmd[-2:] == ["test_e2e.py", "test_unittests.py"]


def test_environment_puts_only_the_harness_on_pythonpath(tmp_path):
    env = runner.run_env("trtllm", str(tmp_path), {"PYTHONPATH": "/x", "PYTEST_ADDOPTS": "-q"})
    first = env["PYTHONPATH"].split(os.pathsep)[0]
    assert os.listdir(first) == ["openengine_chain"]
    assert env["PYTHONPATH"].endswith(os.pathsep + "/x")
    assert env[cv.ENV_CHAIN] == "trtllm" and env[cv.ENV_RESULTS_DIR] == str(tmp_path)
    assert env["PYTEST_ADDOPTS"] == f"-q -p {cv.PLUGIN_MODULE}"
    probe = (
        "import importlib.util, openengine_chain.launch as l; "
        "print(importlib.util.find_spec('utils') is None, l.__name__)"
    )
    out = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        env={"PYTHONPATH": first, "PATH": os.environ["PATH"]},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    # defs' top-level packages, such as utils, are not importable through it.
    assert out == ["True", "openengine_chain.launch"]
    # A second run reuses the directory.
    assert runner.run_env("none", str(tmp_path), {})["PYTHONPATH"] == first


JUNIT = """<?xml version="1.0"?>
<testsuites><testsuite>
<testcase classname="{cls}" name="{name}" time="1.0">
<properties><property name="chain_proof" value="ok"/></properties>{body}
</testcase></testsuite></testsuites>
"""


def test_summary_reads_every_outer_and_inner_junit_file(tmp_path):
    (tmp_path / "junit").mkdir()
    (tmp_path / "outer-junit.xml").write_text(JUNIT.format(cls="test_e2e", name="a", body=""))
    (tmp_path / "outer-junit-iso1.xml").write_text(
        JUNIT.format(cls="test_e2e", name="b", body='<failure message="boom"/>')
    )
    (tmp_path / "junit" / "inner-1.xml").write_text(JUNIT.format(cls="x", name="c", body=""))
    rows = runner.summarize(str(tmp_path))
    assert [(r["name"], r["outcome"]) for r in rows] == [
        ("test_e2e::a", "passed"),
        ("test_e2e::b", "failed"),
        ("x::c", "passed"),
    ]
    assert json.loads((tmp_path / "summary.json").read_text())[1]["message"] == "boom"
    assert "Counts: failed 1, passed 2" in (tmp_path / "summary.md").read_text()


def test_dry_run_prints_one_command_per_batch(tmp_path, capsys):
    path = tmp_path / "list.txt"
    path.write_text("test_e2e.py::test_a\ntest_e2e.py::test_b ISOLATION\n")
    assert (
        runner.main(
            ["--chain", "trtllm", "--out", str(tmp_path / "out"), "--list", str(path), "--dry-run"]
        )
        == 0
    )
    assert capsys.readouterr().out.count("cmd: ") == 2


def test_llm_root_defaults_to_the_checkout_as_in_ci(tmp_path):
    root = runner.run_env("trtllm", str(tmp_path), {})["LLM_ROOT"]
    assert os.path.isfile(os.path.join(root, "tests", "integration", "defs", "test_e2e.py"))
    assert runner.run_env("trtllm", str(tmp_path), {"LLM_ROOT": "/x"})["LLM_ROOT"] == "/x"


def test_relative_waives_path_is_resolved_against_the_caller(tmp_path, monkeypatch, capsys):
    (tmp_path / "list.txt").write_text("test_e2e.py::test_a\n")
    (tmp_path / "waives.txt").write_text("")
    monkeypatch.chdir(tmp_path)
    args = ["--chain", "trtllm", "--out", "out", "--list", "list.txt", "--waives", "waives.txt"]
    assert runner.main(args + ["--dry-run"]) == 0
    assert f"--waives-file={tmp_path / 'waives.txt'}" in capsys.readouterr().out


def test_out_with_earlier_results_is_refused(tmp_path):
    (tmp_path / "list.txt").write_text("test_e2e.py::test_a\n")
    out = tmp_path / "out"
    out.mkdir()
    (out / "outer-junit.xml").write_text(JUNIT.format(cls="test_e2e", name="a", body=""))
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--chain",
                "trtllm",
                "--out",
                str(out),
                "--list",
                str(tmp_path / "list.txt"),
                "--dry-run",
            ]
        )


def test_rerun_of_an_inner_test_replaces_its_earlier_result(tmp_path):
    (tmp_path / "junit").mkdir()

    def inner(name, outer, cls, test, body):
        prop = f'<property name="outer_test" value="{outer}"/>'
        text = JUNIT.format(cls=cls, name=test, body=body).replace(
            "<properties>", f"<properties>{prop}"
        )
        (tmp_path / "junit" / name).write_text(text)

    inner("inner-20260101T000001-1-w.xml", "w[a -k x]", "m", "t", '<failure message="flaky"/>')
    inner("inner-20260101T000009-2-w.xml", "w[a -k x]", "m", "t", "")
    inner("inner-20260101T000010-3-v.xml", "v", "m", "t", '<failure message="real"/>')
    rows = runner.summarize(str(tmp_path))
    assert [(r["outer_test"], r["outcome"]) for r in rows] == [
        ("w[a -k x]", "passed"),
        ("v", "failed"),
    ]

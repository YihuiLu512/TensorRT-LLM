# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capprobe.patchcheck: passes on a good patch, goes red on each kind of sabotage, NOT_RUN when
nothing is expected; plus the real probe tree's working tree."""

import os
import subprocess
import sys

import pytest
from probe_testutil import TOOLS, clean_env
from gitfix import BASE_A, PATCHED_A, TEST_SITES, commit_all, git, init, write

from capprobe import patchcheck

PROBE_TREE = os.path.abspath(os.path.join(TOOLS, "..", "probe-tree"))
BASE_COMMIT = "2f22a98d6e8b32ed185bcdfe051ef0e4405ae4e8"


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    init(r)
    write(r, "tensorrt_llm/a.py", BASE_A)
    write(r, "tensorrt_llm/b.py", "X = 1\n")
    write(r, "docs/readme.md", "doc\n")
    base = commit_all(r, "base")
    return r, base


def check(repo, base, text, rev=None, expected=TEST_SITES):
    write(repo, "tensorrt_llm/a.py", text)
    if rev == "commit":
        rev = commit_all(repo, "patch")
    return patchcheck.run_check(str(repo), base, rev, expected_sites=expected)


def test_good_patch_worktree_and_commit(repo):
    r, base = repo
    rep = check(r, base, PATCHED_A)
    assert rep["status"] == "PASS", rep["violations"]
    assert rep["found_site_count"] == rep["expected_site_count"] == 1
    rep = check(r, base, PATCHED_A, rev="commit")
    assert rep["status"] == "PASS" and rep["rev"] != "WORKTREE"


SABOTAGE = {
    "unguarded_call": PATCHED_A.replace("    if _capprobe.ON_F2:\n        _capprobe.F2()\n",
                                        "    _capprobe.F2()\n"),
    "wrong_guard": PATCHED_A.replace("ON_F2", "ON_H1"),
    "else_branch": PATCHED_A.replace("        _capprobe.F2()\n",
                                     "        _capprobe.F2()\n    else:\n        pass\n"),
    "two_stmt_body": PATCHED_A.replace("        _capprobe.F2()\n",
                                       "        _capprobe.F2()\n        _capprobe.F2()\n"),
    "stray_added_line": PATCHED_A.replace("    return y\n", "    y = y * 1\n    return y\n"),
    "deleted_line": PATCHED_A.replace("    y = x + 1\n", "    y = (x\n         + 1)\n"),
    "no_import": PATCHED_A.replace("import capprobe as _capprobe\n", ""),
    "guard_reads_other": PATCHED_A.replace("if _capprobe.ON_F2:", "if _capprobe.ON_F2 or True:"),
    "alias_elsewhere": PATCHED_A.replace("    return y\n", "    return y, _capprobe.COUNT\n"),
    "site_moved": PATCHED_A.replace("    y = x + 1\n    if", "    y = x + 1\n\n    if"),
    "syntax_error": PATCHED_A + "def (\n",
    "guard_second_operand_alias": PATCHED_A.replace("if _capprobe.ON_F2:",
                                                    "if _capprobe.ON_F2 and _capprobe.COUNT:"),
    "call_with_alias_arg": PATCHED_A.replace("_capprobe.F2()", "_capprobe.F2(_capprobe)"),
}


@pytest.mark.parametrize("name", sorted(SABOTAGE))
def test_sabotage_goes_red(repo, name):
    r, base = repo
    text = SABOTAGE[name]
    assert text != PATCHED_A  # positive control: the sabotage edit really changed something
    rep = check(r, base, text)
    assert rep["status"] == "FAIL", (name, rep)
    assert rep["violations"]


def test_other_file_types_and_new_files_are_red(repo):
    r, base = repo
    write(r, "tensorrt_llm/a.py", PATCHED_A)
    write(r, "docs/readme.md", "changed\n")
    rep = patchcheck.run_check(str(r), base, None, expected_sites=TEST_SITES)
    assert rep["status"] == "FAIL" and any("outside tensorrt_llm" in v for v in rep["violations"])
    git(r, "checkout", "--", "docs/readme.md")
    write(r, "tensorrt_llm/new.py", "Y = 2\n")
    rep = patchcheck.run_check(str(r), base, None, expected_sites=TEST_SITES)
    assert rep["status"] == "FAIL" and any("untracked" in v for v in rep["violations"])


def test_undeclared_site_is_red(repo):
    r, base = repo
    rep = check(r, base, PATCHED_A, expected={})
    assert rep["status"] == "NOT_RUN"  # nothing expected -> never PASS
    two = PATCHED_A.replace("    return y\n", "    if _capprobe.ON_F4:\n        _capprobe.F4()\n"
                            "    return y\n")
    rep = check(r, base, two)
    assert rep["status"] == "FAIL" and any("not declared" in v for v in rep["violations"])


def test_missing_site_is_red(repo):
    r, base = repo
    expected = dict(TEST_SITES, H1=dict(TEST_SITES["F2"], line=99, slots=["count.H1.arrivals"]))
    rep = check(r, base, PATCHED_A, expected=expected)
    assert rep["status"] == "FAIL" and any("H1" in v and "not found" in v
                                           for v in rep["violations"])


@pytest.mark.skipif(not os.path.isdir(PROBE_TREE), reason="probe tree not present")
def test_real_probe_tree_cli():
    # The real patch against the real SITES table (working tree, i.e. what would be deployed).
    r = subprocess.run([sys.executable, "-m", "capprobe.patchcheck", "--probe-tree", PROBE_TREE,
                        "--base-commit", BASE_COMMIT], env=clean_env(), capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr
    assert '"status": "PASS"' in r.stdout and '"found_site_count": 7' in r.stdout

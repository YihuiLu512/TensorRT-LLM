# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capprobe.patchcheck: passes on a good patch, goes red on each kind of sabotage, NOT_RUN when
nothing is expected; a site may sit at several code locations; plus the real probe tree's
working tree."""

import json
import subprocess
import sys

import pytest
from probe_testutil import clean_env, require_probe_tree
from gitfix import BASE_A, PATCHED_A, SITE_LINE, TEST_SITES, commit_all, git, init, write

from capprobe import patchcheck
from capprobe import sites as capsites

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
    expected = dict(TEST_SITES, H1=dict(TEST_SITES["F2"], lines=[99],
                                        slots=["count.H1.arrivals"]))
    rep = check(r, base, PATCHED_A, expected=expected)
    assert rep["status"] == "FAIL" and any("H1" in v and "not found" in v
                                           for v in rep["violations"])


# A site whose event is handled on two code paths: the same guarded call at both.
TWO_PATHS_A = PATCHED_A + ("\n\ndef feed(x):\n    if _capprobe.ON_F2:\n        _capprobe.F2()\n"
                           "    return x\n")
BASE_TWO = BASE_A + "\n\ndef feed(x):\n    return x\n"
SECOND_LINE = TWO_PATHS_A.splitlines().index("    if _capprobe.ON_F2:", SITE_LINE) + 1


def _two_path_repo(tmp_path):
    r = tmp_path / "repo2"
    init(r)
    write(r, "tensorrt_llm/a.py", BASE_TWO)
    return r, commit_all(r, "base")


def test_site_at_two_locations(tmp_path):
    r, base = _two_path_repo(tmp_path)
    both = {"F2": dict(TEST_SITES["F2"], lines=[SITE_LINE, SECOND_LINE])}
    rep = check(r, base, TWO_PATHS_A, expected=both)
    assert rep["status"] == "PASS", rep["violations"]
    assert rep["found_site_count"] == rep["expected_site_count"] == 1
    assert rep["found_location_count"] == rep["expected_location_count"] == 2
    assert rep["changed_files"]["tensorrt_llm/a.py"]["sites"] == {"F2": [SITE_LINE, SECOND_LINE]}


@pytest.mark.parametrize("lines, why", [
    ([SITE_LINE], "second location not declared"),
    ([SITE_LINE, SECOND_LINE, SECOND_LINE + 5], "declared location missing"),
    ([SECOND_LINE, SITE_LINE], "not ascending"),
    ([], "empty"),
])
def test_site_locations_must_match_exactly(tmp_path, lines, why):
    r, base = _two_path_repo(tmp_path)
    rep = check(r, base, TWO_PATHS_A, expected={"F2": dict(TEST_SITES["F2"], lines=lines)})
    assert rep["status"] == "FAIL" and rep["violations"], (why, rep)


def test_real_probe_tree_cli():
    # The real patch against the real SITES table (working tree, i.e. what would be deployed).
    tree = require_probe_tree()
    r = subprocess.run([sys.executable, "-m", "capprobe.patchcheck", "--probe-tree", tree,
                        "--base-commit", BASE_COMMIT], env=clean_env(), capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr
    rep = json.loads(r.stdout)
    assert rep["status"] == "PASS" and rep["rev"] == "WORKTREE"
    assert rep["found_site_count"] == len(capsites.SITES)
    assert rep["found_location_count"] == sum(len(d["lines"]) for d in capsites.SITES.values())

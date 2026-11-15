# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of the exception check_ownership.py keeps for WT-55-fix1 in the harness's tests.

WT-55-fix1 registers NT-091 as a test that cancels calls. The exception allows
WT-55-fix1 alone to add one line, once, to the list in
``test_shipped_tables_name_each_test_that_cancels_calls`` of
``test_expectations.py``, a file of the harness: the match of NT-091
(``<test file name>::<test name>``). It allows WT-55-fix1 nothing else in the
file, and nothing at all once WT-55-fix1 has landed. Run from the repository
root::

    python3 -m pytest -p no:cacheprovider scripts/poc3/
"""

from __future__ import annotations

import ast
import re
import subprocess
from functools import lru_cache
from pathlib import Path

# The scripts import poc3_common themselves; pytest puts this directory on sys.path.
import check_ownership  # noqa: I001
import pytest

ITEM = "WT-55-fix1"
PATH = "tests/integration/defs/openengine_chain/tests/test_expectations.py"
TEST = "test_shipped_tables_name_each_test_that_cancels_calls"
# A stage03 commit with PATH as it is before WT-55-fix1's change, and the git blob id of PATH
# with the line added after ANCHOR, as WT-55-fix1 changes it.
BASE = "0f71f2e5bd5e2e32234cffc73a8c5ccbe26dd885"
HEAD_BLOB = "aff399791e41f257316291b5bdc20072f82bd9c2"
# The last entry of TEST's list at BASE, and the line that goes after it.
ANCHOR = '        "_test_openai_misc.py::test_request_cancellation[*",'
ADDED = ['        "_test_openengine_disagg_gen_first.py::test_gen_first_cancel_closes_both_legs",']


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@lru_cache(maxsize=None)
def _show(rev: str) -> str:
    """PATH as at ``rev`` of this checkout; skips the test when the checkout lacks ``rev``."""
    top = Path(__file__).resolve().parents[2]
    known = subprocess.run(["git", "-C", str(top), "cat-file", "-e", f"{rev}^{{commit}}"])
    if known.returncode != 0:
        pytest.skip(f"{rev} is not in this checkout")
    return _git(top, "show", f"{rev}:{PATH}")


def _repo_with(
    tmp_path: Path, base: str, head: str, task: str, landed: str = ""
) -> tuple[Path, str]:
    """A repo holding PATH as ``base``, then as ``head`` in a commit of ``task``.

    With ``landed``, the base commit carries that work item's trailer.
    """
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    file = repo / PATH
    file.parent.mkdir(parents=True)
    (repo / "tensorrt_llm").mkdir(exist_ok=True)
    (repo / "tensorrt_llm/__init__.py").write_text("")
    file.write_text(base)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"base\n\nPoC3-Task: {landed}" if landed else "base")
    base_sha = _git(repo, "rev-parse", "HEAD").strip()
    file.write_text(head)
    _git(repo, "commit", "-q", "-am", f"t\n\nPoC3-Task: {task}")
    return repo, base_sha


def _after(text: str, anchor: str, added: list[str]) -> str:
    """``text`` with ``added`` right after its one line ``anchor``."""
    lines = text.split("\n")
    at = [i for i, line in enumerate(lines) if line == anchor]
    assert len(at) == 1, (anchor, at)
    return "\n".join(lines[: at[0] + 1] + added + lines[at[0] + 1 :])


def _check(tmp_path: Path, capsys, base: str, head: str, task: str = ITEM, landed: str = ""):
    """Check ``task`` changing PATH from ``base`` to ``head``: the exit status and the output."""
    repo, sha = _repo_with(tmp_path, base, head, task, landed)
    found = check_ownership.check(task, f"{sha}..HEAD", str(repo))
    return found, capsys.readouterr().out


def test_the_exception_lists_exactly_the_line():
    rule = check_ownership.match_rule(ITEM, PATH)
    assert rule.kind == "exception" and not rule.lines and not rule.renames
    assert rule.item == ITEM
    assert list(rule.changes) == [
        check_ownership.Change((), tuple(map(re.escape, ADDED)), TEST),
    ]
    assert [(n, r.kind) for n, r in check_ownership.owners(PATH)] == [
        ("WT-38", "shared"),
        ("WT-62", "exception"),
        (ITEM, "exception"),
        ("WT-57-fix1", "exception"),
    ]


def test_the_line_is_the_change_of_the_reference_patch():
    head = _after(_show(BASE), ANCHOR, ADDED)
    found = subprocess.run(
        ["git", "hash-object", "--stdin"], input=head, check=True, capture_output=True, text=True
    )
    assert found.stdout.strip() == HEAD_BLOB


def test_wt55_fix1_may_add_the_line(tmp_path, capsys):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED))
    assert found == 0, out
    assert f"CHECK BY HAND  EXCEPTION  {PATH}" in out


# Work items that own nothing in PATH: the item WT-55-fix1 fixes and that item's other fixes.
OTHERS = ["WT-55", "WT-55-fix2", "WT-55-fix3"]


@pytest.mark.parametrize("task", OTHERS)
def test_no_other_work_item_may_add_the_line(tmp_path, capsys, task):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED), task)
    assert (
        found == 1 and f"OUT OF BOUNDS  {PATH}  (owned by WT-38, WT-62, {ITEM}, WT-57-fix1)" in out
    ), out


# The ways the line can differ from the one allowed: a broader match, another test of the same
# file, the comma left out, other indentation, a trailing space, single quotes, the line twice,
# or another line in its place.
TAMPERS = {
    "pattern-broadened": [
        '        "_test_openengine_disagg_gen_first.py::test_gen_first_cancel_closes_both_legs*",'
    ],
    "other-test": [
        '        "_test_openengine_disagg_gen_first.py::'
        'test_pipelined_transfer_with_chunked_prefill",'
    ],
    "comma-dropped": [ADDED[0].rstrip(",")],
    "indent-changed": ["    " + ADDED[0]],
    "trailing-space": [ADDED[0] + " "],
    "single-quotes": [ADDED[0].replace('"', "'")],
    "line-repeated": [*ADDED, *ADDED],
    "other-line": ['        "_test_openengine_disagg_cancel.py::*",'],
}


@pytest.mark.parametrize("tamper", TAMPERS)
def test_wt55_fix1_may_not_add_another_line(tmp_path, capsys, tamper):
    base = _show(BASE)
    head = _after(base, ANCHOR, TAMPERS[tamper])
    ast.parse(head)  # each tampered file is still Python, so the hunk itself is refused
    found, out = _check(tmp_path, capsys, base, head)
    assert found == 1, out
    assert f"OUT OF BOUNDS  {PATH}  (the exception allows" in out
    assert "not one of the allowed changes" in out


# Other changes to the file, each made along with the line: (the line; what replaces it).
OTHER_CHANGES = {
    # An entry of the list dropped, or altered.
    "entry-dropped": (ANCHOR, None),
    "entry-altered": (
        '        "_test_openai_kv_cache_contamination.py::'
        'test_no_kv_cache_contamination_number_set_probe*",',
        '        "_test_openai_kv_cache_contamination.py::'
        'test_no_kv_cache_contamination_number_set_probe",',
    ),
    # Another line of the same test changed.
    "comment-altered": (
        "    # The relaxed count check must not reach other tests through a broad pattern.",
        "    # The tests that cancel calls.",
    ),
    "assertion-altered": (
        "    matches = sorted(e.match for e in ex.load() if e.kind == ex.CANCELS_CALLS)",
        "    matches = sorted(e.match for e in ex.load())",
    ),
    # A line of another test dropped.
    "other-test-changed": (
        '        "_test_openai_misc.py::test_version[pytorch-with_beam_search-8-32768]",',
        None,
    ),
}


@pytest.mark.parametrize("other", OTHER_CHANGES)
def test_wt55_fix1_may_change_nothing_else_in_the_file(tmp_path, capsys, other):
    base = _show(BASE)
    old, new = OTHER_CHANGES[other]
    lines = _after(base, ANCHOR, ADDED).split("\n")
    assert lines.count(old) == 1, old
    at = lines.index(old)
    lines[at : at + 1] = [] if new is None else [new]
    found, out = _check(tmp_path, capsys, base, "\n".join(lines))
    assert found == 1, out
    assert f"OUT OF BOUNDS  {PATH}  (the exception allows" in out
    assert "not one of the allowed changes" in out


def test_wt55_fix1_may_not_add_the_line_to_another_test(tmp_path, capsys):
    base = _show(BASE)
    other = '        "_test_openai_misc.py::test_version[pytorch-with_beam_search-8-32768]",'
    found, out = _check(tmp_path, capsys, base, _after(base, other, ADDED))
    assert found == 1, out
    assert f"OUT OF BOUNDS  {PATH}  (the exception allows" in out
    assert "not one of the allowed changes" in out


def test_wt55_fix1_may_add_the_line_once(tmp_path, capsys):
    base = _show(BASE)
    head = _after(base, ANCHOR, ADDED)
    # A second copy earlier in the same list.
    earlier = (
        '        "_test_openai_kv_cache_contamination.py::'
        'test_no_kv_cache_contamination_number_set_probe*",'
    )
    found, out = _check(tmp_path, capsys, base, _after(head, earlier, ADDED))
    assert found == 1, out
    assert f"OUT OF BOUNDS  {PATH}  (the exception allows" in out
    assert "not one of the allowed changes" in out


# Work items whose trailers at the base leave the exception in force: those of OTHERS and WT-38,
# which shares PATH.
@pytest.mark.parametrize("landed", [*OTHERS, "WT-38"])
def test_the_exception_is_not_spent_by_another_work_item(tmp_path, capsys, landed):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED), landed=landed)
    assert found == 0, out


def test_the_exception_allows_nothing_once_wt55_fix1_has_landed(tmp_path, capsys):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED), landed=ITEM)
    assert found == 1, out
    assert f"OUT OF BOUNDS  {PATH}  (the exception for {ITEM} is spent" in out


@pytest.mark.parametrize("task", [ITEM, *OTHERS])
def test_the_line_cannot_be_added_again_once_wt55_fix1_has_landed(tmp_path, capsys, task):
    landed = _after(_show(BASE), ANCHOR, ADDED)  # the file once WT-55-fix1 has landed
    again = _after(landed, ADDED[0], ADDED)  # the same line again, right after itself
    found, out = _check(tmp_path, capsys, landed, again, task, landed=ITEM)
    assert found == 1, out
    if task == ITEM:
        assert f"OUT OF BOUNDS  {PATH}  (the exception for {ITEM} is spent" in out
    else:
        assert f"OUT OF BOUNDS  {PATH}  (owned by WT-38, WT-62, {ITEM}, WT-57-fix1)" in out

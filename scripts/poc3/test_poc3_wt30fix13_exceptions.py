# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of the exception check_ownership.py keeps for WT-62 in the harness's tests.

WT-62 registers NT-132 and NT-133 as tests that cancel calls. The exception
allows WT-62 alone to add two lines to the list in
``test_shipped_tables_name_each_test_that_cancels_calls`` of
``test_expectations.py``, a file of the harness: the matches of NT-132 and
NT-133 (``<test file name>::<test name>``, with ``[*`` after the name of
NT-132, a parametrized test), in that order, next to each other and once.
Where in that test they go is not checked. It allows WT-62 nothing else in
the file, and nothing at all once WT-62 has landed. The exception for
WT-55-fix1 in the same list allows WT-55-fix1 only its own line. Run from the
repository root::

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

ITEM = "WT-62"
PATH = "tests/integration/defs/openengine_chain/tests/test_expectations.py"
TEST = "test_shipped_tables_name_each_test_that_cancels_calls"
# A stage03 commit with PATH as it is before WT-62's change, and the git blob id of PATH with the
# lines added after ANCHOR, as WT-62 changes it.
BASE = "78fc298206b2bf98e20d34641f796759a15325c2"
HEAD_BLOB = "e1f9c356f9ef9c7bc245d7e89e39eb01e294d56d"
# The last entry of TEST's list at BASE, and the lines that go after it.
ANCHOR = '        "_test_openai_misc.py::test_request_cancellation[*",'
# The first entry of TEST's list, and a line of another test of the same file.
FIRST = (
    '        "_test_openai_kv_cache_contamination.py::'
    'test_no_kv_cache_contamination_number_set_probe*",'
)
OTHER_TEST_LINE = '        "_test_openai_misc.py::test_version[pytorch-with_beam_search-8-32768]",'
ADDED = [
    '        "_test_openengine_ds_disagg_cancel.py::test_abort_stages[*",',
    '        "_test_openengine_ds_disagg_cancel.py::test_parked_kv_timeout_and_orphans",',
]
# The line the exception for WT-55-fix1 allows. Sorted, it comes before WT-62's lines.
FIX = "WT-55-fix1"
FIX_ADDED = [
    '        "_test_openengine_disagg_gen_first.py::test_gen_first_cancel_closes_both_legs",'
]


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


def _refused(found: int, out: str) -> None:
    """The check refused a hunk of PATH under the exception the work item has."""
    assert found == 1, out
    assert f"OUT OF BOUNDS  {PATH}  (the exception allows" in out
    assert "not one of the allowed changes" in out


def test_the_exception_lists_exactly_the_lines():
    rule = check_ownership.match_rule(ITEM, PATH)
    assert rule.kind == "exception" and not rule.lines and not rule.renames
    assert rule.item == ITEM
    assert list(rule.changes) == [
        check_ownership.Change((), tuple(map(re.escape, ADDED)), TEST),
    ]
    assert [(n, r.kind) for n, r in check_ownership.owners(PATH)] == [
        ("WT-38", "shared"),
        (ITEM, "exception"),
        (FIX, "exception"),
        ("WT-57-fix1", "exception"),
    ]


def test_the_exception_for_wt55_fix1_lists_exactly_its_line():
    rule = check_ownership.match_rule(FIX, PATH)
    assert rule.kind == "exception" and rule.item == FIX
    assert list(rule.changes) == [
        check_ownership.Change((), tuple(map(re.escape, FIX_ADDED)), TEST),
    ]


def test_the_lines_are_the_change_of_the_reference_patch():
    head = _after(_show(BASE), ANCHOR, ADDED)
    found = subprocess.run(
        ["git", "hash-object", "--stdin"], input=head, check=True, capture_output=True, text=True
    )
    assert found.stdout.strip() == HEAD_BLOB


def test_wt62_may_add_the_lines(tmp_path, capsys):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED))
    assert found == 0, out
    assert f"CHECK BY HAND  EXCEPTION  {PATH}" in out


# WT-55-fix1's line in the list at the base, the base carrying WT-55-fix1's trailer or not.
@pytest.mark.parametrize("landed", ["", FIX])
def test_wt62_may_add_the_lines_after_the_line_of_wt55_fix1(tmp_path, capsys, landed):
    base = _after(_show(BASE), ANCHOR, FIX_ADDED)
    found, out = _check(tmp_path, capsys, base, _after(base, FIX_ADDED[0], ADDED), landed=landed)
    assert found == 0, out


def test_wt55_fix1_may_add_its_line(tmp_path, capsys):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, FIX_ADDED), FIX)
    assert found == 0, out


def test_wt55_fix1_may_add_its_line_before_the_lines_of_wt62(tmp_path, capsys):
    base = _after(_show(BASE), ANCHOR, ADDED)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, FIX_ADDED), FIX, ITEM)
    assert found == 0, out


# The exceptions for WT-62 and WT-55-fix1 allow each its own work item its own lines only: neither
# may add the other's lines, nor both sets at once.
@pytest.mark.parametrize(
    "task, added",
    [
        (ITEM, FIX_ADDED),
        (ITEM, [*FIX_ADDED, *ADDED]),
        (ITEM, [*ADDED, *FIX_ADDED]),
        (FIX, ADDED),
        (FIX, [*FIX_ADDED, *ADDED]),
        (FIX, [*ADDED, *FIX_ADDED]),
    ],
    ids=[
        "wt62-fix-line",
        "wt62-both-fix-first",
        "wt62-both-own-first",
        "fix-wt62-lines",
        "fix-both-own-first",
        "fix-both-wt62-first",
    ],
)
def test_neither_work_item_may_add_the_lines_of_the_other(tmp_path, capsys, task, added):
    base = _show(BASE)
    _refused(*_check(tmp_path, capsys, base, _after(base, ANCHOR, added), task))


# Work items no rule allows to change PATH: WT-62's fixes, the item WT-55-fix1 fixes and another
# fix of it.
OTHERS = ["WT-62-fix1", "WT-62-fix2", "WT-62-fix3", "WT-55", "WT-55-fix2"]


@pytest.mark.parametrize("task", OTHERS)
def test_no_other_work_item_may_add_the_lines(tmp_path, capsys, task):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED), task)
    assert (
        found == 1 and f"OUT OF BOUNDS  {PATH}  (owned by WT-38, {ITEM}, {FIX}, WT-57-fix1)" in out
    ), out


# The ways the lines can differ from the ones allowed: one of them alone, the two swapped, a
# broader or narrower match (a bare ``*`` after the name of NT-132 is broader: it also matches
# other tests whose names start with that name), another test of the same file, the comma left
# out, other indentation, a trailing space, single quotes, the lines twice, a third line after
# them, or other lines in their place.
TAMPERS = {
    "first-only": ADDED[:1],
    "second-only": ADDED[1:],
    "swapped": ADDED[::-1],
    "pattern-broadened": [ADDED[0], ADDED[1].replace('orphans",', 'orphans*",')],
    "bare-star": [ADDED[0].replace('stages[*",', 'stages*",'), ADDED[1]],
    "pattern-narrowed": [ADDED[0].replace('stages[*",', 'stages",'), ADDED[1]],
    "other-test": [
        ADDED[0],
        ADDED[1].replace("test_parked_kv_timeout_and_orphans", "test_engine_loss"),
    ],
    "comma-dropped": [ADDED[0], ADDED[1].rstrip(",")],
    "indent-changed": ["    " + line for line in ADDED],
    "trailing-space": [ADDED[0] + " ", ADDED[1]],
    "single-quotes": [line.replace('"', "'") for line in ADDED],
    "lines-repeated": [*ADDED, *ADDED],
    "line-added": [*ADDED, '        "_test_openengine_ds_disagg_cancel.py::*",'],
    "other-lines": ['        "_test_openengine_ds_disagg_cancel.py::*",'] * 2,
}


@pytest.mark.parametrize("tamper", TAMPERS)
def test_wt62_may_not_add_other_lines(tmp_path, capsys, tamper):
    base = _show(BASE)
    head = _after(base, ANCHOR, TAMPERS[tamper])
    ast.parse(head)  # each tampered file is still Python, so the hunk itself is refused
    _refused(*_check(tmp_path, capsys, base, head))


def test_wt62_may_not_add_the_lines_apart(tmp_path, capsys):
    base = _show(BASE)
    head = _after(_after(base, FIRST, ADDED[:1]), ANCHOR, ADDED[1:])
    _refused(*_check(tmp_path, capsys, base, head))


# Other changes to the file, each made along with the lines: (the line; what replaces it).
OTHER_CHANGES = {
    # An entry of the list dropped, or altered.
    "entry-dropped": (ANCHOR, None),
    "entry-altered": (FIRST, FIRST.replace('probe*",', 'probe",')),
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
    "other-test-changed": (OTHER_TEST_LINE, None),
}


@pytest.mark.parametrize("other", OTHER_CHANGES)
def test_wt62_may_change_nothing_else_in_the_file(tmp_path, capsys, other):
    base = _show(BASE)
    old, new = OTHER_CHANGES[other]
    lines = _after(base, ANCHOR, ADDED).split("\n")
    assert lines.count(old) == 1, old
    at = lines.index(old)
    lines[at : at + 1] = [] if new is None else [new]
    _refused(*_check(tmp_path, capsys, base, "\n".join(lines)))


def test_wt62_may_not_add_the_lines_to_another_test(tmp_path, capsys):
    base = _show(BASE)
    _refused(*_check(tmp_path, capsys, base, _after(base, OTHER_TEST_LINE, ADDED)))


def test_wt62_may_add_the_lines_once(tmp_path, capsys):
    base = _show(BASE)
    head = _after(base, ANCHOR, ADDED)
    # A second copy earlier in the same list.
    _refused(*_check(tmp_path, capsys, base, _after(head, FIRST, ADDED)))


# Work items whose trailers at the base leave the exception in force: those of OTHERS, WT-55-fix1
# and WT-38, which shares PATH.
@pytest.mark.parametrize("landed", [*OTHERS, FIX, "WT-38"])
def test_the_exception_is_not_spent_by_another_work_item(tmp_path, capsys, landed):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED), landed=landed)
    assert found == 0, out


def test_the_exception_allows_nothing_once_wt62_has_landed(tmp_path, capsys):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED), landed=ITEM)
    assert found == 1, out
    assert f"OUT OF BOUNDS  {PATH}  (the exception for {ITEM} is spent" in out


@pytest.mark.parametrize("task", [ITEM, *OTHERS])
def test_the_lines_cannot_be_added_again_once_wt62_has_landed(tmp_path, capsys, task):
    landed = _after(_show(BASE), ANCHOR, ADDED)  # the file once WT-62 has landed
    again = _after(landed, ADDED[1], ADDED)  # the same lines again, right after themselves
    found, out = _check(tmp_path, capsys, landed, again, task, landed=ITEM)
    assert found == 1, out
    if task == ITEM:
        assert f"OUT OF BOUNDS  {PATH}  (the exception for {ITEM} is spent" in out
    else:
        assert f"OUT OF BOUNDS  {PATH}  (owned by WT-38, {ITEM}, {FIX}, WT-57-fix1)" in out

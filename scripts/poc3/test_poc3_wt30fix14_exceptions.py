# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of the exception check_ownership.py keeps for WT-57-fix1 in the harness's tests.

WT-57-fix1 registers NT-199 as a test that cancels calls. The exception
allows WT-57-fix1 alone to add one line, once, to the list in
``test_shipped_tables_name_each_test_that_cancels_calls`` of
``test_expectations.py``, a file of the harness: the match of NT-199
(``<test file name>::<test name>``). Where in that test it goes is not
checked. It allows WT-57-fix1 nothing else in the file, and nothing at all
once WT-57-fix1 has landed. The exceptions for WT-55-fix1 and WT-62 in the
same list allow each of them only its own lines. Run from the repository
root::

    python3 -m pytest -p no:cacheprovider scripts/poc3/
"""

from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

# The scripts import poc3_common themselves; pytest puts this directory on sys.path.
import check_ownership  # noqa: I001
import pytest

ITEM = "WT-57-fix1"
PATH = "tests/integration/defs/openengine_chain/tests/test_expectations.py"
TEST = "test_shipped_tables_name_each_test_that_cancels_calls"
# A stage03 commit with PATH as it is before WT-57-fix1's change, and the git blob id of PATH with
# the line added after ANCHOR, as WT-57-fix1 changes it.
BASE = "239bbeca6394a0145919fdb86a6db186032e2d3e"
HEAD_BLOB = "7caa151602c1efb6362633b95c27fba9aefe7c5f"
# The last entry of TEST's list at BASE (the line WT-55-fix1 added), and the line that goes after
# it.
ANCHOR = '        "_test_openengine_disagg_gen_first.py::test_gen_first_cancel_closes_both_legs",'
ADDED = [
    '        "_test_openengine_disagg_transceiver.py::'
    'test_cancel_during_transfer_is_the_same_on_both_forms",'
]
# The first entry of TEST's list, and a line of another test of the same file.
FIRST = (
    '        "_test_openai_kv_cache_contamination.py::'
    'test_no_kv_cache_contamination_number_set_probe*",'
)
OTHER_TEST_LINE = '        "_test_openai_misc.py::test_version[pytorch-with_beam_search-8-32768]",'
# The lines the exceptions for WT-55-fix1 and WT-62 allow. Sorted, WT-55-fix1's line comes before
# WT-57-fix1's and WT-62's lines after it, so whichever of WT-57-fix1 and WT-62 lands second puts
# its lines next to those of the other.
WT55_FIX1 = "WT-55-fix1"
WT55_FIX1_ADDED = [ANCHOR]
WT62 = "WT-62"
WT62_ADDED = [
    '        "_test_openengine_ds_disagg_cancel.py::test_abort_stages[*",',
    '        "_test_openengine_ds_disagg_cancel.py::test_parked_kv_timeout_and_orphans",',
]
OWNED = f"OUT OF BOUNDS  {PATH}  (owned by WT-38, {WT62}, {WT55_FIX1}, {ITEM})"


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


# --- the exception and the reference patch ------------------------------------------------------


def test_the_exception_lists_exactly_the_line():
    rule = check_ownership.match_rule(ITEM, PATH)
    assert rule.kind == "exception" and not rule.lines and not rule.renames
    assert rule.item == ITEM
    assert list(rule.changes) == [
        check_ownership.Change((), tuple(map(re.escape, ADDED)), TEST),
    ]
    assert [(n, r.kind) for n, r in check_ownership.owners(PATH)] == [
        ("WT-38", "shared"),
        (WT62, "exception"),
        (WT55_FIX1, "exception"),
        (ITEM, "exception"),
    ]


@pytest.mark.parametrize("task, added", [(WT55_FIX1, WT55_FIX1_ADDED), (WT62, WT62_ADDED)])
def test_the_exceptions_for_wt55_fix1_and_wt62_list_exactly_their_lines(task, added):
    rule = check_ownership.match_rule(task, PATH)
    assert rule.kind == "exception" and rule.item == task
    assert list(rule.changes) == [
        check_ownership.Change((), tuple(map(re.escape, added)), TEST),
    ]


def test_the_line_is_the_change_of_the_reference_patch():
    head = _after(_show(BASE), ANCHOR, ADDED)
    found = subprocess.run(
        ["git", "hash-object", "--stdin"], input=head, check=True, capture_output=True, text=True
    )
    assert found.stdout.strip() == HEAD_BLOB


# --- what the exception allows ------------------------------------------------------------------


def test_wt57_fix1_may_add_the_line(tmp_path, capsys):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED))
    assert found == 0, out
    assert f"CHECK BY HAND  EXCEPTION  {PATH}" in out


# WT-62's lines in the list at the base (WT-62 landed first), the base carrying WT-62's trailer or
# not: WT-57-fix1's line goes before them.
@pytest.mark.parametrize("landed", ["", WT62], ids=["unlanded", "landed"])
def test_wt57_fix1_may_add_the_line_before_the_lines_of_wt62(tmp_path, capsys, landed):
    base = _after(_show(BASE), ANCHOR, WT62_ADDED)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED), landed=landed)
    assert found == 0, out


# WT-57-fix1's line in the list at the base (WT-57-fix1 landed first), the base carrying its
# trailer or not: WT-62's lines go after it.
@pytest.mark.parametrize("landed", ["", ITEM], ids=["unlanded", "landed"])
def test_wt62_may_add_its_lines_after_the_line_of_wt57_fix1(tmp_path, capsys, landed):
    base = _after(_show(BASE), ANCHOR, ADDED)
    head = _after(base, ADDED[0], WT62_ADDED)
    found, out = _check(tmp_path, capsys, base, head, WT62, landed)
    assert found == 0, out


# --- what it does not allow ---------------------------------------------------------------------

# Each exception allows its own work item its own lines only: no work item among WT-57-fix1,
# WT-55-fix1 and WT-62 may add the lines of another, nor its own along with another's. WT-55-fix1's
# line is in the list at BASE already, so adding it is adding a second copy.
CROSS = {
    "wt57-fix1-adds-wt62-lines": (ITEM, WT62_ADDED),
    "wt57-fix1-adds-wt55-fix1-line": (ITEM, WT55_FIX1_ADDED),
    "wt57-fix1-adds-own-and-wt62-lines": (ITEM, [*ADDED, *WT62_ADDED]),
    "wt62-adds-wt57-fix1-line": (WT62, ADDED),
    "wt62-adds-wt57-fix1-and-own-lines": (WT62, [*ADDED, *WT62_ADDED]),
    "wt55-fix1-adds-wt57-fix1-line": (WT55_FIX1, ADDED),
    "wt55-fix1-adds-own-and-wt57-fix1-lines": (WT55_FIX1, [*WT55_FIX1_ADDED, *ADDED]),
}


@pytest.mark.parametrize("case", CROSS)
def test_no_work_item_may_add_the_lines_of_another(tmp_path, capsys, case):
    task, added = CROSS[case]
    base = _show(BASE)
    _refused(*_check(tmp_path, capsys, base, _after(base, ANCHOR, added), task))


# Work items no rule allows to change PATH: the item WT-57-fix1 fixes and that item's other fixes.
OTHERS = ["WT-57", "WT-57-fix2", "WT-57-fix3"]


@pytest.mark.parametrize("task", OTHERS)
def test_no_other_work_item_may_add_the_line(tmp_path, capsys, task):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED), task)
    assert found == 1 and OWNED in out, out


# The ways the line can differ from the one allowed: another match of the same test (``*`` after
# the name matches more tests, ``[*`` matches only parametrized ones), another test of the same
# test file, another test file, the comma left out, other indentation, a trailing space, single
# quotes, the line twice, a second line after it, or another line in its place.
TAMPERS = {
    "pattern-broadened": [ADDED[0].replace('forms",', 'forms*",')],
    "pattern-parametrized": [ADDED[0].replace('forms",', 'forms[*",')],
    "other-test": [ADDED[0].replace("test_cancel_during_transfer", "test_abort_during_transfer")],
    "other-file": [ADDED[0].replace("_transceiver.py::", "_topology.py::")],
    "comma-dropped": [ADDED[0].rstrip(",")],
    "indent-changed": ["    " + ADDED[0]],
    "trailing-space": [ADDED[0] + " "],
    "single-quotes": [ADDED[0].replace('"', "'")],
    "line-repeated": [*ADDED, *ADDED],
    "line-added": [*ADDED, '        "_test_openengine_disagg_transceiver.py::*",'],
    "other-line": ['        "_test_openengine_disagg_transceiver.py::*",'],
}


@pytest.mark.parametrize("tamper", TAMPERS)
def test_wt57_fix1_may_not_add_other_lines(tmp_path, capsys, tamper):
    base = _show(BASE)
    _refused(*_check(tmp_path, capsys, base, _after(base, ANCHOR, TAMPERS[tamper])))


# Other changes to the file, each made along with the line: (the line; what replaces it).
OTHER_CHANGES = {
    # An entry of the list dropped, or altered.
    "entry-dropped": (FIRST, None),
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
def test_wt57_fix1_may_change_nothing_else_in_the_file(tmp_path, capsys, other):
    base = _show(BASE)
    old, new = OTHER_CHANGES[other]
    lines = _after(base, ANCHOR, ADDED).split("\n")
    assert lines.count(old) == 1, old
    at = lines.index(old)
    lines[at : at + 1] = [] if new is None else [new]
    _refused(*_check(tmp_path, capsys, base, "\n".join(lines)))


# The line in place of an entry of the list, rather than added to it.
@pytest.mark.parametrize("entry", [FIRST, ANCHOR], ids=["first", "last"])
def test_wt57_fix1_may_not_put_the_line_in_place_of_an_entry(tmp_path, capsys, entry):
    base = _show(BASE)
    lines = base.split("\n")
    assert lines.count(entry) == 1, entry
    lines[lines.index(entry)] = ADDED[0]
    _refused(*_check(tmp_path, capsys, base, "\n".join(lines)))


def test_wt57_fix1_may_not_add_the_line_to_another_test(tmp_path, capsys):
    base = _show(BASE)
    _refused(*_check(tmp_path, capsys, base, _after(base, OTHER_TEST_LINE, ADDED)))


def test_wt57_fix1_may_add_the_line_once(tmp_path, capsys):
    base = _show(BASE)
    head = _after(base, ANCHOR, ADDED)
    # A second copy earlier in the same list.
    _refused(*_check(tmp_path, capsys, base, _after(head, FIRST, ADDED)))


# --- once WT-57-fix1 has landed -----------------------------------------------------------------


# Work items whose trailers at the base leave the exception in force: those of OTHERS, WT-55-fix1,
# WT-62, and WT-38, which shares PATH.
@pytest.mark.parametrize("landed", [*OTHERS, WT55_FIX1, WT62, "WT-38"])
def test_the_exception_is_not_spent_by_another_work_item(tmp_path, capsys, landed):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED), landed=landed)
    assert found == 0, out


def test_the_exception_allows_nothing_once_wt57_fix1_has_landed(tmp_path, capsys):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED), landed=ITEM)
    assert found == 1, out
    assert f"OUT OF BOUNDS  {PATH}  (the exception for {ITEM} is spent" in out


@pytest.mark.parametrize("task", [ITEM, *OTHERS])
def test_the_line_cannot_be_added_again_once_wt57_fix1_has_landed(tmp_path, capsys, task):
    landed = _after(_show(BASE), ANCHOR, ADDED)  # the file once WT-57-fix1 has landed
    again = _after(landed, ADDED[0], ADDED)  # the same line again, right after itself
    found, out = _check(tmp_path, capsys, landed, again, task, landed=ITEM)
    assert found == 1, out
    if task == ITEM:
        assert f"OUT OF BOUNDS  {PATH}  (the exception for {ITEM} is spent" in out
    else:
        assert OWNED in out

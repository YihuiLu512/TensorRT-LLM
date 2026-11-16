# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of the exceptions check_ownership.py keeps for WT-60-fix1 in the harness.

WT-60-fix1 gives XD-313 a time limit on the chain, waits longer for processes
to exit after SIGKILL, and stops the orchestrator's wait for chain workers that
have exited. Its change touches five files of the harness that WT-37 and WT-38
own, and creates a sixth, the harness's self-tests of those waits. The
exceptions allow WT-60-fix1 alone the hunks of its reference change in the
five files, each once, and creating the sixth with exactly the reference
content; nothing at all once WT-60-fix1 has landed. Run from the repository
root::

    python3 -m pytest -p no:cacheprovider scripts/poc3/
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

# The scripts import poc3_common themselves; pytest puts this directory on sys.path.
import check_ownership  # noqa: I001
import pytest

ITEM = "WT-60-fix1"
CHAIN = check_ownership.CHAIN
APPS = check_ownership.APPS
# The stage03 commit WT-60-fix1's reference change is based on, and that change.
BASE = "317d6c10b5a49920d1c8dfcebde79db30f031d44"
REF = "b013947903aef6683037297278a46b765d7b3b0f"
# The files of other work items the change touches, with their git blob ids in REF.
CHANGED = {
    f"{CHAIN}/expectations/__init__.py": "6465b5651a9b1cab4b4fddc85cd59ac2b9b2e62b",
    f"{CHAIN}/pytest_plugin.py": "183ea803d154ac7bc8003054a4d9139e708f8e84",
    f"{CHAIN}/launch.py": "4defc05920289947f9bcb4b49138aea0ff0e9d4c",
    f"{CHAIN}/remote.py": "77dbeae641764fe71a737b563fa3914d8fe9b8af",
    f"{APPS}/openai_server.py": "fb4551da20409e6d130963325c11c04b3142ea55",
}
# The file the change creates, and WT-60-fix1's own expectation table, which it owns anyway.
CREATED = f"{CHAIN}/tests/test_waits.py"
OWN = f"{CHAIN}/expectations/{ITEM}.py"
# The other work items with a rule for each of the five files, and the kind of the rule.
OWNERS = {
    f"{CHAIN}/expectations/__init__.py": [("WT-38", "own")],
    f"{CHAIN}/pytest_plugin.py": [("WT-38", "own")],
    f"{CHAIN}/launch.py": [("WT-37", "shared"), ("WT-38", "shared"), ("WT-56", "shared")],
    f"{CHAIN}/remote.py": [("WT-37", "shared")],
    f"{APPS}/openai_server.py": [("WT-37", "shared")],
}
# Work items no rule allows to make the change: the item WT-60-fix1 fixes, that item's other fixes,
# the owners of the files, and the work item that runs next.
OTHERS = ["WT-60", "WT-60-fix2", "WT-60-fix3", "WT-37", "WT-38", "WT-61"]


def _git(repo: Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, input=stdin
    ).stdout


TOP = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def _show(rev: str, path: str) -> str:
    """``path`` as at ``rev`` of this checkout; skips the test when the checkout lacks ``rev``."""
    known = subprocess.run(["git", "-C", str(TOP), "cat-file", "-e", f"{rev}^{{commit}}"])
    if known.returncode != 0:
        pytest.skip(f"{rev} is not in this checkout")
    return _git(TOP, "show", f"{rev}:{path}")


def _base() -> dict[str, str]:
    return {path: _show(BASE, path) for path in CHANGED}


def _ref() -> dict[str, str]:
    return {path: _show(REF, path) for path in (*CHANGED, CREATED, OWN)}


@lru_cache(maxsize=None)
def _hunks() -> tuple[tuple[str, int, int], ...]:
    """(path, first added line, number of added lines) of each hunk of the change with additions."""
    _show(REF, CREATED)  # skips without REF
    found = []
    for path in CHANGED:
        diff = _git(
            TOP,
            "diff",
            "-U0",
            "--diff-algorithm=myers",
            "--indent-heuristic",
            BASE,
            REF,
            "--",
            path,
        )
        for line in diff.split("\n"):
            match = check_ownership._HUNK.match(line)
            if match:
                new_start, new_count = int(match.group(3)), int(match.group(4) or 1)
                if new_count:
                    found.append((path, new_start, new_count))
    return tuple(found)


def _repo_with(
    tmp_path: Path,
    base: dict[str, str],
    head: dict[str, str],
    task: str = ITEM,
    landed: str = "",
    executable: tuple[str, ...] = (),
) -> tuple[Path, str]:
    """A repo holding the files ``base``, then ``head`` written over them in a commit of ``task``.

    With ``landed``, the base commit carries that work item's trailer; the files
    named in ``executable`` are executable at the head.
    """
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "core.fileMode", "true")
    (repo / "tensorrt_llm").mkdir()
    (repo / "tensorrt_llm/__init__.py").write_text("")

    def write(files: dict[str, str]) -> None:
        for path, text in files.items():
            file = repo / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text)

    write(base)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"base\n\nPoC3-Task: {landed}" if landed else "base")
    sha = _git(repo, "rev-parse", "HEAD").strip()
    write(head)
    for path in executable:
        os.chmod(repo / path, 0o755)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"t\n\nPoC3-Task: {task}")
    return repo, sha


def _check(tmp_path: Path, capsys, base: dict, head: dict, task: str = ITEM, **kwargs):
    """Check ``task`` changing the files from ``base`` to ``head``: the exit status and output."""
    repo, sha = _repo_with(tmp_path, base, head, task, **kwargs)
    found = check_ownership.check(task, f"{sha}..HEAD", str(repo))
    return found, capsys.readouterr().out


def _refused(found: int, out: str, path: str, why: str = "not one of the allowed changes") -> None:
    """The check refused ``path`` under the exception WT-60-fix1 has, for ``why``."""
    assert found == 1, out
    assert f"OUT OF BOUNDS  {path}  (the exception allows" in out
    assert why in out


# --- the exceptions and the reference change ---------------------------------------------------


def test_the_exceptions_are_for_wt60_fix1_alone():
    for path in CHANGED:
        rule = check_ownership.match_rule(ITEM, path)
        assert rule.kind == "exception" and rule.item == ITEM and rule.changes, path
        assert not rule.lines and not rule.redact and not rule.blob and not rule.renames, path
        owners = [(name, r.kind) for name, r in check_ownership.owners(path)]
        assert owners == [*OWNERS[path], (ITEM, "exception")], path
    rule = check_ownership.match_rule(ITEM, CREATED)
    assert rule.kind == "exception" and rule.item == ITEM
    assert rule.blob == check_ownership.WT60_FIX1_TEST_WAITS and not rule.changes
    assert [(n, r.kind) for n, r in check_ownership.owners(CREATED)] == [(ITEM, "exception")]
    assert check_ownership.match_rule(ITEM, OWN).kind == "own"


def test_the_hunks_allowed_are_those_of_the_reference_change():
    allowed = sum(len(check_ownership.match_rule(ITEM, path).changes) for path in CHANGED)
    assert allowed == 17 == len(_hunks())


def test_the_reference_change_is_as_registered():
    for path, blob in CHANGED.items():
        assert _git(TOP, "rev-parse", f"{REF}:{path}").strip() == blob, path
    created = _git(TOP, "ls-tree", "--full-tree", REF, "--", CREATED)
    assert created == f"100644 blob {check_ownership.WT60_FIX1_TEST_WAITS}\t{CREATED}\n"
    changed = _git(TOP, "diff", "--name-only", BASE, REF).split()
    assert sorted(changed) == sorted([*CHANGED, CREATED, OWN])


def test_wt60_fix1_may_make_the_reference_change_in_this_checkout(capsys):
    _show(REF, CREATED)  # skips without REF
    assert check_ownership.check(ITEM, f"{BASE}..{REF}", str(TOP)) == 0
    out = capsys.readouterr().out
    for path in (*CHANGED, CREATED):
        assert f"CHECK BY HAND  EXCEPTION  {path}" in out, path


# --- what the exceptions allow ------------------------------------------------------------------


def test_wt60_fix1_may_make_the_reference_change(tmp_path, capsys):
    found, out = _check(tmp_path, capsys, _base(), _ref())
    assert found == 0, out


# Each hunk is allowed on its own, the others not made: say, when a later version of the change
# drops one.
@pytest.mark.parametrize("path", CHANGED)
def test_wt60_fix1_may_change_one_file_alone(tmp_path, capsys, path):
    found, out = _check(tmp_path, capsys, _base(), {path: _ref()[path]})
    assert found == 0, out


# --- what they do not allow ---------------------------------------------------------------------


def _with_line(text: str, number: int, line: str | None) -> str:
    """``text`` with its line ``number`` (from 1) replaced by ``line``, or dropped for None."""
    lines = text.split("\n")
    lines[number - 1 : number] = [] if line is None else [line]
    return "\n".join(lines)


# Each hunk with one of its added lines differing from the reference by a trailing space.
@pytest.mark.parametrize("hunk", range(17))
def test_wt60_fix1_may_not_change_a_line_of_a_hunk(tmp_path, capsys, hunk):
    path, start, count = _hunks()[hunk]
    ref = _ref()
    lines = ref[path].split("\n")
    at = start + count - 1  # the hunk's last added line
    head = {**ref, path: _with_line(ref[path], at, lines[at - 1] + " ")}
    _refused(*_check(tmp_path, capsys, _base(), head), path)


# A line added at the end of each file, along with the reference change.
@pytest.mark.parametrize("path", CHANGED)
def test_wt60_fix1_may_change_nothing_else_in_a_file(tmp_path, capsys, path):
    ref = _ref()
    head = {**ref, path: ref[path] + "# another change\n"}
    _refused(*_check(tmp_path, capsys, _base(), head), path)


# The first line of each file dropped, along with the reference change.
@pytest.mark.parametrize("path", CHANGED)
def test_wt60_fix1_may_drop_no_line_of_a_file(tmp_path, capsys, path):
    ref = _ref()
    _refused(
        *_check(tmp_path, capsys, _base(), {**ref, path: _with_line(ref[path], 1, None)}), path
    )


# The two new top-level functions, each with the blank lines after it, moved from where they are
# to just before another top-level statement of their files: the same lines, in another place.
@pytest.mark.parametrize(
    "path, first, before",
    [
        (f"{CHAIN}/expectations/__init__.py", "def time_limit(", "def outcome_expectation("),
        (f"{CHAIN}/remote.py", "def exited_workers(", "class DisaggTopology:"),
    ],
    ids=["time_limit", "exited_workers"],
)
def test_wt60_fix1_may_not_add_a_function_elsewhere(tmp_path, capsys, path, first, before):
    ref = _ref()
    lines = ref[path].split("\n")
    (hunk,) = [(s, c) for p, s, c in _hunks() if p == path and lines[s - 1].startswith(first)]
    moved = lines[hunk[0] - 1 : hunk[0] - 1 + hunk[1]]
    rest = lines[: hunk[0] - 1] + lines[hunk[0] - 1 + hunk[1] :]
    (at,) = [i for i, line in enumerate(rest) if line.startswith(before)]
    head = "\n".join(rest[:at] + moved + rest[at:])
    repo, sha = _repo_with(tmp_path, _base(), {**ref, path: head})
    # The hunk is the same lines as the reference's, only the statement it follows differs.
    diff = _git(repo, "diff", "-U0", sha, "HEAD", "--", path).split("\n")
    assert [line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")][
        -len(moved) :
    ] == moved
    found = check_ownership.check(ITEM, f"{sha}..HEAD", str(repo))
    _refused(found, capsys.readouterr().out, path)


# A change naming the statements it touches allows the hunk only where it touches exactly those:
# not where it touches fewer.
@pytest.mark.parametrize(
    "around, allowed", [(("A",), True), (("A", "B"), False), ((None, "A"), False)]
)
def test_a_change_around_statements_needs_exactly_those(tmp_path, around, allowed):
    path = "m.py"
    repo, sha = _repo_with(tmp_path, {path: "A = 1\nB = 2\n"}, {path: "A = 3\nB = 2\n"})
    rule = check_ownership._exception(
        path,
        "n",
        item=ITEM,
        changes=(check_ownership._change_around(around, ("A = 1",), ("A = 3",)),),
    )
    found = check_ownership.disallowed_lines(rule, str(repo), sha, "HEAD", path)
    assert (found == []) == allowed, found


# A hunk made twice: the import of handles_since once more right after the first.
def test_wt60_fix1_may_make_each_hunk_once(tmp_path, capsys):
    path = f"{CHAIN}/remote.py"
    ref = _ref()
    head = ref[path].replace("    handles_since,\n", "    handles_since,\n    handles_since,\n", 1)
    _refused(*_check(tmp_path, capsys, _base(), {**ref, path: head}), path)


@pytest.mark.parametrize("task", OTHERS)
def test_no_other_work_item_may_make_the_change(tmp_path, capsys, task):
    found, out = _check(tmp_path, capsys, _base(), _ref(), task)
    assert found == 1, out
    for path in CHANGED:
        names = [name for name, _ in OWNERS[path]]
        if task in names:
            continue  # its own rule for the file; the others refuse it
        assert f"OUT OF BOUNDS  {path}  (owned by {', '.join([*names, ITEM])})" in out, path
    assert f"OUT OF BOUNDS  {CREATED}  (owned by {ITEM})" in out


# --- the created file ---------------------------------------------------------------------------


def test_wt60_fix1_may_not_create_the_file_with_other_content(tmp_path, capsys):
    ref = _ref()
    head = {**ref, CREATED: ref[CREATED] + "\n"}
    _refused(*_check(tmp_path, capsys, _base(), head), CREATED, "is not the regular file")


def test_wt60_fix1_may_not_create_the_file_executable(tmp_path, capsys):
    found, out = _check(tmp_path, capsys, _base(), _ref(), executable=(CREATED,))
    _refused(found, out, CREATED, "is not the regular file")


# The file at the base already, with other content: the range changes it into the registered one.
def test_wt60_fix1_may_not_change_the_file_once_it_exists(tmp_path, capsys):
    ref = _ref()
    base = {**_base(), CREATED: ref[CREATED].replace("def test_", "def test_x_", 1)}
    _refused(*_check(tmp_path, capsys, base, ref), CREATED, "exists at the merge base")


def test_wt60_fix1_may_not_create_another_file_by_the_rule(tmp_path, capsys):
    ref = _ref()
    other = f"{CHAIN}/tests/test_waits_more.py"
    found, out = _check(tmp_path, capsys, _base(), {**ref, other: ref[CREATED]})
    assert found == 1 and f"OUT OF BOUNDS  {other}  (owned by nobody)" in out, out


# --- once WT-60-fix1 has landed -----------------------------------------------------------------


# Work items whose trailers at the base leave the exceptions in force.
@pytest.mark.parametrize("landed", OTHERS)
def test_the_exceptions_are_not_spent_by_another_work_item(tmp_path, capsys, landed):
    found, out = _check(tmp_path, capsys, _base(), _ref(), landed=landed)
    assert found == 0, out


def test_the_exceptions_allow_nothing_once_wt60_fix1_has_landed(tmp_path, capsys):
    found, out = _check(tmp_path, capsys, _base(), _ref(), landed=ITEM)
    assert found == 1, out
    for path in (*CHANGED, CREATED):
        assert f"OUT OF BOUNDS  {path}  (the exception for {ITEM} is spent" in out, path


# The reference change in place at a base carrying WT-60-fix1's trailer, and one of its hunks
# made again: handles_since imported a second time.
@pytest.mark.parametrize("task", [ITEM, "WT-60", "WT-60-fix2"])
def test_no_hunk_can_be_made_again_once_wt60_fix1_has_landed(tmp_path, capsys, task):
    ref = _ref()
    path = f"{CHAIN}/remote.py"
    again = {path: ref[path].replace("    handles_since,\n", "    handles_since,\n" * 2, 1)}
    found, out = _check(tmp_path, capsys, ref, again, task, landed=ITEM)
    assert found == 1, out
    if task == ITEM:
        assert f"OUT OF BOUNDS  {path}  (the exception for {ITEM} is spent" in out
    else:
        assert f"OUT OF BOUNDS  {path}  (owned by WT-37, {ITEM})" in out


# --- the forms of an exception ------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"blob": "0" * 40, "lines": ("a",)}, "or a blob alone"),
        (
            {"blob": "0" * 40, "changes": (check_ownership._change("f", added=("a",)),)},
            "or a blob alone",
        ),
        ({"blob": "0" * 39}, "full git blob id"),
        ({"blob": "G" * 40}, "full git blob id"),
        ({"changes": (check_ownership.Change((), ("a",), "f", ("f",)),)}, "either the statement"),
        ({"changes": (check_ownership.Change((), ("a",), ""),)}, "either the statement"),
    ],
    ids=[
        "blob-and-lines",
        "blob-and-changes",
        "short-blob",
        "not-hex",
        "within-and-around",
        "neither",
    ],
)
def test_an_exception_is_well_formed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        check_ownership._exception("x.py", "n", item=ITEM, **kwargs)


def test_an_exception_creating_a_file_names_its_work_item():
    table = {"WT-60": [check_ownership._exception("x.py", "n", blob="0" * 40)]}
    with pytest.raises(ValueError, match="an exception creating a file names no item"):
        check_ownership._named_fixes(table)


# Regions as _regions gives them: a docstring, A = 1, a comment, B = 2.
REGIONS = [None, "A", "A", "B"]


@pytest.mark.parametrize(
    "span, after, touched",
    [
        ([1, 0, 2, 1], [None, None, "A", "A", "B"], {None}),  # a docstring line
        ([3, 0, 4, 2], [None, "A", "A", "C", "C", "B"], {"A", "C"}),  # a new statement after A
        ([2, 1, 2, 1], [None, "A", "A", "B"], {"A"}),  # A changed in place
        ([3, 1, 3, 2], [None, "A", "B", "B", "B"], {"A", "B"}),  # A's comment moved to B
        ([0, 0, 1, 1], [None, None, "A", "A", "B"], {None}),  # a line before everything
        ([4, 2, 4, 1], [None, "A", "A", "B"], None),  # past the end of the file
    ],
    ids=["docstring", "new-statement", "in-place", "comment-moved", "first-line", "past-end"],
)
def test_the_statements_a_hunk_touches(span, after, touched):
    assert check_ownership._hunk_touches(REGIONS, after, span) == touched

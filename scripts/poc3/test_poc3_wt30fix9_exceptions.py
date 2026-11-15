# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of the exception check_ownership.py keeps for WT-51-fix2 in WT-55's tests.

The exception allows WT-51-fix2 alone to add one block, once, to
``test_the_generation_leg_never_sends_schedule_style_in_extra`` of
``test_poc3_wt55_gen_first.py``, a file WT-55 owns: its undeclared case asserts
that the mapping refuses the decode call of a generation-first request
(SV-DSG-8). It allows WT-51-fix2 nothing else in the file, and nothing at all
once WT-51-fix2 has landed. Run from the repository root::

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

ITEM = "WT-51-fix2"
PATH = "tests/unittest/serve/openengine_poc/wt55/test_poc3_wt55_gen_first.py"
TEST = "test_the_generation_leg_never_sends_schedule_style_in_extra"
# A stage03 commit with PATH as it is before WT-51-fix2's change, and the git blob id of PATH
# with the block added after ANCHOR, as WT-51-fix2 changes it.
BASE = "9f180985529b07e63557773f6e0bee3b4b5431d2"
HEAD_BLOB = "59b5ef42ee3026f270ef896b159b1e8b06575917"
# The line of TEST the block goes after, and the block.
ANCHOR = (
    "    # EX-DSG-7: schedule_style is a prefill-only extra key; the decode call has it in"
    " kv.session."
)
ADDED = [
    "    if not generation_first:",
    "        # SV-DSG-8: a generation-first decode call needs the engine to declare the capability.",
    '        with pytest.raises(ValueError, match=r"generation-first.*trtllm\\.disagg\\.generation_first"):',
    "            _map(_gen_first_gen_params(), _caps(generation_first=False))",
    "        return",
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


def test_the_exception_lists_exactly_the_block():
    rule = check_ownership.match_rule(ITEM, PATH)
    assert rule.kind == "exception" and not rule.lines and not rule.renames
    assert rule.item == ITEM
    assert list(rule.changes) == [
        check_ownership.Change((), tuple(map(re.escape, ADDED)), TEST),
    ]
    assert [(n, r.kind) for n, r in check_ownership.owners(PATH)] == [
        ("WT-55", "own"),
        (ITEM, "exception"),
    ]


def test_the_block_is_the_change_of_the_reference_patch():
    head = _after(_show(BASE), ANCHOR, ADDED)
    found = subprocess.run(
        ["git", "hash-object", "--stdin"], input=head, check=True, capture_output=True, text=True
    )
    assert found.stdout.strip() == HEAD_BLOB


def test_wt51_fix2_may_add_the_block(tmp_path, capsys):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED))
    assert found == 0, out
    assert f"CHECK BY HAND  EXCEPTION  {PATH}" in out


# Work items that own nothing in PATH: the item WT-51-fix2 fixes and that item's other fixes.
OTHERS = ["WT-51", "WT-51-fix1", "WT-51-fix3"]


@pytest.mark.parametrize("task", OTHERS)
def test_no_other_work_item_may_add_the_block(tmp_path, capsys, task):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED), task)
    assert found == 1 and f"OUT OF BOUNDS  {PATH}  (owned by WT-55" in out, out


# The ways the block can differ from the one allowed: a line fewer, one more, one altered, the
# declared case refused as well, or another line in its place.
TAMPERS = {
    "line-dropped": ADDED[:-1],
    "line-repeated": [*ADDED, ADDED[-1]],
    "line-altered": [*ADDED[:-1], "        return None"],
    "both-cases": ["    if True:", *ADDED[1:]],
    "other-line": ["    pass"],
}


@pytest.mark.parametrize("tamper", TAMPERS)
def test_wt51_fix2_may_not_add_another_block(tmp_path, capsys, tamper):
    base = _show(BASE)
    head = _after(base, ANCHOR, TAMPERS[tamper])
    ast.parse(head)  # each tampered file is still Python, so the hunk itself is refused
    found, out = _check(tmp_path, capsys, base, head)
    assert found == 1, out
    assert f"OUT OF BOUNDS  {PATH}  (the exception allows" in out
    assert "not one of the allowed changes" in out


# Other changes to the file, each made along with the block: (the line; what replaces it).
OTHER_CHANGES = {
    # The declared case dropped from the parametrization, or the parametrization changed.
    "declared-dropped": (
        '@pytest.mark.parametrize("generation_first", [True, False],'
        ' ids=["declared", "undeclared"])',
        '@pytest.mark.parametrize("generation_first", [False], ids=["undeclared"])',
    ),
    "ids-renamed": (
        '@pytest.mark.parametrize("generation_first", [True, False],'
        ' ids=["declared", "undeclared"])',
        '@pytest.mark.parametrize("generation_first", [True, False], ids=["yes", "no"])',
    ),
    # An assertion of the declared case dropped.
    "assertion-dropped": ("    assert request.request_type is None", None),
    # Another parametrized test of the file changed.
    "other-parametrization": (
        '@pytest.mark.parametrize("style", [None, DisaggScheduleStyle.CONTEXT_FIRST],'
        ' ids=["unset", "0"])',
        '@pytest.mark.parametrize("style", [None], ids=["unset"])',
    ),
}


@pytest.mark.parametrize("other", OTHER_CHANGES)
def test_wt51_fix2_may_change_nothing_else_in_the_file(tmp_path, capsys, other):
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


def test_wt51_fix2_may_not_add_the_block_to_another_test(tmp_path, capsys):
    base = _show(BASE)
    other = "    params = _ctx_params(DisaggScheduleStyle.GENERATION_FIRST)"
    found, out = _check(tmp_path, capsys, base, _after(base, other, ADDED))
    assert found == 1 and "not one of the allowed changes" in out, out


def test_wt51_fix2_may_add_the_block_once(tmp_path, capsys):
    base = _show(BASE)
    head = _after(base, ANCHOR, ADDED)
    # A second copy further down the same test, so that the diff shows it as a hunk of its own.
    later = "    assert request.request_type is None"
    found, out = _check(tmp_path, capsys, base, _after(head, later, ADDED))
    assert found == 1, out
    assert "not one of the allowed changes" in out


# Work items whose trailers at the base leave the exception in force: those of OTHERS and the
# item that owns PATH.
@pytest.mark.parametrize("landed", [*OTHERS, "WT-55"])
def test_the_exception_is_not_spent_by_another_work_item(tmp_path, capsys, landed):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED), landed=landed)
    assert found == 0, out


def test_the_exception_allows_nothing_once_wt51_fix2_has_landed(tmp_path, capsys):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _after(base, ANCHOR, ADDED), landed=ITEM)
    assert found == 1
    assert f"OUT OF BOUNDS  {PATH}  (the exception for {ITEM} is spent" in out


def test_the_block_cannot_be_added_again_once_wt51_fix2_has_landed(tmp_path, capsys):
    landed = _after(_show(BASE), ANCHOR, ADDED)  # the file once WT-51-fix2 has landed
    again = _after(landed, ADDED[-1], ADDED)  # the same block again, right after itself
    found, out = _check(tmp_path, capsys, landed, again, landed=ITEM)
    assert found == 1, out
    assert f"OUT OF BOUNDS  {PATH}  (the exception for {ITEM} is spent" in out

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of the exception check_ownership.py keeps for WT-71 in WT-70's arm-table test.

WT-71 appends the case-03 arms to the perf harness's table of disaggregated
arms. The exception allows WT-71 alone two hunks, each at most once, in
``test_poc3_wt70_arms.py``, a file of WT-70: the E3 arms added to
``EXPECTED_DISAGG_ARMS`` (the check does not pin where in it), and in
``test_disagg_arm_table`` the expected case taken per arm (case-03 for an E3
arm, case-04 for any other). It allows WT-71 nothing else in the file, and
nothing at all once WT-71 has landed. Run from the repository root::

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

ITEM = "WT-71"
PATH = "tests/unittest/serve/openengine_poc/wt70/test_poc3_wt70_arms.py"
TABLE = "EXPECTED_DISAGG_ARMS"
TEST = "test_disagg_arm_table"
# A stage03 commit with PATH as it is before WT-71's change, and the git blob id of PATH with both
# of WT-71's hunks applied.
BASE = "de6cee6c5e1e8001609041e52f6e20520d75ce50"
HEAD_BLOB = "d56d1f939e376f7551dba17c40e81b7f05c2c88a"
# The last arm of TABLE, and the lines WT-71's change puts after it.
ANCHOR = '    "E4-direct": ("direct", None, None),'
ARMS = [
    "    # The case-03 arms (the ds engine), after the case-04 ones.",
    '    "E3-native": ("disagg", ("none", None, None), ("none", None, None)),',
    '    "E3-native-al": ("disagg", ("none", 1, None), ("none", 1, None)),',
    '    "E3-mid": ("disagg", ("none", 1, None), ("none", 1, None)),',
    '    "E3-chain": ("disagg", ("ds", 1, None), ("ds", 1, None)),',
    '    "E3-chain-lean": ("disagg", ("ds", 1, 0), ("ds", 1, 0)),',
    '    "E3-direct": ("direct", None, None),',
]
# The case TEST expects, before and after.
CASE = '            "case-04",'
CASE_PER_ARM = '            "case-03" if name.startswith("E3-") else "case-04",'


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@lru_cache(maxsize=None)
def _show(rev: str) -> str:
    """PATH as at ``rev`` of this checkout; skips the test when the checkout lacks ``rev``."""
    top = Path(__file__).resolve().parents[2]
    known = subprocess.run(
        ["git", "-C", str(top), "cat-file", "-e", f"{rev}^{{commit}}"], capture_output=True
    )
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


def _replace(text: str, old: str, new: list[str]) -> str:
    """``text`` with its one line ``old`` replaced by the lines ``new`` (none: dropped)."""
    lines = text.split("\n")
    assert lines.count(old) == 1, old
    at = lines.index(old)
    return "\n".join(lines[:at] + new + lines[at + 1 :])


def _changed(text: str) -> str:
    """``text`` with both of WT-71's hunks."""
    return _replace(_after(text, ANCHOR, ARMS), CASE, [CASE_PER_ARM])


def _check(tmp_path: Path, capsys, base: str, head: str, task: str = ITEM, landed: str = ""):
    """Check ``task`` changing PATH from ``base`` to ``head``: the exit status and the output."""
    repo, sha = _repo_with(tmp_path, base, head, task, landed)
    found = check_ownership.check(task, f"{sha}..HEAD", str(repo))
    return found, capsys.readouterr().out


REFUSED = f"OUT OF BOUNDS  {PATH}  (the exception allows"
NOT_ALLOWED = "not one of the allowed changes"


def test_the_exception_lists_exactly_the_changes():
    rule = check_ownership.match_rule(ITEM, PATH)
    assert rule.kind == "exception" and not rule.lines and not rule.renames
    assert rule.item == ITEM
    assert list(rule.changes) == [
        check_ownership.Change((), tuple(map(re.escape, ARMS)), TABLE),
        check_ownership.Change((re.escape(CASE),), (re.escape(CASE_PER_ARM),), TEST),
    ]
    assert [(n, r.kind) for n, r in check_ownership.owners(PATH)] == [
        ("WT-70", "own"),
        (ITEM, "exception"),
    ]
    # WT-71 changes the ds-engine repository too; the exception is for this repository alone.
    assert check_ownership.match_rule(ITEM, PATH, check_ownership.DS) is None


def test_the_changes_are_those_of_the_reference_patch():
    found = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), "hash-object", "--stdin"],
        input=_changed(_show(BASE)),
        check=True,
        capture_output=True,
        text=True,
    )
    assert found.stdout.strip() == HEAD_BLOB


def test_wt71_may_make_the_changes(tmp_path, capsys):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _changed(base))
    assert found == 0, out
    assert f"CHECK BY HAND  EXCEPTION  {PATH}" in out


# No change is required: either one alone is allowed too.
@pytest.mark.parametrize("alone", ["arms", "case"])
def test_wt71_may_make_either_change_alone(tmp_path, capsys, alone):
    base = _show(BASE)
    head = _after(base, ANCHOR, ARMS) if alone == "arms" else _replace(base, CASE, [CASE_PER_ARM])
    found, out = _check(tmp_path, capsys, base, head)
    assert found == 0, out


# Work items that own nothing in PATH: WT-71's fixes and the perf items after it.
OTHERS = ["WT-71-fix1", "WT-71-fix2", "WT-71-fix3", "WT-72", "WT-73"]


@pytest.mark.parametrize("task", OTHERS)
def test_no_other_work_item_may_make_the_changes(tmp_path, capsys, task):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _changed(base), task)
    assert found == 1 and f"OUT OF BOUNDS  {PATH}  (owned by WT-70, {ITEM})" in out, out


# The ways the added lines can differ from those allowed: an arm with other settings (among them
# the E3-native line of an earlier reference patch), an arm more or less, the lines in another
# order, another comment or none, a line with more after it, an E4 arm redefined after the E4
# ones, or the arms twice in one hunk.
ARM_TAMPERS = {
    "arm-settings-changed": [
        line.replace('"ds"', '"trtllm"') if line.startswith('    "E3-chain":') else line
        for line in ARMS
    ],
    "earlier-e3-native": [
        '    "E3-native": ("disagg", ("none", 1, None), ("none", 1, None)),'
        if line.startswith('    "E3-native":')
        else line
        for line in ARMS
    ],
    "arm-added": [*ARMS, '    "E3-extra": ("disagg", ("ds", 1, None), ("ds", 1, None)),'],
    "arm-left-out": [line for line in ARMS if not line.startswith('    "E3-mid":')],
    "arms-reordered": [ARMS[0], *reversed(ARMS[1:])],
    "comment-changed": ["    # The case-03 arms.", *ARMS[1:]],
    "comment-dropped": ARMS[1:],
    "line-extended": [*ARMS[:-1], ARMS[-1] + "  # replayed"],
    "e4-arm-redefined": [
        *ARMS,
        '    "E4-chain": ("disagg", ("none", 1, None), ("none", 1, None)),',
    ],
    "arms-twice": [*ARMS, *ARMS],
}


@pytest.mark.parametrize("tamper", ARM_TAMPERS)
def test_wt71_may_not_append_other_lines(tmp_path, capsys, tamper):
    base = _show(BASE)
    head = _replace(_after(base, ANCHOR, ARM_TAMPERS[tamper]), CASE, [CASE_PER_ARM])
    ast.parse(head)  # each tampered file is still Python, so the hunk itself is refused
    found, out = _check(tmp_path, capsys, base, head)
    assert found == 1, out
    assert REFUSED in out and NOT_ALLOWED in out


# The ways the expected case can differ from the one allowed: E3 matched more broadly, the case
# taken from the arm itself, case-03 expected of the E4 arms too, or more after the line.
CASE_TAMPERS = {
    "prefix-broadened": '            "case-03" if name.startswith("E") else "case-04",',
    "case-from-the-arm": "            arm.case,",
    "either-case": '            arm.case if arm.case in ("case-03", "case-04") else "case-04",',
    "case-03-for-all": '            "case-03",',
    "line-extended": CASE_PER_ARM + "  # E4 stays case-04",
}


@pytest.mark.parametrize("tamper", CASE_TAMPERS)
def test_wt71_may_not_expect_the_case_otherwise(tmp_path, capsys, tamper):
    base = _show(BASE)
    head = _replace(_after(base, ANCHOR, ARMS), CASE, [CASE_TAMPERS[tamper]])
    ast.parse(head)
    found, out = _check(tmp_path, capsys, base, head)
    assert found == 1, out
    assert REFUSED in out and NOT_ALLOWED in out


# Other changes to the file, each made along with both hunks: (the line; what replaces it, None
# to drop it).
OTHER_CHANGES = {
    # An E4 arm of the table dropped, or expected otherwise.
    "e4-arm-dropped": ('    "E4-ctx": ("disagg", ("trtllm", 1, None), ("none", 1, None)),', None),
    "e4-arm-altered": (
        '    "E4-chain": ("disagg", ("trtllm", 1, None), ("trtllm", 1, None)),',
        '    "E4-chain": ("disagg", ("none", 1, None), ("none", 1, None)),',
    ),
    # Assertions of test_disagg_arm_table relaxed or dropped.
    "order-relaxed": (
        "    assert list(arms.DISAGG_ARMS) == list(EXPECTED_DISAGG_ARMS)",
        "    assert set(arms.DISAGG_ARMS) >= set(EXPECTED_DISAGG_ARMS)",
    ),
    "topology-relaxed": ("            topology,", "            arm.topology,"),
    "single-server-check-dropped": (
        "        assert (arm.glm_http, arm.frontends, arm.postprocess_workers) == (False, None, None)",
        None,
    ),
    # Lines of other tests.
    "all-arms-relaxed": (
        "    assert set(arms.all_arms()) == set(POC2_ARMS) | set(EXPECTED_DISAGG_ARMS)",
        "    assert set(arms.all_arms()) >= set(POC2_ARMS) | set(EXPECTED_DISAGG_ARMS)",
    ),
    "other-test-changed": (
        '    with pytest.raises(arms.ArmError, match="direct replay"):',
        '    with pytest.raises(arms.ArmError, match="direct"):',
    ),
}


@pytest.mark.parametrize("other", OTHER_CHANGES)
def test_wt71_may_change_nothing_else_in_the_file(tmp_path, capsys, other):
    base = _show(BASE)
    old, new = OTHER_CHANGES[other]
    head = _replace(_changed(base), old, [] if new is None else [new])
    found, out = _check(tmp_path, capsys, base, head)
    assert found == 1, out
    assert REFUSED in out and NOT_ALLOWED in out


def test_wt71_may_add_the_arms_once(tmp_path, capsys):
    base = _show(BASE)
    # A second copy in a hunk of its own, after the first E4 arm.
    head = _after(
        _changed(base), '    "E4-base": ("disagg", ("none", 1, None), ("none", 1, None)),', ARMS
    )
    found, out = _check(tmp_path, capsys, base, head)
    assert found == 1, out
    assert REFUSED in out and NOT_ALLOWED in out


def test_wt71_may_not_append_the_arms_to_another_statement(tmp_path, capsys):
    base = _show(BASE)
    head = _replace(_after(base, '    "stream_interval": 20,', ARMS), CASE, [CASE_PER_ARM])
    found, out = _check(tmp_path, capsys, base, head)
    assert found == 1, out
    assert REFUSED in out and NOT_ALLOWED in out


# Work items whose trailers at the base leave the exception in force: those of OTHERS and the
# item that owns PATH.
@pytest.mark.parametrize("landed", [*OTHERS, "WT-70"])
def test_the_exception_is_not_spent_by_another_work_item(tmp_path, capsys, landed):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _changed(base), landed=landed)
    assert found == 0, out


def test_the_exception_allows_nothing_once_wt71_has_landed(tmp_path, capsys):
    base = _show(BASE)
    found, out = _check(tmp_path, capsys, base, _changed(base), landed=ITEM)
    assert found == 1, out
    assert f"OUT OF BOUNDS  {PATH}  (the exception for {ITEM} is spent" in out


@pytest.mark.parametrize("task", [ITEM, *OTHERS])
def test_the_arms_cannot_be_appended_again_once_wt71_has_landed(tmp_path, capsys, task):
    landed = _changed(_show(BASE))  # the file once WT-71 has landed
    again = _after(landed, ARMS[-1], ARMS)  # the same lines again, right after themselves
    found, out = _check(tmp_path, capsys, landed, again, task, landed=ITEM)
    assert found == 1, out
    if task == ITEM:
        assert f"OUT OF BOUNDS  {PATH}  (the exception for {ITEM} is spent" in out
    else:
        assert f"OUT OF BOUNDS  {PATH}  (owned by WT-70, {ITEM})" in out

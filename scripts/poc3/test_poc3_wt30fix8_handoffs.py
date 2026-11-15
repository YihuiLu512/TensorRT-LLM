# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of where the cases in HANDED_TO wait: for WT-52-fix1, WT-53-fix2 or WT-55-fix1.

HANDED_TO gives the fix each case waits for, MOVED the rows each fix owes for
them; WT-37 owes none of these cases. Run from the repository root::

    python3 -m pytest -p no:cacheprovider scripts/poc3/
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# The scripts import poc3_common themselves; pytest puts this directory on sys.path.
import pending_e2e  # noqa: I001
import pytest
import yaml

# The same set as in test_poc3_scripts.py, kept here so that this module imports no other test
# module.
LINK = {"WT-35", "WT-37", "WT-38", "WT-40", "WT-50", "WT-51"}
# Items taken as landed: the dependencies of the rows below, the fixes excepted.
LANDED = LINK | {"WT-52", "WT-53", "WT-55"}

# The rows each fix owes for these cases, with their cases and their dependencies but the fix.
MOVED = {
    "WT-52-fix1": {
        "F-25+WT-52-fix1": (["XD-219", "NT-065"], LINK | {"WT-52"}),
        "EG-06+WT-52-fix1": (["XD-219"], LINK | {"WT-52"}),
    },
    "WT-53-fix2": {
        "DG-19+WT-53-fix2": (["XD-121", "XD-123", "NT-081"], LINK | {"WT-53"}),
        "F-43": (["XD-120", "NT-073"], LINK | {"WT-53"}),
        "DG-24": (["XD-026", "XD-119", "NT-086"], LINK | {"WT-53"}),
    },
    "WT-55-fix1": {
        "DG-39b": (["NT-091"], LINK | {"WT-52", "WT-55"}),
        "DG-81": (["XD-215", "XD-207", "XD-015", "XD-016", "XD-227"], LINK | {"WT-53", "WT-55"}),
        "EG-20": (["NT-200"], LINK | {"WT-55"}),
    },
}
MOVED_ROWS = {row for rows in MOVED.values() for row in rows}
# The cases the rows XD-121, XD-123 and XD-219 left keep, with their dependencies.
LEFT = {
    "DG-19": (["XD-025", "XD-007", "XD-107", "XD-058", "XD-059"], LINK | {"WT-53"}),
    "F-25": (["T2"], LINK | {"WT-52"}),
    "EG-06": (["NT-026"], LINK | {"WT-52"}),
}
# The fix each of these cases waits for.
HANDED_TO = {
    **dict.fromkeys(["XD-121", "XD-123", "XD-120", "XD-026", "XD-119"], "WT-53-fix2"),
    "XD-219": "WT-52-fix1",
    **dict.fromkeys(
        ["NT-091", "NT-200", "XD-215", "XD-207", "XD-015", "XD-016", "XD-227"], "WT-55-fix1"
    ),
}
# Rows holding one of them that do not wait for its fix: EG-03, which depends on the link and WT-57,
# is not one of the rows handed over.
NOT_WAITING = {"XD-215": {"EG-03"}}


@pytest.fixture(scope="module")
def rows():
    return {row.id: row for row in pending_e2e.load_rows()[1]}


def _owed(task, landed, rows):
    return {row.id for row in pending_e2e.obligations(task, landed, rows.values()).owed}


def test_the_moved_rows_hold_their_cases_and_wait_for_their_fix(rows):
    for fix, moved in MOVED.items():
        for row_id, (cases, deps) in moved.items():
            assert list(rows[row_id].cases) == cases, row_id
            assert rows[row_id].deps == deps | {fix}, row_id
    for row_id, (cases, deps) in LEFT.items():
        assert list(rows[row_id].cases) == cases and rows[row_id].deps == deps, row_id


def test_a_handed_over_case_waits_for_its_fix_in_every_row_but_those_not_waiting(rows):
    # Worked out from the file: a row added to the tables with one of these cases is covered too.
    for case, fix in HANDED_TO.items():
        holding = [row for row in rows.values() if case in row.cases]
        assert holding, case
        for row in holding:
            if row.id in NOT_WAITING.get(case, ()):
                assert fix not in row.deps, (case, row.id)
            else:
                assert fix in row.deps, (case, row.id)
    assert {row.id for row in rows.values() if "XD-219" in row.cases} == {
        "F-25+WT-52-fix1",
        "EG-06+WT-52-fix1",
    }


def test_wt37_owes_none_of_the_moved_cases(rows):
    owed = _owed("WT-37", LANDED - {"WT-37"}, rows)
    assert not owed & MOVED_ROWS
    assert set(LEFT) <= owed
    cases = {case for row_id in owed for case in rows[row_id].cases}
    assert not cases & {case for moved in MOVED.values() for c, _ in moved.values() for case in c}


@pytest.mark.parametrize("fix", sorted(MOVED))
def test_each_fix_owes_its_moved_rows_once_their_other_dependencies_have_landed(rows, fix):
    owed = _owed(fix, LANDED, rows)
    assert owed & MOVED_ROWS == set(MOVED[fix])
    if fix == "WT-55-fix1":  # no other row waits for it, landed or not
        assert owed == set(MOVED[fix])
        assert {row.id for row in rows.values() if fix in row.deps} == set(MOVED[fix])
    for other in sorted(set(MOVED) - {fix}):
        assert not set(MOVED[fix]) & _owed(other, LANDED, rows)


def test_wt55_owes_its_rows_to_its_fix_and_hears_of_them(rows):
    result = pending_e2e.obligations("WT-55", LANDED - {"WT-55"}, rows.values())
    assert not {row.id for row in result.owed} & set(MOVED["WT-55-fix1"])
    assert {row.id for row in result.other_fix} == set(MOVED["WT-55-fix1"])
    # The fix landing before WT-55 leaves the rows to WT-55, as any dependency would.
    landed = (LANDED - {"WT-55"}) | {"WT-55-fix1"}
    assert set(MOVED["WT-55-fix1"]) <= _owed("WT-55", landed, rows)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def test_pending_on_a_constructed_history(tmp_path, capsys):
    """WT-37, landing after WT-55, owes none of these rows; each fix, landing after it, owes its own."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")

    def land(*items: str) -> None:
        for item in items:
            _git(repo, "commit", "-q", "--allow-empty", "-m", f"{item}\n\nPoC3-Task: {item}")

    def owed(task: str) -> set[str]:
        argv = ["--repo", str(repo), "--ref", "main", "--task", task, "--json"]
        assert pending_e2e.main(argv) == 0
        return set(yaml.safe_load(capsys.readouterr().out)["owed"])

    land("WT-30", *sorted(LANDED - {"WT-37"}))
    wt37 = owed("WT-37")
    assert not wt37 & MOVED_ROWS and set(LEFT) <= wt37
    land("WT-37")
    for fix in sorted(MOVED):
        assert owed(fix) & MOVED_ROWS == set(MOVED[fix])
        land(fix)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of how the rows WT-57 completes are shared with WT-57-fix1 and WT-57-fix2.

WT-57 keeps the CPU and one-GPU cases (KEPT_WHOLE but DG-43+rest, and the cases
PARTIAL keeps); WT-57-fix1 takes the new multi-GPU cases and WT-57-fix2 the
existing multi-GPU ones. R11 (WT-60) waits for both fixes and verifies DG-43+rest; the R12 rows
whose cases WT-57-fix1 writes wait for it. The tests also cover the split
mechanism these rows need: splits of a ``+rest`` row, a partial and a whole
split on one row, notes that name only cases that stay, ``GROUP_ADDITIONS``,
``VERIFIERS``, ``ROW_WAITS``, and the refusal of fixes named in the tables of
tasks.md. Run from the repository root::

    python3 -m pytest -p no:cacheprovider scripts/poc3/
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# The scripts import poc3_common themselves; pytest puts this directory on sys.path.
import check_ownership  # noqa: I001
import pending_e2e
import pytest
import sync_feature_deps
import yaml

# The same set as in test_poc3_scripts.py, kept here so that this module imports no other test
# module.
LINK = {"WT-35", "WT-37", "WT-38", "WT-40", "WT-50", "WT-51"}
FIX1, FIX2 = "WT-57-fix1", "WT-57-fix2"
# Items taken as landed: every dependency of the rows below but WT-57 and its fixes.
LANDED = LINK | {"WT-52", "WT-54"}

# The rows WT-57 owed whole before the split, their cases not looked at here; all but DG-43+rest
# (TO_R11) stay with WT-57.
KEPT_WHOLE = {
    *("F-01+rest", "F-07", "F-34+rest", "F-38", "DG-13+rest", "F-02+rest", "F-08", "F-14+rest"),
    *("F-24", "F-03", "F-04", "F-27", "DG-15", "F-28", "DG-05", "DG-42", "F-05a+rest", "DG-07"),
    *("DG-09", "DG-80", "F-05b+rest", "F-05c+rest", "F-05d", "F-11", "F-12", "F-13", "F-15"),
    *("F-16+rest", "F-17", "F-18", "EG-11", "F-22", "DG-65", "F-29", "F-35", "EG-04", "F-40"),
    *("F-42a", "F-42b", "F-44a", "F-44b", "F-44c", "F-45", "F-46", "EG-08", "F-47a", "F-47b"),
    *("F-47c", "DG-21", "DG-37+rest", "DG-41", "DG-43+rest", "DG-45", "DG-46+rest", "DG-47+rest"),
    *("DG-49+rest", "DG-50+rest", "DG-63", "EG-12", "EG-15", "EG-30"),
}
# Rows split by case: the cases WT-57 keeps, and those that move to <row>+WT-57-fix2.
PARTIAL = {
    "F-20": (["T5", "T1"], ["XD-062"]),
    "F-09": (["NT-057"], ["XD-060", "XD-061"]),
    "DG-20": (["NT-082"], ["XD-008"]),
    "DG-34+rest": (["NT-088"], ["XD-312"]),
    "DG-11": (["NT-002"], ["XD-105"]),
    "EG-22a": (["NT-041"], ["XD-017", "XD-040", "XD-211", "XD-223", "XD-226", "XD-209", "XD-230"]),
}
# Rows that wait for one fix whole, under their own ids.
FIX1_WHOLE = {"EG-01e", "EG-16b", "EG-16c", "EG-31", "EP-04+rest"}
FIX2_WHOLE = {
    *("F-23", "F-30+rest", "EG-01f", "EG-13", "EG-10a", "F-31", "EG-18+rest", "F-32", "EG-03"),
    *("EG-27", "EP-09+rest", "EG-01b", "EG-14e+rest", "EG-26", "EG-24"),
}
# Rows that wait for WT-57-fix1 whole after moving their existing cases to <row>+WT-57-fix2.
BOTH = {
    "EG-01a": (["NT-189"], ["XD-225", "XD-205"]),
    "DG-67": (["NT-098"], ["XD-057", "XD-058", "XD-208"]),
}
# The row R11 verifies instead of WT-57: its only case, NT-011, is not written.
TO_R11 = "DG-43+rest"
# R12 rows whose cases WT-57-fix1 writes.
R12_WAITING = {"EG-01g", "EG-22b", "EG-22c", "EG-22d", "EG-33", "EG-34", "EG-36", "EG-37", "EG-38"}

WT57_OWES = (KEPT_WHOLE - {TO_R11}) | set(PARTIAL)
FIX1_OWES = FIX1_WHOLE | set(BOTH)
FIX2_OWES = FIX2_WHOLE | {f"{row}+{FIX2}" for row in [*PARTIAL, *BOTH]}
# The existing cases WT-57's expectation table re-expects (tasks.md §3.3): WT-57-fix2 runs them.
# Not XD-313, re-expected only if R11 or R12 finds it failing (a conditional 【改】).
RE_EXPECTED = {
    *("XD-018", "XD-040", "XD-060", "XD-061", "XD-203", "XD-205", "XD-221", "XD-228", "XD-231"),
}


@pytest.fixture(scope="module")
def rows():
    return {row.id: row for row in pending_e2e.load_rows()[1]}


@pytest.fixture(scope="module")
def raw_rows():
    return {row["id"]: row for row in yaml.safe_load(pending_e2e.FEATURE_DEPS.read_text())["rows"]}


def _result(task, landed, rows):
    return pending_e2e.obligations(task, landed, rows.values())


def _owed(task, landed, rows):
    return {row.id for row in _result(task, landed, rows).owed}


# --- the rows of feature_deps.yaml --------------------------------------------------------------


def test_the_lists_add_up_to_the_89_rows_wt57_owed():
    # 61 whole rows and 6 partial ones stay, 5 + 2 go to WT-57-fix1, 15 + 6 + 2 to WT-57-fix2;
    # DG-43+rest goes to R11.
    assert (len(KEPT_WHOLE), len(PARTIAL), len(FIX1_WHOLE), len(BOTH), len(FIX2_WHOLE)) == (
        61,
        6,
        5,
        2,
        15,
    )
    assert TO_R11 in KEPT_WHOLE
    assert (len(WT57_OWES), len(FIX1_OWES), len(FIX2_OWES)) == (66, 7, 23)


@pytest.mark.parametrize("order", [(FIX1, FIX2), (FIX2, FIX1)], ids=["fix1-first", "fix2-first"])
def test_wt57_and_its_fixes_owe_their_rows_in_either_landing_order(rows, order):
    assert _owed("WT-57", LANDED, rows) == WT57_OWES
    landed = LANDED | {"WT-57"}
    for fix in order:
        assert _owed(fix, landed, rows) == {FIX1: FIX1_OWES, FIX2: FIX2_OWES}[fix]
        landed |= {fix}


def test_a_fix_landing_before_wt57_leaves_its_rows_to_wt57(rows):
    # A fix can push first only if the coordinator lets it; the rows are not lost then.
    for fix, owes in ((FIX1, FIX1_OWES), (FIX2, FIX2_OWES)):
        assert owes <= _owed("WT-57", LANDED | {fix}, rows)
        assert owes <= {row.id for row in _result("WT-57", LANDED, rows).other_fix}


def test_the_partial_rows_keep_their_one_gpu_cases(rows):
    for row_id, (kept, moved) in PARTIAL.items():
        base, split = rows[row_id], rows[f"{row_id}+{FIX2}"]
        assert list(base.cases) == kept and FIX2 not in base.deps, row_id
        assert sorted(split.cases) == sorted(moved) and split.deps == base.deps | {FIX2}, row_id


def test_the_two_way_rows_wait_for_fix1_whole_after_moving_the_existing_cases(rows, raw_rows):
    for row_id, (kept, moved) in BOTH.items():
        base, split = rows[row_id], rows[f"{row_id}+{FIX2}"]
        assert list(base.cases) == kept and raw_rows[row_id]["author"] == [FIX1], row_id
        assert FIX1 in base.deps and FIX2 not in base.deps, row_id
        assert sorted(split.cases) == sorted(moved) and split.deps == base.deps - {FIX1} | {FIX2}
    # EG-01a keeps its Hopper and unrunnable cases.
    assert raw_rows["EG-01a"]["not_runnable"] == ["XD-202"] and raw_rows["EG-01a"]["hopper_cases"]


def test_the_whole_rows_keep_their_ids_and_cases(rows, raw_rows):
    for fix, whole in ((FIX1, FIX1_WHOLE), (FIX2, FIX2_WHOLE)):
        for row_id in whole:
            assert fix in rows[row_id].deps and "WT-57" in rows[row_id].deps, row_id
            assert raw_rows[row_id]["author"] == [fix], row_id
            assert f"{row_id}+{fix}" not in rows, row_id
    # EG-26 keeps its note on XD-020, which stays unrunnable on the row.
    assert raw_rows["EG-26"]["not_runnable"] == ["XD-020"] and raw_rows["EG-26"]["notes"]
    assert list(rows["F-32"].cases)[-2:] == ["T31", "T32"] and "T23" in rows["F-23"].cases


def test_xd215_waits_for_wt57_fix2_in_eg03_and_for_wt55_fix1_in_dg81(rows):
    assert {row.id for row in rows.values() if "XD-215" in row.cases} == {"EG-03", "DG-81"}
    assert FIX2 in rows["EG-03"].deps and "WT-55-fix1" not in rows["EG-03"].deps
    assert "WT-55-fix1" in rows["DG-81"].deps and FIX2 not in rows["DG-81"].deps


def test_every_re_expected_case_waits_for_wt57_fix2(rows):
    # Worked out from the file: a re-expected case in a row WT-57-fix2 does not complete would be
    # re-expected without anybody re-running it there. R11 rows wait for it through their group.
    holding = [row for row in rows.values() if set(row.cases) & RE_EXPECTED and not row.deferred]
    assert holding
    for row in holding:
        assert FIX2 in row.deps, row.id


def test_a_fix_case_waits_for_its_fix_in_every_row(rows):
    # Worked out from the file, not from the lists above.
    for fix in (FIX1, FIX2):
        cases = {
            c for row in rows.values() if fix in row.deps and not row.assignee for c in row.cases
        }
        for row in rows.values():
            if set(row.cases) & cases and fix not in row.deps:
                # Existing cases also in rows completed earlier (DG-81's XD-215, EG-28's XD-010, ...)
                # are not re-expected; NT and T cases are written or moved by the fix and must
                # wait everywhere.
                assert not {c for c in set(row.cases) & cases if c.startswith(("NT-", "T"))}, row.id
                assert not set(row.cases) & cases & RE_EXPECTED, row.id


def _r11_rows(raw_rows):
    return {row_id for row_id, row in raw_rows.items() if "r11-prereqs" in row["deps"]}


def test_r11_waits_for_both_fixes(rows, raw_rows):
    groups = pending_e2e.load_rows()[0]
    assert groups["r11-prereqs"][-2:] == [FIX1, FIX2]
    r11_rows = _r11_rows(raw_rows)
    assert len(r11_rows) == 17
    prereqs = set(groups["r11-prereqs"])
    for missing in (FIX1, FIX2):
        result = _result("WT-60", LINK | prereqs - {missing}, rows)
        assert not r11_rows & {row.id for row in result.owed}, missing
        assert r11_rows <= {row.id for row in result.too_early}, missing
    assert r11_rows <= _owed("WT-60", LINK | prereqs, rows)


def test_r11_verifies_dg43_rest_and_wt57_does_not(rows, raw_rows):
    row = rows[TO_R11]
    assert row.assignee == "WT-60" and list(row.cases) == ["NT-011"]
    assert row.deps == LINK | {"WT-57"} and "r11-prereqs" not in raw_rows[TO_R11]["deps"]
    assert list(raw_rows[TO_R11])[:7] == [
        *("id", "features", "cases", "author", "deps", "chain", "assignee")
    ]
    assert TO_R11 not in _owed("WT-57", LANDED, rows)
    assert TO_R11 in _owed("WT-60", LANDED | {"WT-57"}, rows)
    # How R11 verifies it is a note on the row, which names WT-60.
    (note,) = raw_rows[TO_R11]["notes"]
    assert "NT-011" in note and "DQ-02" in note
    assert TO_R11 in {row.id for row in _result("WT-60", LANDED | {"WT-57"}, rows).noted}


def test_the_r12_rows_wt57_fix1_writes_wait_for_it(rows, raw_rows):
    for row_id in R12_WAITING:
        assert raw_rows[row_id]["deps"] == ["link", "WT-57", FIX1], row_id
        assert raw_rows[row_id]["author"] == [FIX1], row_id
        assert rows[row_id].assignee == "WT-61", row_id
    landed = LANDED | {"WT-57"}
    assert R12_WAITING <= {row.id for row in _result("WT-61", landed, rows).too_early}
    assert R12_WAITING <= _owed("WT-61", landed | {FIX1}, rows)
    # Worked out from the file: no other row with a fixed verifier holds a case WT-57-fix1 writes.
    fix1_cases = {c for row_id in FIX1_OWES for c in rows[row_id].cases}
    assert fix1_cases & {"NT-202"}
    holding = {r.id for r in rows.values() if r.assignee and set(r.cases) & fix1_cases}
    assert holding <= R12_WAITING
    assert {r.id for r in rows.values() if r.assignee and FIX1 in raw_rows[r.id]["deps"]} == (
        R12_WAITING
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def test_pending_on_a_constructed_history(tmp_path, capsys):
    """WT-57 lands, then its fixes; then R11 owes DG-43+rest and the R11 rows."""
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

    land("WT-30", *sorted(LANDED))
    assert owed("WT-57") & (WT57_OWES | FIX1_OWES | FIX2_OWES) == WT57_OWES
    land("WT-57")
    assert owed(FIX2) & (FIX1_OWES | FIX2_OWES) == FIX2_OWES
    land(FIX2)
    assert owed(FIX1) & (FIX1_OWES | FIX2_OWES) == FIX1_OWES
    land(FIX1)
    r11_rows = {
        row["id"]
        for row in yaml.safe_load(pending_e2e.FEATURE_DEPS.read_text())["rows"]
        if "r11-prereqs" in row["deps"]
    }
    assert len(r11_rows) == 17
    assert owed("WT-60") >= r11_rows | {TO_R11}


# --- the split, group, verifier and wait rules on a made-up table -------------------------------

TABLES = """\
### 0.2 仓库分支与合入标记
| 仓库 | 地址 | 集成分支 | 起点 | 适用工作项 |
|---|---|---|---|---|
| ds-engine 自有仓库 | x | y | z | WT-41、WT-42、WT-43 |
### 3.2 依赖集合的缩写
| 缩写 | 成员 | 用在哪里 |
|---|---|---|
| 链路 | WT-35、WT-37 | x |
| R11 前提 | WT-37、WT-57 | x |
| ds 链路 | WT-33、WT-62 | x |
### 3.6 表 D（5 个主行）
| 功能 | 用例 | 用例作者 | 依赖集合 |
|---|---|---|---|
| EG-ZZX a（✅） | NT-030、XD-311、NT-080、XD-312（Hopper 分片：XD-036） | \
WT-40（NT-030）；WT-57（NT-080、XD-312） | 链路；其余用例另等 WT-57 |
| EG-ZZY b（✅） | NT-003、NT-081、XD-082 | WT-57 | {DEPS} |
| EG-ZZZ c（✅） | XD-213 | 现成（R11 执行） | R11 前提 |
| EG-ZZW d（✅） | NT-209 | WT-57 | 链路、WT-57；R12 验收 |
| EG-ZZV e（—） | XD-001 | — | 不设 E2E：x |
### 3.7 表 C（1 个主行）
| 功能 | 用例 | 用例作者 | 依赖集合 |
|---|---|---|---|
| DS-ZZ f（✅） | NT-147 | WT-62 | ds 链路 |
"""


def _derive(splits=(), deps="链路、WT-57", verifiers=(), waits=()):
    text = TABLES.replace("{DEPS}", deps)
    rows = sync_feature_deps.derive(text, tuple(splits), tuple(verifiers), tuple(waits))
    return {row["id"]: row for row in rows}


def test_a_whole_split_of_a_rest_row_keeps_its_id():
    rows = _derive([("EG-ZZX+rest", ("*", "NT-080", "XD-312"), FIX1)])
    assert list(rows)[:2] == ["EG-ZZX", "EG-ZZX+rest"]
    assert rows["EG-ZZX+rest"]["deps"] == ["link", "WT-57", FIX1]
    assert rows["EG-ZZX+rest"]["author"] == [FIX1]
    assert rows["EG-ZZX"]["deps"] == ["link"] and rows["EG-ZZX"]["cases"] == ["NT-030", "XD-311"]
    assert rows["EG-ZZX"]["hopper_cases"] == ["XD-036"]


def test_a_partial_split_of_a_rest_row_moves_its_cases_to_a_row_named_after_both():
    rows = _derive([("EG-ZZX+rest", ("XD-312",), FIX2), ("EG-ZZX", ("XD-311",), FIX2)])
    assert list(rows)[:4] == ["EG-ZZX", "EG-ZZX+rest", f"EG-ZZX+{FIX2}", f"EG-ZZX+rest+{FIX2}"]
    assert rows["EG-ZZX+rest"]["cases"] == ["NT-080"] and rows["EG-ZZX+rest"]["deps"] == [
        "link",
        "WT-57",
    ]
    split = rows[f"EG-ZZX+rest+{FIX2}"]
    assert split["cases"] == ["XD-312"] and split["deps"] == ["link", "WT-57", FIX2]
    assert split["author"] == [FIX2] and "hopper_cases" not in split
    assert rows[f"EG-ZZX+{FIX2}"]["deps"] == ["link", FIX2]


@pytest.mark.parametrize("whole_first", [False, True])
def test_a_row_takes_a_partial_and_a_whole_split(whole_first):
    splits = [("EG-ZZY", ("XD-082",), FIX2), ("EG-ZZY", ("*", "NT-003", "NT-081"), FIX1)]
    rows = _derive(splits[::-1] if whole_first else splits)
    assert rows["EG-ZZY"]["cases"] == ["NT-003", "NT-081"] and rows["EG-ZZY"]["author"] == [FIX1]
    assert rows["EG-ZZY"]["deps"] == ["link", "WT-57", FIX1]
    assert rows[f"EG-ZZY+{FIX2}"]["cases"] == ["XD-082"]
    assert rows[f"EG-ZZY+{FIX2}"]["deps"] == ["link", "WT-57", FIX2]


@pytest.mark.parametrize(
    "splits, message",
    [
        (
            [("EG-ZZY", ("*", "NT-003"), FIX1), ("EG-ZZY", ("*", "NT-081", "XD-082"), FIX2)],
            "two splits of one row",
        ),
        ([("EG-ZZY", ("NT-003",), FIX1), ("EG-ZZY", ("NT-081",), FIX2)], "two splits of one row"),
        (
            [("EG-ZZY", ("XD-082",), FIX1), ("EG-ZZY", ("*", "NT-003", "NT-081"), FIX1)],
            "both splits name",
        ),
        (
            [("EG-ZZY", ("XD-082",), FIX2), ("EG-ZZY", ("*", "NT-003"), FIX1)],
            "marked whole, but the row also keeps",
        ),
        (
            [("EG-ZZY", ("XD-082",), FIX2), ("EG-ZZY", ("*", "NT-003", "NT-081", "XD-082"), FIX1)],
            "not among the cases",
        ),
        ([("EG-ZZY+rest", ("*", "NT-003"), FIX1)], r"split of EG-ZZY\+rest: no such row"),
        ([("EG-ZZQ+rest", ("*", "NT-003"), FIX1)], r"split of EG-ZZQ\+rest: no such row"),
        ([("EG-ZZX+rest", ("NT-030",), FIX1)], "not among the cases"),
        ([("EG-ZZX+rest", ("NT-080", "XD-312"), FIX1)], "no case would be left"),
        ([("EG-ZZX+rest", ("*", "NT-080"), FIX1)], "marked whole, but the row also keeps"),
        ([("EG-ZZX+rest", ("XD-312",), "WT-57")], "dependency of the row already"),
        ([("EG-ZZX+rest", ("XD-036",), FIX1)], "two-node or unrunnable"),
        ([("EG-ZZZ", ("*", "XD-213"), FIX1)], "has a fixed verifier"),
        ([("EG-ZZW", ("*", "NT-209"), FIX1)], "has a fixed verifier"),
    ],
    ids=[
        "two-whole",
        "two-partial",
        "same-item",
        "whole-leaves-a-case",
        "whole-takes-a-moved-case",
        "no-rest-row",
        "no-row",
        "main-case-in-rest-split",
        "rest-emptied",
        "rest-whole-incomplete",
        "rest-dependency",
        "rest-hopper",
        "r11",
        "r12",
    ],
)
def test_a_split_the_rows_leave_no_room_for_is_refused(splits, message):
    with pytest.raises(ValueError, match=message):
        _derive(splits)


def test_notes_on_cases_that_stay_do_not_block_a_split():
    deps = "链路、WT-57；补充用例 XD-020 随 PB-3（WT-72）"
    for splits in (
        [("EG-ZZY", ("*", "NT-003", "NT-081", "XD-082"), FIX2)],
        [("EG-ZZY", ("XD-082",), FIX2)],
    ):
        rows = _derive(splits, deps)
        assert rows["EG-ZZY"]["notes"] == ["补充用例 XD-020 随 PB-3（WT-72）"]
        assert all("notes" not in row for row_id, row in rows.items() if "+" in row_id)


@pytest.mark.parametrize(
    "deps",
    ["链路、WT-57；NT-081 由 WT-40 先补跑", "链路（只在 B200）、WT-57"],
    ids=["names-a-moved-case", "names-no-case"],
)
def test_notes_on_a_moved_case_or_on_no_case_block_a_split(deps):
    for cases in (("NT-081",), ("*", "NT-003", "NT-081", "XD-082")):
        with pytest.raises(ValueError, match="the row has notes"):
            _derive([("EG-ZZY", cases, FIX2)], deps)
    if "NT-081" in deps:  # a split that leaves NT-081 on the row goes through
        assert _derive([("EG-ZZY", ("XD-082",), FIX2)], deps)[f"EG-ZZY+{FIX2}"]


def test_group_additions_append_fixes_and_refuse_what_they_cannot_add():
    groups = sync_feature_deps.derive_groups(TABLES)
    added = sync_feature_deps.add_group_members(groups, {"r11-prereqs": (FIX1, FIX2)})
    assert added["r11-prereqs"] == ["WT-37", "WT-57", FIX1, FIX2]
    assert groups["r11-prereqs"] == ["WT-37", "WT-57"]  # the tables' groups are left alone
    assert sync_feature_deps.GROUP_ADDITIONS == {"r11-prereqs": (FIX1, FIX2)}
    for additions, message in (
        ({"r12": (FIX1,)}, "no such group"),
        ({"r11-prereqs": ("WT-57-fx1",)}, "not a PoC-S3 work item"),
        ({"r11-prereqs": ("WT-58",)}, "WT-58 is not a fix"),
        ({"r11-prereqs": ("WT-57",)}, "WT-57 is not a fix"),
        ({"r11-prereqs": ("WT-55-fix1",)}, "fixes WT-55, which is not a member"),
        ({"r11-prereqs": (FIX1, FIX1)}, "a member already"),
    ):
        with pytest.raises(ValueError, match=message):
            sync_feature_deps.add_group_members(groups, additions)


ZZY = ("NT-003", "NT-081", "XD-082")


def test_a_verifier_becomes_the_fixed_verifier_of_its_row_with_its_note():
    rows = _derive(verifiers=[("EG-ZZY", ZZY, "WT-60", "how")])
    assert rows["EG-ZZY"]["assignee"] == "WT-60" and rows["EG-ZZY"]["deps"] == ["link", "WT-57"]
    assert list(rows["EG-ZZY"])[:7] == [
        *("id", "features", "cases", "author", "deps", "chain", "assignee")
    ]
    assert rows["EG-ZZY"]["notes"] == ["how"] and rows["EG-ZZY"]["author"] == ["WT-57"]
    # A note of the tables stays first.
    rows = _derive(
        deps="链路、WT-57；补充用例 XD-020 随 PB-3", verifiers=[("EG-ZZY", ZZY, "WT-60", "how")]
    )
    assert rows["EG-ZZY"]["notes"] == ["补充用例 XD-020 随 PB-3", "how"]
    assert [entry[:3] for entry in sync_feature_deps.VERIFIERS] == [(TO_R11, ("NT-011",), "WT-60")]


@pytest.mark.parametrize(
    "verifiers, splits, message",
    [
        ([("EG-ZZQ", ZZY, "WT-60", "x")], (), "no such row"),
        ([("EG-ZZY", ZZY[:2], "WT-60", "x")], (), "the row's cases are"),
        ([("EG-ZZY", (*ZZY, "NT-030"), "WT-60", "x")], (), "the row's cases are"),
        ([("EG-ZZY", ZZY, "WT-65", "x")], (), "verifies no row of the tables"),
        ([("EG-ZZY", ZZY, "WT-6", "x")], (), "verifies no row of the tables"),
        ([("EG-ZZY", ZZY, "WT-60", " ")], (), "no note says how"),
        ([("EG-ZZZ", ("XD-213",), "WT-61", "x")], (), "has a fixed verifier"),
        ([("EG-ZZV", ("XD-001",), "WT-60", "x")], (), "deferred"),
        ([("DS-ZZ", ("NT-147",), "WT-60", "x")], (), "case-03"),
        (
            [("EG-ZZY", ZZY, "WT-60", "x"), ("EG-ZZY", ZZY, "WT-61", "x")],
            (),
            "two verifiers of one row",
        ),
        # A split refuses a row with a fixed verifier; the verifier cannot come after the split.
        (
            [("EG-ZZY", ZZY[:2], "WT-60", "x")],
            [("EG-ZZY", ("XD-082",), FIX2)],
            "a split names or adds",
        ),
        (
            [(f"EG-ZZY+{FIX2}", ("XD-082",), "WT-60", "x")],
            [("EG-ZZY", ("XD-082",), FIX2)],
            "a split names",
        ),
        ([("EG-ZZY", ZZY, "WT-60", "x")], [("EG-ZZY", ("*", *ZZY), FIX2)], "a split"),
        (
            [("EG-ZZX+rest", ("NT-080",), "WT-60", "x")],
            [("EG-ZZX+rest", ("XD-312",), FIX2)],
            "a split names",
        ),
    ],
    ids=[
        *("no-row", "fewer-cases", "more-cases", "unknown-verifier", "malformed", "no-note"),
        *("has-one", "deferred", "case-03", "twice", "split-row", "split-product"),
        *("whole-split-row", "split-rest-row"),
    ],
)
def test_a_verifier_the_row_cannot_take_is_refused(verifiers, splits, message):
    with pytest.raises(ValueError, match=message):
        _derive(splits, verifiers=verifiers)


ZZW, ZZZ = ("NT-209",), ("XD-213",)


def test_a_wait_makes_a_row_with_a_fixed_verifier_also_wait_for_the_item():
    rows = _derive(waits=[("EG-ZZW", ZZW, FIX1), ("EG-ZZZ", ZZZ, "WT-55-fix1")])
    assert rows["EG-ZZW"]["deps"] == ["link", "WT-57", FIX1]
    assert rows["EG-ZZW"]["assignee"] == "WT-61" and rows["EG-ZZW"]["author"] == [FIX1]
    assert rows["EG-ZZZ"]["deps"] == ["r11-prereqs", "WT-55-fix1"]
    assert {(row_id, item) for row_id, _, item in sync_feature_deps.ROW_WAITS} == {
        (row_id, FIX1) for row_id in R12_WAITING
    }


@pytest.mark.parametrize(
    "waits, message",
    [
        ([("EG-ZZQ", ZZW, FIX1)], "no such row"),
        ([("EG-ZZW", (), FIX1)], "the row's cases are"),
        ([("EG-ZZW", ("NT-209", "NT-210"), FIX1)], "the row's cases are"),
        ([("EG-ZZY", ZZY, FIX1)], "has no fixed verifier"),
        ([("EG-ZZV", ("XD-001",), FIX1)], "has no fixed verifier"),  # a deferred row has none
        ([("EG-ZZW", ZZW, "WT-57-fx1")], "not a PoC-S3 work item"),
        ([("EG-ZZW", ZZW, "WT-61")], "is the row's verifier"),
        ([("EG-ZZW", ZZW, "WT-57")], "dependency of the row already"),
        ([("EG-ZZZ", ZZZ, "WT-37")], "dependency of the row already"),  # through R11 前提
        ([("EG-ZZZ", ZZZ, FIX2)], "dependency of the row already"),  # through GROUP_ADDITIONS
        ([("EG-ZZW", ZZW, FIX1), ("EG-ZZW", ZZW, FIX2)], "two waits of one row"),
    ],
    ids=[
        *("no-row", "no-cases", "more-cases", "no-verifier", "deferred", "malformed"),
        *("own-verifier", "dependency", "group", "added", "twice"),
    ],
)
def test_a_wait_the_row_cannot_take_is_refused(waits, message):
    with pytest.raises(ValueError, match=message):
        _derive(waits=waits)


def test_a_performance_row_cannot_wait():
    text = TABLES.replace("{DEPS}", "链路、WT-57").replace(
        "| NT-209 | WT-57 | 链路、WT-57；R12 验收 |",
        "| NT-209、PU-07 | WT-57 | 链路、WT-57；R12 验收；PU-07 取证（WT-54） |",
    )
    rows = {row["id"]: row for row in sync_feature_deps.derive(text)}
    assert rows["EG-ZZW+PU-07"]["assignee"] == "WT-54"
    with pytest.raises(ValueError, match="the row is performance evidence"):
        sync_feature_deps.derive(text, waits=(("EG-ZZW+PU-07", ("PU-07",), FIX1),))


def test_a_row_verifiers_gives_a_verifier_can_also_wait():
    rows = _derive(verifiers=[("EG-ZZY", ZZY, "WT-60", "x")], waits=[("EG-ZZY", ZZY, FIX1)])
    assert rows["EG-ZZY"]["assignee"] == "WT-60" and rows["EG-ZZY"]["deps"][-1] == FIX1


@pytest.mark.parametrize(
    "split, cases",
    [(("EG-ZZY", ("XD-082",), FIX2), ZZY[:2]), (("EG-ZZY", ("*", *ZZY), FIX2), ZZY)],
    ids=["partial", "whole"],
)
def test_a_split_row_cannot_wait(split, cases):
    # It has no fixed verifier: a split refuses one, and VERIFIERS refuses the split rows.
    with pytest.raises(ValueError, match="has no fixed verifier"):
        _derive([split], waits=[("EG-ZZY", cases, FIX1)])


@pytest.mark.parametrize("deps", ["不设 E2E：x", "链路；R12 验收", "链路（只在 B200）、WT-57"])
def test_a_missing_rest_row_is_reported_as_missing_first(deps):
    with pytest.raises(ValueError, match=r"split of EG-ZZY\+rest: no such row"):
        _derive([("EG-ZZY+rest", ("*", "NT-003"), FIX1)], deps)


@pytest.mark.parametrize("spelling", ["WT-57-fix1", "WT-57-FIX1", "WT-57 fix1", "WT-57（fix1）"])
def test_a_fix_in_the_group_table_is_refused(spelling):
    with pytest.raises(ValueError, match="fixes go in GROUP_ADDITIONS"):
        sync_feature_deps.derive_groups(TABLES.replace("WT-37、WT-57 |", f"WT-37、{spelling} |"))


@pytest.mark.parametrize(
    "old, new, message",
    [
        ("WT-57（NT-080、XD-312）", "WT-57-fix1（NT-080、XD-312）", "names a fix"),
        ("其余用例另等 WT-57", "其余用例另等 WT-57-fix1", "names a fix"),
        ("| NT-209 | WT-57 |", "| NT-209 | WT-57 Fix1 |", "names a fix"),
        ("WT-41、WT-42、WT-43", "WT-41、WT-42-fix1", "repository's items name a fix"),
    ],
    ids=["author", "rest-clause", "author-spelling", "ds-repo"],
)
def test_a_fix_in_the_other_tables_is_refused(old, new, message):
    text = TABLES.replace("{DEPS}", "链路、WT-57")
    sync_feature_deps.derive_groups(text)
    sync_feature_deps.derive(text)
    with pytest.raises(ValueError, match=message):
        sync_feature_deps.derive_groups(text.replace(old, new))
        sync_feature_deps.derive(text.replace(old, new))


def test_applying_splits_leaves_the_authors_it_is_given_alone():
    authors = {"NT-003": "WT-57", "NT-081": "WT-57", "XD-082": "existing"}
    before = dict(authors)
    _, deps, _, after = sync_feature_deps._apply_splits(
        "EG-ZZY", "EG-ZZY", "case-04", list(authors), authors, ["link"], [(("*", *authors), FIX1)]
    )
    assert authors == before and after == dict.fromkeys(authors, FIX1) and deps == ["link", FIX1]


# --- ownership ----------------------------------------------------------------------------------

APPS = "tests/unittest/llmapi/apps"
CHAIN = "tests/integration/defs/openengine_chain"
POC_TESTS = "tests/unittest/serve/openengine_poc"


@pytest.mark.parametrize("fix", [FIX1, FIX2])
def test_the_fixes_own_what_wt57_owns_and_their_own_files(fix):
    for path in (
        f"{APPS}/_test_openengine_disagg_spec.py",
        f"{APPS}/_test_openengine_disagg_per_role.py",
        f"{APPS}/_test_openengine_disagg_common.py",
        f"{POC_TESTS}/wt57/test_poc3_wt57_two_forms.py",
        f"{CHAIN}/expectations/{fix}.py",
        f"{CHAIN}/verified/{fix}.yaml",
        f"{CHAIN}/smoke/{fix}.txt",
        f"{CHAIN}/new_cases/{fix}.txt",
        f"{CHAIN}/migrated/{fix}.txt",
    ):
        assert check_ownership.match_rule(fix, path) is not None, path
        assert check_ownership.match_rule("WT-57", path) is not None or fix in path, path


@pytest.mark.parametrize("fix, other", [(FIX1, FIX2), (FIX2, FIX1)])
def test_a_fix_owns_neither_the_other_fixs_files_nor_other_items_files(fix, other):
    for path in (
        f"{CHAIN}/expectations/{other}.py",
        f"{CHAIN}/verified/{other}.yaml",
        f"{CHAIN}/new_cases/{other}.txt",
        f"{APPS}/_test_openengine_disagg_rejects.py",  # WT-52
        f"{APPS}/_test_openengine_disagg_gen_first.py",  # WT-55
        f"{APPS}/_test_openengine_ds_disagg_basic.py",  # WT-56, then WT-62
        f"{POC_TESTS}/wt52/test_poc3_wt52_rejects.py",
        f"{CHAIN}/split.py",  # WT-37
        "scripts/poc3/feature_deps.yaml",  # WT-30 and its fixes only
        "tensorrt_llm/serve/openengine_poc/facade.py",
    ):
        assert check_ownership.match_rule(fix, path) is None, path
    assert check_ownership.match_rule("WT-57", f"{CHAIN}/expectations/{fix}.py") is None

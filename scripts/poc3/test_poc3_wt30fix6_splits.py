# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of the splits that make GPU-layer NT cases wait for WT-52-fix1 and WT-53-fix2.

The rows whose dependencies are exactly the link and WT-52 (or WT-53) are
completed by WT-37's landing. Their GPU-layer NT cases (the layer column of
the plan's functional tests, §6.3) also wait for WT-52-fix1 (WT-53-fix2);
WT-37 keeps the rows' XD, T and CPU-layer NT cases. Run from the repository root::

    python3 -m pytest -p no:cacheprovider scripts/poc3/
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

# The scripts import poc3_common themselves; pytest puts this directory on sys.path.
import pending_e2e  # noqa: I001
import pytest
import sync_feature_deps
import yaml
from poc3_common import parse_task

# The same set as in test_poc3_scripts.py, kept here so that this module imports no other test
# module.
LINK = {"WT-35", "WT-37", "WT-38", "WT-40", "WT-50", "WT-51"}
# Landed items the obligations below depend on, WT-37 excepted (the integration branch also has
# items no row here depends on).
BEFORE_WT37 = (LINK - {"WT-37"}) | {"WT-51-fix1", "WT-52", "WT-53", "WT-54"}

# The GPU-layer NT cases of those rows: layer GPU-直连 or GPU-E2E-case04 in §6.3 of the plan's
# 02_functional-tests.md (test_the_gpu_cases_are_the_gpu_layer_of_the_plan checks it there).
GPU_CASES = {
    *("NT-055", "NT-058", "NT-064", "NT-065", "NT-066", "NT-069", "NT-073", "NT-081", "NT-083"),
    *("NT-084", "NT-085", "NT-086", "NT-087", "NT-089", "NT-090", "NT-092", "NT-180", "NT-181"),
    *("NT-182", "NT-183", "NT-184", "NT-185", "NT-186", "NT-187", "NT-190", "NT-191", "NT-192"),
    *("NT-197", "NT-201", "NT-204", "NT-216", "NT-040"),
}
# The cases each of those rows keeps for WT-37; the rows in WHOLE_ROWS keep none and wait for the
# fix under their own ids.
KEPT = {
    "F-25": ["XD-219", "T2"],
    "F-30": ["XD-014", "T2", "T1"],
    "F-34": ["XD-014", "XD-059", "XD-208"],
    "F-43": ["XD-120"],
    "DG-19": ["XD-025", "XD-007", "XD-107", "XD-058", "XD-121", "XD-123", "XD-059"],
    "DG-22": ["NT-004"],
    "DG-24": ["XD-026", "XD-119"],
    "DG-35": ["NT-004"],
    "DG-38": ["XD-055", "NT-009", "XD-073", "XD-125"],
    "DG-39c": ["XD-055", "NT-009"],
    "DG-84": ["NT-020", "NT-031"],
    "DG-85": ["NT-021"],
    "DG-86": ["NT-022"],
    "DG-87": ["XD-110", "XD-111", "XD-114"],
    "DG-88": ["NT-023", "NT-032"],
    "DG-90": ["NT-024", "NT-033"],
    "EG-05b": ["NT-025"],
    "EG-06": ["XD-219", "NT-026"],
    "EG-07": ["NT-028", "NT-034"],
    "EG-16a": ["XD-055"],
    "EG-19": ["NT-027"],
    "EG-21": ["NT-035", "NT-171"],
    "EP-01": ["NT-172", "NT-173"],
}
WHOLE_ROWS = {
    "WT-52-fix1": ["F-10", "F-19", "F-26", "DG-39", "EG-05a", "EG-10b", "EG-29"],
    "WT-53-fix2": ["F-06", "F-39", "DG-23", "DG-25", "DG-89"],
}
PARTIAL_ROWS = {
    "WT-52-fix1": [
        "F-25", "DG-38", "DG-39c", "DG-84", "DG-85", "DG-86", "DG-87", "DG-88", "DG-90",
        "EG-05b", "EG-07", "EG-16a", "EG-19",
    ],
    "WT-53-fix2": ["F-43", "DG-19", "DG-22", "DG-24", "DG-35", "EG-21"],
}  # fmt: skip
UNSPLIT_ROWS = {"F-30", "F-34", "EG-06", "EP-01"}  # no GPU-layer NT case
SOURCE_ROWS = set(KEPT) | {row for rows in WHOLE_ROWS.values() for row in rows}
FIX_OF = {"WT-52-fix1": "WT-52", "WT-53-fix2": "WT-53"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    return repo


def _land(repo: Path, *items: str) -> None:
    for item in items:
        _git(repo, "commit", "-q", "--allow-empty", "-m", f"{item}\n\nPoC3-Task: {item}")


@pytest.fixture(scope="module")
def rows():
    return {row.id: row for row in pending_e2e.load_rows()[1]}


@pytest.fixture(scope="module")
def raw_rows():
    return {row["id"]: row for row in yaml.safe_load(pending_e2e.FEATURE_DEPS.read_text())["rows"]}


def _owed(task, landed, rows):
    return [row.id for row in pending_e2e.obligations(task, landed, rows.values()).owed]


def _fix_rows(fix):
    return set(WHOLE_ROWS[fix]) | {f"{row}+{fix}" for row in PARTIAL_ROWS[fix]}


# --- the rows of feature_deps.yaml --------------------------------------------------------------


def test_the_splits_of_the_fixes_move_the_gpu_cases_of_the_source_rows(rows):
    splits = {}
    for row_id, cases, item in sync_feature_deps.SPLITS:
        splits[row_id] = (*sync_feature_deps.split_cases(cases), item)
    fixed = {row for row, (_, _, item) in splits.items() if item in FIX_OF}
    assert fixed == SOURCE_ROWS - UNSPLIT_ROWS
    for row_id in SOURCE_ROWS:
        base = rows[row_id]
        if row_id in UNSPLIT_ROWS:
            assert list(base.cases) == KEPT[row_id] and base.deps == LINK | {"WT-52"}
            continue
        whole, moved, item = splits[row_id]
        assert set(moved) <= GPU_CASES and whole == (row_id in WHOLE_ROWS[item])
        if whole:
            assert row_id not in KEPT and set(base.cases) == set(moved)
            assert base.deps == LINK | {FIX_OF[item], item}
        else:
            assert row_id in PARTIAL_ROWS[item] and list(base.cases) == KEPT[row_id]
            assert base.deps == LINK | {FIX_OF[item]}
            split = rows[f"{row_id}+{item}"]
            assert set(split.cases) == set(moved) and split.deps == LINK | {FIX_OF[item], item}
    assert not GPU_CASES & {case for cases in KEPT.values() for case in cases}


def test_no_link_wt52_or_wt53_row_has_a_gpu_case_left(rows):
    # Worked out from the file, not from the lists above: a row added to the tables is covered too.
    completed = [
        row
        for row in rows.values()
        if row.chain == "case-04" and row.deps in (LINK | {"WT-52"}, LINK | {"WT-53"})
    ]
    assert {row.id for row in completed} == set(KEPT)
    for row in completed:
        assert not GPU_CASES & set(row.cases), row.id


def test_wt37_owes_the_cases_the_source_rows_keep(rows):
    owed = set(_owed("WT-37", BEFORE_WT37, rows))
    assert owed & SOURCE_ROWS == set(KEPT)
    assert not owed & (_fix_rows("WT-52-fix1") | _fix_rows("WT-53-fix2"))
    for row_id in KEPT:
        assert list(rows[row_id].cases) == KEPT[row_id]


def test_only_the_split_cases_wait_for_a_fix(rows):
    waiting = {row.id for row in rows.values() if any(parse_task(d)[1] for d in row.deps)}
    assert waiting == _fix_rows("WT-52-fix1") | _fix_rows("WT-53-fix2") | {
        "DG-16+WT-51-fix2",
        "DG-82+WT-51-fix2",
    }


def test_a_case_waits_for_a_fix_in_every_row_or_in_none(rows):
    waiting, not_waiting = set(), set()
    for row in rows.values():
        fixed = any(parse_task(d)[1] for d in row.deps)
        (waiting if fixed else not_waiting).update(c for c in row.cases if c.startswith("NT-"))
    assert not waiting & not_waiting


def test_the_gpu_cases_are_the_gpu_layer_of_the_plan():
    tasks_md = os.environ.get("POC3_TASKS_MD")
    doc = (
        Path(tasks_md).parents[1] / "02_poc-goals" / "02_functional-tests.md" if tasks_md else None
    )
    if doc is None or not doc.is_file():
        pytest.skip("POC3_TASKS_MD does not point into the plan")
    layer = {}
    for line in doc.read_text().splitlines():
        found = re.match(r"\| (NT-\d{3}) \| ([^|]+?) \|", line)
        if found:
            layer.setdefault(found.group(1), found.group(2))
    rows = yaml.safe_load(pending_e2e.FEATURE_DEPS.read_text())["rows"]
    cases = {c for row in rows if row["id"].split("+")[0] in SOURCE_ROWS for c in row["cases"]}
    nt = {case for case in cases if case.startswith("NT-")}
    assert {case for case in nt if layer[case].startswith("GPU")} == GPU_CASES


@pytest.mark.parametrize("fix", sorted(FIX_OF))
def test_the_fix_owes_its_rows_whole_or_split_once_wt37_has_landed(rows, fix):
    landed = BEFORE_WT37 | {"WT-37"}
    # Whole rows keep their ids, so nobody loses them; WT-37 does not owe them.
    assert set(_owed(fix, landed, rows)) == _fix_rows(fix)
    assert not _fix_rows(fix) & set(_owed("WT-37", BEFORE_WT37, rows))
    other = next(name for name in FIX_OF if name != fix)
    assert not _fix_rows(fix) & set(_owed(other, landed, rows))
    # A fix landing before WT-37 leaves the rows to WT-37, as any dependency would.
    assert _fix_rows(fix) <= set(_owed("WT-37", BEFORE_WT37 | {fix}, rows))


def test_the_item_of_each_fix_hears_of_the_rows_waiting_for_it(rows):
    for fix, item in FIX_OF.items():
        result = pending_e2e.obligations(item, LINK, rows.values())
        assert {row.id for row in result.other_fix} == _fix_rows(fix)
    # test_poc3_scripts.py uses WT-54 as an item that no row waits for a fix of.
    fixes_of_wt54 = [
        name for row in rows.values() for name in row.deps if parse_task(name)[0] == "WT-54"
    ]
    assert set(fixes_of_wt54) <= {"WT-54"}
    assert pending_e2e.obligations("WT-54", LINK, rows.values()).other_fix == []


def test_the_supplementary_cases_stay_with_dg19_and_dg24(raw_rows):
    assert raw_rows["DG-19"]["hopper_cases"] == ["XD-049", "XD-122"]
    assert raw_rows["DG-24"]["hopper_cases"] == ["XD-050", "XD-051"]
    assert raw_rows["DG-24"]["not_runnable"] == ["XD-118"]
    for row_id in ("DG-19", "DG-24"):
        assert raw_rows[row_id]["deps"] == ["link", "WT-53"]
        split = raw_rows[f"{row_id}+WT-53-fix2"]
        assert not {"hopper_cases", "two_node_cases", "not_runnable"} & set(split)


def test_pending_on_a_constructed_history(tmp_path, capsys):
    """WT-37 owes the rows keeping cases for it; a fix landing after WT-37 owes its rows."""
    repo = _repo(tmp_path)
    _land(repo, "WT-30", *sorted(BEFORE_WT37))

    def pending(task):
        argv = ["--repo", str(repo), "--ref", "main", "--task", task, "--json"]
        assert pending_e2e.main(argv) == 0
        return yaml.safe_load(capsys.readouterr().out)

    owed = set(pending("WT-37")["owed"])
    assert owed & SOURCE_ROWS == set(KEPT)
    assert not owed & (_fix_rows("WT-52-fix1") | _fix_rows("WT-53-fix2"))
    _land(repo, "WT-37")
    for fix in sorted(FIX_OF):
        assert set(pending(fix)["owed"]) == _fix_rows(fix)
        _land(repo, fix)


# --- the split rules on a made-up table ---------------------------------------------------------

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
| ds 链路 | WT-33（探针通过）、WT-62 | x |
### 3.5 表 B（2 个主行）
| 功能 | 用例 | 用例作者 | 依赖集合 |
|---|---|---|---|
| DG-ZZY x（04：✅） | 04：NT-030、NT-080、XD-312【改】 | \
04：WT-40（NT-030）；WT-51（NT-080）；WT-54（XD-312） | 04：链路；其余用例另等 WT-54 |
| DG-ZZZ y（04：✅） | 04：NT-003、NT-081、XD-082{NOTE} | 04：WT-51 | 04：{DEPS} |
"""
ALL_ZZZ = ("*", "NT-003", "NT-081", "XD-082")


def _derive(tmp_path, split, note="", deps="链路"):
    text = TABLES.replace("{NOTE}", note).replace("{DEPS}", deps)
    rows = sync_feature_deps.derive(text, (split,))
    path = tmp_path / "feature_deps.yaml"
    path.write_text(sync_feature_deps.render(sync_feature_deps.derive_groups(text), rows))
    return {row["id"]: row for row in rows}, {row.id: row for row in pending_e2e.load_rows(path)[1]}


def test_a_whole_split_keeps_the_row_and_its_debt(tmp_path):
    raw, rows = _derive(tmp_path, ("DG-ZZZ", ALL_ZZZ, "WT-51-fix2"))
    assert "DG-ZZZ+WT-51-fix2" not in raw
    assert raw["DG-ZZZ"]["cases"] == ["NT-003", "NT-081", "XD-082"]
    assert raw["DG-ZZZ"]["author"] == ["WT-51-fix2"]
    assert raw["DG-ZZZ"]["deps"] == ["link", "WT-51-fix2"]
    # The link's last item no longer owes the row; the fix does when it lands after it.
    assert "DG-ZZZ" not in _owed("WT-37", {"WT-35"}, rows)
    assert "DG-ZZZ" in _owed("WT-51-fix2", {"WT-35", "WT-37"}, rows)
    assert "DG-ZZZ" in _owed("WT-37", {"WT-35", "WT-51-fix2"}, rows)


def test_a_partial_split_leaves_the_dependencies_of_the_row_alone(tmp_path):
    raw, _ = _derive(tmp_path, ("DG-ZZZ", ("NT-081",), "WT-51-fix2"))
    assert raw["DG-ZZZ"]["cases"] == ["NT-003", "XD-082"]
    assert raw["DG-ZZZ"]["deps"] == ["link"]
    assert raw["DG-ZZZ+WT-51-fix2"]["deps"] == ["link", "WT-51-fix2"]


def test_a_whole_split_leaves_the_rest_row_alone(tmp_path):
    raw, _ = _derive(tmp_path, ("DG-ZZY", ("*", "NT-030", "NT-080"), "WT-51-fix2"))
    assert list(raw) == ["DG-ZZY", "DG-ZZY+rest", "DG-ZZZ"]
    assert raw["DG-ZZY"]["deps"] == ["link", "WT-51-fix2"]
    assert raw["DG-ZZY+rest"]["deps"] == ["link", "WT-54"]
    assert raw["DG-ZZY+rest"]["cases"] == ["XD-312"]


@pytest.mark.parametrize(
    "note, key",
    [
        ("（Hopper 分片：XD-036）", "hopper_cases"),
        ("（2 节点组：XD-301）", "two_node_cases"),
        ("（PoC 机型不可跑：XD-202）", "not_runnable"),
    ],
)
def test_supplementary_cases_stay_on_the_row_for_a_part_or_a_whole_split(tmp_path, note, key):
    case = note[1:-1].split("：")[1]
    raw, _ = _derive(tmp_path, ("DG-ZZZ", ("NT-081",), "WT-51-fix2"), note)
    assert raw["DG-ZZZ"][key] == [case] and key not in raw["DG-ZZZ+WT-51-fix2"]
    raw, _ = _derive(tmp_path, ("DG-ZZZ", ALL_ZZZ, "WT-51-fix2"), note)
    assert raw["DG-ZZZ"][key] == [case]


@pytest.mark.parametrize(
    "split, note, message",
    [
        (("DG-ZZZ", ALL_ZZZ[1:], "WT-51-fix2"), "", "no case would be left"),
        (("DG-ZZZ", ALL_ZZZ[:3], "WT-51-fix2"), "", r"marked whole, but the row also keeps"),
        (("DG-ZZZ", ("*",), "WT-51-fix2"), "", "no case to move"),
        (("DG-ZZZ", ("NT-003", "*"), "WT-51-fix2"), "", "'\\*' not first"),
        (("DG-ZZZ", ("*", "NT-081", "NT-081"), "WT-51-fix2"), "", "a case listed twice"),
        (("DG-ZZZ", ("*", *ALL_ZZZ[1:], "NT-099"), "WT-51-fix2"), "", "not among the cases"),
        (("DG-ZZY", ("*", "NT-030", "NT-080", "XD-312"), "WT-51-fix2"), "", "not among the"),
        (("DG-ZZY", ("*", "NT-030", "NT-080"), "WT-54"), "", "dependency of the row already"),
        (("DG-ZZY", ("NT-080",), "WT-54"), "", "dependency of the row already"),
    ]
    + [
        (("DG-ZZZ", cases, "WT-51-fix2"), f"（{kind}：NT-081）", "two-node or unrunnable")
        for cases in (("NT-081",), ALL_ZZZ)
        for kind in ("Hopper 分片", "2 节点组", "PoC 机型不可跑")
    ],
)
def test_a_split_that_does_not_say_what_it_moves_is_refused(tmp_path, split, note, message):
    with pytest.raises(ValueError, match=message):
        _derive(tmp_path, split, note)


@pytest.mark.parametrize(
    "deps, item, message",
    [
        ("链路", "WT-35", "dependency of the row already"),
        ("链路、WT-52", "WT-52", "dependency of the row already"),
        ("链路", "WT-51-fx2", "not a PoC-S3 work item"),
        ("不设 E2E：x", "WT-51-fix2", "deferred, has a fixed verifier"),
        ("R11 前提", "WT-51-fix2", "deferred, has a fixed verifier"),
        ("链路；R12 验收", "WT-51-fix2", "deferred, has a fixed verifier"),
        ("链路；由 WT-73 补", "WT-51-fix2", "deferred, has a fixed verifier"),
        ("链路；PU-02 取证（WT-70）", "WT-51-fix2", "performance evidence"),
    ],
    ids=["group-member", "dependency", "bad-item", "deferred", "r11", "r12", "fallback", "pu"],
)
def test_a_whole_split_is_refused_where_a_partial_one_is(tmp_path, deps, item, message):
    for cases in (ALL_ZZZ, ("NT-081",)):
        with pytest.raises(ValueError, match=message):
            _derive(tmp_path, ("DG-ZZZ", cases, item), deps=deps)


def test_a_row_with_notes_cannot_be_split_even_whole(tmp_path):
    for cases in (("NT-081",), ALL_ZZZ):
        with pytest.raises(ValueError, match="the row has notes"):
            _derive(tmp_path, ("DG-ZZZ", cases, "WT-51-fix2"), "（只在 B200：NT-081）")

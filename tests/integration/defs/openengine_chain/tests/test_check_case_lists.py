# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The new-test and migrated-test lists against the functional test plan."""

import os

import pytest
from openengine_chain import check_case_lists as cc

pytestmark = pytest.mark.cpu_only

_E2E = "unittest/llmapi/apps/_test_openengine_disagg_basic.py::test_chat"
_DS_E2E = "unittest/llmapi/apps/_test_openengine_ds_disagg_basic.py::test_chat"
_CPU = "openengine_chain/tests/test_proof_disagg.py::test_orphan_handoffs_are_counted"
_CHAT = "unittest/llmapi/apps/_test_openai_chat.py"
_PLAN = f"""
| # | 层 | 拟定 nodeid | 覆盖的行 |
|---|---|---|---|
| NT-024 | CPU-fake链 | `{_CPU}` | DG-90 |
| NT-050 | GPU-E2E-case04 | `{_E2E}`（参数 `a`、`b`） | F-01 |
| NT-100 | GPU-E2E-case03 | `{_DS_E2E}` | F-01 |
| NT-205 | — | `unittest/llmapi/apps/_test_openengine_disagg_gone.py::test_x` | 废止 |

| F-01 | **【搬迁】** T2 `{_CHAT}::test_single_chat_session`、`::test_chat_streaming`，搬自 agg |
| F-02 | **【搬迁】** XD-704 `unittest/llmapi/apps/_test_openai_lora.py`，搬自 agg |
| F-03 | **【搬迁】** = 测试函数本体不动 |
"""


def _write(root, directory, name, *lines):
    os.makedirs(root / directory, exist_ok=True)
    (root / directory / name).write_text(
        "# SPDX-License-Identifier: Apache-2.0\n" + "".join(f"{line}\n" for line in lines)
    )


def test_the_plan_is_read():
    assert cc.plan_new_tests(_PLAN) == {
        _CPU: ("NT-024", "CPU-fake链"),
        _E2E: ("NT-050", "GPU-E2E-case04"),
        _DS_E2E: ("NT-100", "GPU-E2E-case03"),
    }
    assert cc.plan_migrated_tests(_PLAN) == {
        "unittest/llmapi/apps/_test_openai_chat.py::test_single_chat_session": "T2",
        "unittest/llmapi/apps/_test_openai_chat.py::test_chat_streaming": "T2",
        "unittest/llmapi/apps/_test_openai_lora.py": "XD-704",
    }


def test_lists_that_agree_with_the_plan_pass(tmp_path):
    _write(tmp_path, cc.NEW_DIR, "WT-38.txt", f"NT-024 {_CPU}", _E2E)
    _write(
        tmp_path,
        cc.MIGRATED_DIR,
        "WT-40-fix1.txt",
        "T-2 unittest/llmapi/apps/_test_openai_chat.py::test_chat_streaming[pytorch]",
        "unittest/llmapi/apps/_test_openai_lora.py::test_lora[x]  # by file",
    )
    assert cc.check(_PLAN, root=str(tmp_path)) == []


@pytest.mark.parametrize(
    "new, migrated, expected",
    [
        ([_E2E.replace("basic", "other")], [], "not a new test of the plan"),
        ([f"NT-050 {_CPU}"], [], "the plan's id is NT-024"),
        (["unittest/llmapi/apps/_test_openengine_disagg_gone.py::test_x"], [], "not a new test"),
        (
            [],
            ["unittest/llmapi/apps/_test_openai_chat.py::test_multi_turn_dialogue"],
            "not a migrated",
        ),
        (
            [],
            ["unittest/llmapi/apps/_test_openai_chat.py::test_chat_streaming_x"],
            "not a migrated",
        ),
        ([], ["XD-704 unittest/llmapi/apps/_test_openai_chat.py::test_chat_streaming"], "'T2'"),
        ([_E2E], [_E2E], "listed in both"),
    ],
)
def test_lists_that_disagree_with_the_plan_fail(tmp_path, new, migrated, expected):
    _write(tmp_path, cc.NEW_DIR, "WT-38.txt", *new)
    _write(tmp_path, cc.MIGRATED_DIR, "WT-40.txt", *migrated)
    problems = cc.check(_PLAN, root=str(tmp_path))
    assert any(expected in p for p in problems), problems


def test_a_planned_end_to_end_test_outside_its_file_fails(tmp_path):
    misplaced = "unittest/llmapi/apps/_test_openai_x.py::test_y"
    plan = _PLAN + f"| NT-051 | GPU-E2E-case04 | `{misplaced}` | F-01 |\n"
    _write(tmp_path, cc.NEW_DIR, "WT-38.txt", misplaced)
    problems = cc.check(plan, root=str(tmp_path))
    assert sum("_test_openengine_disagg_" in p for p in problems) == 2, problems


def test_complete_requires_every_planned_test(tmp_path):
    _write(tmp_path, cc.NEW_DIR, "WT-38.txt", _CPU)
    problems = cc.check(_PLAN, root=str(tmp_path), complete=True)
    assert sorted(p.split(" is in no ")[0] for p in problems) == sorted(
        [
            f"plan: new test NT-050 {_E2E}",
            f"plan: new test NT-100 {_DS_E2E}",
            "plan: migrated test T2 unittest/llmapi/apps/_test_openai_chat.py::test_single_chat_session",
            "plan: migrated test T2 unittest/llmapi/apps/_test_openai_chat.py::test_chat_streaming",
            "plan: migrated test XD-704 unittest/llmapi/apps/_test_openai_lora.py",
        ]
    )
    assert cc.check(_PLAN, root=str(tmp_path)) == []


@pytest.mark.parametrize("plan", ["", "| NT-024 | CPU-fake链 | not a node id |"])
def test_a_plan_without_marks_fails(tmp_path, plan):
    _write(tmp_path, cc.NEW_DIR, "WT-38.txt", _CPU)
    problems = cc.check(plan, root=str(tmp_path))
    assert any("no new test" in p for p in problems)
    assert any("no migrated test" in p for p in problems)


def test_no_lists_fail(tmp_path):
    assert cc.check(_PLAN, root=str(tmp_path)) == [
        "no test is listed in new_cases/*.txt or migrated/*.txt"
    ]


def test_a_list_not_named_after_a_work_item_fails(tmp_path):
    _write(tmp_path, cc.NEW_DIR, "WT-38.txt", _CPU)
    _write(tmp_path, cc.NEW_DIR, "notes.txt", _E2E)
    problems = cc.check(_PLAN, root=str(tmp_path))
    assert len(problems) == 1 and "named after its work item" in problems[0]


def test_a_malformed_line_raises(tmp_path):
    _write(tmp_path, cc.NEW_DIR, "WT-38.txt", f"NT-024 {_CPU} extra")
    with pytest.raises(ValueError, match="WT-38.txt:2"):
        cc.check(_PLAN, root=str(tmp_path))


def test_main_reports_pass_and_fail(tmp_path, capsys):
    doc = tmp_path / "plan.md"
    doc.write_text("")
    assert cc.main(["--doc", str(doc)]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "FAIL"


def test_the_checked_in_lists_agree_with_the_plan_rows_they_name():
    """Every list in this checkout names a test of a plan holding exactly its rows."""
    rows = []
    for directory, marker in ((cc.NEW_DIR, "new"), (cc.MIGRATED_DIR, "migrated")):
        entries, problems = cc.read_lists(os.path.join(os.path.dirname(cc.__file__), directory))
        assert problems == []
        for entry in entries:
            if marker == "new":
                rows.append(f"| {entry.case_id or 'NT-999'} | CPU-fake链 | `{entry.nodeid}` | x |")
            else:
                rows.append(f"**【搬迁】** {entry.case_id or 'T1'} `{entry.nodeid}`")
    assert rows, "the WT-38 list itself is checked in"
    plan = "\n".join(rows) + "\n**【搬迁】** T1 `x.py::y`\n"
    assert cc.check(plan) == []

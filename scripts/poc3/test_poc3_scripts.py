# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of the PoC-S3 gate scripts.

Run from the repository root::

    python3 -m pytest -p no:cacheprovider scripts/poc3/test_poc3_scripts.py

``POC3_TASKS_MD`` (the plan's tasks.md) enables the check that
``feature_deps.yaml`` matches the plan's tables; without it that test skips.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path

# The scripts import poc3_common themselves; pytest puts this directory on sys.path.
import check_invariants  # noqa: I001
import check_ownership
import pending_e2e
import poc3_common
import pytest
import sync_feature_deps
import yaml
from poc3_common import landed_tasks, parse_task, task_lines, trailer_log

# tasks.md §1.1: the work items PoC-S3 uses.
WORK_ITEMS = {
    *(f"WT-{n}" for n in (30, 31, 32, 33, 34, 35, 36, 37, 38)),
    *(f"WT-{n}" for n in (40, 41, 42, 43)),
    *(f"WT-{n}" for n in (50, 51, 52, 53, 54, 55, 56, 57)),
    *(f"WT-{n}" for n in (60, 61, 62)),
    *(f"WT-{n}" for n in (70, 71, 72, 73, 79)),
}
LINK = {"WT-35", "WT-37", "WT-38", "WT-40", "WT-50", "WT-51"}
R11_PREREQS = {"WT-37", "WT-38", "WT-40", "WT-50", "WT-51", "WT-52", "WT-57"}
DS_LINK = {"WT-33", "WT-37", "WT-38", "WT-42", "WT-43", "WT-50", "WT-51", "WT-52", "WT-56", "WT-62"}
# tasks.md §3.1 rule 6: the nine FB-2 rows whose case-04 dependency set is the link alone.
FB2_ROWS = ["F-01", "F-02", "F-14", "F-16", "DG-12", "DG-13", "DG-16", "DG-82", "DG-83"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _repo(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    return repo


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for path, text in files.items():
        file = repo / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)
        _git(repo, "add", path)
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


# --- names and trailers -------------------------------------------------------------------------


def test_task_names_cover_wt30_to_wt79_and_fixes():
    assert parse_task("WT-30") == ("WT-30", None)
    assert parse_task("WT-79") == ("WT-79", None)
    assert parse_task("WT-51-fix2") == ("WT-51", 2)
    for name in (f"WT-{n}" for n in range(30, 80)):
        assert parse_task(name)[0] == name


@pytest.mark.parametrize(
    "name",
    [
        "WT-01",
        "WT-16",
        "WT-20-R2",
        "WT-20-R3",
        "WT-21",
        "WT-22",
        "WT-12-contract",
        "WT-29",
        "WT-80",
        "WT-30-fix0",
        "wt-30",
        "WT-3",
        "WT-30-R2",
    ],
)
def test_task_names_reject_poc2_and_malformed(name):
    with pytest.raises(ValueError):
        parse_task(name)


def test_trailers_read_only_the_poc3_key_in_the_trailer_block(tmp_path):
    repo = _repo(tmp_path)
    _commit(
        repo, {}, "Subject\n\nBody\n\nPoC2-Task: WT-10\nPoC3-Task: WT-30\nSigned-off-by: x <x@y>"
    )
    _commit(repo, {}, "Subject\n\nPoC2-Task: WT-10")
    _commit(repo, {}, "Subject\n\nPoC3-Task: WT-51\nis quoted in the body\n\nPoC3-Task: WT-35")
    values = [record.values for record in trailer_log(repo, "main")]
    assert values == [["WT-35"], [], ["WT-30"]]
    assert [record.malformed for record in trailer_log(repo, "main")] == [["WT-51"], [], []]
    assert landed_tasks(repo, "main") == {"WT-30", "WT-35"}


@pytest.mark.parametrize(
    "message",
    ["add\n\nPoC3-Task: WT-51\nquoted in the body\n\nPoC3-Task: WT-30", "PoC3-Task: WT-51"],
    ids=["earlier-paragraph", "subject"],
)
def test_an_exact_line_outside_the_trailers_fails_the_gate(tmp_path, capsys, message):
    # The §0.2 grep would find the line and count WT-51 as landed.
    repo, base = _two_commit_repo(tmp_path, {"scripts/poc3/x.py": "x\n"}, message)
    [record] = trailer_log(repo, f"{base}..HEAD")
    assert record.malformed == ["WT-51"]
    assert check_ownership.check("WT-30", f"{base}..HEAD", str(repo)) == 1
    assert "BAD TRAILER" in capsys.readouterr().out
    assert "WT-51" not in landed_tasks(repo, "main")
    assert "malformed PoC3-Task line ('WT-51')" in capsys.readouterr().err


@pytest.mark.parametrize(
    "message, values, malformed",
    [
        ("s\n\nPoC3-Task: WT-30\n   \n", ["WT-30"], []),
        ("s\n\nPoC3-Task: WT-35\n \nPoC3-Task: WT-30", ["WT-30"], ["WT-35"]),
        ("s\n\nPoC3-Task:", [], ["PoC3-Task:"]),
        ("s\n\nPoC3-Task: WT-30\r", [], ["WT-30"]),
    ],
    ids=["trailing-blank-line", "blank-line-separates", "empty-value", "carriage-return"],
)
def test_task_lines_of_unusual_messages(message, values, malformed):
    assert task_lines(message) == (values, malformed)


def test_a_message_that_is_not_utf8_is_read_without_failing(tmp_path, capsys):
    repo = _repo(tmp_path)
    tree = _git(repo, "mktree").strip()
    person = "t <t@example.com> 0 +0000"
    # git commit would re-encode the message; write the commit object as it is.
    commit = f"tree {tree}\nauthor {person}\ncommitter {person}\n\n".encode()
    sha = (
        subprocess.run(
            ["git", "-C", str(repo), "hash-object", "-t", "commit", "-w", "--stdin"],
            input=commit + b"s3 \xff\n\nPoC3-Task: WT-30\xff\n",
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .strip()
    )
    _git(repo, "update-ref", "refs/heads/main", sha)
    assert landed_tasks(repo, "main") == set()
    assert "is not a PoC-S3 work item" in capsys.readouterr().err


def test_a_well_formed_line_naming_no_poc3_item_is_reported(tmp_path, capsys):
    repo = _repo(tmp_path)
    _commit(repo, {}, "s3\n\nPoC3-Task: WT-35: contract")
    assert landed_tasks(repo, "main") == set()
    assert "names 'WT-35: contract', which is not a PoC-S3 work item" in capsys.readouterr().err


def test_landed_tasks_ignore_poc2_trailers(tmp_path):
    repo = _repo(tmp_path)
    _commit(repo, {}, "s2\n\nPoC2-Task: WT-10")
    _commit(repo, {}, "s2 round\n\nPoC2-Task: WT-20-R3")
    _commit(repo, {}, "s3\n\nPoC3-Task: WT-30")
    _commit(repo, {}, "s3 fix\n\nPoC3-Task: WT-51-fix1")
    _commit(repo, {}, "bad\n\nPoC3-Task: WT-12")
    assert landed_tasks(repo, "main") == {"WT-30", "WT-51-fix1"}


@pytest.mark.parametrize(
    "trailer",
    [
        "poc3-task: WT-30",
        "PoC3-Task:WT-30",
        "PoC3-Task : WT-30",
        "PoC3-Task:  WT-30",
        "PoC3-Task: WT-30 ",
        "PoC3-Task: WT-30\npoc3-task: WT-30",
    ],
    ids=["lower-case", "no-space", "space-before-colon", "two-spaces", "trailing-space", "twice"],
)
def test_a_trailer_not_written_exactly_neither_passes_nor_lands(tmp_path, capsys, trailer):
    # tasks.md §0.2 finds a landed item with --grep='^PoC3-Task: WT-xx$'; only that spelling counts.
    repo = _repo(tmp_path)
    _git(repo, "config", "commit.cleanup", "verbatim")  # keep the trailing space
    (repo / "tensorrt_llm").mkdir()
    base = _commit(repo, {"tensorrt_llm/__init__.py": ""}, "base")
    _commit(repo, {"scripts/poc3/x.py": "x\n"}, f"add\n\n{trailer}\n")
    [record] = trailer_log(repo, f"{base}..HEAD")
    assert record.malformed == ["WT-30"]
    assert check_ownership.check("WT-30", f"{base}..HEAD", str(repo)) == 1
    assert "is not written as the exact line" in capsys.readouterr().out
    assert landed_tasks(repo, "main") == ({"WT-30"} if trailer.count("WT-30") == 2 else set())
    assert "malformed PoC3-Task line" in capsys.readouterr().err


def test_an_exact_line_in_the_body_does_not_excuse_a_malformed_trailer(tmp_path, capsys):
    message = "add\n\nPoC3-Task: WT-30\nquoted in the body\n\npoc3-task: WT-30"
    repo, base = _two_commit_repo(tmp_path, {"scripts/poc3/x.py": "x\n"}, message)
    [record] = trailer_log(repo, f"{base}..HEAD")
    assert record.values == [] and record.malformed == ["WT-30", "WT-30"]
    assert check_ownership.check("WT-30", f"{base}..HEAD", str(repo)) == 1
    assert landed_tasks(repo, "main") == set()


@pytest.mark.parametrize(
    "message",
    [
        "PoC3-Task: WT-30",
        "add\n\nPoC3-Task: WT-30\n  folded onto the next line",
        "add\n\nPoC3-Task: WT-30\n\tfolded with a tab",
    ],
    ids=["subject-only", "folded", "folded-with-a-tab"],
)
def test_lines_git_would_not_read_as_the_trailer_name_nothing(tmp_path, message):
    repo = _repo(tmp_path)
    _commit(repo, {}, message)
    assert landed_tasks(repo, "main") == set()


def test_landed_tasks_do_not_depend_on_the_grep_configuration(tmp_path):
    repo = _repo(tmp_path)
    _commit(repo, {}, "s3\n\nPoC3-Task: WT-30")
    for pattern_type in ("fixed", "extended", "perl"):
        _git(repo, "config", "grep.patternType", pattern_type)
        assert landed_tasks(repo, "main") == {"WT-30"}


@pytest.mark.parametrize(
    "key, value",
    [
        ("trailer.separators", "#"),
        ("trailer.poc3-task.key", "Other"),
        ("log.showSignature", "true"),
        ("i18n.logOutputEncoding", "UTF-16"),
    ],
)
def test_trailers_do_not_depend_on_the_git_configuration(tmp_path, key, value):
    repo = _repo(tmp_path)
    _commit(repo, {}, "s3\n\nPoC3-Task: WT-30")
    _git(repo, "config", key, value)
    assert landed_tasks(repo, "main") == {"WT-30"}


def test_a_commit_with_two_task_trailers_fails_the_gate(tmp_path, capsys):
    repo, base = _two_commit_repo(
        tmp_path, {"scripts/poc3/x.py": "x\n"}, "add\n\nPoC3-Task: WT-30\nPoC3-Task: WT-35"
    )
    assert [record.values for record in trailer_log(repo, f"{base}..HEAD")] == [["WT-30", "WT-35"]]
    assert landed_tasks(repo, "main") == {"WT-30", "WT-35"}
    assert check_ownership.check("WT-30", f"{base}..HEAD", str(repo)) == 1
    assert "BAD TRAILER" in capsys.readouterr().out


# --- ownership ----------------------------------------------------------------------------------


def test_ownership_table_names_exactly_the_work_items():
    assert set(check_ownership.OWNERSHIP) == WORK_ITEMS


@pytest.mark.parametrize(
    "path, owner",
    [
        ("scripts/poc3/check_invariants.py", "WT-30"),
        ("tensorrt_llm/serve/openengine_poc/contracts.py", "WT-35"),
        ("tensorrt_llm/serve/disagg_auth.py", "WT-35"),
        ("tests/unittest/grpc/openengine_client/fake_openengine.py", "WT-36"),
        ("tests/integration/defs/openengine_chain/split.py", "WT-37"),
        ("tests/integration/defs/openengine_chain/proof.py", "WT-38"),
        ("tensorrt_llm/grpc/openengine/disagg.py", "WT-40"),
        ("tensorrt_llm/commands/serve.py", "WT-50"),
        ("tensorrt_llm/serve/openengine_poc/facade.py", "WT-51"),
        ("tests/unittest/llmapi/apps/_test_openengine_disagg_rejects.py", "WT-52"),
        ("tests/unittest/llmapi/apps/_test_openengine_disagg_kv_aware.py", "WT-53"),
        ("tests/unittest/llmapi/apps/_test_openengine_disagg_sampling.py", "WT-57"),
        ("tests/unittest/llmapi/apps/_test_openengine_ds_disagg_gsm8k.py", "WT-62"),
        ("tests/unittest/serve/openengine_poc/wt52/test_poc3_wt52_rejects.py", "WT-52"),
        ("tests/integration/defs/openengine_chain/verified/WT-60.yaml", "WT-60"),
        ("tests/integration/defs/openengine_chain/records/WT-31/b200.txt", "WT-31"),
    ],
)
def test_owner_of(path, owner):
    owning = [name for name, rule in check_ownership.owners(path) if rule.kind == "own"]
    assert owning == [owner]


def test_the_invariant_registry_belongs_to_wt35_with_a_creation_exception_for_wt30():
    found = {
        name: rule.kind
        for name, rule in check_ownership.owners(check_ownership.INVARIANT_ADDITIONS)
    }
    assert found == {"WT-30": "exception", "WT-35": "own"}


@pytest.mark.parametrize(
    "path",
    [
        *check_invariants.ORCHESTRATOR_FILES,
        "scripts/poc2/check_ownership.py",
        "tests/integration/defs/openengine_chain/smoke/WT-10.txt",
    ],
)
def test_paths_nobody_owns(path):
    assert check_ownership.owners(path) == []


def test_ds_paths_belong_to_ds_items_only():
    owning = [n for n, r in check_ownership.owners("python/dsv4_serve/openengine/server.py", "ds")]
    assert owning == ["WT-41", "WT-42", "WT-43"]
    assert check_ownership.owners("python/dsv4_serve/openengine/server.py") == []


def _two_commit_repo(tmp_path, files, message):
    repo = _repo(tmp_path)
    (repo / "tensorrt_llm").mkdir()
    base = _commit(repo, {"tensorrt_llm/__init__.py": ""}, "base")
    _commit(repo, files, message)
    return repo, base


def test_check_passes_an_in_bounds_change(tmp_path, capsys):
    repo, base = _two_commit_repo(tmp_path, {"scripts/poc3/x.py": "x\n"}, "add\n\nPoC3-Task: WT-30")
    assert check_ownership.check("WT-30", f"{base}..HEAD", str(repo)) == 0
    assert "PASS: WT-30" in capsys.readouterr().out


def test_check_fails_a_deliberately_out_of_bounds_change(tmp_path, capsys):
    repo, base = _two_commit_repo(
        tmp_path,
        {"scripts/poc3/x.py": "x\n", "tensorrt_llm/serve/router.py": "changed\n"},
        "add\n\nPoC3-Task: WT-30",
    )
    assert check_ownership.check("WT-30", f"{base}..HEAD", str(repo)) == 1
    out = capsys.readouterr().out
    assert "OUT OF BOUNDS  tensorrt_llm/serve/router.py  (owned by nobody)" in out


@pytest.mark.parametrize("message", ["add\n\nPoC2-Task: WT-30", "add", "add\n\nPoC3-Task: WT-35"])
def test_check_fails_a_commit_without_the_task_trailer(tmp_path, capsys, message):
    repo, base = _two_commit_repo(tmp_path, {"scripts/poc3/x.py": "x\n"}, message)
    assert check_ownership.check("WT-30", f"{base}..HEAD", str(repo)) == 1
    assert "BAD TRAILER" in capsys.readouterr().out


def test_a_fix_owns_what_its_item_owns_and_its_own_records(tmp_path):
    files = {
        "tensorrt_llm/serve/openengine_poc/facade.py": "x\n",
        "tests/integration/defs/openengine_chain/verified/WT-51-fix1.yaml": "entries: []\n",
    }
    repo, base = _two_commit_repo(tmp_path, files, "fix\n\nPoC3-Task: WT-51-fix1")
    assert check_ownership.check("WT-51-fix1", f"{base}..HEAD", str(repo)) == 0


@pytest.mark.parametrize(
    "task, path, allowed",
    [
        ("WT-51-fix1", f"{check_ownership.CHAIN}/verified/WT-51.yaml", True),
        ("WT-51-fix1", f"{check_ownership.CHAIN}/verified/WT-51-fix2.yaml", False),
        ("WT-51", f"{check_ownership.CHAIN}/verified/WT-51-fix1.yaml", False),
        ("WT-51-fix1", f"{check_ownership.POC_TESTS}/wt51/test_x.py", True),
    ],
)
def test_per_item_files_belong_to_the_exact_name(task, path, allowed):
    assert (check_ownership.match_rule(task, path) is not None) == allowed


@pytest.mark.parametrize(
    "task, path, allowed",
    [
        ("WT-30-fix1", "scripts/poc3/check_ownership.py", True),
        ("WT-54", "scripts/poc3/check_ownership.py", False),
        ("WT-54", "scripts/poc3/poc3_common.py", False),
        ("WT-54", "scripts/poc3/wt54_report.py", True),
        ("WT-35", check_ownership.INVARIANT_ADDITIONS, True),
    ],
)
def test_only_wt30_changes_the_gate_files(task, path, allowed):
    assert (check_ownership.match_rule(task, path) is not None) == allowed


def test_no_literal_path_has_two_owners():
    literal = {
        (rule.pattern, rule.repo)
        for rules in check_ownership.OWNERSHIP.values()
        for rule in rules
        if rule.kind == "own" and not set("*?!") & set(rule.pattern)
    }
    for path, repo in literal:
        owning = [n for n, r in check_ownership.owners(path, repo) if r.kind == "own"]
        assert len(owning) == 1, (path, owning)


def test_wt57_leaves_the_domains_of_other_items_alone():
    taken = [
        rule.pattern
        for name, rules in check_ownership.OWNERSHIP.items()
        if name != "WT-57"
        for rule in rules
        if rule.kind == "own" and "/_test_openengine_disagg_" in rule.pattern
    ]
    assert len(taken) == 18  # the 17 domains of other items, and WT-40's NT-040 file
    for path in taken:
        assert check_ownership.match_rule("WT-57", path) is None, path


def test_moving_a_file_counts_its_old_path(tmp_path, capsys):
    repo = _repo(tmp_path)
    base = _commit(
        repo, {"tensorrt_llm/__init__.py": "", "tensorrt_llm/serve/router.py": "r\n"}, "base"
    )
    (repo / "scripts/poc3").mkdir(parents=True)
    _git(repo, "mv", "tensorrt_llm/serve/router.py", "scripts/poc3/router.py")
    _commit(repo, {}, "move\n\nPoC3-Task: WT-30")
    assert check_ownership.check("WT-30", f"{base}..HEAD", str(repo)) == 1
    assert "OUT OF BOUNDS  tensorrt_llm/serve/router.py" in capsys.readouterr().out


def test_check_fails_an_empty_range(tmp_path, capsys):
    repo, _ = _two_commit_repo(tmp_path, {"scripts/poc3/x.py": "x\n"}, "add\n\nPoC3-Task: WT-30")
    assert check_ownership.check("WT-30", "HEAD..HEAD", str(repo)) == 1
    assert "NO COMMITS" in capsys.readouterr().out


def test_the_gate_files_hold_even_if_the_table_grants_them(monkeypatch):
    rules = check_ownership.OWNERSHIP["WT-54"] + check_ownership._own("scripts/poc3/**")
    monkeypatch.setitem(check_ownership.OWNERSHIP, "WT-54", rules)
    assert check_ownership.match_rule("WT-54", "scripts/poc3/check_ownership.py") is None


def test_unusual_path_names_are_matched_as_they_are(tmp_path, capsys):
    path = "tests/integration/defs/openengine_chain/records/WT-31/b200 é.txt"
    repo, base = _two_commit_repo(tmp_path, {path: "x\n"}, "add\n\nPoC3-Task: WT-31")
    assert check_ownership.check("WT-31", f"{base}..HEAD", str(repo)) == 0


def test_a_relative_diff_config_does_not_hide_changes(tmp_path, capsys):
    repo, base = _two_commit_repo(
        tmp_path, {"tensorrt_llm/serve/router.py": "changed\n"}, "add\n\nPoC3-Task: WT-30"
    )
    _git(repo, "config", "diff.relative", "true")
    assert check_ownership.check("WT-30", f"{base}..HEAD", str(repo / "tensorrt_llm")) == 1


def test_a_relative_diff_config_does_not_hide_changes_outside_the_directory(tmp_path, capsys):
    repo, base = _two_commit_repo(tmp_path, {"tests/x.py": "x\n"}, "add\n\nPoC3-Task: WT-30")
    _git(repo, "config", "diff.relative", "true")
    assert check_ownership.check("WT-30", f"{base}..HEAD", str(repo / "tensorrt_llm")) == 1
    assert "OUT OF BOUNDS  tests/x.py" in capsys.readouterr().out


@pytest.mark.parametrize(
    "path, task",
    [
        ("tests/integration/defs/openengine_chain/tests/test_remote_server.py", "WT-37"),
        ("tests/integration/defs/openengine_chain/tests/test_proof.py", "WT-38"),
        ("tests/unittest/grpc/openengine_client/test_fake_openengine.py", "WT-36"),
    ],
)
def test_existing_test_files_are_listed_for_review(path, task):
    rule = check_ownership.match_rule(task, path)
    assert rule is not None and rule.kind == "shared"
    assert [n for n, r in check_ownership.owners(path) if r.kind == "own"] == []


@pytest.mark.parametrize(
    "path, owner",
    [
        ("tests/unittest/llmapi/apps/_test_openengine_disagg_direct.py", "WT-40"),
        ("tests/unittest/llmapi/apps/_test_openengine_ds_disagg_direct.py", "WT-42"),
        ("tests/unittest/grpc/openengine/_test_direct_handoff.py", "WT-32"),
    ],
)
def test_direct_case_files_follow_section_3_3(path, owner):
    assert [n for n, r in check_ownership.owners(path) if r.kind == "own"] == [owner]
    if owner == "WT-32":
        assert check_ownership.match_rule("WT-40", path).kind == "shared"


@pytest.mark.parametrize("revision_range", ["..HEAD", "HEAD..", "HEAD...HEAD", "HEAD"])
def test_check_needs_both_ends_of_a_two_dot_range(tmp_path, revision_range):
    repo, _ = _two_commit_repo(tmp_path, {"scripts/poc3/x.py": "x\n"}, "add\n\nPoC3-Task: WT-30")
    with pytest.raises(ValueError):
        check_ownership.check("WT-30", revision_range, str(repo))


def test_shared_and_exception_paths_are_listed_for_review(tmp_path, capsys):
    files = {"scripts/poc3/invariant_additions.yaml": "additions: []\n"}
    repo, base = _two_commit_repo(tmp_path, files, "add\n\nPoC3-Task: WT-30")
    assert check_ownership.check("WT-30", f"{base}..HEAD", str(repo)) == 0
    assert (
        "CHECK BY HAND  EXCEPTION  scripts/poc3/invariant_additions.yaml" in capsys.readouterr().out
    )


def test_an_unused_work_item_number_has_no_entry(tmp_path, capsys):
    repo, base = _two_commit_repo(tmp_path, {"a.txt": "a\n"}, "add\n\nPoC3-Task: WT-39")
    assert check_ownership.check("WT-39", f"{base}..HEAD", str(repo)) == 1
    assert "no entry in the ownership table" in capsys.readouterr().out


def test_the_ds_repository_is_detected(tmp_path):
    repo = _repo(tmp_path)
    _commit(repo, {"python/dsv4_serve/__init__.py": ""}, "base")
    assert check_ownership.detect_repo(str(repo)) == check_ownership.DS


# The three assertions that pin the contract and conventions versions WT-35 raises.
VERSION_PINS = {
    "tests/unittest/serve/openengine_poc/wt01/test_poc2_wt01_contracts.py": (
        "    assert contracts.POC2_CONTRACT_VERSION == 9\n",
        "    assert contracts.POC2_CONTRACT_VERSION == 10\n",
    ),
    "tests/unittest/serve/openengine_poc/wt01/test_poc2_wt01_conventions.py": (
        "    assert conv.CONVENTIONS_VERSION == 3\n",
        "    assert conv.CONVENTIONS_VERSION == 4\n",
    ),
    "tests/unittest/grpc/openengine/test_control.py": (
        "    assert isinstance(version, float) and version == 3\n",
        "    assert isinstance(version, float) and version == 4\n",
    ),
}


def _pin_repo(tmp_path, path, change):
    old = VERSION_PINS[path][0]
    text = f"def test_version():\n{old}\n\ndef test_other():\n    assert True\n"
    repo = _repo(tmp_path)
    base = _commit(repo, {"tensorrt_llm/__init__.py": "", path: text}, "base")
    _commit(repo, {path: change(text)}, "bump\n\nPoC3-Task: WT-35")
    return repo, base


@pytest.mark.parametrize("path", VERSION_PINS)
def test_wt35_may_raise_the_pinned_version(tmp_path, capsys, path):
    old, new = VERSION_PINS[path]
    repo, base = _pin_repo(tmp_path, path, lambda text: text.replace(old, new))
    assert check_ownership.check("WT-35", f"{base}..HEAD", str(repo)) == 0
    assert f"CHECK BY HAND  EXCEPTION  {path}  -- only the expected" in capsys.readouterr().out


@pytest.mark.parametrize("path", VERSION_PINS)
@pytest.mark.parametrize(
    "change",
    [
        lambda text, old, new: text.replace(old, new).replace("assert True", "assert 1"),
        lambda text, old, new: text.replace(old, ""),
        lambda text, old, new: text.replace(old, new + new),
        lambda text, old, new: text.replace(old, new.replace("\n", "\r\n")),
        lambda text, old, new: text.replace(old, new.replace("    ", "  ")),
        lambda text, old, new: text.replace(old, "") + new,
        lambda text, old, new: text.replace(old, "").replace("    assert True\n", new),
        lambda text, old, new: text.replace(old, new.replace(" == ", " != ")),
    ],
    ids=[
        "another-line",
        "assertion-removed",
        "line-added",
        "crlf",
        "reindented",
        "moved-to-the-end",
        "moved-into-another-test",
        "not-equal",
    ],
)
def test_wt35_may_not_change_anything_else_in_those_files(tmp_path, capsys, path, change):
    old, new = VERSION_PINS[path]
    repo, base = _pin_repo(tmp_path, path, lambda text: change(text, old, new))
    assert check_ownership.check("WT-35", f"{base}..HEAD", str(repo)) == 1
    assert f"OUT OF BOUNDS  {path}  (the exception allows only the expected" in (
        capsys.readouterr().out
    )


def test_a_version_pin_exception_rejects_dropping_the_final_newline(tmp_path, capsys):
    path = next(iter(VERSION_PINS))
    old, new = VERSION_PINS[path]
    repo = _repo(tmp_path)
    base = _commit(
        repo, {"tensorrt_llm/__init__.py": "", path: f"def test_version():\n{old}"}, "base"
    )
    _commit(repo, {path: f"def test_version():\n{new.rstrip()}"}, "bump\n\nPoC3-Task: WT-35")
    assert check_ownership.check("WT-35", f"{base}..HEAD", str(repo)) == 1
    assert "No newline at end of file" in capsys.readouterr().out


def _pin_rule():
    path = next(iter(VERSION_PINS))
    return path, check_ownership.match_rule("WT-35", path)


def test_a_version_pin_exception_ignores_the_hunk_context_configuration(tmp_path):
    path, rule = _pin_rule()
    old, new = VERSION_PINS[path]
    text = f"{old}x = 1\n{old}"
    repo = _repo(tmp_path)
    base = _commit(repo, {path: text}, "base")
    _commit(repo, {path: text.replace(old, new)}, "bump")
    _git(repo, "config", "diff.interHunkContext", "5")
    _git(repo, "config", "diff.context", "5")
    assert check_ownership.disallowed_lines(rule, str(repo), base, "HEAD", path) == []


@pytest.mark.parametrize(
    "config",
    [
        {"diff.relative": "true"},
        {"diff.external": "true"},
        {"diff.hide.textconv": "true"},
    ],
    ids=["relative", "external", "textconv"],
)
def test_a_version_pin_exception_sees_every_change_whatever_the_configuration(
    tmp_path, capsys, config
):
    path = next(iter(VERSION_PINS))
    old, new = VERSION_PINS[path]
    repo, base = _pin_repo(
        tmp_path, path, lambda text: text.replace(old, new).replace("assert True", "assert 1")
    )
    (repo / ".git/info/attributes").write_text("*.py diff=hide\n")
    for key, value in config.items():
        _git(repo, "config", key, value)
    # Run from a directory that does not contain the file.
    assert check_ownership.check("WT-35", f"{base}..HEAD", str(repo / "tensorrt_llm")) == 1
    assert f"OUT OF BOUNDS  {path}" in capsys.readouterr().out


def test_a_version_pin_exception_ignores_the_color_and_context_settings(tmp_path, monkeypatch):
    path, rule = _pin_rule()
    old, new = VERSION_PINS[path]
    repo, base = _pin_repo(tmp_path, path, lambda text: text.replace(old, new))
    _git(repo, "config", "color.ui", "always")
    _git(repo, "config", "color.diff", "always")
    monkeypatch.setenv("GIT_DIFF_OPTS", "--unified=3")
    assert check_ownership.disallowed_lines(rule, str(repo), base, "HEAD", path) == []


PATHSPEC_VARIABLES = (
    "GIT_LITERAL_PATHSPECS",
    "GIT_GLOB_PATHSPECS",
    "GIT_NOGLOB_PATHSPECS",
    "GIT_ICASE_PATHSPECS",
)


@pytest.mark.parametrize("variable", PATHSPEC_VARIABLES)
def test_a_version_pin_exception_sees_every_change_whatever_the_pathspec_variables(
    tmp_path, capsys, monkeypatch, variable
):
    path = next(iter(VERSION_PINS))
    old, new = VERSION_PINS[path]
    repo, base = _pin_repo(
        tmp_path, path, lambda text: text.replace(old, new).replace("assert True", "assert 1")
    )
    monkeypatch.setenv(variable, "1")
    assert check_ownership.check("WT-35", f"{base}..HEAD", str(repo)) == 1
    assert f"OUT OF BOUNDS  {path}" in capsys.readouterr().out


def test_git_runs_without_the_variables_that_change_its_reading(monkeypatch):
    for variable in ("GIT_DIFF_OPTS", *PATHSPEC_VARIABLES):
        monkeypatch.setenv(variable, "1")
    monkeypatch.setenv("GIT_DIR", "/kept")
    env = poc3_common.git_env()
    assert not {"GIT_DIFF_OPTS", *PATHSPEC_VARIABLES} & set(env)
    assert env["GIT_DIR"] == "/kept"


def test_a_submodule_pointer_is_seen_whatever_the_configuration(tmp_path, capsys):
    repo = _repo(tmp_path)
    base = _commit(repo, {"tensorrt_llm/__init__.py": ""}, "base")
    sha = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{sha},tensorrt_llm/serve/vendored")
    _git(repo, "commit", "-q", "-m", "vendor\n\nPoC3-Task: WT-30")
    _git(repo, "config", "diff.ignoreSubmodules", "all")
    assert check_ownership.check("WT-30", f"{base}..HEAD", str(repo)) == 1
    assert "OUT OF BOUNDS  tensorrt_llm/serve/vendored" in capsys.readouterr().out


def test_a_version_pin_exception_reads_the_change_against_the_merge_base(tmp_path):
    path, rule = _pin_rule()
    old, new = VERSION_PINS[path]
    text = f"def test_version():\n{old}\n\ndef test_other():\n    assert True\n"
    repo = _repo(tmp_path)
    fork = _commit(repo, {path: text}, "base")
    # The base branch moves on and changes another line; the head only raises the version.
    _commit(repo, {path: text.replace("assert True", "assert 1")}, "elsewhere")
    _git(repo, "branch", "moved")
    _git(repo, "checkout", "-q", "-b", "head", fork)
    _commit(repo, {path: text.replace(old, new)}, "bump")
    assert check_ownership.disallowed_lines(rule, str(repo), "moved", "head", path) == []


def test_a_version_pin_exception_rejects_a_mode_change(tmp_path, capsys):
    path = next(iter(VERSION_PINS))
    old, new = VERSION_PINS[path]
    repo, base = _pin_repo(tmp_path, path, lambda text: text.replace(old, new))
    (repo / path).chmod(0o755)
    _git(repo, "add", path)
    _commit(repo, {}, "mode\n\nPoC3-Task: WT-35")
    assert check_ownership.check("WT-35", f"{base}..HEAD", str(repo)) == 1
    assert f"OUT OF BOUNDS  {path}" in capsys.readouterr().out


@pytest.mark.parametrize("path", VERSION_PINS)
def test_the_version_pin_exceptions_are_wt35s(path):
    assert check_ownership.match_rule("WT-36", path) is None
    assert check_ownership.match_rule("WT-35-fix1", path).lines


@pytest.mark.parametrize("path", VERSION_PINS)
def test_the_version_pin_patterns_match_the_files_at_the_frozen_base(path):
    repo = Path(__file__).resolve().parents[2]
    found = subprocess.run(
        ["git", "-C", str(repo), "show", f"{check_invariants.FROZEN_BASE}:{path}"],
        capture_output=True,
        text=True,
    )
    if found.returncode != 0:
        pytest.skip("the frozen base is not in this checkout")
    rule = check_ownership.match_rule("WT-35", path)
    lines = found.stdout.split("\n")
    assert VERSION_PINS[path][0].rstrip("\n") in lines
    matching = [line for line in lines if any(re.fullmatch(p, line) for p in rule.lines)]
    assert matching == [VERSION_PINS[path][0].rstrip("\n")]


def test_a_missing_git_fails_with_a_message(tmp_path, capsys, monkeypatch):
    repo, base = _two_commit_repo(tmp_path, {"scripts/poc3/x.py": "x\n"}, "add\n\nPoC3-Task: WT-30")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert check_ownership.main(["--task", "WT-30", f"{base}..HEAD", "--repo-dir", str(repo)]) == 1
    assert capsys.readouterr().out.startswith("FAIL: FileNotFoundError")


@pytest.mark.parametrize(
    "args",
    [["--task", "WT-5", "HEAD~1..HEAD"], ["--task", "WT-30", "HEAD"], ["--task", "WT-30", "x..y"]],
    ids=["bad-task", "bad-range", "unknown-revisions"],
)
def test_bad_arguments_fail_with_a_message(tmp_path, capsys, args):
    repo, _ = _two_commit_repo(tmp_path, {"scripts/poc3/x.py": "x\n"}, "add\n\nPoC3-Task: WT-30")
    assert check_ownership.main([*args, "--repo-dir", str(repo)]) == 1
    assert capsys.readouterr().out.startswith("FAIL: ")


# --- end-to-end obligations ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def rows():
    return pending_e2e.load_rows()[1]


def _owed(task, landed, rows, verified=(), unknown=()):
    result = pending_e2e.obligations(task, landed, rows, verified, unknown)
    return (
        [row.id for row in result.owed],
        [row.id for row in result.left_to_fallback],
        [row.id for row in result.unverified],
        [row.id for row in result.unknown],
    )


def test_a_bad_task_name_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        pending_e2e.main(["--task", "WT-5", "--repo", str(tmp_path)])
    assert raised.value.code == 2
    assert "not a PoC-S3 work item" in capsys.readouterr().err


def test_listing_what_landed_does_not_read_the_task(tmp_path, capsys):
    repo = _repo(tmp_path)
    _commit(repo, {}, "s3\n\nPoC3-Task: WT-30")
    assert (
        pending_e2e.main(["--landed", "--task", "WT-5", "--repo", str(repo), "--ref", "main"]) == 0
    )
    assert capsys.readouterr().out.split() == ["WT-30"]


def test_an_unknown_ref_is_a_usage_error(tmp_path, capsys):
    repo = _repo(tmp_path)
    _commit(repo, {}, "s3\n\nPoC3-Task: WT-30")
    with pytest.raises(SystemExit) as raised:
        pending_e2e.main(["--task", "WT-51", "--repo", str(repo), "--ref", "nosuch"])
    assert raised.value.code == 2
    assert "nosuch" in capsys.readouterr().err


def test_a_verification_record_that_is_not_yaml_is_reported_and_ignored(tmp_path, capsys):
    repo = _repo(tmp_path)
    _commit(repo, {f"{pending_e2e.VERIFIED_DIR}/WT-51.yaml": "entries: [\n"}, "bad record")
    assert pending_e2e.read_verified(repo, "main") == set()
    assert "WT-51.yaml is not valid YAML" in capsys.readouterr().err


@pytest.mark.parametrize(
    "data",
    [b"entries:\n  - row: F-01\n    date: 2026-02-30\n", b"entries: [\xff]\n"],
    ids=["bad-date", "not-utf8"],
)
def test_a_verification_record_yaml_cannot_convert_is_reported_and_ignored(tmp_path, capsys, data):
    repo = _repo(tmp_path)
    record = repo / pending_e2e.VERIFIED_DIR / "WT-51.yaml"
    record.parent.mkdir(parents=True)
    record.write_bytes(data)
    _git(repo, "add", ".")
    _commit(repo, {}, "bad record")
    assert pending_e2e.read_verified(repo, "main") == set()
    assert "WT-51.yaml is not valid YAML" in capsys.readouterr().err


def test_verification_entries_that_are_not_a_list_are_reported_and_ignored(tmp_path, capsys):
    repo = _repo(tmp_path)
    _commit(repo, {f"{pending_e2e.VERIFIED_DIR}/WT-51.yaml": "entries: 5\n"}, "bad record")
    assert pending_e2e.read_verified(repo, "main") == set()
    assert "entries are not a list" in capsys.readouterr().err


def test_a_record_name_that_is_not_utf8_is_skipped(tmp_path):
    repo = _repo(tmp_path)
    directory = repo / pending_e2e.VERIFIED_DIR
    directory.mkdir(parents=True)
    (directory / "WT-51.yaml").write_text("entries:\n  - row: F-01\n")
    (directory / os.fsdecode(b"WT-5\xff.yaml")).write_text("x: 1\n")
    _git(repo, "add", ".")
    _commit(repo, {}, "records")
    _git(repo, "config", "core.quotePath", "false")
    assert pending_e2e.read_verified(repo, "main") == {"F-01"}


def test_a_missing_git_is_a_usage_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(SystemExit) as raised:
        pending_e2e.main(["--landed", "--repo", str(tmp_path)])
    assert raised.value.code == 2
    assert "FileNotFoundError" in capsys.readouterr().err


def test_groups_match_tasks_md_section_3_2():
    groups = pending_e2e.load_rows()[0]
    assert set(groups["link"]) == LINK
    assert set(groups["r11-prereqs"]) == R11_PREREQS
    assert set(groups["ds-link"]) == DS_LINK
    assert set(groups["ds-repo"]) == {"WT-41", "WT-42", "WT-43"}


def test_rows_name_only_poc3_items(rows):
    for row in rows:
        for name in [*row.deps, row.assignee, row.fallback]:
            if name is not None:
                assert parse_task(name)[0] in WORK_ITEMS, (row.id, name)


def test_wt30_and_wt35_alone_owe_nothing(rows):
    assert _owed("WT-30", set(), rows)[:3] == ([], [], [])
    assert _owed("WT-35", {"WT-30"}, rows)[:3] == ([], [], [])


def test_the_last_link_item_owes_the_fb2_rows(rows):
    # Order A: WT-51 completes the link.
    owed, _, _, _ = _owed("WT-51", LINK - {"WT-51"}, rows)
    assert set(FB2_ROWS) <= set(owed)
    # F-03 also needs WT-57 (§3.4: 链路、WT-57), F-01's T2 waits for WT-57 (+rest).
    assert "F-03" not in owed and "F-01+rest" not in owed
    # Order B: WT-35 completes the link instead; WT-51 then owes nothing for those rows.
    owed_b = _owed("WT-35", LINK - {"WT-35"}, rows)[0]
    assert set(FB2_ROWS) <= set(owed_b)
    assert not set(FB2_ROWS) & set(_owed("WT-51", LINK - {"WT-51"}, rows, verified=FB2_ROWS)[0])


def test_rows_with_an_item_on_top_of_the_link_follow_landing_order(rows):
    # §3.4 F-03: 链路、WT-57. If WT-57 lands after the link, WT-57 owes it ...
    owed = _owed("WT-57", LINK, rows)[0]
    assert {"F-03", "F-01+rest", "F-02+rest"} <= set(owed)
    assert "F-01" not in owed  # the link alone completed F-01 earlier
    # ... and if the link's last item lands after WT-57, that item owes it.
    owed = _owed("WT-50", (LINK - {"WT-50"}) | {"WT-57"}, rows)[0]
    assert {"F-01", "F-03", "F-01+rest"} <= set(owed)
    # F-30 (§3.4: 链路、WT-52; 其余用例另等 WT-57): its +rest row waits for both.
    assert "F-30" not in owed and "F-30+rest" not in owed
    owed = _owed("WT-52", LINK | {"WT-57"}, rows)[0]
    assert {"F-30", "F-30+rest"} <= set(owed)


def test_r11_and_r12_rows_are_owed_only_by_their_round(rows):
    landed = LINK | R11_PREREQS
    owed, _, unverified, _ = _owed("WT-57", landed - {"WT-57"}, rows)
    assert "F-33" not in owed  # §3.4 F-33: R11 前提
    assert "F-33" in _owed("WT-60", landed, rows)[0]
    assert "F-33" in _owed("WT-61", landed | {"WT-60"}, rows)[2]  # landed, unverified, not owed
    # §3.6 EG-39: 链路、WT-38（CT-15）；R12 验收.
    assert "EG-39" in _owed("WT-61", LINK, rows)[0]
    assert "EG-39" not in _owed("WT-38", LINK - {"WT-38"}, rows)[0]


def test_case03_rows_go_to_r13_or_its_fallback(rows):
    ds_landed = (DS_LINK | {"WT-35", "WT-36", "WT-41"}) - {"WT-62"}
    owed = _owed("WT-62", ds_landed, rows)[0]
    assert "F-01@03" in owed and "DS-S01" in owed
    # F-06@03 (§3.4: ds 链路、WT-53): WT-62 cannot verify it before WT-53 lands ...
    assert "F-06@03" not in owed
    # ... so WT-53 landing later completes it and leaves it to R14 ...
    owed, left, _, _ = _owed("WT-53", ds_landed | {"WT-62"}, rows)
    assert "F-06@03" in left and "F-06@03" not in owed
    # ... which owes it.
    assert "F-06@03" in _owed("WT-79", ds_landed | {"WT-62", "WT-53"}, rows)[0]
    # DS-S19a (§3.7: R13 时 WT-71 未合入的由 WT-73 补) falls back to WT-73, not R14.
    assert "DS-S19a" in _owed("WT-71", ds_landed | {"WT-62"}, rows)[1]
    assert "DS-S19a" in _owed("WT-73", ds_landed | {"WT-62", "WT-71"}, rows)[0]
    assert "DS-S19a" not in _owed("WT-79", ds_landed | {"WT-62", "WT-71"}, rows)[0]


def test_a_fallback_landing_before_r13_owes_and_reports_no_case03_row(rows):
    ds_landed = (DS_LINK | {"WT-35", "WT-36", "WT-41"}) - {"WT-62"}
    result = pending_e2e.obligations("WT-79", ds_landed, rows)
    reported = (
        result.owed
        + result.too_early
        + result.left_to_fallback
        + result.unverified
        + result.orphaned
    )
    case03 = [row.id for row in reported if row.chain == "case-03"]
    assert case03 == []


def test_rows_missing_a_checked_item_are_incomplete_not_unknown(rows):
    landed = (DS_LINK | {"WT-35", "WT-36", "WT-41"}) - {"WT-62", "WT-42", "WT-52"}
    owed, _, _, unknown = _owed("WT-62", landed, rows, unknown={"WT-41", "WT-42", "WT-43"})
    assert "F-01@03" not in unknown and "F-01@03" not in owed
    # With WT-52 landed, the row waits only on the unchecked WT-42.
    owed, _, _, unknown = _owed("WT-62", landed | {"WT-52"}, rows, unknown={"WT-42"})
    assert "F-01@03" in unknown


def test_fixed_performance_verifiers(rows):
    # §3.2: DG-27's GPU static-URL start-up goes with PU-02 (WT-70); DS-S19b with WT-73.
    assert "DG-27+PU-02" in _owed("WT-70", LINK, rows)[0]
    assert "DG-27+PU-02" not in _owed("WT-51", LINK - {"WT-51"}, rows)[0]
    assert "DG-27" in _owed("WT-51", LINK - {"WT-51"}, rows)[0]
    landed = DS_LINK | {"WT-71"}
    assert "DS-S19b" in _owed("WT-73", landed, rows)[0]
    assert "DS-S19b" not in _owed("WT-62", landed - {"WT-62"}, rows)[0]


def test_fixes_and_deferred_rows_owe_nothing(rows):
    assert _owed("WT-51-fix1", LINK, rows)[0] == []
    everything = WORK_ITEMS
    for task in sorted(everything):
        assert "F-21" not in _owed(task, everything - {task}, rows)[0]


def test_ds_items_unknown_without_the_ds_repo(rows):
    landed = (DS_LINK | {"WT-35", "WT-36", "WT-41"}) - {"WT-62", "WT-42", "WT-43"}
    owed, _, _, unknown = _owed("WT-62", landed, rows, unknown={"WT-41", "WT-42", "WT-43"})
    assert "F-01@03" in unknown and "F-01@03" not in owed


def _extra(task, landed, rows):
    result = pending_e2e.obligations(task, landed, rows)
    return (
        [row.id for row in result.too_early],
        [row.id for row in result.orphaned],
        [row.id for row in result.noted],
    )


def test_a_fix_landing_does_not_land_its_item(rows):
    # A WT-37-fix for a PoC-S2 path can land before WT-37 (§0.6); the link is still incomplete.
    landed = (LINK - {"WT-51", "WT-37"}) | {"WT-37-fix1"}
    assert "F-01" not in _owed("WT-51", landed, rows)[0]
    assert "F-01" in _owed("WT-51", landed | {"WT-37"}, rows)[0]


def test_rows_left_without_a_verifier_are_reported(rows):
    # WT-60 running before the R11 prerequisites landed: F-33 is listed, not silently dropped.
    assert "F-33" in _extra("WT-60", LINK, rows)[0]
    # F-33 completed by WT-57 after WT-60 landed: nobody owes it any more.
    landed = (LINK | R11_PREREQS | {"WT-60"}) - {"WT-57"}
    assert "F-33" in _extra("WT-57", landed, rows)[1]
    assert "F-33" not in _owed("WT-57", landed, rows)[0]


def test_case03_rows_left_without_a_verifier_are_reported(rows):
    ds_landed = (DS_LINK | {"WT-35", "WT-36", "WT-41"}) - {"WT-62"}
    # R14 landing while F-06@03 still waits for WT-53.
    assert "F-06@03" in _extra("WT-79", ds_landed | {"WT-62"}, rows)[0]
    # WT-53 completing F-06@03 after R14 landed.
    assert "F-06@03" in _extra("WT-53", ds_landed | {"WT-62", "WT-79"}, rows)[1]


def test_notes_on_deferred_rows_reach_the_item_they_name(rows):
    # §3.6 EG-25 (不设 E2E): PB-3 走 GB 路线时由 WT-72 顺带取证.
    assert "EG-25" in _extra("WT-72", set(), rows)[2]


def test_notes_naming_an_item_reach_it(rows):
    # §3.6 EG-28: XD-009 在 CO-12 合入后由 WT-40 先补跑.
    assert "EG-28" in _extra("WT-40", set(), rows)[2]
    assert "EG-28" not in _extra("WT-41", set(), rows)[2]


def test_only_poc3_records_are_read_from_any_directory(tmp_path, capsys):
    repo = _repo(tmp_path)
    records = {
        f"{pending_e2e.VERIFIED_DIR}/WT-10.yaml": yaml.safe_dump({"entries": [{"row": "F-06"}]}),
        f"{pending_e2e.VERIFIED_DIR}/WT-20-R2.yaml": yaml.safe_dump({"entries": [{"row": "F-07"}]}),
        f"{pending_e2e.VERIFIED_DIR}/WT-51.yaml": yaml.safe_dump(
            {"entries": [{"row": "F-01"}, {"row": "F-0x"}]}
        ),
        f"{pending_e2e.VERIFIED_DIR}/WT-51-fix1.yaml": yaml.safe_dump(
            {"entries": [{"row": "F-02"}]}
        ),
    }
    _commit(repo, records, "records")
    known = {"F-01", "F-02", "F-06", "F-07"}
    sub = repo / "tests"
    assert pending_e2e.read_verified(sub, "main", known) == {"F-01", "F-02"}
    assert "unknown row 'F-0x'" in capsys.readouterr().err


def test_verified_records_need_a_row_id(capsys):
    records = [{"entries": [{"row": "F-01"}, {"features": ["F-02"]}]}, None, ["not a mapping"]]
    assert pending_e2e.verified_rows(records) == {"F-01"}
    err = capsys.readouterr().err
    assert "without a row id" in err and "not a mapping" in err


def test_pending_on_a_constructed_history(tmp_path, capsys):
    """Two landing orders on real git histories, read back through the command line."""
    for order, last in (
        (["WT-30", "WT-35", "WT-37", "WT-38", "WT-40", "WT-50"], "WT-51"),
        (["WT-30", "WT-51", "WT-37", "WT-38", "WT-40", "WT-50"], "WT-35"),
    ):
        repo = _repo(tmp_path, f"repo-{last}")
        _commit(repo, {"README": "x\n"}, "s2 history\n\nPoC2-Task: WT-10")
        for item in order:
            _commit(repo, {}, f"{item}\n\nPoC3-Task: {item}")
        verified = {"entries": [{"row": "DG-12", "commit": "x", "junit": "y"}]}
        _commit(
            repo,
            {f"{pending_e2e.VERIFIED_DIR}/WT-40.yaml": yaml.safe_dump(verified)},
            "record\n\nPoC3-Task: WT-40",
        )
        assert (
            pending_e2e.main(["--repo", str(repo), "--ref", "main", "--task", last, "--json"]) == 0
        )
        out = yaml.safe_load(capsys.readouterr().out)
        assert set(FB2_ROWS) - {"DG-12"} <= set(out["owed"]), last
        assert "DG-12" not in out["owed"]
        assert set(out["landed"]) == set(order)


# --- feature_deps.yaml against the plan ---------------------------------------------------------


def _plan_tasks_md() -> Path | None:
    path = os.environ.get("POC3_TASKS_MD")
    return Path(path) if path and Path(path).is_file() else None


def test_feature_deps_matches_the_plan_tables():
    tasks_md = _plan_tasks_md()
    if tasks_md is None:
        pytest.skip("POC3_TASKS_MD is not set")
    assert sync_feature_deps.main(["--tasks-md", str(tasks_md), "--check"]) == 0


def test_feature_deps_has_every_main_row_once(rows):
    features = {feature for row in rows for feature in row.features}
    tasks_md = _plan_tasks_md()
    if tasks_md is not None:
        assert len(features) == sync_feature_deps.main_row_count(tasks_md.read_text())
    assert len({row.id for row in rows}) == len(rows)


F90_ROW = (
    "| F-90 x（04：✅；03：✅） | "
    "04：XD-001、T2、NT-050（Hopper 分片：XD-036；PoC 机型不可跑：XD-202）；03：NT-101 | "
    "04：现成（FB-2，随 WT-51）；WT-57（T2、NT-050）；03：WT-62 | "
    "04：链路；其余用例另等 WT-57；03：ds 链路、WT-53 |"
)
EG91_ROW = (
    "| EG-91 w（✅） | NT-209、XD-009（Hopper 分片：XD-041；第二批：XD-042；XD-101 是死用例） | "
    "WT-57 | 链路、WT-38（CT-15）；R12 验收；XD-009 由 WT-40 先补跑 |"
)
GROUPS_TABLE = """\
### 3.2 依赖集合的缩写
| 缩写 | 成员 | 用在哪里 |
|---|---|---|
| 链路 | WT-35、WT-37 | x |
| R11 前提 | WT-37、WT-57 | x |
| ds 链路 | WT-33（探针通过）、WT-62 | x |
"""
SECTION = """\
### 3.4 表 A（2 个主行）
| 功能 | 用例 | 用例作者 | 依赖集合 |
|---|---|---|---|
{F90_ROW}
| F-91 y（04：—；03：—） | — | — | 不设 E2E：两个 case 都是 —（不适用）；XD-623 留在 CPU 门槛里 |
### 3.6 表 D（3 个主行）
| 功能 | 用例 | 用例作者 | 依赖集合 |
|---|---|---|---|
| EG-90 z（✅） | XD-213 | 现成（R11 执行） | R11 前提 |
{EG91_ROW}
| EG-92 u（✅） | XD-005、PU-02 | 现成 | 链路；GPU 起法随 PB-1a 的 PU-02 取证（WT-70）：PU-02 跑通 |
### 3.7 表 C
| 功能 | 用例 | 用例作者 | 依赖集合 |
|---|---|---|---|
| DS-90 v（✅） | NT-147 | WT-62 | ds 链路、WT-71（CP-11 (a)、CP-06）；R13 时 WT-71 未合入的由 WT-73 补 |
### 3.8 其他
| 功能 | 用例 | 用例作者 | 依赖集合 |
| F-99 ignored | — | — | — |
""".replace("{F90_ROW}", F90_ROW).replace("{EG91_ROW}", EG91_ROW)


def test_the_table_parser():
    rows = {row["id"]: row for row in sync_feature_deps.derive(SECTION.replace("（3 个", "（4 个"))}
    assert list(rows) == [
        "F-90",
        "F-90+rest",
        "F-90@03",
        "F-91",
        "EG-90",
        "EG-91",
        "EG-92",
        "EG-92+PU-02",
        "DS-90",
    ]
    assert rows["F-90"]["cases"] == ["XD-001"] and rows["F-90"]["deps"] == ["link"]
    assert rows["F-90"]["hopper_cases"] == ["XD-036"] and rows["F-90"]["not_runnable"] == ["XD-202"]
    assert rows["F-90"]["author"] == ["existing"]
    assert rows["F-90+rest"]["cases"] == ["T2", "NT-050"]
    assert rows["F-90+rest"]["deps"] == ["link", "WT-57"] and rows["F-90+rest"]["author"] == [
        "WT-57"
    ]
    assert rows["F-90@03"]["chain"] == "case-03" and rows["F-90@03"]["deps"] == ["ds-link", "WT-53"]
    assert rows["F-91"]["deferred"] is True and rows["F-91"]["notes"] == [
        "不设 E2E：两个 case 都是 —（不适用）",
        "XD-623 留在 CPU 门槛里",
    ]
    assert rows["EG-90"]["assignee"] == "WT-60" and rows["EG-90"]["deps"] == ["r11-prereqs"]
    assert rows["EG-91"]["assignee"] == "WT-61" and rows["EG-91"]["deps"] == ["link", "WT-38"]
    assert rows["EG-91"]["hopper_cases"] == ["XD-041", "XD-042"]
    assert rows["EG-91"]["notes"] == [
        "XD-101 是死用例",
        "链路、WT-38（CT-15）",
        "XD-009 由 WT-40 先补跑",
    ]
    assert rows["EG-92"]["cases"] == ["XD-005"] and "assignee" not in rows["EG-92"]
    assert rows["EG-92+PU-02"]["cases"] == ["PU-02"] and rows["EG-92+PU-02"]["assignee"] == "WT-70"
    assert rows["EG-92+PU-02"]["deps"] == ["link", "WT-70"]
    assert rows["DS-90"]["fallback"] == "WT-73" and rows["DS-90"]["deps"] == ["ds-link", "WT-71"]


REPO_TABLE = """\
### 0.2 仓库分支与合入标记
| 仓库 | 地址 | 集成分支 | 起点 | 适用工作项 |
|---|---|---|---|---|
| ds-engine 自有仓库 | x | y | z | WT-41、WT-42、WT-43；WT-71 的 `E3-mid` 部分（CD-09） |
"""


def test_the_groups_come_from_section_3_2():
    assert sync_feature_deps.derive_groups(REPO_TABLE + GROUPS_TABLE) == {
        "link": ["WT-35", "WT-37"],
        "ds-link": ["WT-33", "WT-62"],
        "r11-prereqs": ["WT-37", "WT-57"],
        "ds-repo": ["WT-41", "WT-42", "WT-43"],
    }
    with pytest.raises(ValueError, match="does not define"):
        sync_feature_deps.derive_groups(
            REPO_TABLE + GROUPS_TABLE.replace("| R11 前提 |", "| R12 |")
        )
    with pytest.raises(ValueError, match="ds-repo"):
        sync_feature_deps.derive_groups(GROUPS_TABLE)


def test_the_table_parser_rejects_unknown_dependencies():
    bad = SECTION.replace("04：链路；", "04：链路、某事；")
    with pytest.raises(ValueError, match="unknown dependency"):
        sync_feature_deps.derive(bad)


def test_the_table_parser_rejects_a_row_without_cases():
    bad = SECTION.replace("| EG-90 z（✅） | XD-213 |", "| EG-90 z（✅） | — |")
    with pytest.raises(ValueError, match="no case verifies"):
        sync_feature_deps.derive(bad.replace("（3 个", "（4 个"))


def test_qualifiers_and_reasons_are_kept_as_notes():
    deps = sync_feature_deps.parse_deps(["链路（断言随 WT-52）", "R12 验收（条件：DQ-42）"])
    assert deps["deps"] == ["link"] and deps["assignee"] == "WT-61"
    assert deps["notes"] == ["链路（断言随 WT-52）", "R12 验收（条件：DQ-42）"]
    deferred = sync_feature_deps.parse_deps(["不设 E2E：—（不适用）", "另见 DG-1"])
    assert deferred["deferred"] and deferred["notes"] == ["不设 E2E：—（不适用）", "另见 DG-1"]
    plain = sync_feature_deps.parse_deps(["链路、WT-57", "其余用例另等 WT-57"])
    assert plain["notes"] == []


def test_r11_prerequisites_make_r11_the_verifier_with_other_dependencies_too():
    deps = sync_feature_deps.parse_deps(["R11 前提、WT-53"])
    assert deps["deps"] == ["r11-prereqs", "WT-53"] and deps["assignee"] == "WT-60"


def test_case_ids_next_to_chinese_text_are_found():
    assert sync_feature_deps._cases("用XD-001与NT-002、T3跑") == ["XD-001", "NT-002", "T3"]
    assert sync_feature_deps._cases("CT-15、XD-0011") == []


def test_text_before_the_chain_segments_is_rejected():
    bad = SECTION.replace("| 04：链路；其余", "| 链路；04：链路；其余")
    with pytest.raises(ValueError, match="before the 04/03 segments"):
        sync_feature_deps.derive(bad.replace("（3 个", "（4 个"))


def test_the_table_parser_counts_main_rows():
    with pytest.raises(ValueError, match="distinct feature rows"):
        sync_feature_deps.derive(SECTION)


# --- design invariants --------------------------------------------------------------------------

PROTOCOL_SRC = """\
from typing import Optional


class DisaggregatedParams(OpenAIBaseModel):
    request_type: str
    ctx_request_id: Optional[int] = None


def to_disaggregated_params(params):
    return params


def to_llm_disaggregated_params(params):
    return params
"""
SERVE_SRC = """\
import click


@click.command("disaggregated")
@click.option("-c", "--config_file")
def disaggregated(config_file):
    return config_file


@click.command("disaggregated_mpi_worker")
def disaggregated_mpi_worker():
    return None
"""
AUTH_SRC = """\
_INTERNAL_DISAGG_AUTH_FIELDS = ("ctx_info_endpoint", "encoded_opaque_state")


def _auth_payload(params):
    return {name: getattr(params, name) for name in _INTERNAL_DISAGG_AUTH_FIELDS}
"""
NEW_FIELD = "    openengine_handoff_attributes: Optional[str] = None\n"
NEW_AUTH = 'EXTRA_FIELD = "openengine_handoff_attributes"\n'
EMPTY_REGISTRY = "version: 1\nadditions: []\n"
TEST_REGISTRY = yaml.safe_dump(
    {
        "version": 1,
        "additions": [
            {
                "change": "CS-20",
                "file": check_invariants.PROTOCOL,
                "region": "DisaggregatedParams",
                "lines": [NEW_FIELD.rstrip("\n")],
            },
            {
                "change": "CS-20",
                "file": check_invariants.AUTH,
                "region": "file",
                "lines": [NEW_AUTH.rstrip("\n")],
            },
        ],
    }
)


@pytest.fixture
def invariant_repo(tmp_path):
    repo = _repo(tmp_path)
    files = {path: f"# {path}\n" for path in check_invariants.ORCHESTRATOR_FILES}
    files |= {
        check_invariants.PROTOCOL: PROTOCOL_SRC,
        check_invariants.SERVE: SERVE_SRC,
        check_invariants.AUTH: AUTH_SRC,
        check_invariants.REGISTRY_PATH: EMPTY_REGISTRY,
    }
    base = _commit(repo, files, "base")
    return repo, base


def _invariants(repo, base, files, registry=None):
    _commit(repo, files, "change")
    results = check_invariants.run(repo, base, "HEAD", registry)
    return {name for name, passed, _ in results if not passed}


def test_invariants_hold_on_the_base(invariant_repo):
    repo, base = invariant_repo
    assert _invariants(repo, base, {}) == set()


@pytest.mark.parametrize(
    "path, old, new, failing",
    [
        ("tensorrt_llm/serve/router.py", None, "# changed\n", "file tensorrt_llm/serve/router.py"),
        (
            check_invariants.PROTOCOL,
            "    return params\n\n\ndef to_llm",
            "    return dict(params)\n\n\ndef to_llm",
            f"def {check_invariants.PROTOCOL}:to_disaggregated_params",
        ),
        (
            check_invariants.PROTOCOL,
            "    return params\n",
            "    return params\n    # a trailing comment in the body\n",
            f"def {check_invariants.PROTOCOL}:to_disaggregated_params",
        ),
        (
            check_invariants.PROTOCOL,
            "def to_llm_disaggregated_params(params):\n    return params\n",
            "def to_llm_disaggregated_params(params):\n    return dict(params)\n",
            f"def {check_invariants.PROTOCOL}:to_llm_disaggregated_params",
        ),
        (
            check_invariants.PROTOCOL,
            "\n\ndef to_llm",
            "\n\nto_disaggregated_params = dict\n\n\ndef to_llm",
            f"def {check_invariants.PROTOCOL}:to_disaggregated_params",
        ),
        (
            check_invariants.SERVE,
            '@click.option("-c", "--config_file")',
            '@click.option("--config_file")',
            f"def {check_invariants.SERVE}:disaggregated",
        ),
        (
            check_invariants.SERVE,
            "    return None\n",
            "    return 0\n",
            f"def {check_invariants.SERVE}:disaggregated_mpi_worker",
        ),
        (
            check_invariants.PROTOCOL,
            "    request_type: str\n",
            "    request_type: Optional[str]\n",
            f"additions {check_invariants.PROTOCOL}:DisaggregatedParams",
        ),
        (
            check_invariants.AUTH,
            '"encoded_opaque_state")',
            '"encoded_opaque_state", "x")',
            f"additions {check_invariants.AUTH}:file",
        ),
        (
            check_invariants.PROTOCOL,
            "    request_type: str\n",
            "    request_type: str\n" + NEW_FIELD,
            f"additions {check_invariants.PROTOCOL}:DisaggregatedParams",
        ),
        (
            check_invariants.AUTH,
            'state")\n',
            'state")\n' + NEW_AUTH,
            f"additions {check_invariants.AUTH}:file",
        ),
    ],
    ids=[
        "orchestrator-file",
        "conversion-function",
        "trailing-comment",
        "other-conversion-function",
        "rebinding",
        "subcommand-decorator",
        "subcommand-body",
        "class-existing-line",
        "auth-existing-line",
        "class-unregistered-line",
        "auth-unregistered-line",
    ],
)
def test_each_kind_of_violation_fails(invariant_repo, path, old, new, failing):
    repo, base = invariant_repo
    source = {
        check_invariants.PROTOCOL: PROTOCOL_SRC,
        check_invariants.SERVE: SERVE_SRC,
        check_invariants.AUTH: AUTH_SRC,
    }.get(path)
    text = new if old is None else source.replace(old, new, 1)
    assert text != (source or "")
    assert _invariants(repo, base, {path: text}) == {failing}


def test_registered_additions_pass(invariant_repo):
    repo, base = invariant_repo
    files = {
        check_invariants.PROTOCOL: PROTOCOL_SRC.replace(
            "    request_type: str\n", "    request_type: str\n" + NEW_FIELD
        ),
        check_invariants.AUTH: AUTH_SRC.replace('state")\n', 'state")\n' + NEW_AUTH),
    }
    assert _invariants(repo, base, files, TEST_REGISTRY) == set()
    # A line registered once may be added once; a second copy is unregistered.
    doubled = PROTOCOL_SRC.replace(
        "    request_type: str\n", "    request_type: str\n" + NEW_FIELD * 2
    )
    assert _invariants(repo, base, {check_invariants.PROTOCOL: doubled}, TEST_REGISTRY) == {
        f"additions {check_invariants.PROTOCOL}:DisaggregatedParams"
    }


@pytest.mark.parametrize("change", ["crlf", "mode"])
def test_whole_files_compare_bytes_and_mode(invariant_repo, change):
    repo, base = invariant_repo
    path = "tensorrt_llm/serve/router.py"
    if change == "crlf":
        (repo / path).write_bytes(f"# {path}\r\n".encode())
    else:
        (repo / path).chmod(0o755)
    _git(repo, "add", path)
    assert _invariants(repo, base, {}) == {f"file {path}"}


@pytest.mark.parametrize(
    "addition",
    [
        "if True:\n    def disaggregated():\n        return 1\n",
        "try:\n    from x import y as disaggregated\nexcept ImportError:\n    pass\n",
        "for disaggregated in ():\n    pass\n",
        "with open(__file__) as disaggregated:\n    pass\n",
        "if (disaggregated := None):\n    pass\n",
        "try:\n    pass\nexcept Exception as disaggregated:\n    pass\n",
        "match 1:\n    case disaggregated:\n        pass\n",
        "match 1:\n    case int() as disaggregated:\n        pass\n",
        "match []:\n    case [*disaggregated]:\n        pass\n",
        "match {}:\n    case {**disaggregated}:\n        pass\n",
        "match 1:\n    case 1 if (disaggregated := 1):\n        pass\n",
        "match 1:\n    case 1:\n        disaggregated = 2\n",
        "def _swap():\n    global disaggregated\n    disaggregated = print\n",
        "def g(a=(disaggregated := print)):\n    return a\n",
        "@(disaggregated := print)\ndef g():\n    pass\n",
        "class K((disaggregated := object)):\n    pass\n",
        "f = lambda a=(disaggregated := 1): a\n",
        "[(disaggregated := x) for x in ()]\n",
        "del disaggregated\n",
        "disaggregated += 1\n",
        "(a, [disaggregated]) = (1, [2])\n",
        "disaggregated: int = 1\n",
        "f = lambda *, a=(disaggregated := 1): a\n",
        "names = [x for x in () if (disaggregated := x)]\n",
        "try:\n    pass\nexcept (disaggregated := Exception):\n    pass\n",
        "x: (disaggregated := int)\n",
        "(disaggregated := print).x: int\n",
    ],
    ids=[
        "def-in-if",
        "import-in-try",
        "for-target",
        "with-target",
        "walrus",
        "except-as",
        "match-capture",
        "match-as",
        "match-star",
        "match-rest",
        "match-guard-walrus",
        "match-case-body",
        "global",
        "default-walrus",
        "decorator-walrus",
        "base-walrus",
        "lambda-default-walrus",
        "comprehension-walrus",
        "del",
        "augmented",
        "nested-target",
        "annotated-assignment",
        "lambda-keyword-default-walrus",
        "comprehension-condition-walrus",
        "except-type-walrus",
        "bare-annotation-walrus",
        "bare-annotation-target-walrus",
    ],
)
def test_a_second_module_scope_binding_fails(invariant_repo, addition):
    repo, base = invariant_repo
    files = {check_invariants.SERVE: SERVE_SRC + "\n\n" + addition}
    assert _invariants(repo, base, files) == {f"def {check_invariants.SERVE}:disaggregated"}


def test_a_star_import_may_bind_any_name(invariant_repo):
    repo, base = invariant_repo
    files = {check_invariants.SERVE: SERVE_SRC + "\n\nfrom os.path import *\n"}
    assert _invariants(repo, base, files) == {
        f"def {check_invariants.SERVE}:disaggregated",
        f"def {check_invariants.SERVE}:disaggregated_mpi_worker",
    }


@pytest.mark.parametrize(
    "addition",
    [
        "disaggregated.help = 'x'\n",
        "registry = {}\nregistry[disaggregated] = 1\n",
        "registry = {}\nregistry[disaggregated.name] = disaggregated\n",
        "f = lambda: (disaggregated := 1)\n",
        "names = [disaggregated for disaggregated in ()]\n",
        "class K:\n    disaggregated = 1\n",
        "def g(disaggregated=1):\n    return disaggregated\n",
        "x = disaggregated\n",
        "disaggregated: int\n",
    ],
    ids=[
        "attribute-target",
        "subscript-key",
        "subscript-value",
        "lambda-body-walrus",
        "comprehension-target",
        "class-body",
        "parameter",
        "use",
        "bare-annotation",
    ],
)
def test_uses_of_the_name_are_not_bindings(invariant_repo, addition):
    repo, base = invariant_repo
    files = {check_invariants.SERVE: SERVE_SRC + "\n\n" + addition}
    assert _invariants(repo, base, files) == set()


def test_local_names_in_nested_bodies_are_not_bindings(invariant_repo):
    repo, base = invariant_repo
    addition = (
        "if True:\n    def helper():\n        disaggregated = 1\n        return disaggregated\n"
    )
    files = {check_invariants.SERVE: SERVE_SRC + "\n\n" + addition}
    assert _invariants(repo, base, files) == set()


@pytest.mark.parametrize("setting", ["fileMode", "autocrlf", "attributes", "assume", "skip"])
def test_working_tree_settings_do_not_hide_a_changed_file(invariant_repo, setting):
    repo, base = invariant_repo
    path = "tensorrt_llm/serve/router.py"
    file = repo / path
    if setting == "fileMode":
        _git(repo, "config", "core.fileMode", "false")
        file.chmod(0o755)
    elif setting in ("autocrlf", "attributes"):
        if setting == "autocrlf":
            _git(repo, "config", "core.autocrlf", "true")
        else:
            (repo / ".git/info/attributes").write_text("*.py text\n")
        file.write_bytes(f"# {path}\r\n".encode())
    else:
        flag = "--assume-unchanged" if setting == "assume" else "--skip-worktree"
        _git(repo, "update-index", flag, path)
        file.write_text("# changed\n")
    results = check_invariants.run(repo, base, "")
    assert {name for name, passed, _ in results if not passed} == {f"file {path}"}


@pytest.mark.parametrize("variable", PATHSPEC_VARIABLES)
@pytest.mark.parametrize("head", ["HEAD", ""], ids=["head", "working-tree"])
def test_whole_files_are_compared_whatever_the_pathspec_variables(
    invariant_repo, monkeypatch, variable, head
):
    repo, base = invariant_repo
    path = "tensorrt_llm/serve/router.py"
    (repo / path).write_text("# changed\n")
    if head:
        _git(repo, "commit", "-q", "-a", "-m", "change")
    monkeypatch.setenv(variable, "1")
    results = check_invariants.run(repo, base, head)
    assert {name for name, passed, _ in results if not passed} == {f"file {path}"}


def test_a_definition_file_under_a_symlinked_directory_fails(invariant_repo, tmp_path):
    repo, base = invariant_repo
    copy = tmp_path / "commands"
    copy.mkdir()
    (copy / "serve.py").write_bytes((repo / check_invariants.SERVE).read_bytes())
    (repo / check_invariants.SERVE).unlink()
    (repo / check_invariants.SERVE).parent.rmdir()
    (repo / check_invariants.SERVE).parent.symlink_to(copy)
    results = check_invariants.run(repo, base, "")
    assert {name for name, passed, _ in results if not passed} == {
        f"def {check_invariants.SERVE}:disaggregated",
        f"def {check_invariants.SERVE}:disaggregated_mpi_worker",
    }


def test_a_whole_file_under_a_symlinked_directory_fails(invariant_repo, tmp_path):
    repo, base = invariant_repo
    llmapi = repo / "tensorrt_llm/llmapi"
    copy = tmp_path / "llmapi"
    copy.mkdir()
    for file in llmapi.iterdir():
        (copy / file.name).write_bytes(file.read_bytes())
        file.unlink()
    llmapi.rmdir()
    llmapi.symlink_to(copy)
    results = check_invariants.run(repo, base, "")
    assert {name for name, passed, _ in results if not passed} == {
        "file tensorrt_llm/llmapi/disagg_utils.py"
    }


def test_an_owner_only_executable_bit_is_a_mode_change(invariant_repo):
    repo, base = invariant_repo
    path = "tensorrt_llm/serve/router.py"
    (repo / path).chmod(0o744)
    results = check_invariants.run(repo, base, "")
    assert {name for name, passed, _ in results if not passed} == {f"file {path}"}


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_an_unreadable_whole_file_fails_its_own_check(invariant_repo):
    repo, base = invariant_repo
    path = "tensorrt_llm/serve/router.py"
    (repo / path).chmod(0)
    try:
        results = check_invariants.run(repo, base, "")
    finally:
        (repo / path).chmod(0o644)
    assert {name for name, passed, _ in results if not passed} == {f"file {path}"}


def test_a_failing_diff_fails_only_the_file_checks(invariant_repo, monkeypatch):
    repo, base = invariant_repo

    def differ(_repo):
        def differs(base, head, path):
            raise RuntimeError("bad object")

        return differs

    monkeypatch.setattr(check_invariants, "git_differ", differ)
    failed = {name for name, passed, _ in check_invariants.run(repo, base, "HEAD") if not passed}
    assert failed == {f"file {path}" for path in check_invariants.ORCHESTRATOR_FILES}


def test_source_too_deep_to_parse_fails_only_its_own_check(invariant_repo, monkeypatch):
    repo, base = invariant_repo
    real = check_invariants.definition_text

    def definition_text(source, name, kind=ast.FunctionDef):
        if name == "disaggregated":
            raise RecursionError("maximum recursion depth exceeded")
        return real(source, name, kind)

    monkeypatch.setattr(check_invariants, "definition_text", definition_text)
    failed = {name for name, passed, _ in check_invariants.run(repo, base, "HEAD") if not passed}
    assert failed == {f"def {check_invariants.SERVE}:disaggregated"}


def test_a_class_too_deep_to_parse_fails_only_its_own_check(invariant_repo, monkeypatch):
    repo, base = invariant_repo
    real = check_invariants.definition_text

    def definition_text(source, name, kind=ast.FunctionDef):
        if name == "DisaggregatedParams":
            raise RecursionError("maximum recursion depth exceeded")
        return real(source, name, kind)

    monkeypatch.setattr(check_invariants, "definition_text", definition_text)
    failed = {name for name, passed, _ in check_invariants.run(repo, base, "HEAD") if not passed}
    assert failed == {f"additions {check_invariants.PROTOCOL}:DisaggregatedParams"}


def test_a_registry_too_deep_to_parse_fails_only_its_own_check(invariant_repo, monkeypatch):
    repo, base = invariant_repo

    def safe_load(text):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(check_invariants.yaml, "safe_load", safe_load)
    failed = {name for name, passed, _ in check_invariants.run(repo, base, "HEAD") if not passed}
    assert failed == {"registry"}


def test_a_symlink_on_disk_is_a_change_even_to_the_same_bytes(invariant_repo, tmp_path):
    repo, base = invariant_repo
    path = "tensorrt_llm/serve/router.py"
    copy = tmp_path / "router.py"
    copy.write_bytes((repo / path).read_bytes())
    (repo / path).unlink()
    (repo / path).symlink_to(copy)
    results = check_invariants.run(repo, base, "")
    assert {name for name, passed, _ in results if not passed} == {f"file {path}"}


def test_working_tree_files_match_the_base_byte_for_byte(invariant_repo):
    repo, base = invariant_repo
    _git(repo, "config", "core.autocrlf", "true")
    results = check_invariants.run(repo, base, "")
    assert all(passed for _, passed, _ in results)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_an_unreadable_file_on_disk_fails_its_own_check(invariant_repo):
    repo, base = invariant_repo
    (repo / check_invariants.SERVE).chmod(0)
    try:
        results = check_invariants.run(repo, base, "")
    finally:
        (repo / check_invariants.SERVE).chmod(0o644)
    failed = {name for name, passed, _ in results if not passed}
    assert failed == {
        f"def {check_invariants.SERVE}:disaggregated",
        f"def {check_invariants.SERVE}:disaggregated_mpi_worker",
    }


def test_crlf_inside_a_frozen_definition_fails(invariant_repo):
    repo, base = invariant_repo
    text = SERVE_SRC.replace("    return None\n", "    return None\r\n")
    assert _invariants(repo, base, {check_invariants.SERVE: text}) == {
        f"def {check_invariants.SERVE}:disaggregated_mpi_worker"
    }


def test_crlf_on_disk_inside_a_frozen_definition_fails(invariant_repo):
    repo, base = invariant_repo
    text = SERVE_SRC.replace("    return None\n", "    return None\r\n")
    (repo / check_invariants.SERVE).write_bytes(text.encode())
    results = check_invariants.run(repo, base, "")
    assert {name for name, passed, _ in results if not passed} == {
        f"def {check_invariants.SERVE}:disaggregated_mpi_worker"
    }


@pytest.mark.parametrize(
    "path, data, failing",
    [
        (check_invariants.SERVE, b"\xff\xfe not utf-8\n", "disaggregated"),
        (check_invariants.SERVE, SERVE_SRC.encode() + b"x = 1\x00\n", "disaggregated"),
        (check_invariants.AUTH, b"\xff\n", "file"),
        (check_invariants.REGISTRY_PATH, b"additions: [\n", "registry"),
        (check_invariants.REGISTRY_PATH, b"\xff\n", "registry"),
        (check_invariants.REGISTRY_PATH, b"version: 1\nwhen: 2026-02-30\n", "registry"),
    ],
    ids=[
        "non-utf8-definition",
        "nul-byte",
        "non-utf8-region",
        "registry-yaml",
        "registry-bytes",
        "registry-value",
    ],
)
def test_unreadable_input_fails_its_check_without_a_traceback(invariant_repo, path, data, failing):
    repo, base = invariant_repo
    (repo / path).write_bytes(data)
    _git(repo, "add", path)
    _commit(repo, {}, "bad input")
    failed = {name for name, passed, _ in check_invariants.run(repo, base, "HEAD") if not passed}
    assert failed and all(failing in name or name == "registry" for name in failed), failed
    assert any(failing in name for name in failed)


def test_bad_arguments_fail_the_invariants_with_a_message(invariant_repo, tmp_path, capsys):
    repo, base = invariant_repo
    (tmp_path / "bad.yaml").write_bytes(b"\xff\n")
    assert (
        check_invariants.main(["--repo", str(repo), "--registry", str(tmp_path / "bad.yaml")]) == 1
    )
    assert "FAIL: UnicodeDecodeError" in capsys.readouterr().out
    assert check_invariants.main(["--repo", str(repo), "--base", "no-such-rev"]) == 1
    assert "FAIL: RuntimeError: unknown revision 'no-such-rev'" in capsys.readouterr().out
    assert (
        check_invariants.main(["--repo", str(repo), "--base", base, "--head", "no-such-rev"]) == 1
    )
    assert "FAIL: RuntimeError: unknown revision 'no-such-rev'" in capsys.readouterr().out
    assert check_invariants.main(["--repo", str(repo), "--registry", str(tmp_path / "none")]) == 1
    assert "FAIL: FileNotFoundError" in capsys.readouterr().out
    (tmp_path / "plain").mkdir()
    assert check_invariants.main(["--repo", str(tmp_path / "plain")]) == 1
    assert "not a git repository" in capsys.readouterr().out


def test_a_head_whose_gate_file_is_not_utf8_still_needs_its_registry(invariant_repo):
    repo, base = invariant_repo
    (repo / "scripts/poc3/check_invariants.py").write_bytes(b"\xff\n")
    _git(repo, "add", "scripts/poc3/check_invariants.py")
    _git(repo, "rm", "-q", check_invariants.REGISTRY_PATH)
    results = {name: detail for name, passed, detail in _results(repo, base, {}) if not passed}
    assert list(results) == ["registry"] and "missing" in results["registry"]


@pytest.mark.parametrize(
    "old, new",
    [
        ('"encoded_opaque_state")\n', '"encoded_opaque_state")\n'.replace("\n", "")),
        (
            '_INTERNAL_DISAGG_AUTH_FIELDS = ("ctx_info_endpoint", "encoded_opaque_state")\n\n\n',
            "\n\n",
        ),
    ],
    ids=["joined-line", "deleted-line"],
)
def test_removing_an_existing_auth_line_fails(invariant_repo, old, new):
    repo, base = invariant_repo
    text = AUTH_SRC.replace(old, new, 1)
    assert text != AUTH_SRC
    assert _invariants(repo, base, {check_invariants.AUTH: text}, TEST_REGISTRY) == {
        f"additions {check_invariants.AUTH}:file"
    }


def test_reordering_existing_lines_fails(invariant_repo):
    repo, base = invariant_repo
    lines = AUTH_SRC.splitlines(keepends=True)
    reordered = "".join([lines[-1], *lines[:-1]])
    assert _invariants(repo, base, {check_invariants.AUTH: reordered}) == {
        f"additions {check_invariants.AUTH}:file"
    }


def test_a_head_with_the_gate_but_no_registry_fails(invariant_repo):
    repo, base = invariant_repo
    _commit(repo, {"scripts/poc3/check_invariants.py": "# gate\n"}, "gate")
    _git(repo, "rm", "-q", check_invariants.REGISTRY_PATH)
    results = {name: detail for name, passed, detail in _results(repo, base, {}) if not passed}
    assert list(results) == ["registry"] and "missing" in results["registry"]


def _results(repo, base, files, head="HEAD"):
    _commit(repo, files, "change")
    return check_invariants.run(repo, base, head)


def test_the_head_and_the_working_tree_are_told_apart(invariant_repo):
    repo, base = invariant_repo
    (repo / "tensorrt_llm/serve/router.py").write_text("# on disk only\n")
    assert {n for n, ok, _ in check_invariants.run(repo, base, "HEAD") if not ok} == set()
    assert {n for n, ok, _ in check_invariants.run(repo, base, "") if not ok} == {
        "file tensorrt_llm/serve/router.py"
    }


def test_a_relative_diff_config_does_not_hide_a_changed_file(invariant_repo):
    repo, base = invariant_repo
    _git(repo, "config", "diff.relative", "true")
    _commit(repo, {"tensorrt_llm/serve/router.py": "# changed\n"}, "change")
    results = check_invariants.run(repo / "tensorrt_llm/serve", base, "HEAD")
    assert {n for n, ok, _ in results if not ok} == {"file tensorrt_llm/serve/router.py"}


def test_a_relative_diff_config_does_not_hide_a_file_outside_the_directory(invariant_repo):
    repo, base = invariant_repo
    _git(repo, "config", "diff.relative", "true")
    _commit(repo, {"tensorrt_llm/llmapi/disagg_utils.py": "# changed\n"}, "change")
    results = check_invariants.run(repo / "tensorrt_llm/serve", base, "HEAD")
    assert {n for n, ok, _ in results if not ok} == {"file tensorrt_llm/llmapi/disagg_utils.py"}


def test_the_working_tree_is_checked_from_a_subdirectory(invariant_repo):
    repo, base = invariant_repo
    (repo / "tensorrt_llm/serve/router.py").write_text("# changed on disk\n")
    sub = repo / "tensorrt_llm"
    results = check_invariants.run(sub, base, "")
    assert {name for name, passed, _ in results if not passed} == {
        "file tensorrt_llm/serve/router.py"
    }


def test_moving_a_frozen_definition_is_fine(invariant_repo):
    repo, base = invariant_repo
    head, tail = PROTOCOL_SRC.split("\n\ndef to_disaggregated_params")
    moved = head + "\n\nX = 1\n\n\ndef to_disaggregated_params" + tail
    assert _invariants(repo, base, {check_invariants.PROTOCOL: moved}) == set()


@pytest.mark.parametrize(
    "entry, problem",
    [
        (
            {"change": "CS-04", "file": check_invariants.AUTH, "region": "file", "lines": ["x"]},
            "only CS-20",
        ),
        (
            {
                "change": "CS-20",
                "file": "tensorrt_llm/serve/router.py",
                "region": "file",
                "lines": ["x"],
            },
            "not a guarded region",
        ),
        ({"change": "CS-20", "file": check_invariants.AUTH, "region": "file"}, "no list of lines"),
    ],
)
def test_registry_entries_are_validated(entry, problem):
    _, problems = check_invariants.load_registry(yaml.safe_dump({"additions": [entry]}))
    assert len(problems) == 1 and problem in problems[0]


def test_a_missing_definition_fails(invariant_repo):
    repo, base = invariant_repo
    gone = SERVE_SRC.replace("def disaggregated_mpi_worker", "def renamed_worker")
    assert _invariants(repo, base, {check_invariants.SERVE: gone}) == {
        f"def {check_invariants.SERVE}:disaggregated_mpi_worker"
    }


def test_the_shipped_registry_is_valid():
    assert check_invariants.load_registry(check_invariants.REGISTRY.read_text())[1] == []


def test_a_registry_that_is_not_a_mapping_is_a_problem():
    assert check_invariants.load_registry("- a\n- b\n")[1] == [
        "the registry is not a mapping with a list of additions"
    ]


def test_invariants_hold_on_the_head_of_this_checkout():
    repo = Path(__file__).resolve().parents[2]
    if (
        subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", check_invariants.FROZEN_BASE + "^{commit}"],
            capture_output=True,
        ).returncode
        != 0
    ):
        pytest.skip("the frozen base is not in this checkout")
    assert check_invariants.main(["--repo", str(repo), "--head", "HEAD"]) == 0

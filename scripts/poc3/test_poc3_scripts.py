# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of the PoC-S3 gate scripts.

Run from the repository root::

    python3 -m pytest -p no:cacheprovider scripts/poc3/test_poc3_scripts.py

``POC3_TASKS_MD`` (the plan's tasks.md) enables the check that
``feature_deps.yaml`` matches the plan's tables; without it that test skips.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# The scripts import poc3_common themselves; pytest puts this directory on sys.path.
import check_invariants  # noqa: I001
import check_ownership
import pending_e2e
import pytest
import sync_feature_deps
import yaml
from poc3_common import landed_tasks, parse_task, trailers

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


def test_trailers_read_only_the_poc3_key():
    message = "Subject\n\nBody\n\nPoC2-Task: WT-10\nPoC3-Task: WT-30\n"
    assert trailers(message) == ["WT-30"]
    assert trailers("PoC2-Task: WT-10\n") == []


def test_landed_tasks_ignore_poc2_trailers(tmp_path):
    repo = _repo(tmp_path)
    _commit(repo, {}, "s2\n\nPoC2-Task: WT-10")
    _commit(repo, {}, "s2 round\n\nPoC2-Task: WT-20-R3")
    _commit(repo, {}, "s3\n\nPoC3-Task: WT-30")
    _commit(repo, {}, "s3 fix\n\nPoC3-Task: WT-51-fix1")
    _commit(repo, {}, "bad\n\nPoC3-Task: WT-12")
    assert landed_tasks(repo, "main") == {"WT-30", "WT-51-fix1"}


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
    assert len(taken) == 17
    for path in taken:
        assert check_ownership.match_rule("WT-57", path) is None, path


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
    assert rows["F-91"]["deferred"] is True and rows["F-91"]["notes"] == ["XD-623 留在 CPU 门槛里"]
    assert rows["EG-90"]["assignee"] == "WT-60" and rows["EG-90"]["deps"] == ["r11-prereqs"]
    assert rows["EG-91"]["assignee"] == "WT-61" and rows["EG-91"]["deps"] == ["link", "WT-38"]
    assert rows["EG-91"]["hopper_cases"] == ["XD-041", "XD-042"]
    assert rows["EG-91"]["notes"] == ["XD-101 是死用例", "XD-009 由 WT-40 先补跑"]
    assert rows["EG-92"]["cases"] == ["XD-005"] and "assignee" not in rows["EG-92"]
    assert rows["EG-92+PU-02"]["cases"] == ["PU-02"] and rows["EG-92+PU-02"]["assignee"] == "WT-70"
    assert rows["EG-92+PU-02"]["deps"] == ["link", "WT-70"]
    assert rows["DS-90"]["fallback"] == "WT-73" and rows["DS-90"]["deps"] == ["ds-link", "WT-71"]


def test_the_groups_come_from_section_3_2():
    assert sync_feature_deps.derive_groups(GROUPS_TABLE) == {
        "link": ["WT-35", "WT-37"],
        "ds-link": ["WT-33", "WT-62"],
        "r11-prereqs": ["WT-37", "WT-57"],
        "ds-repo": ["WT-41", "WT-42", "WT-43"],
    }
    with pytest.raises(ValueError, match="does not define"):
        sync_feature_deps.derive_groups(GROUPS_TABLE.replace("| R11 前提 |", "| R12 |"))


def test_the_table_parser_rejects_unknown_dependencies():
    bad = SECTION.replace("04：链路；", "04：链路、某事；")
    with pytest.raises(ValueError, match="unknown dependency"):
        sync_feature_deps.derive(bad)


def test_the_table_parser_rejects_a_row_without_cases():
    bad = SECTION.replace("| EG-90 z（✅） | XD-213 |", "| EG-90 z（✅） | — |")
    with pytest.raises(ValueError, match="no case verifies"):
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

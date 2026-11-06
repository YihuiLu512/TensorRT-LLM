# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the PoC-S2 tooling: the ownership checker and the end-to-end obligation script."""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

__extra_import_path__ = ["~/scripts/poc2"]
import check_ownership  # noqa: E402
import pending_e2e  # noqa: E402
from poc2_common import landed_tasks, parse_task  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

LINK_ROWS = ["F-01+F-02", "F-05", "F-06", "F-07", "F-14", "F-41", "F-47"]
WT11_LINK_ROWS = ["F-08", "F-09", "F-11+F-12", "F-13", "F-19", "F-23", "F-25+F-26", "F-35"]
WT12_ROWS = ["F-03+F-04", "F-15+F-16", "F-17+F-18+F-30", "F-20+F-21", "F-24", "F-27+F-28"]
LINK = {"WT-01", "WT-03", "WT-04", "WT-10"}
TRTLLM_FEATURES = {f"WT-0{i}" for i in range(1, 6)} | {f"WT-1{i}" for i in range(0, 7)}


@pytest.fixture(scope="module")
def rows():
    return pending_e2e.load_rows()[1]


def _owed(task, landed, rows, verified=()):
    result = pending_e2e.obligations(task, landed, rows, verified)
    return (
        [row.id for row in result.owed],
        [row.id for row in result.left_to_r3],
        [row.id for row in result.unverified],
    )


def test_task_names():
    assert parse_task("WT-01") == ("WT-01", None)
    assert parse_task("WT-11-fix2") == ("WT-11", 2)
    assert parse_task("WT-20-R3") == ("WT-20-R3", None)
    for bad in ("WT-08", "WT-17", "WT-1", "WT-11-fix0", "wt-11", "WT-20"):
        with pytest.raises(ValueError):
            parse_task(bad)


def test_every_work_item_name_has_an_ownership_entry():
    names = [f"WT-{i:02d}" for i in range(100)] + ["WT-20-R2", "WT-20-R3"]
    accepted = set()
    for name in names:
        try:
            accepted.add(parse_task(name)[0])
        except ValueError:
            pass
    assert accepted == set(check_ownership.OWNERSHIP)


def test_wt01_owes_nothing(rows):
    assert _owed("WT-01", set(), rows) == ([], [], [])


def test_last_link_item_owes_the_link_rows(rows):
    # Order A: WT-03, WT-04, WT-01 landed; WT-10 completes the link.
    owed, r3, unverified = _owed("WT-10", {"WT-03", "WT-04", "WT-01"}, rows)
    assert owed == LINK_ROWS
    assert r3 == []
    assert unverified == ["F-33+F-34", "T17"]
    # With WT-03's rows verified, they are not reported.
    assert _owed("WT-10", {"WT-03", "WT-04", "WT-01"}, rows, {"F-33+F-34", "T17"})[2] == []


def test_wt03_landing_last_owes_its_own_rows_and_the_link(rows):
    # Order B: WT-01, WT-10, WT-04 landed; WT-03 completes the link.
    owed, _, _ = _owed("WT-03", {"WT-01", "WT-10", "WT-04"}, rows)
    assert owed == ["F-33+F-34", "T17"] + LINK_ROWS


def test_feature_items_landing_before_the_link_hand_their_rows_on(rows):
    # Order C: WT-11 lands before the link is complete.
    assert _owed("WT-11", {"WT-01", "WT-02"}, rows)[0] == ["F-44-cpu"]
    # WT-10 then completes the link and owes WT-11's link rows too.
    landed = {"WT-01", "WT-02", "WT-11", "WT-03", "WT-04"}
    verified = {"F-33+F-34", "T17", "F-44-cpu"}
    owed, _, unverified = _owed("WT-10", landed, rows, verified)
    assert owed == LINK_ROWS + WT11_LINK_ROWS
    assert unverified == []


def test_feature_item_after_the_link(rows):
    landed = LINK | {"WT-02", "WT-11"}
    owed, _, _ = _owed("WT-12", landed, rows)
    assert owed == WT12_ROWS
    # WT-05 last: logit_bias and the chain scheduling row; thinking budget needs WT-12 too.
    owed, _, _ = _owed("WT-05", landed, rows)
    assert owed == ["F-22", "F-44-chain"]
    owed, _, _ = _owed("WT-05", landed | {"WT-12"}, rows)
    assert owed == ["F-22", "F-29", "F-44-chain"]


def test_case01_rows(rows):
    # A glm item completing a case-01 row owes it.
    assert _owed("WT-07", LINK, rows)[0] == ["F-15-case01"]
    # A glm item landing before its row is complete owes nothing for it ...
    assert _owed("WT-06", LINK, rows)[0] == []
    # ... and the TRT-LLM item completing it leaves it to R3.
    owed, r3, _ = _owed("WT-12", LINK | {"WT-11", "WT-06"}, rows)
    assert owed == WT12_ROWS
    assert r3 == ["F-17+F-30+F-19-case01"]
    # A TRT-LLM item completing a glm row without its own dependency never owes it.
    assert "F-15-case01" not in _owed("WT-10", {"WT-01", "WT-03", "WT-04", "WT-07"}, rows)[0]


def test_rows_waiting_only_on_unchecked_glm_items_are_reported(rows):
    # WT-07 landed only in the glm repository, which was not read.
    result = pending_e2e.obligations(
        "WT-20-R3",
        TRTLLM_FEATURES | {"WT-06", "WT-20-R2"},
        rows,
        {"F-32"},
        unknown={"WT-06", "WT-07"},
    )
    unknown = [row.id for row in result.unknown]
    assert "F-15-case01" in unknown and "F-31+F-36+F-37+GLM" in unknown
    assert not {row.id for row in result.owed} & set(unknown)
    # An item known to have landed is not unknown.
    result = pending_e2e.obligations("WT-12", LINK | {"WT-11", "WT-06"}, rows, unknown={"WT-06"})
    assert [row.id for row in result.left_to_r3] == ["F-17+F-30+F-19-case01"]


def test_the_glm_group_comes_from_feature_deps(rows):
    owed, r3, _ = _owed("WT-07", LINK, rows)
    assert owed == ["F-15-case01"]
    result = pending_e2e.obligations("WT-07", LINK, rows, glm_tasks=())
    assert [row.id for row in result.left_to_r3] == ["F-15-case01"]
    groups, _ = pending_e2e.load_rows()
    assert groups["glm"] == ["WT-06", "WT-07"]


def test_entries_without_a_known_row_are_reported(rows, capsys):
    records = [
        {"entries": [{"features": ["F-15"]}, {"row": "F-99"}, {"row": "F-05"}]},
        None,
    ]
    assert pending_e2e.verified_rows(records, [row.id for row in rows]) == {"F-05"}
    err = capsys.readouterr().err
    assert "without a row id" in err and "unknown row 'F-99'" in err


def test_integration_rounds(rows):
    everything = TRTLLM_FEATURES | {"WT-06", "WT-07"}
    assert _owed("WT-20-R2", everything, rows)[0] == ["F-32"]
    # The last feature item does not owe the R2 row.
    assert "F-32" not in _owed("WT-16", TRTLLM_FEATURES - {"WT-16"}, rows)[0]
    owed, _, _ = _owed("WT-20-R3", everything | {"WT-20-R2"}, rows, {"F-32"})
    assert owed == ["F-15-case01", "F-17+F-30+F-19-case01", "F-31+F-36+F-37+GLM"]


def test_fixes_and_deferred_rows_owe_nothing(rows):
    assert _owed("WT-11-fix1", LINK | {"WT-11"}, rows)[0] == []
    assert all(row.id != "F-48" for row in rows if not row.deferred)
    everything = TRTLLM_FEATURES | {"WT-06", "WT-07", "WT-20-R2", "WT-20-R3"}
    assert "F-48" not in _owed("WT-20-R3", everything, rows)[0]


def _git(repo, *args):
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t",
    )
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env
    ).stdout


def _commit(repo, message, files=()):
    for path, text in files:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        _git(repo, "add", path)
    _git(repo, "commit", "--allow-empty", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _commit(tmp_path, "base", [("tensorrt_llm/__init__.py", "")])
    _git(tmp_path, "branch", "integration")
    return tmp_path


def test_landed_tasks_and_verified_records_from_git(repo):
    _git(repo, "checkout", "-q", "integration")
    _commit(repo, "a\n\nPoC2-Task: WT-01")
    _commit(repo, "b\n\nPoC2-Task: WT-11-fix1\n")
    _commit(repo, "c\n\nPoC2-Task: bogus\nPoC2-Task: WT-03")
    record = {"entries": [{"row": "F-33+F-34", "features": ["F-33", "F-34"]}]}
    _commit(
        repo,
        "d\n\nPoC2-Task: WT-03",
        [(f"{pending_e2e.VERIFIED_DIR}/WT-03.yaml", yaml.safe_dump(record))],
    )
    assert landed_tasks(repo, "integration") == {"WT-01", "WT-03", "WT-11-fix1"}
    assert pending_e2e.read_verified(repo, "integration") == {"F-33+F-34"}
    assert pending_e2e.read_verified(repo, "main") == set()
    assert pending_e2e.main(["--repo", str(repo), "--ref", "integration", "--task", "WT-10"]) == 0


def test_glm_items_are_read_from_the_glm_repository(repo, tmp_path, capsys):
    glm = tmp_path / "glm"
    glm.mkdir()
    _git(glm, "init", "-q", "-b", "main")
    _commit(glm, "base")
    _commit(glm, "x\n\nPoC2-Task: WT-07")
    # A TRT-LLM item's trailer in the glm repository is not counted ...
    _commit(glm, "y\n\nPoC2-Task: WT-11")
    _git(repo, "checkout", "-q", "integration")
    _commit(repo, "a\n\nPoC2-Task: WT-01")
    # ... nor a glm item's trailer on the TRT-LLM branch.
    _commit(repo, "b\n\nPoC2-Task: WT-06")
    base = ["--repo", str(repo), "--ref", "integration", "--json"]
    assert pending_e2e.main([*base, "--landed"]) == 0
    out = capsys.readouterr()
    assert yaml.safe_load(out.out) == ["WT-01"] and "--glm-repo not given" in out.err
    assert pending_e2e.main([*base, "--landed", "--glm-repo", str(glm), "--glm-ref", "main"]) == 0
    assert yaml.safe_load(capsys.readouterr().out) == ["WT-01", "WT-07"]


def test_ownership_in_bounds(repo, capsys):
    _git(repo, "checkout", "-q", "-b", "work")
    _commit(
        repo,
        "map\n\nPoC2-Task: WT-11",
        [
            ("tensorrt_llm/serve/openengine_poc/mapping.py", "x"),
            ("tests/unittest/serve/openengine_poc/wt11/test_x.py", "x"),
            ("tests/integration/defs/openengine_chain/smoke/WT-11.txt", "x"),
        ],
    )
    assert check_ownership.check("WT-11", "main..work", str(repo)) == 0
    assert "PASS" in capsys.readouterr().out
    # A fix owns what its work item owns.
    _commit(
        repo,
        "fix\n\nPoC2-Task: WT-11-fix1",
        [("tensorrt_llm/serve/openengine_poc/mapping.py", "y")],
    )
    assert check_ownership.check("WT-11-fix1", "work~1..work", str(repo)) == 0


def test_ownership_rejects_a_deliberately_out_of_bounds_change(repo, capsys):
    _git(repo, "checkout", "-q", "-b", "work")
    _commit(
        repo,
        "map\n\nPoC2-Task: WT-11",
        [
            ("tensorrt_llm/serve/openengine_poc/mapping.py", "x"),
            ("tensorrt_llm/serve/openengine_poc/response.py", "x"),
        ],
    )
    assert check_ownership.check("WT-11", "main..work", str(repo)) == 1
    out = capsys.readouterr().out
    assert "OUT OF BOUNDS  tensorrt_llm/serve/openengine_poc/response.py  (owned by WT-12)" in out
    assert "FAIL" in out


def test_ownership_rejects_commits_without_the_trailer(repo, capsys):
    _git(repo, "checkout", "-q", "-b", "work")
    _commit(repo, "map", [("tensorrt_llm/serve/openengine_poc/mapping.py", "x")])
    _commit(repo, "other\n\nPoC2-Task: WT-12")
    assert check_ownership.check("WT-11", "main..work", str(repo)) == 1
    out = capsys.readouterr().out
    assert out.count("BAD TRAILER") == 2


def test_shared_files_are_listed_for_review(repo, capsys):
    _git(repo, "checkout", "-q", "-b", "work")
    _commit(
        repo,
        "wire\n\nPoC2-Task: WT-01",
        [("tensorrt_llm/serve/openai_server.py", "x"), ("scripts/poc2/x.py", "x")],
    )
    assert check_ownership.check("WT-01", "main..work", str(repo)) == 0
    out = capsys.readouterr().out
    assert "CHECK BY HAND  SHARED" in out and "openai_server.py" in out
    assert "scripts/poc2/x.py" not in out


def test_glm_repository_rules(tmp_path):
    assert check_ownership.match_rule("WT-07", "csrc/kernels/sampling.cu", "glm") is not None
    assert check_ownership.match_rule("WT-07", "csrc/kernels/sampling.cu") is None
    assert check_ownership.match_rule("WT-06", "python/glm5_serve/openengine/servicer.py", "glm")
    assert check_ownership.match_rule(
        "WT-06", "tests/unittest/llmapi/apps/_test_openengine_glm53.py"
    )
    _git(tmp_path, "init", "-q")
    assert check_ownership.detect_repo(str(tmp_path)) == "glm"


def test_per_item_files_belong_to_their_item_only():
    chain = "tests/integration/defs/openengine_chain"
    assert [n for n, _ in check_ownership.owners(f"{chain}/verified/WT-10.yaml")] == ["WT-10"]
    assert [n for n, _ in check_ownership.owners(f"{chain}/verified/WT-20-R2.yaml")] == ["WT-20-R2"]
    assert [n for n, _ in check_ownership.owners(f"{chain}/harness.py")] == ["WT-03"]
    assert [n for n, _ in check_ownership.owners(f"{chain}/smoke/WT-03.txt")] == ["WT-03"]
    fake = "tests/unittest/grpc/openengine_client/fake_engine.py"
    assert {n: r.kind for n, r in check_ownership.owners(fake)} == {
        "WT-02": "own",
        "WT-01": "exception",
    }
    assert [
        n for n, _ in check_ownership.owners("tests/unittest/grpc/openengine_client/test_client.py")
    ] == ["WT-10"]


def test_no_path_has_two_exclusive_owners():
    literal = [
        rule
        for rules in check_ownership.OWNERSHIP.values()
        for rule in rules
        if rule.kind == "own" and not re.search(r"[*?!]", rule.pattern)
    ]
    for rule in literal:
        exclusive = [
            n for n, r in check_ownership.owners(rule.pattern, rule.repo) if r.kind == "own"
        ]
        assert len(exclusive) == 1, (rule.pattern, exclusive)


# tasks.md lives in the planning directory, outside the repository; set
# POC2_TASKS_MD to its path to run the comparison.
TASKS_MD = Path(os.environ["POC2_TASKS_MD"]) if os.environ.get("POC2_TASKS_MD") else None


def _tasks_md_rows():
    text = TASKS_MD.read_text()
    section = text.split("## 3.", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0] in ("功能",) or set(cells[0]) <= {"-"}:
            continue
        rows.append(cells)
    return rows


def _normalize_deps(text):
    if "暂缓" in text:
        return None
    if "全部 TRT-LLM 侧功能工作项" in text:
        return {"trtllm-features"}
    deps = set(re.findall(r"WT-\d\d", text))
    if "链路" in text:
        deps.add("link")
    if re.search(r"\bR2\b", text):
        deps.add("WT-20-R2")
    return deps


@pytest.mark.skipif(
    TASKS_MD is None or not TASKS_MD.is_file(), reason="POC2_TASKS_MD does not name tasks.md"
)
def test_feature_deps_matches_tasks_md_section_3():
    data = yaml.safe_load(pending_e2e.FEATURE_DEPS.read_text())
    md_rows = _tasks_md_rows()
    assert len(md_rows) == len(data["rows"])
    for (feature, _cases, author, deps), row in zip(md_rows, data["rows"]):
        expected_author = {"R2": "WT-20-R2", "R3": "WT-20-R3", "—": "-"}.get(author, author)
        assert row["author"] == expected_author, (feature, row["id"])
        expected = _normalize_deps(deps)
        if expected is None:
            assert row.get("deferred") is True, row["id"]
        else:
            assert set(row["deps"]) == expected, (feature, row["id"])
        for feature_id in row["features"]:
            if feature_id.startswith("F-"):
                assert feature_id in feature, (feature, row["id"])
        assert ("case-01" in feature) == (row.get("chain") == "case-01"), row["id"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of the exceptions check_ownership.py keeps for one work item each.

An exception that lists its changes names the work item it is for and allows
that item alone, until the item lands. Neither the item, nor the item it fixes
or that item's other fixes, can then make those changes again under it (lines
it only adds, for example). WT-55 has two such exceptions in WT-40's tests.
Run from the repository root::

    python3 -m pytest -p no:cacheprovider scripts/poc3/
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

# The scripts import poc3_common themselves; pytest puts this directory on sys.path.
import check_ownership  # noqa: I001
import pytest

# A stage03 commit before WT-53-fix1 landed; WT-40's two files there are as at WT-55's base.
WT55_BASE = "d636f2e4cba849dbc14235c176cd7922ef6e50cb"
WT55_CONTROL = "tests/unittest/grpc/openengine/test_disagg_control.py"
WT55_SERVICER = "tests/unittest/grpc/openengine/test_disagg_servicer.py"
GENERATION_FIRST_TEST_WAS = "test_schedule_style_is_unimplemented_without_generation_first"
GENERATION_FIRST_TEST = "test_context_only_takes_a_generation_first_schedule_style"
WT55_RENAMES = {WT55_SERVICER: {GENERATION_FIRST_TEST_WAS: GENERATION_FIRST_TEST}}
# WT-55's changes to WT-40's tests (DQ-06, rulings N45 and N54), as (the statement, by its name
# at the head; the removed lines; the added lines).
WT55_CHANGES = {
    WT55_CONTROL: [
        (
            "test_model_info_declares_every_version_4_capability_key",
            ['        "trtllm.disagg.generation_first": False,'],
            ['        "trtllm.disagg.generation_first": enabled,'],
        ),
    ],
    WT55_SERVICER: [
        (
            GENERATION_FIRST_TEST,
            [
                f"def {GENERATION_FIRST_TEST_WAS}():",
                '    request = _request(extra={"request_type": "context_only", "schedule_style": 1})',
            ],
            [
                f"def {GENERATION_FIRST_TEST}():",
                "    llm = _Llm([_prefill_result()])",
                "    request = _request(",
                "        extra={",
                '            "request_type": "context_only",',
                '            "disagg_request_id": str(CALLER_ID),',
                '            "schedule_style": 1,',
                "        }",
                "    )",
            ],
        ),
        (
            GENERATION_FIRST_TEST,
            ["    _, context = _generate(_servicer(), request)"],
            ["    _, context = _generate(_servicer(llm), request)"],
        ),
        (
            GENERATION_FIRST_TEST,
            [
                "    assert context.abort_code == grpc.StatusCode.UNIMPLEMENTED",
                '    assert "schedule_style" in context.abort_details',
            ],
            [
                "    assert context.abort_code is None, context.abort_details",
                '    params = llm.calls[0]["disaggregated_params"]',
                "    assert params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST",
                "    assert params.disagg_request_id == CALLER_ID",
            ],
        ),
        (
            "test_disagg_request_keys_are_unimplemented_without_a_transceiver",
            ['    for extra in ({"disagg_request_id": str(CALLER_ID)}, {"orig_elapsed_s": 0.5}):'],
            [
                "    for extra in (",
                '        {"disagg_request_id": str(CALLER_ID)},',
                '        {"schedule_style": 1},',
                '        {"orig_elapsed_s": 0.5},',
                "    ):",
            ],
        ),
    ],
}
WT55_CHANGE_IDS = [
    (path, index) for path in WT55_CHANGES for index in range(len(WT55_CHANGES[path]))
]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _show(rev: str, path: str) -> str:
    """``path`` as at ``rev`` of this checkout; skips the test when the checkout lacks ``rev``."""
    top = Path(__file__).resolve().parents[2]
    known = subprocess.run(["git", "-C", str(top), "cat-file", "-e", f"{rev}^{{commit}}"])
    if known.returncode != 0:
        pytest.skip(f"{rev} is not in this checkout")
    return _git(top, "show", f"{rev}:{path}")


def _repo_with(
    tmp_path: Path, path: str, base: str, head: str, task: str, landed: str = ""
) -> tuple[Path, str]:
    """A repo holding ``path`` as ``base``, then as ``head`` in a commit of ``task``.

    With ``landed``, the base commit carries that work item's trailer.
    """
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    file = repo / path
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


def _replace(path: str, text: str, change: tuple, by: tuple | None = None) -> str:
    """``text`` with ``change`` made in its statement; with ``by``, its removed lines become those.

    A renamed statement goes by its name at the head here, as in a change.
    """
    within, removed, added = change
    lines = text.split("\n")
    renamed = WT55_RENAMES.get(path, {})
    regions = [renamed.get(n, n) for n in check_ownership._regions(text)]
    for i in range(len(regions)):
        if regions[i] == within and lines[i : i + len(removed)] == removed:
            return "\n".join(
                lines[:i] + list(added if by is None else by) + lines[i + len(removed) :]
            )
    raise AssertionError(f"no place for the change in {within}")


def _wt55_head(
    path: str, text: str, replaced: int | None = None, by: list | None = None, within: str = ""
) -> str:
    """``text`` with WT-55's changes to ``path``.

    Change ``replaced`` adds ``by`` instead, or is made in the statement ``within``. It is made
    last, so that the others still find their statements.
    """
    changes = list(enumerate(WT55_CHANGES[path]))
    if replaced is not None:
        changes = [c for c in changes if c[0] != replaced] + [changes[replaced]]
    for index, change in changes:
        if index == replaced and within:
            change = (within, *change[1:])
        text = _replace(path, text, change, by if index == replaced else None)
    return text


@pytest.mark.parametrize("path", WT55_CHANGES)
def test_the_wt55_exceptions_list_exactly_its_changes(path):
    rule = check_ownership.match_rule("WT-55", path)
    assert rule.kind == "exception" and not rule.lines and rule.item == "WT-55"
    assert list(rule.changes) == [
        check_ownership.Change(tuple(map(re.escape, old)), tuple(map(re.escape, new)), within)
        for within, old, new in WT55_CHANGES[path]
    ]
    assert dict(rule.renames) == WT55_RENAMES.get(path, {})
    assert [(n, r.kind) for n, r in check_ownership.owners(path)] == [
        ("WT-40", "own"),
        ("WT-55", "exception"),
    ]


@pytest.mark.parametrize("path", WT55_CHANGES)
def test_wt55_may_make_its_changes_to_the_files_at_its_base(tmp_path, capsys, path):
    base = _show(WT55_BASE, path)
    repo, sha = _repo_with(tmp_path, path, base, _wt55_head(path, base), "WT-55")
    assert check_ownership.check("WT-55", f"{sha}..HEAD", str(repo)) == 0
    assert f"CHECK BY HAND  EXCEPTION  {path}" in capsys.readouterr().out


# The ways a change can differ from the one allowed: a line fewer, one more, one altered, or
# another line in place of its added ones.
TAMPERS = {
    "line-dropped": lambda added: added[:-1],
    "line-repeated": lambda added: [*added, added[-1]],
    "line-altered": lambda added: [*added[:-1], added[-1] + " "],
    "other-line": lambda added: ["    pass"],
}


@pytest.mark.parametrize("path, index", WT55_CHANGE_IDS)
@pytest.mark.parametrize("tamper", TAMPERS)
def test_wt55_may_not_change_anything_else(tmp_path, capsys, path, index, tamper):
    base = _show(WT55_BASE, path)
    added = TAMPERS[tamper](WT55_CHANGES[path][index][2])
    head = _wt55_head(path, base, index, added)
    repo, sha = _repo_with(tmp_path, path, base, head, "WT-55")
    assert check_ownership.check("WT-55", f"{sha}..HEAD", str(repo)) == 1
    out = capsys.readouterr().out
    assert f"OUT OF BOUNDS  {path}  (the exception allows" in out
    # The reason: a file Python does not parse, a renamed test without its new name, or else a
    # hunk that is not one of the changes.
    try:
        ast.parse(head)
    except SyntaxError:
        assert "is not Python this interpreter parses" in out
        return
    if index == 0 and path in WT55_RENAMES and added[:1] != WT55_CHANGES[path][0][2][:1]:
        assert "does not rename" in out
    else:
        assert "not one of the allowed changes" in out


# A test appended to each file holding the removed lines of a change: the change made there,
# and not where it names, lies in another statement.
OTHER_TESTS = {
    (WT55_CONTROL, 0): [
        "def test_appended(enabled):",
        "    assert {",
        '        "trtllm.disagg.generation_first": False,',
        "    }",
    ],
    (WT55_SERVICER, 1): [
        "def test_appended(request):",
        "    _, context = _generate(_servicer(), request)",
    ],
    (WT55_SERVICER, 3): [
        "def test_appended():",
        '    for extra in ({"disagg_request_id": str(CALLER_ID)}, {"orig_elapsed_s": 0.5}):',
        "        pass",
    ],
}


@pytest.mark.parametrize("path, index", OTHER_TESTS)
def test_wt55_may_not_make_its_changes_in_another_statement(tmp_path, capsys, path, index):
    other = OTHER_TESTS[path, index]
    base = _show(WT55_BASE, path).rstrip("\n") + "\n\n\n" + "\n".join(other) + "\n"
    head = _wt55_head(path, base, index, within="test_appended")
    repo, sha = _repo_with(tmp_path, path, base, head, "WT-55")
    assert check_ownership.check("WT-55", f"{sha}..HEAD", str(repo)) == 1
    assert "not one of the allowed changes" in capsys.readouterr().out


def test_wt55_needs_the_rename_to_change_the_renamed_test(tmp_path):
    path = WT55_SERVICER
    rule = check_ownership.match_rule("WT-55", path)
    base = _show(WT55_BASE, path)
    repo, sha = _repo_with(tmp_path, path, base, _wt55_head(path, base), "WT-55")
    assert check_ownership.disallowed_lines(rule, str(repo), sha, "HEAD", path) == []
    without = check_ownership._exception(path, rule.note, changes=rule.changes, item=rule.item)
    found = check_ownership.disallowed_lines(without, str(repo), sha, "HEAD", path)
    # The three hunks in the renamed test.
    assert len(found) == 3 and all("not one of the allowed changes" in line for line in found)


@pytest.mark.parametrize("name", ["kept", "another"])
def test_wt55_may_not_give_the_renamed_test_another_name(tmp_path, capsys, name):
    path = WT55_SERVICER
    _, removed, added = WT55_CHANGES[path][0]
    if name == "kept":  # the changes in the test made, but its old name kept
        new = [removed[0], *added[1:]]
    else:
        new = [added[0].replace("def test_", "def test_x_"), *added[1:]]
    base = _show(WT55_BASE, path)
    repo, sha = _repo_with(tmp_path, path, base, _wt55_head(path, base, 0, new), "WT-55")
    assert check_ownership.check("WT-55", f"{sha}..HEAD", str(repo)) == 1
    expected = f"does not rename {GENERATION_FIRST_TEST_WAS} to {GENERATION_FIRST_TEST} at the head"
    assert expected in capsys.readouterr().out


# --- an exception for one work item -------------------------------------------------------------

# Every exception that lists the changes it allows, with the work item it is for.
KEYED = [
    (name, rule)
    for name, rules in check_ownership.OWNERSHIP.items()
    for rule in rules
    if rule.kind == "exception" and rule.changes
]


def test_every_exception_listing_its_changes_is_for_one_work_item():
    found = {(rule.pattern, rule.item) for _, rule in KEYED}
    oe = check_ownership.OE_SERVER_TESTS
    poc = check_ownership.POC_TESTS
    client = check_ownership.CLIENT_TESTS
    assert found == {
        (f"{oe}/test_control.py", "WT-40"),
        (f"{oe}/test_servicer.py", "WT-40"),
        (f"{oe}/test_multimodal.py", "WT-40"),
        (f"{poc}/wt50/wt50_chain.py", "WT-52"),
        (f"{poc}/wt50/test_poc3_wt50_internal_auth.py", "WT-52"),
        (f"{poc}/wt11/test_poc2_wt11_no_silent_drop.py", "WT-52"),
        ("tensorrt_llm/serve/openengine_poc/engine_client.py", "WT-53-fix1"),
        ("tensorrt_llm/serve/openengine_poc/engine_client_impl.py", "WT-53-fix1"),
        (f"{client}/test_openengine_poc_serve.py", "WT-53-fix1"),
        (f"{poc}/wt01/test_poc2_wt01_contracts.py", "WT-53-fix1"),
        (f"{poc}/wt35/test_poc3_wt35_contracts.py", "WT-53-fix1"),
        (f"{poc}/wt15/test_poc2_wt15_metrics.py", "WT-53-fix1"),
        (f"{oe}/test_disagg_control.py", "WT-55"),
        (f"{oe}/test_disagg_servicer.py", "WT-55"),
        (f"{poc}/wt55/test_poc3_wt55_gen_first.py", "WT-51-fix2"),
        (f"{check_ownership.CHAIN}/tests/test_expectations.py", "WT-55-fix1"),
        (f"{poc}/wt70/test_poc3_wt70_arms.py", "WT-71"),
        (f"{check_ownership.CHAIN}/tests/test_expectations.py", "WT-62"),
        (f"{check_ownership.CHAIN}/tests/test_expectations.py", "WT-57-fix1"),
    }
    assert check_ownership.NAMED_FIXES == [
        "WT-30-fix15",
        "WT-51-fix2",
        "WT-53-fix1",
        "WT-55-fix1",
        "WT-57-fix1",
    ]


def _others(item: str) -> list[str]:
    """The work item ``item`` is (or fixes) and that item's fixes 1 to 3, ``item`` itself left out."""
    base = check_ownership.parse_task(item)[0]
    return [name for name in (base, *(f"{base}-fix{n}" for n in (1, 2, 3))) if name != item]


@pytest.mark.parametrize("name, rule", KEYED, ids=[f"{r.item}:{r.pattern}" for _, r in KEYED])
def test_an_exception_for_a_work_item_allows_no_other(name, rule):
    assert check_ownership.match_rule(rule.item, rule.pattern, rule.repo) is rule
    for other in _others(rule.item):
        assert check_ownership.match_rule(other, rule.pattern, rule.repo) is None, other
    assert (rule.item, rule) in check_ownership.owners(rule.pattern, rule.repo)


def test_an_exception_is_for_a_work_item_by_its_name():
    with pytest.raises(ValueError, match="not a PoC-S3 work item"):
        check_ownership._exception("x.py", "test", item="WT-5")


A_TO_2 = check_ownership._change("f", ("    a = 1",), ("    a = 2",))


def test_an_exception_listing_changes_names_its_work_item():
    table = {"WT-52": [check_ownership._exception("x.py", "test", changes=(A_TO_2,))]}
    with pytest.raises(ValueError, match="an exception listing changes names no item"):
        check_ownership._named_fixes(table)


@pytest.mark.parametrize("item", ["WT-53", "WT-52-fix1"])
def test_an_exception_is_listed_under_the_item_it_is_for_or_fixes(item):
    table = {"WT-52": [check_ownership._exception("x.py", "test", item=item)]}
    if item == "WT-53":
        with pytest.raises(ValueError, match="an exception for WT-53 is under WT-52"):
            check_ownership._named_fixes(table)
    else:
        assert check_ownership._named_fixes(table) == ["WT-52-fix1"]


# Lines an exception lets its work item add, and nothing else: (the item; the file; the commit
# holding the file once the item has landed; the line the item adds them after, when that commit
# does not hold them yet; the line they are added again after, if not right after themselves;
# the lines).
ADDED_ONLY = [
    (
        "WT-40",
        "tests/unittest/grpc/openengine/test_control.py",
        WT55_BASE,
        None,
        None,
        ['    other_keys.discard("trtllm.stopping.min_tokens")'],
    ),
    (
        "WT-40",
        "tests/unittest/grpc/openengine/test_control.py",
        WT55_BASE,
        None,
        None,
        [
            "from tensorrt_llm.grpc.openengine_conventions import (  # noqa: E402",
            "    COUNTER_GENERATE_ACCEPTED_BUCKETS,",
            "    VERSION_4_ENGINE_COUNTERS,",
            ")",
        ],
    ),
    (
        "WT-52",
        "tests/unittest/serve/openengine_poc/wt11/test_poc2_wt11_no_silent_drop.py",
        WT55_BASE,
        None,
        # Not right after them: the copy would share its last lines with the probe's next branch,
        # and git would show the added lines rotated.
        "        caps = facade.client.model_info()",
        [
            '        if name == "min_tokens" and (',
            "            conv.conventions_version(facade.client.server_info().extra)"
            " == conv.CONVENTIONS_VERSION",
            "            and not caps.supports(conv.CAP_STOPPING_MIN_TOKENS)",
            "        ):",
            "            # DQ-48: the facade refuses min_tokens to an engine on the current",
            "            # conventions that does not declare it.",
            '            with pytest.raises(ValueError, match="min_tokens"):',
            "                await facade.generate_async(GRPC_PROMPT, _sampling(probe.params),"
            " **options)",
            "            assert engine.requests == []",
            "            return",
        ],
    ),
    (
        "WT-53-fix1",
        "tensorrt_llm/serve/openengine_poc/engine_client.py",
        WT55_BASE,
        "    kv_connector: KvConnectorInfo | None = None",
        None,
        ["    kv_block_size: int | None = None", "    total_kv_blocks: int | None = None"],
    ),
]


def _after(lines: list[str], anchor: list[str], added: list[str]) -> list[str]:
    """``lines`` with ``added`` right after the one place they hold ``anchor``."""
    at = [i for i in range(len(lines)) if lines[i : i + len(anchor)] == anchor]
    assert len(at) == 1, (anchor, at)
    end = at[0] + len(anchor)
    return lines[:end] + added + lines[end:]


@pytest.mark.parametrize(
    "item, path, rev, anchor, again, added",
    ADDED_ONLY,
    ids=[f"{c[0]}:{c[1].rsplit('/', 1)[1]}:{len(c[-1])}" for c in ADDED_ONLY],
)
def test_lines_an_exception_adds_cannot_be_added_again_once_its_item_has_landed(
    tmp_path, capsys, item, path, rev, anchor, again, added
):
    lines = _show(rev, path).split("\n")
    if anchor is not None:  # the item's change, as it stands once the item has landed
        lines = _after(lines, [anchor], added)
    base = "\n".join(lines)
    head = "\n".join(_after(lines, [again] if again else added, added))  # the same lines again
    # Before the item lands, the change is one its exception allows.
    repo, sha = _repo_with(tmp_path / "before", path, base, head, item)
    assert check_ownership.check(item, f"{sha}..HEAD", str(repo)) == 0, capsys.readouterr().out
    capsys.readouterr()
    for task in [item, *_others(item)]:
        repo, sha = _repo_with(tmp_path / task, path, base, head, task, landed=item)
        found = check_ownership.check(task, f"{sha}..HEAD", str(repo))
        out = capsys.readouterr().out
        if task == item:
            expected = f"OUT OF BOUNDS  {path}  (the exception for {item} is spent"
        else:
            expected = f"OUT OF BOUNDS  {path}  (owned by "
        assert found == 1 and expected in out, (task, out)


@pytest.mark.parametrize("path", WT55_CHANGES)
def test_an_exception_allows_nothing_once_its_item_has_landed(tmp_path, capsys, path):
    base = _show(WT55_BASE, path)
    head = _wt55_head(path, base)
    repo, sha = _repo_with(tmp_path, path, base, head, "WT-55", landed="WT-55")
    assert check_ownership.check("WT-55", f"{sha}..HEAD", str(repo)) == 1
    out = capsys.readouterr().out
    assert f"OUT OF BOUNDS  {path}  (the exception for WT-55 is spent" in out
    # Another item landed at the base does not spend it.
    repo, sha = _repo_with(tmp_path / "other", path, base, head, "WT-55", landed="WT-55-fix1")
    assert check_ownership.check("WT-55", f"{sha}..HEAD", str(repo)) == 0


def test_an_exception_is_spent_by_its_item_at_the_base_not_at_the_merge_base(tmp_path, capsys):
    path = WT55_CONTROL
    base = _show(WT55_BASE, path)
    head = _wt55_head(path, base)
    repo, fork_point = _repo_with(tmp_path, path, base, head, "WT-55")
    # WT-55 lands on a branch from the fork point; the change is then made again on a copy of
    # the fork point that was never rebased onto it.
    _git(repo, "checkout", "-q", "-b", "landed", fork_point)
    _git(repo, "commit", "-q", "--allow-empty", "-m", "landed\n\nPoC3-Task: WT-55")
    assert check_ownership.check("WT-55", "landed..main", str(repo)) == 1
    assert f"OUT OF BOUNDS  {path}  (the exception for WT-55 is spent" in capsys.readouterr().out


# Work items whose trailers at the base leave WT-53-fix1's exceptions in force: the item it fixes
# and that item's other fixes.
@pytest.mark.parametrize("landed", _others("WT-53-fix1"))
def test_an_exception_for_a_fix_is_not_spent_by_the_item_it_fixes_or_that_items_other_fixes(
    tmp_path, capsys, landed
):
    # WT-53-fix1's lines in engine_client.py, made on a base where WT-53, or another fix of it,
    # has landed and WT-53-fix1 has not.
    item, path, rev, anchor, _, added = next(c for c in ADDED_ONLY if c[0] == "WT-53-fix1")
    lines = _show(rev, path).split("\n")
    base, head = "\n".join(lines), "\n".join(_after(lines, [anchor], added))
    repo, sha = _repo_with(tmp_path, path, base, head, item, landed=landed)
    assert check_ownership.check(item, f"{sha}..HEAD", str(repo)) == 0, capsys.readouterr().out
    assert f"CHECK BY HAND  EXCEPTION  {path}" in capsys.readouterr().out

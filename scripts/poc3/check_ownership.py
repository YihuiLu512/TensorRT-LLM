#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Check that a PoC-S3 work item changes only the paths it owns.

Every path changed in ``<base>...<head>`` must match one of the work item's
rules in :data:`OWNERSHIP`, every commit in ``<base>..<head>`` must carry
exactly one ``PoC3-Task`` line, reading exactly ``PoC3-Task: <work item>`` (read
as ``poc3_common`` describes), and the range must hold at least one commit. A
fix (``WT-xx-fixN``) owns what WT-xx owns, plus the per-item files named after
the fix itself; WT-xx and its other fixes do not own those. Likewise an
``exception`` that names the work item it is for allows that item alone: an
exception for WT-xx-fixN allows neither WT-xx nor WT-xx's other fixes, and one
for WT-xx allows none of WT-xx's fixes. It allows nothing once that item has
landed, that is once a commit with its trailer is in the history of
``<base>``. The gate files
(``scripts/poc3/`` except the invariant registry and ``wt54_*``) can only be
changed by WT-30 and its fixes, whatever the table says.

A rule is ``own`` (no other item owns the path; another item may change it only
under a ``shared`` or ``exception`` rule of its own), ``shared`` (this item may
change only the part the note names; the rest of the file belongs to its owner,
which may be another item or code outside PoC-S3) or ``exception`` (a one-off
allowance the note describes). An ``exception`` may also limit what it allows,
in one of four forms (patterns are regular expressions in which ``\d`` and
``\w`` match ASCII characters only):

- ``lines``: the range may only replace lines of that file in place, each
  removed and each added line matching one of the patterns in full;
- ``changes``: each hunk of the range's diff (``-U0``) must be one of the
  listed changes, each listed change used at most once: in the top-level
  statement the change names (or, for a change that names the statements it
  touches instead, touching exactly those), with the same number of removed
  and of added lines, each matching its counterpart in the change in full, in
  order;
- ``redact``: each hunk of the range's diff (``-U0``) replaces lines in place,
  as many added as removed, each added line differing from its removed
  counterpart only where it has one of the listed placeholders, each standing
  for a whole path start its pattern matches (see :func:`_redacts`); with a
  ``header``, one hunk may instead add only that line, as the second line of
  the file;
- ``blob``: the range creates the file, absent at the merge base, as a regular
  file that is not executable with exactly the content of that git blob.

A top-level statement runs from its first line (its first decorator) up to the
line before the next top-level statement of any kind. A ``def`` or ``class`` is
named by its name, an assignment (plain or annotated) whose first target is a
bare name by that name, and any import by ``import``; other statements have no name.
A hunk lies in a named statement when its removed lines lie in it in the file at
the merge base, and its added lines lie in it in the file at the head; a hunk
that only adds lines must also follow or precede a line of that statement at the
merge base, and the file must be Python the interpreter running the check parses
at both. A hunk touches the statements its removed lines lie in at the merge
base, those its added lines lie in at the head and, when it removes nothing,
the one the line it follows lies in at the merge base; lines in no named
statement (a module docstring, say) count as one more. A ``changes``
exception may also list the statements it renames, each from one name to
another: the file must have a statement by the first name and none by the
second at the merge base, and the other way round at the head; the changes
are matched as if the statement already went by the second at the merge base.
With any form nothing else about the file may change (its mode or final
newline, for example);
otherwise the path is out of bounds. Paths changed under ``shared`` and
``exception`` rules are listed for a reviewer to check by hand.
A pattern starting with ``!`` excludes paths from the rules before it.

The orchestrator's main files belong to no work item. Some code inside files
that work items own or share belongs to no work item either: the two
conversion functions of ``openai_protocol.py`` and the two ``commands/serve.py``
subcommands keep their source text, and the existing lines of the HTTP
``DisaggregatedParams`` class and of ``disagg_auth.py`` stay as they are (only
registered lines may be added). ``check_invariants.py`` guards all of these and
runs separately.

Usage::

    python3 scripts/poc3/check_ownership.py --task WT-51 fork/feat/engine_integration-stage03..HEAD
    cd ../deepseek-v4.1-flash
    python3 <trtllm>/scripts/poc3/check_ownership.py --task WT-41 origin/feat/engine_integration-stage03..HEAD

Both ends of the range are required. Exit status: 0 when every path and commit
is in bounds, 1 otherwise (a bad ``--task`` or range, or a git error, included);
2 when the arguments cannot be parsed.
The scripts need Python 3.9 or newer.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from poc3_common import (
    TRAILER,
    UNREADABLE,
    git,
    git_error,
    landed_tasks,
    parse_task,
    run_git,
    trailer_log,
)

TRTLLM = "trtllm"
DS = "ds"


class Change(NamedTuple):
    """A hunk an exception allows."""

    removed: tuple[str, ...]  # patterns of its removed lines, in order
    added: tuple[str, ...]  # patterns of its added lines, in order
    # The top-level statement it lies in (see _hunk_regions); one the exception renames goes by
    # its name at the head. Empty for a change that names ``around`` instead.
    within: str
    # Instead of ``within``: every top-level statement the hunk touches (see _hunk_touches), None
    # standing for lines in no named statement.
    around: tuple[str | None, ...] = ()


@dataclass(frozen=True)
class Rule:
    pattern: str
    kind: str = "own"
    note: str = ""
    repo: str = TRTLLM
    # For an exception: regular expressions, one of which every changed line must match in full,
    # for lines replaced in place ...
    lines: tuple[str, ...] = ()
    # ... or the only hunks allowed, each used at most once.
    changes: tuple[Change, ...] = ()
    # With changes: the top-level statements the hunks rename, as (name at the merge base, name at
    # the head). A change names a renamed statement by its name at the head.
    renames: tuple[tuple[str, str], ...] = ()
    # For an exception: the placeholders that may stand for the start of a path in lines replaced
    # in place, each with the pattern of the starts it stands for, and the one line that may be
    # added as the file's second line.
    redact: tuple[tuple[str, str], ...] = ()
    header: str = ""
    # For an exception: the one work item it allows (a fix by its own name), until that item has
    # landed; empty for the work item whose entry lists it and all of that item's fixes.
    item: str = ""
    # For an exception: the git blob id of the file, which the range must create with exactly
    # this content (a regular, non-executable file absent at the merge base).
    blob: str = ""


def _own(*patterns: str, repo: str = TRTLLM) -> list[Rule]:
    return [Rule(p, "own", repo=repo) for p in patterns]


def _shared(pattern: str, note: str, repo: str = TRTLLM) -> Rule:
    return Rule(pattern, "shared", note, repo)


def _exception(
    pattern: str,
    note: str,
    repo: str = TRTLLM,
    lines: tuple[str, ...] = (),
    changes: tuple[Change, ...] = (),
    renames: tuple[tuple[str, str], ...] = (),
    item: str = "",
    redact: tuple[tuple[str, str], ...] = (),
    header: str = "",
    blob: str = "",
) -> Rule:
    if item:
        parse_task(item)  # raises ValueError for a name that is not a work item
    if sum(map(bool, (lines, changes, redact, blob))) > 1:
        raise ValueError(
            f"{pattern}: an exception lists one of lines, changes and redact, or a blob alone"
        )
    if blob and not re.fullmatch(r"[0-9a-f]{40}", blob):
        raise ValueError(f"{pattern}: an exception's blob is a full git blob id, not {blob!r}")
    for change in changes:
        if bool(change.within) == bool(change.around):
            raise ValueError(f"{pattern}: a change names either the statement it lies in or around")
    if header and not redact:
        raise ValueError(f"{pattern}: an exception adds a header line only with redact")
    for placeholder, start in redact:
        if not placeholder or not start:
            raise ValueError(
                f"{pattern}: an exception's placeholders and their patterns are not empty"
            )
        re.compile(start, re.ASCII)  # raises re.error for a pattern that is not one
    if renames and not changes:
        raise ValueError(f"{pattern}: an exception renames statements only with its changes")
    olds, news = {old for old, _ in renames}, {new for _, new in renames}
    if len(olds) < len(renames) or len(news) < len(renames):
        raise ValueError(
            f"{pattern}: an exception renames each statement once, each to its own name"
        )
    named = {change.within for change in changes}
    for old, new in renames:
        if new not in named or old in named:
            raise ValueError(
                f"{pattern}: renaming {old} to {new} needs a change in {new}, none in {old}"
            )
    return Rule(
        pattern, "exception", note, repo, lines, changes, renames, redact, header, item, blob
    )


def _change(within: str, removed: tuple[str, ...] = (), added: tuple[str, ...] = ()) -> Change:
    """A hunk in ``within`` that removes exactly the lines ``removed`` and adds exactly ``added``."""
    return Change(
        tuple(re.escape(line) for line in removed),
        tuple(re.escape(line) for line in added),
        within,
    )


def _change_around(
    around: tuple[str | None, ...], removed: tuple[str, ...] = (), added: tuple[str, ...] = ()
) -> Change:
    """A hunk touching exactly the statements ``around`` that removes ``removed`` and adds ``added``.

    For a hunk that adds a top-level statement, or a comment or docstring
    outside one, which no single statement holds at both ends.
    """
    return Change(
        tuple(re.escape(line) for line in removed),
        tuple(re.escape(line) for line in added),
        "",
        tuple(around),
    )


POC = "tensorrt_llm/serve/openengine_poc"
CHAIN = "tests/integration/defs/openengine_chain"
APPS = "tests/unittest/llmapi/apps"
POC_TESTS = "tests/unittest/serve/openengine_poc"
CLIENT_TESTS = "tests/unittest/grpc/openengine_client"
OE_SERVER = "tensorrt_llm/grpc/openengine"
OE_SERVER_TESTS = "tests/unittest/grpc/openengine"
PERF = "tests/integration/defs/perf"
DS_OE = "python/dsv4_serve/openengine"
_SPECULATION_TEST = (
    "test_model_info_reports_only_no_effect_min_p_and_penalties_under_one_model_speculation"
)
_TAMPERED_HANDOFF_TEST = "test_gen_worker_frontend_rejects_a_tampered_handoff_signature"
_SIGNED_HANDOFF_TEST = "test_a_signed_handoff_is_not_refused_for_a_missing_transceiver"
_UNSUPPORTED_HANDLER_TEST = "test_an_unsupported_error_from_a_handler_answers_501"
_NO_EVENTS_HTTP_TEST_WAS = "test_kv_cache_events_the_frontend_cannot_read_answer_501"
_NO_EVENTS_HTTP_TEST = "test_kv_cache_events_the_frontend_cannot_read_answer_no_events"
_NO_EVENTS_SUBSCRIPTION_TEST_WAS = "test_kv_cache_events_unsupported_raise_a_new_error_each_call"
_NO_EVENTS_SUBSCRIPTION_TEST = "test_kv_cache_events_unsupported_answer_no_events_each_call"
_GENERATION_FIRST_TEST_WAS = "test_schedule_style_is_unimplemented_without_generation_first"
_GENERATION_FIRST_TEST = "test_context_only_takes_a_generation_first_schedule_style"
_GENERATION_LEG_TEST = "test_the_generation_leg_never_sends_schedule_style_in_extra"
_CANCELS_CALLS_TEST = "test_shipped_tables_name_each_test_that_cancels_calls"
_ARM_TABLE_TEST = "test_disagg_arm_table"
# The registry of lines check_invariants.py lets CS-20 add.
INVARIANT_ADDITIONS = "scripts/poc3/invariant_additions.yaml"
# Gate files only WT-30 and its fixes may change (see the module docstring).
GATE_FILES = ("scripts/poc3/**", f"!{INVARIANT_ADDITIONS}", "!scripts/poc3/wt54_*")
EXISTING_TESTS = "add tests; existing assertions unchanged (tasks.md §0.3)"
WT60_FIX1 = "WT-60-fix1"
# The git blob id of the harness's self-tests of its waits that WT-60-fix1 adds.
WT60_FIX1_TEST_WAITS = "0101d96484c318e1b6b3b7f0ab5718acc3c3125a"


def _per_item(item: str) -> list[Rule]:
    """The files a work item (or a fix) owns under its exact name."""
    return _own(
        f"{CHAIN}/expectations/{item}.py",
        f"{CHAIN}/verified/{item}.yaml",
        f"{CHAIN}/smoke/{item}.txt",
        f"{CHAIN}/new_cases/{item}.txt",
        f"{CHAIN}/migrated/{item}.txt",
        # Machine-readable records of probes and integration rounds (case lists, summaries).
        f"{CHAIN}/records/{item}/**",
    )


# U41 (N127, N128): records in which WT-30-fix15 may only put a placeholder in place of the start
# of each path on a local disk under /home: <scratch-1> for a disk whose name ends in _1,
# <scratch> for any other. In the driver scripts the placeholders are ${SCRATCH1:?} and
# ${SCRATCH:?}, which bash does not read as redirections, and the scripts may also get a fixed
# second line saying what they stand for.
REDACTED_RECORDS = tuple(
    f"{CHAIN}/verified/{name}.yaml"
    for name in (
        "WT-03",
        "WT-03-fix1",
        "WT-05-fix1",
        "WT-09-fix1",
        "WT-10",
        "WT-10-fix1",
        "WT-11",
        "WT-12",
        "WT-13",
        "WT-15",
        "WT-16",
        "WT-20-R2",
        "WT-20-R3",
    )
) + tuple(f"{CHAIN}/records/{name}/README.md" for name in ("WT-31", "WT-33", "WT-34"))
REDACTED_SCRIPTS = tuple(
    f"{CHAIN}/records/WT-31/scripts/{name}"
    for name in ("collect.sbatch", "collect.sh", "r10_shard.sbatch", "submit.sh")
)
_DISK_1 = r"/home/\w[\w.-]*_1"
_DISK = r"/home/\w[\w.-]*(?<!_1)"
REDACTED_PLACEHOLDERS = (("<scratch-1>", _DISK_1), ("<scratch>", _DISK))
REDACTED_SCRIPT_PLACEHOLDERS = (("${SCRATCH1:?}", _DISK_1), ("${SCRATCH:?}", _DISK))
REDACTED_HEADER = (
    "# Historical copy kept as a record, not meant to be run again: SCRATCH and SCRATCH1 stand for"
    " the local disks it used to name."
)
_REDACTED = "U41 only: the start of paths under /home replaced by <scratch> or <scratch-1>"


# Built from each card's change-scope row (改动范围) and the shared-file table of tasks.md §0.3.
# Where a card leaves a file name to its work item, this table fixes one (direct_replay*.py,
# cpu_disagg_chain*.py, check_case_lists.py, handoff*.py, wt54_*, records/<item>/, and the ds
# scripts dev/openengine_disagg* and dev/openengine_stress*).
OWNERSHIP: dict[str, list[Rule]] = {
    "WT-30": _own(*GATE_FILES)
    + [_exception(INVARIANT_ADDITIONS, "only creating it with an empty list of additions")]
    + [
        _exception(path, _REDACTED, item="WT-30-fix15", redact=REDACTED_PLACEHOLDERS)
        for path in REDACTED_RECORDS
    ]
    + [
        _exception(
            path,
            "U41 only: the start of paths under /home replaced by ${SCRATCH:?} or ${SCRATCH1:?},"
            " and the line that says so",
            item="WT-30-fix15",
            redact=REDACTED_SCRIPT_PLACEHOLDERS,
            header=REDACTED_HEADER,
        )
        for path in REDACTED_SCRIPTS
    ],
    "WT-31": [],
    # WT-32 creates the NT-041 file; §3.3 makes NT-041 WT-40's to finish and re-run.
    "WT-32": _own(f"{CHAIN}/direct_replay*.py", f"{OE_SERVER_TESTS}/_test_direct_handoff.py"),
    "WT-33": [],
    "WT-34": [],
    "WT-35": _own(
        "tensorrt_llm/grpc/openengine_conventions.py",
        f"{POC}/contracts.py",
        f"{POC}/engine_client.py",
        f"{POC_TESTS}/fakes.py",
        "tensorrt_llm/serve/disagg_auth.py",
        INVARIANT_ADDITIONS,
    )
    + [
        _shared(f"{POC}/__init__.py", "lazy exports of the new contract types only"),
        _shared(
            "tensorrt_llm/serve/openai_protocol.py",
            "CS-20: only the registered additions to the HTTP DisaggregatedParams class body"
            " (check_invariants.py)",
        ),
        _exception(
            "tests/unittest/disaggregated/test_disagg_internal_auth.py",
            "CS-20's flip only: the new field name in test_protected_fields_come_from_protocol_metadata",
        ),
        # Contract v10 and conventions version 4 break the assertions pinning the old versions.
        _exception(
            f"{POC_TESTS}/wt01/test_poc2_wt01_contracts.py",
            "only the expected contract version in test_contract_version",
            lines=(r"    assert contracts\.POC2_CONTRACT_VERSION == \d+",),
        ),
        _exception(
            f"{POC_TESTS}/wt01/test_poc2_wt01_conventions.py",
            "only the expected conventions version in test_version_and_keys",
            lines=(r"    assert conv\.CONVENTIONS_VERSION == \d+",),
        ),
        _exception(
            f"{OE_SERVER_TESTS}/test_control.py",
            "only the expected conventions version in"
            " test_server_info_reports_the_conventions_version",
            lines=(r"    assert isinstance\(version, float\) and version == \d+",),
        ),
    ],
    "WT-36": _own(
        f"{CLIENT_TESTS}/fake_openengine.py",
        f"{CLIENT_TESTS}/test_fake_openengine_*.py",
        f"{POC_TESTS}/cpu_disagg_chain*.py",
    )
    + [_shared(f"{CLIENT_TESTS}/test_fake_openengine.py", EXISTING_TESTS)],
    "WT-37": _own(f"{CHAIN}/split.py")
    + [
        _shared(f"{CHAIN}/tests/test_split.py", EXISTING_TESTS),
        _shared(f"{CHAIN}/tests/test_launch.py", EXISTING_TESTS),
        _shared(
            f"{CHAIN}/tests/test_remote_server.py",
            EXISTING_TESTS + "; CT-14 ④ flips the role refusal (:230-232)",
        ),
        _shared(f"{CHAIN}/remote.py", "CT-03: allowing role (CT-14 ④ asserts it)"),
        _shared(
            f"{CHAIN}/launch.py",
            "CT-04: wait_engine_ready and launch_serve_command; _engine_prefix is WT-56's and"
            " ServeHandle's role fields are WT-38's",
        ),
        _shared(
            f"{CHAIN}/conventions.py",
            "TRTLLM_TEST_OPENENGINE_TOPOLOGY only; the ds chain values are WT-56's",
        ),
        _shared(f"{CHAIN}/runner.py", "telling the PoC-S2 and PoC-S3 smoke unions apart (§0.5)"),
        _shared("tests/integration/defs/disaggregated/disagg_test_utils.py", "CT-01: _run_worker"),
        _shared(
            "tests/integration/defs/accuracy/test_disaggregated_serving.py",
            "CT-02: launch_disaggregated_llm",
        ),
        _shared(f"{APPS}/openai_server.py", "CT-06: RemoteOpenAIServer's environment branch"),
    ],
    "WT-38": _own(
        f"{CHAIN}/proof.py",
        f"{CHAIN}/pytest_plugin.py",
        f"{CHAIN}/runner.py",
        f"{CHAIN}/expectations/__init__.py",
        f"{CHAIN}/check_case_lists.py",
        f"{CHAIN}/new_cases/README.md",
        f"{CHAIN}/migrated/README.md",
        f"{CHAIN}/tests/test_proof_disagg.py",
        f"{CHAIN}/tests/test_check_case_lists.py",
    )
    + [
        _shared(f"{CHAIN}/launch.py", "CT-05: ServeHandle's role fields"),
        *(
            _shared(f"{CHAIN}/tests/{name}", EXISTING_TESTS)
            for name in (
                "test_proof.py",
                "test_expectations.py",
                "test_plugin.py",
                "test_runner.py",
            )
        ),
    ],
    "WT-40": _own(
        f"{OE_SERVER}/disagg.py",
        f"{OE_SERVER}/servicer.py",
        f"{OE_SERVER}/control.py",
        f"{OE_SERVER}/capabilities.py",
        f"{OE_SERVER_TESTS}/test_disagg_servicer.py",
        f"{OE_SERVER_TESTS}/test_disagg_control.py",
        f"{APPS}/_test_openengine_disagg_direct.py",
    )
    + [
        _shared(f"{OE_SERVER_TESTS}/_test_direct_handoff.py", "NT-041 (§3.3)"),
        # CO-03 refuses disagg requests without a KV cache transceiver and CO-06 adds the
        # version 4 capability keys and counters, which break these expectations.
        _exception(
            f"{OE_SERVER_TESTS}/test_control.py",
            "only the version 4 capability keys and counters in the expectations,"
            ' kv_transfer_backend="" for two GetModelInfo cases, and min_tokens following'
            " one-model speculation",
            item="WT-40",
            changes=(
                _change(
                    "import",
                    added=(
                        "from tensorrt_llm.grpc.openengine_conventions import (  # noqa: E402",
                        "    COUNTER_GENERATE_ACCEPTED_BUCKETS,",
                        "    VERSION_4_ENGINE_COUNTERS,",
                        ")",
                    ),
                ),
                _change(
                    "EXPECTED_EXTRA_CAPABILITIES",
                    added=(
                        '    "trtllm.stopping.min_tokens": True,',
                        "    # No KV cache transceiver is configured.",
                        '    "trtllm.request.disagg_request_id": False,',
                        '    "trtllm.disagg.generation_first": False,',
                        '    "trtllm.disagg.orig_elapsed_s": False,',
                    ),
                ),
                _change(
                    "test_model_info_without_a_tokenizer_reports_capabilities_but_no_fingerprint",
                    ("    info = await _servicer(_tokenizer=None).GetModelInfo(",),
                    (
                        '    info = await _servicer(kv_transfer_backend="", _tokenizer=None).GetModelInfo(',
                    ),
                ),
                _change(
                    _SPECULATION_TEST,
                    ("    info = await _servicer(speculative_config=make_config()).GetModelInfo(",),
                    (
                        '    info = await _servicer(kv_transfer_backend="",'
                        " speculative_config=make_config()).GetModelInfo(",
                    ),
                ),
                _change(
                    _SPECULATION_TEST,
                    added=(
                        "    # The one-model speculative sampler rejects min_tokens as well.",
                        '    assert info.extra["trtllm.stopping.min_tokens"] == supported',
                    ),
                ),
                _change(
                    _SPECULATION_TEST,
                    added=('    other_keys.discard("trtllm.stopping.min_tokens")',),
                ),
                _change(
                    "test_generate_counters_start_at_zero_and_only_increase",
                    ("    assert counters.attributes() == {RECEIVED: 0, ACCEPTED: 0}",),
                    (
                        "    zero = {key: 0 for key in VERSION_4_ENGINE_COUNTERS}",
                        "    assert counters.attributes() == {RECEIVED: 0, ACCEPTED: 0, **zero}",
                    ),
                ),
                _change(
                    "test_generate_counters_start_at_zero_and_only_increase",
                    ("    assert counters.attributes() == {RECEIVED: 2, ACCEPTED: 1}",),
                    (
                        '    aggregated = COUNTER_GENERATE_ACCEPTED_BUCKETS["aggregated"]',
                        "    assert counters.attributes() =="
                        " {RECEIVED: 2, ACCEPTED: 1, **zero, aggregated: 1}",
                    ),
                ),
            ),
        ),
        _exception(
            f"{OE_SERVER_TESTS}/test_servicer.py",
            'only kv_transfer_backend="NIXL" for the two servicers that receive disagg requests',
            item="WT-40",
            changes=(
                _change(
                    "test_generate_preserves_child_identity_and_authenticates_parent_affinity",
                    (
                        '        llm, model="test-model", subagent_affinity_auth_key="shared-secret"',
                    ),
                    (
                        "        llm,",
                        '        model="test-model",',
                        '        subagent_affinity_auth_key="shared-secret",',
                        '        kv_transfer_backend="NIXL",',
                    ),
                ),
                _change(
                    "test_generate_context_only_ends_at_prefill_ready",
                    (
                        "    servicer = OpenEngineInferenceServicer(_FakeLlm(results),"
                        ' model="test-model", metrics=metrics)',
                    ),
                    (
                        "    servicer = OpenEngineInferenceServicer(",
                        '        _FakeLlm(results), model="test-model", metrics=metrics,'
                        ' kv_transfer_backend="NIXL"',
                        "    )",
                    ),
                ),
            ),
        ),
        _exception(
            f"{OE_SERVER_TESTS}/test_multimodal.py",
            "only a kv_transfer_backend for the servicers of"
            " test_audio_text_prompt_survives_preparation_without_placeholders",
            item="WT-40",
            changes=(
                _change(
                    "test_audio_text_prompt_survives_preparation_without_placeholders",
                    (
                        '    inference = OpenEngineInferenceServicer(llm, "audio-model")',
                        '    control = OpenEngineControlServicer(llm, "audio-model", inference)',
                    ),
                    (
                        '    backend = "NIXL" if context_only else ""',
                        '    inference = OpenEngineInferenceServicer(llm, "audio-model",'
                        " kv_transfer_backend=backend)",
                        '    control = OpenEngineControlServicer(llm, "audio-model", inference,'
                        " kv_transfer_backend=backend)",
                    ),
                ),
            ),
        ),
    ],
    "WT-41": _own(
        f"{DS_OE}/**",
        "tests/openengine/**",
        "!tests/openengine/test_disagg_roles.py",
        repo=DS,
    )
    + [
        _shared("python/dsv4_serve/cli.py", "the OpenEngine entry branch", DS),
        _shared("python/dsv4_serve/engine_client.py", "request_abort", DS),
        _shared("python/dsv4_serve/engine_api.py", "MockEngine extensions", DS),
        _shared("python/pyproject.toml", "the openengine extra and packages", DS),
        _shared("python/requirements-server.txt", "the openengine extra", DS),
        _shared("README.md", "the OpenEngine section", DS),
    ],
    "WT-42": _own("tests/openengine/test_disagg_roles.py", "dev/openengine_disagg*", repo=DS)
    # NT-042/043: §3.3 names WT-42 the author of this TRT-LLM test file.
    + _own(f"{APPS}/_test_openengine_ds_disagg_direct.py")
    + [
        _shared(
            f"{DS_OE}/**",
            "the disagg branches: the handoff module, the disagg paths of servicer, validation and"
            " capabilities, the sub-agent affinity check",
            DS,
        ),
        _shared("python/dsv4_serve/engine_api.py", "MockEngine's disagg behaviour", DS),
        _shared(
            "python/dsv4_serve/cli.py",
            "the OpenEngine entry branch: disagg roles and the affinity key",
            DS,
        ),
        _shared("README.md", "the OpenEngine section", DS),
        # WT-41 serves aggregated requests only and its tests pin that down (ruling U14); WT-42
        # opens the disagg roles and flips those assertions in the tests that make them. The
        # whole file is allowed and listed for review.
        _exception(
            "tests/openengine/test_cli_entry.py",
            "U14 flip only: test_openengine_refuses_a_disagg_role turned into the roles'"
            " acceptance, with the fixture and module docstring that describe it",
            DS,
        ),
        _exception(
            "tests/openengine/test_openengine_server.py",
            "U14 flips only: conversation_id and disagg_request_id reported true in"
            " test_ctl_4_model_info_explicit; disagg_request_id INVALID_ARGUMENT instead of"
            " UNIMPLEMENTED in test_extra_conventions",
            DS,
        ),
        _exception(
            "tests/openengine/test_validation.py",
            "U14 flip only: disagg_request_id INVALID_ARGUMENT instead of UNIMPLEMENTED in"
            " test_disagg_request_keys_refused",
            DS,
        ),
    ],
    "WT-43": _own("dev/openengine_stress*", repo=DS)
    + [
        _shared("csrc/engine.h", "recording and exposing the failure reason", DS),
        _shared("csrc/engine.cpp", "recording and exposing the failure reason", DS),
        _shared("csrc/bindings.cpp", "get_responses raising on an empty poll", DS),
        _shared(f"{DS_OE}/server.py", "the concurrency design", DS),
        _shared(f"{DS_OE}/servicer.py", "the concurrency design", DS),
    ],
    "WT-50": _own(
        "tensorrt_llm/commands/serve.py",
        f"{POC}/launch.py",
        f"{POC}/multi_frontend.py",
        f"{APPS}/_test_openengine_disagg_cpu_control.py",
    )
    + [
        _shared(f"{POC}/facade.py", "CS-18: the conventions-version check opening _check_engine"),
        _exception(
            f"{CLIENT_TESTS}/test_openengine_poc_serve.py",
            "CT-14 ①②: the CLI and YAML refusals of disagg options turned into acceptance",
        ),
    ],
    "WT-51": _own(
        f"{POC}/facade.py",
        f"{POC}/mapping.py",
        f"{POC}/field_classes.py",
        f"{POC}/engine_client_impl.py",
        f"{POC}/response.py",
        f"{POC}/postproc_pool.py",
        f"{POC}/handoff*.py",
        f"{APPS}/_test_openengine_disagg_handoff.py",
    )
    + [
        _shared(
            f"{POC}/contracts.py",
            "CS-05 StepJoiner and PPEvent, CS-13 GenerateCounter; types, keys and the contract"
            " version stay WT-35's and change only with the user's approval (§0.3)",
        ),
        _shared(f"{POC}/engine_client.py", "CS-04 EngineRequest, as contract v10 defines it"),
        _shared(
            "tensorrt_llm/serve/openai_server.py",
            "facade gaps only; unchanged behaviour without the switch",
        ),
        _shared(
            "tensorrt_llm/serve/postprocess_handlers.py",
            "facade gaps only; unchanged behaviour without the switch",
        ),
        _exception(
            f"{POC_TESTS}/wt11/test_poc2_wt11_mapping.py",
            "CT-14 ③: disaggregated_params asserted as mapped instead of ValueError",
        ),
        _exception(
            f"{POC_TESTS}/wt01/test_poc2_wt01_contracts.py",
            "CT-14 ③: disaggregated_params asserted as mapped instead of ValueError",
        ),
        _exception(
            f"{POC_TESTS}/wt11/test_poc2_wt11_no_silent_drop.py",
            "CT-14 ③: disaggregated_params asserted as mapped instead of ValueError",
        ),
        # SV-DSG-8, which WT-51-fix2 makes: the undeclared case of WT-55's test of the generation
        # leg asserts that the decode call of a generation-first request is refused.
        _exception(
            f"{POC_TESTS}/wt55/test_poc3_wt55_gen_first.py",
            "SV-DSG-8 flip only: the undeclared case of"
            f" {_GENERATION_LEG_TEST} asserts the refusal of the decode call",
            item="WT-51-fix2",
            changes=(
                _change(
                    _GENERATION_LEG_TEST,
                    added=(
                        "    if not generation_first:",
                        "        # SV-DSG-8: a generation-first decode call needs the engine to"
                        " declare the capability.",
                        "        with pytest.raises(ValueError,"
                        r' match=r"generation-first.*trtllm\.disagg\.generation_first"):',
                        "            _map(_gen_first_gen_params(), _caps(generation_first=False))",
                        "        return",
                    ),
                ),
            ),
        ),
    ],
    "WT-52": _own(
        *(
            f"{APPS}/_test_openengine_disagg_{domain}.py"
            for domain in (
                "cancel",
                "first_token",
                "leg_consistency",
                "engine_loss",
                "rejects",
                "startup",
                "orphans",
                "arbitrary_kv",
            )
        )
    )
    + [
        _shared("tensorrt_llm/serve/postprocess_handlers.py", "CS-07: the ctx_usage fill"),
        _shared(f"{POC}/response.py", "CS-07"),
        _shared(
            f"{POC}/facade.py",
            "CS-08: check_frontend_options and the role x kv_connector check in _check_engine;"
            " CS-16: the health watch and self-stop",
        ),
        _shared(f"{POC}/field_classes.py", "CS-08: the min_tokens capability gate"),
        _shared(f"{POC}/mapping.py", "CS-08: the min_tokens capability gate"),
        _shared(
            f"{POC}/engine_client_impl.py",
            "CS-19: the debug dump, appended where GenerateRequest is built",
        ),
        # CS-08: a worker frontend with a role needs an engine that declares a KV connector, so
        # wt50_chain's fake engine declares one, and a handoff past the signature check reaches it.
        _exception(
            f"{POC_TESTS}/wt50/wt50_chain.py",
            "only engine() serving a disagg_profile, with its docstring and import",
            item="WT-52",
            changes=(
                _change(
                    "import",
                    (
                        "from fake_openengine import PROFILES, FakeEngine, running_in_thread"
                        "  # noqa: E402",
                    ),
                    (
                        "from fake_openengine import PROFILES, FakeEngine, disagg_profile,"
                        " running_in_thread  # noqa: E402",
                    ),
                ),
                _change(
                    "engine",
                    (
                        '    """A ``trtllm-like`` fake engine serving the tokenizer in ``model_path``."""',
                    ),
                    (
                        '    """A ``trtllm-like`` fake engine with KV transfer, serving the tokenizer'
                        ' in ``model_path``."""',
                    ),
                ),
                _change(
                    "engine",
                    ('    profile = PROFILES["trtllm-like"].replace(',),
                    ('    profile = disagg_profile(PROFILES["trtllm-like"]).replace(',),
                ),
            ),
        ),
        _exception(
            f"{POC_TESTS}/wt50/test_poc3_wt50_internal_auth.py",
            "only the two handoffs past the signature check expecting 200 and one engine call,"
            " with their comment and docstrings, and _refused_by_the_facade removed",
            item="WT-52",
            changes=(
                _change(
                    "_refused_by_the_facade",
                    (
                        "def _refused_by_the_facade(response: requests.Response) -> bool:",
                        '    """A 400 about ``disaggregated_params``, past the signature and'
                        ' transceiver checks."""',
                        "    text = response.text",
                        "    return (",
                        "        response.status_code == 400",
                        '        and "disaggregated_params" in text',
                        "        and AUTH_FAILED not in text",
                        "        and NO_TRANSCEIVER not in text",
                        "    )",
                        "",
                        "",
                    ),
                ),
                _change(
                    _TAMPERED_HANDOFF_TEST,
                    (
                        '    """NT-008 (DG-34, DG-37): a handoff whose signature does not match is'
                        ' refused before the engine."""',
                    ),
                    (
                        '    """NT-008 (DG-34, DG-37): a keyed worker refuses a handoff whose signature'
                        ' does not match."""',
                    ),
                ),
                _change(
                    _TAMPERED_HANDOFF_TEST,
                    ("    # to the facade's refusal of the handoff fields.",),
                    (
                        "    # to the engine.",
                        "    keyless_before = chain.generate_calls(keyless_engine)",
                    ),
                ),
                _change(
                    _TAMPERED_HANDOFF_TEST,
                    (
                        "    assert _refused_by_the_facade(response),"
                        " (response.status_code, response.text)",
                        "    assert chain.generate_calls(keyless_engine) == 0",
                    ),
                    (
                        "    assert response.status_code == 200, response.text",
                        "    assert chain.generate_calls(keyless_engine) == keyless_before + 1",
                    ),
                ),
                _change(
                    _SIGNED_HANDOFF_TEST,
                    ("    facade, which refuses the handoff fields it does not map.",),
                    ("    engine.",),
                ),
                _change(
                    _SIGNED_HANDOFF_TEST,
                    (
                        "    assert _refused_by_the_facade(response),"
                        " (response.status_code, response.text)",
                        "    assert chain.generate_calls(keyed_engine) == before",
                    ),
                    (
                        "    assert response.status_code == 200, response.text",
                        "    assert chain.generate_calls(keyed_engine) == before + 1",
                    ),
                ),
            ),
        ),
        _exception(
            f"{POC_TESTS}/wt11/test_poc2_wt11_no_silent_drop.py",
            "DQ-48 flip only: the over-gRPC min_tokens probe expecting the facade to refuse"
            " min_tokens to an engine on the current conventions that does not declare"
            " trtllm.stopping.min_tokens; the mapping-level probes are unchanged",
            item="WT-52",
            changes=(
                _change(
                    "test_over_grpc_the_engine_receives_what_the_mapping_sends",
                    added=(
                        '        if name == "min_tokens" and (',
                        "            conv.conventions_version(facade.client.server_info().extra)"
                        " == conv.CONVENTIONS_VERSION",
                        "            and not caps.supports(conv.CAP_STOPPING_MIN_TOKENS)",
                        "        ):",
                        "            # DQ-48: the facade refuses min_tokens to an engine on the current",
                        "            # conventions that does not declare it.",
                        '            with pytest.raises(ValueError, match="min_tokens"):',
                        "                await facade.generate_async(GRPC_PROMPT,"
                        " _sampling(probe.params), **options)",
                        "            assert engine.requests == []",
                        "            return",
                    ),
                ),
            ),
        ),
    ],
    "WT-53": _own(
        *(
            f"{APPS}/_test_openengine_disagg_{domain}.py"
            for domain in (
                "kv_aware",
                "subagent_affinity",
                "multi_frontend",
                "server_info",
                "admission",
            )
        )
    )
    + [
        _shared(f"{POC}/mapping.py", "CS-09 subagent_affinity_id; CS-15"),
        _shared(f"{POC}/facade.py", "CS-09 _metadata; CS-10 engine_block and /server_info"),
        _shared(
            f"{POC}/contracts.py",
            "CS-11 kv_cache_events; contract changes need the user's approval (§0.3)",
        ),
        _shared(f"{POC}/metrics.py", "CS-11"),
        _shared("tensorrt_llm/serve/openai_server.py", "CS-10: get_server_info, minimal"),
        _shared(f"{POC}/multi_frontend.py", "CS-17"),
        _shared(f"{POC}/launch.py", "CS-17"),
        # Contract v11 (rulings U24 and N46), which WT-53-fix1 makes: EngineInfo carries the
        # engine's KV-cache block size and block count, and a frontend subscribes to KV-cache
        # events at its start, so the default /kv_cache_events answers no events.
        _exception(
            f"{POC}/engine_client.py",
            "contract v11 (U24, N46) only: the EngineInfo.kv_block_size and total_kv_blocks"
            " fields and their docstring",
            item="WT-53-fix1",
            changes=(
                _change(
                    "EngineInfo",
                    added=(
                        "        kv_block_size: ``ServerInfo.capacity.kv_block_size``, the tokens in"
                        " one",
                        "            KV-cache block; None when the engine reports none.",
                        "        total_kv_blocks: ``ServerInfo.capacity.total_kv_blocks``, the"
                        " KV-cache",
                        "            blocks the engine can allocate; None when the engine reports"
                        " none.",
                    ),
                ),
                _change(
                    "EngineInfo",
                    added=(
                        "    kv_block_size: int | None = None",
                        "    total_kv_blocks: int | None = None",
                    ),
                ),
            ),
        ),
        _exception(
            f"{POC}/engine_client_impl.py",
            "contract v11 (U24, N46) only: _engine_info filling kv_block_size and"
            " total_kv_blocks from the capacity",
            item="WT-53-fix1",
            changes=(
                _change(
                    "_engine_info",
                    added=(
                        "        kv_block_size=capacity.kv_block_size or None,",
                        "        total_kv_blocks=capacity.total_kv_blocks or None,",
                    ),
                ),
            ),
        ),
        _exception(
            f"{CLIENT_TESTS}/test_openengine_poc_serve.py",
            "contract v11 (U24) only: the 501 handler case raising OpenEngineUnsupportedError"
            " itself, since the default /kv_cache_events answers no events",
            item="WT-53-fix1",
            changes=(
                _change(
                    _UNSUPPORTED_HANDLER_TEST,
                    added=(
                        "    async def unsupported(client, timeout):",
                        '        raise contracts.OpenEngineUnsupportedError("no KV-cache events")',
                        "",
                    ),
                ),
                _change(
                    _UNSUPPORTED_HANDLER_TEST,
                    added=("        facade._metrics.kv_cache_events = unsupported",),
                ),
            ),
        ),
        _exception(
            f"{POC_TESTS}/wt01/test_poc2_wt01_contracts.py",
            "U24 flips only: contract version 11 in test_contract_version, and the default"
            " metrics answering no KV-cache events instead of raising OpenEngineUnsupportedError",
            item="WT-53-fix1",
            changes=(
                _change(
                    "test_contract_version",
                    ("    assert contracts.POC2_CONTRACT_VERSION == 10",),
                    ("    assert contracts.POC2_CONTRACT_VERSION == 11",),
                ),
                _change(
                    "test_default_metrics_record_nothing",
                    (
                        "    with pytest.raises(contracts.OpenEngineUnsupportedError):",
                        "        asyncio.run(recorder.kv_cache_events(FakeEngineClient(), None))",
                    ),
                    (
                        "    asyncio.run(recorder.subscribe_kv_cache_events(FakeEngineClient()))",
                        "    assert asyncio.run(recorder.kv_cache_events(FakeEngineClient(), None))"
                        " == []",
                    ),
                ),
            ),
        ),
        _exception(
            f"{POC_TESTS}/wt35/test_poc3_wt35_contracts.py",
            "U24 flip only: contract version 11 in test_contract_version",
            item="WT-53-fix1",
            changes=(
                _change(
                    "test_contract_version",
                    ("    assert contracts.POC2_CONTRACT_VERSION == 10",),
                    ("    assert contracts.POC2_CONTRACT_VERSION == 11",),
                ),
            ),
        ),
        _exception(
            f"{POC_TESTS}/wt15/test_poc2_wt15_metrics.py",
            "U24 flips only, two tests renamed to say so: /kv_cache_events answering no events"
            " instead of 501 for no sources or ones the frontend cannot read, and the subscription"
            " answering no events instead of raising OpenEngineUnsupportedError for an engine"
            " without sources; N46: the test client's server_info() and load(); a comment on the"
            " first call",
            item="WT-53-fix1",
            changes=(
                _change(
                    "test_kv_cache_events_arrive_through_get_kv_event_sources",
                    ("        # The first call subscribes; nothing was received before it.",),
                    (
                        "        # The frontend subscribed at its start; nothing was published"
                        " since, and",
                        "        # the engine's load reports no total_kv_blocks for a created"
                        " event.",
                    ),
                ),
                _change(
                    _NO_EVENTS_HTTP_TEST,
                    (f"def {_NO_EVENTS_HTTP_TEST_WAS}(tmp_path, sources):",),
                    (f"def {_NO_EVENTS_HTTP_TEST}(tmp_path, sources):",),
                ),
                _change(
                    _NO_EVENTS_HTTP_TEST,
                    (
                        "    assert [r.status_code for r in responses] == [501, 501],"
                        " responses[0].text",
                    ),
                    (
                        "    assert [r.status_code for r in responses] == [200, 200],"
                        " responses[0].text",
                        "    assert [r.json() for r in responses] == [[], []]",
                    ),
                ),
                _change(
                    "_Client",
                    ('        return SimpleNamespace(engine_name="e", extra={})',),
                    (
                        '        return SimpleNamespace(engine_name="e", extra={},'
                        " total_kv_blocks=None)",
                        "",
                        "    async def load(self):",
                        "        return None",
                    ),
                ),
                _change(
                    _NO_EVENTS_SUBSCRIPTION_TEST,
                    (f"def {_NO_EVENTS_SUBSCRIPTION_TEST_WAS}():",),
                    (f"def {_NO_EVENTS_SUBSCRIPTION_TEST}():",),
                ),
                _change(
                    _NO_EVENTS_SUBSCRIPTION_TEST,
                    (
                        "    async def raised():",
                        "        try:",
                        "            await subscription.events(client, 0)",
                        "        except contracts.OpenEngineUnsupportedError as error:",
                        "            return error",
                        '        raise AssertionError("not raised")',
                        "",
                    ),
                ),
                _change(
                    _NO_EVENTS_SUBSCRIPTION_TEST,
                    ("        return [await raised() for _ in range(2)]",),
                    ("        return [await subscription.events(client, 0) for _ in range(2)]",),
                ),
                _change(
                    _NO_EVENTS_SUBSCRIPTION_TEST,
                    (
                        "    first, second = asyncio.run(twice())",
                        "    # Raising the same error again would lengthen its traceback at every"
                        " call.",
                        "    assert first is not second",
                        "    assert str(first) == str(second)",
                    ),
                    (
                        "    assert asyncio.run(twice()) == [[], []]",
                        "    # The engine's answer is kept: asked once.",
                    ),
                ),
            ),
            renames=(
                (_NO_EVENTS_HTTP_TEST_WAS, _NO_EVENTS_HTTP_TEST),
                (_NO_EVENTS_SUBSCRIPTION_TEST_WAS, _NO_EVENTS_SUBSCRIPTION_TEST),
            ),
        ),
    ],
    "WT-54": _own("scripts/poc3/wt54_*", f"{APPS}/_test_openengine_disagg_perf_metrics.py")
    + [
        _shared(f"{POC}/metrics.py", "CS-12"),
        _shared("tensorrt_llm/serve/openai_server.py", "CS-12: the role segment of the JSONL name"),
        _shared(f"{OE_SERVER}/streaming.py", "CO-08: durations only"),
        _shared(f"{OE_SERVER}/servicer.py", "CO-08: durations only"),
    ],
    "WT-55": _own(f"{APPS}/_test_openengine_disagg_gen_first.py")
    + [
        _shared(f"{POC}/mapping.py", "CS-14 gen-first"),
        _shared(f"{POC}/facade.py", "CS-14 gen-first"),
        _shared(f"{POC}/handoff*.py", "CS-14: the gen worker front end building KvSessionRef"),
        _shared(f"{OE_SERVER}/disagg.py", "CO-09: reading extra on prefill requests"),
        # CO-09: an engine with a KV cache transceiver declares generation-first and serves a
        # generation-first context-only request that carries its disagg_request_id (DQ-06).
        _exception(
            f"{OE_SERVER_TESTS}/test_disagg_control.py",
            "DQ-06 flip only: trtllm.disagg.generation_first declared with a transceiver in"
            " test_model_info_declares_every_version_4_capability_key",
            item="WT-55",
            changes=(
                _change(
                    "test_model_info_declares_every_version_4_capability_key",
                    ('        "trtllm.disagg.generation_first": False,',),
                    ('        "trtllm.disagg.generation_first": enabled,',),
                ),
            ),
        ),
        _exception(
            f"{OE_SERVER_TESTS}/test_disagg_servicer.py",
            "DQ-06 flip only: the schedule_style refusal turned into the acceptance of a"
            " generation-first context-only request, renamed to say so; schedule_style among the"
            " keys refused without a transceiver",
            item="WT-55",
            changes=(
                _change(
                    _GENERATION_FIRST_TEST,
                    (
                        f"def {_GENERATION_FIRST_TEST_WAS}():",
                        '    request = _request(extra={"request_type": "context_only",'
                        ' "schedule_style": 1})',
                    ),
                    (
                        f"def {_GENERATION_FIRST_TEST}():",
                        "    llm = _Llm([_prefill_result()])",
                        "    request = _request(",
                        "        extra={",
                        '            "request_type": "context_only",',
                        '            "disagg_request_id": str(CALLER_ID),',
                        '            "schedule_style": 1,',
                        "        }",
                        "    )",
                    ),
                ),
                _change(
                    _GENERATION_FIRST_TEST,
                    ("    _, context = _generate(_servicer(), request)",),
                    ("    _, context = _generate(_servicer(llm), request)",),
                ),
                _change(
                    _GENERATION_FIRST_TEST,
                    (
                        "    assert context.abort_code == grpc.StatusCode.UNIMPLEMENTED",
                        '    assert "schedule_style" in context.abort_details',
                    ),
                    (
                        "    assert context.abort_code is None, context.abort_details",
                        '    params = llm.calls[0]["disaggregated_params"]',
                        "    assert params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST",
                        "    assert params.disagg_request_id == CALLER_ID",
                    ),
                ),
                _change(
                    "test_disagg_request_keys_are_unimplemented_without_a_transceiver",
                    (
                        '    for extra in ({"disagg_request_id": str(CALLER_ID)},'
                        ' {"orig_elapsed_s": 0.5}):',
                    ),
                    (
                        "    for extra in (",
                        '        {"disagg_request_id": str(CALLER_ID)},',
                        '        {"schedule_style": 1},',
                        '        {"orig_elapsed_s": 0.5},',
                        "    ):",
                    ),
                ),
            ),
            renames=((_GENERATION_FIRST_TEST_WAS, _GENERATION_FIRST_TEST),),
        ),
        # WT-55-fix1 registers NT-091 as a test that cancels calls (DG-39b); the harness test that
        # lists every such test names it too.
        _exception(
            f"{CHAIN}/tests/test_expectations.py",
            f"N96 only: name NT-091 among the tests that cancel calls in {_CANCELS_CALLS_TEST}",
            item="WT-55-fix1",
            changes=(
                _change(
                    _CANCELS_CALLS_TEST,
                    added=(
                        '        "_test_openengine_disagg_gen_first.py::'
                        'test_gen_first_cancel_closes_both_legs",',
                    ),
                ),
            ),
        ),
    ],
    "WT-56": _own(
        f"{APPS}/_test_openengine_ds_disagg_basic.py",
        f"{APPS}/_test_openengine_ds_disagg_frontend.py",
    )
    + [
        _shared(f"{CHAIN}/conventions.py", "CT-08: the ds chain value and its command prefix"),
        _shared(f"{CHAIN}/launch.py", "CT-08: _engine_prefix"),
    ],
    # The domains of §6.3 of 02_functional-tests.md; the files other items own are excluded below.
    "WT-57": _own(f"{APPS}/_test_openengine_disagg_*.py")
    + [
        # WT-57-fix1 registers NT-199 as a test that cancels calls; the harness test that lists
        # every such test names it too.
        _exception(
            f"{CHAIN}/tests/test_expectations.py",
            f"N123 only: name NT-199 among the tests that cancel calls in {_CANCELS_CALLS_TEST}",
            item="WT-57-fix1",
            changes=(
                _change(
                    _CANCELS_CALLS_TEST,
                    added=(
                        '        "_test_openengine_disagg_transceiver.py::'
                        'test_cancel_during_transfer_is_the_same_on_both_forms",',
                    ),
                ),
            ),
        ),
    ],
    # WT-60-fix1 (N132): XD-313's time limit on the chain (the expectations and the plugin,
    # WT-38's), SURVIVOR_WAIT_S and the exit note (launch.py), and the orchestrator's wait for
    # chain workers that have exited (remote.py, openai_server.py; WT-37's), with the harness's
    # self-tests of the three in a new file.
    "WT-60": [
        _exception(
            f"{CHAIN}/expectations/__init__.py",
            "WT-60-fix1 only: the time_limit kind (TIME_LIMIT, KINDS, Expectation.seconds and its"
            " checks, applicable, time_limit)",
            item=WT60_FIX1,
            changes=(
                _change_around(
                    (None,),
                    added=(
                        "",
                        "A ``time_limit`` entry gives the test ``seconds`` in place of the limit"
                        " of its",
                        "own ``timeout`` mark (the pytest plugin adds the mark at collection).",
                    ),
                ),
                _change_around(
                    ("LEG_EXPECT", "TIME_LIMIT"),
                    added=(
                        "# The test may run for ``seconds`` instead of the limit of its own"
                        " timeout mark, for",
                        "# example because the chain starts its two workers' engines one after"
                        " the other.",
                        'TIME_LIMIT = "time_limit"',
                    ),
                ),
                _change(
                    "KINDS",
                    (
                        "KINDS = (HTTP_STATUS, STARTUP_REJECTED, NO_ENGINE_TRAFFIC, CANCELS_CALLS,"
                        " SKIP, LEG_EXPECT)",
                    ),
                    (
                        "KINDS = (",
                        "    HTTP_STATUS,",
                        "    STARTUP_REJECTED,",
                        "    NO_ENGINE_TRAFFIC,",
                        "    CANCELS_CALLS,",
                        "    SKIP,",
                        "    LEG_EXPECT,",
                        "    TIME_LIMIT,",
                        ")",
                    ),
                ),
                _change("Expectation", added=("    seconds: int | None = None",)),
                _change(
                    "Expectation",
                    added=(
                        "        if (self.kind == TIME_LIMIT) != (self.seconds is not None):",
                        "            raise ValueError(",
                        '                f"{self.wt}: {self.match!r}: seconds is required for,'
                        ' and only for, {TIME_LIMIT}"',
                        "            )",
                        "        if self.seconds is not None and (type(self.seconds) is not int"
                        " or self.seconds <= 0):",
                        "            raise ValueError(",
                        '                f"{self.wt}: {self.match!r}: seconds must be a positive'
                        ' int, not {self.seconds!r}"',
                        "            )",
                    ),
                ),
                _change(
                    "applicable",
                    ("    for kinds in (OUTCOME_KINDS, (LEG_EXPECT,)):",),
                    ("    for kinds in (OUTCOME_KINDS, (LEG_EXPECT,), (TIME_LIMIT,)):",),
                ),
                _change_around(
                    ("leg_expectation", "time_limit"),
                    added=(
                        "def time_limit(found: Sequence[Expectation]) -> Expectation | None:",
                        '    """The test\'s ``time_limit`` entry; :func:`applicable` allows at most'
                        ' one."""',
                        "    return next((e for e in found if e.kind == TIME_LIMIT), None)",
                        "",
                        "",
                    ),
                ),
            ),
        ),
        _exception(
            f"{CHAIN}/pytest_plugin.py",
            "WT-60-fix1 only: the timeout mark of a time_limit entry in"
            " pytest_collection_modifyitems",
            item=WT60_FIX1,
            changes=(
                _change(
                    "pytest_collection_modifyitems",
                    added=(
                        "        limit = ex.time_limit(found)",
                        "        if limit is not None:",
                        "            # First among the item's own marks, so pytest-timeout reads it"
                        " before the test's.",
                        "            item.add_marker(pytest.mark.timeout(limit.seconds),"
                        " append=False)",
                    ),
                ),
            ),
        ),
        _exception(
            f"{CHAIN}/launch.py",
            "WT-60-fix1 only: SURVIVOR_WAIT_S and the note of processes slow to exit in"
            " LoggedProcess.stop",
            item=WT60_FIX1,
            changes=(
                _change_around(
                    ("TERMINATE_GRACE_S", "SURVIVOR_WAIT_S"),
                    ("SURVIVOR_WAIT_S = 5.0",),
                    (
                        "# A process holding GPUs exits only once their memory is freed, which"
                        " takes a large-model",
                        "# engine well over 5 s after SIGKILL.",
                        "SURVIVOR_WAIT_S = 60.0",
                    ),
                ),
                _change(
                    "LoggedProcess",
                    ("        deadline = time.monotonic() + SURVIVOR_WAIT_S",),
                    (
                        "        started = time.monotonic()",
                        "        deadline = started + SURVIVOR_WAIT_S",
                    ),
                ),
                _change("LoggedProcess", added=("        lingered = sorted(survivors)",)),
                _change(
                    "LoggedProcess",
                    added=(
                        "        elif lingered:",
                        "            self.notes.append(",
                        '                f"{self.role} pid {self.pid}: processes {lingered} took "',
                        '                f"{time.monotonic() - started:.1f} s to exit after'
                        ' SIGKILL"',
                        "            )",
                    ),
                ),
            ),
        ),
        _exception(
            f"{CHAIN}/remote.py",
            "WT-60-fix1 only: exited_workers and its import of handles_since",
            item=WT60_FIX1,
            changes=(
                _change("import", added=("    handles_since,",)),
                _change_around(
                    ("launch_for_remote_server", "exited_workers"),
                    added=(
                        "def exited_workers() -> list[str]:",
                        '    """Why each disaggregated-server worker started here, and not'
                        " stopped, has exited.",
                        "",
                        "    A server outside the chain, such as the disaggregated server those"
                        " workers",
                        "    register with, cannot tell that one has exited: without this it"
                        " waits for",
                        "    the worker until its own limit.",
                        '    """',
                        "    reasons = []",
                        "    for handle in handles_since(0):",
                        "        if not handle.alive or handle.role is None:",
                        "            continue",
                        "        rc = handle.poll()",
                        "        if rc is None:",
                        "            continue",
                        "        why = handle.frontend_port_problem() or (",
                        '            f"logs {handle.frontend_log} and {handle.engine_log}"',
                        "        )",
                        '        reasons.append(f"the {handle.role} worker on port {handle.port}'
                        ' exited ({rc}): {why}")',
                        "    return reasons",
                        "",
                        "",
                    ),
                ),
            ),
        ),
        _exception(
            f"{APPS}/openai_server.py",
            "WT-60-fix1 only: watch_workers and _check_chain_workers in RemoteOpenAIServer",
            item=WT60_FIX1,
            changes=(
                _change(
                    "RemoteOpenAIServer",
                    added=(
                        "        # A server outside the chain while one is selected, such as the",
                        "        # disaggregated server that chain workers register with.",
                        "        watch_workers = self._chain_handle is None and bool(",
                        '            os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN")',
                        '            or os.environ.get("TRTLLM_TEST_OPENENGINE_TOPOLOGY"))',
                    ),
                ),
                _change(
                    "RemoteOpenAIServer",
                    added=(
                        "            elif watch_workers:",
                        "                self._check_chain_workers(start, timeout)",
                    ),
                ),
                _change(
                    "RemoteOpenAIServer",
                    added=(
                        "    def _check_chain_workers(self, start: float, timeout: float):",
                        "        # Checked before every /health request: a server outside the"
                        " chain may",
                        "        # answer it with another status, such as 503, while it waits"
                        " for workers.",
                        "        from openengine_chain.remote import exited_workers",
                        "        exited = exited_workers()",
                        "        if not exited and time.time() - start <= timeout:",
                        "            return",
                        "        self.terminate()",
                        "        if exited:",
                        '            raise RuntimeError("Server failed to start: " + "; ".join(exited))',
                        '        raise RuntimeError("Server failed to start in time.")',
                        "",
                    ),
                ),
            ),
        ),
        _exception(
            f"{CHAIN}/tests/test_waits.py",
            "WT-60-fix1 only: creating the harness's self-tests of its waits, as registered",
            item=WT60_FIX1,
            blob=WT60_FIX1_TEST_WAITS,
        ),
    ],
    "WT-61": [],
    "WT-62": _own(
        f"{APPS}/_test_openengine_ds_disagg_*.py",
        f"!{APPS}/_test_openengine_ds_disagg_basic.py",
        f"!{APPS}/_test_openengine_ds_disagg_frontend.py",
        f"!{APPS}/_test_openengine_ds_disagg_direct.py",
    )
    + [
        _shared(
            f"{APPS}/_test_openengine_ds_disagg_basic.py",
            "the skeleton WT-56 wrote, taken over once WT-56 has landed (§3.3)",
        ),
        _shared(
            f"{APPS}/_test_openengine_ds_disagg_frontend.py",
            "the skeleton WT-56 wrote, taken over once WT-56 has landed (§3.3)",
        ),
        # WT-62 registers NT-132 and NT-133 as tests that cancel calls; the harness test that lists
        # every such test names them too.
        _exception(
            f"{CHAIN}/tests/test_expectations.py",
            "N107 only: name NT-132 and NT-133 among the tests that cancel calls in"
            f" {_CANCELS_CALLS_TEST}",
            item="WT-62",
            changes=(
                _change(
                    _CANCELS_CALLS_TEST,
                    added=(
                        '        "_test_openengine_ds_disagg_cancel.py::test_abort_stages[*",',
                        '        "_test_openengine_ds_disagg_cancel.py::'
                        'test_parked_kv_timeout_and_orphans",',
                    ),
                ),
            ),
        ),
    ],
    "WT-70": [
        _shared(
            f"{PERF}/test_perf_sanity.py",
            "append only: CP-01 arm guard, CP-02 disagg run_cmd branches, CP-03 log line",
        ),
        _shared(f"{PERF}/openengine_arms.py", "append only: CP-01, CP-04, CP-09"),
        _shared("jenkins/scripts/perf/local/**", "CP-05: the new single-node disagg driver"),
        _shared(f"{OE_SERVER}/servicer.py", "CP-09: the timestamp probe"),
        _shared(f"{POC}/facade.py", "CP-09: the worker front end's timestamp probes"),
        _shared(f"{CHAIN}/direct_replay*.py", "CT-13: the full replay tool"),
    ],
    "WT-71": [
        _shared(
            f"{PERF}/openengine_arms.py", "append only: CP-06, CP-08 ds arms; CP-11 ds arm launch"
        ),
        _shared(f"{PERF}/test_perf_sanity.py", "append only: CP-07 the client restriction"),
        _shared("python/dsv4_serve/protocol.py", "CD-09: the four DisaggregatedParams fields", DS),
        # WT-71 appends the case-03 arms to DISAGG_ARMS (N102), so WT-70's test of the arm table
        # has to expect them, each arm with its own case.
        _exception(
            f"{POC_TESTS}/wt70/test_poc3_wt70_arms.py",
            "N102 only: the E3 arms added to EXPECTED_DISAGG_ARMS, and the case expected per arm"
            f" in {_ARM_TABLE_TEST}",
            item="WT-71",
            changes=(
                _change(
                    "EXPECTED_DISAGG_ARMS",
                    added=(
                        "    # The case-03 arms (the ds engine), after the case-04 ones.",
                        '    "E3-native": ("disagg", ("none", None, None), ("none", None, None)),',
                        '    "E3-native-al": ("disagg", ("none", 1, None), ("none", 1, None)),',
                        '    "E3-mid": ("disagg", ("none", 1, None), ("none", 1, None)),',
                        '    "E3-chain": ("disagg", ("ds", 1, None), ("ds", 1, None)),',
                        '    "E3-chain-lean": ("disagg", ("ds", 1, 0), ("ds", 1, 0)),',
                        '    "E3-direct": ("direct", None, None),',
                    ),
                ),
                _change(
                    _ARM_TABLE_TEST,
                    ('            "case-04",',),
                    ('            "case-03" if name.startswith("E3-") else "case-04",',),
                ),
            ),
        ),
    ],
    "WT-72": [],
    "WT-73": [],
    "WT-79": [],
}
OWNERSHIP["WT-57"] += [
    Rule(f"!{rule.pattern}", repo=rule.repo)
    for name, rules in OWNERSHIP.items()
    if name != "WT-57"
    for rule in rules
    if rule.kind == "own" and rule.pattern.startswith(f"{APPS}/_test_openengine_disagg_")
]
for _name in OWNERSHIP:
    OWNERSHIP[_name] = OWNERSHIP[_name] + _per_item(_name)
    OWNERSHIP[_name] += _own(f"{POC_TESTS}/wt{_name.removeprefix('WT-')}/**")


def _named_fixes(table: dict[str, list[Rule]]) -> list[str]:
    """The fixes the exceptions of ``table`` are for, in the order of the table.

    Raises:
        ValueError: An exception lists its changes or placeholders but no work item, or is for
            neither the work item it is listed under nor one of that item's fixes.
    """
    fixes: list[str] = []
    for name, rules in table.items():
        for rule in rules:
            if rule.changes and not rule.item:
                raise ValueError(f"{rule.pattern}: an exception listing changes names no item")
            if rule.redact and not rule.item:
                raise ValueError(f"{rule.pattern}: an exception listing placeholders names no item")
            if rule.blob and not rule.item:
                raise ValueError(f"{rule.pattern}: an exception creating a file names no item")
            if not rule.item:
                continue
            if parse_task(rule.item)[0] != name:
                raise ValueError(f"{rule.pattern}: an exception for {rule.item} is under {name}")
            if rule.item not in table and rule.item not in fixes:
                fixes.append(rule.item)
    return fixes


NAMED_FIXES = _named_fixes(OWNERSHIP)


@lru_cache(maxsize=None)
def _regex(pattern: str) -> re.Pattern[str]:
    """Compile a glob: ``**`` crosses directories, ``*`` and ``?`` do not."""
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _matches(patterns: tuple[str, ...], path: str) -> bool:
    matched = False
    for pattern in patterns:
        if pattern.startswith("!"):
            matched = matched and not _regex(pattern[1:]).match(path)
        elif _regex(pattern).match(path):
            matched = True
    return matched


def match_rule(task: str, path: str, repo: str = TRTLLM) -> Rule | None:
    """The rule of ``task`` that allows changing ``path``, or None."""
    base, fix = parse_task(task)
    if base != "WT-30" and repo == TRTLLM and _matches(GATE_FILES, path):
        return None
    rules = OWNERSHIP[base] + (_per_item(task) if fix is not None else [])
    matched: Rule | None = None
    for rule in rules:
        if rule.repo != repo or rule.item not in ("", task):
            continue
        if rule.pattern.startswith("!"):
            if matched is not None and _regex(rule.pattern[1:]).match(path):
                matched = None
            continue
        if matched is None and _regex(rule.pattern).match(path):
            matched = rule
    return matched


def owners(path: str, repo: str = TRTLLM) -> list[tuple[str, Rule]]:
    """Every work item with a rule for ``path``, and that rule.

    A fix is listed only where an exception for it by name covers the path. An
    exception whose work item has landed is listed too, though it allows nothing
    any more (see :func:`check`).
    """
    found = []
    for name in OWNERSHIP:
        rule = match_rule(name, path, repo)
        if rule is not None:
            found.append((name, rule))
    for name in NAMED_FIXES:
        rule = match_rule(name, path, repo)
        if rule is not None and rule.item == name:
            found.append((name, rule))
    return found


def detect_repo(repo_dir: str) -> str:
    """``trtllm`` or ``ds``, from the checkout's layout."""
    top = Path(git(repo_dir, "rev-parse", "--show-toplevel").strip())
    if (top / "tensorrt_llm" / "__init__.py").is_file():
        return TRTLLM
    if (top / "python" / "dsv4_serve").is_dir():
        return DS
    raise ValueError(f"{top} is neither a TensorRT-LLM nor a ds-engine checkout")


# A diff header line that may precede the hunks of a line-limited exception's file.
_PLAIN_HEADER = ("diff --git ", "index ", "--- ", "+++ ")
_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _statement_name(node: ast.stmt) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "import"
    targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(targets[0], ast.Name):
        return targets[0].id
    return None


def _regions(source: str) -> list[str | None]:
    """The name of the top-level statement each line of ``source`` lies in (index 0 is line 1).

    A statement's lines run from its first line (its first decorator) to the
    line before the next top-level statement; lines before the first statement
    lie in none (None), as do the lines of statements :func:`_statement_name`
    does not name.

    Raises:
        SyntaxError, ValueError: ``source`` is not Python, or splits into lines
            differently for Python and for git (a lone carriage return).
    """
    if "\r" in source.replace("\r\n", ""):
        raise ValueError("a lone carriage return")
    count = len(source.split("\n")) - (1 if source.endswith("\n") else 0)
    regions: list[str | None] = [None] * count
    nodes = ast.parse(source).body
    starts = [min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])]) for n in nodes]
    for i, node in enumerate(nodes):
        end = starts[i + 1] - 1 if i + 1 < len(nodes) else count
        regions[starts[i] - 1 : end] = [_statement_name(node)] * (end - starts[i] + 1)
    return regions


def _hunk_regions(base: list[str | None], head: list[str | None], span: list[int]) -> set[str]:
    """The named statement a hunk lies in (a set of at most one), given the regions of the file.

    ``span`` is the hunk's (old start, old count, new start, new count). Its
    removed lines must all lie in one statement of ``base``, its added lines in
    one of ``head``, the same one; a hunk that only adds lines must also follow
    or precede a line of that statement in ``base``.
    """

    def one(regions: list[str | None], start: int, count: int) -> set[str | None]:
        found = set(regions[start - 1 : start - 1 + count])
        return found if len(found) == 1 and start - 1 + count <= len(regions) else set()

    old_start, old_count, new_start, new_count = span
    if old_count:
        names = one(base, old_start, old_count)
    else:
        names = {base[i] for i in (old_start - 1, old_start) if 0 <= i < len(base)}
    if new_count:
        names &= one(head, new_start, new_count)
    return {name for name in names if name is not None}


def _hunk_touches(
    base: list[str | None], head: list[str | None], span: list[int]
) -> set[str | None] | None:
    """Every top-level statement a hunk touches, given the regions of the file; None if none fit.

    ``span`` is the hunk's (old start, old count, new start, new count). The
    hunk touches the statements its removed lines lie in at the merge base,
    those its added lines lie in at the head, and, when it removes nothing, the
    one the line it follows lies in at the merge base. None in the set stands
    for lines in no named statement, the lines before the first statement
    included.
    """
    old_start, old_count, new_start, new_count = span
    if old_start - 1 + old_count > len(base) or new_start - 1 + new_count > len(head):
        return None
    touched = set(base[old_start - 1 : old_start - 1 + old_count])
    if not old_count:
        touched.add(base[old_start - 1] if old_start else None)
    touched.update(head[new_start - 1 : new_start - 1 + new_count])
    return touched


def _regions_at(repo_dir: str, rev: str, path: str) -> list[str | None] | None:
    """:func:`_regions` of ``path`` at ``rev``; None when it cannot be read or parsed."""
    found = run_git(repo_dir, "show", f"{rev}:{path}", check=False)
    if found.returncode != 0:
        return None
    try:
        return _regions(found.stdout.decode())
    except (SyntaxError, ValueError, RuntimeError, MemoryError):
        # ValueError: not UTF-8, a NUL byte or a lone carriage return; RuntimeError
        # (RecursionError) and MemoryError: nested too deep to parse.
        return None


def _redacts(placeholders: tuple[tuple[str, str], ...], removed: str, added: str) -> bool:
    """Whether ``added`` is ``removed`` with the start of one or more paths replaced.

    Each placeholder in ``added`` stands for a path start in ``removed`` that its
    pattern matches in full: one that neither follows a word character, ``.`` or
    ``/``, nor goes on with anything but ``/``, ``:``, white space, ``}`` or the
    end of the line. The rest of the two lines is the same. A line with U+FFFD, which every byte that is not UTF-8
    reads as, does not count.
    """
    if "\ufffd" in removed or "\ufffd" in added:
        return False
    starts = dict(placeholders)
    alternatives = "|".join(re.escape(p) for p in sorted(starts, key=len, reverse=True))
    parts = re.split(f"({alternatives})", added)
    if len(parts) == 1:
        return False
    pattern = "".join(
        rf"(?<![\w./])(?:{starts[part]})(?=[/:\s}}]|$)" if i % 2 else re.escape(part)
        for i, part in enumerate(parts)
    )
    return re.fullmatch(pattern, removed, re.ASCII) is not None


def disallowed_lines(
    rule: Rule, repo_dir: str, base_ref: str, head_ref: str, path: str
) -> list[str]:
    """What ``base_ref...head_ref`` changes in ``path`` beyond the lines ``rule`` allows.

    Empty when the diff changes nothing but lines and, with ``rule.lines``,
    every hunk replaces lines in place (as many removed as added, at the same
    line number) and every removed and added line matches one of
    ``rule.lines``; with ``rule.changes``, every hunk is one of the changes,
    none used twice; with ``rule.redact``, every hunk replaces lines in place,
    each added line redacting its removed counterpart (:func:`_redacts`),
    except one hunk that adds ``rule.header`` as the second line of the file.
    """
    # Bytes, not text mode: universal newlines would hide a CRLF line ending. --text: a file git
    # takes for binary (core.bigFileThreshold, a -diff attribute) is still diffed by line.
    diff = run_git(
        repo_dir,
        "diff",
        "-U0",
        "--inter-hunk-context=0",
        "--diff-algorithm=myers",
        "--indent-heuristic",
        "--text",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--no-relative",
        "--ignore-submodules=none",
        f"{base_ref}...{head_ref}",
        "--",
        f":(top,literal){path}",
    ).stdout.decode("utf-8", errors="replace")
    patterns = [re.compile(line, re.ASCII) for line in rule.lines]
    if rule.blob:
        return _created_differently(rule, repo_dir, base_ref, head_ref, path)
    unused = [
        (
            [re.compile(p, re.ASCII) for p in change.removed],
            [re.compile(p, re.ASCII) for p in change.added],
            change.within,
            set(change.around),
        )
        for change in rule.changes
    ]
    base_regions: list[str | None] | None = []
    head_regions: list[str | None] | None = []
    if rule.changes:
        merge_base = run_git(repo_dir, "merge-base", base_ref, head_ref).stdout.decode().strip()
        base_regions = _regions_at(repo_dir, merge_base, path)
        head_regions = _regions_at(repo_dir, head_ref, path)
        if base_regions is None or head_regions is None:
            return [f"{path} is not Python this interpreter parses, at the merge base or the head"]
        for old, new in rule.renames:
            if old not in base_regions or new in base_regions:
                return [f"{path} has no {old} to rename, or has {new} already, at the merge base"]
            if new not in head_regions or old in head_regions:
                return [f"{path} does not rename {old} to {new} at the head"]
        renamed = dict(rule.renames)
        base_regions = [renamed.get(name, name) for name in base_regions]
    bad = []
    # The hunk being read: its header, the statement it lies in (see _hunk_regions), the
    # statements it touches (see _hunk_touches), its removed and added lines.
    hunk: tuple[str, set[str], set[str | None] | None, list[str], list[str]] | None = None

    def matches(
        change: tuple,
        where: set[str],
        touched: set[str | None] | None,
        removed: list[str],
        added: list[str],
    ) -> bool:
        old, new, within, around = change
        return (
            (touched == around if around else within in where)
            and len(old) == len(removed)
            and len(new) == len(added)
            and all(p.fullmatch(line) for p, line in zip(old, removed))
            and all(p.fullmatch(line) for p, line in zip(new, added))
        )

    def end_hunk() -> None:
        if hunk is None or not (rule.changes or rule.redact):
            return
        header, where, touched, removed, added = hunk
        if rule.redact:
            span = _HUNK.match(header)
            if span is None:
                return  # reported as not a hunk header
            # A hunk that adds the header and nothing else must add it right after the first line;
            # only one hunk of a diff can.
            if rule.header and added == [rule.header]:
                if span.group(1, 3) != ("1", "2"):
                    bad.append(f"{header}: the header is not added as the second line")
            elif len(removed) != len(added) or not removed:
                bad.append(f"{header}: not a replacement in place")
            else:
                bad.extend(
                    f"{header}: +{new} does not redact -{old}"
                    for old, new in zip(removed, added)
                    if not _redacts(rule.redact, old, new)
                )
            return
        index = next(
            (i for i, c in enumerate(unused) if matches(c, where, touched, removed, added)), None
        )
        if index is None:
            bad.append(f"{header}: not one of the allowed changes")
        else:
            del unused[index]

    for line in diff.split("\n")[:-1]:
        if line.startswith("@@"):
            end_hunk()
            match = _HUNK.match(line)
            # (old start, old count, new start, new count); a missing count is 1.
            span = [int(n) if n is not None else 1 for n in match.groups()] if match else None
            where = _hunk_regions(base_regions, head_regions, span) if span else set()
            touched = _hunk_touches(base_regions, head_regions, span) if span else None
            hunk = (line, where, touched, [], [])
            if span is None:
                bad.append(f"{line}: not a hunk header")
            elif rule.lines and span[:2] != span[2:]:
                bad.append(f"{line}: not a replacement in place")
        elif hunk is None:
            if not line.startswith(_PLAIN_HEADER):
                bad.append(line)
        elif line[:1] in ("+", "-"):
            hunk[3 if line[0] == "-" else 4].append(line[1:])
            if rule.lines and not any(p.fullmatch(line[1:]) for p in patterns):
                bad.append(line)
        else:
            bad.append(line)  # "\ No newline at end of file", or anything unexpected
    end_hunk()
    return bad


def _created_differently(
    rule: Rule, repo_dir: str, base_ref: str, head_ref: str, path: str
) -> list[str]:
    """How ``base_ref...head_ref`` differs from creating ``path`` as the blob ``rule`` names."""
    merge_base = run_git(repo_dir, "merge-base", base_ref, head_ref).stdout.decode().strip()
    bad = []

    def entry(rev: str) -> str:
        found = run_git(
            repo_dir, "ls-tree", "-z", "--full-name", rev, "--", f":(top,literal){path}"
        )
        return found.stdout.decode("utf-8", errors="replace")

    if entry(merge_base):
        bad.append(f"{path} exists at the merge base")
    at_head = entry(head_ref)
    if at_head != f"100644 blob {rule.blob}\t{path}\0":
        bad.append(f"{path} at the head is not the regular file {rule.blob}: {at_head!r}")
    return bad


def check(task: str, revision_range: str, repo_dir: str = ".", repo: str | None = None) -> int:
    """Print the check of ``revision_range`` for ``task``; return the exit status."""
    if ".." not in revision_range or "..." in revision_range:
        raise ValueError(f"expected <base>..<head>, got {revision_range!r}")
    base_ref, head_ref = revision_range.split("..", 1)
    if not base_ref or not head_ref:
        raise ValueError(f"both ends of the range are required, got {revision_range!r}")
    repo = repo or detect_repo(repo_dir)
    base, _ = parse_task(task)
    if base not in OWNERSHIP:
        print(f"FAIL: {task} has no entry in the ownership table")
        return 1

    status = 0
    # The work items landed at <base>, read once a rule naming its work item matches a path.
    landed: set[str] | None = None
    changed = git(
        repo_dir,
        "diff",
        "--name-only",
        "-z",
        "--no-renames",
        "--no-relative",
        "--ignore-submodules=none",
        f"{base_ref}...{head_ref}",
    ).split("\0")
    review = []
    for path in filter(None, changed):
        rule = match_rule(task, path, repo)
        if rule is not None and rule.item and landed is None:
            landed = landed_tasks(repo_dir, base_ref)
        if rule is None:
            others = ", ".join(name for name, _ in owners(path, repo)) or "nobody"
            print(f"OUT OF BOUNDS  {path}  (owned by {others})")
            status = 1
        elif rule.item and rule.item in (landed or ()):
            print(
                f"OUT OF BOUNDS  {path}  (the exception for {rule.item} is spent: {rule.item}"
                " has landed at the base)"
            )
            status = 1
        elif (rule.lines or rule.changes or rule.redact or rule.blob) and (
            bad := disallowed_lines(rule, repo_dir, base_ref, head_ref, path)
        ):
            print(
                f"OUT OF BOUNDS  {path}  (the exception allows {rule.note}; also changed: {bad[0]!r})"
            )
            status = 1
        elif rule.kind != "own":
            review.append(f"{rule.kind.upper():<9}  {path}  -- {rule.note}")
    for line in review:
        print(f"CHECK BY HAND  {line}")

    commits = trailer_log(repo_dir, f"{base_ref}..{head_ref}")
    if not commits:
        print(f"NO COMMITS     in {revision_range}")
        status = 1
    for record in commits:
        sha = record.sha[:12]
        if record.readable:
            problem = f"is not written as the exact line '{TRAILER}: <work item>'"
        else:
            problem = f"does not count: {UNREADABLE}"
        for value in record.malformed:
            print(f"BAD TRAILER    {sha}  {TRAILER} line with {value!r} {problem}")
            status = 1
        if not record.malformed and record.values != [task]:
            found = ", ".join(record.values) or "no trailer"
            print(f"BAD TRAILER    {sha}  has {found}, expected {TRAILER}: {task}")
            status = 1
    print(f"{'PASS' if status == 0 else 'FAIL'}: {task} {revision_range} ({repo})")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, help="the work item, e.g. WT-51")
    parser.add_argument(
        "range", help="<base>..<head>, e.g. fork/feat/engine_integration-stage03..HEAD"
    )
    parser.add_argument("--repo-dir", default=".", help="the checkout to check")
    parser.add_argument("--repo", choices=(TRTLLM, DS), help="default: detected from the checkout")
    args = parser.parse_args(argv)
    try:
        return check(args.task, args.range, args.repo_dir, args.repo)
    except ValueError as error:
        print(f"FAIL: {error}")
    except OSError as error:
        print(f"FAIL: {type(error).__name__}: {error}")
    except subprocess.CalledProcessError as error:
        print(f"FAIL: {git_error(error)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

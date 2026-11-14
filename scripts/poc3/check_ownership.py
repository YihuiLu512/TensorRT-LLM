#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Check that a PoC-S3 work item changes only the paths it owns.

Every path changed in ``<base>...<head>`` must match one of the work item's
rules in :data:`OWNERSHIP`, every commit in ``<base>..<head>`` must carry
exactly one ``PoC3-Task`` line, reading exactly ``PoC3-Task: <work item>`` (read
as ``poc3_common`` describes), and the range must hold at least one commit. A
fix (``WT-xx-fixN``) owns what WT-xx owns, plus the per-item files named after
the fix itself; WT-xx and its other fixes do not own those. The gate files
(``scripts/poc3/`` except the invariant registry and ``wt54_*``) can only be
changed by WT-30 and its fixes, whatever the table says.

A rule is ``own`` (no other item owns the path; another item may change it only
under a ``shared`` or ``exception`` rule of its own), ``shared`` (this item may
change only the part the note names; the rest of the file belongs to its owner,
which may be another item or code outside PoC-S3) or ``exception`` (a one-off
allowance the note describes). An ``exception`` may also limit what it allows,
in one of three forms (patterns are regular expressions in which ``\d`` and
``\w`` match ASCII characters only):

- ``lines``: the range may only replace lines of that file in place, each
  removed and each added line matching one of the patterns in full;
- ``changes``: each hunk of the range's diff (``-U0``) must be one of the
  listed changes, each listed change used at most once: in the top-level
  statement the change names, with the same number of removed and of added
  lines, each matching its counterpart in the change in full, in order;
- ``within``: each hunk must lie in one of the named top-level statements,
  whatever it changes.

A hunk lies in a top-level statement of the file as it is at the merge base:
a ``def`` or ``class`` (from its first decorator), an assignment to a name, or
any import (named ``import``), up to the line before the next top-level
statement. A hunk that removes lines lies where those lines are; one that only
adds lines lies where the lines before and after it are. With any of the three forms
nothing else about the file may change (its mode or final newline, for
example); otherwise the path is out of bounds. Paths changed under
``shared`` and ``exception`` rules are listed for a reviewer to check by hand.
Test files that already exist at the base are ``shared`` even for the item they
belong to: tasks.md §0.3 lets existing assertions change only for the flips it
names, so changes there are listed too.
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
from typing import NamedTuple, Optional

from poc3_common import TRAILER, UNREADABLE, git, git_error, parse_task, run_git, trailer_log

TRTLLM = "trtllm"
DS = "ds"


class Change(NamedTuple):
    """A hunk an exception allows."""

    removed: tuple[str, ...]  # patterns of its removed lines, in order
    added: tuple[str, ...]  # patterns of its added lines, in order
    within: Optional[str]  # the top-level statement it lies in (see _regions), or anywhere


@dataclass(frozen=True)
class Rule:
    pattern: str
    kind: str = "own"
    note: str = ""
    repo: str = TRTLLM
    # For an exception: regular expressions, one of which every changed line must match in full,
    # for lines replaced in place ...
    lines: tuple[str, ...] = ()
    # ... or the only hunks allowed, each used at most once ...
    changes: tuple[Change, ...] = ()
    # ... or the top-level statements every hunk must lie in, whatever it changes.
    within: tuple[str, ...] = ()


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
    within: tuple[str, ...] = (),
) -> Rule:
    if sum(map(bool, (lines, changes, within))) > 1:
        raise ValueError(f"{pattern}: an exception lists lines, changes or within, not two")
    return Rule(pattern, "exception", note, repo, lines, changes, within)


def _change(within: str, removed: tuple[str, ...] = (), added: tuple[str, ...] = ()) -> Change:
    """A hunk in ``within`` that removes exactly the lines ``removed`` and adds exactly ``added``."""
    return Change(
        tuple(re.escape(line) for line in removed),
        tuple(re.escape(line) for line in added),
        within,
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
# The registry of lines check_invariants.py lets CS-20 add.
INVARIANT_ADDITIONS = "scripts/poc3/invariant_additions.yaml"
# Gate files only WT-30 and its fixes may change (see the module docstring).
GATE_FILES = ("scripts/poc3/**", f"!{INVARIANT_ADDITIONS}", "!scripts/poc3/wt54_*")
EXISTING_TESTS = "add tests; existing assertions unchanged (tasks.md §0.3)"


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


# Built from each card's change-scope row (改动范围) and the shared-file table of tasks.md §0.3.
# Where a card leaves a file name to its work item, this table fixes one (direct_replay*.py,
# cpu_disagg_chain*.py, check_case_lists.py, handoff*.py, wt54_*, records/<item>/, and the ds
# scripts dev/openengine_disagg* and dev/openengine_stress*).
OWNERSHIP: dict[str, list[Rule]] = {
    "WT-30": _own(*GATE_FILES)
    + [_exception(INVARIANT_ADDITIONS, "only creating it with an empty list of additions")],
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
    "WT-57": _own(f"{APPS}/_test_openengine_disagg_*.py"),
    "WT-60": [],
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
        if rule.repo != repo:
            continue
        if rule.pattern.startswith("!"):
            if matched is not None and _regex(rule.pattern[1:]).match(path):
                matched = None
            continue
        if matched is None and _regex(rule.pattern).match(path):
            matched = rule
    return matched


def owners(path: str, repo: str = TRTLLM) -> list[tuple[str, Rule]]:
    """Every work item that may change ``path``, with the rule that allows it."""
    found = []
    for name in OWNERSHIP:
        rule = match_rule(name, path, repo)
        if rule is not None:
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


def _hunk_regions(regions: list[str | None], span: list[int]) -> set[str]:
    """The named statements a hunk with ``span`` (old start, old count, ...) lies in.

    One that removes lines lies in the statement of all of them, if they share
    one; one that only adds lines lies in the statements of the lines before
    and after it.
    """
    start, count = span[0], span[1]
    if count:
        names = set(regions[start - 1 : start - 1 + count]) if start >= 1 else {None}
        names = names if len(names) == 1 and start - 1 + count <= len(regions) else set()
    else:
        names = {regions[i] for i in (start - 1, start) if 0 <= i < len(regions)}
    return {name for name in names if name is not None}


def _base_regions(repo_dir: str, base_ref: str, head_ref: str, path: str) -> list[str | None]:
    """:func:`_regions` of ``path`` at the merge base; empty when it cannot be read or parsed."""
    merge_base = run_git(repo_dir, "merge-base", base_ref, head_ref).stdout.decode().strip()
    found = run_git(repo_dir, "show", f"{merge_base}:{path}", check=False)
    if found.returncode != 0:
        return []
    try:
        return _regions(found.stdout.decode())
    except (SyntaxError, ValueError, RuntimeError, MemoryError):
        # ValueError: not UTF-8, a NUL byte or a lone carriage return; RuntimeError
        # (RecursionError) and MemoryError: nested too deep to parse.
        return []


def disallowed_lines(
    rule: Rule, repo_dir: str, base_ref: str, head_ref: str, path: str
) -> list[str]:
    """What ``base_ref...head_ref`` changes in ``path`` beyond the lines ``rule`` allows.

    Empty when the diff changes nothing but lines and, with ``rule.lines``,
    every hunk replaces lines in place (as many removed as added, at the same
    line number) and every removed and added line matches one of
    ``rule.lines``; with ``rule.changes``, every hunk is one of the changes,
    none used twice; with ``rule.within``, every hunk lies in one of the named
    statements.
    """
    # Bytes, not text mode: universal newlines would hide a CRLF line ending. --text: a file git
    # takes for binary (core.bigFileThreshold, a -diff attribute) is still diffed by line.
    diff = run_git(
        repo_dir,
        "diff",
        "-U0",
        "--inter-hunk-context=0",
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
    unused = [
        (
            [re.compile(p, re.ASCII) for p in change.removed],
            [re.compile(p, re.ASCII) for p in change.added],
            change.within,
        )
        for change in rule.changes
    ]
    regions = _base_regions(repo_dir, base_ref, head_ref, path) if not rule.lines else []
    bad = []
    # The hunk being read: its header, the statements it lies in, its removed and added lines.
    hunk: tuple[str, set[str], list[str], list[str]] | None = None

    def matches(change: tuple, where: set[str], removed: list[str], added: list[str]) -> bool:
        old, new, within = change
        return (
            within in where
            and len(old) == len(removed)
            and len(new) == len(added)
            and all(p.fullmatch(line) for p, line in zip(old, removed))
            and all(p.fullmatch(line) for p, line in zip(new, added))
        )

    def end_hunk() -> None:
        if hunk is None or not rule.changes:
            return
        header, where, removed, added = hunk
        index = next((i for i, c in enumerate(unused) if matches(c, where, removed, added)), None)
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
            where = _hunk_regions(regions, span) if span else set()
            hunk = (line, where, [], [])
            if span is None:
                bad.append(f"{line}: not a hunk header")
            elif rule.lines and span[:2] != span[2:]:
                bad.append(f"{line}: not a replacement in place")
            elif rule.within and not where & set(rule.within):
                bad.append(f"{line}: not in {', '.join(rule.within)}")
        elif hunk is None:
            if not line.startswith(_PLAIN_HEADER):
                bad.append(line)
        elif line[:1] in ("+", "-"):
            hunk[2 if line[0] == "-" else 3].append(line[1:])
            if rule.lines and not any(p.fullmatch(line[1:]) for p in patterns):
                bad.append(line)
        else:
            bad.append(line)  # "\ No newline at end of file", or anything unexpected
    end_hunk()
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
        if rule is None:
            others = ", ".join(name for name, _ in owners(path, repo)) or "nobody"
            print(f"OUT OF BOUNDS  {path}  (owned by {others})")
            status = 1
        elif (rule.lines or rule.changes or rule.within) and (
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

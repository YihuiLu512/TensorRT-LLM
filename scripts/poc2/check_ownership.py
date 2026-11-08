#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that a PoC-S2 work item changes only the paths it owns.

Every path changed in ``<base>...<head>`` must match one of the work item's
rules in :data:`OWNERSHIP`, and every commit in ``<base>..<head>`` must carry
the trailer ``PoC2-Task: <work item>``. A fix (``WT-xx-fixN``) owns what WT-xx
owns.

A rule is ``own`` (this item owns the path; another item may change it only
under a ``shared`` or ``exception`` rule of its own), ``shared`` (this item may
change only the part the note names; the rest of the file belongs to its owner,
which may be another item or code outside PoC-S2) or ``exception`` (a one-off
allowance the note describes). Paths changed under ``shared`` and ``exception``
rules are listed for a reviewer to check by hand.
A pattern starting with ``!`` excludes paths from the rules before it.

Usage::

    python3 scripts/poc2/check_ownership.py --task WT-10 fork/feat/engine_integration..HEAD

Exit status: 0 when every path and commit is in bounds, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from poc2_common import TRAILER, git, parse_task, trailers

TRTLLM = "trtllm"
GLM = "glm"


@dataclass(frozen=True)
class Rule:
    pattern: str
    kind: str = "own"
    note: str = ""
    repo: str = TRTLLM


def _own(*patterns: str, repo: str = TRTLLM) -> list[Rule]:
    return [Rule(p, "own", repo=repo) for p in patterns]


def _shared(pattern: str, note: str) -> Rule:
    return Rule(pattern, "shared", note)


POC = "tensorrt_llm/serve/openengine_poc"
CHAIN = "tests/integration/defs/openengine_chain"
CLIENT_TESTS = "tests/unittest/grpc/openengine_client"
OE_SERVER = "tensorrt_llm/grpc/openengine"
OE_SERVER_TESTS = "tests/unittest/grpc/openengine"


def _per_item(name: str) -> list[Rule]:
    """The files every work item owns under its own name."""
    rules = _own(
        f"{CHAIN}/expectations/{name}.py",
        f"{CHAIN}/verified/{name}.yaml",
        f"{CHAIN}/smoke/{name}.txt",
    )
    match = re.fullmatch(r"WT-(\d\d)", name)
    if match:
        number = match.group(1)
        rules += _own(
            f"tests/unittest/serve/openengine_poc/wt{number}/**",
            f"tests/unittest/llmapi/apps/_test_openengine_wt{number}*.py",
        )
    return rules


OWNERSHIP: dict[str, list[Rule]] = {
    "WT-01": _own(
        "tensorrt_llm/grpc/openengine_conventions.py",
        f"{POC}/engine_client.py",
        f"{POC}/contracts.py",
        "tests/unittest/serve/openengine_poc/fakes.py",
        "scripts/poc2/**",
    )
    + [
        _shared(f"{POC}/__init__.py", "lazy exports only"),
        _shared("tensorrt_llm/serve/openai_server.py", "wiring points only; WT-13 owns it"),
        _shared("tensorrt_llm/commands/serve.py", "wiring points only; WT-10 owns it"),
        Rule(
            f"{CLIENT_TESTS}/fake_openengine.py",
            "exception",
            "only to switch WT-02's local convention constants to openengine_conventions",
        ),
        Rule(
            f"{CHAIN}/conventions.py",
            "exception",
            "only the note on its local copies of the convention constants",
        ),
        Rule(
            f"{OE_SERVER}/capabilities.py",
            "exception",
            "only to take WT-04's convention names and tokenizer fingerprint from openengine_conventions",
        ),
    ],
    "WT-02": _own(f"{CLIENT_TESTS}/fake_openengine.py", f"{CLIENT_TESTS}/test_fake_openengine*.py")
    + [
        _shared(
            f"{CLIENT_TESTS}/test_openengine_poc_serve.py",
            "only to run the PoC-S1 request-equivalence test against the gRPC fake; WT-10 owns it",
        )
    ],
    "WT-03": _own(
        f"{CHAIN}/**",
        f"!{CHAIN}/expectations/WT-*.py",
        f"!{CHAIN}/verified/WT-*.yaml",
        f"!{CHAIN}/smoke/WT-*.txt",
    )
    + [
        _shared(
            "tests/unittest/llmapi/apps/openai_server.py", "the chain hook of RemoteOpenAIServer"
        ),
        _shared("tests/integration/defs/examples/serve/test_serve.py", "the H-2 chain hook"),
        _shared(
            "tests/integration/defs/kv_cache/test_prefix_aware_scheduling.py", "the H-2 chain hook"
        ),
        _shared("tests/integration/defs/accuracy/test_glm53_flash.py", "the H-2 chain hook"),
    ],
    "WT-04": _own(
        f"{OE_SERVER}/control.py",
        f"{OE_SERVER}/capabilities.py",
        f"{OE_SERVER_TESTS}/test_control.py",
        f"{OE_SERVER_TESTS}/test_capability_conformance.py",
    )
    + [Rule(f"{OE_SERVER}/servicer.py", "exception", "only the Generate counter calls")],
    "WT-05": _own(
        f"{OE_SERVER}/request_mapping.py",
        f"{OE_SERVER}/servicer.py",
        f"{OE_SERVER}/lora_registry.py",
        f"{OE_SERVER_TESTS}/test_servicer.py",
        f"{OE_SERVER_TESTS}/test_request_mapping*.py",
        f"{OE_SERVER_TESTS}/test_lora_registry*.py",
        # WT-05-fix1 verifies F-44-chain, the row of WT-05's chain cases.
        f"{CHAIN}/verified/WT-05-fix1.yaml",
    )
    + [
        _shared(
            f"{OE_SERVER}/control.py",
            "only taking the request keys, priority, LoRA and max_loras from the request"
            " capabilities, max_num_sequences from max_num_sequences_for, and delegating"
            " the LoRA RPCs; WT-04 owns it",
        ),
        Rule(
            f"{OE_SERVER_TESTS}/test_control.py",
            "exception",
            "only the expected request-key values, the comments and docstrings that"
            " state what Generate accepts, and (WT-05-fix1) the _inference and _servicer"
            " helpers giving Control and the inference servicer one engine, with the cases"
            " that change because of it, and the expected CONVENTIONS_VERSION",
        ),
        Rule(
            f"{OE_SERVER_TESTS}/test_capability_conformance.py",
            "exception",
            "only (WT-05-fix1) the LoRA probe registering its adapter with LoadLora first,"
            " and the servicer argument it passes to _accepted",
        ),
        Rule(
            "tensorrt_llm/executor/result.py",
            "exception",
            "only (WT-05-fix1) streaming detokenization of several candidates: an id"
            " cursor per output, a flush at each output's finish and when a stop string"
            " ends the request, so no text is decoded twice or follows an output's finish,"
            " and a stop-string check of the outputs after the one that ends the request",
        ),
        Rule(
            "tests/integration/test_lists/test-db/l0_cpu.yml",
            "exception",
            "only (WT-05-fix1) listing test_request_mapping_features.py and test_lora_registry.py",
        ),
        Rule(
            "tensorrt_llm/grpc/openengine_conventions.py",
            "exception",
            "only (WT-05-fix1) CONVENTIONS_VERSION",
        ),
        Rule(
            "tests/unittest/serve/openengine_poc/wt01/test_poc2_wt01_conventions.py",
            "exception",
            "only (WT-05-fix1) the expected CONVENTIONS_VERSION",
        ),
        Rule(
            f"{OE_SERVER}/capabilities.py",
            "exception",
            "only the comment on MAX_NUM_SEQUENCES",
        ),
        Rule(
            f"{OE_SERVER}/README.md",
            "exception",
            "only the sentences on priority metadata, lora_name and the LoRA RPCs",
        ),
        Rule(
            f"{POC}/mapping.py",
            "exception",
            "only (WT-05-fix1) sending kv_hints to the engine as the envelope's JSON text",
        ),
        Rule(
            "tests/unittest/serve/openengine_poc/wt11/test_poc2_wt11_mapping.py",
            "exception",
            "only (WT-05-fix1) the kv_hints cases, which expect JSON text",
        ),
        Rule(
            "tests/unittest/serve/openengine_poc/wt11/test_poc2_wt11_no_silent_drop.py",
            "exception",
            "only (WT-05-fix1) the kv_hints probe reading JSON text",
        ),
        Rule(
            f"{CLIENT_TESTS}/fake_openengine.py",
            "exception",
            "only (WT-05-fix1) the request check taking kv_hints as JSON text or an object",
        ),
        Rule(
            "tests/unittest/serve/openengine_poc/wt01/test_poc2_wt01_scripts.py",
            "exception",
            "only (WT-05-fix1) the owners it expects of fake_openengine.py",
        ),
        Rule(
            "docs/source/features/kv-hints.md",
            "exception",
            "only the known-limitations paragraphs on gRPC transport",
        ),
        Rule(
            "scripts/poc2/check_ownership.py",
            "exception",
            "only the WT-05 rules",
        ),
    ],
    "WT-06": _own("python/glm5_serve/openengine/**", "tests/openengine/**", repo=GLM)
    + _own("tests/unittest/llmapi/apps/_test_openengine_glm53.py"),
    "WT-07": _own(
        "csrc/**",
        "tests/test_min_tokens*.py",
        "tests/test_min_tokens*.cpp",
        "tests/test_ban_logits.py",
        repo=GLM,
    )
    + [
        Rule(
            "python/glm5_serve/engine_api.py",
            "exception",
            "only the MockEngine's min_tokens masking, the masked-id limit constant and the"
            " min_tokens API notes",
            repo=GLM,
        ),
        Rule(
            "python/glm5_serve/openengine/servicer.py",
            "exception",
            "only the module docstring's min_tokens note; WT-06 owns the file",
            repo=GLM,
        ),
        Rule(
            "tests/openengine/test_openengine_server.py",
            "exception",
            "only the min_tokens tests and their imports; WT-06 owns the file",
            repo=GLM,
        ),
        Rule(
            "README.md", "exception", "only the min_tokens notes and test-table entries", repo=GLM
        ),
        Rule("scripts/poc2/check_ownership.py", "exception", "only WT-07's entry in OWNERSHIP"),
    ],
    "WT-09": _own("tests/integration/defs/perf/openengine_arms.py")
    + [
        _shared("tests/integration/defs/perf/test_perf_sanity.py", "the arm hook only"),
        _shared("tests/integration/defs/perf/utils.py", "the arm hook only"),
    ],
    "WT-10": _own(
        f"{POC}/adapter.py",
        f"{POC}/launch.py",
        f"{POC}/engine_client_impl.py",
        f"{POC}/facade.py",
        "tensorrt_llm/grpc/openengine_client/**",
        "tensorrt_llm/commands/serve.py",
        "tests/unittest/api_stability/references/trtllm_serve_cli.yaml",
        f"{CLIENT_TESTS}/**",
        f"!{CLIENT_TESTS}/fake_openengine.py",
        f"!{CLIENT_TESTS}/test_fake_openengine*.py",
    )
    + [
        _shared(f"{POC}/__init__.py", "exports of the facade"),
        Rule(
            f"{POC}/contracts.py",
            "exception",
            "only contract v4: FrontendLaunchConfig's allow_newer_schema and control_timeout,"
            " and the default launch passing them and report_addr on",
        ),
        Rule(
            "tests/unittest/serve/openengine_poc/wt01/test_poc2_wt01_contracts.py",
            "exception",
            "only the contract v4 version and the default launch's arguments",
        ),
        Rule(
            "scripts/poc2/check_ownership.py",
            "exception",
            "only the contract v4 exceptions of WT-10",
        ),
    ],
    "WT-11": _own(f"{POC}/mapping.py", f"{POC}/field_classes.py")
    # WT-11-fix2 (spec 06, CAP-*): sampling capabilities as accepted values, on both sides.
    + [
        Rule(path, "exception", f"WT-11-fix2 only: {what}")
        for path, what in (
            (
                "tensorrt_llm/grpc/openengine_conventions.py",
                "trtllm.params and its parsing; no built-in capability table",
            ),
            (
                f"{POC}/contracts.py",
                "contract v6 and the default map_request's explicit values checked against them",
            ),
            (
                f"{POC}/facade.py",
                "priority sent only when given; the no-conventions warning names no built-in table;"
                " a malformed capability declaration fails startup or loses a restarted engine",
            ),
            ("tensorrt_llm/serve/openai_server.py", "the request's priority only when it set one"),
            ("tensorrt_llm/sampling_params.py", "every field name a request supplied is recorded"),
            (
                f"{OE_SERVER}/request_mapping.py",
                "the accepted sampling values, declared and checked by Generate",
            ),
            (f"{OE_SERVER}/control.py", "reporting the accepted sampling values"),
            (
                f"{OE_SERVER}/capabilities.py",
                "the note on the conventions; no aliases of the version 2 sampling keys",
            ),
            (
                "tests/unittest/serve/openengine_poc/wt01/test_poc2_wt01_contracts.py",
                "the tests of the contract v6 changes",
            ),
            (
                "tests/unittest/serve/openengine_poc/wt01/test_poc2_wt01_conventions.py",
                "the tests of trtllm.params",
            ),
            (f"{OE_SERVER_TESTS}/test_control.py", "the tests of the reported sampling values"),
            (
                f"{OE_SERVER_TESTS}/test_request_mapping_features.py",
                "the tests of the accepted sampling values",
            ),
            (
                "tests/unittest/grpc/openengine_client/test_openengine_poc_serve.py",
                "the PoC-S1 mapping's and the facade's priority tests",
            ),
            ("scripts/poc2/check_ownership.py", "these WT-11-fix2 exceptions"),
        )
    ]
    + [
        Rule(
            f"{POC}/{path}",
            "exception",
            "only the request-data contract: a thinking budget and logit_bias as request data",
        )
        for path in ("contracts.py", "facade.py")
    ]
    + [
        Rule(path, "exception", f"only the request-data contract: {what}")
        for path, what in (
            (
                "tensorrt_llm/serve/openai_server.py",
                "no thinking-budget processor or bias tensor for a generator that takes"
                " request data, and logit_bias, the reasoning parser and the tokenizer passed to it",
            ),
            ("tensorrt_llm/serve/openai_protocol.py", "the switch that does not materialize"),
            ("tensorrt_llm/serve/responses_utils.py", "the thinking-budget processor switch"),
            (
                "tests/unittest/serve/openengine_poc/wt01/test_poc2_wt01_contracts.py",
                "the version",
            ),
            ("scripts/poc2/check_ownership.py", "these WT-11 exceptions"),
        )
    ]
    + [
        Rule(
            "tests/unittest/grpc/openengine_client/test_openengine_poc_serve.py",
            "exception",
            "only its test profile declaring no stop strings: its tests check the frontend's"
            " matching, and the request mapping sends stop strings to an engine that declares them",
        ),
    ],
    "WT-12": _own(
        f"{POC}/response.py",
        "tensorrt_llm/serve/detokenize.py",
        "tests/unittest/llmapi/apps/test_serve_detokenize.py",
    )
    + [
        Rule(
            f"{POC}/contracts.py",
            "exception",
            "only contract v3: engine_stop_texts of create_response_assembler and"
            " PostprocPool.open, and DetokOptions.engine_stop",
        ),
        Rule(
            "tests/unittest/serve/openengine_poc/wt01/test_poc2_wt01_contracts.py",
            "exception",
            "only the contract v3 version, the engine_stop_texts tests and keeping the"
            " slot-default test independent of the slot modules that exist",
        ),
        Rule(
            "scripts/poc2/check_ownership.py",
            "exception",
            "only the exceptions of WT-12",
        ),
        Rule(
            f"{CLIENT_TESTS}/test_openengine_poc_serve.py",
            "exception",
            "only the completion_tokens of the stop-string case: the stop token matched on its id"
            " is not counted, as with LLM",
        ),
    ],
    "WT-13": _own(
        f"{POC}/postproc_pool.py",
        "tensorrt_llm/serve/openai_server.py",
    )
    + [
        Rule(
            f"{POC}/contracts.py",
            "exception",
            "only contract v2: the postprocessing pool interface, PPOpen and the default"
            " dispatch's check for the facade",
        ),
        Rule(
            "tests/unittest/serve/openengine_poc/wt01/test_poc2_wt01_contracts.py",
            "exception",
            "only the contract v2 version, slot table, postprocessing defaults, error statuses and PPOpen fields",
        ),
        Rule(
            "scripts/poc2/check_ownership.py",
            "exception",
            "only the exceptions of WT-13",
        ),
        Rule(
            f"{CLIENT_TESTS}/test_openengine_poc_serve.py",
            "exception",
            "only the cases that expected the postprocessing options refused while their slots"
            " had the default implementation",
        ),
        _shared(
            "tensorrt_llm/serve/postprocess_handlers.py", "making PostprocArgs plain data only"
        ),
        _shared("tensorrt_llm/executor/postproc_worker.py", "making PostprocArgs plain data only"),
    ],
    "WT-14": _own(f"{POC}/multi_frontend.py", "tensorrt_llm/serve/_frontend_processes.py")
    + [
        Rule(
            "tensorrt_llm/commands/serve.py",
            "exception",
            (
                "only dropping the single-frontend default of --openengine-endpoint "
                "(SV-MF-1), and the comment and --num_serve_frontends help about it"
            ),
        ),
        Rule(
            f"{CLIENT_TESTS}/test_openengine_poc_serve.py",
            "exception",
            "only the CLI tests that assumed a single OpenEngine frontend",
        ),
        Rule(
            "tests/unittest/serve/openengine_poc/wt01/test_poc2_wt01_contracts.py",
            "exception",
            "only calling the default launch_frontends directly",
        ),
        Rule(
            "scripts/poc2/feature_deps.yaml",
            "exception",
            "only adding WT-15 to the dependencies of F-39",
        ),
        Rule("scripts/poc2/check_ownership.py", "exception", "only the exceptions of WT-14"),
    ],
    "WT-15": _own(f"{POC}/metrics.py"),
    "WT-16": _own(f"{POC}/media.py", f"{POC}/lora.py")
    + [
        Rule(
            "tensorrt_llm/serve/openai_server.py",
            "exception",
            "only the chat and Responses paths' media-references hook for the OpenEngine facade",
        ),
        Rule(
            "tensorrt_llm/serve/responses_utils.py",
            "exception",
            "only the media-references hook of _create_input_tokens and request_preprocess",
        ),
        Rule(
            "tests/unittest/serve/openengine_poc/wt01/test_poc2_wt01_contracts.py",
            "exception",
            "only to aim the default media and LoRA slot tests at the defaults",
        ),
        Rule(
            f"{CLIENT_TESTS}/test_openengine_poc_serve.py",
            "exception",
            "only the media-flow, default LoRA registry and cancelled-while-starting LoRA tests, "
            "and the comment on when a request with an adapter is assembled",
        ),
        Rule(
            f"{POC}/facade.py",
            "exception",
            "only media references, unexpanded placeholders, mapping media once, LoRA checks",
        ),
        Rule(
            f"{CHAIN}/split.py",
            "exception",
            "only --video_pruning_rate to both sides",
        ),
        Rule(
            "tensorrt_llm/commands/serve.py",
            "exception",
            "only refusing --media_io_kwargs with --openengine-endpoint",
        ),
        Rule(
            "scripts/poc2/check_ownership.py",
            "exception",
            "only the exceptions of WT-16",
        ),
    ],
    "WT-20-R2": [],
    "WT-20-R3": [],
    "WT-21": [],
    "WT-22": [],
}
for _name in OWNERSHIP:
    OWNERSHIP[_name] = OWNERSHIP[_name] + _per_item(_name)


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


def match_rule(task: str, path: str, repo: str = TRTLLM) -> Rule | None:
    """The rule of ``task`` that allows changing ``path``, or None."""
    base, _ = parse_task(task)
    matched: Rule | None = None
    for rule in OWNERSHIP[base]:
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
    top = Path(git(repo_dir, "rev-parse", "--show-toplevel").strip())
    return TRTLLM if (top / "tensorrt_llm" / "__init__.py").is_file() else GLM


def check(task: str, revision_range: str, repo_dir: str = ".", repo: str | None = None) -> int:
    """Print the check of ``revision_range`` for ``task``; return the exit status."""
    if ".." not in revision_range or "..." in revision_range:
        raise ValueError(f"expected <base>..<head>, got {revision_range!r}")
    base_ref, head_ref = revision_range.split("..", 1)
    head_ref = head_ref or "HEAD"
    repo = repo or detect_repo(repo_dir)
    parse_task(task)

    status = 0
    changed = git(
        repo_dir, "diff", "--name-only", "--no-renames", f"{base_ref}...{head_ref}"
    ).split("\n")
    review = []
    for path in filter(None, changed):
        rule = match_rule(task, path, repo)
        if rule is None:
            others = ", ".join(name for name, _ in owners(path, repo)) or "nobody"
            print(f"OUT OF BOUNDS  {path}  (owned by {others})")
            status = 1
        elif rule.kind != "own":
            review.append(f"{rule.kind.upper():<9}  {path}  -- {rule.note}")
    for line in review:
        print(f"CHECK BY HAND  {line}")

    log = git(repo_dir, "log", "--format=%H%x00%B%x01", f"{base_ref}..{head_ref}")
    for record in filter(None, (r.strip("\n") for r in log.split("\x01"))):
        sha, _, message = record.partition("\0")
        values = trailers(message)
        if values != [task]:
            print(
                f"BAD TRAILER    {sha[:12]}  has {values or 'no trailer'}, expected {TRAILER}: {task}"
            )
            status = 1
    print(f"{'PASS' if status == 0 else 'FAIL'}: {task} {revision_range} ({repo})")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, help="the work item, e.g. WT-10")
    parser.add_argument("range", help="<base>..<head>, e.g. fork/feat/engine_integration..HEAD")
    parser.add_argument("--repo-dir", default=".", help="the checkout to check")
    parser.add_argument("--repo", choices=(TRTLLM, GLM), help="default: detected from the checkout")
    args = parser.parse_args(argv)
    return check(args.task, args.range, args.repo_dir, args.repo)


if __name__ == "__main__":
    sys.exit(main())

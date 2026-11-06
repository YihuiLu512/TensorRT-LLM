#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that a PoC-S2 work item changes only the paths it owns.

Every path changed in ``<base>...<head>`` must match one of the work item's
rules in :data:`OWNERSHIP`, and every commit in ``<base>..<head>`` must carry
the trailer ``PoC2-Task: <work item>``. A fix (``WT-xx-fixN``) owns what WT-xx
owns.

A rule is ``own`` (only this item changes the path), ``shared`` (this item may
change only the part the note names; the rest of the file belongs to its owner,
which may be another item or code outside PoC-S2) or ``exception`` (a one-off
allowance the note describes). Paths changed under
``shared`` and ``exception`` rules are listed for a reviewer to check by hand.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from poc2_common import TRAILER, git, parse_task, trailers  # noqa: E402

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
            f"{CLIENT_TESTS}/fake_engine.py",
            "exception",
            "only to switch WT-02's local convention constants to openengine_conventions",
        ),
        Rule(
            f"{OE_SERVER}/control.py",
            "exception",
            "only to switch WT-04's local convention constants to openengine_conventions",
        ),
        Rule(
            f"{OE_SERVER}/capabilities.py",
            "exception",
            "only to switch WT-04's local convention constants to openengine_conventions",
        ),
    ],
    "WT-02": _own(f"{CLIENT_TESTS}/fake_engine.py", f"{CLIENT_TESTS}/test_fake_engine*.py")
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
    ],
    "WT-04": _own(
        f"{OE_SERVER}/control.py",
        f"{OE_SERVER}/capabilities.py",
        f"{OE_SERVER_TESTS}/test_control.py",
        f"{OE_SERVER_TESTS}/test_capability_conformance.py",
    ),
    "WT-05": _own(
        f"{OE_SERVER}/request_mapping.py",
        f"{OE_SERVER}/servicer.py",
        f"{OE_SERVER}/lora_registry.py",
        f"{OE_SERVER_TESTS}/test_servicer.py",
        f"{OE_SERVER_TESTS}/test_request_mapping*.py",
        f"{OE_SERVER_TESTS}/test_lora_registry*.py",
    )
    + [_shared(f"{OE_SERVER}/control.py", "only the LoRA RPC delegation, after WT-04 landed")],
    "WT-06": _own("python/glm5_serve/openengine/**", "tests/openengine/**", repo=GLM)
    + _own("tests/unittest/llmapi/apps/_test_openengine_glm53.py"),
    "WT-07": _own("csrc/**", "tests/test_min_tokens*.py", repo=GLM),
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
        f"!{CLIENT_TESTS}/fake_engine.py",
        f"!{CLIENT_TESTS}/test_fake_engine*.py",
    )
    + [_shared(f"{POC}/__init__.py", "exports of the facade")],
    "WT-11": _own(f"{POC}/mapping.py", f"{POC}/field_classes.py"),
    "WT-12": _own(
        f"{POC}/response.py",
        "tensorrt_llm/serve/detokenize.py",
        "tests/unittest/llmapi/apps/test_serve_detokenize.py",
    ),
    "WT-13": _own(
        f"{POC}/postproc_pool.py",
        "tensorrt_llm/serve/openai_server.py",
    )
    + [
        _shared(
            "tensorrt_llm/serve/postprocess_handlers.py", "making PostprocArgs plain data only"
        ),
        _shared("tensorrt_llm/executor/postproc_worker.py", "making PostprocArgs plain data only"),
    ],
    "WT-14": _own(f"{POC}/multi_frontend.py", "tensorrt_llm/serve/_frontend_processes.py"),
    "WT-15": _own(f"{POC}/metrics.py"),
    "WT-16": _own(f"{POC}/media.py", f"{POC}/lora.py"),
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

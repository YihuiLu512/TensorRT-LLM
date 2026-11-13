#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the PoC-S3 design invariants against the frozen base.

PoC-S3 puts OpenEngine worker front ends behind the existing disaggregated
orchestrator without changing the orchestrator. Three checks hold that line:

1. **Whole files unchanged**: the orchestrator's seven main files are byte for
   byte the same as at the base (what ``git diff --stat <base> -- <files>``
   printing nothing means).
2. **Definitions unchanged**: the two conversion functions of
   ``openai_protocol.py`` and the ``disaggregated`` / ``disaggregated_mpi_worker``
   subcommands of ``commands/serve.py`` have the same source text as at the
   base. Each definition is located with ``ast`` and compared from its first
   decorator to its last line, so moving it inside the file is fine but any
   change to its decorators, signature or body is not.
3. **Registered additions only**: the HTTP ``DisaggregatedParams`` class of
   ``openai_protocol.py`` and the whole of ``disagg_auth.py`` may only gain the
   lines listed in ``invariant_additions.yaml``: the base text must be a
   subsequence of the current text, and the extra lines must all be registered.
   Every registry entry must belong to change item CS-20.

Usage::

    python3 scripts/poc3/check_invariants.py                      # HEAD against the frozen base
    python3 scripts/poc3/check_invariants.py --head <rev>
    python3 scripts/poc3/check_invariants.py --working-tree       # files on disk

Exit status: 0 when all three checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import yaml

# fork/feat/engine_integration when PoC-S3 started; recorded in the stage STATUS by WT-30.
FROZEN_BASE = "2f22a98d6e8b32ed185bcdfe051ef0e4405ae4e8"
REGISTRY = Path(__file__).resolve().parent / "invariant_additions.yaml"
REGISTRY_PATH = "scripts/poc3/invariant_additions.yaml"
ALLOWED_CHANGE = "CS-20"
PROTOCOL = "tensorrt_llm/serve/openai_protocol.py"
SERVE = "tensorrt_llm/commands/serve.py"
AUTH = "tensorrt_llm/serve/disagg_auth.py"

ORCHESTRATOR_FILES = (
    "tensorrt_llm/serve/openai_disagg_server.py",
    "tensorrt_llm/serve/openai_disagg_service.py",
    "tensorrt_llm/serve/router.py",
    "tensorrt_llm/serve/disagg_coordinator.py",
    "tensorrt_llm/serve/coordinator_server.py",
    "tensorrt_llm/serve/openai_client.py",
    "tensorrt_llm/llmapi/disagg_utils.py",
)
FROZEN_DEFINITIONS = (
    (PROTOCOL, "to_disaggregated_params"),
    (PROTOCOL, "to_llm_disaggregated_params"),
    (SERVE, "disaggregated"),
    (SERVE, "disaggregated_mpi_worker"),
)
# (file, region): the region is a top-level class name, or "file" for the whole file.
ADDITION_REGIONS = ((PROTOCOL, "DisaggregatedParams"), (AUTH, "file"))

Reader = Callable[[str, str], "str | None"]


def git_reader(repo: str | Path) -> Reader:
    """Read ``path`` at revision ``rev`` (None when it does not exist there); rev "" is the disk."""

    def read(rev: str, path: str) -> str | None:
        if not rev:
            file = Path(repo) / path
            return file.read_text() if file.is_file() else None
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True, text=True
        )
        return result.stdout if result.returncode == 0 else None

    return read


def definition_text(source: str, name: str, kind: type = ast.FunctionDef) -> str:
    """The source lines of the single top-level ``kind`` named ``name``, decorators included.

    Raises:
        LookupError: There is no such definition, or more than one.
    """
    nodes = [n for n in ast.parse(source).body if isinstance(n, kind) and n.name == name]
    if len(nodes) != 1:
        raise LookupError(f"expected one top-level {kind.__name__} {name!r}, found {len(nodes)}")
    node = nodes[0]
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "".join(source.splitlines(keepends=True)[first - 1 : node.end_lineno])


def extra_lines(base: list[str], head: list[str]) -> list[str] | None:
    """The lines ``head`` adds to ``base``, or None when ``base`` is not a subsequence of ``head``.

    The extra lines, as a multiset, do not depend on how ``base`` is matched.
    """
    i = 0
    for line in head:
        if i < len(base) and line == base[i]:
            i += 1
    if i != len(base):
        return None
    return list((Counter(head) - Counter(base)).elements())


def load_registry(text: str | None) -> tuple[dict[tuple[str, str], Counter[str]], list[str]]:
    """Registered lines per region, and the problems found in the registry itself."""
    registered: dict[tuple[str, str], Counter[str]] = {
        region: Counter() for region in ADDITION_REGIONS
    }
    if text is None:
        return registered, [f"registry {REGISTRY_PATH} is missing"]
    data = yaml.safe_load(text) or {}
    problems = []
    for index, entry in enumerate(data.get("additions") or []):
        where = f"registry entry {index}"
        if not isinstance(entry, dict):
            problems.append(f"{where} is not a mapping")
            continue
        region = (entry.get("file"), entry.get("region"))
        if entry.get("change") != ALLOWED_CHANGE:
            problems.append(
                f"{where} belongs to {entry.get('change')!r}; only {ALLOWED_CHANGE} may register lines"
            )
        elif region not in registered:
            problems.append(f"{where} names {region}, which is not a guarded region")
        elif not isinstance(entry.get("lines"), list) or not all(
            isinstance(x, str) for x in entry["lines"]
        ):
            problems.append(f"{where} has no list of lines")
        else:
            registered[region].update(
                line if line.endswith("\n") else line + "\n" for line in entry["lines"]
            )
    return registered, problems


def run(
    read: Reader, base: str, head: str, registry_text: str | None = None
) -> list[tuple[str, bool, str]]:
    """Every check as ``(name, passed, detail)``.

    The registry is ``registry_text`` when given, else the one in ``head``, else
    (for a head without ``scripts/poc3``, such as the frozen base) the one next
    to this script.
    """
    results = []
    for path in ORCHESTRATOR_FILES:
        old, new = read(base, path), read(head, path)
        same = old is not None and old == new
        results.append(
            (f"file {path}", same, "" if same else "differs from the base (or is missing)")
        )

    for path, name in FROZEN_DEFINITIONS:
        try:
            old_text = definition_text(read(base, path) or "", name)
            new_text = definition_text(read(head, path) or "", name)
        except (LookupError, SyntaxError) as error:
            results.append((f"def {path}:{name}", False, str(error)))
            continue
        same = old_text == new_text
        results.append((f"def {path}:{name}", same, "" if same else "source text changed"))

    source = "given"
    if registry_text is None:
        registry_text, source = read(head, REGISTRY_PATH), "head"
    if registry_text is None and REGISTRY.is_file():
        registry_text, source = REGISTRY.read_text(), str(REGISTRY)
    registered, problems = load_registry(registry_text)
    results.append(("registry", not problems, "; ".join(problems) or f"from {source}"))
    for path, region in ADDITION_REGIONS:
        label = f"additions {path}:{region}"
        try:
            old_src, new_src = read(base, path), read(head, path)
            if old_src is None or new_src is None:
                raise LookupError("file missing")
            if region == "file":
                old_text, new_text = old_src, new_src
            else:
                old_text = definition_text(old_src, region, ast.ClassDef)
                new_text = definition_text(new_src, region, ast.ClassDef)
        except (LookupError, SyntaxError) as error:
            results.append((label, False, str(error)))
            continue
        extra = extra_lines(old_text.splitlines(keepends=True), new_text.splitlines(keepends=True))
        if extra is None:
            results.append((label, False, "an existing line was changed or removed"))
            continue
        unregistered = Counter(extra) - registered[(path, region)]
        detail = "; ".join(
            f"unregistered line {line.rstrip()!r}" for line in unregistered.elements()
        )
        results.append(
            (label, not unregistered, detail or f"{len(extra)} registered line(s) added")
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=".", help="the TRT-LLM checkout")
    parser.add_argument(
        "--base", default=FROZEN_BASE, help="the frozen base (default: %(default)s)"
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--head", default="HEAD", help="the revision to check (default: HEAD)")
    where.add_argument("--working-tree", action="store_true", help="check the files on disk")
    parser.add_argument("--registry", type=Path, help="use this registry instead of the head's")
    args = parser.parse_args(argv)
    head = "" if args.working_tree else args.head
    registry_text = args.registry.read_text() if args.registry else None
    results = run(git_reader(args.repo), args.base, head, registry_text)
    for name, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    failed = [name for name, passed, _ in results if not passed]
    target = "working tree" if args.working_tree else args.head
    print(f"{'FAIL' if failed else 'PASS'}: invariants of {target} against {args.base[:12]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

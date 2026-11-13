#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the PoC-S3 design invariants against the frozen base.

PoC-S3 puts OpenEngine worker front ends behind the existing disaggregated
orchestrator without changing the orchestrator. Three checks hold that line:

1. **Whole files unchanged**: ``git diff --quiet <base> [<head>] -- <file>``
   finds no difference in any of the orchestrator's seven main files (contents,
   line endings and file mode).
2. **Definitions unchanged**: the two conversion functions of
   ``openai_protocol.py`` and the ``disaggregated`` / ``disaggregated_mpi_worker``
   subcommands of ``commands/serve.py`` have the same source text as at the
   base. Each definition is located with ``ast`` and compared from its first
   decorator to its last line, comment lines indented into its body included,
   so moving it inside the file is fine but any change to its decorators,
   signature or body is not. Nothing else in the file may bind or delete the
   name at module scope: assignments and ``del``, imports (``import *``
   included), definitions, ``for`` / ``with`` / ``except`` targets, ``match``
   captures, walrus targets evaluated at module scope (in decorators, default
   values and base classes too), and a ``global`` declaration of the name
   anywhere in the file all fail the check. Bindings made without that syntax,
   such as through ``globals()`` or ``setattr`` on the module, are not seen.
   Helpers the definitions call, and the command table that maps CLI names to
   the subcommands, are not covered.
3. **Registered additions only**: the HTTP ``DisaggregatedParams`` class of
   ``openai_protocol.py`` and the whole of ``disagg_auth.py`` may only gain the
   lines listed in ``invariant_additions.yaml``: the base text must be a
   subsequence of the current text, and the extra lines must all be registered.
   Every registry entry must belong to change item CS-20. Where a registered
   line is inserted is not checked; a reviewer reads it.

Usage::

    python3 scripts/poc3/check_invariants.py                      # HEAD against the frozen base
    python3 scripts/poc3/check_invariants.py --head <rev>
    python3 scripts/poc3/check_invariants.py --working-tree       # files on disk

Exit status: 0 when all three checks pass, 1 otherwise. A checked file or a
registry in the repository that cannot be read or parsed fails its own check;
an unknown revision or an unreadable ``--registry`` file fails the whole run
with one line. The scripts need Python 3.9 or newer.
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

# The commit PoC-S3 starts from: the PoC-S2 branch fork/feat/engine_integration at that point.
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
Differ = Callable[[str, str, str], bool]


def git_reader(repo: str | Path) -> Reader:
    """Read ``path`` at revision ``rev`` (None when it does not exist there); rev "" is the disk."""
    top = Path(
        subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    def read(rev: str, path: str) -> str | None:
        # Line endings are kept as they are, so a CRLF change is a change.
        if not rev:
            file = top / path
            return file.read_bytes().decode() if file.is_file() else None
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True
        )
        return result.stdout.decode() if result.returncode == 0 else None

    return read


def _exists(read: Reader, rev: str, path: str) -> bool:
    try:
        return read(rev, path) is not None
    except ValueError:  # there, but not UTF-8
        return True


def git_differ(repo: str | Path) -> Differ:
    """Whether ``path`` differs between ``base`` and ``head`` (head "" is the disk), by git diff."""

    def differs(base: str, head: str, path: str) -> bool:
        revs = [base, head] if head else [base]
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "--quiet",
                "--no-ext-diff",
                "--no-relative",
                *revs,
                "--",
                f":(top){path}",
            ],
            capture_output=True,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode(errors="replace").strip())
        return result.returncode == 1

    return differs


# Statement fields holding nested statements; everything else is evaluated where the statement runs.
_BLOCKS = ("body", "orelse", "finalbody", "handlers", "cases")


def _same_scope(nodes: list[ast.AST]) -> list[ast.AST]:
    """The nodes under ``nodes`` evaluated in the scope they appear in.

    A lambda's body and a comprehension's ``for`` targets have scopes of their
    own and are skipped; a lambda's default values and a comprehension's
    iterables and conditions are not (walrus targets in a comprehension bind
    in the enclosing scope).
    """
    found: list[ast.AST] = []
    stack = list(nodes)
    while stack:
        node = stack.pop()
        found.append(node)
        if isinstance(node, ast.Lambda):
            stack += [d for d in (*node.args.defaults, *node.args.kw_defaults) if d is not None]
        elif isinstance(node, ast.comprehension):
            stack += [node.iter, *node.ifs]
        else:
            stack += ast.iter_child_nodes(node)
    return found


def _expression_bindings(values: list[object], name: str) -> int:
    """How many times expressions and targets in ``values`` bind or delete ``name``."""
    roots = [
        item
        for value in values
        for item in (value if isinstance(value, list) else [value])
        if isinstance(item, ast.AST)
    ]
    return sum(
        isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, (ast.Store, ast.Del))
        for node in _same_scope(roots)
    )


def _pattern_bindings(pattern: ast.AST, name: str) -> int:
    """How many capture names of a ``match`` pattern are ``name``."""
    return sum(
        getattr(node, "name", None) == name or getattr(node, "rest", None) == name
        for node in ast.walk(pattern)
        if type(node).__name__ in ("MatchAs", "MatchStar", "MatchMapping")
    )


def _statement_bindings(body: list[ast.stmt], name: str) -> int:
    """How many times statements of a module body bind or delete ``name`` at module scope.

    Statements nested in blocks that run at module scope (``if``, ``try``,
    ``with``, ``for``, ``while``, ``match``) count; function and class bodies
    do not, but their decorators, default values, annotations and base classes
    do. ``global`` declarations are counted by :func:`module_bindings`.
    """
    count = 0
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            count += node.name == name
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            count += sum(
                a.name == "*" or (a.asname or a.name.split(".")[0]) == name for a in node.names
            )
            continue
        if isinstance(node, ast.AnnAssign) and node.value is None:
            # A bare annotation binds nothing; only its expression is evaluated.
            fields = [node.annotation]
        else:
            fields = [value for field, value in ast.iter_fields(node) if field not in _BLOCKS]
        count += _expression_bindings(fields, name)
        for handler in getattr(node, "handlers", None) or []:
            count += handler.name == name
            count += _expression_bindings([handler.type], name)
            count += _statement_bindings(handler.body, name)
        for case in getattr(node, "cases", None) or []:
            count += _pattern_bindings(case.pattern, name)
            count += _expression_bindings([case.guard], name)
            count += _statement_bindings(case.body, name)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for field in ("body", "orelse", "finalbody"):
                count += _statement_bindings(getattr(node, field, None) or [], name)
    return count


def module_bindings(tree: ast.Module, name: str) -> int:
    """How many times the module binds or deletes ``name`` at module scope, as far as the syntax shows.

    A ``global`` declaration of ``name`` anywhere in the module counts once.
    """
    declared = sum(name in node.names for node in ast.walk(tree) if isinstance(node, ast.Global))
    return _statement_bindings(tree.body, name) + declared


def definition_text(source: str, name: str, kind: type = ast.FunctionDef) -> str:
    """The source lines of the single top-level ``kind`` named ``name``.

    The text runs from the first decorator to the last line of the body,
    including comment lines that follow it indented deeper than the ``def``.

    Raises:
        LookupError: There is no such definition, or anything else binds or
            deletes ``name`` at module scope (:func:`module_bindings`).
    """
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, kind) and n.name == name]
    if len(nodes) != 1:
        raise LookupError(f"expected one top-level {kind.__name__} {name!r}, found {len(nodes)}")
    bindings = module_bindings(tree, name)
    if bindings != 1:
        raise LookupError(f"{name!r} is bound or deleted {bindings} times at module scope")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    last = node.end_lineno
    probe = last
    while probe < len(lines):
        line = lines[probe]
        if line.strip() and len(line) - len(line.lstrip()) <= node.col_offset:
            break
        probe += 1
        if line.strip():
            last = probe
    return "".join(lines[first - 1 : last])


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


def _no_registered_lines() -> dict[tuple[str, str], Counter[str]]:
    return {region: Counter() for region in ADDITION_REGIONS}


def load_registry(text: str | None) -> tuple[dict[tuple[str, str], Counter[str]], list[str]]:
    """Registered lines per region, and the problems found in the registry itself."""
    registered = _no_registered_lines()
    if text is None:
        return registered, [f"registry {REGISTRY_PATH} is missing"]
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as error:
        return registered, [f"the registry is not valid YAML: {error}"]
    if not isinstance(data, dict) or not isinstance(data.get("additions") or [], list):
        return registered, ["the registry is not a mapping with a list of additions"]
    problems = []
    for index, entry in enumerate(data.get("additions") or []):
        where = f"registry entry {index}"
        if not isinstance(entry, dict):
            problems.append(f"{where} is not a mapping")
            continue
        region = (entry.get("file"), entry.get("region"))
        if not all(isinstance(x, str) for x in region):
            problems.append(f"{where} has no file and region names")
        elif entry.get("change") != ALLOWED_CHANGE:
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
    repo: str | Path, base: str, head: str, registry_text: str | None = None
) -> list[tuple[str, bool, str]]:
    """Every check as ``(name, passed, detail)``; ``head`` "" checks the files on disk.

    Raises:
        RuntimeError: ``base`` or ``head`` is not a commit.

    The registry is ``registry_text`` when given, else the one in ``head``, else
    (for a head without ``scripts/poc3``, such as the frozen base) the one next
    to this script.
    """
    for rev in (base, head):
        if (
            rev
            and subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
                capture_output=True,
            ).returncode
        ):
            raise RuntimeError(f"unknown revision {rev!r}")
    read, differs = git_reader(repo), git_differ(repo)
    results = []
    for path in ORCHESTRATOR_FILES:
        same = _exists(read, base, path) and not differs(base, head, path)
        results.append(
            (f"file {path}", same, "" if same else "differs from the base (or is missing)")
        )

    for path, name in FROZEN_DEFINITIONS:
        try:
            old_text = definition_text(read(base, path) or "", name)
            new_text = definition_text(read(head, path) or "", name)
        except (LookupError, SyntaxError, ValueError) as error:
            # ValueError: not UTF-8, or (before Python 3.12) a NUL byte ast cannot parse.
            results.append((f"def {path}:{name}", False, f"{type(error).__name__}: {error}"))
            continue
        same = old_text == new_text
        results.append((f"def {path}:{name}", same, "" if same else "source text changed"))

    source = "given"
    try:
        if registry_text is None:
            registry_text = read(head, REGISTRY_PATH)
            source = "head" if head else "the working tree"
        head_has_gate = _exists(read, head, f"{Path(REGISTRY_PATH).parent}/check_invariants.py")
        if registry_text is None and not head_has_gate and REGISTRY.is_file():
            # The head predates scripts/poc3 (the frozen base, for example).
            registry_text, source = REGISTRY.read_bytes().decode(), str(REGISTRY)
    except ValueError as error:
        registered = _no_registered_lines()
        problems = [f"the registry cannot be read: {type(error).__name__}: {error}"]
    else:
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
        except (LookupError, SyntaxError, ValueError) as error:
            results.append((label, False, f"{type(error).__name__}: {error}"))
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
    try:
        registry_text = args.registry.read_bytes().decode() if args.registry else None
        results = run(args.repo, args.base, head, registry_text)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {type(error).__name__}: {error}")
        return 1
    for name, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    failed = [name for name, passed, _ in results if not passed]
    target = "working tree" if args.working_tree else args.head
    print(f"{'FAIL' if failed else 'PASS'}: invariants of {target} against {args.base[:12]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

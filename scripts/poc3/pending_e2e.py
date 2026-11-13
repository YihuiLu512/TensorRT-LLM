#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which work items have landed, and which end-to-end checks a work item owes when it lands.

A feature row of ``feature_deps.yaml`` is owed by the work item whose landing
completes the row's dependencies, provided nobody has recorded it as verified:

- a row with an ``assignee`` is owed only by that item (an integration round or
  a performance item);
- a case-03 row without an ``assignee`` is owed by R13 (WT-62) when R13's
  landing completes it. A row another item completes after R13 landed is left
  to the row's ``fallback`` (R14, WT-79, unless the row names another item); a
  fallback landing after R13 owes only the completed rows that name it as
  their fallback (R14: those that name no other item);
- a fix (``WT-xx-fixN``) owes nothing new, and its landing does not count as
  WT-xx landing;
- deferred rows are owed by nobody.

When the row's fixed verifier (its ``assignee``, or the ``fallback`` of a
case-03 row once R13 has landed) runs before the row is complete, or the row
completes after that verifier landed, nobody owes the row any more; both cases
are listed so the coordinator can hand the row to someone. A fallback landing
before R13 lists nothing: R13 still owes the rows its landing completes.

Supplementary cases (``hopper_cases``, ``two_node_cases``) never create an
obligation; integration rounds run them.

A row counts as verified when a PoC-S3 verification record on the integration
branch (``tests/integration/defs/openengine_chain/verified/<item>.yaml``, where
``<item>`` is a PoC-S3 work item such as ``WT-51`` or ``WT-51-fix1``) has an
entry whose ``row`` is the row id (feature ids alone are ambiguous: a feature
can have a case-04 row, a case-03 row and a ``+rest`` row). PoC-S2's records in
the same directory reuse row ids such as ``F-06`` for other rows and are not
read. Entries without ``row``, or naming no row of ``feature_deps.yaml``, are
reported and ignored::

    entries:
      - row: F-01
        features: [F-01]
        cases: ...
        commit: <sha>
        junit: <path>

A ds-engine work item (``groups.ds-repo``) counts as landed when the ds-engine
repository's integration branch (``--ds-repo``, ``--ds-ref``) has its trailer.
Without ``--ds-repo`` their landing is unknown: rows waiting only on them are
listed separately rather than counted as owed or not.

Run it after the last rebase onto the integration branch: pushes only fast
forward, so whichever item lands second sees the first one as landed. The rows
come from the ``feature_deps.yaml`` beside this script, so run the copy in a
checkout rebased onto ``--ref``.

Usage::

    python3 scripts/poc3/pending_e2e.py --task WT-51 --ds-repo ../deepseek-v4.1-flash
    python3 scripts/poc3/pending_e2e.py --landed            # just list what landed

The scripts need Python 3.9 or newer.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from poc3_common import DEFAULT_REF, DS_DEFAULT_REF, git, landed_tasks, parse_task

FEATURE_DEPS = Path(__file__).resolve().parent / "feature_deps.yaml"
VERIFIED_DIR = "tests/integration/defs/openengine_chain/verified"
R13 = "WT-62"
R14 = "WT-79"
CHAINS = ("case-04", "case-03")


@dataclass(frozen=True)
class Row:
    id: str
    features: tuple[str, ...]
    cases: tuple[str, ...]
    deps: frozenset[str]
    chain: str = "case-04"
    assignee: str | None = None
    fallback: str | None = None
    deferred: bool = False
    notes: tuple[str, ...] = ()


@dataclass
class Obligations:
    """The result for one work item.

    Attributes:
        owed: Rows this item must verify before it lands.
        left_to_fallback: case-03 rows this item completes but another item verifies.
        unverified: Rows whose dependencies have all landed, without a
            verification record, that this item does not owe.
        unknown: Rows whose only missing dependencies are items whose landing
            was not checked; the result for them is not known.
        too_early: Rows this item is the fixed verifier of, whose dependencies
            have not all landed yet; nobody owes them once this item lands.
        orphaned: Rows this item completes whose fixed verifier has already
            landed; nobody owes them.
        noted: Rows, deferred ones included, whose notes name this work item
            (``WT-xx`` for a fix ``WT-xx-fixN``): instructions the tables give
            outside the dependency set, such as an early re-run.
    """

    task: str
    landed: list[str]
    owed: list[Row] = field(default_factory=list)
    left_to_fallback: list[Row] = field(default_factory=list)
    unverified: list[Row] = field(default_factory=list)
    unknown: list[Row] = field(default_factory=list)
    too_early: list[Row] = field(default_factory=list)
    orphaned: list[Row] = field(default_factory=list)
    noted: list[Row] = field(default_factory=list)


def load_rows(path: Path = FEATURE_DEPS) -> tuple[dict[str, list[str]], list[Row]]:
    """Read ``feature_deps.yaml``; dependency groups are expanded."""
    data = yaml.safe_load(path.read_text())
    groups: dict[str, list[str]] = data["groups"]
    rows = []
    seen = set()
    for raw in data["rows"]:
        if raw["id"] in seen:
            raise ValueError(f"duplicate row id {raw['id']!r} in {path}")
        seen.add(raw["id"])
        if raw.get("chain", "case-04") not in CHAINS:
            raise ValueError(f"row {raw['id']!r}: unknown chain {raw['chain']!r}")
        deps: set[str] = set()
        for dep in raw["deps"]:
            if dep in groups:
                deps.update(groups[dep])
            else:
                parse_task(dep)
                deps.add(dep)
        for name in (raw.get("assignee"), raw.get("fallback")):
            if name is not None:
                parse_task(name)
        rows.append(
            Row(
                id=raw["id"],
                features=tuple(raw["features"]),
                cases=tuple(raw["cases"]),
                deps=frozenset(deps),
                chain=raw.get("chain", "case-04"),
                assignee=raw.get("assignee"),
                fallback=raw.get("fallback"),
                deferred=bool(raw.get("deferred", False)),
                notes=tuple(raw.get("notes", ())),
            )
        )
    return groups, rows


def verified_rows(
    records: Iterable[Mapping[str, Any]], known: Iterable[str] | None = None
) -> set[str]:
    """The row ids named by verification records.

    Warns about entries without a ``row`` and, given the ``known`` row ids,
    about rows that are not among them.
    """
    known_ids = None if known is None else set(known)
    done = set()
    for record in records:
        if record is None:
            continue
        if not isinstance(record, Mapping):
            print(
                f"warning: verification record is not a mapping, ignored: {record!r}",
                file=sys.stderr,
            )
            continue
        for entry in record.get("entries") or []:
            row = entry.get("row") if isinstance(entry, Mapping) else None
            if not isinstance(row, str):
                print(
                    f"warning: verification entry without a row id ignored: {entry!r}",
                    file=sys.stderr,
                )
            elif known_ids is not None and row not in known_ids:
                print(f"warning: verification entry names an unknown row {row!r}", file=sys.stderr)
            else:
                done.add(row)
    return done


def read_verified(repo: str | Path, ref: str, known: Iterable[str] | None = None) -> set[str]:
    """The row ids verified by the PoC-S3 records on ``ref``."""
    listing = git(repo, "ls-tree", "--full-tree", "--name-only", ref, f"{VERIFIED_DIR}/")
    records = []
    for path in listing.split("\n"):
        name = Path(path).name
        if not name.endswith(".yaml"):
            continue
        try:
            parse_task(name[: -len(".yaml")])
        except ValueError:
            continue
        try:
            records.append(yaml.safe_load(git(repo, "show", f"{ref}:{VERIFIED_DIR}/{name}")))
        except yaml.YAMLError as error:
            print(f"warning: {name} is not valid YAML, ignored: {error}", file=sys.stderr)
    return verified_rows(records, known)


def _names(task: str, notes: Iterable[str]) -> bool:
    return any(re.search(rf"(?<![\w-]){re.escape(task)}(?![\w-])", note) for note in notes)


def obligations(
    task: str,
    landed: Iterable[str],
    rows: Iterable[Row],
    verified: Iterable[str] = (),
    unknown: Iterable[str] = (),
) -> Obligations:
    """Work out what ``task`` owes, given what has landed and what has been verified.

    Args:
        landed: Work items with a commit on the integration branch, fixes under
            their own names; a fix does not make its work item count as landed.
        unknown: Work items whose landing was not checked.
    """
    base, fix = parse_task(task)
    new = fix is None
    landed_bases = {name for name in landed if parse_task(name)[1] is None}
    done = set(verified)
    unchecked = set(unknown) - landed_bases - {base}
    result = Obligations(task=task, landed=sorted(landed))
    after = landed_bases | {base} if new else landed_bases
    for row in rows:
        if row.id in done:
            continue
        if _names(base, row.notes):
            result.noted.append(row)
        if row.deferred:
            continue
        missing = row.deps - after
        if missing and missing <= unchecked:
            result.unknown.append(row)
            continue
        complete = not missing
        completed_by_task = complete and base in row.deps and new
        if row.assignee is not None:
            if row.assignee == base and new:
                (result.owed if complete else result.too_early).append(row)
            elif completed_by_task and row.assignee in landed_bases:
                result.orphaned.append(row)
            elif row.deps <= landed_bases:
                result.unverified.append(row)
            continue
        if row.chain == "case-03":
            fallback = row.fallback or R14
            if base == R13 and complete and new:
                result.owed.append(row)
            elif base == fallback and new and R13 in landed_bases:
                (result.owed if complete else result.too_early).append(row)
            elif completed_by_task:
                # R13 is a dependency of every case-03 row, so it has landed already.
                if fallback in landed_bases:
                    result.orphaned.append(row)
                else:
                    result.left_to_fallback.append(row)
            elif row.deps <= landed_bases:
                result.unverified.append(row)
            continue
        if completed_by_task:
            result.owed.append(row)
        elif row.deps <= landed_bases:
            result.unverified.append(row)
    return result


def _format(result: Obligations) -> str:
    lines = [f"Work item: {result.task}", f"Landed: {', '.join(result.landed) or '(none)'}"]

    def section(title: str, rows: list[Row]) -> None:
        lines.append(f"\n{title} ({len(rows)}):")
        for row in rows:
            lines.append(f"  {row.id:<16} {row.chain:<8} {', '.join(row.cases)}")
        if not rows:
            lines.append("  (none)")

    section("Owed end-to-end checks", result.owed)
    for row in result.owed:
        for note in row.notes:
            lines.append(f"    note on {row.id}: {note}")
    section("Completed here, left to the row's fallback (case-03)", result.left_to_fallback)
    section("Dependencies landed but not verified, owed by others", result.unverified)
    section("Waiting only on items whose landing was not checked", result.unknown)
    section("Fixed verifier is this item, dependencies not all landed: report it", result.too_early)
    section("Completed here after its fixed verifier landed: report it", result.orphaned)
    base = parse_task(result.task)[0]
    lines.append(f"\nRows whose notes name {base}, deferred rows included ({len(result.noted)}):")
    for row in result.noted:
        lines.append(f"  {row.id:<16} " + " / ".join(row.notes))
    if not result.noted:
        lines.append("  (none)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", help="the work item, e.g. WT-51 or WT-51-fix1")
    parser.add_argument("--ref", default=DEFAULT_REF, help="the TRT-LLM integration branch")
    parser.add_argument("--repo", default=".", help="the TRT-LLM checkout")
    parser.add_argument("--ds-repo", help="the ds-engine checkout, to see which ds items landed")
    parser.add_argument(
        "--ds-ref", default=DS_DEFAULT_REF, help="the ds-engine repository's integration branch"
    )
    parser.add_argument(
        "--landed",
        action="store_true",
        help="only list the work items that have landed (on --ref, and --ds-ref if given)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)
    if args.task and not args.landed:
        try:
            parse_task(args.task)
        except ValueError as error:
            parser.error(str(error))
    try:
        return _run(args, parser)
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if isinstance(error.stderr, str) else ""
        parser.exit(2, f"{parser.prog}: error: {' '.join(map(str, error.cmd))}: {stderr}\n")


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    groups, rows = load_rows()
    ds_tasks = set(groups.get("ds-repo", ()))
    landed = {t for t in landed_tasks(args.repo, args.ref) if parse_task(t)[0] not in ds_tasks}
    if args.ds_repo:
        landed |= {
            t for t in landed_tasks(args.ds_repo, args.ds_ref) if parse_task(t)[0] in ds_tasks
        }
        unknown: set[str] = set()
    else:
        unknown = ds_tasks
        print(
            f"warning: --ds-repo not given; not checked whether {', '.join(sorted(ds_tasks))} "
            "landed",
            file=sys.stderr,
        )
    if args.landed:
        print(json.dumps(sorted(landed)) if args.json else "\n".join(sorted(landed)))
        return 0
    if not args.task:
        parser.error("--task is required unless --landed is given")
    if args.task in landed:
        print(
            f"warning: {args.task} has already landed; its obligations were fixed when it landed",
            file=sys.stderr,
        )
    verified = read_verified(args.repo, args.ref, known=(row.id for row in rows))
    result = obligations(args.task, landed, rows, verified, unknown=unknown)
    if args.json:
        print(
            json.dumps(
                {
                    "task": result.task,
                    "landed": result.landed,
                    "owed": [row.id for row in result.owed],
                    "left_to_fallback": [row.id for row in result.left_to_fallback],
                    "unverified": [row.id for row in result.unverified],
                    "unknown": [row.id for row in result.unknown],
                    "too_early": [row.id for row in result.too_early],
                    "orphaned": [row.id for row in result.orphaned],
                    "noted": [row.id for row in result.noted],
                },
                indent=2,
            )
        )
    else:
        print(_format(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which work items have landed, and which end-to-end checks a work item owes when it lands.

A feature row of ``feature_deps.yaml`` is owed by the work item whose landing
completes the row's dependencies, provided nobody has recorded it as verified:

- a row with an ``assignee`` is owed only by that integration round;
- a case-01 row is owed by a glm work item (``groups.glm``) that completes it;
  when a TRT-LLM work item completes it instead, it is left to R3 (WT-20-R3),
  which also owes every completed case-01 row nobody verified;
- a fix (``WT-xx-fixN``) owes nothing new;
- deferred rows are owed by nobody.

A row counts as verified when a verification record on the integration branch
(``tests/integration/defs/openengine_chain/verified/*.yaml``) has an entry whose
``row`` is the row id (feature ids alone are ambiguous: a feature can have a
case-01 row and a case-02 row). Entries without ``row`` are reported and
ignored::

    entries:
      - row: F-01+F-02
        features: [F-01, F-02]
        cases: ...
        commit: <sha>
        junit: <path>

A glm work item counts as landed when the glm repository's integration branch
(``--glm-repo``, ``--glm-ref``) has its trailer; its commits on the TRT-LLM
branch do not count. Without ``--glm-repo`` the landing of glm items is unknown:
rows waiting only on them are listed separately rather than counted as owed or
not.

Run it after the last rebase onto the integration branch: pushes only fast
forward, so whichever item lands second sees the first one as landed.

Usage::

    python3 scripts/poc2/pending_e2e.py --task WT-10 --glm-repo ../glm-5.3-flash
    python3 scripts/poc2/pending_e2e.py --landed            # just list what landed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from poc2_common import DEFAULT_REF, GLM_DEFAULT_REF, git, landed_tasks, parse_task

FEATURE_DEPS = Path(__file__).resolve().parent / "feature_deps.yaml"
VERIFIED_DIR = "tests/integration/defs/openengine_chain/verified"
R3 = "WT-20-R3"


@dataclass(frozen=True)
class Row:
    id: str
    features: tuple[str, ...]
    cases: str
    author: str
    deps: frozenset[str]
    chain: str = "case-02"
    assignee: str | None = None
    deferred: bool = False


@dataclass
class Obligations:
    """The result for one work item.

    Attributes:
        owed: Rows this item must verify before it lands.
        left_to_r3: case-01 rows this item completes but R3 verifies.
        unverified: Rows whose dependencies have all landed, without a
            verification record, that this item does not owe.
        unknown: Rows whose only missing dependencies are items whose landing
            was not checked; the result for them is not known.
    """

    task: str
    landed: list[str]
    owed: list[Row] = field(default_factory=list)
    left_to_r3: list[Row] = field(default_factory=list)
    unverified: list[Row] = field(default_factory=list)
    unknown: list[Row] = field(default_factory=list)


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
        deps: set[str] = set()
        for dep in raw["deps"]:
            if dep in groups:
                deps.update(groups[dep])
            else:
                parse_task(dep)
                deps.add(dep)
        rows.append(
            Row(
                id=raw["id"],
                features=tuple(raw["features"]),
                cases=raw["cases"],
                author=raw["author"],
                deps=frozenset(deps),
                chain=raw.get("chain", "case-02"),
                assignee=raw.get("assignee"),
                deferred=bool(raw.get("deferred", False)),
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
        for entry in (record or {}).get("entries") or []:
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


def read_verified(repo: str | Path, ref: str, rows: Iterable[Row] | None = None) -> set[str]:
    """The row ids verified by the records on ``ref``."""
    listing = git(repo, "ls-tree", "--name-only", ref, f"{VERIFIED_DIR}/")
    names = [Path(path).name for path in listing.split()]
    records = [
        yaml.safe_load(git(repo, "show", f"{ref}:{VERIFIED_DIR}/{name}"))
        for name in names
        if name.endswith(".yaml")
    ]
    return verified_rows(records, None if rows is None else (row.id for row in rows))


def obligations(
    task: str,
    landed: Iterable[str],
    rows: Iterable[Row],
    verified: Iterable[str] = (),
    glm_tasks: Iterable[str] | None = None,
    unknown: Iterable[str] = (),
) -> Obligations:
    """Work out what ``task`` owes, given what has landed and what has been verified.

    Args:
        glm_tasks: The glm work items; ``groups.glm`` of ``feature_deps.yaml`` if None.
        unknown: Work items whose landing was not checked.
    """
    base, fix = parse_task(task)
    landed_bases = {parse_task(name)[0] for name in landed}
    done = set(verified)
    glm = set(load_rows()[0].get("glm", ()) if glm_tasks is None else glm_tasks)
    unchecked = set(unknown) - landed_bases - {base}
    result = Obligations(task=task, landed=sorted(landed))
    after = landed_bases | {base}
    for row in rows:
        if row.deferred or row.id in done:
            continue
        missing = row.deps - after
        if missing and missing <= unchecked:
            result.unknown.append(row)
            continue
        complete = row.deps <= after
        completed_by_task = complete and base in row.deps and fix is None
        if row.assignee is not None:
            if row.assignee == base and complete and fix is None:
                result.owed.append(row)
            elif row.deps <= landed_bases:
                result.unverified.append(row)
            continue
        if row.chain == "case-01":
            if base == R3 and complete and fix is None:
                result.owed.append(row)
            elif completed_by_task and base in glm:
                result.owed.append(row)
            elif completed_by_task:
                result.left_to_r3.append(row)
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
            lines.append(f"  {row.id:<24} {', '.join(row.features):<20} {row.cases}")
        if not rows:
            lines.append("  (none)")

    section("Owed end-to-end checks", result.owed)
    section("Completed here, left to R3 (case-01)", result.left_to_r3)
    section("Dependencies landed but not verified, owed by others", result.unverified)
    section("Waiting only on items whose landing was not checked", result.unknown)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", help="the work item, e.g. WT-10 or WT-11-fix1")
    parser.add_argument("--ref", default=DEFAULT_REF, help="the integration branch")
    parser.add_argument("--repo", default=".", help="the TRT-LLM checkout")
    parser.add_argument("--glm-repo", help="the glm checkout, to see which glm items landed")
    parser.add_argument(
        "--glm-ref", default=GLM_DEFAULT_REF, help="the glm repository's integration branch"
    )
    parser.add_argument(
        "--landed",
        action="store_true",
        help="only list the work items that have landed (on --ref, and --glm-ref if given)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)

    groups, rows = load_rows()
    glm_tasks = set(groups.get("glm", ()))
    landed = {t for t in landed_tasks(args.repo, args.ref) if parse_task(t)[0] not in glm_tasks}
    if args.glm_repo:
        landed |= {
            t for t in landed_tasks(args.glm_repo, args.glm_ref) if parse_task(t)[0] in glm_tasks
        }
        unknown: set[str] = set()
    else:
        unknown = glm_tasks
        print(
            f"warning: --glm-repo not given; not checked whether {', '.join(sorted(glm_tasks))} "
            "landed",
            file=sys.stderr,
        )
    if args.landed:
        print(json.dumps(sorted(landed)) if args.json else "\n".join(sorted(landed)))
        return 0
    if not args.task:
        parser.error("--task is required unless --landed is given")
    result = obligations(
        args.task,
        landed,
        rows,
        read_verified(args.repo, args.ref, rows),
        glm_tasks=glm_tasks,
        unknown=unknown,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "task": result.task,
                    "landed": result.landed,
                    "owed": [row.id for row in result.owed],
                    "left_to_r3": [row.id for row in result.left_to_r3],
                    "unverified": [row.id for row in result.unverified],
                    "unknown": [row.id for row in result.unknown],
                },
                indent=2,
            )
        )
    else:
        print(_format(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the new-test and migrated-test lists against the functional test plan.

    python3 tests/integration/defs/openengine_chain/check_case_lists.py --doc <02_functional-tests.md>

The plan marks a new test ``NT-nnn`` in its list of new tests (a table row
``| NT-nnn | <layer> | `<node id>`... | ...``; a row whose layer is ``—`` is
withdrawn) and a migrated aggregated test as
``**【搬迁】** <T-n | XD-7nn> `<file>::<test>`、`::<test>`...``. Each work item
lists the node ids it wrote in ``new_cases/<WT>.txt`` and the ones it migrated
in ``migrated/<WT>.txt``: one per line, ``<id> <node id>`` or the node id
alone, ``#`` starting a comment. A migrated entry may add a parameter id to
the plan's ``<file>::<test>``; a plan entry naming a file alone covers its tests.

The checks: (1) every listed new test is a new test of the plan, with the
plan's id if it gives one; (2) every listed migrated test is a migrated test of
the plan; (3) no node id is listed twice, in one list file or two; (4) a new GPU end-to-end test
lives in the file the plan requires for its case. With ``--complete`` the
lists must also hold every test the plan marks, as at the end of the stage.
An empty list set, or a plan in which no mark is found, fails.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
NEW_DIR = "new_cases"
MIGRATED_DIR = "migrated"
_LIST_NAME = re.compile(r"^WT-\d{2}(?:-fix\d+|-R\d+)?\.txt$")
_ID = re.compile(r"^(?:NT-\d{3}|T-?\d+|XD-\d{3})$")
_NEW_ROW = re.compile(r"^\| (NT-\d{3}) \| ([^|]+?) \| `([^`]+)`[^|]*\|", re.MULTILINE)
_MIGRATED = re.compile(r"\*\*【搬迁】\*\* (T-?\d+|XD-\d{3}) `([^`]+)`((?:、`::[^`]+`)*)")
# The file a new GPU end-to-end test of each case must be in.
_E2E_FILES = {
    "GPU-E2E-case04": re.compile(r"^unittest/llmapi/apps/_test_openengine_disagg_[^/]+\.py$"),
    "GPU-E2E-case03": re.compile(r"^unittest/llmapi/apps/_test_openengine_ds_disagg_[^/]+\.py$"),
}


def _case_id(text: str) -> str:
    """``T-n`` written as the plan writes it, ``Tn``; other ids as they are."""
    return re.sub(r"^T-(\d+)$", r"T\1", text)


@dataclass(frozen=True)
class Entry:
    path: str  # the list file
    line: int
    case_id: str | None
    nodeid: str


def read_list(path: str, problems: list[str] | None = None) -> list[Entry]:
    """The entries of one list; a malformed line goes to ``problems`` (raises without it)."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for number, raw in enumerate(f, 1):
            text = raw.split("#", 1)[0].strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) == 2 and _ID.match(parts[0]):
                entries.append(Entry(path, number, _case_id(parts[0]), parts[1]))
            elif len(parts) == 1:
                entries.append(Entry(path, number, None, parts[0]))
            else:
                problem = f"{path}:{number}: expected '<id> <node id>' or a node id: {text!r}"
                if problems is None:
                    raise ValueError(problem)
                problems.append(problem)
    return entries


def read_lists(directory: str) -> tuple[list[Entry], list[str]]:
    """The entries of every work item's list in ``directory``, and the problems found reading them."""
    entries, problems = [], []
    for path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
        if not _LIST_NAME.match(os.path.basename(path)):
            problems.append(
                f"{path}: a list is named after its work item, WT-<nn>[-fix<n> | -R<n>].txt"
            )
            continue
        entries.extend(read_list(path, problems))
    return entries, problems


def plan_new_tests(text: str) -> dict[str, tuple[str, str]]:
    """``{node id: (NT id, layer)}`` of the plan's new tests, withdrawn ones left out."""
    found = {}
    for case_id, layer, nodeid in _NEW_ROW.findall(text):
        layer = layer.strip()
        if layer not in ("—", "-"):
            found[nodeid] = (case_id, layer)
    return found


def plan_migrated_tests(text: str) -> dict[str, str]:
    """``{<file>::<test> or <file>: T/XD id}`` of the plan's migrated tests."""
    found = {}
    for case_id, first, more in _MIGRATED.findall(text):
        case_id = _case_id(case_id)
        path = first.partition("::")[0]
        found[first] = case_id
        for extra in re.findall(r"`(::[^`]+)`", more):
            found[path + extra] = case_id
    return found


def _covers(planned: str, nodeid: str) -> bool:
    """Whether the plan's ``planned`` entry covers a listed node id.

    An entry ``<file>::<test>`` covers that test and its parameters; an entry
    ``<file>`` or ``<file>::<class>`` covers every test in it.
    """
    return nodeid == planned or nodeid.startswith((planned + "[", planned + "::"))


def check(doc_text: str, root: str = _HERE, complete: bool = False) -> list[str]:
    """The problems found; an empty list means the lists agree with the plan."""
    problems = []
    planned_new = plan_new_tests(doc_text)
    planned_migrated = plan_migrated_tests(doc_text)
    if not planned_new:
        problems.append("no new test (| NT-nnn | <layer> | `<node id>` |) was found in the plan")
    if not planned_migrated:
        problems.append("no migrated test (**【搬迁】** <id> `<node id>`) was found in the plan")
    new, bad_names = read_lists(os.path.join(root, NEW_DIR))
    migrated, more_bad = read_lists(os.path.join(root, MIGRATED_DIR))
    problems += bad_names + more_bad
    if not new and not migrated:
        problems.append(f"no test is listed in {NEW_DIR}/*.txt or {MIGRATED_DIR}/*.txt")
    for entry in new:
        where = f"{entry.path}:{entry.line}: {entry.nodeid}"
        planned = planned_new.get(entry.nodeid)
        if planned is None:
            problems.append(f"{where}: not a new test of the plan")
            continue
        if entry.case_id is not None and entry.case_id != planned[0]:
            problems.append(f"{where}: listed as {entry.case_id}, the plan's id is {planned[0]}")
        rule = _E2E_FILES.get(planned[1])
        if rule is not None and not rule.match(entry.nodeid.split("::")[0]):
            problems.append(
                f"{where}: a {planned[1]} test must be in a file matching {rule.pattern}"
            )
    for entry in migrated:
        where = f"{entry.path}:{entry.line}: {entry.nodeid}"
        ids = {i for p, i in planned_migrated.items() if _covers(p, entry.nodeid)}
        if not ids:
            problems.append(f"{where}: not a migrated test of the plan")
        elif entry.case_id is not None and entry.case_id not in ids:
            problems.append(f"{where}: listed as {entry.case_id}, the plan's id is {sorted(ids)}")
    seen: dict[str, Entry] = {}
    for entry in new + migrated:
        other = seen.setdefault(entry.nodeid, entry)
        if other is entry:
            continue
        if other.path != entry.path:
            problems.append(f"{entry.nodeid}: listed in both {other.path} and {entry.path}")
        else:
            problems.append(f"{entry.path}:{entry.line}: {entry.nodeid}: listed twice")
    for nodeid, (case_id, layer) in planned_new.items():
        rule = _E2E_FILES.get(layer)
        if rule is not None and not rule.match(nodeid.split("::")[0]):
            problems.append(f"plan: {case_id} {nodeid}: a {layer} test must match {rule.pattern}")
    if complete:
        listed_new = {e.nodeid for e in new}
        problems += [
            f"plan: new test {case_id} {nodeid} is in no {NEW_DIR}/*.txt"
            for nodeid, (case_id, _) in planned_new.items()
            if nodeid not in listed_new
        ]
        problems += [
            f"plan: migrated test {case_id} {planned} is in no {MIGRATED_DIR}/*.txt"
            for planned, case_id in planned_migrated.items()
            if not any(_covers(planned, e.nodeid) for e in migrated)
        ]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--doc", required=True, help="the plan: 02_functional-tests.md")
    parser.add_argument(
        "--complete", action="store_true", help="also require every marked test to be listed"
    )
    args = parser.parse_args(argv)
    with open(args.doc, encoding="utf-8") as f:
        problems = check(f.read(), complete=args.complete)
    for problem in problems:
        print(f"FAIL: {problem}")
    print("FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

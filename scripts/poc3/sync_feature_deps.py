#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive ``feature_deps.yaml`` from the feature tables of the PoC-S3 plan, or check it.

The plan's ``tasks.md`` lists every feature row in §3.4 to §3.7 with four
columns: the feature, its cases, who writes them, and the work items that must
have landed (its dependency set); §3.2 names the dependency groups, and the
section headings give the number of main rows. This script reads those tables
and writes the rows ``pending_e2e.py`` works from; ``--check`` tells when the
file no longer matches the tables:

- A table A / B row splits into a case-04 row (id ``F-01``) and a case-03 row
  (id ``F-01@03``) when its cells have ``04：`` / ``03：`` segments; table D
  rows are case-04 and table C rows case-03.
- "其余用例另等 WT-xx" (the remaining cases wait for WT-xx) adds a row
  ``<id>+rest`` whose cases are the ones WT-xx writes and whose dependencies
  add WT-xx.
- "R11 前提" and "R12 验收" fix the verifier (``assignee``) to R11 (WT-60) and
  R12 (WT-61); "由 WT-xx 做" fixes it to WT-xx; "由 WT-xx 补" names the item
  that verifies a case-03 row WT-62 could not (``fallback``; R14 by default).
- "<PU-nn> 取证（WT-xx）" (performance case PU-nn, run by WT-xx, is the
  evidence) moves that case to its own row ``<id>+<PU-nn>`` verified by WT-xx.
- "不设 E2E" (no E2E) makes the row ``deferred``.
- Cases in parentheses are not the row's evidence: Hopper-shard cases (with
  their "第一批" / "第二批" batch notes) and two-node cases are kept as
  supplementary lists, unrunnable ones as ``not_runnable``.
- The dependency set when it carries a parenthesised qualifier, the reason
  of a "不设 E2E" row, any clause of the dependency cell that is not exactly
  one of the forms above, and any other parenthesised note of the case cell
  that names a case are kept verbatim in the row's ``notes``.
- The ds-engine repository's items (group ``ds-repo``) come from the
  repository table of tasks.md §0.2.

``SPLITS`` lists rows split after the tables were written: each makes some
cases of a row also wait for ``<item>``, a work item or fix (such as
``WT-51-fix2``); the moved cases are credited to ``<item>``. The row is a main
row (``F-01``) or its ``+rest`` row (``F-01+rest``). A split normally
moves the cases to a row ``<row>+<item>``, and the authors of the row are worked
out again from the cases it keeps. A whole split, whose cases are ``*`` followed by every case in the
row's ``cases`` (not those of the main row's ``+rest`` row, nor its Hopper,
two-node or unrunnable ones), adds no row: the row itself, under its own id,
waits for ``<item>``. The row's Hopper, two-node and unrunnable cases stay with
it. A row takes one partial split and one whole split at most; with both, the
partial split moves its cases first and the whole split lists every case left.
A split is refused when its
row is missing, deferred, has a fixed verifier or performance evidence; when
the row has notes that name no case or name a case that moves;
when it lists no case, a case twice, or ``*`` other than first; when a case is
not a case of the row (the main row's ``+rest`` cases included) or is also among its
Hopper, two-node or unrunnable cases; when a partial split would leave the row
no case, or a whole split leaves it any; when both splits of a row name one
item; and when the item is malformed or
a dependency of the row already, groups expanded. So ``--check`` still compares
every row with the tables, and fails when the tables add a main case to the row
of a whole split.

``GROUP_ADDITIONS`` adds fixes to the groups of §3.2, whose table names work
items only. ``VERIFIERS`` sets the fixed verifier of a row whose verifier was
decided after the tables were written; the verifier must be one the tables
already use, and the row must have none, be case-04 and not be deferred.
``ROW_WAITS`` makes a row with a fixed verifier also wait for an item, which
must be well formed and not a dependency of the row already, groups expanded.

The scripts need Python 3.9 or newer.

Usage::

    python3 scripts/poc3/sync_feature_deps.py --tasks-md <plan>/01_overall-plan/tasks.md --check
    python3 scripts/poc3/sync_feature_deps.py --tasks-md <plan>/01_overall-plan/tasks.md --write

``--tasks-md`` defaults to ``$POC3_TASKS_MD``. Exit status of ``--check``: 0
when the file matches the tables, 1 otherwise.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from poc3_common import parse_task

FEATURE_DEPS = Path(__file__).resolve().parent / "feature_deps.yaml"

_GROUP_WORDS = {"链路": "link", "ds 链路": "ds-link", "R11 前提": "r11-prereqs"}
R11, R12 = "WT-60", "WT-61"

HEADER = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PoC-S3 features: who writes their end-to-end cases and which work items must
# have landed before they can be verified. pending_e2e.py reads this file.
# Generated by sync_feature_deps.py from tasks.md section 3.4 to 3.7 of the
# PoC-S3 plan; regenerate it rather than editing rows by hand.
#
# Row fields:
#   id            Unique row id; verification records name the row they verify.
#                 F-01 is the case-04 row of F-01, F-01@03 its case-03 row,
#                 F-01+rest the cases that wait for a later work item.
#   features      Feature ids.
#   cases         The cases whose passing verifies the row.
#   author        The work items that write, move or re-expect those cases (a
#                 list: one row's cases can come from several items); cases
#                 split out of a row are credited to the item they wait for.
#   deps          Work items, fixes of them (WT-51-fix2) or groups that must
#                 have landed; a fix lands under its own name only.
#   chain         case-04 (the TRT-LLM engine) or case-03 (ds-engine).
#   assignee      A fixed verifier (an integration round or a performance item).
#   fallback      case-03 only: who verifies the row when it completes after
#                 R13 (WT-62) landed; R14 (WT-79) when absent.
#   deferred      True for rows nobody verifies (no E2E in this stage).
#   hopper_cases, two_node_cases
#                 Supplementary evidence run only by integration rounds; they
#                 never make a work item owe the row.
#   not_runnable  Cases the PoC machines cannot run.
#   notes         Clauses of the tables this file does not model, verbatim.
#
# A row <id>+<item> (DG-16+WT-51-fix2; DG-34+rest+WT-57-fix2 for a +rest row)
# holds cases split out of row <id> after the tables were written; it waits for
# <item> too. A whole split covers all the cases row <id> keeps without adding
# a row: row <id> itself waits for <item> instead.
# sync_feature_deps.SPLITS lists these splits.
#
# A row with an assignee is verified by it. Otherwise the verifier of a case-04
# row is the dependency that lands last, and a case-03 row is verified by R13
# (WT-62), or by its fallback when it completes after R13 landed.
"""

# Not \b: in the plan's Chinese text a case id can touch a CJK character, which \b treats as a word
# character.
_CASE_RE = re.compile(r"(?<![A-Za-z0-9-])(XD-\d{3}|NT-\d{3}|T\d+|PU-\d+)(?![0-9])")
_TASK_RE = re.compile(r"WT-\d\d")
_SECTION_RE = re.compile(r"^### 3\.([2-7]) ")
_MAIN_ROWS_RE = re.compile(r"（(\d+) 个主行）")
_PU_RULE_RE = re.compile(r"(PU-\d+) 取证（(WT-\d\d)）")
_OPEN, _CLOSE = "（(", "）)"
WHOLE = "*"  # first of a split's cases: the cases listed are every case in the row's cases

# Rows split after the tables were written: (row id, cases that wait for the item, the item). The
# cases of a whole split start with WHOLE and list every case in the row's cases (neither its +rest
# cases nor its Hopper, two-node or unrunnable ones); the row itself then waits for the item.
# WT-51-fix2 writes the GPU file of NT-080 and NT-179 once the link has landed, so the item that
# completes the link cannot verify them. On the rows whose dependencies are exactly the link and
# WT-52 (WT-53), every GPU-layer NT case (the layer column of the plan's functional tests, §6.3)
# also waits for WT-52-fix1 (WT-53-fix2); the CPU-layer NT cases stay with the row. The XD and T
# cases stay with the row too, except XD-121 and XD-123 (DG-19), XD-120 (F-43), XD-026 and XD-119
# (DG-24), which also wait for WT-53-fix2, so that F-43 and DG-24 wait for it whole, and XD-219,
# which waits for WT-52-fix1 in both its rows (F-25, EG-06). DG-39b, DG-81 and EG-20, whose
# dependencies include WT-55, wait for WT-55-fix1 whole.
# Of the rows WT-57 completes, WT-57 keeps the CPU and one-GPU cases. The new multi-GPU cases wait
# for WT-57-fix1 and the existing multi-GPU cases (with T23, T31 and T32) for WT-57-fix2: rows with
# both kinds keep the one-GPU cases and move the others to <row>+WT-57-fix2, rows with only one kind
# wait for that fix whole, and EG-01a and DG-67, with new and existing multi-GPU cases only, move
# the existing ones to <row>+WT-57-fix2 and wait for WT-57-fix1 whole. XD-215 waits for WT-57-fix2
# in EG-03 and for WT-55-fix1 in DG-81: neither fix changes what it expects.
SPLITS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("DG-16", ("NT-080",), "WT-51-fix2"),
    ("DG-82", ("NT-179",), "WT-51-fix2"),
    ("F-06", ("*", "NT-055"), "WT-53-fix2"),
    ("F-10", ("*", "NT-058"), "WT-52-fix1"),
    ("F-19", ("*", "NT-064"), "WT-52-fix1"),
    ("F-25", ("NT-065", "XD-219"), "WT-52-fix1"),
    ("F-26", ("*", "NT-066"), "WT-52-fix1"),
    ("F-39", ("*", "NT-069"), "WT-53-fix2"),
    ("F-43", ("*", "XD-120", "NT-073"), "WT-53-fix2"),
    ("DG-19", ("NT-081", "XD-123", "XD-121"), "WT-53-fix2"),
    ("DG-22", ("NT-083", "NT-084"), "WT-53-fix2"),
    ("DG-23", ("*", "NT-085"), "WT-53-fix2"),
    ("DG-24", ("*", "XD-026", "XD-119", "NT-086"), "WT-53-fix2"),
    ("DG-25", ("*", "NT-087"), "WT-53-fix2"),
    ("DG-35", ("NT-083", "NT-084"), "WT-53-fix2"),
    ("DG-38", ("NT-089",), "WT-52-fix1"),
    ("DG-39", ("*", "NT-090"), "WT-52-fix1"),
    ("DG-39b", ("*", "NT-091"), "WT-55-fix1"),
    ("DG-39c", ("NT-092",), "WT-52-fix1"),
    ("DG-81", ("*", "XD-215", "XD-207", "XD-015", "XD-016", "XD-227"), "WT-55-fix1"),
    ("DG-84", ("NT-180", "NT-040"), "WT-52-fix1"),
    ("DG-85", ("NT-181",), "WT-52-fix1"),
    ("DG-86", ("NT-182",), "WT-52-fix1"),
    ("DG-87", ("NT-183", "NT-184"), "WT-52-fix1"),
    ("DG-88", ("NT-185",), "WT-52-fix1"),
    ("DG-89", ("*", "NT-186"), "WT-53-fix2"),
    ("DG-90", ("NT-187",), "WT-52-fix1"),
    ("EG-05a", ("*", "NT-190"), "WT-52-fix1"),
    ("EG-05b", ("NT-190",), "WT-52-fix1"),
    ("EG-06", ("XD-219",), "WT-52-fix1"),
    ("EG-07", ("NT-191",), "WT-52-fix1"),
    ("EG-10b", ("*", "NT-192"), "WT-52-fix1"),
    ("EG-16a", ("NT-197",), "WT-52-fix1"),
    ("EG-19", ("NT-216",), "WT-52-fix1"),
    ("EG-20", ("*", "NT-200"), "WT-55-fix1"),
    ("EG-21", ("NT-201",), "WT-53-fix2"),
    ("EG-29", ("*", "NT-204"), "WT-52-fix1"),
    ("F-09", ("XD-060", "XD-061"), "WT-57-fix2"),
    ("F-20", ("XD-062",), "WT-57-fix2"),
    ("F-23", ("*", "XD-206", "XD-073", "XD-074", "T23"), "WT-57-fix2"),
    ("F-30+rest", ("*", "XD-060", "XD-061"), "WT-57-fix2"),
    ("F-31", ("*", "XD-205", "XD-203", "XD-221"), "WT-57-fix2"),
    (
        "F-32",
        ("*", "XD-225", "XD-203", "XD-011", "XD-029", "XD-030", "XD-031", "XD-032", "XD-033")
        + ("XD-034", "T31", "T32"),
        "WT-57-fix2",
    ),
    ("DG-11", ("XD-105",), "WT-57-fix2"),
    ("DG-20", ("XD-008",), "WT-57-fix2"),
    ("DG-34+rest", ("XD-312",), "WT-57-fix2"),
    ("DG-67", ("XD-057", "XD-058", "XD-208"), "WT-57-fix2"),
    ("DG-67", ("*", "NT-098"), "WT-57-fix1"),
    ("EG-01a", ("XD-225", "XD-205"), "WT-57-fix2"),
    ("EG-01a", ("*", "NT-189"), "WT-57-fix1"),
    ("EG-01b", ("*", "XD-210", "XD-059", "XD-231"), "WT-57-fix2"),
    ("EG-01e", ("*", "NT-188"), "WT-57-fix1"),
    ("EG-01f", ("*", "XD-228", "XD-060", "XD-061"), "WT-57-fix2"),
    ("EG-03", ("*", "XD-212", "XD-217", "XD-215", "XD-221", "XD-224"), "WT-57-fix2"),
    ("EG-10a", ("*", "XD-062"), "WT-57-fix2"),
    (
        "EG-13",
        ("*", "XD-043", "XD-053", "XD-222", "XD-220", "XD-224", "XD-226", "XD-229", "XD-060")
        + ("XD-061", "XD-010"),
        "WT-57-fix2",
    ),
    ("EG-14e+rest", ("*", "XD-231"), "WT-57-fix2"),
    ("EG-16b", ("*", "NT-198"), "WT-57-fix1"),
    ("EG-16c", ("*", "NT-199"), "WT-57-fix1"),
    ("EG-18+rest", ("*", "XD-205"), "WT-57-fix2"),
    (
        "EG-22a",
        ("XD-017", "XD-040", "XD-211", "XD-223", "XD-226", "XD-209", "XD-230"),
        "WT-57-fix2",
    ),
    ("EG-24", ("*", "XD-018"), "WT-57-fix2"),
    ("EG-26", ("*", "XD-231"), "WT-57-fix2"),
    ("EG-27", ("*", "XD-221", "XD-072", "XD-003"), "WT-57-fix2"),
    ("EG-31", ("*", "NT-207"), "WT-57-fix1"),
    ("EP-04+rest", ("*", "NT-202"), "WT-57-fix1"),
    ("EP-09+rest", ("*", "XD-221"), "WT-57-fix2"),
)

# Members added to the groups of tasks.md §3.2, whose table names work items only: R11 (WT-60)
# also waits for WT-57's two fixes.
GROUP_ADDITIONS: dict[str, tuple[str, ...]] = {"r11-prereqs": ("WT-57-fix1", "WT-57-fix2")}

# Rows whose fixed verifier was set after the tables were written: (row id, verifier). NT-011, the
# only case of DG-43+rest, is not written: the orchestrator holds no executor to raise
# CppExecutorError, so R11 verifies DG-43 by comparing its run with R10's and by the orchestrator's
# unchanged files (DQ-02).
VERIFIERS: tuple[tuple[str, str], ...] = (("DG-43+rest", R11),)

# Rows with a fixed verifier that also wait for an item the tables do not name: (row id, item). The
# row keeps its verifier and its authors; SPLITS covers the rows without a fixed verifier.
# WT-57-fix1 writes the cases of these R12 rows (NT-202, NT-208 to NT-213).
ROW_WAITS: tuple[tuple[str, str], ...] = tuple(
    (row_id, "WT-57-fix1")
    for row_id in ("EG-01g", "EG-22b", "EG-22c", "EG-22d", "EG-33", "EG-34", "EG-36", "EG-37")
    + ("EG-38",)
)


def split_cases(cases: tuple[str, ...]) -> tuple[bool, tuple[str, ...]]:
    """Whether a split's ``cases`` mark a whole split, and the cases without the marker."""
    whole = cases[:1] == (WHOLE,)
    return whole, cases[1:] if whole else cases


def split_top(text: str, sep: str) -> list[str]:
    """Split ``text`` on ``sep`` outside parentheses (full-width or ASCII)."""
    parts, depth, current = [], 0, []
    for char in text:
        if char in _OPEN:
            depth += 1
        elif char in _CLOSE:
            depth -= 1
        if char == sep and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def outside_parens(text: str) -> tuple[str, list[str]]:
    """The text with its top-level parenthesised groups removed, and those groups."""
    outer, groups, depth, current = [], [], 0, []
    for char in text:
        if char in _OPEN:
            depth += 1
            if depth == 1:
                current = []
                continue
        elif char in _CLOSE:
            depth -= 1
            if depth == 0:
                groups.append("".join(current))
                continue
        (current if depth else outer).append(char)
    return "".join(outer), groups


def segments(cell: str) -> dict[str, list[str]]:
    """Split a cell into its ``04：`` / ``03：`` segments; unprefixed text goes to ``""``."""
    result: dict[str, list[str]] = {}
    key = ""
    for part in split_top(cell, "；"):
        match = re.match(r"^(04|03)：", part)
        if match:
            key = match.group(1)
            part = part[match.end() :]
        result.setdefault(key, []).append(part)
    return result


def _cases(text: str) -> list[str]:
    return list(dict.fromkeys(_CASE_RE.findall(re.sub(r"【[^】]*】", "", text))))


def parse_cases(parts: list[str]) -> dict[str, list[str]]:
    """Main cases, the supplementary and unrunnable ones in parentheses, and other notes."""
    out: dict[str, list[str]] = {
        "cases": [],
        "hopper_cases": [],
        "two_node_cases": [],
        "not_runnable": [],
        "notes": [],
    }
    kinds = {
        "Hopper 分片": "hopper_cases",
        "2 节点组": "two_node_cases",
        "PoC 机型不可跑": "not_runnable",
    }
    for part in parts:
        outer, groups = outside_parens(part)
        out["cases"] += _cases(outer)
        for group in groups:
            current = None
            for note in split_top(group, "；"):
                found = [kind for word, kind in kinds.items() if word in note]
                if found:
                    current = found[0]
                elif not (current == "hopper_cases" and note.startswith(("第一批", "第二批"))):
                    current = None
                if current:
                    out[current] += _cases(note)
                elif _cases(note):
                    out["notes"].append(note)
    return {key: list(dict.fromkeys(value)) for key, value in out.items()}


def parse_authors(parts: list[str], cases: list[str]) -> dict[str, str]:
    """Map each case to the work item that writes it ("existing" for ready-made cases).

    ``WT-xx（<cases>）`` names the cases; a bare ``WT-xx`` writes every case no
    other part names.
    """
    owner: dict[str, str] = {}
    whole_row = None
    for part in parts:
        match = re.match(r"^(WT-\d\d)", part)
        if not match:
            continue
        named = [case for group in outside_parens(part)[1] for case in _cases(group)]
        if named:
            owner.update(dict.fromkeys(named, match.group(1)))
        else:
            whole_row = match.group(1)
    return {case: owner.get(case, whole_row or "existing") for case in cases}


def parse_deps(parts: list[str]) -> dict[str, Any]:
    """The dependency set (first part) and the verifier rules of the parts after it."""
    first = parts[0]
    out: dict[str, Any] = {
        "deps": [],
        "rest": [],
        "assignee": None,
        "fallback": None,
        "deferred": False,
        "performance": [],
        "notes": [],
    }
    if first.startswith("不设 E2E"):
        out["deferred"] = True
        out["notes"] = parts
        return out
    tokens = split_top(first, "、")
    if any(outside_parens(token)[1] for token in tokens):
        out["notes"].append(first)
    for token in tokens:
        word = outside_parens(token)[0].strip()
        if word in _GROUP_WORDS:
            out["deps"].append(_GROUP_WORDS[word])
        elif _TASK_RE.fullmatch(word):
            out["deps"].append(word)
        else:
            raise ValueError(f"unknown dependency {token!r} in {first!r}")
    if "r11-prereqs" in out["deps"]:
        out["assignee"] = R11
    exact = re.compile(r"其余用例另等 WT-\d\d(、WT-\d\d)*|R12 验收|由 WT-\d\d 做|.*由 WT-\d\d 补")
    for part in parts[1:]:
        if not exact.fullmatch(part) and not _PU_RULE_RE.search(part):
            out["notes"].append(part)
        if part.startswith("其余用例另等"):
            out["rest"] = _TASK_RE.findall(part)
        elif part.startswith("R12 验收"):
            out["assignee"] = R12
        elif match := re.match(r"^由 (WT-\d\d) 做", part):
            out["assignee"] = match.group(1)
        elif match := re.search(r"由 (WT-\d\d) 补$", part):
            out["fallback"] = match.group(1)
        elif match := _PU_RULE_RE.search(part):
            out["performance"].append((match.group(1), match.group(2), part))
    return out


def _row(
    row_id: str,
    feature: str,
    chain: str,
    cases: dict[str, list[str]],
    authors: dict[str, str],
    deps: list[str],
    **extra: Any,
) -> dict[str, Any]:
    if not cases["cases"]:
        raise ValueError(f"{row_id}: no case verifies the row")
    row: dict[str, Any] = {
        "id": row_id,
        "features": [feature],
        "cases": cases["cases"],
        "author": sorted({a for a in authors.values() if a != "existing"}) or ["existing"],
        "deps": deps,
        "chain": chain,
    }
    row.update({key: value for key, value in extra.items() if value})
    for key in ("hopper_cases", "two_node_cases", "not_runnable", "notes"):
        if cases.get(key):
            row[key] = cases[key]
    return row


Split = tuple[tuple[str, ...], str]  # the cases of a split (WHOLE first for a whole one), its item


def rows_for(
    feature: str,
    chain: str,
    row_id: str,
    case_parts: list[str],
    author_parts: list[str],
    dep_parts: list[str],
    splits: dict[str, list[Split]] | None = None,
    groups: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """The feature_deps rows of one feature on one chain.

    ``splits`` maps ``row_id`` and ``<row_id>+rest`` to their splits (cases,
    work item) of ``SPLITS``, which make those cases of the row wait for the
    item too: they move to a row ``<row>+<item>``, or, for a whole split (cases
    starting with ``WHOLE``), the row itself waits for it. ``groups`` expands
    the dependency groups when checking the splits.
    """
    splits = splits or {}
    cases = parse_cases(case_parts)
    authors = parse_authors(author_parts, cases["cases"])
    deps = parse_deps(dep_parts)
    notes = cases["notes"] + deps["notes"]
    for target, target_splits in splits.items():
        for split in target_splits:
            _check_split(target, split, cases, deps, groups or {})
    if deps["deferred"]:
        row = {
            "id": row_id,
            "features": [feature],
            "cases": cases["cases"],
            "author": ["-"],
            "deps": [],
            "chain": chain,
            "deferred": True,
        }
        return [row | ({"notes": notes} if notes else {})]
    extra = {"assignee": deps["assignee"], "fallback": deps["fallback"]}
    main = {**cases, "notes": notes}
    out = []
    for pu, task, clause in deps["performance"]:
        if pu not in main["cases"]:
            raise ValueError(f"{row_id}: {pu} is not among the row's cases")
        main = {**main, "cases": [c for c in main["cases"] if c != pu]}
        out.append(
            _row(
                f"{row_id}+{pu}",
                feature,
                chain,
                {"cases": [pu], "notes": [clause]},
                {pu: task},
                deps["deps"] + [task],
                assignee=task,
            )
        )
    later = [c for c in main["cases"] if authors[c] in deps["rest"]]
    if deps["rest"] and not later:
        raise ValueError(f"{row_id}: no case is written by {deps['rest']}")
    now = [c for c in main["cases"] if c not in later]
    now, row_deps, now_rows = _apply_splits(
        row_id, feature, chain, now, authors, deps["deps"], splits.get(row_id, [])
    )
    rows = [
        _row(
            row_id,
            feature,
            chain,
            {**main, "cases": now},
            {c: authors[c] for c in now},
            row_deps,
            **extra,
        )
    ]
    rest_rows = []
    if later:
        rest_id = f"{row_id}+rest"
        later, rest_deps, rest_rows = _apply_splits(
            rest_id,
            feature,
            chain,
            later,
            authors,
            deps["deps"] + deps["rest"],
            splits.get(rest_id, []),
        )
        rows.append(
            _row(
                rest_id,
                feature,
                chain,
                {"cases": later},
                {c: authors[c] for c in later},
                rest_deps,
                **extra,
            )
        )
    elif splits.get(f"{row_id}+rest"):
        raise ValueError(f"split of {row_id}+rest: no such row")
    return rows + now_rows + rest_rows + out


def _apply_splits(
    row_id: str,
    feature: str,
    chain: str,
    cases: list[str],
    authors: dict[str, str],
    deps: list[str],
    splits: list[Split],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Apply the splits of row ``row_id``: the cases it keeps, its dependencies and the new rows.

    The partial split moves its cases to a row ``<row_id>+<item>`` first; the
    whole split then makes the row wait for its item. ``authors`` is updated
    with the cases a whole split credits to its item.
    """
    partial = [split for split in splits if not split_cases(split[0])[0]]
    whole = [split for split in splits if split_cases(split[0])[0]]
    if len(partial) > 1 or len(whole) > 1:
        raise ValueError(
            f"two splits of one row: {row_id} takes one partial and one whole split at most"
        )
    if partial and whole and partial[0][1] == whole[0][1]:
        raise ValueError(f"split of {row_id}: both splits name {whole[0][1]}")
    kept, row_deps, rows = list(cases), list(deps), []
    for split in partial + whole:
        is_whole, moved = split_cases(split[0])
        item = split[1]
        missing = [c for c in moved if c not in kept]
        if missing:
            # A case of the +rest row is split in the tables already, or another split moves it.
            raise ValueError(f"split of {row_id}: {missing} not among the cases row {row_id} keeps")
        left = [c for c in kept if c not in moved]
        if is_whole:
            if left:
                raise ValueError(f"split of {row_id}: marked whole, but the row also keeps {left}")
            # Every case the row keeps waits for the item: the row itself does, under its own id.
            authors.update(dict.fromkeys(moved, item))
            row_deps = row_deps + [item]
            continue
        if not left:
            raise ValueError(
                f"split of {row_id}: no case would be left; a whole split starts with {WHOLE!r}"
            )
        rows.append(
            _row(
                f"{row_id}+{item}",
                feature,
                chain,
                {"cases": [c for c in kept if c in moved]},
                dict.fromkeys(moved, item),
                deps + [item],
            )
        )
        kept = left
    return kept, row_deps, rows


def _check_split(
    row_id: str,
    split: Split,
    cases: dict[str, list[str]],
    deps: dict[str, Any],
    groups: dict[str, list[str]],
) -> None:
    """Refuse a split the rows of the tables do not leave room for."""
    _, moved = split_cases(split[0])
    item = split[1]
    try:
        parse_task(item)
    except ValueError:
        raise ValueError(f"split of {row_id}: {item!r} is not a PoC-S3 work item") from None
    if not moved or len(set(moved)) != len(moved) or WHOLE in moved:
        raise ValueError(
            f"split of {row_id}: no case to move, or a case listed twice, or {WHOLE!r} not first"
        )
    if deps["deferred"] or deps["assignee"] or deps["fallback"] or deps["performance"]:
        raise ValueError(
            f"split of {row_id}: the row is deferred, has a fixed verifier or performance evidence"
        )
    members = {*deps["deps"], *deps["rest"]}
    members |= {member for name in deps["deps"] for member in groups.get(name, ())}
    if item in members:
        raise ValueError(f"split of {row_id}: {item} is a dependency of the row already")
    # Notes may concern the moved cases; the tables must say which. A note that names only cases
    # that stay (EG-26's PB-3 note on its unrunnable XD-020) is left with the row.
    for note in cases["notes"] + deps["notes"]:
        named = _cases(note)
        if not named or set(named) & set(moved):
            raise ValueError(
                f"split of {row_id}: the row has notes that name no case or one it moves"
            )
    # The row keeps its Hopper, two-node and unrunnable cases; none of them may move.
    extra = {c for key in ("hopper_cases", "two_node_cases", "not_runnable") for c in cases[key]}
    supplementary = sorted(extra & set(moved))
    if supplementary:
        raise ValueError(f"split of {row_id}: {supplementary} are Hopper, two-node or unrunnable")


def table_rows(tasks_md: str) -> list[tuple[str, list[str]]]:
    """``(section, cells)`` of every feature row in §3.4 to §3.7 (section is "4" .. "7")."""
    found, section = [], None
    for line in tasks_md.splitlines():
        if line.startswith("### "):
            match = _SECTION_RE.match(line)
            section = match.group(1) if match and match.group(1) >= "4" else None
            continue
        if section is None or not line.startswith("| ") or line.startswith("| 功能 "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
        if len(cells) != 4:
            raise ValueError(f"expected 4 cells in §3.{section}: {line!r}")
        found.append((section, cells))
    return found


def main_row_count(tasks_md: str) -> int:
    """The number of main rows the headings of §3.4 to §3.7 announce."""
    counts = [
        int(count)
        for line in tasks_md.splitlines()
        if (match := _SECTION_RE.match(line)) and match.group(1) >= "4"
        for count in _MAIN_ROWS_RE.findall(line)
    ]
    if not counts:
        raise ValueError("no heading of §3.4 to §3.7 gives its number of main rows")
    return sum(counts)


def derive_groups(tasks_md: str) -> dict[str, list[str]]:
    """The dependency groups of §3.2, plus the ds-engine repository's items (§0.2)."""
    groups: dict[str, list[str]] = {}
    in_section = False
    for line in tasks_md.splitlines():
        if line.startswith("| ds-engine 自有仓库 |"):
            # The last cell lists the items on that repository before the first "；".
            last = line.strip().strip("|").split(" | ")[-1]
            groups["ds-repo"] = _TASK_RE.findall(last.split("；")[0])
        if line.startswith("### "):
            match = _SECTION_RE.match(line)
            in_section = bool(match) and match.group(1) == "2"
            continue
        if in_section and line.startswith("| "):
            cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
            if cells[0] in _GROUP_WORDS:
                groups[_GROUP_WORDS[cells[0]]] = _TASK_RE.findall(cells[1])
    names = [*_GROUP_WORDS.values(), "ds-repo"]
    missing = [name for name in names if not groups.get(name)]
    if missing:
        raise ValueError(f"tasks.md does not define the groups {missing}")
    return {name: groups[name] for name in names}


def add_group_members(
    groups: dict[str, list[str]],
    additions: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, list[str]]:
    """``groups`` with the members of ``additions`` (``GROUP_ADDITIONS`` by default) appended.

    Raises:
        ValueError: A group is missing, a member is not a PoC-S3 work item, or is in the group
            already.
    """
    groups = {name: list(members) for name, members in groups.items()}
    for name, members in (GROUP_ADDITIONS if additions is None else additions).items():
        if name not in groups:
            raise ValueError(f"members added to {name}: no such group")
        for member in members:
            try:
                parse_task(member)
            except ValueError:
                raise ValueError(
                    f"members added to {name}: {member!r} is not a PoC-S3 work item"
                ) from None
            if member in groups[name]:
                raise ValueError(f"members added to {name}: {member} is a member already")
            groups[name].append(member)
    return groups


def set_verifiers(rows: list[dict[str, Any]], verifiers: tuple[tuple[str, str], ...]) -> None:
    """Make each verifier of ``verifiers`` (row id, work item) the fixed verifier of its row.

    The verifier must already verify a row of the tables (as ``assignee`` or
    ``fallback``); the row must be case-04, not deferred and without a fixed
    verifier.

    Raises:
        ValueError: A row is missing or named twice, or does not take the verifier.
    """
    by_id = {row["id"]: index for index, row in enumerate(rows)}
    known = {row.get(key) for row in rows for key in ("assignee", "fallback")} - {None}
    if len({row_id for row_id, _ in verifiers}) != len(verifiers):
        raise ValueError("two verifiers of one row")
    for row_id, verifier in verifiers:
        if row_id not in by_id:
            raise ValueError(f"verifier of {row_id}: no such row")
        if verifier not in known:
            raise ValueError(f"verifier of {row_id}: {verifier!r} verifies no row of the tables")
        row = rows[by_id[row_id]]
        if (
            row.get("deferred")
            or row.get("assignee")
            or row.get("fallback")
            or row["chain"] != "case-04"
        ):
            raise ValueError(
                f"verifier of {row_id}: the row is deferred, case-03 or has a fixed verifier"
            )
        # The same key order as a row the tables give a verifier.
        keys = ("id", "features", "cases", "author", "deps", "chain")
        rows[by_id[row_id]] = (
            {key: row[key] for key in keys}
            | {"assignee": verifier}
            | {key: value for key, value in row.items() if key not in keys}
        )


def add_waits(
    rows: list[dict[str, Any]],
    waits: tuple[tuple[str, str], ...],
    groups: dict[str, list[str]],
) -> None:
    """Make each row of ``waits`` (row id, item), which has a fixed verifier, also wait for the item.

    Raises:
        ValueError: A row is missing or named twice, is deferred or has no fixed verifier, or the
            item is malformed or a dependency of the row already.
    """
    by_id = {row["id"]: row for row in rows}
    if len({row_id for row_id, _ in waits}) != len(waits):
        raise ValueError("two waits of one row")
    for row_id, item in waits:
        row = by_id.get(row_id)
        if row is None:
            raise ValueError(f"wait of {row_id}: no such row")
        try:
            parse_task(item)
        except ValueError:
            raise ValueError(f"wait of {row_id}: {item!r} is not a PoC-S3 work item") from None
        if row.get("deferred") or not row.get("assignee"):
            raise ValueError(f"wait of {row_id}: the row is deferred or has no fixed verifier")
        members = {*row["deps"]} | {m for name in row["deps"] for m in groups.get(name, ())}
        if item in members:
            raise ValueError(f"wait of {row_id}: {item} is a dependency of the row already")
        row["deps"] = row["deps"] + [item]


def derive(
    tasks_md: str,
    splits: tuple[tuple[str, tuple[str, ...], str], ...] = (),
    verifiers: tuple[tuple[str, str], ...] = (),
    waits: tuple[tuple[str, str], ...] = (),
) -> list[dict[str, Any]]:
    """The rows of the plan's §3.4 to §3.7, with ``splits``, ``verifiers`` and ``waits`` applied."""
    pending: dict[str, list[Split]] = {}
    for row_id, moved, item in splits:
        pending.setdefault(row_id, []).append((moved, item))
    groups = add_group_members(derive_groups(tasks_md)) if splits or waits else {}

    def take(row_id: str) -> dict[str, list[Split]]:
        return {
            target: pending.pop(target)
            for target in (row_id, f"{row_id}+rest")
            if target in pending
        }

    rows: list[dict[str, Any]] = []
    features = []
    for section, (feature_cell, case_cell, author_cell, dep_cell) in table_rows(tasks_md):
        feature = feature_cell.split()[0]
        features.append(feature)
        cases, authors, deps = segments(case_cell), segments(author_cell), segments(dep_cell)
        if section in ("4", "5"):  # tables A and B: one row per chain segment
            chains = [k for k in ("04", "03") if k in deps] or [""]
            for cell in (cases, authors, deps):
                if "" in cell and ("04" in cell or "03" in cell):
                    raise ValueError(f"{feature}: text before the 04/03 segments: {cell['']}")
            for key in chains:
                chain = "case-03" if key == "03" else "case-04"
                row_id = f"{feature}@03" if key == "03" else feature
                rows += rows_for(
                    feature,
                    chain,
                    row_id,
                    cases.get(key, []),
                    authors.get(key, []),
                    deps[key],
                    take(row_id),
                    groups,
                )
        else:  # table D is case-04, table C case-03
            chain = "case-04" if section == "6" else "case-03"
            rows += rows_for(
                feature,
                chain,
                feature,
                cases.get("", []),
                authors.get("", []),
                deps[""],
                take(feature),
                groups,
            )
    if pending:
        # One name per split: a row can take two.
        missing = sorted(row_id for row_id, target in pending.items() for _ in target)
        raise ValueError(f"split of {', '.join(missing)}: no such row")
    expected = main_row_count(tasks_md)
    if len(set(features)) != len(features) or len(features) != expected:
        raise ValueError(f"expected {expected} distinct feature rows, got {len(features)}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row ids")
    set_verifiers(rows, verifiers)
    add_waits(rows, waits, groups)
    return rows


def render(groups: dict[str, list[str]], rows: list[dict[str, Any]]) -> str:
    body = yaml.safe_dump(
        {"version": 1, "groups": groups, "rows": rows},
        allow_unicode=False,
        sort_keys=False,
        default_flow_style=None,
        width=100,
    )
    return HEADER + "\n" + body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--tasks-md", default=os.environ.get("POC3_TASKS_MD"), help="the plan's tasks.md"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--check", action="store_true", help="compare feature_deps.yaml with the tables"
    )
    action.add_argument("--write", action="store_true", help="rewrite feature_deps.yaml")
    args = parser.parse_args(argv)
    if not args.tasks_md:
        parser.error("--tasks-md (or $POC3_TASKS_MD) is required")
    tasks_md = Path(args.tasks_md).read_text()
    try:
        expected = render(
            add_group_members(derive_groups(tasks_md)),
            derive(tasks_md, SPLITS, VERIFIERS, ROW_WAITS),
        )
    except ValueError as error:
        # The tables, or a split of SPLITS, cannot be read.
        print(f"FAIL: {args.tasks_md}: {error}")
        return 1
    if args.write:
        FEATURE_DEPS.write_text(expected)
        print(f"wrote {FEATURE_DEPS}")
        return 0
    actual = FEATURE_DEPS.read_text() if FEATURE_DEPS.exists() else ""
    if actual == expected:
        print(f"PASS: {FEATURE_DEPS.name} matches {args.tasks_md}")
        return 0
    sys.stdout.writelines(
        difflib.unified_diff(
            actual.splitlines(True), expected.splitlines(True), "feature_deps.yaml", "tables"
        )
    )
    print(f"FAIL: {FEATURE_DEPS.name} differs from {args.tasks_md}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

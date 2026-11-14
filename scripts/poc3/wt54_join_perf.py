#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join the perf records of disaggregated requests offline, by their ``disagg_request_id``.

A request served through OpenEngine leaves up to five records, each in its own
process's file:

- the orchestrator's, in ``perf_metrics-disagg-*.jsonl``;
- each worker frontend's, in ``perf_metrics-<role>-*.jsonl``: ``context`` or
  ``generation`` for a frontend started with ``--server_role``, ``server``
  for one without (static URLs). A ``server`` record is the leg of the
  engine record with its ``request_id``. Otherwise the leg is inferred from
  the request's other records: a file is one frontend, so the records of a
  file are one leg; if one leg has no record yet and the rest come from one
  file, they are that leg; if both have none and the rest come from two
  files, the file whose record arrived first is the context leg. That last
  rule assumes the orchestrator sends the context leg first, which does not
  hold under generation-first scheduling: give that run's engine records too.
  A frontend record of any other role (an encoder's, say) is left out;
- each TensorRT-LLM OpenEngine server's leg record, in
  ``openengine_leg_durations-*.jsonl``: the durations its engine measured,
  never a time, since the engine's clock is not aligned with the frontends'.

This script reads every such file under the given directories and writes, for
each ``disagg_request_id`` of an orchestrator record, one JSON line holding the
five records (None for a missing one), the KV-cache transfer time and size the
engines measured, the parts missing, the number of extra records of a part
(``duplicates``; the first one is kept), the number of ``server`` records
whose leg could not be inferred (``unassigned``) and whether the legs of two
``server`` files were told apart by arrival alone (``legs_by_arrival``). A
file found under more than one of the directories is read once.

Usage::

    python3 scripts/poc3/wt54_join_perf.py <dir> [<dir> ...] [--require engines|frontends]
        [--output <file>]

``--require engines`` (the default) asks for all five records, ``frontends``
for the orchestrator's and both frontends' (an engine that writes no leg
records). Exit status: 0 when at least one request has every required record;
1 when a given directory does not exist, no record is found, or no request
has them all; 2 when the arguments cannot be parsed. The script needs
Python 3.9 or newer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

FRONTEND_PREFIX = "perf_metrics-"
# The role segment of a frontend started without --server_role.
SERVER_KIND = "server"
ENGINE_PREFIX = "openengine_leg_durations-"
ORCHESTRATOR_KIND = "disagg"
LEGS = ("context", "generation")
PARTS = (
    "orchestrator",
    "context_frontend",
    "generation_frontend",
    "context_engine",
    "generation_engine",
)
REQUIRED = {
    "engines": PARTS,
    "frontends": PARTS[:3],
}


def _records(path: Path) -> Iterator[dict[str, Any]]:
    """The JSON objects of a JSONL file; a line that is not one is skipped."""
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _files(directories: Iterable[Path], prefix: str) -> list[Path]:
    return sorted(
        {
            path.resolve()
            for directory in directories
            for path in directory.rglob(f"{prefix}*.jsonl")
            if path.is_file()
        }
    )


def _key(value: Any) -> str | None:
    """An id as compared across files, as text: an id may be written as a number or as text."""
    return None if value is None else str(value)


# A record as loaded: the kind its file name gives (the frontend's role
# segment, or ``engine``), the record, and its file.
Found = tuple[str, dict[str, Any], Path]


def load(directories: Iterable[Path]) -> dict[str, list[Found]]:
    """The records found, by source: ``orchestrator``, ``frontend`` and ``engine``."""
    directories = list(directories)
    found: dict[str, list[Found]] = {"orchestrator": [], "frontend": [], "engine": []}
    for path in _files(directories, FRONTEND_PREFIX):
        kind = path.name[len(FRONTEND_PREFIX) :].split("-", 1)[0]
        source = "orchestrator" if kind == ORCHESTRATOR_KIND else "frontend"
        found[source].extend((kind, record, path) for record in _records(path))
    for path in _files(directories, ENGINE_PREFIX):
        found["engine"].extend(("engine", record, path) for record in _records(path))
    return found


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _arrival(record: dict[str, Any]) -> float:
    """When the frontend received the request, on the orchestrator's clock; +inf when unknown."""
    value = _dict(_dict(record.get("perf_metrics")).get("timing_metrics")).get(
        "server_arrival_time"
    )
    try:
        return float(value) if isinstance(value, (int, float)) else float("inf")
    except OverflowError:
        return float("inf")


def _frontend_legs(
    records: list[Found], engine_legs: dict[str, str]
) -> tuple[dict[str, list[dict[str, Any]]], int, bool]:
    """The frontend records of one request by leg, the unassigned count, and ``legs_by_arrival``.

    ``engine_legs`` maps the ``request_id`` of each of the request's engine
    records to its leg. See the module's description for how a ``server``
    record's leg is found.
    """
    legs: dict[str, list[dict[str, Any]]] = {leg: [] for leg in LEGS}
    file_legs: dict[Path, str] = {}
    unknown: dict[Path, list[dict[str, Any]]] = {}
    for kind, record, path in records:
        leg = kind if kind in LEGS else engine_legs.get(_key(record.get("request_id")) or "")
        if leg in LEGS:
            legs[leg].append(record)
            file_legs.setdefault(path, leg)
        else:
            unknown.setdefault(path, []).append(record)
    for path in [path for path in unknown if path in file_legs]:
        legs[file_legs[path]].extend(unknown.pop(path))
    empty = [leg for leg in LEGS if not legs[leg]]
    files = sorted(unknown.values(), key=lambda rs: min(_arrival(r) for r in rs))
    by_arrival = False
    if len(empty) == 1 and len(files) == 1:
        legs[empty[0]].extend(files[0])
        files = []
    elif len(empty) == 2 and len(files) == 2:
        first, second = (min(_arrival(r) for r in rs) for rs in files)
        if first < second < float("inf"):
            legs["context"].extend(files[0])
            legs["generation"].extend(files[1])
            files = []
            by_arrival = True
    return legs, sum(len(rs) for rs in files), by_arrival


def _engine_reading(engines: list[dict[str, Any] | None], read: Any) -> Any:
    """``read(record)`` of the generation engine's record, else of the context engine's."""
    for record in engines:
        if record is not None:
            value = read(record)
            if value is not None:
                return value
    return None


def join(found: dict[str, list[Found]], require: str) -> list[dict[str, Any]]:
    """One joined entry per ``disagg_request_id`` of an orchestrator record, in file order."""
    frontends: dict[str, list[Found]] = {}
    for item in found["frontend"]:
        disagg_id = _key(item[1].get("disagg_request_id"))
        if disagg_id is not None and (item[0] in LEGS or item[0] == SERVER_KIND):
            frontends.setdefault(disagg_id, []).append(item)
    engines: dict[tuple[str, str], list[dict[str, Any]]] = {}
    # By disagg_request_id: the leg of each engine record's request_id.
    engine_legs: dict[str, dict[str, str]] = {}
    for _, record, _ in found["engine"]:
        disagg_id = _key(record.get("disagg_request_id"))
        leg = record.get("leg")
        if disagg_id is None or leg not in LEGS:
            continue
        engines.setdefault((disagg_id, leg), []).append(record)
        request_id = _key(record.get("request_id"))
        if request_id is not None:
            engine_legs.setdefault(disagg_id, {})[request_id] = leg

    joined = []
    orchestrators: dict[str, list[dict[str, Any]]] = {}
    for _, record, _ in found["orchestrator"]:
        disagg_id = _key(record.get("disagg_request_id"))
        if disagg_id is not None:
            orchestrators.setdefault(disagg_id, []).append(record)
    for disagg_id, records in orchestrators.items():
        legs, unassigned, by_arrival = _frontend_legs(
            frontends.get(disagg_id, []), engine_legs.get(disagg_id, {})
        )
        entry: dict[str, Any] = {
            "disagg_request_id": records[0]["disagg_request_id"],
            "orchestrator": records[0],
        }
        duplicates = len(records) - 1
        for leg in LEGS:
            parts = {
                f"{leg}_frontend": legs[leg],
                f"{leg}_engine": engines.get((disagg_id, leg), []),
            }
            for part, found_records in parts.items():
                entry[part] = found_records[0] if found_records else None
                duplicates += max(len(found_records) - 1, 0)
        by_engine = [entry["generation_engine"], entry["context_engine"]]
        entry["kv_cache_transfer_ms"] = _engine_reading(
            by_engine, lambda r: _dict(r.get("durations_ms")).get("kv_cache_transfer")
        )
        entry["kv_cache_size"] = _engine_reading(by_engine, lambda r: r.get("kv_cache_size"))
        entry["missing"] = [part for part in REQUIRED[require] if entry[part] is None]
        entry["duplicates"] = duplicates
        entry["unassigned"] = unassigned
        entry["legs_by_arrival"] = by_arrival
        joined.append(entry)
    return joined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--require", choices=sorted(REQUIRED), default="engines")
    parser.add_argument("--output", type=Path, help="the joined JSONL (default: standard output)")
    args = parser.parse_args(argv)
    missing = [str(path) for path in args.directories if not path.is_dir()]
    if missing:
        print(f"not a directory: {', '.join(missing)}", file=sys.stderr)
        return 1
    found = load(args.directories)
    counts = {source: len(records) for source, records in found.items()}
    joined = join(found, args.require)
    complete = sum(1 for entry in joined if not entry["missing"])
    lines = "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in joined)
    if args.output is None:
        sys.stdout.write(lines)
    else:
        args.output.write_text(lines, encoding="utf-8")
    print(
        f"records: orchestrator {counts['orchestrator']}, frontend {counts['frontend']}, "
        f"engine {counts['engine']}; requests {len(joined)}, complete ({args.require}) {complete}",
        file=sys.stderr,
    )
    if not any(counts.values()):
        print("no perf records found", file=sys.stderr)
        return 1
    if complete == 0:
        print("no request has every required record", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Work-item names and git helpers shared by the PoC-S3 scripts.

A work item lands by pushing its commits to the integration branch once, each
commit carrying the trailer ``PoC3-Task: <item>``. So an item has landed exactly
when the integration branch has a commit with its trailer. A commit names an
item when git parses a ``PoC3-Task`` trailer (the trailer block at the end of
the message) with that value and the message has a line reading exactly
``PoC3-Task: <item>``, the line tasks.md §0.2 looks for with
``git log --grep='^PoC3-Task: <item>$'``. A trailer git accepts in another
spelling (``poc3-task: WT-41``, ``PoC3-Task:WT-42``, a trailing space) is
malformed: the ownership gate rejects the commit and it does not land anything.
A ``PoC3-Task:`` line outside the trailer block names nothing. The branch was
cut from the PoC-S2 branch, whose history carries ``PoC2-Task`` trailers; those
name PoC-S2 items and never count here.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import NamedTuple

TRAILER = "PoC3-Task"
DEFAULT_REF = "fork/feat/engine_integration-stage03"
DS_DEFAULT_REF = "origin/feat/engine_integration-stage03"

# WT-30 .. WT-79 and their fixes. PoC-S2 names (WT-01 .. WT-22, WT-20-R2, ...) do not match.
_TASK_RE = re.compile(r"^(?P<base>WT-[3-7]\d)(?:-fix(?P<fix>[1-9]\d*))?$")


def parse_task(name: str) -> tuple[str, int | None]:
    """Split ``WT-51-fix2`` into ``("WT-51", 2)``; ``WT-51`` gives ``("WT-51", None)``.

    Raises:
        ValueError: Not a PoC-S3 work item name.
    """
    match = _TASK_RE.match(name)
    if match is None:
        raise ValueError(f"not a PoC-S3 work item: {name!r}")
    fix = match.group("fix")
    return match.group("base"), int(fix) if fix else None


def git(repo: str | Path, *args: str) -> str:
    """Run git in ``repo`` and return its stdout."""
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


class TrailerRecord(NamedTuple):
    """The ``PoC3-Task`` trailers of one commit, in message order."""

    sha: str
    values: list[str]  # well-formed: the message has the exact line ``PoC3-Task: <value>``
    malformed: list[str]  # parsed by git, but without that exact line


def _message(repo: str | Path, sha: str) -> list[str]:
    """The lines of the commit message, as stored in the commit object."""
    raw = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "commit", sha], check=True, capture_output=True
    ).stdout.decode("utf-8", errors="replace")
    return raw.partition("\n\n")[2].split("\n")


def trailer_log(repo: str | Path, *log_args: str) -> list[TrailerRecord]:
    """The ``PoC3-Task`` trailers of every commit ``git log <log_args>`` lists."""
    log = git(
        repo,
        "log",
        "--no-show-signature",
        f"--format=%H%x00%(trailers:key={TRAILER},valueonly,separator=%x02)%x01",
        *log_args,
    )
    records = []
    for record in log.split("\x01"):
        record = record.strip("\n")
        if not record:
            continue
        sha, _, joined = record.partition("\0")
        parsed = [v.strip() for v in joined.split("\x02") if v.strip()]
        lines = _message(repo, sha) if parsed else []
        values, malformed = [], []
        for value in parsed:
            line = f"{TRAILER}: {value}"
            if line in lines:
                lines.remove(line)  # each exact line vouches for one trailer
                values.append(value)
            else:
                malformed.append(value)
        records.append(TrailerRecord(sha, values, malformed))
    return records


def landed_tasks(repo: str | Path, ref: str) -> set[str]:
    """The work items with a commit on ``ref``, fixes included under their own names."""
    landed: set[str] = set()
    for record in trailer_log(repo, ref):
        for value in record.values:
            try:
                parse_task(value)
            except ValueError:
                continue
            landed.add(value)
    return landed

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Work-item names and git helpers shared by the PoC-S3 scripts.

A work item lands by pushing its commits to the integration branch once, each
commit carrying the trailer ``PoC3-Task: <item>``. So an item has landed exactly
when the integration branch has a commit with its trailer. The branch was cut
from the PoC-S2 branch, whose history carries ``PoC2-Task`` trailers; those name
PoC-S2 items and never count here.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

TRAILER = "PoC3-Task"
DEFAULT_REF = "fork/feat/engine_integration-stage03"
DS_DEFAULT_REF = "origin/feat/engine_integration-stage03"

# WT-30 .. WT-79 and their fixes. PoC-S2 names (WT-01 .. WT-22, WT-20-R2, ...) do not match.
_TASK_RE = re.compile(r"^(?P<base>WT-[3-7]\d)(?:-fix(?P<fix>[1-9]\d*))?$")
_TRAILER_RE = re.compile(rf"^{TRAILER}: (\S+)$", re.MULTILINE)


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


def trailers(message: str) -> list[str]:
    """The ``PoC3-Task`` trailer values of a commit message."""
    return _TRAILER_RE.findall(message)


def landed_tasks(repo: str | Path, ref: str) -> set[str]:
    """The work items with a commit on ``ref``, fixes included under their own names."""
    log = git(repo, "log", ref, f"--grep=^{TRAILER}: ", "--format=%B%x00")
    landed: set[str] = set()
    for message in log.split("\0"):
        for value in trailers(message):
            try:
                parse_task(value)
            except ValueError:
                continue
            landed.add(value)
    return landed

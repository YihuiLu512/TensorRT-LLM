# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Work-item names and git helpers shared by the PoC-S2 scripts.

A work item lands by pushing its commits to the integration branch once, each
commit carrying the trailer ``PoC2-Task: <item>``. So an item has landed exactly
when the integration branch has a commit with its trailer.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

TRAILER = "PoC2-Task"
DEFAULT_REF = "fork/feat/engine_integration"

# WT-01 .. WT-16 (no WT-08), the integration rounds, WT-21 and WT-22, and fixes.
_TASK_RE = re.compile(r"^(?P<base>WT-(?:0[1-79]|1[0-6]|20-R[23]|21|22))(?:-fix(?P<fix>[1-9]\d*))?$")
_TRAILER_RE = re.compile(rf"^{TRAILER}: (\S+)$", re.MULTILINE)


def parse_task(name: str) -> tuple[str, int | None]:
    """Split ``WT-11-fix2`` into ``("WT-11", 2)``; ``WT-11`` gives ``("WT-11", None)``.

    Raises:
        ValueError: Not a PoC-S2 work item name.
    """
    match = _TASK_RE.match(name)
    if match is None:
        raise ValueError(f"not a PoC-S2 work item: {name!r}")
    fix = match.group("fix")
    return match.group("base"), int(fix) if fix else None


def git(repo: str | Path, *args: str) -> str:
    """Run git in ``repo`` and return its stdout."""
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def trailers(message: str) -> list[str]:
    """The ``PoC2-Task`` trailer values of a commit message."""
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

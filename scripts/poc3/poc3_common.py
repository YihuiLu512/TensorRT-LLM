# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Work-item names and git helpers shared by the PoC-S3 scripts.

A work item lands by pushing its commits to the integration branch once, each
commit carrying the trailer ``PoC3-Task: <item>``. So an item has landed exactly
when the integration branch has a commit with its trailer.

The scripts read the trailer from the raw commit message, not through git's
trailer parsing: a ``PoC3-Task`` line is any line of the message whose key, the
text before the first colon with surrounding spaces ignored, is ``PoC3-Task`` in
any letter case. It names ``<item>`` only when it is in the message's last
paragraph (a message that is only a subject has no trailers), reads exactly
``PoC3-Task: <item>`` and is not folded onto the next line; that
is the line tasks.md §0.2 looks for with ``git log --grep='^PoC3-Task: <item>$'``.
Any other ``PoC3-Task`` line (``poc3-task: WT-41``, ``PoC3-Task:WT-42``, extra
spaces, or an exact line in the subject or an earlier paragraph, which the §0.2
grep would also find) is malformed: the ownership gate rejects the commit, and
the line lands nothing. The branch was cut from the PoC-S2 branch, whose
history carries ``PoC2-Task`` trailers; those name PoC-S2 items and never count
here.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
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


def git_env() -> dict[str, str]:
    """The environment for git, without the variable that sets a diff's context lines."""
    return {key: value for key, value in os.environ.items() if key != "GIT_DIFF_OPTS"}


def git(repo: str | Path, *args: str) -> str:
    """Run git in ``repo`` and return its stdout."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=git_env(),
    ).stdout


def git_error(error: subprocess.CalledProcessError) -> str:
    """One line describing a failed git command: the command and what it printed on stderr."""
    stderr = error.stderr
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    return f"{' '.join(map(str, error.cmd))}: {(stderr or '').strip()}"


class TrailerRecord(NamedTuple):
    """The ``PoC3-Task`` lines of one commit, in message order."""

    sha: str
    values: list[str]  # the items named by lines reading exactly ``PoC3-Task: <item>``
    malformed: list[str]  # the other ``PoC3-Task`` lines (their stripped value, or the line)


def task_lines(message: str) -> tuple[list[str], list[str]]:
    """The well-formed values and the malformed ``PoC3-Task`` lines of a commit message."""
    lines = message.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    # The last paragraph; a message of one paragraph (the subject) has no trailers.
    start = max((i + 1 for i, line in enumerate(lines) if not line.strip()), default=len(lines))
    values, malformed = [], []
    for i, line in enumerate(lines):
        key, colon, rest = line.partition(":")
        if not colon or key.strip().lower() != TRAILER.lower():
            continue
        value = rest.strip()
        folded = i + 1 < len(lines) and lines[i + 1][:1] in (" ", "\t")
        if i >= start and value and not folded and line == f"{TRAILER}: {value}":
            values.append(value)
        else:
            malformed.append(value or line)
    return values, malformed


def _messages(repo: str | Path, shas: list[str]) -> dict[str, str]:
    """The raw commit message of each of ``shas``."""
    out = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input="".join(f"{sha}\n" for sha in shas).encode(),
        check=True,
        capture_output=True,
    ).stdout
    messages, pos = {}, 0
    for sha in shas:
        header_end = out.index(b"\n", pos)
        name, kind, size = out[pos:header_end].decode().split()
        if kind != "commit":
            raise ValueError(f"{sha} is a {kind}, not a commit")
        body = out[header_end + 1 : header_end + 1 + int(size)]
        messages[sha] = body.decode("utf-8", errors="replace").partition("\n\n")[2]
        pos = header_end + 1 + int(size) + 1
    return messages


def trailer_log(repo: str | Path, *log_args: str) -> list[TrailerRecord]:
    """The ``PoC3-Task`` lines of every commit ``git rev-list <log_args>`` lists."""
    # --encoding: --grep matches the message in the output encoding (i18n.logOutputEncoding).
    shas = git(repo, "rev-list", "--encoding=UTF-8", *log_args).split()
    messages = _messages(repo, shas) if shas else {}
    return [TrailerRecord(sha, *task_lines(messages[sha])) for sha in shas]


def landed_tasks(repo: str | Path, ref: str) -> set[str]:
    """The work items with a commit on ``ref``, fixes included under their own names.

    Malformed ``PoC3-Task`` lines, and lines naming something that is not a
    PoC-S3 work item, are reported on stderr and not counted.
    """
    landed: set[str] = set()
    # Only a superset of the commits with a PoC3-Task line is read; the options pin the match.
    candidates = ("--fixed-strings", "--regexp-ignore-case", f"--grep={TRAILER}")
    for record in trailer_log(repo, *candidates, ref):
        for value in record.malformed:
            print(
                f"warning: {record.sha[:12]} has a malformed {TRAILER} line ({value!r});"
                " it is not counted as landed",
                file=sys.stderr,
            )
        for value in record.values:
            try:
                parse_task(value)
            except ValueError:
                print(
                    f"warning: {record.sha[:12]} names {value!r}, which is not a PoC-S3 work item;"
                    " it is not counted as landed",
                    file=sys.stderr,
                )
                continue
            landed.add(value)
    return landed

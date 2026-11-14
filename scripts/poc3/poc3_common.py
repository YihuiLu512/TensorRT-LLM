# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Work-item names and git helpers shared by the PoC-S3 scripts.

A work item lands by pushing its commits to the integration branch once, each
commit carrying the trailer ``PoC3-Task: <item>``. So an item has landed exactly
when the integration branch has a commit with its trailer.

The scripts read the trailer from the raw commit message, not through git's
trailer parsing: a ``PoC3-Task`` line is any line of the message whose key, the
text before the first colon with surrounding whitespace ignored, is ``PoC3-Task`` in
any letter case. It names ``<item>`` only when it is in the message's last
paragraph (a message that is only a subject has no trailers; as in git, a
blank line holds only spaces, tabs and carriage returns, and blank lines before
the subject are skipped), reads exactly ``PoC3-Task: <item>`` and is not folded onto the
next line; that is the line tasks.md §0.2 looks for with
``git log --grep='^PoC3-Task: <item>$'``. Any other ``PoC3-Task`` line
(``poc3-task: WT-41``, ``PoC3-Task:WT-42``, extra spaces, or an exact line in
the subject or an earlier paragraph, which the §0.2 grep would also find) is
malformed: the ownership gate rejects the commit, and the line lands nothing.
So is every ``PoC3-Task`` line of a commit that declares an encoding other than
UTF-8 or holds a NUL byte anywhere: ``git log --grep`` reads such a message
re-encoded or not at all. The branch was cut from the PoC-S2 branch, whose
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
_TASK_RE = re.compile(r"^(?P<base>WT-[3-7][0-9])(?:-fix(?P<fix>[1-9][0-9]*))?$")


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


# Errors a YAML document the scripts read may raise besides yaml.YAMLError (measured with PyYAML
# 5.4 and 6.0): ValueError for a value it cannot convert (the date 2026-02-30) or text that is
# not UTF-8, LookupError (KeyError, IndexError) and AttributeError for an explicit tag it cannot
# apply (!!bool x, !!int '-', !!timestamp x), RuntimeError (RecursionError) for nesting too deep.
YAML_LOAD_ERRORS = (ValueError, LookupError, AttributeError, RuntimeError)

# Environment variables that change how git reads pathspecs or how many context lines a diff has.
# GIT_DIR and GIT_WORK_TREE are kept: a caller sets them to name the repository on purpose.
_GIT_ENV_DROPPED = (
    "GIT_DIFF_OPTS",
    "GIT_LITERAL_PATHSPECS",
    "GIT_GLOB_PATHSPECS",
    "GIT_NOGLOB_PATHSPECS",
    "GIT_ICASE_PATHSPECS",
)


def git_env() -> dict[str, str]:
    """The environment for every git command the scripts run.

    It drops ``_GIT_ENV_DROPPED`` and turns off replace refs (``refs/replace/``)
    and grafts (``info/grafts``), so git reads the objects and parents as they
    are. Other local repository state (a shallow clone, the commit-graph file,
    alternates) is trusted.
    """
    env = {key: value for key, value in os.environ.items() if key not in _GIT_ENV_DROPPED}
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    env["GIT_GRAFT_FILE"] = os.devnull
    return env


def run_git(
    repo: str | Path, *args: str, check: bool = True, stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Run git in ``repo`` with :func:`git_env`; stdout and stderr are captured as bytes."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        input=stdin,
        env=git_env(),
    )


def git(repo: str | Path, *args: str) -> str:
    """Run git in ``repo`` and return its stdout, decoded as UTF-8."""
    return run_git(repo, *args).stdout.decode()


def git_error(error: subprocess.CalledProcessError) -> str:
    """One line describing a failed git command: the command and what it printed on stderr."""
    stderr = error.stderr
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    return f"{' '.join(map(str, error.cmd))}: {(stderr or '').strip()}"


class TrailerRecord(NamedTuple):
    """The ``PoC3-Task`` lines of one commit, in message order."""

    sha: str
    values: list[str]  # the items named by well-formed lines (see the module docstring)
    malformed: list[str]  # the other ``PoC3-Task`` lines (their stripped value, or the line)
    # False when git log does not read the message as it is (see the module docstring); then
    # every PoC3-Task line is malformed.
    readable: bool = True


# What git's isspace() accepts; a line of form feeds or no-break spaces is not blank to git.
_GIT_WHITESPACE = " \t\n\r"


def _blank(line: str) -> bool:
    return not line.strip(_GIT_WHITESPACE)


def _task_lines(message: str) -> list[tuple[str, bool]]:
    """Each ``PoC3-Task`` line of ``message`` in order, as (value or line, well-formed)."""
    lines = message.split("\n")
    while lines and _blank(lines[-1]):
        lines.pop()
    while lines and _blank(lines[0]):
        lines.pop(0)
    # The last paragraph; a message of one paragraph (the subject) has no trailers.
    start = max((i + 1 for i, line in enumerate(lines) if _blank(line)), default=len(lines))
    found = []
    for i, line in enumerate(lines):
        key, colon, rest = line.partition(":")
        if not colon or key.strip().lower() != TRAILER.lower():
            continue
        value = rest.strip()
        folded = i + 1 < len(lines) and lines[i + 1][:1] in (" ", "\t")
        well_formed = i >= start and value and not folded and line == f"{TRAILER}: {value}"
        found.append((value or line, bool(well_formed)))
    return found


def task_lines(message: str) -> tuple[list[str], list[str]]:
    """The well-formed values and the malformed ``PoC3-Task`` lines of a commit message."""
    found = _task_lines(message)
    return [v for v, ok in found if ok], [v for v, ok in found if not ok]


# Why every PoC3-Task line of a commit that is not ``readable`` is malformed.
UNREADABLE = "the commit declares an encoding other than UTF-8 or holds a NUL byte"


def _messages(repo: str | Path, shas: list[str]) -> dict[str, tuple[str, bool]]:
    """The raw commit message of each of ``shas``, and whether git log reads it as it is.

    git log re-encodes a message whose commit declares another encoding, and
    its ``--grep`` stops at a NUL byte, in the message or in a header.
    """
    out = run_git(repo, "cat-file", "--batch", stdin="".join(f"{sha}\n" for sha in shas).encode())
    messages, pos = {}, 0
    for sha in shas:
        header_end = out.stdout.index(b"\n", pos)
        name, kind, size = out.stdout[pos:header_end].decode().split()
        if kind != "commit":
            raise ValueError(f"{sha} is a {kind}, not a commit")
        body = out.stdout[header_end + 1 : header_end + 1 + int(size)]
        headers, _, message = body.partition(b"\n\n")
        encodings = [
            line[len(b"encoding ") :].strip().lower()
            for line in headers.split(b"\n")
            if line.startswith(b"encoding ")
        ]
        readable = b"\0" not in body and all(e in (b"utf-8", b"utf8") for e in encodings)
        messages[sha] = (message.decode("utf-8", errors="replace"), readable)
        pos = header_end + 1 + int(size) + 1
    return messages


def trailer_log(repo: str | Path, *log_args: str) -> list[TrailerRecord]:
    """The ``PoC3-Task`` lines of every commit ``git rev-list <log_args>`` lists."""
    # --encoding: --grep matches the message in the output encoding (i18n.logOutputEncoding).
    shas = git(repo, "rev-list", "--encoding=UTF-8", *log_args).split()
    messages = _messages(repo, shas) if shas else {}
    records = []
    for sha in shas:
        message, readable = messages[sha]
        if readable:
            records.append(TrailerRecord(sha, *task_lines(message)))
        else:
            lines = [value for value, _ in _task_lines(message)]
            records.append(TrailerRecord(sha, [], lines, readable=False))
    return records


def landed_tasks(repo: str | Path, ref: str) -> set[str]:
    """The work items with a commit on ``ref``, fixes included under their own names.

    Malformed ``PoC3-Task`` lines, and lines naming something that is not a
    PoC-S3 work item, are reported on stderr and not counted.
    """
    landed: set[str] = set()
    # Only a superset of the commits whose PoC3-Task lines can count is read (a message with a
    # NUL byte can hide its lines from --grep, but such lines never count); the options pin the
    # match.
    candidates = ("--fixed-strings", "--regexp-ignore-case", f"--grep={TRAILER}")
    for record in trailer_log(repo, *candidates, ref):
        why = "" if record.readable else f" ({UNREADABLE})"
        for value in record.malformed:
            print(
                f"warning: {record.sha[:12]} has a malformed {TRAILER} line ({value!r}){why};"
                " the line is not counted",
                file=sys.stderr,
            )
        for value in record.values:
            try:
                parse_task(value)
            except ValueError:
                print(
                    f"warning: {record.sha[:12]} names {value!r}, which is not a PoC-S3 work item;"
                    " the line is not counted",
                    file=sys.stderr,
                )
                continue
            landed.add(value)
    return landed

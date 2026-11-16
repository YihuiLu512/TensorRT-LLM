# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of the exceptions check_ownership.py keeps for WT-30-fix15 in the records.

Some verification records and probe records on the integration branch name
local paths. The exceptions allow WT-30-fix15 alone, and only until it has
landed, to replace the start of each such path (``/home/<disk>``) by a
placeholder, line by line in place, in the files ``REDACTED_RECORDS`` and
``REDACTED_SCRIPTS`` name: ``<scratch-1>`` for a disk whose name ends in
``_1`` and ``<scratch>`` for any other in the records, ``${SCRATCH1:?}`` and
``${SCRATCH:?}`` in the driver scripts, which may also get one fixed line, as
their second line, saying what they stand for. Nothing else in those files may change: no
entry added, removed or edited. Run from the repository root::

    python3 -m pytest -p no:cacheprovider scripts/poc3/
"""

from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

# The scripts import poc3_common themselves; pytest puts this directory on sys.path.
import check_ownership  # noqa: I001
import pytest
import yaml

FIX = "WT-30-fix15"
CHAIN = check_ownership.CHAIN
RECORDS = check_ownership.REDACTED_RECORDS
SCRIPTS = check_ownership.REDACTED_SCRIPTS
HEADER = check_ownership.REDACTED_HEADER
PLACEHOLDERS = {False: ("<scratch-1>", "<scratch>"), True: ("${SCRATCH1:?}", "${SCRATCH:?}")}
# A stage03 commit with the files as they are before WT-30-fix15, and the git blob id of each file
# as WT-30-fix15 leaves it.
BASE = "e74f136fdb0401c2d92f9a1a28cbb7fa2a05b62f"
HEAD_BLOBS = {
    f"{CHAIN}/records/WT-31/README.md": "7f7208764370b9d2c053e4b00169e2c2daf97873",
    f"{CHAIN}/records/WT-31/scripts/collect.sbatch": "f0ea9d427c6594cf4e5337cf228f7e7422a015bb",
    f"{CHAIN}/records/WT-31/scripts/collect.sh": "4a17c0584c542e66c97506714fad484faacf356e",
    f"{CHAIN}/records/WT-31/scripts/r10_shard.sbatch": "fd95942f9e310cb63393264a97302e53aa7a00ca",
    f"{CHAIN}/records/WT-31/scripts/submit.sh": "9f990068a56be242ed8869122adbca655e0592d7",
    f"{CHAIN}/records/WT-33/README.md": "a7e9c50be3f7b6b2ad6608cef698bd4fde8ef316",
    f"{CHAIN}/records/WT-34/README.md": "cabe2c14b86998d2924518cba7686986e1565df5",
    f"{CHAIN}/verified/WT-03-fix1.yaml": "98ccdaaa3119c44c3aaf49bcc2c76be77e1aa350",
    f"{CHAIN}/verified/WT-03.yaml": "556fc683ffff2343aead5534b19f95a74878dd84",
    f"{CHAIN}/verified/WT-05-fix1.yaml": "bbef75a3e3c3afee583d96cb4d534f6640e90307",
    f"{CHAIN}/verified/WT-09-fix1.yaml": "b058ba8863485d7d1ef25c684822b2dbe28af55d",
    f"{CHAIN}/verified/WT-10-fix1.yaml": "7b5eb88973d257156988426b84af7853cd3d83b4",
    f"{CHAIN}/verified/WT-10.yaml": "3f0d914cf98a922c2664352755d3f99aee2a8037",
    f"{CHAIN}/verified/WT-11.yaml": "e6ac4ec755976949c3d89f9ad3b8142edcd4f383",
    f"{CHAIN}/verified/WT-12.yaml": "c00469fe7cdf0d549d2cf74af92417efa0c5f489",
    f"{CHAIN}/verified/WT-13.yaml": "6ab5b3bdb9b1dea4909e9870269ac6c5efc74d3d",
    f"{CHAIN}/verified/WT-15.yaml": "c06ea47275f6af5f60da9ef25ed9afd2496d65e1",
    f"{CHAIN}/verified/WT-16.yaml": "d675c950aaa476f96ff9b028743724911e0b44bb",
    f"{CHAIN}/verified/WT-20-R2.yaml": "c2e5f09c316092cc6cec8f9eafb450e7c63a42a8",
    f"{CHAIN}/verified/WT-20-R3.yaml": "77888e5faf096f4a98402722f50a5d1c8ea9924c",
}
# What WT-30-fix15 replaces at BASE: /home and the name of a disk under it.
LOCAL_START = re.compile(r"/home/[\w.-]+")
RECORD = f"{CHAIN}/verified/WT-10.yaml"
SCRIPT = f"{CHAIN}/records/WT-31/scripts/submit.sh"
# A made-up record and script, with made-up local paths.
RECORD_BASE = """\
# SPDX-License-Identifier: Apache-2.0
entries:
  - row: F-01
    features: [F-01]
    cases: [x/y/a.py::test_a]
    # https://example.com/a/b/c
    commit: 0123456789abcdef0123456789abcdef01234567
    junit:
      - /home/disk.someone/results/p1/outer-junit.xml
      - /home/disk.someone/results/p1/junit/
    result: passed
  - row: F-02
    features: [F-02]
    cases: [b.py::test_b]
    commit: 0123456789abcdef0123456789abcdef01234567
    junit:
      - /home/disk.someone/results/p2/outer-junit.xml
    result: failed
"""
SCRIPT_BASE = """\
#!/bin/bash
# Usage: submit.sh <key>
BASE=${BASE:-/home/disk.someone/tmp/run}
MOUNTS=${MOUNTS:-/home/disk.someone:/home/disk.someone_1}
MPI=/opt/hpcx/ompi
echo "$BASE"
"""


def _redacted(text: str, script: bool = False) -> str:
    """``text`` with the made-up local paths of RECORD_BASE or SCRIPT_BASE redacted."""
    disk_1, disk = PLACEHOLDERS[script]
    return text.replace("/home/disk.someone_1", disk_1).replace("/home/disk.someone", disk)


def _with_header(text: str) -> str:
    first, rest = text.split("\n", 1)
    return f"{first}\n{HEADER}\n{rest}"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@lru_cache(maxsize=None)
def _show(obj: str) -> str:
    """The blob ``obj`` (``<rev>:<path>`` or a blob id) of this checkout.

    Skips the test when the checkout lacks it.
    """
    top = Path(__file__).resolve().parents[2]
    known = subprocess.run(["git", "-C", str(top), "cat-file", "-e", obj], capture_output=True)
    if known.returncode != 0:
        pytest.skip(f"{obj} is not in this checkout")
    return _git(top, "cat-file", "-p", obj)


def _write(path: Path, content: str | bytes) -> None:
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


def _repo_with(
    tmp_path: Path,
    files: dict[str, str | bytes],
    heads: dict[str, str | bytes | None],
    task: str,
    landed: str = "",
) -> tuple[Path, str]:
    """A repo holding ``files``, then ``heads`` (None: deleted) in a commit of ``task``.

    With ``landed``, the base commit carries that work item's trailer.
    """
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / "tensorrt_llm").mkdir()
    (repo / "tensorrt_llm/__init__.py").write_text("")
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        _write(repo / path, text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"base\n\nPoC3-Task: {landed}" if landed else "base")
    base_sha = _git(repo, "rev-parse", "HEAD").strip()
    for path, text in heads.items():
        if text is None:
            (repo / path).unlink()
        else:
            (repo / path).parent.mkdir(parents=True, exist_ok=True)
            _write(repo / path, text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"t\n\nPoC3-Task: {task}")
    return repo, base_sha


def _check(tmp_path, capsys, files, heads, task=FIX, landed=""):
    """Check ``task`` changing ``files`` to ``heads``: the exit status and the output."""
    repo, sha = _repo_with(tmp_path, files, heads, task, landed)
    found = check_ownership.check(task, f"{sha}..HEAD", str(repo))
    return found, capsys.readouterr().out


def _refused(found: int, out: str, path: str) -> None:
    """The check refused a change of ``path`` under the exception for WT-30-fix15."""
    assert found == 1, out
    assert f"OUT OF BOUNDS  {path}  (the exception allows U41 only" in out


def _local_starts_replaced(value, script: bool = False):
    """``value`` (text, or data read from YAML) with the start of every local path replaced.

    As WT-30-fix15 replaces it: ``/home/<disk>_1`` by ``<scratch-1>`` (``${SCRATCH1:?}`` in a
    script), any other ``/home/<disk>`` by ``<scratch>`` (``${SCRATCH:?}``).
    """
    disk_1, disk = PLACEHOLDERS[script]
    if isinstance(value, str):
        return LOCAL_START.sub(lambda m: disk_1 if m[0].endswith("_1") else disk, value)
    if isinstance(value, list):
        return [_local_starts_replaced(v, script) for v in value]
    if isinstance(value, dict):
        return {k: _local_starts_replaced(v, script) for k, v in value.items()}
    return value


# --- the exceptions -----------------------------------------------------------------------------


def test_the_exceptions_name_exactly_the_files_with_local_paths():
    assert len(RECORDS) == len(set(RECORDS)) == 16 and len(SCRIPTS) == len(set(SCRIPTS)) == 4
    assert set(RECORDS) | set(SCRIPTS) == set(HEAD_BLOBS)
    _show(BASE)  # skips the test when the checkout lacks BASE
    top = Path(__file__).resolve().parents[2]
    found = _git(
        top, "grep", "-l", "-I", "/home/", BASE, "--", f"{CHAIN}/verified", f"{CHAIN}/records"
    )
    assert {line.split(":", 1)[1] for line in found.split()} == set(HEAD_BLOBS)


@pytest.mark.parametrize("path", sorted(HEAD_BLOBS))
def test_each_exception_is_for_wt30_fix15_alone(path):
    rule = check_ownership.match_rule(FIX, path)
    assert rule.kind == "exception" and rule.item == FIX
    disk_1, disk = PLACEHOLDERS[path in SCRIPTS]
    assert [p for p, _ in rule.redact] == [disk_1, disk]
    assert rule.header == (HEADER if path in SCRIPTS else "")
    assert not rule.lines and not rule.changes and not rule.renames
    assert (FIX, rule) in check_ownership.owners(path)
    for other in ("WT-30", "WT-30-fix14", "WT-30-fix16", "WT-31", "WT-37"):
        found = check_ownership.match_rule(other, path)
        assert found is None or (found.kind == "own" and not found.item), (other, found)


def test_an_exception_listing_placeholders_names_its_work_item():
    rule = check_ownership._exception("x.yaml", "test", redact=(("<x>", "/x"),))
    with pytest.raises(ValueError, match="an exception listing placeholders names no item"):
        check_ownership._named_fixes({"WT-30": [rule]})


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"redact": (("<x>", "/x"),), "lines": ("a",)}, "one of lines, changes and redact"),
        ({"header": "# h"}, "adds a header line only with redact"),
        ({"redact": (("", "/x"),)}, "placeholders and their patterns are not empty"),
        ({"redact": (("<x>", ""),)}, "placeholders and their patterns are not empty"),
        ({"redact": (("<x>", "("),)}, "missing \\)"),
    ],
    ids=["redact-and-lines", "header-alone", "empty-placeholder", "empty-pattern", "bad-pattern"],
)
def test_a_redact_exception_is_well_formed(kwargs, message):
    with pytest.raises((ValueError, re.error), match=message):
        check_ownership._exception("x.yaml", "test", item=FIX, **kwargs)


# --- the redacted records -----------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(HEAD_BLOBS))
def test_wt30_fix15_may_redact_each_file(tmp_path, capsys, path):
    base, head = _show(f"{BASE}:{path}"), _show(HEAD_BLOBS[path])
    found, out = _check(tmp_path, capsys, {path: base}, {path: head})
    assert found == 0, out
    assert f"CHECK BY HAND  EXCEPTION  {path}  -- U41 only" in out


@pytest.mark.parametrize("task", ["WT-30", "WT-30-fix14", "WT-30-fix16"])
def test_no_other_fix_may_redact_the_files(tmp_path, capsys, task):
    path = RECORD
    found, out = _check(
        tmp_path, capsys, {path: _show(f"{BASE}:{path}")}, {path: _show(HEAD_BLOBS[path])}, task
    )
    assert found == 1 and f"OUT OF BOUNDS  {path}  (owned by {FIX})" in out, out


@pytest.mark.parametrize("path", sorted(p for p in HEAD_BLOBS if p.endswith(".yaml")))
def test_each_verification_record_asserts_what_it_did(path):
    base = yaml.safe_load(_show(f"{BASE}:{path}"))
    head = yaml.safe_load(_show(HEAD_BLOBS[path]))
    assert head != base and head == _local_starts_replaced(base)
    assert not LOCAL_START.search(_show(HEAD_BLOBS[path]))


@pytest.mark.parametrize("path", sorted(p for p in HEAD_BLOBS if not p.endswith(".yaml")))
def test_each_other_record_differs_only_by_the_start_of_its_local_paths(path):
    base, head = _show(f"{BASE}:{path}"), _show(HEAD_BLOBS[path])
    if path in SCRIPTS:
        assert head.split("\n")[1] == HEADER
        head = "\n".join(head.split("\n")[:1] + head.split("\n")[2:])
    assert head != base and head == _local_starts_replaced(base, path in SCRIPTS)
    assert not LOCAL_START.search(head)


# --- what the exceptions allow, on made-up records ----------------------------------------------


def _one_line_redacted(text: str) -> str:
    return text.replace(
        "/home/disk.someone/results/p2/outer-junit.xml", "<scratch>/results/p2/outer-junit.xml"
    )


# SCRIPT_BASE with a new second line that names a local path.
SCRIPT_WITH_PATH_ON_LINE_2 = SCRIPT_BASE.replace(
    "# Usage: submit.sh <key>\n", "LOG=/home/disk.someone/log\n# Usage: submit.sh <key>\n"
)


@pytest.mark.parametrize(
    "path, base, head",
    [
        (RECORD, RECORD_BASE, _redacted(RECORD_BASE)),
        (RECORD, RECORD_BASE, _one_line_redacted(RECORD_BASE)),
        (SCRIPT, SCRIPT_BASE, _with_header(_redacted(SCRIPT_BASE, True))),
        (SCRIPT, SCRIPT_BASE, _redacted(SCRIPT_BASE, True)),
        (SCRIPT, SCRIPT_BASE, _with_header(SCRIPT_BASE)),
    ],
    ids=["record", "record-one-line", "script-and-header", "script-no-header", "header-alone"],
)
def test_wt30_fix15_may_redact_local_paths(tmp_path, capsys, path, base, head):
    found, out = _check(tmp_path, capsys, {path: base}, {path: head})
    assert found == 0, out


# Each changes RECORD_BASE in a way the exception does not allow; the record would assert
# something else, or its paths would keep a local disk or point somewhere else.
TAMPERED_RECORDS = {
    "result": _redacted(RECORD_BASE).replace("result: failed", "result: passed"),
    "result-on-a-path-line": _redacted(RECORD_BASE).replace(
        "      - <scratch>/results/p2/outer-junit.xml\n    result: failed",
        "      - <scratch>/results/p2/outer-junit.xml\n    result: passed",
    ),
    "commit": _redacted(RECORD_BASE).replace("0123456789abcdef", "fedcba9876543210", 1),
    "row": _redacted(RECORD_BASE).replace("row: F-02", "row: F-03"),
    "entry-added": _redacted(RECORD_BASE)
    + "  - row: F-03\n    junit:\n      - <scratch>/results/p3/outer-junit.xml\n    result: passed\n",
    "entry-removed": _redacted(RECORD_BASE).split("  - row: F-02")[0],
    "path-line-removed": _redacted(RECORD_BASE).replace(
        "      - <scratch>/results/p1/junit/\n", ""
    ),
    "path-line-added": _redacted(RECORD_BASE).replace(
        "      - <scratch>/results/p1/junit/\n",
        "      - <scratch>/results/p1/junit/\n      - <scratch>/results/p9/junit/\n",
    ),
    "path-tail": _redacted(RECORD_BASE).replace("results/p1/outer", "results/p9/outer"),
    "three-components": _redacted(RECORD_BASE).replace(
        "<scratch>/results/p1/outer-junit.xml", "<scratch>/p1/outer-junit.xml"
    ),
    "one-component": RECORD_BASE.replace(
        "/home/disk.someone/results/p1/outer-junit.xml",
        "/data/<scratch>/results/p1/outer-junit.xml",
    ),
    "another-root": RECORD_BASE.replace("/home/disk.someone/", "/srv/disk.other/"),
    "unknown-placeholder": RECORD_BASE.replace("/home/disk.someone/", "<disk>/"),
    "indent": _redacted(RECORD_BASE).replace(
        "      - <scratch>/results/p1/junit/", "    - <scratch>/results/p1/junit/"
    ),
    "lines-swapped": _redacted(RECORD_BASE).replace(
        "      - <scratch>/results/p1/outer-junit.xml\n      - <scratch>/results/p1/junit/\n",
        "      - <scratch>/results/p1/junit/\n      - <scratch>/results/p1/outer-junit.xml\n",
    ),
    "header": _with_header(_redacted(RECORD_BASE)),
    "no-final-newline": _redacted(RECORD_BASE).rstrip("\n"),
    # Path starts the placeholders do not stand for, or that do not start a path.
    "case-id": _redacted(RECORD_BASE).replace("[x/y/a.py::test_a]", "[x<scratch>/a.py::test_a]"),
    "url": _redacted(RECORD_BASE).replace("https://example.com/a/b/c", "https:/<scratch>/b/c"),
    "user-name-kept": RECORD_BASE.replace(
        "/home/disk.someone/results/p1/outer-junit.xml",
        "/home/disk.someone<scratch>/outer-junit.xml",
    ),
    "part-of-the-disk": RECORD_BASE.replace(
        "/home/disk.someone/results/p1/outer-junit.xml",
        "<scratch>.someone/results/p1/outer-junit.xml",
    ),
    "other-disk": RECORD_BASE.replace("/home/disk.someone/", "<scratch-1>/"),
    "script-placeholder": _redacted(RECORD_BASE, True),
}


@pytest.mark.parametrize("tamper", sorted(TAMPERED_RECORDS))
def test_wt30_fix15_may_change_nothing_else_in_a_record(tmp_path, capsys, tamper):
    head = TAMPERED_RECORDS[tamper]
    assert head != RECORD_BASE
    found, out = _check(tmp_path, capsys, {RECORD: RECORD_BASE}, {RECORD: head})
    _refused(found, out, RECORD)


def _lines(text: str) -> list[str]:
    return text.split("\n")


TAMPERED_SCRIPTS = {
    "header-as-third-line": "\n".join(
        _lines(_redacted(SCRIPT_BASE, True))[:2]
        + [HEADER]
        + _lines(_redacted(SCRIPT_BASE, True))[2:]
    ),
    "header-first": f"{HEADER}\n{_redacted(SCRIPT_BASE, True)}",
    "header-twice": _with_header(_with_header(_redacted(SCRIPT_BASE, True))),
    "other-header": _redacted(SCRIPT_BASE, True).replace("\n", "\n# Paths are redacted.\n", 1),
    "header-at-the-end": _redacted(SCRIPT_BASE, True) + HEADER + "\n",
    "header-for-a-line": _redacted(SCRIPT_BASE, True).replace("# Usage: submit.sh <key>", HEADER),
    "header-and-command": _with_header(_redacted(SCRIPT_BASE, True)).replace(
        '"$BASE"', '"$MOUNTS"'
    ),
    "record-placeholder": _with_header(_redacted(SCRIPT_BASE)),
    "other-disk": _redacted(SCRIPT_BASE, True).replace("${SCRATCH1:?}", "${SCRATCH:?}"),
    "container-path": _redacted(SCRIPT_BASE, True).replace("/opt/hpcx/ompi", "${SCRATCH:?}/ompi"),
}


@pytest.mark.parametrize("tamper", sorted(TAMPERED_SCRIPTS))
def test_wt30_fix15_may_add_only_the_header_as_the_second_line(tmp_path, capsys, tamper):
    found, out = _check(tmp_path, capsys, {SCRIPT: SCRIPT_BASE}, {SCRIPT: TAMPERED_SCRIPTS[tamper]})
    _refused(found, out, SCRIPT)


def test_a_placeholder_stands_only_for_a_path_that_starts_at_home(tmp_path, capsys):
    # /home/y inside a relative path, with the rest of the record redacted as allowed.
    base = RECORD_BASE.replace("[x/y/a.py::test_a]", "[x/home/y/a.py::test_a]")
    head = _redacted(base).replace("[x/home/y/a.py::test_a]", "[x<scratch>/a.py::test_a]")
    found, out = _check(tmp_path, capsys, {RECORD: base}, {RECORD: head})
    _refused(found, out, RECORD)


# A placeholder stands for a whole disk name that starts with a word character, never for "." or
# "..", nor for part of a name that goes on with another character.
@pytest.mark.parametrize(
    "disk, head_start",
    [
        (".", "<scratch>/"),
        ("..", "<scratch>/"),
        ("disk+someone", "<scratch>+someone/"),
        ("disk@someone", "<scratch>@someone/"),
        ("disk\u00e9", "<scratch>\u00e9/"),
    ],
    ids=["dot", "dot-dot", "plus", "at", "non-ascii"],
)
def test_a_placeholder_stands_for_a_whole_disk_name(tmp_path, capsys, disk, head_start):
    base = RECORD_BASE.replace("/home/disk.someone/", f"/home/{disk}/")
    head = RECORD_BASE.replace("/home/disk.someone/", head_start)
    found, out = _check(tmp_path, capsys, {RECORD: base}, {RECORD: head})
    _refused(found, out, RECORD)


def test_a_hunk_adding_the_header_and_redacting_the_second_line_is_refused(tmp_path, capsys):
    # -U0 makes one hunk of the two; only a hunk that adds the header alone is allowed.
    head = _with_header(_redacted(SCRIPT_WITH_PATH_ON_LINE_2, True))
    found, out = _check(tmp_path, capsys, {SCRIPT: SCRIPT_WITH_PATH_ON_LINE_2}, {SCRIPT: head})
    _refused(found, out, SCRIPT)
    found, out = _check(
        tmp_path / "again",
        capsys,
        {SCRIPT: SCRIPT_WITH_PATH_ON_LINE_2},
        {SCRIPT: _redacted(SCRIPT_WITH_PATH_ON_LINE_2, True)},
    )
    assert found == 0, out


# A byte that is not UTF-8 reads as U+FFFD whatever it is, so a line with one is not redacted, even
# with the same byte kept.
@pytest.mark.parametrize("byte", [b"\xff", b"\xfe"], ids=["same-byte", "other-byte"])
def test_a_line_that_is_not_utf8_is_not_redacted(tmp_path, capsys, byte):
    base = RECORD_BASE.encode().replace(b"p2/outer-junit.xml", b"p2/outer-\xff.xml")
    head = (
        _redacted(RECORD_BASE)
        .encode()
        .replace(b"p2/outer-junit.xml", b"p2/outer-" + byte + b".xml")
    )
    found, out = _check(tmp_path, capsys, {RECORD: base}, {RECORD: head})
    _refused(found, out, RECORD)


def test_wt30_fix15_may_not_delete_a_record(tmp_path, capsys):
    found, out = _check(tmp_path, capsys, {RECORD: RECORD_BASE}, {RECORD: None})
    _refused(found, out, RECORD)


def test_wt30_fix15_may_not_add_a_record(tmp_path, capsys):
    other = f"{CHAIN}/verified/WT-13.yaml"
    found, out = _check(tmp_path, capsys, {other: RECORD_BASE}, {RECORD: _redacted(RECORD_BASE)})
    _refused(found, out, RECORD)


def test_wt30_fix15_may_not_change_the_mode_of_a_script(tmp_path, capsys):
    repo, sha = _repo_with(
        tmp_path, {SCRIPT: SCRIPT_BASE}, {SCRIPT: _redacted(SCRIPT_BASE, True)}, FIX
    )
    (repo / SCRIPT).chmod(0o755)
    _git(repo, "commit", "-q", "--amend", "-a", "--no-edit")
    found = check_ownership.check(FIX, f"{sha}..HEAD", str(repo))
    _refused(found, capsys.readouterr().out, SCRIPT)


def test_wt30_fix15_may_not_redact_other_records(tmp_path, capsys):
    other = f"{CHAIN}/verified/WT-37.yaml"
    found, out = _check(tmp_path, capsys, {other: RECORD_BASE}, {other: _redacted(RECORD_BASE)})
    assert found == 1 and f"OUT OF BOUNDS  {other}  (owned by WT-37)" in out, out


# --- once WT-30-fix15 has landed ----------------------------------------------------------------


@pytest.mark.parametrize("landed", ["WT-30", "WT-30-fix14", "WT-31"])
def test_the_exceptions_are_not_spent_by_another_work_item(tmp_path, capsys, landed):
    heads = {RECORD: _redacted(RECORD_BASE), SCRIPT: _with_header(_redacted(SCRIPT_BASE, True))}
    files = {RECORD: RECORD_BASE, SCRIPT: SCRIPT_BASE}
    found, out = _check(tmp_path, capsys, files, heads, landed=landed)
    assert found == 0, out


def test_the_exceptions_allow_nothing_once_wt30_fix15_has_landed(tmp_path, capsys):
    heads = {RECORD: _redacted(RECORD_BASE), SCRIPT: _with_header(_redacted(SCRIPT_BASE, True))}
    files = {RECORD: RECORD_BASE, SCRIPT: SCRIPT_BASE}
    found, out = _check(tmp_path, capsys, files, heads, landed=FIX)
    assert found == 1, out
    for path in (RECORD, SCRIPT):
        assert f"OUT OF BOUNDS  {path}  (the exception for {FIX} is spent" in out

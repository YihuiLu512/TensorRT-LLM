# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run test-list entries through the chain the way CI runs them, then summarize every test.

    python3 tests/integration/defs/openengine_chain/runner.py --chain trtllm \
        --out <dir> --list tests/integration/defs/openengine_chain/lists/case02_full.txt
    python3 tests/integration/defs/openengine_chain/runner.py --chain trtllm --out <dir> --smoke

Entries use the test-db syntax (``test_e2e.py::...``, ``unittest/... -k ...``,
``TIMEOUT (N)``, ``ISOLATION``). pytest runs in ``tests/integration/defs`` with
``--test-list``, once for the entries without ``ISOLATION`` and once for each
entry with it, as CI does; tests that start an inner pytest pass the
environment on, so the inner runs load the plugin too and write their junit
files to ``<out>/junit``. ``<out>/summary.md`` lists every test with its chain
markers. ``LLM_ROOT`` defaults to this checkout, as CI sets it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

_DEFS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, ""):
    sys.path.insert(0, _DEFS)

from openengine_chain import conventions as cv  # noqa: E402

_HERE = os.path.join(_DEFS, "openengine_chain")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_DEFS)))
_PLUGIN_OPT = f"-p {cv.PLUGIN_MODULE}"
_PROPS = (
    "openengine_chain",
    "chain_proof",
    "engine_calls",
    "engine_name",
    "expectation",
    "outer_test",
)


def read_entries(paths: list[str]) -> list[str]:
    entries: list[str] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line and line not in entries:
                    entries.append(line)
    return entries


_ISOLATION = re.compile(r"\s+ISOLATION\b")


def batches(entries: list[str]) -> list[tuple[str, list[str]]]:
    """``(name, entries)`` per pytest run: one for the shared entries, one per ISOLATION entry.

    The ISOLATION marker is removed; the test-list parser does not know it.
    """
    shared = [e for e in entries if not _ISOLATION.search(e)]
    isolated = [_ISOLATION.sub("", e) for e in entries if _ISOLATION.search(e)]
    runs = [("", shared)] if shared else []
    return runs + [(f"-iso{i}", [e]) for i, e in enumerate(isolated, 1)]


def import_dir(out: str) -> str:
    """A directory holding only the harness package, for PYTHONPATH.

    Putting ``tests/integration/defs`` itself there would let its top-level
    packages (``utils``, ``llmapi``, ...) shadow modules of the same name in the
    processes the tests start.
    """
    path = os.path.join(out, "pythonpath")
    os.makedirs(path, exist_ok=True)
    link = os.path.join(path, "openengine_chain")
    if not os.path.islink(link):
        os.symlink(_HERE, link)
    elif os.path.realpath(link) != os.path.realpath(_HERE):
        raise SystemExit(f"{link} points to {os.path.realpath(link)}, not {_HERE}")
    return path


def smoke_lists() -> list[str]:
    return sorted(glob.glob(os.path.join(_HERE, "smoke", "WT-*.txt")))


def test_files(entries: list[str]) -> list[str]:
    files: list[str] = []
    for entry in entries:
        path = entry.split()[0].split("::")[0]
        path = "test_unittests.py" if path.startswith("unittest/") else path
        if path not in files:
            files.append(path)
    return files


def run_env(chain: str, out: str, base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    if chain == cv.CHAIN_NONE:
        env.pop(cv.ENV_CHAIN, None)
    else:
        env[cv.ENV_CHAIN] = chain
    env[cv.ENV_RESULTS_DIR] = out
    env.setdefault("LLM_ROOT", _REPO)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (import_dir(out), env.get("PYTHONPATH", "")) if p
    )
    addopts = env.get("PYTEST_ADDOPTS", "")
    if _PLUGIN_OPT not in addopts:
        env["PYTEST_ADDOPTS"] = f"{addopts} {_PLUGIN_OPT}".strip()
    return env


def pytest_command(
    entries: list[str],
    out: str,
    *,
    timeout: int,
    waives: str | None,
    extra: list[str],
    name: str = "",
) -> list[str]:
    test_list = os.path.join(out, f"test-list{name}.txt")
    with open(test_list, "w", encoding="utf-8") as f:
        f.write("\n".join(entries) + "\n")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-vv",
        "--timeout-method=thread",
        f"--timeout={timeout}",
        "--rootdir",
        _DEFS,
        f"--test-list={test_list}",
        f"--output-dir={os.path.join(out, 'output')}",
        f"--junitxml={os.path.join(out, f'outer-junit{name}.xml')}",
    ]
    if waives:
        cmd.append(f"--waives-file={waives}")
    return cmd + extra + test_files(entries)


def _cases(path: str):
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        yield {"file": path, "name": f"<unreadable junit: {error}>", "outcome": "error"}
        return
    for case in root.iter("testcase"):
        props = {p.get("name"): p.get("value") for p in case.iter("property")}
        outcome, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            node = case.find(tag)
            if node is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                message = (
                    (node.get("message") or "").splitlines()[0][:200] if node.get("message") else ""
                )
                break
        row = {
            "file": os.path.basename(path),
            "name": f"{case.get('classname', '')}::{case.get('name', '')}",
            "outcome": outcome,
            "time": case.get("time", ""),
            "message": message,
        }
        row.update({key: props.get(key, "") for key in _PROPS})
        yield row


def junit_files(out: str) -> list[str]:
    # The shared run first, then the ISOLATION runs in order (-iso2 before -iso10).
    outer = sorted(glob.glob(os.path.join(out, "outer-junit*.xml")), key=lambda p: (len(p), p))
    return outer + sorted(glob.glob(os.path.join(out, "junit", "*.xml")))


def merge_reruns(rows: list[dict]) -> list[dict]:
    """Keep the last run of each inner test of one wrapper test.

    A wrapper reruns the tests that failed (``--lf``) in a pytest of its own,
    whose junit file comes later; as CI merges the two reports, the rerun counts.
    """
    merged: dict = {}
    for i, row in enumerate(rows):
        inner = row["file"].startswith("inner-")
        merged[(row["outer_test"], row["name"]) if inner else i] = row
    return list(merged.values())


def summarize(out: str) -> list[dict]:
    files = junit_files(out)
    rows = merge_reruns([row for path in files for row in _cases(path)])
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    cols = ["file", "name", "outcome", "time", *_PROPS, "message"]
    lines = [
        "# openengine_chain run summary",
        "",
        "Counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row.get(c, "")).replace("|", "\\|") for c in cols) + " |"
        )
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chain", required=True, choices=(cv.CHAIN_NONE,) + cv.CHAINS)
    parser.add_argument(
        "--out", required=True, help="results directory (created; must hold no junit files yet)"
    )
    parser.add_argument("--list", action="append", default=[], help="test list file")
    parser.add_argument("--smoke", action="store_true", help="add every work item's smoke/WT-*.txt")
    parser.add_argument("--timeout", type=int, default=7200, help="per-test timeout (s)")
    parser.add_argument(
        "--waives", help="waives file (CI: tests/integration/test_lists/waives.txt)"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the command and stop")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="only rebuild summary.md from the junit files in --out",
    )
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="-- extra pytest arguments")
    args = parser.parse_args(argv)
    out = os.path.abspath(args.out)
    if args.summarize_only:
        summarize(out)
        return 0
    lists = args.list + (smoke_lists() if args.smoke else [])
    entries = read_entries(lists)
    if not entries:
        parser.error("no test entries; give --list and/or --smoke")
    if junit_files(out):
        parser.error(f"{out} already holds junit files; the summary would mix runs")
    os.makedirs(out, exist_ok=True)
    # pytest runs in tests/integration/defs, not in the caller's directory.
    waives = os.path.abspath(args.waives) if args.waives else None
    extra = [a for a in args.extra if a != "--"]
    env = run_env(args.chain, out, dict(os.environ))
    print("cwd:", _DEFS)
    print(
        "env:",
        {
            k: env.get(k)
            for k in (cv.ENV_CHAIN, cv.ENV_RESULTS_DIR, "LLM_ROOT", "PYTHONPATH", "PYTEST_ADDOPTS")
        },
    )
    rc = 0
    for name, batch in batches(entries):
        cmd = pytest_command(
            batch, out, timeout=args.timeout, waives=waives, extra=extra, name=name
        )
        print("cmd:", " ".join(cmd))
        if args.dry_run:
            continue
        batch_rc = subprocess.call(cmd, cwd=_DEFS, env=env)
        print(f"pytest{name} rc={batch_rc}")
        rc = rc or batch_rc
    if args.dry_run:
        return 0
    rows = summarize(out)
    print(f"pytest rc={rc}; {len(rows)} test results in {os.path.join(out, 'summary.md')}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

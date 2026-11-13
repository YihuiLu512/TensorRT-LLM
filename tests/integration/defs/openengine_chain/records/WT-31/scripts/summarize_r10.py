# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge the R10 (WT-31) shard results into one line per planned test-list entry.

    python3 summarize_r10.py --cases <r10_cases.tsv> --results <results root> --out <tsv>
        [--rerun-shard <shard>]...

r10_shard.sh runs one runner.py per entry of a shard and writes, per node,
``<results root>/<group>/<shard>/job-*/node<N>/results/e<NN>/summary.json`` and ``entries.tsv``
(name, entry, start, wall seconds, runner rc, timeout used, leftover processes found, still alive
after the kill). The outer junit rows (one per test-list entry) are matched to the planned entries. An entry
without a row is ``not_started`` (job budget used up), ``no_result`` (its pytest ended without a
junit row, e.g. a per-test timeout), ``killed`` (running when the job ended), ``not_reached`` (the
job ended before its turn) or ``not_run`` (its shard never started an entry); ``not_started``
also covers a 2-node entry that one node skipped because the other node had stopped. When a shard
started entries in more than one job, the latest such job counts (``job-<id>`` order). A
``--rerun-shard`` holds entries of other shards run again; its result for an entry goes to the
``rerun_*`` columns and the baseline columns keep the planned shard's result. Multi-node
shards report one row per node; the entry outcome is node 0's (the node that runs the client;
``node0_missing`` when only node 1 reported), node 1's is kept in its own column.
``timeout_s_used`` is the ``--timeout`` runner.py got; a test with its own
``@pytest.mark.timeout`` runs under that mark instead, within ``timeout_s_used`` + 60 s
(r10_shard.sh runs runner.py under timeout(1)).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter

TIMEOUT_MARK = re.compile(r" TIMEOUT \(?\d+\)?")


def junit_key(entry: str) -> str:
    """The ``classname::name`` runner.py reports for one test-list entry, without TIMEOUT marks.

    The mark is removed from both the planned entry and the reported name (``load_rows``), so the
    match does not depend on whether a ``unittest/`` parameter id carries it.
    """
    entry = TIMEOUT_MARK.sub("", entry).strip()
    if entry.startswith("unittest/"):
        return f"test_unittests::test_unittests_v2[{entry}]"
    parts = entry.split("::")
    module = parts[0][:-3].replace("/", ".")
    classname = ".".join([module, *parts[1:-1]])
    return f"{classname}::{parts[-1]}"


def job_order(path: str) -> tuple:
    m = re.search(r"/job-(\d+)(?:_(\d+))?-r(\d+)/", path)
    return tuple(int(x or 0) for x in m.groups()) if m else (0, 0, 0)


def latest_jobs(results: str) -> dict[str, str]:
    """Shard -> directory of its latest job that started an entry (``node*/test-started.txt``).

    Results are matched by shard name, so a shard name used under two group directories (a rerun
    group that reuses a planned shard's name) is refused rather than merged.
    """
    latest: dict[str, str] = {}
    groups: dict[str, set] = {}
    for marker in glob.glob(os.path.join(results, "*", "*", "job-*", "node*", "test-started.txt")):
        job_dir = os.path.dirname(os.path.dirname(marker)) + os.sep
        shard_dir = os.path.dirname(job_dir.rstrip(os.sep))
        shard = os.path.basename(shard_dir)
        groups.setdefault(shard, set()).add(os.path.basename(os.path.dirname(shard_dir)))
        if len(groups[shard]) > 1:
            raise SystemExit(f"shard {shard} appears under groups {sorted(groups[shard])}")
        if shard not in latest or job_order(job_dir) > job_order(latest[shard]):
            latest[shard] = job_dir
    return latest


def load_rows(results: str) -> dict[tuple[str, str], dict]:
    """(shard, junit key) -> {node id: row}; a row without junit data carries only ``outcome``."""
    rows: dict[tuple[str, str], dict] = {}
    for shard, job_dir in latest_jobs(results).items():
        job = os.path.basename(job_dir.rstrip(os.sep))
        # Only the node directories: the job directory also holds files such as node-precheck.log.
        for node_dir in sorted(glob.glob(os.path.join(job_dir, "node[0-9]*"))):
            match = re.fullmatch(r"node(\d+)", os.path.basename(node_dir))
            if not match or not os.path.isdir(node_dir):
                continue
            node = match.group(1)
            ran: dict[str, dict] = {}
            if os.path.exists(os.path.join(node_dir, "entries.tsv")):
                with open(os.path.join(node_dir, "entries.tsv"), encoding="utf-8") as f:
                    for line in f:
                        name, entry, _, wall, rc, timeout, *reaped = line.rstrip("\n").split("\t")
                        ran[junit_key(entry)] = dict(
                            name=name, wall=wall, rc=rc, timeout=timeout, leftovers="/".join(reaped)
                        )
            for listed in glob.glob(os.path.join(node_dir, "list-e*.txt")):
                with open(listed, encoding="utf-8") as f:
                    key = junit_key(f.read())
                if key not in ran:
                    ran[key] = dict(
                        name=os.path.basename(listed)[5:-4],
                        wall="",
                        rc="killed",
                        timeout="",
                        leftovers="",
                    )
            # Entries of the frozen shard list the node never got to (the job ended first).
            if os.path.exists(os.path.join(job_dir, "selected.txt")):
                with open(os.path.join(job_dir, "selected.txt"), encoding="utf-8") as f:
                    for line in f:
                        if line.strip() and junit_key(line) not in ran:
                            ran[junit_key(line)] = dict(
                                name="", wall="", rc="not_reached", timeout="", leftovers=""
                            )
            for key, info in ran.items():
                junit = []
                summary = os.path.join(node_dir, "results", info["name"], "summary.json")
                if os.path.exists(summary):
                    with open(summary, encoding="utf-8") as f:
                        junit = [r for r in json.load(f) if not r["file"].startswith("inner-")]
                if info["rc"] == "-":
                    row = {"outcome": "not_started"}
                elif info["rc"] in ("killed", "not_reached"):
                    row = {"outcome": info["rc"]}
                elif not junit:
                    row = {"outcome": "no_result", "message": f"runner rc={info['rc']}"}
                else:
                    row = junit[0]
                    reported = TIMEOUT_MARK.sub("", re.sub(r"^defs\.", "", row["name"]))
                    if len(junit) != 1 or reported != key:
                        row = dict(row, message=f"unexpected junit rows: {len(junit)} {reported}")
                rows.setdefault((shard, key), {})[node] = dict(
                    row,
                    job=job,
                    wall_s=info["wall"],
                    rc=info["rc"],
                    timeout_s_used=info["timeout"],
                    leftovers=info["leftovers"],
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rerun-shard", action="append", default=[])
    args = parser.parse_args()
    rows = load_rows(args.results)
    with open(args.cases, encoding="utf-8") as f:
        cases = list(csv.DictReader(f, delimiter="\t"))
    reruns: dict[str, dict] = {}
    rerun_shards = {shard for shard, _ in rows if shard in args.rerun_shard}
    missing = sorted(set(args.rerun_shard) - rerun_shards)
    if missing:
        raise SystemExit(f"no started job for --rerun-shard {missing}")
    for (shard, key), found in rows.items():
        if shard in rerun_shards and "0" in found:
            if key in reruns:
                raise SystemExit(f"entry {key} is in more than one rerun shard")
            reruns[key] = found["0"]
    out_cols = [
        "xd",
        "entry",
        "group",
        "gpus",
        "shard",
        "outcome",
        "time_s",
        "wall_s",
        "timeout_s_used",
        "leftovers",
        "job",
        "node1_outcome",
        "message",
        "waives",
        "rerun_outcome",
        "rerun_time_s",
        "rerun_job",
        "rerun_message",
    ]
    counts: Counter = Counter()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\t".join(out_cols) + "\n")
        for case in cases:
            found = rows.get((case["shard"], junit_key(case["entry"])), {})
            first = found.get("0")
            if first is None and found:
                first = {"outcome": "node0_missing", "job": next(iter(found.values()))["job"]}
            rerun = reruns.get(junit_key(case["entry"]), {})
            line = dict(
                case,
                rerun_outcome=rerun.get("outcome", ""),
                rerun_time_s=rerun.get("time", ""),
                rerun_job=rerun.get("job", ""),
                rerun_message=rerun.get("message", "").replace("\t", " ")[:200],
                outcome=first["outcome"] if first else "not_run",
                time_s=first.get("time", "") if first else "",
                wall_s=first.get("wall_s", "") if first else "",
                timeout_s_used=first.get("timeout_s_used", "") if first else "",
                leftovers=first.get("leftovers", "") if first else "",
                job=first["job"] if first else "",
                node1_outcome=found["1"]["outcome"] if "1" in found else "",
                message=(first.get("message", "") if first else "").replace("\t", " ")[:200],
            )
            counts[(case["group"], line["outcome"])] += 1
            if rerun:
                counts[(case["group"] + "-rerun", rerun["outcome"])] += 1
            f.write("\t".join(str(line.get(c, "")) for c in out_cols) + "\n")
    print(json.dumps({f"{g}:{o}": n for (g, o), n in sorted(counts.items())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

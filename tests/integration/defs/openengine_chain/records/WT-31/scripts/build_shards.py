# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the R10 (WT-31) case table and shard lists from the XD index and the collected node ids.

    python3 build_shards.py --functional-tests <02_functional-tests.md> --collect <collect dir> \
        --waives <waives.txt> --out <dir>

Input: the XD-001..XD-319 rows of §2.0 of 02_functional-tests.md (one nodeid, model, GPU count,
test-db registration, CI stage, waive/skip markers, duration / upper bound per row) and the
``pytest --collect-only -q`` output of the collect job. Output in ``--out``:

* ``r10_cases.tsv``: one line per test-list entry (a parametrized row expands to one line per
  collected parameter); the columns are named in its header line.
* ``lists/<shard>.txt`` (test-db syntax, ``TIMEOUT (min)`` when the bound exceeds 60 min, capped at
  ``LIST_TIMEOUT_CAP_MIN`` so that a test can end inside the 4 h job) and one ``cases-<group>.psv``
  per sbatch array (``<shard>|<list path>``).

Grouping: ``skip_no_hopper`` rows go to the Hopper shard; the TG-4 multi-node rows (XD-301..311,
XD-314) to the 2-node shard; the rest run on single-node B200 with
``GPUs = max(skip_less_device N, smallest GPU count of the CI stages that register it, else the
nominal count)`` rounded up to 1, 2, 4 or 8. The tests that bind fixed ports (etcd 2379 and the
service-discovery tests; ``examples/test_ray.py``, ports 8000-8002) of the single-node group run
one after another in a single 8-GPU ``-ports`` shard, which submit.sh requests as a whole node, so that no other job on
the node can hold those ports; r10_shard.sh adds ``--run-ray`` to the test_ray entries as the CI
Ray stages do. The 2-node rows form one shard (the multi-node group is requested once) ordered by
estimate, shortest first, so that as many as possible finish inside one 4 h allocation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict

TIERS = (1, 2, 4, 8)
SHARD_BUDGET_S = 8400  # sum of estimates per single-node shard (job wall clock is 4 h)
LIST_TIMEOUT_CAP_MIN = 215  # per-test TIMEOUT written to the lists; leaves room for the preflight
HOPPER_ROWS = {
    *(f"XD-{n:03d}" for n in (35, 36, 37, 38, 39, 41, 42)),
    *(f"XD-{n:03d}" for n in range(44, 53)),
    "XD-066",
    "XD-068",
    "XD-069",
    "XD-122",
    "XD-124",
    "XD-201",
}
TWO_NODE_ROWS = {*(f"XD-{n}" for n in range(301, 312)), "XD-314"}
UNIT_PREFIX = "unittest/"


def parse_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^\| (XD-[0-3]\d\d) \|", line)
            if not m:
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 10:
                raise SystemExit(f"unexpected column count in {m.group(1)}: {len(cells)}")
            rows.append(
                dict(
                    xd=cells[0],
                    nodeid=cells[1],
                    group=cells[2],
                    model=cells[3],
                    gpus=cells[4],
                    testdb=cells[5],
                    stage=cells[6],
                    skip=cells[7],
                    dur=cells[8],
                    hp=cells[9],
                )
            )
    return rows


def collected(collect_dir: str) -> tuple[set[str], set[str]]:
    defs, unit = set(), set()
    for name in os.listdir(collect_dir):
        if not name.endswith(".txt") or not name.startswith(("defs-", "unit-")):
            continue
        target = defs if name.startswith("defs-") else unit
        with open(os.path.join(collect_dir, name), encoding="utf-8") as f:
            target.update(line.strip() for line in f if "::" in line and " " not in line.strip())
    if not defs:
        raise SystemExit("no collected node ids")
    return defs, unit


def entries_for(row: dict, defs: set[str], unit: set[str]) -> list[str]:
    """The exact test-list entries of one row, checked against the collected node ids."""
    text = row["nodeid"]
    m = re.match(r"`([^`]+)`", text)
    if not m:
        raise SystemExit(f"{row['xd']}: no node id in {text[:80]!r}")
    base = m.group(1)
    params = re.findall(r"`(\[[^`]*\])`", text[len(base) + 2 :])
    if base.startswith(UNIT_PREFIX):
        inner = base[len(UNIT_PREFIX) :]
        if inner not in unit:
            raise SystemExit(f"{row['xd']}: {inner} not collected")
        return [base]
    if params:  # "<func>`；参数 N 个：`[a]`、`[…]`": every collected parameter of the function
        found = sorted(n for n in defs if n.startswith(base + "["))
        if len(found) != len(params):
            raise SystemExit(
                f"{row['xd']}: {len(params)} parameters listed, {len(found)} collected"
            )
        return found
    if base not in defs:
        raise SystemExit(f"{row['xd']}: {base} not collected")
    return [base]


def stage_gpus(stage: str) -> int | None:
    counts = []
    for part in re.findall(r"L0：([^`（]+)", stage):
        m = re.search(r"-(\d)_GPUs", part)
        counts.append(int(m.group(1)) if m else 1)
    return min(counts) if counts else None


def first_int(text: str) -> int:
    m = re.search(r"\d+", text)
    return int(m.group()) if m else 1


def plan(row: dict) -> tuple[str, int]:
    if row["xd"] in HOPPER_ROWS:
        group = "hopper"
    elif row["xd"] in TWO_NODE_ROWS:
        return "2node", 8
    else:
        group = "b200"
    need = [int(n) for n in re.findall(r"skip_less_device\((\d+)\)", row["skip"] + row["gpus"])]
    registered = stage_gpus(row["stage"])
    base = registered if registered is not None else first_int(row["gpus"])
    gpus = max([base, *need])
    return group, next(t for t in TIERS if t >= min(gpus, 8))


def bounds(row: dict, model: str, gpus: int) -> tuple[int, int]:
    """(estimate s, timeout s). Estimate: the CI duration x1.5 when known, else a guess by size."""
    text = re.sub(r"(?:WT|XD)-\d+", "", row["dur"])
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", text.split("/")[0])]
    # "<duration> / <upper bound>"; a cell without "/" has no bound.
    upper = re.findall(r"(\d+)", text.split("/")[-1]) if "/" in text else []
    timeout = int(upper[0]) if upper else 3600
    if nums:
        est = int(max(nums) * 1.5)
    elif timeout > 3600:
        est = min(timeout, 9000)
    elif "0.6B" in model:
        est = 300
    else:
        est = 900 if gpus <= 2 else 1800
    return est, timeout


def uses_etcd(entry: str) -> bool:
    return "etcd" in entry or "service_discovery" in entry


def uses_ray(entry: str) -> bool:
    return entry.startswith("examples/test_ray.py::")


def read_waives(path: str) -> list[tuple[int, str, str]]:
    """(line number, test name without the ``full:<platform>/`` prefix, line) per waive line."""
    waives = []
    with open(path, encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            if not line.strip() or line.startswith("#"):
                continue
            name = re.sub(r"^full:[^/]+/", "", line.split(" SKIP")[0].split()[0])
            waives.append((no, name, line.strip()))
    return waives


def waive_lines(waives: list[tuple[int, str, str]], entry: str) -> list[str]:
    return [
        f"{no}:{line[:120]}"
        for no, name, line in waives
        if name == entry or entry.startswith(name + "[") or entry.startswith(name + "::")
    ]


def pack(cases: list[dict], prefix: str) -> list[list[dict]]:
    """First-fit decreasing by estimate; a case above the budget gets a shard of its own."""
    shards: list[list[dict]] = []
    loads: list[int] = []
    for case in sorted(cases, key=lambda c: -c["est"]):
        for i, load in enumerate(loads):
            if load + case["est"] <= SHARD_BUDGET_S:
                shards[i].append(case)
                loads[i] += case["est"]
                break
        else:
            shards.append([case])
            loads.append(case["est"])
    for i, shard in enumerate(shards):
        for case in shard:
            case["shard"] = f"{prefix}-s{i:02d}"
    return shards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--functional-tests", required=True)
    parser.add_argument("--collect", required=True)
    parser.add_argument("--waives", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = parse_rows(args.functional_tests)
    defs, unit = collected(args.collect)
    waives = read_waives(args.waives)
    cases = []
    for row in rows:
        group, gpus = plan(row)
        for entry in entries_for(row, defs, unit):
            est, timeout = bounds(row, row["model"], gpus)
            cases.append(
                dict(
                    xd=row["xd"],
                    entry=entry,
                    group=group,
                    gpus=gpus,
                    est=est,
                    timeout=timeout,
                    etcd=uses_etcd(entry),
                    ray=uses_ray(entry),
                    waives=waive_lines(waives, entry),
                    model=row["model"],
                    skip=row["skip"],
                )
            )
    os.makedirs(os.path.join(args.out, "lists"), exist_ok=True)
    by_array: dict[str, list[dict]] = defaultdict(list)
    port_cases = [c for c in cases if c["group"] == "b200" and (c["etcd"] or c["ray"])]
    port_gpus = 8  # the whole node (see the module docstring)
    for case in cases:
        if case in port_cases:
            # One shard, so that no two of them hold a fixed port at once.
            case["gpus"] = port_gpus
            key = f"b200-{port_gpus}g-ports"
        else:
            key = f"{case['group']}-{case['gpus']}g"
        by_array[key].append(case)
    arrays = {}
    for key, members in sorted(by_array.items()):
        if key.endswith("-ports") or key.startswith("2node"):
            if key.startswith("2node"):
                members.sort(key=lambda c: c["est"])
            for case in members:
                case["shard"] = f"{key}-s00"
            shards = [members]
        else:
            shards = pack(members, key)
        arrays[key] = shards
        with open(os.path.join(args.out, f"cases-{key}.psv"), "w", encoding="utf-8") as f:
            for shard in shards:
                name = shard[0]["shard"]
                path = os.path.join(os.path.abspath(args.out), "lists", f"{name}.txt")
                with open(path, "w", encoding="utf-8") as lf:
                    for case in shard:
                        minutes = min(math.ceil(case["timeout"] / 60), LIST_TIMEOUT_CAP_MIN)
                        # runner.py's --timeout: the TIMEOUT mark, else TEST_TIMEOUT (3600 s).
                        # r10_shard.sh may cut it to the time left in the job, and a test's
                        # own timeout mark wins over it.
                        case["list_timeout"] = minutes * 60 if minutes > 60 else 3600
                        lf.write(
                            case["entry"] + (f" TIMEOUT ({minutes})" if minutes > 60 else "") + "\n"
                        )
                f.write(f"{name}|{path}\n")
    with open(os.path.join(args.out, "r10_cases.tsv"), "w", encoding="utf-8") as f:
        f.write(
            "xd\tentry\tgroup\tgpus\tshard\test_s\ttimeout_s\tlist_timeout_s\tetcd\tray"
            "\twaives\tmodel\n"
        )
        keys = ("xd", "entry", "group", "gpus", "shard", "est", "timeout", "list_timeout", "etcd")
        for case in cases:
            f.write("\t".join(str(case[k]) for k in (*keys, "ray")))
            f.write("\t" + ";".join(case["waives"]) + "\t" + case["model"] + "\n")
    summary = {
        key: [
            dict(shard=s[0]["shard"], cases=len(s), est_s=sum(c["est"] for c in s)) for s in shards
        ]
        for key, shards in arrays.items()
    }
    with open(os.path.join(args.out, "shards.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(
        json.dumps(
            {k: [(s["shard"], s["cases"], s["est_s"]) for s in v] for k, v in summary.items()}
        )
    )
    print("rows", len(rows), "entries", len(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

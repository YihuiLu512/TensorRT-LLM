# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The worker frontends' perf records of a disaggregated request, on the CPU chain (SV-DSG-12, CS-12).

Each chain (``cpu_disagg_chain``) runs the orchestrator and two worker
frontends, each writing its perf records to one directory, with the
frontends started with ``--server_role`` or not (static URLs) and with the
postprocessing pool off or on. Both frontends' records of a request carry
``ctx_request_id`` and the orchestrator's ``disagg_request_id``; the file
name's role segment is ``context`` / ``generation`` with a role and ``server``
without one; the times the engine measures (queueing, KV-cache transfer and
its size, iterations, KV-cache blocks) are left out, never written as 0; and
``scripts/poc3/wt54_join_perf.py`` joins the three records of each request.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

import pytest
import yaml

__extra_import_path__ = [".."]
from cpu_disagg_chain import CpuDisaggChain  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

ROLES = ("context", "generation")
JOIN_SCRIPT = Path(__file__).resolve().parents[5] / "scripts" / "poc3" / "wt54_join_perf.py"
# (shape, frontends started with --server_role, postprocessing processes per frontend)
CHAINS = [
    ("trtllm", True, 0),
    ("trtllm", True, 2),
    ("trtllm", False, 0),
    ("trtllm", False, 2),
    ("ds", True, 0),
]
REQUESTS = [
    ("/v1/completions", {"prompt": "He llo", "max_tokens": 3}),
    ("/v1/completions", {"prompt": "He llo", "max_tokens": 3, "stream": True}),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "He"}], "max_tokens": 2}),
]


def _ids(chain: tuple) -> str:
    shape, role, workers = chain
    return f"{shape}-{'role' if role else 'static'}-pool{workers}"


@pytest.fixture(scope="module", params=CHAINS, ids=[_ids(c) for c in CHAINS])
def served(request, tmp_path_factory):
    """A chain that has served :data:`REQUESTS`, stopped so that its files are complete."""
    shape, role, workers = request.param
    directory = tmp_path_factory.mktemp(f"chain-{_ids(request.param)}")
    output_dir = directory / "perf"
    config = directory / "frontend.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                # SV-DSG-1: without it OpenAIServer refuses a generation leg's handoff fields.
                "cache_transceiver_config": {"backend": "DEFAULT"},
                "return_perf_metrics": True,
                "perf_metrics_output_dir": str(output_dir),
            }
        )
    )
    args = {}
    for leg in ROLES:
        args[leg] = ["--config", str(config)]
        if workers:
            args[leg] += ["--num_postprocess_workers", str(workers)]
        if role:
            args[leg] += ["--server_role", leg]
    statuses = []
    with CpuDisaggChain(
        directory,
        shape,
        frontend_args=args,
        orchestrator_config={
            "return_perf_metrics": True,
            "perf_metrics_output_dir": str(output_dir),
        },
    ) as chain:
        for path, body in REQUESTS:
            response = chain.post(path, body)
            statuses.append((response.status_code, response.text[:2000]))
        log = chain.orchestrator_log()
    # Stopping the servers flushes their writers.
    return {"role": role, "dir": output_dir, "statuses": statuses, "log": log}


def _read(output_dir: Path) -> dict[str, list[dict]]:
    """The perf records, by the role segment of their file name."""
    records: dict[str, list[dict]] = {}
    for path in sorted(output_dir.glob("perf_metrics-*.jsonl")):
        kind = path.name.removeprefix("perf_metrics-").split("-", 1)[0]
        for line in path.read_text().splitlines():
            if line:
                records.setdefault(kind, []).append(json.loads(line))
    return records


def _wait_records(output_dir: Path, count: int, timeout: float = 10.0) -> dict[str, list[dict]]:
    deadline = time.monotonic() + timeout
    while True:
        records = _read(output_dir)
        if sum(len(v) for v in records.values()) >= count or time.monotonic() > deadline:
            return records
        time.sleep(0.1)


def test_every_request_is_served(served):
    for status, text in served["statuses"]:
        assert status == 200, (text, served["log"][-4000:])


def test_the_file_names_carry_the_frontends_role(served):
    records = _wait_records(served["dir"], 3 * len(REQUESTS))
    expected = {"disagg", *ROLES} if served["role"] else {"disagg", "server"}
    assert set(records) == expected
    assert len(records["disagg"]) == len(REQUESTS)


def test_both_legs_carry_the_ids_of_the_orchestrator(served):
    records = _wait_records(served["dir"], 3 * len(REQUESTS))
    orchestrator_ids = [record["disagg_request_id"] for record in records["disagg"]]
    assert len(set(orchestrator_ids)) == len(REQUESTS), orchestrator_ids
    workers = [record for kind, items in records.items() if kind != "disagg" for record in items]
    assert len(workers) == 2 * len(REQUESTS)
    by_id: dict[int, list[dict]] = {}
    for record in workers:
        assert record.get("disagg_request_id") is not None, record
        assert record.get("ctx_request_id") is not None, record
        by_id.setdefault(record["disagg_request_id"], []).append(record)
    assert sorted(by_id) == sorted(orchestrator_ids)
    for legs in by_id.values():
        assert len(legs) == 2
        assert legs[0]["ctx_request_id"] == legs[1]["ctx_request_id"]
    if served["role"]:
        for kind in ROLES:
            assert sorted(r["disagg_request_id"] for r in records[kind]) == sorted(orchestrator_ids)


def test_the_times_the_engine_measures_are_left_out_not_zero(served):
    records = _wait_records(served["dir"], 3 * len(REQUESTS))
    for kind, items in records.items():
        if kind == "disagg":
            continue
        for record in items:
            metrics = record["perf_metrics"]
            timing = metrics["timing_metrics"]
            assert timing.get("first_scheduled_time") is None
            for name in ("kv_cache_transfer_start", "kv_cache_transfer_end", "kv_cache_size"):
                assert timing.get(name) is None, (name, record)
            assert metrics.get("first_iter") is None and metrics.get("last_iter") is None
            assert set((metrics.get("kv_cache_metrics") or {}).values()) <= {None}
            # What the frontend measured is there.
            for name in ("arrival_time", "first_token_time", "last_token_time"):
                assert timing[name] > 0, (name, record)
            assert timing["arrival_time"] <= timing["first_token_time"] <= timing["last_token_time"]


def test_the_join_script_joins_each_request(served, tmp_path):
    _wait_records(served["dir"], 3 * len(REQUESTS))
    output = tmp_path / "joined.jsonl"
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            str(JOIN_SCRIPT),
            str(served["dir"]),
            "--require",
            "frontends",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    joined = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(joined) == len(REQUESTS)
    for entry in joined:
        assert entry["missing"] == [], entry
        assert entry["duplicates"] == 0
        assert entry["context_frontend"]["disagg_request_id"] == entry["disagg_request_id"]
        assert entry["generation_frontend"]["disagg_request_id"] == entry["disagg_request_id"]
        if served["role"]:
            # On the orchestrator's clock, the generation leg arrives after the context leg.
            # (Without roles the script orders the two legs by that very arrival.)
            ctx = entry["context_frontend"]["perf_metrics"]["timing_metrics"]
            gen = entry["generation_frontend"]["perf_metrics"]["timing_metrics"]
            assert ctx["server_arrival_time"] < gen["server_arrival_time"]
        assert entry["legs_by_arrival"] is not served["role"]

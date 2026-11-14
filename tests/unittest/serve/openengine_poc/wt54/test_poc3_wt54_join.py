# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``scripts/poc3/wt54_join_perf.py`` on made-up records (DQ-13: offline correlation).

The script joins the orchestrator's record, both frontends' and both
engines' leg records of a request by ``disagg_request_id``, takes the KV-cache
transfer time and size from the engines, and fails, not passes, on no input
or when no request has every required record.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

SCRIPT = Path(__file__).resolve().parents[5] / "scripts" / "poc3" / "wt54_join_perf.py"
ID = 4_611_686_018_427_387_905


def _module():
    spec = importlib.util.spec_from_file_location("wt54_join_perf", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


join_perf = _module()


def _write(path: Path, *records: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")


def _orchestrator(disagg_id: int = ID) -> dict:
    return {
        "ctx_server": "h:1",
        "gen_server": "h:2",
        "disagg_server_arrival_time": 1.0,
        "disagg_ctx_dispatch_time": 1.1,
        "disagg_server_first_token_time": 2.0,
        "status": "complete",
        "disagg_request_id": disagg_id,
    }


def _frontend(request_id: str, arrival: float, disagg_id: int = ID) -> dict:
    return {
        "request_id": request_id,
        "perf_metrics": {"timing_metrics": {"server_arrival_time": arrival}},
        "ctx_request_id": disagg_id,
        "disagg_request_id": disagg_id,
        "status": "complete",
    }


def _engine(leg: str, request_id: str, disagg_id: int = ID, transfer=None, size=None) -> dict:
    return {
        "leg": leg,
        "request_id": request_id,
        "disagg_request_id": disagg_id,
        "ctx_request_id": disagg_id,
        "finish_reason": "length",
        "durations_ms": {"queue": 1.0, "kv_cache_transfer": transfer},
        "kv_cache_size": size,
    }


def _five(directory: Path, kinds=("context", "generation"), engines: bool = True) -> None:
    _write(directory / "perf_metrics-disagg-h-1-t.jsonl", _orchestrator())
    _write(directory / "ctx" / f"perf_metrics-{kinds[0]}-h-2-t.jsonl", _frontend("c1", 1.2))
    _write(directory / "gen" / f"perf_metrics-{kinds[1]}-h-3-t.jsonl", _frontend("g1", 1.6))
    if engines:
        _write(
            directory / "engines" / "openengine_leg_durations-h-4-t.jsonl",
            _engine("context", "c1", transfer=7.0, size=111),
            _engine("generation", "g1", transfer=9.5, size=222),
        )


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # nosec B603
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=60
    )


def test_the_five_records_of_a_request_are_joined(tmp_path):
    _five(tmp_path)
    output = tmp_path / "joined.jsonl"

    result = _run(str(tmp_path), "--output", str(output))

    assert result.returncode == 0, result.stderr
    (entry,) = [json.loads(line) for line in output.read_text().splitlines()]
    assert entry["disagg_request_id"] == ID
    assert entry["missing"] == []
    assert entry["duplicates"] == 0
    assert entry["context_frontend"]["request_id"] == "c1"
    assert entry["generation_frontend"]["request_id"] == "g1"
    assert entry["context_engine"]["leg"] == "context"
    assert entry["generation_engine"]["leg"] == "generation"
    # The generation engine's reading comes first.
    assert entry["kv_cache_transfer_ms"] == 9.5
    assert entry["kv_cache_size"] == 222


def test_the_context_engine_stands_in_for_a_reading_the_generation_engine_lacks(tmp_path):
    _write(tmp_path / "perf_metrics-disagg-h-1-t.jsonl", _orchestrator())
    _write(
        tmp_path / "openengine_leg_durations-h-4-t.jsonl",
        _engine("context", "c1", transfer=7.0, size=111),
        _engine("generation", "g1"),
    )

    (entry,) = join_perf.join(join_perf.load([tmp_path]), "engines")

    assert entry["kv_cache_transfer_ms"] == 7.0
    assert entry["kv_cache_size"] == 111
    assert entry["missing"] == ["context_frontend", "generation_frontend"]


def test_server_records_take_the_leg_of_the_engine_record_with_their_request_id(tmp_path):
    # Arrival order would say the opposite: the engine records decide.
    _write(tmp_path / "perf_metrics-disagg-h-1-t.jsonl", _orchestrator())
    _write(tmp_path / "a" / "perf_metrics-server-h-2-t.jsonl", _frontend("c1", 1.9))
    _write(tmp_path / "b" / "perf_metrics-server-h-3-t.jsonl", _frontend("g1", 1.2))
    _write(
        tmp_path / "openengine_leg_durations-h-4-t.jsonl",
        _engine("context", "c1"),
        _engine("generation", "g1"),
    )

    (entry,) = join_perf.join(join_perf.load([tmp_path]), "engines")

    assert entry["context_frontend"]["request_id"] == "c1"
    assert entry["generation_frontend"]["request_id"] == "g1"
    assert entry["legs_by_arrival"] is False


def test_an_engine_record_decides_the_leg_of_its_own_request_only(tmp_path):
    # Two requests whose context and generation calls reuse each other's request ids.
    other = ID + 1
    _write(tmp_path / "perf_metrics-disagg-h-1-t.jsonl", _orchestrator(), _orchestrator(other))
    _write(
        tmp_path / "a" / "perf_metrics-server-h-2-t.jsonl",
        _frontend("x", 1.9),
        _frontend("y", 1.1, other),
    )
    _write(
        tmp_path / "b" / "perf_metrics-server-h-3-t.jsonl",
        _frontend("y", 1.2),
        _frontend("x", 1.8, other),
    )
    _write(
        tmp_path / "openengine_leg_durations-h-4-t.jsonl",
        _engine("context", "x"),
        _engine("generation", "y"),
        _engine("context", "y", other),
        _engine("generation", "x", other),
    )

    first, second = join_perf.join(join_perf.load([tmp_path]), "engines")

    assert first["context_frontend"]["request_id"] == "x"
    assert second["context_frontend"]["request_id"] == "y"
    assert first["missing"] == second["missing"] == []


def test_records_of_other_frontend_roles_are_left_out(tmp_path):
    _five(tmp_path)
    _write(tmp_path / "perf_metrics-mm_encoder-h-5-t.jsonl", _frontend("e1", 1.1))

    (entry,) = join_perf.join(join_perf.load([tmp_path]), "engines")

    assert entry["missing"] == []
    assert entry["duplicates"] == 0
    assert entry["unassigned"] == 0


def test_a_file_under_two_given_directories_is_read_once(tmp_path):
    _five(tmp_path)
    output = tmp_path / "joined.jsonl"

    result = _run(str(tmp_path), str(tmp_path / "ctx"), "--output", str(output))

    assert result.returncode == 0, result.stderr
    (entry,) = [json.loads(line) for line in output.read_text().splitlines()]
    assert entry["duplicates"] == 0


def test_without_engine_records_the_first_server_record_to_arrive_is_the_context_leg(tmp_path):
    _five(tmp_path, kinds=("server", "server"), engines=False)

    (entry,) = join_perf.join(join_perf.load([tmp_path]), "frontends")

    assert entry["missing"] == []
    assert entry["context_frontend"]["request_id"] == "c1"
    assert entry["generation_frontend"]["request_id"] == "g1"
    assert entry["legs_by_arrival"] is True


def test_a_server_record_takes_the_leg_left_when_the_other_is_known(tmp_path):
    # Only the context engine wrote its record: the generation frontend's record is the other leg.
    _write(tmp_path / "perf_metrics-disagg-h-1-t.jsonl", _orchestrator())
    _write(tmp_path / "a" / "perf_metrics-server-h-2-t.jsonl", _frontend("c1", 1.9))
    _write(tmp_path / "b" / "perf_metrics-server-h-3-t.jsonl", _frontend("g1", 1.2))
    _write(tmp_path / "openengine_leg_durations-h-4-t.jsonl", _engine("context", "c1"))

    (entry,) = join_perf.join(join_perf.load([tmp_path]), "frontends")

    assert entry["context_frontend"]["request_id"] == "c1"
    assert entry["generation_frontend"]["request_id"] == "g1"
    assert entry["unassigned"] == 0


def test_a_resent_leg_of_server_records_counts_as_a_duplicate(tmp_path):
    # Each frontend's records are in its own file, so a resent leg stays with its file.
    _five(tmp_path, kinds=("server", "server"), engines=False)
    _write(tmp_path / "ctx" / "perf_metrics-server-h-2-t.jsonl", _frontend("c2", 1.4))

    (entry,) = join_perf.join(join_perf.load([tmp_path]), "frontends")

    assert entry["missing"] == []
    assert entry["context_frontend"]["request_id"] == "c1"
    assert entry["generation_frontend"]["request_id"] == "g1"
    assert entry["duplicates"] == 1
    assert entry["unassigned"] == 0


def test_server_records_of_three_files_are_left_unassigned(tmp_path):
    _five(tmp_path, kinds=("server", "server"), engines=False)
    _write(tmp_path / "other" / "perf_metrics-server-h-5-t.jsonl", _frontend("x1", 1.3))

    (entry,) = join_perf.join(join_perf.load([tmp_path]), "frontends")

    assert entry["context_frontend"] is None and entry["generation_frontend"] is None
    assert entry["unassigned"] == 3


def test_records_of_an_unexpected_shape_are_read_as_missing_fields(tmp_path):
    _write(tmp_path / "perf_metrics-disagg-h-1-t.jsonl", _orchestrator())
    _write(
        tmp_path / "perf_metrics-server-h-2-t.jsonl",
        {**_frontend("c1", 1.2), "perf_metrics": ["not", "a", "dict"]},
        {
            **_frontend("c1", 1.2),
            "perf_metrics": {"timing_metrics": {"server_arrival_time": 10**400}},
        },
    )
    _write(tmp_path / "perf_metrics-server-h-3-t.jsonl", _frontend("g1", 1.6))
    _write(
        tmp_path / "openengine_leg_durations-h-4-t.jsonl",
        {**_engine("generation", "g9"), "durations_ms": [1, 2]},
    )

    (entry,) = join_perf.join(join_perf.load([tmp_path]), "engines")

    # The records without a usable arrival time cannot be ordered, so none is assigned.
    assert entry["unassigned"] == 3
    assert entry["kv_cache_transfer_ms"] is None


def test_frontends_only_needs_no_engine_record(tmp_path):
    _five(tmp_path, engines=False)

    assert _run(str(tmp_path), "--require", "frontends").returncode == 0
    result = _run(str(tmp_path))
    assert result.returncode == 1
    assert "no request has every required record" in result.stderr


def test_records_of_other_requests_do_not_join(tmp_path):
    _write(tmp_path / "perf_metrics-disagg-h-1-t.jsonl", _orchestrator(ID))
    _write(tmp_path / "perf_metrics-context-h-2-t.jsonl", _frontend("c1", 1.2, ID + 1))
    _write(tmp_path / "perf_metrics-generation-h-3-t.jsonl", _frontend("g1", 1.6, ID + 1))

    result = _run(str(tmp_path), "--require", "frontends")

    assert result.returncode == 1
    (entry,) = join_perf.join(join_perf.load([tmp_path]), "frontends")
    assert entry["missing"] == ["context_frontend", "generation_frontend"]


def test_duplicates_are_counted_and_the_first_kept(tmp_path):
    _five(tmp_path)
    _write(tmp_path / "ctx" / "perf_metrics-context-h-2-t.jsonl", _frontend("c2", 1.3))

    (entry,) = join_perf.join(join_perf.load([tmp_path]), "engines")

    assert entry["duplicates"] == 1
    assert entry["context_frontend"]["request_id"] == "c1"


@pytest.mark.parametrize("content", [None, "", "not json\n"], ids=["no-file", "empty", "garbage"])
def test_no_records_fails(tmp_path, content):
    if content is not None:
        (tmp_path / "perf_metrics-disagg-h-1-t.jsonl").write_text(content)

    result = _run(str(tmp_path))

    assert result.returncode == 1
    assert "no perf records found" in result.stderr


def test_a_missing_directory_fails(tmp_path):
    assert _run(str(tmp_path / "absent")).returncode == 1

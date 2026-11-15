# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The direct replay load tool (openengine_chain/direct_replay_load.py, CT-13) against two fake engines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("grpc", reason="gRPC runtime not installed")
pytest.importorskip("grpc_health", reason="gRPC health runtime not installed")

__extra_import_path__ = ["~/tests/unittest/grpc/openengine_client"]
from fake_openengine import FakeEngine, disagg_profile, running_in_thread  # noqa: E402

pytestmark = pytest.mark.cpu_only

MODEL = "fake-model"


@pytest.fixture
def replay(harness_path, arms, monkeypatch):
    """``openengine_chain.direct_replay_load``, finding the arms module the test loaded."""
    import importlib

    module = importlib.import_module("openengine_chain.direct_replay_load")
    monkeypatch.setattr(module, "_arms", lambda: arms)
    return module


@pytest.fixture
def engines():
    ctx, gen = FakeEngine(disagg_profile()), FakeEngine(disagg_profile())
    with running_in_thread(ctx) as ctx_engine, running_in_thread(gen) as gen_engine:
        yield ctx_engine, gen_engine


def _endpoint(engine) -> str:
    return f"127.0.0.1:{engine.port}"


def _config(replay, **changes):
    values = dict(
        model=MODEL, input_len=8, output_len=5, concurrency=3, num_requests=7, max_token_id=500
    )
    values.update(changes)
    return replay.LoadConfig(**values)


def test_load_runs_both_legs_of_every_request(replay, engines):
    ctx, gen = engines
    run = replay.measure(_endpoint(ctx), _endpoint(gen), _config(replay))
    assert len(run.records) == 7 and all(r.ok for r in run.records), [r.error for r in run.records]
    for record in run.records:
        assert record.ctx_status == record.gen_status == "OK"
        assert record.prefill_ready_s is not None and record.output_tokens == 5
        assert record.ttft_s > 0 and record.e2el_s >= record.ttft_s
        assert len(record.itl_s()) == len(record.token_event_s) - 1
    counts = replay.leg_counts(run.before, run.after)
    assert (
        counts["ctx"]["generate_received"] == 7 and counts["gen"]["accepted_generation_only"] == 7
    )
    assert counts["ctx"]["prefill_ready_sent"] == 7 and counts["gen"]["accepted_aggregated"] == 0


def test_result_matches_the_e4_direct_schema(replay, arms, engines, tmp_path):
    ctx, gen = engines
    run = replay.measure(_endpoint(ctx), _endpoint(gen), _config(replay))
    doc = replay.build_result(
        run, test_id="direct-fake", gpus={"ctx": 1, "gen": 1}, kv_runtime="PYTHON"
    )
    assert arms.validate_disagg_result(doc) == []
    assert (doc["arm"], doc["topology"], doc["concurrency"], doc["output_tokens"]) == (
        "E4-direct",
        "direct",
        3,
        35,
    )
    assert doc["chain_proof"]["status"] == {"overall": "ok", "ctx": "ok", "gen": "ok"}
    assert doc["chain_proof"]["handoffs"] == 7 and doc["chain_proof"]["orphan_handoffs"] == 0
    assert doc["chain_proof"]["ctx"]["http_received"] is None, (
        "no front end: front-end counts are null"
    )
    assert doc["kv_transfer"]["backend"] == {"ctx": "DEFAULT", "gen": "DEFAULT"}
    assert doc["metrics"]["ttft_ms"]["median"] > 0 and doc["metrics"]["output_tok_per_s"] > 0
    assert doc["first_output"] is None and doc["notes"] == []


def test_counts_that_disagree_fail_the_proof(replay):
    counts = {
        "ctx": {"generate_received": 3, "accepted_context_only": 3, "prefill_ready_sent": None},
        "gen": {"generate_received": 2, "accepted_generation_only": 2, "accepted_aggregated": 0},
    }
    status, details = replay.judge_counts(counts, sent=3, handoffs=3)
    assert status == {"ctx": "incomplete", "gen": "fail", "overall": "fail"}
    assert "gen.generate_received = 2, expected 3" in details


def test_a_failed_context_leg_sends_no_generation_leg(replay):
    record = replay.replay_one(
        _Unavailable(), None, _config(replay, num_requests=1), [1, 2, 3], 0, 0.0
    )
    assert not record.ok and record.ctx_status == "UNAVAILABLE" and record.gen_status == ""
    assert "context leg" in record.error


class _Unavailable:
    def Generate(self, request, timeout=None):
        import grpc

        class _Error(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

            def details(self):
                return "no engine"

        raise _Error()


def test_main_writes_the_result_and_the_records(replay, engines, tmp_path, capsys):
    ctx, gen = engines
    argv = [
        "--ctx", _endpoint(ctx), "--gen", _endpoint(gen), "--model", MODEL,
        "--input-len", "8", "--output-len", "4", "--concurrency", "2", "--num-requests", "4",
        "--warmup-requests", "2", "--max-token-id", "500", "--out-dir", str(tmp_path),
    ]  # fmt: skip
    assert replay.main(argv) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    result = Path(summary["result"])
    assert json.loads(result.read_text())["arm"] == "E4-direct"
    records = result.with_suffix(".requests.jsonl").read_text().splitlines()
    assert len(records) == 4 and all(json.loads(line)["ok"] for line in records)


def test_invalid_arguments(replay, capsys):
    argv = [
        "--ctx",
        "a:1",
        "--gen",
        "b:2",
        "--model",
        "m",
        "--num-requests",
        "0",
        "--out-dir",
        "/x",
    ]
    assert replay.main(argv) == 2
    assert "num_requests must be at least 1" in capsys.readouterr().err

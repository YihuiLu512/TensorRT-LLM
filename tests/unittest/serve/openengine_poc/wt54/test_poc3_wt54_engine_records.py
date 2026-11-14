# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TensorRT-LLM OpenEngine server's leg-duration records (EN-TRTD-8, CO-08), on a stub LLM.

With ``perf_metrics_output_dir``, each finished leg of a disaggregated request
leaves one record, keyed by its ``disagg_request_id``, of the durations the
engine measured: never a time. It does so even if its caller goes away at
the last response. An aggregated request leaves none, and a server without
the directory writes nothing. Records past the writer's limit of records
waiting to be written are dropped and counted.
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

grpc = pytest.importorskip(
    "grpc", reason='gRPC runtime not installed (pip install "grpcio>=1.73.1,<2")'
)
pytest.importorskip("grpc_health", reason="gRPC health runtime not installed")

from tensorrt_llm.disaggregated_params import DisaggregatedParams  # noqa: E402
from tensorrt_llm.grpc.openengine import streaming  # noqa: E402
from tensorrt_llm.grpc.openengine.bindings import generation_pb2, kv_pb2  # noqa: E402
from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer  # noqa: E402
from tensorrt_llm.grpc.openengine.streaming import (  # noqa: E402
    LEG_DURATIONS_FILE_PREFIX,
    LegDurationsWriter,
    leg_durations_record,
)

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"
CALLER_ID = 1_234_567_890_123_456
CTX_REQUEST_ID = 99
CTX_ENDPOINT = "tcp://10.0.0.1:5555"
# Steady-clock offsets in seconds, as the engine's timing metrics hold them.
TIMES = {
    "arrival_time": 100.000,
    "first_scheduled_time": 100.010,
    "first_token_time": 100.050,
    "last_token_time": 100.090,
    "kv_cache_transfer_start": 100.055,
    "kv_cache_transfer_end": 100.080,
}
DURATIONS = {
    "queue": 10.0,
    "scheduled_to_first_token": 40.0,
    "kv_cache_transfer": 25.0,
    "arrival_to_first_token": 50.0,
    "arrival_to_last_token": 90.0,
}
RECORD_KEYS = {
    "leg",
    "request_id",
    "disagg_request_id",
    "ctx_request_id",
    "finish_reason",
    "durations_ms",
    "kv_cache_size",
}


class _Context:
    """The parts of grpc.aio.ServicerContext that Generate uses."""

    def __init__(self) -> None:
        self.abort_code = None
        self.abort_details = None

    def invocation_metadata(self) -> list:
        return []

    def cancelled(self) -> bool:
        return False

    def code(self):
        return self.abort_code

    def time_remaining(self):
        return None

    def add_done_callback(self, callback: Any) -> None:
        pass

    async def abort(self, code, details: str = "") -> None:
        self.abort_code = code
        self.abort_details = details
        raise _Aborted(details)


class _Aborted(Exception):
    pass


class _Tokenizer:
    def decode(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
        return "".join(f"<{token}>" for token in token_ids)


def _timing(**times: float | None) -> SimpleNamespace:
    values = {**TIMES, **times}
    timing = {
        name: (None if value is None else timedelta(seconds=value))
        for name, value in values.items()
    }
    return SimpleNamespace(**timing, kv_cache_size=4096)


def _result(params: DisaggregatedParams | None, timing: Any) -> SimpleNamespace:
    output = SimpleNamespace(
        index=0,
        token_ids=[7],
        text="<7>",
        logprobs=[],
        finish_reason="length",
        stop_reason=None,
        disaggregated_params=params,
        request_perf_metrics=None if timing is None else SimpleNamespace(timing_metrics=timing),
    )
    return SimpleNamespace(
        prompt_token_ids=[1, 2], outputs=[output], cached_tokens=0, error=None, finished=True
    )


class _Handle:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.finished = False
        self.outputs: list[Any] = []
        self.prompt_token_ids: list[int] = []
        self.error = None

    async def __aiter__(self):
        self.finished = True
        self.outputs = self._result.outputs
        self.prompt_token_ids = self._result.prompt_token_ids
        yield self._result

    def abort(self) -> None:
        pass


class _Llm:
    """The LLM surface Generate uses; ends each request with one finished result."""

    def __init__(self, output_dir: Path | None, timing: Any = "default") -> None:
        self.args = SimpleNamespace(
            guided_decoding_backend=None,
            perf_metrics_output_dir=None if output_dir is None else str(output_dir),
        )
        self.tokenizer = _Tokenizer()
        self.timing = _timing() if timing == "default" else timing

    def generate_async(self, **kwargs: Any) -> _Handle:
        kwargs["sampling_params"]._validate()
        params = kwargs["disaggregated_params"]
        if params is not None and params.request_type == "context_only":
            # The engine gives the context leg its ctx_request_id; the rest is the request's.
            params = DisaggregatedParams(
                request_type="context_only",
                ctx_request_id=CTX_REQUEST_ID,
                ctx_info_endpoint=CTX_ENDPOINT,
                first_gen_tokens=[7],
                disagg_request_id=params.disagg_request_id,
            )
        return _Handle(_result(params, self.timing))


def _request(request_id: str, *, extra: dict | None = None, session=None):
    request = generation_pb2.GenerateRequest(
        request_id=request_id,
        model=MODEL,
        prompt="hello",
        stopping=generation_pb2.StoppingOptions(max_tokens=4),
    )
    if extra:
        request.extra.update(extra)
    if session is not None:
        request.kv.session.CopyFrom(session)
    return request


def _context_request(request_id: str = "ctx-1", disagg_id: int = CALLER_ID):
    return _request(
        request_id, extra={"request_type": "context_only", "disagg_request_id": str(disagg_id)}
    )


def _generation_request(request_id: str = "gen-1", disagg_id: int = CALLER_ID):
    session = kv_pb2.KvSessionRef(session_id=str(CTX_REQUEST_ID))
    session.attributes_struct.update(
        {
            "ctx_info_endpoint": CTX_ENDPOINT,
            "first_gen_tokens": [7],
            "disagg_request_id": str(disagg_id),
        }
    )
    return _request(request_id, session=session)


def _run(servicer: OpenEngineInferenceServicer, *requests) -> list[_Context]:
    async def one(request) -> _Context:
        context = _Context()
        try:
            async for _ in servicer.Generate(request, context):
                pass
        except _Aborted:
            pass
        return context

    async def all_of() -> list[_Context]:
        return [await one(request) for request in requests]

    contexts = asyncio.run(all_of())
    if servicer._leg_durations is not None:
        # Writes everything submitted and stops the writer's thread.
        servicer._leg_durations.close()
    return contexts


def _servicer(llm: _Llm) -> OpenEngineInferenceServicer:
    return OpenEngineInferenceServicer(llm, MODEL, kv_transfer_backend="NIXL")


def _records(directory: Path) -> list[dict]:
    files = sorted(directory.glob(f"{LEG_DURATIONS_FILE_PREFIX}-*.jsonl"))
    return [json.loads(line) for path in files for line in path.read_text().splitlines()]


def test_each_finished_leg_leaves_one_record_keyed_by_its_disagg_request_id(tmp_path):
    servicer = _servicer(_Llm(tmp_path))

    contexts = _run(servicer, _context_request(), _generation_request())

    assert [context.abort_code for context in contexts] == [None, None]
    records = _records(tmp_path)
    assert [record["leg"] for record in records] == ["context", "generation"]
    for record, request_id in zip(records, ("ctx-1", "gen-1")):
        assert set(record) == RECORD_KEYS
        assert record["request_id"] == request_id
        assert record["disagg_request_id"] == CALLER_ID
        assert record["ctx_request_id"] == CTX_REQUEST_ID
        assert record["finish_reason"] == "length"
        assert record["durations_ms"] == pytest.approx(DURATIONS)
        assert record["kv_cache_size"] == 4096
    # One file per process, not named like the frontends' perf records.
    assert len(list(tmp_path.iterdir())) == 1
    assert not list(tmp_path.glob("perf_metrics-*"))


def test_a_leg_whose_caller_leaves_at_its_last_response_still_leaves_its_record(tmp_path):
    servicer = _servicer(_Llm(tmp_path))

    async def first_response_only() -> None:
        responses = servicer.Generate(_context_request(), _Context())
        await responses.__anext__()
        await responses.aclose()

    asyncio.run(first_response_only())
    servicer._leg_durations.close()

    assert [record["leg"] for record in _records(tmp_path)] == ["context"]


def test_records_hold_no_time_only_durations(tmp_path):
    _run(_servicer(_Llm(tmp_path)), _context_request())

    (record,) = _records(tmp_path)
    flat = json.dumps(record)
    for value in TIMES.values():
        # Neither the seconds nor the milliseconds of an engine time appear.
        assert str(value) not in flat
        assert str(value * 1000) not in flat
    assert not any(key.endswith("_time") for key in record["durations_ms"])


def test_every_request_of_a_leg_gets_its_own_record(tmp_path):
    servicer = _servicer(_Llm(tmp_path))
    ids = [CALLER_ID + i for i in range(3)]

    _run(servicer, *(_context_request(f"ctx-{i}", disagg_id) for i, disagg_id in enumerate(ids)))

    assert [record["disagg_request_id"] for record in _records(tmp_path)] == ids


def test_an_aggregated_request_leaves_no_record(tmp_path):
    servicer = _servicer(_Llm(tmp_path))

    (context,) = _run(servicer, _request("agg-1"))

    assert context.abort_code is None
    assert _records(tmp_path) == []


def test_a_server_without_an_output_directory_writes_nothing(tmp_path):
    servicer = _servicer(_Llm(None))

    (context,) = _run(servicer, _context_request())

    assert context.abort_code is None
    assert servicer._leg_durations is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "timing",
    [
        None,
        _timing(first_scheduled_time=None, kv_cache_transfer_start=None),
        _timing(first_token_time=99.0),
    ],
    ids=["no-perf-metrics", "missing-times", "negative"],
)
def test_a_duration_the_engine_did_not_measure_is_none_not_zero(tmp_path, timing):
    _run(_servicer(_Llm(tmp_path, timing=timing)), _generation_request())

    (record,) = _records(tmp_path)
    durations = record["durations_ms"]
    assert 0 not in durations.values()
    if timing is None:
        assert set(durations.values()) == {None}
        assert record["kv_cache_size"] is None
    elif timing.first_scheduled_time is None:
        assert durations["queue"] is None
        assert durations["scheduled_to_first_token"] is None
        assert durations["kv_cache_transfer"] is None
        assert durations["arrival_to_first_token"] == pytest.approx(50.0)
    else:
        assert durations["scheduled_to_first_token"] is None
        assert durations["arrival_to_first_token"] is None
        assert durations["queue"] == pytest.approx(10.0)


def test_a_failure_to_write_does_not_fail_the_request(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    servicer = _servicer(_Llm(blocker))

    (context,) = _run(servicer, _context_request())

    assert context.abort_code is None
    assert servicer._leg_durations.dropped_records == 1


def test_a_context_leg_falls_back_to_the_request_ids():
    request_params = DisaggregatedParams(request_type="context_only", disagg_request_id=CALLER_ID)
    result = _result(DisaggregatedParams(request_type="context_only", ctx_request_id=5), None)

    record = leg_durations_record(result, request_params, "ctx-1")

    assert record["disagg_request_id"] == CALLER_ID
    assert record["ctx_request_id"] == 5


def test_the_writer_names_its_file_once_per_process(tmp_path):
    writer = LegDurationsWriter(str(tmp_path / "out"))
    writer.submit({"leg": "context"})
    writer.submit({"leg": "generation"})
    writer.close()

    assert writer.path is not None
    assert writer.path.name.startswith(f"{LEG_DURATIONS_FILE_PREFIX}-")
    assert [json.loads(line)["leg"] for line in writer.path.read_text().splitlines()] == [
        "context",
        "generation",
    ]


def test_records_past_the_pending_limit_are_dropped_and_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(streaming, "_LEG_DURATIONS_MAX_PENDING", 2)
    writer = LegDurationsWriter(str(tmp_path))
    release = threading.Event()
    write = writer._write

    def stalled(line: str) -> None:
        # A file system that does not answer until released.
        release.wait(10)
        write(line)

    monkeypatch.setattr(writer, "_write", stalled)
    for index in range(4):
        writer.submit({"leg": "context", "index": index})
    release.set()
    writer.close()

    assert writer.dropped_records == 2
    assert [json.loads(line)["index"] for line in writer.path.read_text().splitlines()] == [0, 1]

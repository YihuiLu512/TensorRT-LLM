# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The first-output timestamps of the OpenEngine server and the worker front end, on stubs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

grpc = pytest.importorskip("grpc", reason="gRPC runtime not installed")
pytest.importorskip("grpc_health", reason="gRPC health runtime not installed")

from tensorrt_llm.grpc.openengine import servicer as servicer_module  # noqa: E402
from tensorrt_llm.grpc.openengine.bindings import generation_pb2  # noqa: E402
from tensorrt_llm.grpc.openengine.servicer import (  # noqa: E402
    ENGINE_FIRST_OUTPUT_PROBE,
    FIRST_OUTPUT_DIR_ENV,
    FRONTEND_FIRST_OUTPUT_PROBE,
    FirstOutputTimestamps,
    OpenEngineInferenceServicer,
)
from tensorrt_llm.serve.openengine_poc import facade as facade_module  # noqa: E402

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"


def _records(directory: Path) -> list[dict]:
    return [
        json.loads(line)
        for path in sorted(directory.glob("*.jsonl"))
        for line in path.read_text().splitlines()
    ]


def test_writer_appends_one_line_per_record(tmp_path):
    writer = FirstOutputTimestamps(str(tmp_path / "d"), ENGINE_FIRST_OUTPUT_PROBE)
    writer.record("r1", 10)
    writer.record("r2", 20)
    writer._executor.shutdown(wait=True)
    assert _records(tmp_path / "d") == [
        {"probe": ENGINE_FIRST_OUTPUT_PROBE, "request_id": "r1", "t_ns": 10},
        {"probe": ENGINE_FIRST_OUTPUT_PROBE, "request_id": "r2", "t_ns": 20},
    ]
    assert Path(writer.path).name.startswith(f"{ENGINE_FIRST_OUTPUT_PROBE}-")
    assert writer.dropped_records == 0


def test_writer_drops_what_it_cannot_write(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    writer = FirstOutputTimestamps(str(blocker / "d"), ENGINE_FIRST_OUTPUT_PROBE)
    writer.record("r1", 10)
    writer.record("r2", 20)
    writer._executor.shutdown(wait=True)
    assert writer.dropped_records == 2 and writer._pending == 0


def test_writer_drops_records_past_the_pending_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(servicer_module, "_FIRST_OUTPUT_MAX_PENDING", 0)
    writer = FirstOutputTimestamps(str(tmp_path), ENGINE_FIRST_OUTPUT_PROBE)
    writer.record("r1", 10)
    writer._executor.shutdown(wait=True)
    assert writer.dropped_records == 1 and _records(tmp_path) == []


@pytest.mark.parametrize("value", [None, ""])
def test_no_directory_no_writer(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(FIRST_OUTPUT_DIR_ENV, raising=False)
    else:
        monkeypatch.setenv(FIRST_OUTPUT_DIR_ENV, value)
    assert FirstOutputTimestamps.from_env(ENGINE_FIRST_OUTPUT_PROBE) is None
    assert facade_module._first_output_timestamps() is None


def test_front_end_writer(monkeypatch, tmp_path):
    monkeypatch.setenv(FIRST_OUTPUT_DIR_ENV, str(tmp_path))
    writer = facade_module._first_output_timestamps()
    assert Path(writer.path).parent == tmp_path
    assert Path(writer.path).name.startswith(f"{FRONTEND_FIRST_OUTPUT_PROBE}-")


# ----------------------------------------------------------------------------- the server's probe


class _Context:
    """The parts of grpc.aio.ServicerContext that Generate uses."""

    def invocation_metadata(self) -> list:
        return []

    def cancelled(self) -> bool:
        return False

    def code(self):
        return None

    def time_remaining(self):
        return None

    def add_done_callback(self, callback: Any) -> None:
        pass

    async def abort(self, code, details: str = "") -> None:
        raise AssertionError(f"aborted: {code} {details}")


class _Tokenizer:
    def decode(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
        return "".join(f"<{token}>" for token in token_ids)


class _Handle:
    """Two results: one token, then one more token and the end."""

    def __init__(self) -> None:
        self.finished = False
        self.outputs: list[Any] = []
        self.prompt_token_ids = [1, 2]
        self.error = None
        self.request_id = 5

    async def __aiter__(self):
        for tokens, finished in (([7], False), ([7, 8], True)):
            output = SimpleNamespace(
                index=0,
                token_ids=tokens,
                text="".join(f"<{t}>" for t in tokens),
                logprobs=[],
                finish_reason="length" if finished else None,
                stop_reason=None,
                disaggregated_params=None,
                request_perf_metrics=None,
            )
            self.outputs, self.finished = [output], finished
            yield SimpleNamespace(
                prompt_token_ids=self.prompt_token_ids,
                outputs=[output],
                cached_tokens=0,
                error=None,
                finished=finished,
            )

    def abort(self) -> None:
        pass


class _Llm:
    def __init__(self) -> None:
        self.args = SimpleNamespace(guided_decoding_backend=None, perf_metrics_output_dir=None)
        self.tokenizer = _Tokenizer()

    def generate_async(self, **kwargs: Any) -> _Handle:
        return _Handle()


def _generate(servicer, request_id: str) -> list[str]:
    request = generation_pb2.GenerateRequest(
        request_id=request_id,
        model=MODEL,
        prompt="hello",
        stopping=generation_pb2.StoppingOptions(max_tokens=2),
    )

    async def run() -> list[str]:
        return [r.WhichOneof("event") async for r in servicer.Generate(request, _Context())]

    return asyncio.run(run())


def test_server_records_each_calls_first_response_once(monkeypatch, tmp_path):
    monkeypatch.setenv(FIRST_OUTPUT_DIR_ENV, str(tmp_path))
    servicer = OpenEngineInferenceServicer(_Llm(), MODEL, kv_transfer_backend="NIXL")
    events = _generate(servicer, "call-1") + _generate(servicer, "call-2")
    assert events.count("token") >= 2 and "finished" in events
    servicer._first_output._executor.shutdown(wait=True)
    records = _records(tmp_path)
    assert [(r["probe"], r["request_id"]) for r in records] == [
        (ENGINE_FIRST_OUTPUT_PROBE, "call-1"),
        (ENGINE_FIRST_OUTPUT_PROBE, "call-2"),
    ]
    assert all(isinstance(r["t_ns"], int) for r in records)


def test_server_without_the_directory_records_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv(FIRST_OUTPUT_DIR_ENV, raising=False)
    servicer = OpenEngineInferenceServicer(_Llm(), MODEL, kv_transfer_backend="NIXL")
    assert servicer._first_output is None
    assert "finished" in _generate(servicer, "call-1")


# ----------------------------------------------------------------------------- the front end's probe


class _Stream:
    def __init__(self, request_id: str, events: list[Any]) -> None:
        self.request_id = request_id
        self.sent = True
        self._events = events

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self._events:
            yield event

    def cancel(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


def _pump(first_output, request_id: str, events: list[Any]) -> list[Any]:
    output = facade_module.OpenEngineRequestOutput.__new__(facade_module.OpenEngineRequestOutput)
    output._facade = SimpleNamespace(
        first_output=first_output, _record_sent=lambda count, bucket: None
    )
    output._http_request = None
    output._metrics_failed = False
    output._request_metrics = None

    async def run() -> list[Any]:
        output._queue = asyncio.Queue()
        await output._pump(_Stream(request_id, events), 0, "aggregated")
        return [output._queue.get_nowait() for _ in range(output._queue.qsize())]

    return asyncio.run(run())


def test_front_end_records_each_calls_first_event_once(tmp_path):
    writer = FirstOutputTimestamps(str(tmp_path), FRONTEND_FIRST_OUTPUT_PROBE)
    queued = _pump(writer, "call-1", ["e1", "e2", "e3"])
    _pump(writer, "call-2", ["e1"])
    _pump(writer, "call-3", [])
    writer._executor.shutdown(wait=True)
    assert [event for _, event in queued[:3]] == ["e1", "e2", "e3"]
    assert [(r["probe"], r["request_id"]) for r in _records(tmp_path)] == [
        (FRONTEND_FIRST_OUTPUT_PROBE, "call-1"),
        (FRONTEND_FIRST_OUTPUT_PROBE, "call-2"),
    ]


def test_front_end_without_a_writer(tmp_path):
    queued = _pump(None, "call-1", ["e1"])
    assert queued[0] == (0, "e1")

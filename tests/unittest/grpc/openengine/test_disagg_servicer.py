# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admission of disaggregated Generate requests by the OpenEngine server.

A context_only request takes the caller's ``disagg_request_id``; a request with a
``kv.session`` takes ``orig_elapsed_s``; both legs ask for one sequence; a server
without a KV cache transceiver refuses both legs; and a kv.session that locates
no context worker passes only in the gen-only benchmark mode. A dead engine is
not asked to cancel a request. The engine is stubbed, so nothing
here needs a GPU.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, Optional

import pytest

grpc = pytest.importorskip(
    "grpc", reason='gRPC runtime not installed (pip install "grpcio>=1.73.1,<2")'
)
pytest.importorskip("grpc_health", reason="gRPC health runtime not installed")
from conftest import AbortError, FakeServicerContext  # noqa: E402

import tensorrt_llm.grpc.openengine.disagg as oe_disagg  # noqa: E402
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle  # noqa: E402
from tensorrt_llm.executor.utils import EngineDeadError  # noqa: E402
from tensorrt_llm.grpc.openengine.bindings import generation_pb2, kv_pb2  # noqa: E402
from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer  # noqa: E402
from tensorrt_llm.grpc.openengine_conventions import (  # noqa: E402
    COUNTER_GENERATE_ACCEPTED_BUCKETS,
    COUNTER_PREFILL_READY_SENT,
    DISAGG_REQUEST_ID_MIN,
)
from tensorrt_llm.llmapi.disagg_utils import MIN_GLOBAL_ID  # noqa: E402

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"
CALLER_ID = 1_234_567_890_123_456
CTX_ENDPOINT = "tcp://10.0.0.1:5555"


class _Tokenizer:
    def decode(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
        return "".join(f"<{token}>" for token in token_ids)


class _Handle:
    """Stands in for GenerationResult: streams fixed results, records an abort."""

    def __init__(self, results: list[Any]) -> None:
        self._results = results
        self.aborted = False
        self.finished = False
        self.prompt_token_ids: list[int] = []
        self.outputs: list[Any] = []
        self.cached_tokens = 0
        self.error = None

    async def __aiter__(self):
        for result in self._results:
            self.finished = result.finished
            self.prompt_token_ids = result.prompt_token_ids
            self.outputs = result.outputs
            yield result

    def abort(self) -> None:
        self.aborted = True


class _Llm:
    """The LLM surface Generate uses; records what generate_async received."""

    def __init__(self, results: Optional[list[Any]] = None) -> None:
        self.args = SimpleNamespace(guided_decoding_backend=None)
        self.tokenizer = _Tokenizer()
        # None: a context_only request ends in a handoff, any other in a finished output.
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def generate_async(self, **kwargs: Any) -> _Handle:
        kwargs["sampling_params"]._validate()
        self.calls.append(kwargs)
        params = kwargs["disaggregated_params"]
        if self.results is None:
            prefill = params is not None and params.request_type == "context_only"
            return _Handle([_prefill_result() if prefill else _finished_result()])
        return _Handle(self.results)


def _output(**fields: Any) -> SimpleNamespace:
    base = dict(
        index=0,
        token_ids=[7],
        text="<7>",
        logprobs=[],
        finish_reason="length",
        stop_reason=None,
        disaggregated_params=None,
    )
    base.update(fields)
    return SimpleNamespace(**base)


def _finished_result() -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_ids=[1, 2], outputs=[_output()], cached_tokens=0, error=None, finished=True
    )


def _prefill_result() -> SimpleNamespace:
    handoff = oe_disagg.DisaggregatedParams(
        request_type="context_only",
        ctx_request_id=99,
        ctx_info_endpoint=CTX_ENDPOINT,
        first_gen_tokens=[7],
        disagg_request_id=CALLER_ID,
    )
    return SimpleNamespace(
        prompt_token_ids=[1, 2],
        outputs=[_output(disaggregated_params=handoff)],
        cached_tokens=0,
        error=None,
        finished=True,
    )


def _session(**attributes: Any) -> kv_pb2.KvSessionRef:
    session = kv_pb2.KvSessionRef(session_id="99")
    session.attributes_struct.update(
        {"ctx_info_endpoint": CTX_ENDPOINT, "first_gen_tokens": [7], **attributes}
    )
    return session


def _request(
    *,
    extra: Optional[dict[str, Any]] = None,
    session: Optional[kv_pb2.KvSessionRef] = None,
    num_sequences: Optional[int] = None,
) -> generation_pb2.GenerateRequest:
    request = generation_pb2.GenerateRequest(
        request_id="request-1",
        model=MODEL,
        prompt="hello",
        stopping=generation_pb2.StoppingOptions(max_tokens=4),
    )
    if extra:
        request.extra.update(extra)
    if session is not None:
        request.kv.session.CopyFrom(session)
    if num_sequences is not None:
        request.sampling.num_sequences = num_sequences
    return request


def _generate(
    servicer: OpenEngineInferenceServicer, request: generation_pb2.GenerateRequest
) -> tuple[list[generation_pb2.GenerateResponse], FakeServicerContext]:
    context = FakeServicerContext()

    async def collect() -> list[generation_pb2.GenerateResponse]:
        responses = []
        try:
            async for response in servicer.Generate(request, context):
                responses.append(response)
        except AbortError:
            pass
        return responses

    return asyncio.run(collect()), context


def _servicer(llm: Optional[_Llm] = None, backend: str = "NIXL") -> OpenEngineInferenceServicer:
    return OpenEngineInferenceServicer(llm or _Llm(), MODEL, kv_transfer_backend=backend)


# -- NT-030: the caller's disagg_request_id (CO-01) ---------------------------------


def test_context_only_uses_the_caller_disagg_request_id():
    llm = _Llm([_prefill_result()])
    request = _request(extra={"request_type": "context_only", "disagg_request_id": str(CALLER_ID)})

    responses, context = _generate(_servicer(llm), request)

    assert context.abort_code is None, context.abort_details
    params = llm.calls[0]["disaggregated_params"]
    assert params.request_type == "context_only"
    assert params.disagg_request_id == CALLER_ID
    assert params.schedule_style == DisaggScheduleStyle.CONTEXT_FIRST
    ready = [response for response in responses if response.HasField("prefill_ready")]
    assert len(ready) == 1
    attributes = ready[0].prefill_ready.kv_session.attributes_struct
    assert attributes["disagg_request_id"] == str(CALLER_ID)


def test_context_only_without_a_caller_id_still_gets_a_random_one():
    first = oe_disagg.disaggregated_params_from_request(
        _request(extra={"request_type": "context_only"})
    )
    second = oe_disagg.disaggregated_params_from_request(
        _request(extra={"request_type": "context_only"})
    )

    # The id range of the conventions starts where the engine's global ids start.
    assert DISAGG_REQUEST_ID_MIN == MIN_GLOBAL_ID
    for params in (first, second):
        assert (1 << 40) <= params.disagg_request_id < (1 << 63)
        assert params.schedule_style == DisaggScheduleStyle.CONTEXT_FIRST
    assert first.disagg_request_id != second.disagg_request_id


@pytest.mark.parametrize(
    "value",
    ["", "0", "0123", "-5", "+5", " 5", "1099511627775", str(1 << 63), "1e3", 12345.0],
    ids=lambda value: repr(value),
)
def test_a_malformed_caller_id_is_invalid_argument(value):
    request = _request(extra={"request_type": "context_only", "disagg_request_id": value})

    _, context = _generate(_servicer(), request)

    assert context.abort_code == grpc.StatusCode.INVALID_ARGUMENT
    assert "disagg_request_id" in context.abort_details


@pytest.mark.parametrize(
    "fields",
    [
        {"extra": {"disagg_request_id": str(CALLER_ID)}},
        {"extra": {"request_type": "context_and_generation", "disagg_request_id": str(CALLER_ID)}},
        {"extra": {"disagg_request_id": str(CALLER_ID)}, "session": _session()},
        {"extra": {"orig_elapsed_s": 1.5}},
        {"extra": {"request_type": "context_only", "orig_elapsed_s": 1.5}},
    ],
    ids=[
        "aggregated-id",
        "context_and_generation-id",
        "session-id",
        "aggregated-elapsed",
        "prefill-elapsed",
    ],
)
def test_a_request_key_on_the_wrong_leg_is_invalid_argument(fields):
    llm = _Llm()

    _, context = _generate(_servicer(llm), _request(**fields))

    assert context.abort_code == grpc.StatusCode.INVALID_ARGUMENT
    assert "is not accepted on" in context.abort_details
    assert llm.calls == []


def test_schedule_style_is_unimplemented_without_generation_first():
    request = _request(extra={"request_type": "context_only", "schedule_style": 1})

    _, context = _generate(_servicer(), request)

    assert context.abort_code == grpc.StatusCode.UNIMPLEMENTED
    assert "schedule_style" in context.abort_details


def test_disagg_request_keys_are_unimplemented_without_a_transceiver():
    for extra in ({"disagg_request_id": str(CALLER_ID)}, {"orig_elapsed_s": 0.5}):
        _, context = _generate(_servicer(backend=""), _request(extra=extra))

        assert context.abort_code == grpc.StatusCode.UNIMPLEMENTED, extra
        assert "is not supported by this engine" in context.abort_details


# -- NT-031: no KV cache transceiver (CO-03) ----------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"extra": {"request_type": "context_only"}},
        {"session": _session()},
        {"extra": {"request_type": "generation_only"}},
    ],
    ids=["context_only", "kv_session", "generation_only"],
)
def test_disagg_request_without_a_transceiver_is_failed_precondition(fields):
    llm = _Llm()
    servicer = _servicer(llm, backend="")

    _, context = _generate(servicer, _request(**fields))

    assert context.abort_code == grpc.StatusCode.FAILED_PRECONDITION
    assert "KV cache transceiver" in context.abort_details
    assert llm.calls == []
    counters = servicer.generate_counters.attributes()
    assert counters["trtllm.counters.generate_received"] == 1
    assert counters["trtllm.counters.generate_accepted"] == 0


def test_requests_needing_no_handoff_still_run_without_a_transceiver():
    for extra in (None, {"request_type": "context_and_generation"}):
        llm = _Llm()

        _, context = _generate(_servicer(llm, backend=""), _request(extra=extra))

        assert context.abort_code is None, (extra, context.abort_details)
        assert len(llm.calls) == 1


# -- NT-032 / NT-034: one sequence per leg (CO-02) ----------------------------------


@pytest.mark.parametrize(
    "fields",
    [{"extra": {"request_type": "context_only"}}, {"session": _session()}],
    ids=["context_only", "kv_session"],
)
def test_two_leg_request_with_several_sequences_is_rejected(fields):
    llm = _Llm()

    _, context = _generate(_servicer(llm), _request(num_sequences=2, **fields))

    assert context.abort_code == grpc.StatusCode.INVALID_ARGUMENT
    assert "num_sequences" in context.abort_details
    assert llm.calls == []


def test_two_phase_request_with_num_sequences_above_one_is_rejected():
    """Only the two legs are limited: an aggregated request may still ask for several."""
    for fields in ({"extra": {"request_type": "context_only"}}, {"session": _session()}):
        _, context = _generate(_servicer(), _request(num_sequences=3, **fields))

        assert context.abort_code == grpc.StatusCode.INVALID_ARGUMENT, fields
    for extra in (None, {"request_type": "context_and_generation"}):
        llm = _Llm(
            [
                SimpleNamespace(
                    prompt_token_ids=[1],
                    outputs=[_output(index=0), _output(index=1)],
                    cached_tokens=0,
                    error=None,
                    finished=True,
                )
            ]
        )

        _, context = _generate(_servicer(llm), _request(extra=extra, num_sequences=2))

        assert context.abort_code is None, (extra, context.abort_details)
        assert llm.calls[0]["sampling_params"].n == 2
    for fields in ({"extra": {"request_type": "context_only"}}, {"session": _session()}):
        _, context = _generate(_servicer(), _request(num_sequences=1, **fields))

        assert context.abort_code is None, (fields, context.abort_details)


# -- NT-035: orig_elapsed_s (CO-10) --------------------------------------------------


def test_orig_elapsed_s_extra_key_reaches_disaggregated_params():
    llm = _Llm()

    _, context = _generate(
        _servicer(llm), _request(extra={"orig_elapsed_s": 2.25}, session=_session())
    )

    assert context.abort_code is None, context.abort_details
    params = llm.calls[0]["disaggregated_params"]
    assert params.request_type == "generation_only"
    assert params.orig_elapsed_s == 2.25
    unset = oe_disagg.disaggregated_params_from_request(_request(session=_session()))
    assert unset.orig_elapsed_s is None


@pytest.mark.parametrize("value", [-0.5, "1.5", True], ids=repr)
def test_a_malformed_orig_elapsed_s_is_invalid_argument(value):
    _, context = _generate(
        _servicer(), _request(extra={"orig_elapsed_s": value}, session=_session())
    )

    assert context.abort_code == grpc.StatusCode.INVALID_ARGUMENT
    assert "orig_elapsed_s" in context.abort_details


# -- NT-036: the gen-only benchmark mode (CO-12) -------------------------------------


def _benchmark_session() -> kv_pb2.KvSessionRef:
    """The session the orchestrator sends in the gen-only benchmark mode: no endpoint."""
    session = kv_pb2.KvSessionRef(session_id="1")
    session.attributes_struct.update({"first_gen_tokens": [7]})
    return session


def test_gen_only_benchmark_session_needs_the_engine_env(monkeypatch):
    request = _request(session=_benchmark_session())

    monkeypatch.delenv(oe_disagg.GEN_ONLY_BENCHMARK_ENV, raising=False)
    _, context = _generate(_servicer(), request)
    assert context.abort_code == grpc.StatusCode.INVALID_ARGUMENT
    assert context.abort_details == (
        "kv.session carries neither a ctx_info_endpoint nor an opaque_state"
    )

    monkeypatch.setenv(oe_disagg.GEN_ONLY_BENCHMARK_ENV, "0")
    _, context = _generate(_servicer(), request)
    assert context.abort_code == grpc.StatusCode.INVALID_ARGUMENT

    monkeypatch.setenv(oe_disagg.GEN_ONLY_BENCHMARK_ENV, "1")
    llm = _Llm()
    _, context = _generate(_servicer(llm), request)
    assert context.abort_code is None, context.abort_details
    params = llm.calls[0]["disaggregated_params"]
    assert params.request_type == "generation_only"
    assert params.ctx_request_id == 1
    assert params.ctx_info_endpoint is None and params.opaque_state is None


def test_gen_only_benchmark_mode_still_needs_a_session_id(monkeypatch):
    monkeypatch.setenv(oe_disagg.GEN_ONLY_BENCHMARK_ENV, "1")
    session = kv_pb2.KvSessionRef()
    session.attributes_struct.update({"first_gen_tokens": [7]})

    _, context = _generate(_servicer(), _request(session=session))

    assert context.abort_code == grpc.StatusCode.INVALID_ARGUMENT
    assert "neither a session_id nor a disagg_request_id" in context.abort_details


# -- the per-leg counters (CO-04) -----------------------------------------------------


def test_accepted_requests_are_counted_by_bucket_and_prefill_ready_events_are_counted():
    servicer = _servicer()
    _generate(servicer, _request(extra={"request_type": "context_only"}))
    _generate(servicer, _request(session=_session()))
    _generate(servicer, _request())
    _generate(servicer, _request(extra={"request_type": "context_and_generation"}))
    _generate(servicer, _request(num_sequences=2, session=_session()))

    counters = servicer.generate_counters.attributes()
    assert counters["trtllm.counters.generate_received"] == 5
    assert counters["trtllm.counters.generate_accepted"] == 4
    assert {bucket: counters[key] for bucket, key in COUNTER_GENERATE_ACCEPTED_BUCKETS.items()} == {
        "context_only": 1,
        "generation_only": 1,
        "aggregated": 2,
    }
    assert counters[COUNTER_PREFILL_READY_SENT] == 1


# -- an engine that dies after acceptance (CO-07) ---------------------------------------


class _FailingHandle(_Handle):
    def __init__(self, error: Exception) -> None:
        super().__init__([])
        self._error = error

    async def __aiter__(self):
        raise self._error
        yield  # pragma: no cover - makes this an async generator


class _FailingLlm(_Llm):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error
        self.handles: list[_FailingHandle] = []

    def generate_async(self, **kwargs: Any) -> _FailingHandle:
        self.calls.append(kwargs)
        self.handles.append(_FailingHandle(self.error))
        return self.handles[-1]


@pytest.mark.parametrize(
    "fields",
    [{"extra": {"request_type": "context_only"}}, {"session": _session()}],
    ids=["context_only", "kv_session"],
)
def test_a_dead_engine_is_not_asked_to_cancel(fields):
    llm = _FailingLlm(EngineDeadError(RuntimeError("worker exited")))
    servicer = _servicer(llm)

    responses, context = _generate(servicer, _request(**fields))

    assert context.abort_code is None
    assert [response.WhichOneof("event") for response in responses] == ["error"]
    assert "Engine has died" in responses[-1].error.message
    assert not llm.handles[0].aborted
    assert servicer.active_request_count() == 0


def test_a_failure_of_a_live_engine_still_cancels_the_request():
    llm = _FailingLlm(RuntimeError("formatting failed"))
    servicer = _servicer(llm)

    responses, _ = _generate(servicer, _request(extra={"request_type": "context_only"}))

    assert [response.WhichOneof("event") for response in responses] == ["error"]
    assert llm.handles[0].aborted


class _BlockingHandle(_Handle):
    """A request that runs until the test ends it."""

    def __init__(self) -> None:
        super().__init__([])
        self.release = asyncio.Event()

    async def __aiter__(self):
        await self.release.wait()
        yield _finished_result()


class _DownLlm(_Llm):
    """An engine whose executor goes down while a request runs.

    ``shutdown`` sets what ``is_shutdown()`` reports; ``dead`` sets only the
    executor's ``_engine_dead``, as a worker's death does before the error is recorded.
    """

    def __init__(self, how: str) -> None:
        super().__init__()
        self.how = how
        self.down = False
        self._executor = SimpleNamespace(is_shutdown=lambda: self.down and how == "shutdown")
        self.handles: list[_BlockingHandle] = []

    def generate_async(self, **kwargs: Any) -> _BlockingHandle:
        self.calls.append(kwargs)
        self.handles.append(_BlockingHandle())
        return self.handles[-1]


@pytest.mark.parametrize("how", ["shutdown", "dead"])
def test_a_down_engine_is_not_asked_to_cancel_by_abort_or_by_the_caller(how):
    llm = _DownLlm(how)
    servicer = _servicer(llm)

    async def scenario() -> bool:
        stream = servicer.Generate(_request(), FakeServicerContext())
        task = asyncio.ensure_future(stream.__anext__())
        for _ in range(1000):
            if llm.handles:
                break
            await asyncio.sleep(0.01)
        llm.down = True
        if how == "dead":
            llm._executor._engine_dead = True
        found = servicer.abort_request_by_id("request-1")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return found

    found = asyncio.run(scenario())

    assert found
    assert not llm.handles[0].aborted
    assert servicer.active_request_count() == 0

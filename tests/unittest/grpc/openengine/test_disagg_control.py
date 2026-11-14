# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the OpenEngine Control service reports for disaggregated serving.

The per-leg counters in ``GetLoad``, the capability keys of conventions version 4
in ``GetModelInfo``, and in ``GetServerInfo`` the context engine's rendezvous
address and the runtime KV block size. The engine is stubbed, so nothing here
needs a GPU.
"""

import asyncio
import threading
from types import SimpleNamespace
from typing import Any, Optional

import pytest

grpc = pytest.importorskip(
    "grpc", reason='gRPC runtime not installed (pip install "grpcio>=1.73.1,<2")'
)
pytest.importorskip("grpc_health", reason="gRPC health runtime not installed")
from conftest import AbortError, FakeServicerContext  # noqa: E402

import tensorrt_llm.grpc.openengine.control as oe_control  # noqa: E402
import tensorrt_llm.grpc.openengine.disagg as oe_disagg  # noqa: E402
from tensorrt_llm.grpc.openengine.bindings import (  # noqa: E402
    generation_pb2,
    kv_pb2,
    lifecycle_pb2,
    model_pb2,
    server_pb2,
)
from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer  # noqa: E402
from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer  # noqa: E402
from tensorrt_llm.grpc.openengine_conventions import (  # noqa: E402
    COUNTER_GENERATE_ACCEPTED,
    COUNTER_GENERATE_ACCEPTED_BUCKETS,
    COUNTER_GENERATE_RECEIVED,
    COUNTER_PREFILL_READY_SENT,
    VERSION_4_CAPABILITY_KEYS,
    VERSION_4_ENGINE_COUNTERS,
)

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"
CTX_ENDPOINT = "tcp://10.0.0.1:5555"


class _Tokenizer:
    def decode(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
        return "".join(f"<{token}>" for token in token_ids)


def _output(disagg: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        index=0,
        token_ids=[7],
        text="<7>",
        logprobs=[],
        finish_reason="length",
        stop_reason=None,
        disaggregated_params=disagg,
    )


class _Handle:
    def __init__(self, prefill: bool) -> None:
        handoff = (
            oe_disagg.DisaggregatedParams(
                request_type="context_only",
                ctx_request_id=99,
                ctx_info_endpoint=CTX_ENDPOINT,
                first_gen_tokens=[7],
            )
            if prefill
            else None
        )
        self._result = SimpleNamespace(
            prompt_token_ids=[1, 2],
            outputs=[_output(handoff)],
            cached_tokens=0,
            error=None,
            finished=True,
        )
        self.finished = False
        self.prompt_token_ids = [1, 2]
        self.outputs = self._result.outputs
        self.cached_tokens = 0
        self.error = None

    async def __aiter__(self):
        self.finished = True
        yield self._result

    def abort(self) -> None:
        pass


def _llm(
    *,
    tokens_per_block: int = 32,
    kv_capacity: Optional[dict] = None,
    disaggregated_params: Any = None,
) -> SimpleNamespace:
    args = SimpleNamespace(
        max_seq_len=4096,
        max_batch_size=16,
        max_num_tokens=8192,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        enable_lora=False,
        guided_decoding_backend=None,
        max_beam_width=1,
        reasoning_parser=None,
        kv_cache_config=SimpleNamespace(tokens_per_block=tokens_per_block),
        _enable_routing_load=False,
    )
    calls = {"disaggregated_params": 0}

    def get_disaggregated_params() -> Any:
        calls["disaggregated_params"] += 1
        return disaggregated_params

    executor = SimpleNamespace(
        is_shutdown=lambda: False,
        check_health=lambda: True,
        get_kv_cache_capacity=lambda: dict(kv_capacity or {}),
        get_openengine_discovery=lambda: {},
        get_kv_cache_load=lambda: {},
        get_disaggregated_params=get_disaggregated_params,
    )

    def generate_async(**kwargs: Any) -> _Handle:
        params = kwargs["disaggregated_params"]
        return _Handle(prefill=params is not None and params.request_type == "context_only")

    return SimpleNamespace(
        args=args,
        llm_id="instance-1",
        tokenizer=_Tokenizer(),
        input_processor=SimpleNamespace(supports_token_ids_input=True),
        _executor=executor,
        _check_health=lambda: True,
        generate_async=generate_async,
        calls=calls,
    )


def _servicers(
    llm: Optional[SimpleNamespace] = None, backend: str = "NIXL"
) -> tuple[OpenEngineInferenceServicer, OpenEngineControlServicer]:
    llm = llm or _llm()
    inference = OpenEngineInferenceServicer(llm, MODEL, kv_transfer_backend=backend)
    control = OpenEngineControlServicer(llm, MODEL, inference, kv_transfer_backend=backend)
    return inference, control


async def _generate(inference: OpenEngineInferenceServicer, **fields: Any) -> None:
    extra = fields.pop("extra", None)
    session = fields.pop("session", None)
    request = generation_pb2.GenerateRequest(
        request_id=fields.pop("request_id"),
        model=MODEL,
        prompt="hello",
        stopping=generation_pb2.StoppingOptions(max_tokens=4),
        **fields,
    )
    if extra:
        request.extra.update(extra)
    if session is not None:
        request.kv.session.CopyFrom(session)
    try:
        async for _ in inference.Generate(request, FakeServicerContext()):
            pass
    except AbortError:
        pass


async def _load(control: OpenEngineControlServicer) -> dict[str, float]:
    load = await control.GetLoad(server_pb2.GetLoadRequest(), FakeServicerContext())
    return dict(load.attributes)


# -- NT-033: the per-leg counters (CO-04) --------------------------------------------


def test_load_info_reports_the_leg_counters():
    inference, control = _servicers()
    session = kv_pb2.KvSessionRef(session_id="99")
    session.attributes_struct.update({"ctx_info_endpoint": CTX_ENDPOINT, "first_gen_tokens": [7]})

    async def scenario() -> tuple[dict, dict]:
        before = await _load(control)
        await _generate(inference, request_id="ctx", extra={"request_type": "context_only"})
        await _generate(inference, request_id="gen", session=session)
        await _generate(inference, request_id="agg")
        # Refused before acceptance: counted as received only.
        await _generate(inference, request_id="bad", extra={"request_type": "bogus"})
        return before, await _load(control)

    before, after = asyncio.run(scenario())

    buckets = COUNTER_GENERATE_ACCEPTED_BUCKETS
    assert set(VERSION_4_ENGINE_COUNTERS) <= set(before)
    assert all(before[key] == 0 for key in VERSION_4_ENGINE_COUNTERS)
    assert after[COUNTER_GENERATE_RECEIVED] == 4
    assert after[COUNTER_GENERATE_ACCEPTED] == 3
    assert after[buckets["context_only"]] == 1
    assert after[buckets["generation_only"]] == 1
    assert after[buckets["aggregated"]] == 1
    assert sum(after[key] for key in buckets.values()) == after[COUNTER_GENERATE_ACCEPTED]
    assert after[COUNTER_PREFILL_READY_SENT] == 1


# -- capability keys of conventions version 4 (CO-06) ---------------------------------


@pytest.mark.parametrize("backend", ["NIXL", ""], ids=["transceiver", "no-transceiver"])
def test_model_info_declares_every_version_4_capability_key(backend):
    _, control = _servicers(backend=backend)

    info = asyncio.run(control.GetModelInfo(model_pb2.GetModelInfoRequest(), FakeServicerContext()))

    declared = {key: info.extra[key] for key in VERSION_4_CAPABILITY_KEYS}
    enabled = bool(backend)
    assert declared == {
        "trtllm.request.disagg_request_id": enabled,
        "trtllm.disagg.generation_first": False,
        "trtllm.disagg.orig_elapsed_s": enabled,
        "trtllm.stopping.min_tokens": True,
    }


# -- the rendezvous address and the runtime block size (CO-05) ------------------------


def _server_info(control: OpenEngineControlServicer) -> server_pb2.ServerInfo:
    return asyncio.run(
        control.GetServerInfo(server_pb2.GetServerInfoRequest(), FakeServicerContext())
    )


def test_kv_connector_publishes_the_context_rendezvous_address():
    llm = _llm(disaggregated_params={"ctx_dp_rank": 0, "ctx_info_endpoint": [CTX_ENDPOINT]})
    _, control = _servicers(llm)

    first = _server_info(control)
    second = _server_info(control)

    for info in (first, second):
        endpoints = list(info.kv_connector.local_endpoints)
        assert endpoints == [kv_pb2.KvEndpoint(host="10.0.0.1", port=5555, protocol="tcp")]
    # The address is fixed for the life of the engine; it is read once.
    assert llm.calls["disaggregated_params"] == 1
    # The same KvEndpoint a PrefillReady carries for this address.
    handoff = oe_disagg.prefill_ready_response(
        "r",
        oe_disagg.DisaggregatedParams(
            request_type="context_only", ctx_request_id=1, ctx_info_endpoint=CTX_ENDPOINT
        ),
        "NIXL",
    )
    assert list(handoff.prefill_ready.kv_session.endpoints) == endpoints


@pytest.mark.parametrize(
    "published",
    [{}, {"ctx_dp_rank": 0, "ctx_info_endpoint": None}, None],
    ids=["no-params", "no-endpoint", "none"],
)
def test_kv_connector_has_no_address_when_the_transceiver_publishes_none(published, monkeypatch):
    llm = _llm(disaggregated_params=published)
    _, control = _servicers(llm)

    info = _server_info(control)

    assert info.kv_connector.enabled
    assert list(info.kv_connector.local_endpoints) == []
    _server_info(control)
    # An empty answer stands for a while before the worker is asked again.
    assert llm.calls["disaggregated_params"] == 1
    monkeypatch.setattr(oe_control, "_CONTEXT_ENDPOINT_RETRY_SECONDS", 0.0)
    control._context_endpoints_retry_at = 0.0
    _server_info(control)
    assert llm.calls["disaggregated_params"] == 2


def test_kv_connector_finds_the_address_after_a_failed_lookup(monkeypatch):
    monkeypatch.setattr(oe_control, "_CONTEXT_ENDPOINT_RETRY_SECONDS", 0.0)
    answers = iter([RuntimeError("worker busy"), {}, {"ctx_info_endpoint": [CTX_ENDPOINT]}])

    def lookup() -> dict:
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    llm = _llm()
    llm._executor.get_disaggregated_params = lookup
    _, control = _servicers(llm)

    replies = [list(_server_info(control).kv_connector.local_endpoints) for _ in range(4)]

    assert replies[0] == replies[1] == []
    assert [endpoint.port for endpoint in replies[2]] == [5555]
    # The address, once found, is kept: the fourth reply asks nothing.
    assert replies[3] == replies[2]


def test_a_slow_address_lookup_does_not_hold_up_server_info(monkeypatch):
    monkeypatch.setattr(oe_control, "_CONTEXT_ENDPOINT_LOOKUP_TIMEOUT_SECONDS", 0.05)
    release = threading.Event()
    llm = _llm()
    calls = {"count": 0}

    def slow_lookup() -> dict:
        calls["count"] += 1
        release.wait(timeout=30)
        return {"ctx_info_endpoint": [CTX_ENDPOINT]}

    llm._executor.get_disaggregated_params = slow_lookup
    _, control = _servicers(llm)
    request = server_pb2.GetServerInfoRequest()

    async def scenario() -> tuple[list, list, list]:
        while_slow = []
        for _ in range(2):
            info = await control.GetServerInfo(request, FakeServicerContext())
            while_slow.append(list(info.kv_connector.local_endpoints))
        release.set()
        for _ in range(100):
            info = await control.GetServerInfo(request, FakeServicerContext())
            if info.kv_connector.local_endpoints:
                break
        return while_slow[0], while_slow[1], list(info.kv_connector.local_endpoints)

    first, second, settled = asyncio.run(scenario())

    assert first == second == []
    assert [endpoint.port for endpoint in settled] == [5555]
    # The slow lookup was not started again while it ran.
    assert calls["count"] == 1


def test_no_kv_connector_without_a_transceiver():
    llm = _llm(disaggregated_params={"ctx_info_endpoint": [CTX_ENDPOINT]})
    _, control = _servicers(llm, backend="")

    info = _server_info(control)

    assert not info.HasField("kv_connector")
    assert llm.calls["disaggregated_params"] == 0


def test_kv_block_size_is_the_runtime_value():
    configured = _server_info(_servicers(_llm(tokens_per_block=32))[1])
    runtime = _server_info(
        _servicers(_llm(tokens_per_block=32, kv_capacity={"tokensPerBlock": 64}))[1]
    )

    assert configured.capacity.kv_block_size == 32
    assert runtime.capacity.kv_block_size == 64


def test_abort_by_kv_session_stays_unimplemented():
    """CO-11 is not done: the server still refuses to abort by kv_session."""
    _, control = _servicers()
    request = lifecycle_pb2.AbortRequest(kv_session=kv_pb2.KvSessionRef(session_id="99"))
    context = FakeServicerContext()

    with pytest.raises(AbortError):
        asyncio.run(control.Abort(request, context))

    assert context.abort_code == grpc.StatusCode.UNIMPLEMENTED
    assert not _server_info(control).kv_connector.supports_abort_cleanup

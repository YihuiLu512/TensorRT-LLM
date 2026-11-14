# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation-first scheduling over OpenEngine, without an engine (EX-DSG-2, EN-TRTD-9, SV-DSG-2/3).

- The worker frontend sends ``extra.schedule_style`` on a generation-first
  prefill call only. It refuses a generation-first prefill to an engine that
  does not declare ``trtllm.disagg.generation_first``, and one without a
  ``disagg_request_id``.
- A generation worker frontend, which has no context response in
  generation-first, builds the ``kv_session`` from the HTTP fields the
  orchestrator sends.
- The TensorRT-LLM OpenEngine server writes ``extra.schedule_style`` of a
  prefill call into ``DisaggregatedParams`` and refuses it on any other call;
  it refuses a generation-first prefill call without
  ``extra.disagg_request_id``.
- The ``ctx_info_endpoint`` that ``/server_info`` rebuilds from
  ``kv_connector.local_endpoints`` is the string the engine reports (UV-38).
"""

from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace
from typing import Any

import grpc
import pytest

from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.grpc.openengine import disagg as oe_disagg
from tensorrt_llm.grpc.openengine.bindings import generation_pb2, kv_pb2
from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openengine_poc import contracts, facade, handoff, mapping

__extra_import_path__ = [".."]
from fakes import trtllm_like  # noqa: E402

pytestmark = pytest.mark.cpu_only

DISAGG_REQUEST_ID = conv.DISAGG_REQUEST_ID_MIN + 55
CTX_ENDPOINT = "tcp://10.1.2.3:4567"


def _caps(*, generation_first: bool = True):
    caps = trtllm_like()[1]
    declared = set(caps.extra.declared) | {conv.CAP_REQ_DISAGG_REQUEST_ID}
    if generation_first:
        declared.add(conv.CAP_DISAGG_GENERATION_FIRST)
    return dataclasses.replace(caps, extra=dataclasses.replace(caps.extra, declared=declared))


def _map(params: DisaggregatedParams, caps=None):
    (request,) = mapping.map_request(
        [1, 2, 3],
        SamplingParams(max_tokens=4),
        caps or _caps(),
        contracts.GenerateOptions(disaggregated_params=params),
        engine_name="eng",
    )
    return request


def _ctx_params(style: Any) -> DisaggregatedParams:
    return DisaggregatedParams(
        request_type="context_only", disagg_request_id=DISAGG_REQUEST_ID, schedule_style=style
    )


def _gen_first_gen_params() -> DisaggregatedParams:
    """What the orchestrator sends a generation worker in generation-first.

    ``openai_disagg_service._get_gen_request`` without a context response:
    both ids are the orchestrator's id, the endpoint and the DP rank come from
    the context worker's ``/server_info``.
    """
    return DisaggregatedParams(
        request_type="generation_only",
        ctx_request_id=DISAGG_REQUEST_ID,
        disagg_request_id=DISAGG_REQUEST_ID,
        schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
        ctx_info_endpoint=CTX_ENDPOINT,
        ctx_dp_rank=None,
    )


# -- CS-14: the worker frontend's mapping -------------------------------------------------


def test_a_generation_first_prefill_sends_schedule_style():
    request = _map(_ctx_params(DisaggScheduleStyle.GENERATION_FIRST))
    assert request.request_type == conv.REQUEST_TYPE_CONTEXT_ONLY
    assert request.extra[conv.REQ_DISAGG_REQUEST_ID] == str(DISAGG_REQUEST_ID)
    assert request.extra[conv.REQ_SCHEDULE_STYLE] == conv.SCHEDULE_STYLE_GENERATION_FIRST
    assert request.kv_session is None


@pytest.mark.parametrize("style", [None, DisaggScheduleStyle.CONTEXT_FIRST], ids=["unset", "0"])
def test_a_context_first_prefill_does_not_send_schedule_style(style):
    request = _map(_ctx_params(style))
    assert conv.REQ_SCHEDULE_STYLE not in request.extra


def test_a_generation_first_prefill_to_an_engine_without_it_is_refused():
    with pytest.raises(ValueError, match=r"generation-first.*trtllm\.disagg\.generation_first"):
        _map(_ctx_params(DisaggScheduleStyle.GENERATION_FIRST), _caps(generation_first=False))


def test_a_generation_first_prefill_without_a_disagg_request_id_is_refused():
    params = _ctx_params(DisaggScheduleStyle.GENERATION_FIRST)
    params.disagg_request_id = None
    with pytest.raises(ValueError, match="generation-first context leg needs a disagg_request_id"):
        _map(params)


@pytest.mark.parametrize("generation_first", [True, False], ids=["declared", "undeclared"])
def test_the_generation_leg_never_sends_schedule_style_in_extra(generation_first):
    # EX-DSG-7: schedule_style is a prefill-only extra key; the decode call has it in kv.session.
    request = _map(_gen_first_gen_params(), _caps(generation_first=generation_first))
    assert request.request_type is None
    assert conv.REQ_SCHEDULE_STYLE not in request.extra
    assert conv.REQ_DISAGG_REQUEST_ID not in request.extra


def test_the_extra_of_the_two_legs_differs_only_in_the_keys_of_the_single_leg_table():
    prefill = _map(_ctx_params(DisaggScheduleStyle.GENERATION_FIRST))
    decode = _map(_gen_first_gen_params())
    leg_keys = {
        name.removeprefix("extra.")
        for name in conv.LEG_SPECIFIC_FIELDS
        if name.startswith("extra.")
    }
    assert set(prefill.extra) ^ set(decode.extra) <= leg_keys
    for key in set(prefill.extra) & set(decode.extra):
        assert prefill.extra[key] == decode.extra[key], key


# -- CS-14: the generation worker frontend builds kv_session from the HTTP fields -----------


def test_the_generation_worker_builds_the_session_from_the_http_fields():
    session = handoff.encode_kv_session(_gen_first_gen_params(), None, "DEFAULT")
    assert session.session_id == str(DISAGG_REQUEST_ID)
    assert session.dp_rank == 0
    assert [(e.host, e.port, e.protocol) for e in session.endpoints] == [("10.1.2.3", 4567, "tcp")]
    # Only mapped attributes: no openengine_handoff_attributes, so no passthrough key.
    assert dict(session.attributes) == {
        "ctx_info_endpoint": CTX_ENDPOINT,
        "disagg_request_id": str(DISAGG_REQUEST_ID),
        "schedule_style": conv.SCHEDULE_STYLE_GENERATION_FIRST,
    }
    assert set(session.attributes) <= set(conv.HANDOFF_MAPPED_ATTRIBUTES)


def test_the_session_built_from_the_http_fields_reaches_the_engine_as_generation_first():
    session = handoff.encode_kv_session(_gen_first_gen_params(), None, "DEFAULT")
    request = generation_pb2.GenerateRequest(request_id="decode-1", model="m", prompt="hello")
    request.kv.session.CopyFrom(_session_proto(session))
    params = oe_disagg.disaggregated_params_from_request(request)
    assert params.request_type == conv.REQUEST_TYPE_GENERATION_ONLY
    assert params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST
    assert params.ctx_request_id == params.disagg_request_id == DISAGG_REQUEST_ID
    assert params.ctx_info_endpoint == CTX_ENDPOINT
    assert params.opaque_state is None and params.first_gen_tokens is None


def _session_proto(session) -> kv_pb2.KvSessionRef:
    proto = kv_pb2.KvSessionRef(
        session_id=session.session_id,
        transfer_backend=session.transfer_backend,
        dp_rank=session.dp_rank,
    )
    for endpoint in session.endpoints:
        proto.endpoints.add(host=endpoint.host, port=endpoint.port, protocol=endpoint.protocol)
    proto.attributes_struct.update(dict(session.attributes))
    return proto


# -- UV-38: /server_info's ctx_info_endpoint is the string the engine reports ----------------


@pytest.mark.parametrize(
    "endpoint, backend",
    [
        (CTX_ENDPOINT, "DEFAULT"),
        ("tcp://[fd00::12]:4567", "NIXL"),
        ("10.1.2.3:4567", "DEFAULT"),
    ],
    ids=["tcp", "ipv6", "no-scheme"],
)
def test_server_info_gives_back_the_engines_endpoint_string(endpoint, backend):
    located = oe_disagg.kv_endpoint(endpoint, backend)
    assert located is not None
    rebuilt = facade._endpoint_text(
        SimpleNamespace(host=located.host, port=located.port, protocol=located.protocol), backend
    )
    assert rebuilt == endpoint


# -- CO-09: the TensorRT-LLM OpenEngine server reads extra.schedule_style -------------------


def _request(
    extra: dict[str, Any] | None = None, session: kv_pb2.KvSessionRef | None = None
) -> generation_pb2.GenerateRequest:
    request = generation_pb2.GenerateRequest(
        request_id="request-1",
        model="test-model",
        prompt="hello",
        stopping=generation_pb2.StoppingOptions(max_tokens=4),
    )
    if extra:
        request.extra.update(extra)
    if session is not None:
        request.kv.session.CopyFrom(session)
    return request


def _prefill(**extra: Any) -> generation_pb2.GenerateRequest:
    return _request(
        {"request_type": "context_only", "disagg_request_id": str(DISAGG_REQUEST_ID), **extra}
    )


def _decode_session() -> kv_pb2.KvSessionRef:
    session = kv_pb2.KvSessionRef(session_id=str(DISAGG_REQUEST_ID))
    session.attributes_struct.update({"ctx_info_endpoint": CTX_ENDPOINT, "schedule_style": 1})
    return session


def test_the_server_declares_generation_first_with_a_transceiver_only():
    assert oe_disagg.disagg_capabilities(True)[conv.CAP_DISAGG_GENERATION_FIRST] is True
    assert oe_disagg.disagg_capabilities(False)[conv.CAP_DISAGG_GENERATION_FIRST] is False


@pytest.mark.parametrize(
    "extra, expected",
    [
        ({"schedule_style": 1}, DisaggScheduleStyle.GENERATION_FIRST),
        ({"schedule_style": 0}, DisaggScheduleStyle.CONTEXT_FIRST),
        ({}, DisaggScheduleStyle.CONTEXT_FIRST),
    ],
    ids=["generation-first", "context-first", "absent"],
)
def test_a_prefill_call_takes_its_schedule_style_from_extra(extra, expected):
    params = oe_disagg.disaggregated_params_from_request(_prefill(**extra))
    assert params.request_type == conv.REQUEST_TYPE_CONTEXT_ONLY
    assert params.disagg_request_id == DISAGG_REQUEST_ID
    assert params.schedule_style == expected


@pytest.mark.parametrize("value", [2, -1, 0.5, True, "1", None], ids=repr)
def test_a_schedule_style_other_than_0_or_1_is_refused(value):
    with pytest.raises(ValueError, match="schedule_style must be 0"):
        oe_disagg.disaggregated_params_from_request(_prefill(schedule_style=value))


@pytest.mark.parametrize(
    "request_",
    [
        _request({"schedule_style": 1}, _decode_session()),
        _request({"schedule_style": 1}),
        _request({"request_type": "context_and_generation", "schedule_style": 1}),
    ],
    ids=["decode", "aggregated", "context_and_generation"],
)
def test_schedule_style_is_refused_on_any_call_but_a_prefill(request_):
    with pytest.raises(ValueError, match="schedule_style is not accepted on"):
        oe_disagg.disaggregated_params_from_request(request_)


def test_a_generation_first_prefill_call_without_a_disagg_request_id_is_refused():
    request = _request({"request_type": "context_only", "schedule_style": 1})
    with pytest.raises(ValueError, match="generation-first context_only request must carry"):
        oe_disagg.disaggregated_params_from_request(request)


def test_a_server_without_a_transceiver_refuses_a_generation_first_prefill():
    with pytest.raises(oe_disagg.DisaggUnavailableError):
        oe_disagg.disaggregated_params_from_request(
            _prefill(schedule_style=1), kv_transfer_enabled=False
        )


class _Context:
    """The subset of grpc.aio.ServicerContext the servicer uses."""

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

    async def abort(self, code, details: str) -> None:
        self.abort_code, self.abort_details = code, details
        raise RuntimeError(details)


class _Handle:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.finished = False
        self.prompt_token_ids: list[int] = []
        self.outputs: list[Any] = []
        self.cached_tokens = 0
        self.error = None

    async def __aiter__(self):
        self.finished = True
        self.prompt_token_ids = self._result.prompt_token_ids
        self.outputs = self._result.outputs
        yield self._result

    def abort(self) -> None:
        pass


class _Llm:
    """Records what generate_async received; answers a prefill with a handoff."""

    def __init__(self) -> None:
        self.args = SimpleNamespace(guided_decoding_backend=None)
        self.tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens=False: "x")
        self.calls: list[dict[str, Any]] = []

    def generate_async(self, **kwargs: Any) -> _Handle:
        self.calls.append(kwargs)
        params = kwargs["disaggregated_params"]
        handoff_params = DisaggregatedParams(
            request_type="context_only",
            ctx_request_id=DISAGG_REQUEST_ID,
            ctx_info_endpoint=CTX_ENDPOINT,
            first_gen_tokens=[7],
            disagg_request_id=params.disagg_request_id,
            schedule_style=params.schedule_style,
        )
        output = SimpleNamespace(
            index=0,
            token_ids=[7],
            text="x",
            logprobs=[],
            finish_reason="length",
            stop_reason=None,
            disaggregated_params=handoff_params,
        )
        return _Handle(
            SimpleNamespace(
                prompt_token_ids=[1, 2],
                outputs=[output],
                cached_tokens=0,
                error=None,
                finished=True,
            )
        )


def test_the_server_hands_a_generation_first_prefill_to_the_llm_as_one():
    llm = _Llm()
    servicer = OpenEngineInferenceServicer(llm, "test-model", kv_transfer_backend="NIXL")
    context = _Context()

    async def collect() -> list:
        return [r async for r in servicer.Generate(_prefill(schedule_style=1), context)]

    responses = asyncio.run(collect())
    assert context.abort_code is None, context.abort_details
    (call,) = llm.calls
    assert call["disaggregated_params"].schedule_style == DisaggScheduleStyle.GENERATION_FIRST
    (ready,) = [r for r in responses if r.HasField("prefill_ready")]
    attributes = ready.prefill_ready.kv_session.attributes_struct
    assert attributes["schedule_style"] == conv.SCHEDULE_STYLE_GENERATION_FIRST


def test_the_server_refuses_a_generation_first_prefill_without_an_id_as_invalid_argument():
    llm = _Llm()
    servicer = OpenEngineInferenceServicer(llm, "test-model", kv_transfer_backend="NIXL")
    context = _Context()
    request = _request({"request_type": "context_only", "schedule_style": 1})

    async def collect() -> list:
        return [r async for r in servicer.Generate(request, context)]

    with pytest.raises(RuntimeError):
        asyncio.run(collect())
    assert context.abort_code == grpc.StatusCode.INVALID_ARGUMENT
    assert "disagg_request_id" in context.abort_details
    assert llm.calls == []

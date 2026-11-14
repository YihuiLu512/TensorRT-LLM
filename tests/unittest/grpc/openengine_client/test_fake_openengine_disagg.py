# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated serving on the CPU fake OpenEngine engine, driven through gRPC stubs."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import grpc
import pytest
from fake_openengine import (
    PROFILES,
    DisaggFaults,
    FakeEngine,
    FakeKvConnector,
    FakeProfile,
    Hang,
    disagg_profile,
)
from google.protobuf import json_format

from tensorrt_llm.grpc.openengine.affinity import AFFINITY_AUTH_HEADER, AFFINITY_ID_HEADER
from tensorrt_llm.grpc.openengine.bindings import (
    generation_pb2,
    kv_pb2,
    lifecycle_pb2,
    model_pb2,
    openengine_pb2_grpc,
    server_pb2,
)
from tensorrt_llm.grpc.openengine_conventions import (
    BUCKET_AGGREGATED,
    BUCKET_CONTEXT_ONLY,
    BUCKET_GENERATION_ONLY,
    CAP_DISAGG_GENERATION_FIRST,
    CAP_DISAGG_ORIG_ELAPSED_S,
    CAP_REQ_DISAGG_REQUEST_ID,
    CAP_STOPPING_MIN_TOKENS,
    CONVENTIONS_VERSION_KEY,
    COUNTER_GENERATE_ACCEPTED,
    COUNTER_GENERATE_ACCEPTED_BUCKETS,
    COUNTER_GENERATE_RECEIVED,
    COUNTER_PREFILL_READY_SENT,
    DISAGG_REQUEST_ID_LIMIT,
    DISAGG_REQUEST_ID_MIN,
    VERSION_4_CAPABILITY_KEYS,
    VERSION_4_ENGINE_COUNTERS,
)

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

WAIT_SECONDS = 10.0
PROMPT = [5, 6, 7, 8]
EOS = PROFILES["trtllm-like"].eos_token_ids[0]
CALLER_ID = str(DISAGG_REQUEST_ID_MIN + 12345)
AFFINITY_KEY = "affinity-key"
# A prefill engine and a decode engine of each shape.
SHAPES = {
    "trtllm": (lambda: disagg_profile("trtllm-like"), lambda: disagg_profile("trtllm-like")),
    "ds": (
        lambda: disagg_profile("ds-like", "context"),
        lambda: disagg_profile("ds-like", "generation"),
    ),
}
# The handoff attribute keys each shape's engine sends without logprobs.
HANDOFF_KEYS = {"ctx_info_endpoint", "first_gen_tokens", "disagg_request_id", "ctx_dp_rank"}


def run(coroutine: Any) -> Any:
    async def bounded() -> Any:
        return await asyncio.wait_for(coroutine, WAIT_SECONDS * 3)

    return asyncio.run(bounded())


@contextlib.asynccontextmanager
async def serving(*engines: FakeEngine) -> AsyncIterator[tuple[FakeEngine, ...]]:
    async with contextlib.AsyncExitStack() as stack:
        for engine in engines:
            await stack.enter_async_context(engine)
        yield engines


def request(
    prompt: Sequence[int] = PROMPT,
    *,
    extra: dict[str, Any] | None = None,
    session: kv_pb2.KvSessionRef | None = None,
    **fields: Any,
) -> generation_pb2.GenerateRequest:
    """A token-id request; ``fields`` are ``GenerateRequest`` fields as dicts."""
    message = generation_pb2.GenerateRequest(
        model="fake-model", token_ids=generation_pb2.TokenIds(ids=list(prompt))
    )
    json_format.ParseDict(fields, message)
    if extra is not None:
        message.extra.update(extra)
    if session is not None:
        message.kv.session.CopyFrom(session)
    return message


def prefill(**fields: Any) -> generation_pb2.GenerateRequest:
    extra = {"request_type": "context_only", **fields.pop("extra", {})}
    return request(extra=extra, **fields)


async def generate(
    engine: FakeEngine,
    message: generation_pb2.GenerateRequest,
    metadata: Sequence[tuple[str, str]] = (),
) -> tuple[list[generation_pb2.GenerateResponse], grpc.StatusCode, str]:
    """Generate on a plain stub: every response, the final status and its details."""
    if not message.request_id:
        message.request_id = f"call-{len(engine.received)}"
    async with grpc.aio.insecure_channel(engine.target) as channel:
        call = openengine_pb2_grpc.InferenceStub(channel).Generate(
            message, metadata=tuple(metadata), timeout=WAIT_SECONDS
        )
        responses = []
        try:
            async for response in call:
                responses.append(response)
        except grpc.aio.AioRpcError as error:
            return responses, error.code(), error.details()
        return responses, await call.code(), await call.details()


async def accepted(
    engine: FakeEngine,
    message: generation_pb2.GenerateRequest,
    metadata: Sequence[tuple[str, str]] = (),
) -> list[generation_pb2.GenerateResponse]:
    responses, code, details = await generate(engine, message, metadata)
    assert code == grpc.StatusCode.OK, details
    return responses


def events(responses: Sequence[generation_pb2.GenerateResponse]) -> list[str]:
    return [response.WhichOneof("event") for response in responses]


def tokens(responses: Sequence[generation_pb2.GenerateResponse]) -> list[int]:
    return [
        info.token_id
        for response in responses
        if response.WhichOneof("event") == "token"
        for info in response.token.tokens
    ]


def handoff(responses: Sequence[generation_pb2.GenerateResponse]) -> kv_pb2.KvSessionRef:
    assert events(responses)[-1] == "prefill_ready", events(responses)
    return responses[-1].prefill_ready.kv_session


def attributes(session: kv_pb2.KvSessionRef) -> dict[str, Any]:
    return json_format.MessageToDict(session.attributes_struct)


def sign(message: generation_pb2.GenerateRequest, parent: str, key: str = AFFINITY_KEY) -> str:
    """The subagent affinity signature, computed as ``affinity.py`` checks it."""
    payload = {
        "purpose": AFFINITY_ID_HEADER,
        "model": message.model,
        "request_id": message.request_id,
        "conversation_id": message.extra["conversation_id"],
        "subagent_affinity_id": parent,
        "request_type": "context_only",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256=" + hmac.new(key.encode(), encoded, hashlib.sha256).hexdigest()


# -- Declarations ----------------------------------------------------------------------


async def _declarations(
    engine: FakeEngine,
) -> tuple[server_pb2.ServerInfo, dict[str, Any], lifecycle_pb2.HealthResponse, dict[str, Any]]:
    async with grpc.aio.insecure_channel(engine.target) as channel:
        control = openengine_pb2_grpc.ControlStub(channel)
        info = await control.GetServerInfo(server_pb2.GetServerInfoRequest())
        model = await control.GetModelInfo(model_pb2.GetModelInfoRequest())
        health = await control.Health(lifecycle_pb2.HealthRequest())
        load = await control.GetLoad(server_pb2.GetLoadRequest())
    return (
        info,
        json_format.MessageToDict(model.extra),
        health,
        json_format.MessageToDict(load.attributes),
    )


def test_a_disaggregated_trtllm_engine_declares_its_connector_and_version_4() -> None:
    async def main() -> None:
        async with serving(FakeEngine(disagg_profile("trtllm-like"))) as (engine,):
            info, extra, health, load = await _declarations(engine)
        connector = info.kv_connector
        assert connector.enabled and connector.supports_remote_prefill
        assert connector.transfer_backend == "DEFAULT"
        assert not connector.supports_decode_pull and not connector.supports_abort_cleanup
        assert connector.HasField("schema_version")
        # Like the server, it names its KV endpoint only in each handoff.
        assert not connector.local_endpoints
        assert info.engine_role == server_pb2.ENGINE_ROLE_UNSPECIFIED
        assert json_format.MessageToDict(info.extra)[CONVENTIONS_VERSION_KEY] == 4
        assert {key: extra[key] for key in VERSION_4_CAPABILITY_KEYS} == {
            CAP_REQ_DISAGG_REQUEST_ID: True,
            CAP_DISAGG_GENERATION_FIRST: False,
            CAP_DISAGG_ORIG_ELAPSED_S: True,
            CAP_STOPPING_MIN_TOKENS: True,
        }
        checks = {check.name: check for check in health.checks}
        assert checks["kv_connector"].state == lifecycle_pb2.HEALTH_STATE_READY
        assert load == dict.fromkeys(
            (COUNTER_GENERATE_RECEIVED, COUNTER_GENERATE_ACCEPTED, *VERSION_4_ENGINE_COUNTERS), 0
        )

    run(main())


def test_ds_instances_declare_their_role_and_the_context_one_its_endpoint() -> None:
    async def main() -> None:
        context = FakeEngine(disagg_profile("ds-like", "context"))
        generation = FakeEngine(disagg_profile("ds-like", "generation"))
        async with serving(context, generation):
            ctx_info, extra, _, _ = await _declarations(context)
            gen_info, _, _, _ = await _declarations(generation)
        assert ctx_info.engine_name == "dsv4_serve"
        assert ctx_info.engine_role == server_pb2.ENGINE_ROLE_PREFILL
        assert gen_info.engine_role == server_pb2.ENGINE_ROLE_DECODE
        [endpoint] = ctx_info.kv_connector.local_endpoints
        assert (endpoint.host, endpoint.port, endpoint.protocol) == (
            "127.0.0.1",
            context.port,
            "tcp",
        )
        assert not gen_info.kv_connector.local_endpoints
        assert {key: extra[key] for key in VERSION_4_CAPABILITY_KEYS} == {
            CAP_REQ_DISAGG_REQUEST_ID: True,
            CAP_DISAGG_GENERATION_FIRST: False,
            CAP_DISAGG_ORIG_ELAPSED_S: False,
            CAP_STOPPING_MIN_TOKENS: False,
        }

    run(main())


@pytest.mark.parametrize("name", ["trtllm-like", "glm-like", "ds-like"])
def test_without_a_connector_there_is_no_kv_connector_and_no_version_4_counter(name: str) -> None:
    async def main() -> None:
        async with serving(FakeEngine(name)) as (engine,):
            info, extra, health, load = await _declarations(engine)
        assert not info.HasField("kv_connector")
        assert "kv_connector" not in {check.name for check in health.checks}
        assert set(load) == {COUNTER_GENERATE_RECEIVED, COUNTER_GENERATE_ACCEPTED}
        # Only an engine that serves disaggregated requests reports the version 4 keys.
        reported = [key for key in VERSION_4_CAPABILITY_KEYS if key in extra]
        assert reported == ([] if name == "glm-like" else list(VERSION_4_CAPABILITY_KEYS))

    run(main())


def test_disagg_profile_takes_a_role_only_for_ds() -> None:
    with pytest.raises(ValueError, match="no role"):
        disagg_profile("trtllm-like", "context")
    with pytest.raises(ValueError, match="needs a role"):
        disagg_profile("ds-like")
    with pytest.raises(ValueError, match="does not serve"):
        disagg_profile("glm-like")
    with pytest.raises(ValueError, match="need a handoff_shape"):
        PROFILES["glm-like"].replace(kv_connector=FakeKvConnector())
    profile = disagg_profile("trtllm-like", kv_connector=FakeKvConnector(transfer_backend="UCX"))
    assert profile.kv_connector.transfer_backend == "UCX"
    # Applied again, it keeps the connector the profile declares.
    assert disagg_profile(profile).kv_connector == profile.kv_connector
    with pytest.raises(ValueError, match="from its arguments"):
        disagg_profile("ds-like", "context", engine_role=server_pb2.ENGINE_ROLE_DECODE)
    # A trtllm profile has no role, so its engine_role is the caller's to set.
    assert (
        disagg_profile("trtllm-like", engine_role=server_pb2.ENGINE_ROLE_AGGREGATED).engine_role
        == server_pb2.ENGINE_ROLE_AGGREGATED
    )
    # A ds profile applied again keeps its role.
    context = disagg_profile("ds-like", "context")
    assert disagg_profile(context) == context
    # Profiles stay hashable, so they can be told apart in sets.
    assert len({profile, disagg_profile("trtllm-like"), PROFILES["trtllm-like"]}) == 3


def test_a_profile_refuses_settings_its_shape_does_not_have() -> None:
    trtllm, ds = PROFILES["trtllm-like"], PROFILES["ds-like"]
    with pytest.raises(ValueError, match="reject_kv_session"):
        trtllm.replace(reject_kv_session=True)
    with pytest.raises(ValueError, match="no instance_role"):
        trtllm.replace(instance_role="context")
    with pytest.raises(ValueError, match="needs an instance_role"):
        ds.replace(kv_connector=FakeKvConnector())
    with pytest.raises(ValueError, match="engine_role"):
        ds.replace(instance_role="context")
    context = disagg_profile("ds-like", "context")
    with pytest.raises(ValueError, match="engine_role"):
        context.replace(instance_role="generation")


# -- The prefill call -------------------------------------------------------------------


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_prefill_call_sends_its_first_token_then_the_handoff(shape: str) -> None:
    async def main() -> None:
        async with serving(FakeEngine(SHAPES[shape][0]())) as (engine,):
            message = prefill(extra={"disagg_request_id": CALLER_ID}, stopping={"maxTokens": 8})
            responses = await accepted(engine, message)
            counters = engine.counters
        assert events(responses) == ["token", "prefill_ready"]
        assert tokens(responses) == PROMPT[:1]
        usage = responses[-1].usage
        assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
            len(PROMPT),
            1,
            len(PROMPT) + 1,
        )
        assert all(response.request_id == message.request_id for response in responses)
        session = handoff(responses)
        # As the TensorRT-LLM server sends it: session_id is the
        # disaggregated request id, and every number is a float.
        assert session.session_id == CALLER_ID
        assert session.transfer_backend == "DEFAULT"
        [endpoint] = session.endpoints
        assert (endpoint.host, endpoint.port, endpoint.protocol) == (
            "127.0.0.1",
            engine.port,
            "tcp",
        )
        values = attributes(session)
        assert set(values) == HANDOFF_KEYS
        assert values["ctx_info_endpoint"] == f"tcp://127.0.0.1:{engine.port}"
        assert values["disagg_request_id"] == CALLER_ID
        assert values["first_gen_tokens"] == [float(PROMPT[0])]
        assert values["ctx_dp_rank"] == 0.0
        assert engine.handoffs == [(message.request_id, session)]
        assert counters[COUNTER_GENERATE_ACCEPTED_BUCKETS[BUCKET_CONTEXT_ONLY]] == 1
        assert counters[COUNTER_PREFILL_READY_SENT] == 1

    run(main())


def test_a_prefill_call_without_an_id_gets_one_in_the_conventions_range() -> None:
    async def main() -> None:
        async with serving(FakeEngine(disagg_profile())) as (engine,):
            first = handoff(await accepted(engine, prefill()))
            second = handoff(await accepted(engine, prefill()))
        for session in (first, second):
            assert DISAGG_REQUEST_ID_MIN <= int(session.session_id) < DISAGG_REQUEST_ID_LIMIT
            assert attributes(session)["disagg_request_id"] == session.session_id
        assert first.session_id != second.session_id

    run(main())


@pytest.mark.parametrize(
    "stopping",
    [
        {"maxTokens": 1},
        {"maxTokens": 4, "conditions": [{"stopTokenId": PROMPT[0]}]},
        {"maxTokens": 4, "ignoreEos": True},
        {"maxTokens": 4, "conditions": [{"stopText": "<5>"}]},
    ],
    ids=["one-token", "first-token-is-a-stop-token", "ignore-eos", "stop-text"],
)
def test_the_stopping_options_do_not_end_a_trtllm_prefill_call_early(
    stopping: dict[str, Any],
) -> None:
    async def main() -> None:
        async with serving(FakeEngine(disagg_profile())) as (engine,):
            responses = await accepted(engine, prefill(stopping=stopping))
        assert events(responses) == ["token", "prefill_ready"]
        assert tokens(responses) == PROMPT[:1]

    run(main())


@pytest.mark.parametrize("stops_on", ["stop-token", "eos"])
def test_a_ds_prefill_call_ends_on_a_first_token_that_stops_it(stops_on: str) -> None:
    eos = PROFILES["ds-like"].eos_token_ids[0]
    if stops_on == "eos":
        message = prefill(prompt=[eos, *PROMPT], stopping={"maxTokens": 4})
    else:
        message = prefill(stopping={"maxTokens": 4, "conditions": [{"stopTokenId": PROMPT[0]}]})

    async def main() -> None:
        async with serving(FakeEngine(disagg_profile("ds-like", "context"))) as (engine,):
            responses = await accepted(engine, message)
            counters = engine.counters
        assert events(responses) == ["finished"]
        finished = responses[-1].finished
        assert finished.reason == generation_pb2.FINISH_REASON_STOP
        if stops_on == "eos":
            assert finished.stop_match.eos_token_id == eos
        else:
            assert finished.stop_match.stop_token_id == PROMPT[0]
        assert responses[-1].usage.completion_tokens == 0
        assert engine.handoffs == [] and counters[COUNTER_PREFILL_READY_SENT] == 0

    run(main())


def test_prompt_logprobs_come_first_and_trtllm_hands_off_the_first_logprobs() -> None:
    async def main() -> None:
        async with serving(FakeEngine(disagg_profile())) as (engine,):
            message = prefill(
                response={
                    "returnPromptLogprobs": True,
                    "promptCandidates": {"topN": 1},
                    "returnOutputLogprobs": True,
                    "outputCandidates": {"topN": 2},
                }
            )
            responses = await accepted(engine, message)
        assert events(responses) == ["prompt", "token", "prefill_ready"]
        values = attributes(handoff(responses))
        assert set(values) == HANDOFF_KEYS | {"first_gen_log_probs"}
        # Per position, [token_id, logprob, rank] of each candidate.
        first = PROMPT[0]
        assert values["first_gen_log_probs"] == [[[first, -0.5, 1.0], [first + 1, -1.5, 2.0]]]
        info = responses[1].token.tokens[0]
        assert (info.token_id, info.logprob) == (first, -0.5)

    run(main())


def test_ds_refuses_output_logprobs_on_a_prefill_call() -> None:
    async def main() -> None:
        async with serving(FakeEngine(disagg_profile("ds-like", "context"))) as (engine,):
            message = prefill(response={"returnOutputLogprobs": True})
            _, code, _ = await generate(engine, message)
        assert code == grpc.StatusCode.UNIMPLEMENTED

    run(main())


def test_a_prefill_call_aborted_before_its_handoff_ends_cancelled() -> None:
    async def main() -> None:
        engine = FakeEngine(disagg_profile(), step_delay=WAIT_SECONDS)
        async with serving(engine):
            message = prefill(requestId="aborted")
            call = asyncio.create_task(generate(engine, message))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + WAIT_SECONDS
            while engine.running_request_ids != ["aborted"]:
                assert loop.time() < deadline
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            assert engine.abort("aborted")
            responses, code, details = await call
            counters = engine.counters
        assert code == grpc.StatusCode.OK, details
        assert events(responses) == ["token", "finished"]
        assert responses[-1].finished.reason == generation_pb2.FINISH_REASON_CANCELLED
        assert responses[-1].usage.completion_tokens == 1
        assert engine.handoffs == [] and counters[COUNTER_PREFILL_READY_SENT] == 0

    run(main())


# -- The decode call --------------------------------------------------------------------


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_request_goes_through_both_engines_as_one_aggregated_call_would(shape: str) -> None:
    stopping = {"maxTokens": 6}

    async def main() -> None:
        context = FakeEngine(SHAPES[shape][0]())
        generation = FakeEngine(SHAPES[shape][1]())
        reference = FakeEngine(PROFILES[f"{shape}-like"])
        async with serving(context, generation, reference):
            first = prefill(
                requestId="ctx-1", extra={"disagg_request_id": CALLER_ID}, stopping=stopping
            )
            session = handoff(await accepted(context, first))
            second = request(requestId="gen-1", session=session, stopping=stopping)
            decoded = await accepted(generation, second)
            aggregated = await accepted(reference, request(stopping=stopping))
            counters = generation.counters
        assert events(decoded)[-1] == "finished"
        assert tokens(decoded) == tokens(aggregated) == [5, 6, 7, 8, 5, 6]
        assert decoded[-1].usage.completion_tokens == 6
        assert counters[COUNTER_GENERATE_ACCEPTED_BUCKETS[BUCKET_GENERATION_ONLY]] == 1
        assert counters[COUNTER_PREFILL_READY_SENT] == 0
        # The decode call's admitted settings start from the handoff.
        admitted = generation.accepted[0].admitted
        assert admitted.bucket == BUCKET_GENERATION_ONLY
        assert admitted.first_gen_tokens == (PROMPT[0],)
        assert admitted.kv_session == session
        # Both calls are recorded whole, and differ only in their leg-specific parts.
        legs = [context.requests[0], generation.requests[0]]
        for message in legs:
            for name in ("request_id", "extra", "kv"):
                message.ClearField(name)
        assert legs[0] == legs[1]

    run(main())


def test_a_decode_call_after_skipped_tokens_returns_the_aggregated_output() -> None:
    # Under min_tokens the first token skips EOS; the decode call walks the
    # prompt from where the prefill call's first token left it.
    prompt = [EOS, PROMPT[0], PROMPT[1]]
    stopping = {"maxTokens": 5, "minTokens": 2}

    async def main() -> None:
        context, generation = FakeEngine(TRTLLM), FakeEngine(TRTLLM)
        reference = FakeEngine(PROFILES["trtllm-like"])
        async with serving(context, generation, reference):
            session = handoff(await accepted(context, prefill(prompt=prompt, stopping=stopping)))
            decoded = await accepted(
                generation, request(prompt, session=session, stopping=stopping)
            )
            aggregated = await accepted(reference, request(prompt, stopping=stopping))
        assert tokens(decoded) == tokens(aggregated) == [PROMPT[0], PROMPT[1]]
        assert decoded[-1].finished.stop_match.eos_token_id == EOS

    run(main())


def test_the_server_reads_the_fakes_handoff() -> None:
    from tensorrt_llm.grpc.openengine.disagg import disaggregated_params_from_request

    async def main() -> tuple[kv_pb2.KvSessionRef, int]:
        async with serving(FakeEngine(disagg_profile())) as (engine,):
            responses = await accepted(engine, prefill(extra={"disagg_request_id": CALLER_ID}))
            return handoff(responses), engine.port

    session, port = run(main())
    params = disaggregated_params_from_request(request(session=session))
    assert params.request_type == "generation_only"
    assert params.ctx_request_id == params.disagg_request_id == int(CALLER_ID)
    assert params.ctx_info_endpoint == f"tcp://127.0.0.1:{port}"
    assert params.first_gen_tokens == [PROMPT[0]] and params.ctx_dp_rank == 0


@pytest.mark.parametrize(
    "host,url", [("10.0.0.7", "tcp://10.0.0.7:41000"), ("fd00::7", "tcp://[fd00::7]:41000")]
)
def test_the_handoff_names_the_configured_ctx_endpoint(host: str, url: str) -> None:
    connector = FakeKvConnector(ctx_info_endpoint=(host, 41000))

    async def main() -> kv_pb2.KvSessionRef:
        async with serving(FakeEngine(disagg_profile(kv_connector=connector))) as (engine,):
            return handoff(await accepted(engine, prefill()))

    session = run(main())
    assert attributes(session)["ctx_info_endpoint"] == url
    assert (session.endpoints[0].host, session.endpoints[0].port) == (host, 41000)


@pytest.mark.parametrize(
    "bound,named",
    [("0.0.0.0", "127.0.0.1"), ("::", "::1"), ("[::]", "::1"), ("[fd00::7]", "fd00::7")],
)
def test_an_engine_on_a_wildcard_address_names_loopback_as_its_ctx_endpoint(
    bound: str, named: str
) -> None:
    engine = FakeEngine(disagg_profile())
    engine.host = bound
    assert engine.ctx_info_endpoint[0] == named


def _session(**changes: Any) -> kv_pb2.KvSessionRef:
    """A handoff as a fake engine on port 17001 sends it, with ``changes`` to its parts."""
    values: dict[str, Any] = {
        "ctx_info_endpoint": "tcp://127.0.0.1:17001",
        "first_gen_tokens": [PROMPT[0]],
        "disagg_request_id": CALLER_ID,
        "ctx_dp_rank": 0,
    }
    session = kv_pb2.KvSessionRef(session_id=CALLER_ID, transfer_backend="DEFAULT")
    session.endpoints.add(host="127.0.0.1", port=17001, protocol="tcp")
    for name in ("session_id", "endpoints"):
        if name in changes:
            value = changes.pop(name)
            if name == "endpoints":
                del session.endpoints[:]
                for host, port in value:
                    session.endpoints.add(host=host, port=port, protocol="tcp")
            else:
                session.session_id = value
    for key, value in changes.items():
        if value is None:
            values.pop(key, None)
        else:
            values[key] = value
    session.attributes_struct.update(values)
    return session


# -- Refusals ---------------------------------------------------------------------------

TRTLLM = disagg_profile("trtllm-like")
DS_CONTEXT = disagg_profile("ds-like", "context")
DS_GENERATION = disagg_profile("ds-like", "generation")
CONTEXT_ONLY = {"request_type": "context_only"}
# (profile, request fields, extra, kv.session, status)
_REFUSALS: list[tuple[FakeProfile | str, dict[str, Any], dict[str, Any], Any, str]] = [
    # Two-stage calls need an enabled connector that takes remote prefill.
    ("trtllm-like", {}, CONTEXT_ONLY, None, "FAILED_PRECONDITION"),
    ("trtllm-like", {}, {}, _session(), "FAILED_PRECONDITION"),
    ("ds-like", {}, CONTEXT_ONLY, None, "FAILED_PRECONDITION"),
    (
        disagg_profile(kv_connector=FakeKvConnector(enabled=False)),
        {},
        CONTEXT_ONLY,
        None,
        "FAILED_PRECONDITION",
    ),
    (
        disagg_profile(kv_connector=FakeKvConnector(supports_remote_prefill=False)),
        {},
        {},
        _session(),
        "FAILED_PRECONDITION",
    ),
    # One output sequence per two-stage call.
    (
        TRTLLM,
        {"sampling": {"numSequences": 2, "temperature": 1.0}},
        CONTEXT_ONLY,
        None,
        "INVALID_ARGUMENT",
    ),
    (
        TRTLLM,
        {"sampling": {"numSequences": 2, "temperature": 1.0}},
        {},
        _session(),
        "INVALID_ARGUMENT",
    ),
    # request_type cannot make a decode call, and takes only its two values.
    (TRTLLM, {}, {"request_type": "generation_only"}, None, "INVALID_ARGUMENT"),
    ("trtllm-like", {}, {"request_type": "generation_only"}, None, "INVALID_ARGUMENT"),
    (TRTLLM, {}, {"request_type": "prefill"}, None, "INVALID_ARGUMENT"),
    (TRTLLM, {}, {"request_type": 1}, None, "INVALID_ARGUMENT"),
    # kv.session is read with the server's mapping, and its first tokens must be in the vocabulary.
    (TRTLLM, {}, {}, _session(ctx_info_endpoint=None, endpoints=[]), "INVALID_ARGUMENT"),
    (TRTLLM, {}, {}, _session(ctx_info_endpoint="tcp://127.0.0.1"), "INVALID_ARGUMENT"),
    (TRTLLM, {}, {}, _session(session_id="", disagg_request_id=None), "INVALID_ARGUMENT"),
    (TRTLLM, {}, {}, _session(session_id="not-a-number"), "INVALID_ARGUMENT"),
    (TRTLLM, {}, {}, _session(first_gen_tokens=[999999]), "INVALID_ARGUMENT"),
    (TRTLLM, {}, {}, _session(ctx_usage=5), "INVALID_ARGUMENT"),
    (TRTLLM, {}, {}, _session(first_gen_tokens=[float("inf")]), "INVALID_ARGUMENT"),
    # The ds decode call needs the endpoint and first tokens, and named attributes only.
    (DS_GENERATION, {}, {}, _session(first_gen_tokens=None), "INVALID_ARGUMENT"),
    (
        DS_GENERATION,
        {},
        {},
        _session(ctx_info_endpoint=None, endpoints=[], opaque_state="AAAA"),
        "INVALID_ARGUMENT",
    ),
    (DS_GENERATION, {}, {}, _session(ds_private=1), "INVALID_ARGUMENT"),
    # A ds instance takes only its own call of a two-stage request.
    (DS_GENERATION, {}, CONTEXT_ONLY, None, "INVALID_ARGUMENT"),
    (DS_CONTEXT, {}, {}, _session(), "INVALID_ARGUMENT"),
    # The disaggregated request keys: declared, on their own call, with valid values.
    (TRTLLM, {}, {**CONTEXT_ONLY, "schedule_style": 1}, None, "UNIMPLEMENTED"),
    (DS_GENERATION, {}, {"orig_elapsed_s": 0.5}, _session(), "UNIMPLEMENTED"),
    (TRTLLM, {}, {"disagg_request_id": CALLER_ID}, None, "INVALID_ARGUMENT"),
    (TRTLLM, {}, {"disagg_request_id": CALLER_ID}, _session(), "INVALID_ARGUMENT"),
    (TRTLLM, {}, {**CONTEXT_ONLY, "disagg_request_id": "1"}, None, "INVALID_ARGUMENT"),
    (TRTLLM, {}, {**CONTEXT_ONLY, "disagg_request_id": float(CALLER_ID)}, None, "INVALID_ARGUMENT"),
    (TRTLLM, {}, {**CONTEXT_ONLY, "orig_elapsed_s": 0.5}, None, "INVALID_ARGUMENT"),
    (TRTLLM, {}, {"orig_elapsed_s": -1}, _session(), "INVALID_ARGUMENT"),
    (TRTLLM, {}, {"orig_elapsed_s": "1"}, _session(), "INVALID_ARGUMENT"),
    # ds does not declare min_tokens, so it refuses it rather than ignore it.
    ("ds-like", {"stopping": {"maxTokens": 2, "minTokens": 1}}, {}, None, "INVALID_ARGUMENT"),
]


@pytest.mark.parametrize(
    "profile,fields,extra,session,status", _REFUSALS, ids=[str(i) for i in range(len(_REFUSALS))]
)
def test_refused_two_stage_calls_get_the_declared_status(
    profile: FakeProfile | str,
    fields: dict[str, Any],
    extra: dict[str, Any],
    session: kv_pb2.KvSessionRef | None,
    status: str,
) -> None:
    message = request(extra=extra, session=session, **{"stopping": {"maxTokens": 2}, **fields})

    async def main() -> None:
        async with serving(FakeEngine(profile)) as (engine,):
            responses, code, details = await generate(engine, message)
            assert (code.name, responses) == (status, []), details
            assert details
            counters = engine.counters
        assert counters[COUNTER_GENERATE_ACCEPTED] == 0
        assert not [key for key in COUNTER_GENERATE_ACCEPTED_BUCKETS.values() if counters.get(key)]

    run(main())


# -- Accepted variations -----------------------------------------------------------------


@pytest.mark.parametrize(
    "profile,extra,session",
    [
        # The decode call keys of each engine.
        (TRTLLM, {"orig_elapsed_s": 0.25}, _session()),
        (TRTLLM, {"orig_elapsed_s": 0}, _session()),
        # The server takes an opaque_state in place of an endpoint, and ignores request_type.
        (
            TRTLLM,
            {"request_type": "context_only"},
            _session(ctx_info_endpoint=None, endpoints=[], opaque_state="AAAA"),
        ),
        # ds accepts the mapped keys it does not use, and the registered passthrough key.
        (DS_GENERATION, {}, _session(ctx_usage={"prompt_tokens": 4}, schedule_style=0)),
        (DS_GENERATION, {}, _session(trtllm_decode_preparation="opaque")),
        # The TensorRT-LLM server passes on attributes it does not know.
        (TRTLLM, {}, _session(engine_private={"a": 1})),
        # An aggregated call naming the other request type.
        (TRTLLM, {"request_type": "context_and_generation"}, None),
    ],
    ids=[
        "orig-elapsed",
        "orig-elapsed-zero",
        "opaque-state",
        "ds-unused-mapped-keys",
        "ds-registered-passthrough",
        "trtllm-unknown-attribute",
        "context-and-generation",
    ],
)
def test_accepted_two_stage_variations(
    profile: FakeProfile, extra: dict[str, Any], session: kv_pb2.KvSessionRef | None
) -> None:
    message = request(extra=extra, session=session, stopping={"maxTokens": 2})

    async def main() -> None:
        async with serving(FakeEngine(profile)) as (engine,):
            responses = await accepted(engine, message)
        assert events(responses)[-1] == "finished"
        bucket = BUCKET_AGGREGATED if session is None else BUCKET_GENERATION_ONLY
        assert engine.accepted[0].admitted.bucket == bucket

    run(main())


def test_glm_like_ignores_request_type_and_refuses_kv_session() -> None:
    async def main() -> None:
        async with serving(FakeEngine("glm-like")) as (engine,):
            responses = await accepted(engine, prefill(stopping={"maxTokens": 2}))
            _, code, _ = await generate(engine, request(session=_session()))
        assert events(responses) == ["token", "token", "finished"]
        assert code == grpc.StatusCode.UNIMPLEMENTED

    run(main())


def test_a_scripted_prefill_ready_ends_the_call_and_is_counted() -> None:
    session = _session()
    script = [
        generation_pb2.GenerateResponse(
            token=generation_pb2.TokenOutput(
                output_index=0, tokens=[generation_pb2.TokenInfo(token_id=5)]
            )
        ),
        generation_pb2.GenerateResponse(
            prefill_ready=generation_pb2.PrefillReady(kv_session=session),
            usage=generation_pb2.Usage(prompt_tokens=4, completion_tokens=1, total_tokens=5),
        ),
        Hang(),
    ]

    async def main() -> None:
        async with serving(FakeEngine(TRTLLM, script=script)) as (engine,):
            responses = await accepted(engine, prefill())
            counters = engine.counters
        assert events(responses) == ["token", "prefill_ready"]
        assert counters[COUNTER_PREFILL_READY_SENT] == 1

    run(main())


# -- Injected faults --------------------------------------------------------------------


def test_a_prefill_call_can_end_on_eos_without_a_handoff() -> None:
    async def main() -> None:
        engine = FakeEngine(TRTLLM)
        engine.disagg_faults = DisaggFaults(finish_prefill_early=True)
        async with serving(engine):
            responses = await accepted(engine, prefill())
            counters = engine.counters
        assert events(responses) == ["finished"]
        finished = responses[0].finished
        assert finished.reason == generation_pb2.FINISH_REASON_STOP
        assert finished.stop_match.eos_token_id == EOS
        usage = responses[0].usage
        assert (usage.prompt_tokens, usage.completion_tokens) == (len(PROMPT), 0)
        assert counters[COUNTER_PREFILL_READY_SENT] == 0 and engine.handoffs == []
        assert counters[COUNTER_GENERATE_ACCEPTED_BUCKETS[BUCKET_CONTEXT_ONLY]] == 1

    run(main())


@pytest.mark.parametrize(
    "key", ["trtllm_decode_preparation", "engine_private"], ids=["registered", "unregistered"]
)
def test_a_handoff_can_carry_an_extra_attribute(key: str) -> None:
    async def main() -> kv_pb2.KvSessionRef:
        engine = FakeEngine(TRTLLM)
        engine.disagg_faults = DisaggFaults(extra_attributes={key: "value"})
        async with serving(engine):
            return handoff(await accepted(engine, prefill()))

    values = attributes(run(main()))
    assert values[key] == "value" and set(values) == HANDOFF_KEYS | {key}


@pytest.mark.parametrize("bucket", [BUCKET_CONTEXT_ONLY, BUCKET_GENERATION_ONLY, BUCKET_AGGREGATED])
def test_a_call_can_end_before_its_terminal_event(bucket: str) -> None:
    message = {
        BUCKET_CONTEXT_ONLY: prefill(),
        BUCKET_GENERATION_ONLY: request(session=_session(), stopping={"maxTokens": 4}),
        BUCKET_AGGREGATED: request(stopping={"maxTokens": 4}),
    }[bucket]

    async def main() -> None:
        engine = FakeEngine(TRTLLM)
        engine.disagg_faults = DisaggFaults(close_before_terminal=frozenset({bucket}))
        async with serving(engine):
            responses = await accepted(engine, message)
            # Other buckets are not affected.
            other = (
                prefill() if bucket != BUCKET_CONTEXT_ONLY else request(stopping={"maxTokens": 1})
            )
            complete = await accepted(engine, other)
        assert events(responses) == ["token"]
        assert tokens(responses) == PROMPT[:1]
        assert not responses[0].HasField("usage")
        assert events(complete)[-1] in ("prefill_ready", "finished")

    run(main())


@pytest.mark.parametrize("bucket", [BUCKET_GENERATION_ONLY, BUCKET_AGGREGATED])
def test_any_call_can_be_made_to_send_a_handoff(bucket: str) -> None:
    session = _session() if bucket == BUCKET_GENERATION_ONLY else None

    async def main() -> None:
        engine = FakeEngine(TRTLLM)
        engine.disagg_faults = DisaggFaults(prefill_ready_on_any_call=True)
        async with serving(engine):
            responses = await accepted(engine, request(session=session, stopping={"maxTokens": 4}))
            counters = engine.counters
        assert events(responses) == ["token", "prefill_ready"]
        assert counters[COUNTER_PREFILL_READY_SENT] == 1
        assert counters[COUNTER_GENERATE_ACCEPTED_BUCKETS[bucket]] == 1

    run(main())


# -- Metadata ----------------------------------------------------------------------------


def _affinity_request(request_id: str = "child-1") -> generation_pb2.GenerateRequest:
    return prefill(requestId=request_id, extra={"conversation_id": "child"})


def test_signed_subagent_affinity_is_checked_and_recorded() -> None:
    profile = disagg_profile(subagent_affinity_key=AFFINITY_KEY)
    message = _affinity_request()
    metadata = (
        (AFFINITY_ID_HEADER, "parent"),
        (AFFINITY_AUTH_HEADER, sign(message, "parent")),
        ("traceparent", "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"),
    )

    async def main() -> None:
        async with serving(FakeEngine(profile)) as (engine,):
            info, _, _, _ = await _declarations(engine)
            await accepted(engine, message, metadata)
        assert json_format.MessageToDict(info.extra)["trtllm_supports_subagent_affinity"] is True
        record = engine.received[0]
        assert record.admitted.subagent_affinity_id == "parent"
        assert record.metadata_value(AFFINITY_ID_HEADER) == "parent"
        assert record.metadata_value("traceparent") == metadata[2][1]

    run(main())


@pytest.mark.parametrize(
    "profile,message,metadata",
    [
        # A wrong signature, a key the engine lacks, a call other than a prefill call.
        (
            disagg_profile(subagent_affinity_key=AFFINITY_KEY),
            _affinity_request(),
            ((AFFINITY_ID_HEADER, "parent"), (AFFINITY_AUTH_HEADER, "sha256=0")),
        ),
        (
            TRTLLM,
            _affinity_request(),
            (
                (AFFINITY_ID_HEADER, "parent"),
                (AFFINITY_AUTH_HEADER, sign(_affinity_request(), "parent")),
            ),
        ),
        (
            disagg_profile(subagent_affinity_key=AFFINITY_KEY),
            request(requestId="child-1", extra={"conversation_id": "child"}),
            (
                (AFFINITY_ID_HEADER, "parent"),
                (AFFINITY_AUTH_HEADER, sign(_affinity_request(), "parent")),
            ),
        ),
        # A header sent twice.
        (
            disagg_profile(subagent_affinity_key=AFFINITY_KEY),
            _affinity_request(),
            (
                (AFFINITY_ID_HEADER, "parent"),
                (AFFINITY_ID_HEADER, "parent"),
                (AFFINITY_AUTH_HEADER, sign(_affinity_request(), "parent")),
            ),
        ),
    ],
    ids=["bad-signature", "no-key", "aggregated-call", "repeated-header"],
)
def test_bad_subagent_affinity_is_invalid(
    profile: FakeProfile, message: generation_pb2.GenerateRequest, metadata: Any
) -> None:
    async def main() -> None:
        async with serving(FakeEngine(profile)) as (engine,):
            _, code, _ = await generate(engine, message, metadata)
        assert code == grpc.StatusCode.INVALID_ARGUMENT

    run(main())


def test_command_line_profile_takes_a_kv_connector() -> None:
    import fake_openengine

    profile_json = json.dumps(
        {
            "kv_connector": {"transfer_backend": "UCX", "ctx_info_endpoint": ["10.0.0.7", 41000]},
            "instance_role": "context",
            "engine_role": server_pb2.ENGINE_ROLE_PREFILL,
        }
    )
    args = fake_openengine._parse_args(["--profile", "ds-like", "--profile-json", profile_json])
    profile = fake_openengine._profile_from_args(args)
    assert profile.kv_connector == FakeKvConnector(
        transfer_backend="UCX", ctx_info_endpoint=("10.0.0.7", 41000)
    )
    assert (profile.instance_role, profile.handoff_shape) == ("context", "ds")


# -- Fidelity ----------------------------------------------------------------------------

# What was read off the TensorRT-LLM OpenEngine server serving disaggregated.
SERVER_READINGS = (
    Path(__file__).resolve().parents[3]
    / "integration/defs/openengine_chain/records/WT-32/readings.json"
)


def test_the_fake_matches_what_was_read_off_the_server() -> None:
    readings = json.loads(SERVER_READINGS.read_text())["m8"]

    async def main() -> tuple[server_pb2.ServerInfo, dict[str, list[str]], kv_pb2.KvSessionRef]:
        async with serving(FakeEngine(disagg_profile())) as (engine,):
            info, _, _, _ = await _declarations(engine)
            kinds = {}
            for name, stopping in (
                ("first_token_is_stop", {"maxTokens": 1, "conditions": [{"stopTokenId": 5}]}),
                ("ctx_max_tokens_4", {"maxTokens": 4}),
                ("ctx_ignore_eos", {"maxTokens": 1, "ignoreEos": True}),
            ):
                kinds[name] = events(await accepted(engine, prefill(stopping=stopping)))
            message = prefill(response={"returnPromptLogprobs": True})
            kinds["prompt_logprobs"] = events(await accepted(engine, message))
            session = handoff(await accepted(engine, prefill()))
        return info, kinds, session

    info, kinds, session = run(main())
    assert (
        json_format.MessageToDict(info.kv_connector, preserving_proto_field_name=True)
        == (readings["a_server_info_kv_connector"])
    )
    for name, terminal in readings["b_ctx_terminals"].items():
        assert kinds[name][-1] == terminal, name
    assert kinds["prompt_logprobs"] == readings["d_prompt_logprobs"]["ctx_kinds"]
    expected = readings["a_prefill_ready_session"]
    actual = json_format.MessageToDict(session, preserving_proto_field_name=True)
    assert set(actual) == set(expected)
    assert actual["transfer_backend"] == expected["transfer_backend"]
    assert [set(endpoint) for endpoint in actual["endpoints"]] == [
        set(endpoint) for endpoint in expected["endpoints"]
    ]
    assert actual["endpoints"][0]["protocol"] == expected["endpoints"][0]["protocol"]
    values, read = actual["attributes_struct"], expected["attributes_struct"]
    assert {key: type(value) for key, value in values.items()} == {
        key: type(value) for key, value in read.items()
    }
    assert actual["session_id"] == values["disagg_request_id"]
    assert expected["session_id"] == read["disagg_request_id"]

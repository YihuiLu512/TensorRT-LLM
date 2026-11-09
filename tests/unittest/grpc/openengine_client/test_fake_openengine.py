# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CPU fake OpenEngine engine, driven through the OpenEngine client library and gRPC stubs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
from collections.abc import AsyncIterator, Sequence
from typing import Any

import grpc
import pytest
from fake_openengine import (
    CAP_BEST_OF,
    CAP_CONVERSATION_ID,
    CAP_FREQUENCY_PENALTY,
    CAP_KV_HINTS,
    CAP_LOGIT_BIAS,
    CAP_MIN_P,
    CAP_PRESENCE_PENALTY,
    CAP_REPETITION_PENALTY,
    CAP_SCHEDULING,
    CAP_SEED_SCOPE,
    CAP_STOP_TEXT,
    CAP_THINKING_TOKEN_BUDGET,
    CONVENTIONS_VERSION,
    COUNTER_GENERATE_ACCEPTED,
    COUNTER_GENERATE_RECEIVED,
    FP_BOS_TOKEN_ID,
    FP_EOS_TOKEN_IDS,
    FP_FILES_SHA256,
    FP_VOCAB_SIZE,
    KEY_CONVENTIONS_VERSION,
    KV_HINTS_TEXT_MAX_BYTES,
    PROFILES,
    Delay,
    Disconnect,
    Fail,
    FakeEngine,
    FakeProfile,
    Hang,
    Rejection,
    admit,
    running_in_thread,
)
from google.protobuf import json_format
from grpc_health.v1 import health_pb2, health_pb2_grpc

from tensorrt_llm.grpc.openengine.bindings import (
    error_pb2,
    generation_pb2,
    kv_pb2,
    lifecycle_pb2,
    lora_pb2,
    openengine_pb2_grpc,
    server_pb2,
)
from tensorrt_llm.grpc.openengine_client import (
    AbortOutcome,
    OpenEngineClient,
    OpenEngineEngineError,
    OpenEngineRpcError,
)
from tensorrt_llm.grpc.openengine_conventions import (
    SAMPLING_PARAM_NAMES,
    AcceptedValues,
    parse_params,
)

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

WAIT_SECONDS = 10.0
PROMPT = [5, 6, 7]
EOS = PROFILES["trtllm-like"].eos_token_ids[0]
# A TensorRT-LLM engine that does not follow the trtllm.* conventions.
NO_CONVENTIONS = PROFILES["trtllm-like"].replace(
    name="no-conventions", follows_conventions=False, max_num_sequences=None
)
# The TensorRT-LLM server under one-model speculative decoding: min_p and the
# penalties only at the values that leave sampling unchanged.
SPECULATIVE = PROFILES["trtllm-like"].replace(
    name="speculative",
    params={
        name: AcceptedValues("values", values=(no_effect,), default=no_effect)
        for name, no_effect in (
            ("min_p", 0.0),
            ("frequency_penalty", 0.0),
            ("presence_penalty", 0.0),
            ("repetition_penalty", 1.0),
        )
    },
)
# A KV hints envelope that the protocol's admission keeps.
KV_HINTS = {
    "protocol_version": "0.1",
    "message_id": "m-1",
    "actions": [
        {"action_id": "a-1", "action_type": "kv.fetch", "action_version": "1", "payload": {}}
    ],
}
CAPABILITY_KEYS = (
    CAP_MIN_P,
    CAP_FREQUENCY_PENALTY,
    CAP_PRESENCE_PENALTY,
    CAP_REPETITION_PENALTY,
    CAP_STOP_TEXT,
    CAP_CONVERSATION_ID,
    CAP_THINKING_TOKEN_BUDGET,
    CAP_LOGIT_BIAS,
    CAP_KV_HINTS,
    CAP_SCHEDULING,
    CAP_BEST_OF,
)


def run(coroutine: Any) -> Any:
    async def bounded() -> Any:
        return await asyncio.wait_for(coroutine, WAIT_SECONDS * 3)

    return asyncio.run(bounded())


@contextlib.asynccontextmanager
async def connected(engine: FakeEngine) -> AsyncIterator[OpenEngineClient]:
    async with engine:
        client = await OpenEngineClient.connect(engine.target, ready_timeout=WAIT_SECONDS)
        try:
            yield client
        finally:
            await client.close()


def request(
    prompt: Sequence[int] = PROMPT, *, extra: dict[str, Any] | None = None, **fields: Any
) -> generation_pb2.GenerateRequest:
    """A token-id request; ``fields`` are ``GenerateRequest`` fields as dicts."""
    message = generation_pb2.GenerateRequest(
        model="fake-model", token_ids=generation_pb2.TokenIds(ids=list(prompt))
    )
    parse_into(fields, message)
    if extra is not None:
        message.extra.update(extra)
    return message


def parse_into(fields: dict[str, Any], message: generation_pb2.GenerateRequest) -> None:
    """Merge ``fields`` (JSON names) into ``message``; ``tokenIds`` replaces the prompt."""
    if "tokenIds" in fields:
        message.ClearField("token_ids")
    json_format.ParseDict(fields, message)


async def collect(
    client: OpenEngineClient, message: generation_pb2.GenerateRequest
) -> list[generation_pb2.GenerateResponse]:
    async with client.generate(message) as stream:
        return [response async for response in stream]


async def raw_generate(
    engine: FakeEngine,
    message: generation_pb2.GenerateRequest,
    metadata: Sequence[tuple[str, str]] = (),
) -> tuple[list[generation_pb2.GenerateResponse], grpc.StatusCode, str]:
    """Generate on a plain stub: every response, the final status and its details."""
    if not message.request_id:
        message.request_id = f"raw-{len(engine.received)}"
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


def tokens_by_output(responses: Sequence[generation_pb2.GenerateResponse]) -> dict[int, list[int]]:
    outputs: dict[int, list[int]] = {}
    for response in responses:
        if response.WhichOneof("event") == "token":
            outputs.setdefault(response.token.output_index, []).extend(
                info.token_id for info in response.token.tokens
            )
    return outputs


def finished_by_output(
    responses: Sequence[generation_pb2.GenerateResponse],
) -> dict[int, generation_pb2.GenerationFinished]:
    return {
        response.finished.output_index: response.finished
        for response in responses
        if response.WhichOneof("event") == "finished"
    }


def check_stream(
    message: generation_pb2.GenerateRequest, responses: Sequence[generation_pb2.GenerateResponse]
) -> None:
    """Assert the OpenEngine stream rules on a finished stream."""
    num_outputs = message.sampling.num_sequences or 1
    wants_logprobs = message.response.return_output_logprobs
    eos = () if message.stopping.ignore_eos else PROFILES["trtllm-like"].eos_token_ids
    top_n = message.response.output_candidates.top_n
    finished: set[int] = set()
    assert responses, "an accepted request must produce events"
    for position, response in enumerate(responses):
        last = position == len(responses) - 1
        assert response.request_id == message.request_id
        assert response.HasField("usage") == last, "usage must be on the last response only"
        event = response.WhichOneof("event")
        assert event in ("prompt", "token", "finished"), event
        if event == "prompt":
            assert position == 0, "the prompt event must come first"
        elif event == "token":
            output = response.token
            assert output.HasField("output_index")
            assert output.output_index not in finished, "a token after its output finished"
            assert output.tokens
            for info in output.tokens:
                assert info.HasField("logprob") == bool(wants_logprobs)
                assert len(info.candidates) <= top_n
                assert info.token_id not in eos, "EOS must not reach the output"
        else:
            event_message = response.finished
            assert event_message.HasField("output_index")
            assert event_message.output_index not in finished
            assert event_message.reason != generation_pb2.FINISH_REASON_UNSPECIFIED
            is_stop = event_message.reason == generation_pb2.FINISH_REASON_STOP
            assert event_message.HasField("stop_match") == is_stop
            finished.add(event_message.output_index)
    assert finished == set(range(num_outputs)), "every output must finish exactly once"
    usage = responses[-1].usage
    assert usage.prompt_tokens == len(message.token_ids.ids)
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.HasField("cached_prompt_tokens")


# -- Declared capabilities --------------------------------------------------------------


@pytest.mark.parametrize(
    "profile",
    [*(PROFILES[name] for name in sorted(PROFILES)), NO_CONVENTIONS],
    ids=lambda p: p.name,
)
def test_profile_declares_every_capability_explicitly(profile: FakeProfile) -> None:
    async def main() -> None:
        async with connected(FakeEngine(profile)) as client:
            info = client.model_info
            server = client.server_info
        for field in (
            "max_context_length",
            "supports_text_input",
            "supports_token_ids_input",
            "supports_lora",
            "supports_multimodal",
        ):
            assert info.HasField(field), field
        generation = info.generation
        for logprobs in (generation.output_logprobs, generation.prompt_logprobs):
            assert logprobs.HasField("supported") and logprobs.HasField("max_top_n")
        assert generation.guided_decoding.HasField("supported")
        for field in ("supports_priority", "supports_stop_in_output", "supports_cache_salt"):
            assert generation.HasField(field), field
        assert info.supports_token_ids_input
        assert server.engine_name == profile.engine_name
        assert server.instance_id and server.capacity.HasField("max_running_requests")
        extra = json_format.MessageToDict(info.extra)
        if profile.follows_conventions:
            assert json_format.MessageToDict(server.extra)[KEY_CONVENTIONS_VERSION] == (
                CONVENTIONS_VERSION
            )
            for key in CAPABILITY_KEYS:
                assert isinstance(extra[key], bool), key
            assert set(parse_params(extra)) == set(SAMPLING_PARAM_NAMES)
            assert extra[CAP_SEED_SCOPE] in ("none", "batch_dependent", "per_request")
            for key in (FP_VOCAB_SIZE, FP_EOS_TOKEN_IDS, FP_BOS_TOKEN_ID, FP_FILES_SHA256):
                assert key in extra, key
        else:
            assert KEY_CONVENTIONS_VERSION not in json_format.MessageToDict(server.extra)
            assert not [key for key in extra if key.startswith("trtllm.")]

    run(main())


def test_profiles_report_the_planned_engine_capabilities() -> None:
    async def model_info(name: str) -> Any:
        async with connected(FakeEngine(name)) as client:
            return client.model_info

    trtllm = run(model_info("trtllm-like"))
    glm = run(model_info("glm-like"))
    for info, text, top_n, prompt_logprobs, sequences, cache_salt in (
        (trtllm, True, 100, True, 16, True),
        (glm, False, 20, False, 1, False),
    ):
        assert info.supports_text_input is text
        assert info.generation.output_logprobs.max_top_n == top_n
        assert info.generation.prompt_logprobs.supported is prompt_logprobs
        assert info.generation.max_num_sequences == sequences
        assert info.generation.supports_cache_salt is cache_salt
        assert info.generation.supports_stop_in_output
        assert not info.supports_lora and not info.supports_multimodal
    assert len(trtllm.generation.guided_decoding.modes) == 5
    assert not glm.generation.guided_decoding.supported
    trtllm_extra = json_format.MessageToDict(trtllm.extra)
    glm_extra = json_format.MessageToDict(glm.extra)
    assert all(trtllm_extra[key] for key in CAPABILITY_KEYS if key != CAP_BEST_OF)
    assert not trtllm_extra[CAP_BEST_OF]
    assert glm_extra[CAP_MIN_P] and glm_extra[CAP_STOP_TEXT]
    assert not any(
        glm_extra[key] for key in CAPABILITY_KEYS if key not in (CAP_MIN_P, CAP_STOP_TEXT)
    )
    assert trtllm_extra[CAP_SEED_SCOPE] == glm_extra[CAP_SEED_SCOPE] == "batch_dependent"
    assert parse_params(trtllm_extra) == PROFILES["trtllm-like"].params
    glm_params = parse_params(glm_extra)
    for name, no_effect in (
        ("frequency_penalty", 0.0),
        ("presence_penalty", 0.0),
        ("repetition_penalty", 1.0),
    ):
        assert glm_params[name].accepts(no_effect) and not glm_params[name].accepts(0.5), name
    assert glm_params["top_p"].accepts(1.0) and not glm_params["top_p"].accepts(0.0)
    # min_p's range is all the protocol allows, so its version 2 key is true (asserted above).
    assert glm_params["min_p"].accept == "range"


def test_a_sampling_key_in_capabilities_overrides_the_declared_values() -> None:
    async def model_extra(profile: FakeProfile) -> dict[str, Any]:
        async with connected(FakeEngine(profile)) as client:
            return json_format.MessageToDict(client.model_info.extra)

    extra = run(
        model_extra(
            PROFILES["glm-like"].replace(
                capabilities={CAP_PRESENCE_PENALTY: True, CAP_MIN_P: False}
            )
        )
    )
    params = parse_params(extra)
    assert params["presence_penalty"].accept == "any" and extra[CAP_PRESENCE_PENALTY] is True
    assert params["min_p"].accept == "none" and extra[CAP_MIN_P] is False
    # Narrowed by params, min_p's version 2 key turns false; the others keep theirs.
    narrowed = AcceptedValues("values", values=(0.0,), default=0.0)
    extra = run(model_extra(PROFILES["trtllm-like"].replace(params={"min_p": narrowed})))
    assert extra[CAP_MIN_P] is False and extra[CAP_REPETITION_PENALTY] is True
    assert parse_params(extra)["min_p"].values == (0.0,)
    # A range short of the protocol's bounds by an excluded end is not every value.
    for open_end in (
        AcceptedValues("range", min=0.0, max=1.0, max_exclusive=True),
        AcceptedValues("range", min=0.0, min_exclusive=True, max=1.0),
    ):
        extra = run(model_extra(PROFILES["trtllm-like"].replace(params={"min_p": open_end})))
        assert extra[CAP_MIN_P] is False, open_end
    # An override keeps the declared default.
    overridden = PROFILES["trtllm-like"].replace(capabilities={CAP_REPETITION_PENALTY: False})
    assert overridden.accepted("repetition_penalty").default == 1.0


def test_trtllm_like_declares_the_servers_sampling_values() -> None:
    from types import SimpleNamespace

    from tensorrt_llm.grpc.openengine.request_mapping import accepted_sampling_values

    # A server without speculative decoding and with the default generation_config.
    llm_args = SimpleNamespace(
        backend="pytorch", generation_config="trtllm", speculative_config=None
    )
    assert dict(PROFILES["trtllm-like"].params) == accepted_sampling_values(llm_args)
    # The same server under a one-model speculative mode that applies top_k and top_p.
    one_model = SimpleNamespace(
        decoding_type="MTP",
        spec_dec_mode=SimpleNamespace(use_one_engine=lambda: True),
        advanced_sampling_mode=None,
    )
    llm_args.speculative_config = one_model
    assert dict(SPECULATIVE.params) == accepted_sampling_values(llm_args)


def test_profile_without_conventions_has_no_sequence_limit_and_no_counters() -> None:
    engine = FakeEngine(NO_CONVENTIONS)

    async def main() -> None:
        async with connected(engine) as client:
            assert not client.model_info.generation.HasField("max_num_sequences")
            await collect(client, request(stopping={"maxTokens": 1}))
            load = await client.get_load()
            async with grpc.aio.insecure_channel(engine.target) as channel:
                sources = await openengine_pb2_grpc.ControlStub(channel).GetKvEventSources(
                    kv_pb2.GetKvEventSourcesRequest(data_parallel_ranks=[1])
                )
        assert load is not None and load.running_requests == 0
        assert not load.attributes.fields
        # KV events are off, so there are no sources for any rank.
        assert not sources.sources

    run(main())


def test_profile_mappings_are_read_only() -> None:
    profile = PROFILES["trtllm-like"]
    with pytest.raises(TypeError):
        profile.capabilities[CAP_MIN_P] = False  # type: ignore[index]
    changed = profile.replace(capabilities={CAP_MIN_P: False})
    assert changed.capability(CAP_MIN_P) is False and profile.capability(CAP_MIN_P) is True
    assert len({profile, changed, PROFILES["glm-like"]}) == 3


def test_a_new_engine_reports_a_new_instance_id() -> None:
    assert FakeEngine().instance_id != FakeEngine().instance_id


# -- The client library against the fake ---------------------------------------------------


def test_client_drives_every_rpc() -> None:
    profile = PROFILES["trtllm-like"].replace(
        supports_lora=True,
        max_loras=2,
        kv_event_sources=(kv_pb2.KvEventSource(transport="zmq", topic="kv", data_parallel_rank=0),),
    )
    engine = FakeEngine(profile, step_delay=0.01)

    async def main() -> None:
        async with connected(engine) as client:
            await client.refresh()
            assert (await client.health()).state == lifecycle_pb2.HEALTH_STATE_READY
            responses = await collect(client, request(stopping={"maxTokens": 4}))
            assert tokens_by_output(responses) == {0: [5, 6, 7, 5]}

            running = request(requestId="running", stopping={"maxTokens": 1000})
            async with client.generate(running) as stream:
                first = await stream.__anext__()
                assert first.WhichOneof("event") == "token"
                assert (await client.get_load()).running_requests == 1
                assert await client.abort(running.request_id) is AbortOutcome.ABORTED
                rest = [response async for response in stream]
            assert rest[-1].finished.reason == generation_pb2.FINISH_REASON_CANCELLED
            assert await client.abort(running.request_id) is AbortOutcome.ALREADY_FINISHED
            all_response = await client.abort_all()
            assert all_response.status == lifecycle_pb2.ABORT_STATUS_ALREADY_FINISHED
            with pytest.raises(OpenEngineRpcError) as error:
                await client.abort_kv_session(kv_pb2.KvSessionRef(session_id="s"))
            assert error.value.code == grpc.StatusCode.UNIMPLEMENTED

            async with grpc.aio.insecure_channel(engine.target) as channel:
                control = openengine_pb2_grpc.ControlStub(channel)
                adapter = lora_pb2.LoraAdapter(lora_id=7, lora_name="a", source_path="/l/a")
                loaded = await control.LoadLora(lora_pb2.LoadLoraRequest(adapter=adapter))
                assert not loaded.already_loaded
                again = await control.LoadLora(lora_pb2.LoadLoraRequest(adapter=adapter))
                assert again.already_loaded
                listed = await control.ListLoras(lora_pb2.ListLorasRequest())
                assert [a.lora_name for a in listed.adapters] == ["a"]
                lora_request = request(stopping={"maxTokens": 1}, loraName="a")
                assert tokens_by_output(await collect(client, lora_request)) == {0: [5]}
                await control.UnloadLora(lora_pb2.UnloadLoraRequest(lora_name="a"))
                with pytest.raises(grpc.aio.AioRpcError) as unload:
                    await control.UnloadLora(lora_pb2.UnloadLoraRequest(lora_name="a"))
                assert unload.value.code() == grpc.StatusCode.NOT_FOUND
                sources = await control.GetKvEventSources(kv_pb2.GetKvEventSourcesRequest())
                assert [source.topic for source in sources.sources] == ["kv"]
                with pytest.raises(grpc.aio.AioRpcError) as rank:
                    await control.GetKvEventSources(
                        kv_pb2.GetKvEventSourcesRequest(data_parallel_ranks=[1])
                    )
                assert rank.value.code() == grpc.StatusCode.INVALID_ARGUMENT
                with pytest.raises(grpc.aio.AioRpcError) as subscribe:
                    async for _ in control.SubscribeKvEvents(kv_pb2.SubscribeKvEventsRequest()):
                        pass
                assert subscribe.value.code() == grpc.StatusCode.UNIMPLEMENTED
        assert engine.aborted == [running.request_id]
        # Abort is not a client cancellation.
        assert engine.cancelled == [] and not engine.cancel_seen.is_set()

    run(main())


def test_load_lora_refuses_a_name_or_id_of_another_adapter() -> None:
    profile = PROFILES["trtllm-like"].replace(supports_lora=True)

    def adapter(name: str, lora_id: int, path: str) -> lora_pb2.LoadLoraRequest:
        return lora_pb2.LoadLoraRequest(
            adapter=lora_pb2.LoraAdapter(lora_name=name, lora_id=lora_id, source_path=path)
        )

    async def main() -> None:
        async with FakeEngine(profile) as engine:
            async with grpc.aio.insecure_channel(engine.target) as channel:
                control = openengine_pb2_grpc.ControlStub(channel)

                async def status(load: lora_pb2.LoadLoraRequest) -> grpc.StatusCode:
                    try:
                        await control.LoadLora(load)
                    except grpc.aio.AioRpcError as error:
                        return error.code()
                    return grpc.StatusCode.OK

                assert await status(adapter("a", 7, "/l/a")) == grpc.StatusCode.OK
                conflict = grpc.StatusCode.ALREADY_EXISTS
                assert await status(adapter("a", 7, "/l/x")) == conflict
                assert await status(adapter("a", 8, "/l/a")) == conflict
                assert await status(adapter("b", 7, "/l/b")) == conflict
                # The same id and path under another name: the id belongs to "a".
                assert await status(adapter("b", 7, "/l/a")) == conflict
                await control.UnloadLora(lora_pb2.UnloadLoraRequest(lora_name="a"))
                # The id stays bound to its first path after the unload.
                assert await status(adapter("b", 7, "/l/b")) == conflict
                assert await status(adapter("b", 7, "/l/a")) == grpc.StatusCode.OK
                invalid = grpc.StatusCode.INVALID_ARGUMENT
                assert await status(adapter("c", -1, "/l/c")) == invalid

    run(main())


def test_load_lora_does_not_limit_adapters_by_max_loras() -> None:
    profile = PROFILES["trtllm-like"].replace(supports_lora=True, max_loras=1)

    async def main() -> None:
        async with FakeEngine(profile) as engine:
            async with grpc.aio.insecure_channel(engine.target) as channel:
                control = openengine_pb2_grpc.ControlStub(channel)
                for lora_id in (1, 2, 3):
                    adapter = lora_pb2.LoraAdapter(
                        lora_name=f"a{lora_id}", lora_id=lora_id, source_path=f"/l/{lora_id}"
                    )
                    await control.LoadLora(lora_pb2.LoadLoraRequest(adapter=adapter))
                listed = await control.ListLoras(lora_pb2.ListLorasRequest())
                assert len(listed.adapters) == 3

    run(main())


def test_load_lora_counts_every_id_it_has_loaded_against_max_registered_loras() -> None:
    profile = PROFILES["trtllm-like"].replace(
        supports_lora=True, max_loras=1, max_registered_loras=2
    )

    async def main() -> None:
        async with FakeEngine(profile) as engine:
            async with grpc.aio.insecure_channel(engine.target) as channel:
                control = openengine_pb2_grpc.ControlStub(channel)

                async def load(name: str, lora_id: int) -> grpc.StatusCode:
                    adapter = lora_pb2.LoraAdapter(
                        lora_name=name, lora_id=lora_id, source_path=f"/l/{lora_id}"
                    )
                    try:
                        await control.LoadLora(lora_pb2.LoadLoraRequest(adapter=adapter))
                    except grpc.aio.AioRpcError as error:
                        return error.code()
                    return grpc.StatusCode.OK

                async def unload(name: str) -> None:
                    await control.UnloadLora(lora_pb2.UnloadLoraRequest(lora_name=name))

                assert await load("a", 1) == grpc.StatusCode.OK
                # max_loras is only reported: a second adapter is accepted.
                assert await load("b", 2) == grpc.StatusCode.OK
                await unload("a")
                await unload("b")
                # Both ids count although neither adapter is loaded.
                assert await load("c", 3) == grpc.StatusCode.RESOURCE_EXHAUSTED
                # An id loaded before does not count again.
                assert await load("c", 1) == grpc.StatusCode.OK

    run(main())


def test_lora_rpcs_are_unimplemented_without_lora() -> None:
    async def main() -> None:
        async with FakeEngine() as engine:
            async with grpc.aio.insecure_channel(engine.target) as channel:
                control = openengine_pb2_grpc.ControlStub(channel)
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await control.ListLoras(lora_pb2.ListLorasRequest())
                assert error.value.code() == grpc.StatusCode.UNIMPLEMENTED

    run(main())


# -- Built-in generator ---------------------------------------------------------------


_LENGTH = {"reason": "LENGTH"}


@pytest.mark.parametrize(
    "stopping,prompt,expected_tokens,expected_finish",
    [
        ({"maxTokens": 5}, PROMPT, [5, 6, 7, 5, 6], _LENGTH),
        ({"maxTokens": 9}, [5, EOS, 7], [5], {"eosTokenId": EOS}),
        ({"maxTokens": 4, "ignoreEos": True}, [5, EOS, 7], [5, EOS, 7, 5], _LENGTH),
        (
            {"maxTokens": 9, "conditions": [{"stopTokenId": 7}]},
            PROMPT,
            [5, 6],
            {"stopTokenId": 7},
        ),
        (
            {"maxTokens": 9, "conditions": [{"stopTokenId": 7}], "includeStopInOutput": True},
            PROMPT,
            [5, 6, 7],
            {"stopTokenId": 7},
        ),
        # Until min_tokens tokens are sampled, EOS and stop tokens are skipped.
        ({"maxTokens": 9, "minTokens": 3}, [5, EOS, 7], [5, 7, 5], {"eosTokenId": EOS}),
        # Both tokens of a two-token stop string are left out.
        (
            {"maxTokens": 9, "conditions": [{"stopText": "<6><7>"}]},
            PROMPT,
            [5],
            {"stopText": "<6><7>"},
        ),
        (
            {"maxTokens": 9, "conditions": [{"stopText": "<6><7>"}], "includeStopInOutput": True},
            PROMPT,
            [5, 6, 7],
            {"stopText": "<6><7>"},
        ),
        # The first "<6>" is within min_tokens; the next one stops the output.
        (
            {"maxTokens": 9, "conditions": [{"stopText": "<6>"}], "minTokens": 2},
            PROMPT,
            [5, 6, 7, 5],
            {"stopText": "<6>"},
        ),
    ],
    ids=[
        "length",
        "eos",
        "ignore-eos",
        "stop-token",
        "stop-token-included",
        "min-tokens",
        "stop-text",
        "stop-text-included",
        "stop-text-after-min-tokens",
    ],
)
def test_generator_applies_the_stopping_options(
    stopping: dict[str, Any],
    prompt: list[int],
    expected_tokens: list[int],
    expected_finish: dict[str, Any],
) -> None:
    message = request(prompt, stopping=stopping)

    async def main() -> list[generation_pb2.GenerateResponse]:
        async with connected(FakeEngine()) as client:
            return await collect(client, message)

    responses = run(main())
    message.request_id = responses[0].request_id
    check_stream(message, responses)
    assert tokens_by_output(responses).get(0, []) == expected_tokens
    finished = finished_by_output(responses)[0]
    if expected_finish is _LENGTH:
        assert finished.reason == generation_pb2.FINISH_REASON_LENGTH
    else:
        assert finished.reason == generation_pb2.FINISH_REASON_STOP
        assert json_format.MessageToDict(finished.stop_match) == expected_finish
    # Only returned tokens count: an EOS, stop token or stop string left out does not.
    assert responses[-1].usage.completion_tokens == len(expected_tokens)


def test_generator_streams_several_outputs_with_logprobs() -> None:
    message = request(
        sampling={"numSequences": 3, "temperature": 0.7},
        stopping={"maxTokens": 4},
        response={
            "returnOutputLogprobs": True,
            "outputCandidates": {"topN": 2},
            "returnPromptLogprobs": True,
            "promptCandidates": {"topN": 1},
        },
    )
    engine = FakeEngine(tokens_per_step=2, cached_prompt_tokens=2)

    async def main() -> list[generation_pb2.GenerateResponse]:
        async with connected(engine) as client:
            return await collect(client, message)

    responses = run(main())
    message.request_id = responses[0].request_id
    check_stream(message, responses)
    assert tokens_by_output(responses) == {0: [5, 6, 7, 5], 1: [6, 7, 5, 6], 2: [7, 5, 6, 7]}
    prompt = responses[0].prompt
    assert [info.token_id for info in prompt.tokens] == PROMPT
    assert not prompt.tokens[0].HasField("logprob")
    assert all(len(info.candidates) == 1 for info in prompt.tokens[1:])
    first = responses[1].token
    assert len(first.tokens) == 2
    assert [c.token_id for c in first.tokens[0].candidates] == [5, 6]
    assert first.tokens[0].candidates[0].logprob > first.tokens[0].candidates[1].logprob
    usage = responses[-1].usage
    assert (usage.completion_tokens, usage.cached_prompt_tokens) == (12, 2)


def test_unset_max_tokens_fills_the_context_window() -> None:
    profile = PROFILES["trtllm-like"].replace(max_context_length=10)

    async def main() -> list[generation_pb2.GenerateResponse]:
        async with connected(FakeEngine(profile)) as client:
            return await collect(client, request())

    responses = run(main())
    assert len(tokens_by_output(responses)[0]) == 7
    assert finished_by_output(responses)[0].reason == generation_pb2.FINISH_REASON_LENGTH


# -- Admission ------------------------------------------------------------------------

_IMAGE_PROFILE = PROFILES["trtllm-like"].replace(modalities=frozenset({"image"}))
_PRIORITY_PROFILE = PROFILES["trtllm-like"].replace(supports_priority=True)

_REJECTIONS = [
    # (profile, request fields, extra, metadata, status)
    ("trtllm-like", {"model": ""}, None, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {"tokenIds": {"ids": []}}, None, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {"tokenIds": {"ids": [40000]}}, None, (), "INVALID_ARGUMENT"),
    ("glm-like", {"prompt": "hi"}, None, (), "UNIMPLEMENTED"),
    ("trtllm-like", {"sampling": {"temperature": -1}}, None, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {"sampling": {"topP": 1.5}}, None, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {"sampling": {"numSequences": 0}}, None, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {"sampling": {"numSequences": 17}}, None, (), "INVALID_ARGUMENT"),
    (
        "trtllm-like",
        {"sampling": {"numSequences": 2, "temperature": 0}},
        None,
        (),
        "INVALID_ARGUMENT",
    ),
    ("glm-like", {"sampling": {"numSequences": 2}}, None, (), "INVALID_ARGUMENT"),
    ("glm-like", {"sampling": {"presencePenalty": 0.5}}, None, (), "UNIMPLEMENTED"),
    ("trtllm-like", {"stopping": {"maxTokens": 0}}, None, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {"stopping": {"maxTokens": 4094}}, None, (), "INVALID_ARGUMENT"),
    (
        "trtllm-like",
        {"stopping": {"conditions": [{"stopTokenId": 9}] * 65}},
        None,
        (),
        "INVALID_ARGUMENT",
    ),
    ("trtllm-like", {"stopping": {"conditions": [{}]}}, None, (), "INVALID_ARGUMENT"),
    ("glm-like", {"response": {"returnPromptLogprobs": True}}, None, (), "UNIMPLEMENTED"),
    (
        "glm-like",
        {"response": {"returnOutputLogprobs": True, "outputCandidates": {"topN": 21}}},
        None,
        (),
        "INVALID_ARGUMENT",
    ),
    (
        "trtllm-like",
        {"response": {"returnOutputLogprobs": True, "outputCandidates": {"all": {}}}},
        None,
        (),
        "UNIMPLEMENTED",
    ),
    ("trtllm-like", {"response": {"promptLogprobStart": 1}}, None, (), "UNIMPLEMENTED"),
    ("trtllm-like", {"guided": {"choice": {"choices": ["a"]}}}, None, (), "UNIMPLEMENTED"),
    ("trtllm-like", {"guided": {"regex": "a", "backend": "x"}}, None, (), "UNIMPLEMENTED"),
    ("glm-like", {"guided": {"regex": "a"}}, None, (), "UNIMPLEMENTED"),
    ("glm-like", {"kv": {"cacheSalt": "s"}}, None, (), "UNIMPLEMENTED"),
    ("trtllm-like", {"kv": {"bypassPrefixCache": True}}, None, (), "UNIMPLEMENTED"),
    (
        "trtllm-like",
        {"media": [{"modality": "MODALITY_IMAGE", "url": "http://x/a.png"}]},
        None,
        (),
        "UNIMPLEMENTED",
    ),
    ("trtllm-like", {"loraName": "a"}, None, (), "UNIMPLEMENTED"),
    ("trtllm-like", {}, {"detokenize": "no"}, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {}, {"trtllm.unknown": 1}, (), "UNIMPLEMENTED"),
    ("trtllm-like", {}, {"trtllm.best_of": 2}, (), "UNIMPLEMENTED"),
    ("glm-like", {}, {"conversation_id": "c"}, (), "UNIMPLEMENTED"),
    ("glm-like", {}, {"trtllm.logit_bias": {"5": 1}}, (), "UNIMPLEMENTED"),
    ("trtllm-like", {}, {"conversation_id": ""}, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {}, {"trtllm.logit_bias": {"5": 101}}, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {}, {"trtllm.logit_bias": {"x": 1}}, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {}, {"trtllm.logit_bias": {"\uff15": 1}}, (), "INVALID_ARGUMENT"),
    (NO_CONVENTIONS, {}, {"conversation_id": "  "}, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {"sampling": {"repetitionPenalty": "NaN"}}, None, (), "INVALID_ARGUMENT"),
    (
        "trtllm-like",
        {},
        {
            "trtllm.thinking_token_budget": {
                "budget": 4,
                "start_token_ids": [],
                "end_token_ids": [3],
            }
        },
        (),
        "INVALID_ARGUMENT",
    ),
    (
        "trtllm-like",
        {},
        {"trtllm.scheduling": {"agent_hierarchy": [["a"]]}},
        (),
        "INVALID_ARGUMENT",
    ),
    ("trtllm-like", {}, {"trtllm.kv_hints": "x"}, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {}, {"trtllm.kv_hints": 3}, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {}, {"trtllm.kv_hints": "[1]"}, (), "INVALID_ARGUMENT"),
    (
        "trtllm-like",
        {},
        {"trtllm.kv_hints": json.dumps(KV_HINTS)[:-1] + ', "x": NaN}'},
        (),
        "INVALID_ARGUMENT",
    ),
    (
        "trtllm-like",
        {},
        {"trtllm.kv_hints": json.dumps(KV_HINTS).ljust(KV_HINTS_TEXT_MAX_BYTES + 1)},
        (),
        "INVALID_ARGUMENT",
    ),
    (
        "trtllm-like",
        {},
        {"trtllm.kv_hints": json.dumps({**KV_HINTS, "protocol_version": "1"})},
        (),
        "INVALID_ARGUMENT",
    ),
    ("trtllm-like", {}, {"trtllm.kv_hints": {"protocol_version": "1"}}, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {}, {"trtllm.kv_hints": {"message_id": "m"}}, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {}, {"trtllm.kv_hints": {**KV_HINTS, "n": 2**53}}, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {}, None, (("openengine-priority", "500"),), "UNIMPLEMENTED"),
    ("trtllm-like", {}, None, (("openengine-routing-key", "k"),), "UNIMPLEMENTED"),
    ("trtllm-like", {}, None, (("openengine-target-dp-rank", "1"),), "INVALID_ARGUMENT"),
    ("trtllm-like", {}, None, (("openengine-target-dp-rank", "x"),), "INVALID_ARGUMENT"),
    (
        "trtllm-like",
        {"sampling": {"numSequences": 2}},
        None,
        (("openengine-target-dp-rank", "0"),),
        "INVALID_ARGUMENT",
    ),
    ("glm-like", {}, None, (("openengine-target-dp-rank", "0"),), "UNIMPLEMENTED"),
    (
        _IMAGE_PROFILE,
        {"media": [{"modality": "MODALITY_AUDIO", "url": "u"}]},
        None,
        (),
        "UNIMPLEMENTED",
    ),
    (_IMAGE_PROFILE, {"media": [{"modality": "MODALITY_IMAGE"}]}, None, (), "INVALID_ARGUMENT"),
    (_PRIORITY_PROFILE, {}, None, (("openengine-priority", "x"),), "INVALID_ARGUMENT"),
    (_PRIORITY_PROFILE, {}, None, (("openengine-priority", "1001"),), "INVALID_ARGUMENT"),
    (
        _PRIORITY_PROFILE,
        {},
        None,
        (("openengine-priority", "1"), ("openengine-priority", "2")),
        "INVALID_ARGUMENT",
    ),
    (
        PROFILES["trtllm-like"].replace(supports_token_ids_input=False),
        {},
        None,
        (),
        "UNIMPLEMENTED",
    ),
    ("trtllm-like", {"tokenIds": {"ids": [1] * 4096}}, None, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {"sampling": {"topK": -1}}, None, (), "INVALID_ARGUMENT"),
    ("trtllm-like", {"sampling": {"minP": 1.5}}, None, (), "INVALID_ARGUMENT"),
    (
        PROFILES["trtllm-like"].replace(capabilities={CAP_STOP_TEXT: False}),
        {"stopping": {"conditions": [{"stopText": "x"}]}},
        None,
        (),
        "UNIMPLEMENTED",
    ),
    ("trtllm-like", {"stopping": {"conditions": [{"stopText": ""}]}}, None, (), "INVALID_ARGUMENT"),
    (
        PROFILES["trtllm-like"].replace(supports_stop_in_output=False),
        {"stopping": {"includeStopInOutput": True}},
        None,
        (),
        "UNIMPLEMENTED",
    ),
    (
        PROFILES["trtllm-like"].replace(output_logprobs_max_top_n=None),
        {"response": {"returnOutputLogprobs": True}},
        None,
        (),
        "UNIMPLEMENTED",
    ),
    ("glm-like", {"kv": {"session": {}}}, None, (), "UNIMPLEMENTED"),
    (
        PROFILES["trtllm-like"].replace(supports_lora=True, max_loras=1),
        {"loraName": "a"},
        None,
        (),
        "FAILED_PRECONDITION",
    ),
    (
        PROFILES["trtllm-like"].replace(capabilities={CAP_BEST_OF: True}),
        {},
        {"trtllm.best_of": 1},
        (),
        "INVALID_ARGUMENT",
    ),
    ("trtllm-like", {}, {"trtllm.thinking_token_budget": {"budget": "x"}}, (), "INVALID_ARGUMENT"),
    (
        "trtllm-like",
        {},
        {"trtllm.logit_bias": {str(token): 1 for token in range(301)}},
        (),
        "INVALID_ARGUMENT",
    ),
    ("trtllm-like", {}, {"trtllm.scheduling": {"other": 1}}, (), "INVALID_ARGUMENT"),
    (
        "trtllm-like",
        {},
        {"trtllm.scheduling": {"attention_dp_relax": 1}},
        (),
        "INVALID_ARGUMENT",
    ),
    ("glm-like", {"sampling": {"repetitionPenalty": 1.1}}, None, (), "UNIMPLEMENTED"),
    # glm checks a value as a 32-bit float and against the protocol's bounds before
    # its declared values.
    ("glm-like", {"sampling": {"topP": 2.0**-150}}, None, (), "INVALID_ARGUMENT"),
    ("glm-like", {"sampling": {"topP": 0}}, None, (), "INVALID_ARGUMENT"),
    ("glm-like", {"sampling": {"temperature": 3.5e38}}, None, (), "INVALID_ARGUMENT"),
    ("glm-like", {"sampling": {"temperature": -1}}, None, (), "INVALID_ARGUMENT"),
    ("glm-like", {"sampling": {"topK": -1}}, None, (), "INVALID_ARGUMENT"),
    ("glm-like", {"sampling": {"minP": 1.5}}, None, (), "INVALID_ARGUMENT"),
    ("glm-like", {"sampling": {"repetitionPenalty": 1e39}}, None, (), "INVALID_ARGUMENT"),
    # A finite 32-bit float above the declared maximum.
    ("glm-like", {"sampling": {"temperature": 3.4028235e38}}, None, (), "UNIMPLEMENTED"),
    (SPECULATIVE, {"sampling": {"minP": 0.1}}, None, (), "UNIMPLEMENTED"),
    # A value the profile does not accept is refused before the protocol's bounds are checked.
    (SPECULATIVE, {"sampling": {"minP": 1.5}}, None, (), "UNIMPLEMENTED"),
    (
        PROFILES["trtllm-like"].replace(capabilities={CAP_PRESENCE_PENALTY: False}),
        {"sampling": {"presencePenalty": 0}},
        None,
        (),
        "UNIMPLEMENTED",
    ),
    # Within the length limit in characters, beyond it in UTF-8 bytes.
    (
        "trtllm-like",
        {},
        {
            "trtllm.kv_hints": json.dumps(
                {**KV_HINTS, "x": "\u4e2d" * 1000}, ensure_ascii=False
            ).ljust(KV_HINTS_TEXT_MAX_BYTES - 1000)
        },
        (),
        "INVALID_ARGUMENT",
    ),
    ("trtllm-like", {}, {"trtllm.kv_hints": {**KV_HINTS, "n": -(2**53)}}, (), "INVALID_ARGUMENT"),
]


@pytest.mark.parametrize(
    "profile,fields,extra,metadata,status",
    _REJECTIONS,
    ids=[str(i) for i in range(len(_REJECTIONS))],
)
def test_refused_requests_get_the_declared_status_and_no_events(
    profile: str | FakeProfile,
    fields: dict[str, Any],
    extra: dict[str, Any] | None,
    metadata,
    status: str,
) -> None:
    message = request(extra=extra, stopping={"maxTokens": 2})
    parse_into(fields, message)

    async def main() -> None:
        async with FakeEngine(profile) as engine:
            responses, code, details = await raw_generate(engine, message, metadata)
            assert (code.name, responses) == (status, []), details
            assert details
            assert engine.counters == {COUNTER_GENERATE_RECEIVED: 1, COUNTER_GENERATE_ACCEPTED: 0}
            assert engine.received[0].rejection is not None
            assert engine.running_request_ids == []

    run(main())


@pytest.mark.parametrize(
    "hints",
    [
        KV_HINTS,
        json.dumps(KV_HINTS),
        # Text may hold an integer beyond 2**53, and whitespace.
        json.dumps({**KV_HINTS, "n": 2**60}, indent=2),
        json.dumps(KV_HINTS).ljust(KV_HINTS_TEXT_MAX_BYTES),
        {**KV_HINTS, "n": 2**53 - 1},
    ],
    ids=[
        "object",
        "text",
        "text-large-integer",
        "text-longest",
        "object-largest-safe-integer",
    ],
)
def test_kv_hints_are_taken_as_json_text_or_as_an_object(hints: str | dict[str, Any]) -> None:
    message = request(stopping={"maxTokens": 1}, extra={"trtllm.kv_hints": hints})

    async def main() -> None:
        async with FakeEngine() as engine:
            _, code, details = await raw_generate(engine, message)
            assert code == grpc.StatusCode.OK, details

    run(main())


def test_kv_hints_text_limit_is_the_servers() -> None:
    from tensorrt_llm.grpc.openengine import request_mapping
    from tensorrt_llm.kv_hints.protocol import MAX_KV_HINTS_BYTES

    per_byte = request_mapping._KV_HINTS_TEXT_BYTES_PER_ENVELOPE_BYTE
    assert KV_HINTS_TEXT_MAX_BYTES == per_byte * MAX_KV_HINTS_BYTES


@pytest.mark.parametrize(
    "profile,sampling",
    [
        ("glm-like", {"repetitionPenalty": 1.0, "frequencyPenalty": 0, "presencePenalty": 0}),
        ("glm-like", {"topP": 2.0**-149, "minP": 1, "topK": 0, "temperature": 3.4e38}),
        (
            SPECULATIVE,
            {"minP": 0, "repetitionPenalty": 1.0, "frequencyPenalty": 0, "presencePenalty": 0},
        ),
        ("trtllm-like", {"minP": 0.5, "repetitionPenalty": 1.3, "presencePenalty": -2}),
    ],
    ids=["glm-no-effect-penalties", "glm-in-range", "speculative-no-effect-values", "trtllm-any"],
)
def test_declared_sampling_values_are_accepted(
    profile: str | FakeProfile, sampling: dict[str, Any]
) -> None:
    message = request(sampling=sampling, stopping={"maxTokens": 1})

    async def main() -> None:
        async with FakeEngine(profile) as engine:
            _, code, details = await raw_generate(engine, message)
            assert code == grpc.StatusCode.OK, details

    run(main())


def test_command_line_overrides_accepted_values() -> None:
    import fake_openengine

    changes = {"params": {"min_p": {"accept": "values", "values": [0.0], "default": 0.0}}}
    profile = fake_openengine._profile_from_args(
        fake_openengine._parse_args(["--profile-json", json.dumps(changes)])
    )
    assert profile.accepted("min_p") == SPECULATIVE.accepted("min_p")
    assert profile.accepted("top_k") == PROFILES["trtllm-like"].accepted("top_k")


def test_a_request_without_input_is_invalid() -> None:
    with pytest.raises(Rejection) as refused:
        admit(generation_pb2.GenerateRequest(model="fake-model"), (), PROFILES["trtllm-like"])
    assert refused.value.code == grpc.StatusCode.INVALID_ARGUMENT


def test_text_prompt_and_declared_media_are_accepted() -> None:
    message = request(stopping={"maxTokens": 2}, response={"returnOutputLogprobs": True})
    parse_into({"prompt": "hi", "media": [{"modality": "MODALITY_IMAGE", "url": "u"}]}, message)

    async def main() -> None:
        async with FakeEngine(_IMAGE_PROFILE) as engine:
            responses, code, details = await raw_generate(engine, message)
            assert code == grpc.StatusCode.OK, details
            # One token per UTF-8 byte of the prompt, replayed as the output.
            assert tokens_by_output(responses)[0] == [ord("h"), ord("i")]
            assert responses[-1].usage.prompt_tokens == 2

    run(main())


def test_profile_without_conventions_ignores_undeclared_and_unknown_keys() -> None:
    message = request(
        sampling={"presencePenalty": 0.5, "minP": 0.1},
        stopping={"maxTokens": 2, "conditions": [{"stopText": "x"}]},
        # trtllm-like's best_of capability is off, so its invalid value is ignored too.
        extra={
            "conversation_id": "c",
            "trtllm.logit_bias": {"5": 1},
            "trtllm.best_of": "x",
            "trtllm.unknown": 1,
        },
    )

    async def main() -> None:
        async with FakeEngine(NO_CONVENTIONS) as engine:
            responses, code, details = await raw_generate(engine, message)
            assert code == grpc.StatusCode.OK, details
            assert finished_by_output(responses)[0].reason == generation_pb2.FINISH_REASON_LENGTH

    run(main())


def test_accepted_extensions_and_metadata_are_recorded() -> None:
    profile = PROFILES["trtllm-like"].replace(supports_priority=True, data_parallel_size=2)
    message = request(
        sampling={"presencePenalty": 0.5, "seed": 3},
        stopping={"maxTokens": 2},
        kv={"cacheSalt": "salt"},
        extra={
            "detokenize": False,
            "conversation_id": "c",
            "trtllm.logit_bias": {"5": -100, "6": 2.5},
            "trtllm.thinking_token_budget": {
                "budget": 0,
                "start_token_ids": [3],
                "end_token_ids": [4],
            },
            "trtllm.scheduling": {"attention_dp_relax": False, "agent_hierarchy": [["a", 1]]},
            "trtllm.kv_hints": KV_HINTS,
        },
    )
    metadata = (
        ("traceparent", "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"),
        ("openengine-priority", "750"),
        ("openengine-target-dp-rank", "1"),
    )

    async def main() -> None:
        async with FakeEngine(profile) as engine:
            _, code, details = await raw_generate(engine, message, metadata)
            assert code == grpc.StatusCode.OK, details
            record = engine.received[0]
            assert record.request == message
            assert record.metadata_value("traceparent") == metadata[0][1]
            assert (record.admitted.priority, record.admitted.target_dp_rank) == (750, 1)

    run(main())


def test_duplicate_and_unready_requests_are_refused() -> None:
    engine = FakeEngine(script=[Hang()])

    async def main() -> None:
        async with engine:
            message = request(requestId="same")
            first = asyncio.create_task(raw_generate(engine, message))
            await _wait_until(lambda: engine.running_request_ids == ["same"])
            _, code, _ = await raw_generate(engine, request(requestId="same"))
            assert code == grpc.StatusCode.ALREADY_EXISTS
            assert engine.abort("same")
            responses, code, _ = await first
            assert code == grpc.StatusCode.OK
            assert responses[-1].finished.reason == generation_pb2.FINISH_REASON_CANCELLED
            engine.set_health(lifecycle_pb2.HEALTH_STATE_NOT_READY)
            _, code, _ = await raw_generate(engine, request(requestId="later"))
            assert code == grpc.StatusCode.UNAVAILABLE

    run(main())


async def _wait_until(predicate, timeout: float = WAIT_SECONDS) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)


# -- Counters and health -----------------------------------------------------------------


def test_counters_count_every_call_and_never_decrease() -> None:
    engine = FakeEngine(step_delay=0.01)

    async def main() -> None:
        seen: list[tuple[float, float]] = []

        async def snapshot(client: OpenEngineClient) -> tuple[float, float]:
            load = await client.get_load()
            values = json_format.MessageToDict(load.attributes)
            seen.append((values[COUNTER_GENERATE_RECEIVED], values[COUNTER_GENERATE_ACCEPTED]))
            return seen[-1]

        async with connected(engine) as client:
            assert await snapshot(client) == (0, 0)
            await collect(client, request(stopping={"maxTokens": 2}))
            assert await snapshot(client) == (1, 1)
            await raw_generate(engine, request(model=""))
            assert await snapshot(client) == (2, 1)
            cancelled = request(requestId="cancelled", stopping={"maxTokens": 1000})
            async with client.generate(cancelled) as stream:
                await stream.__anext__()
                stream.cancel()
            await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
            assert await snapshot(client) == (3, 2)
            aborted = request(requestId="aborted", stopping={"maxTokens": 1000})
            async with client.generate(aborted) as stream:
                await stream.__anext__()
                await client.abort(aborted.request_id)
                [response async for response in stream]
            assert await snapshot(client) == (4, 3)
        assert engine.cancelled == [cancelled.request_id]
        for counter in range(2):
            values = [counters[counter] for counters in seen]
            assert values == sorted(values), seen

    run(main())


def test_health_state_drives_both_health_services() -> None:
    engine = FakeEngine(health_state=lifecycle_pb2.HEALTH_STATE_STARTING)

    async def check(service: str) -> int:
        async with grpc.aio.insecure_channel(engine.target) as channel:
            stub = health_pb2_grpc.HealthStub(channel)
            return (await stub.Check(health_pb2.HealthCheckRequest(service=service))).status

    async def main() -> None:
        async with engine:
            serving = health_pb2.HealthCheckResponse.SERVING
            not_serving = health_pb2.HealthCheckResponse.NOT_SERVING
            assert (await check("inference"), await check("liveness")) == (not_serving, serving)
            connecting = asyncio.create_task(
                OpenEngineClient.connect(engine.target, ready_timeout=WAIT_SECONDS)
            )
            await asyncio.sleep(0.2)
            assert not connecting.done()
            engine.set_health(lifecycle_pb2.HEALTH_STATE_READY)
            client = await connecting
            try:
                assert await check("") == serving
                engine.set_health(lifecycle_pb2.HEALTH_STATE_NOT_READY)
                assert (await client.health()).state == lifecycle_pb2.HEALTH_STATE_NOT_READY
                assert await check("inference") == not_serving
            finally:
                await client.close()

    run(main())


# -- Scripts -------------------------------------------------------------------------


def _token(index: int, *ids: int) -> generation_pb2.GenerateResponse:
    return generation_pb2.GenerateResponse(
        token=generation_pb2.TokenOutput(
            output_index=index, tokens=[generation_pb2.TokenInfo(token_id=i) for i in ids]
        )
    )


def _stop_text_finish(text: str) -> generation_pb2.GenerateResponse:
    response = generation_pb2.GenerateResponse(
        finished=generation_pb2.GenerationFinished(
            output_index=0,
            reason=generation_pb2.FINISH_REASON_STOP,
            stop_match=generation_pb2.StopMatch(stop_text=text),
        )
    )
    response.usage.prompt_tokens = 3
    return response


def test_script_replays_injected_stop_match_after_a_delay() -> None:
    engine = FakeEngine(script=[_token(0, 9), Delay(0.2), _stop_text_finish("END")])

    async def main() -> tuple[list[generation_pb2.GenerateResponse], float]:
        async with connected(engine) as client:
            loop = asyncio.get_running_loop()
            started = loop.time()
            responses = await collect(client, request())
            return responses, loop.time() - started

    responses, elapsed = run(main())
    assert elapsed >= 0.2
    assert responses[-1].finished.stop_match.stop_text == "END"
    assert all(response.request_id for response in responses)


@pytest.mark.parametrize(
    "steps,status,accepted",
    [
        ([Fail(grpc.StatusCode.RESOURCE_EXHAUSTED, "busy")], "RESOURCE_EXHAUSTED", 0),
        ([_token(0, 9), Fail(grpc.StatusCode.INTERNAL, "boom")], "INTERNAL", 1),
        ([_token(0, 9), Disconnect()], "UNAVAILABLE", 1),
        ([_token(0, 9)], "INTERNAL", 1),
        ([], "INTERNAL", 1),
    ],
    ids=["refused", "failed", "disconnected", "ran-out", "empty"],
)
def test_script_ends_a_call_with_a_status(steps: list[Any], status: str, accepted: int) -> None:
    engine = FakeEngine()
    engine.push_script(*steps)

    async def main() -> None:
        async with connected(engine) as client:
            with pytest.raises(OpenEngineRpcError) as error:
                await collect(client, request())
            assert error.value.code.name == status
            # The queued script is used once; the next call is generated.
            await collect(client, request(stopping={"maxTokens": 1}))
        assert engine.counters[COUNTER_GENERATE_ACCEPTED] == accepted + 1

    run(main())


def test_script_injects_an_engine_error() -> None:
    error_event = generation_pb2.GenerateResponse(
        error=error_pb2.EngineError(code=error_pb2.ERROR_CODE_INTERNAL, message="oom")
    )
    error_event.usage.prompt_tokens = 3
    engine = FakeEngine(script=lambda message: [_token(0, 9), error_event])

    async def main() -> None:
        async with connected(engine) as client:
            with pytest.raises(OpenEngineEngineError, match="oom"):
                await collect(client, request())

    run(main())


def test_unsupported_script_steps_are_refused_up_front() -> None:
    with pytest.raises(TypeError, match="unsupported script step"):
        FakeEngine(script=[object()])
    with pytest.raises(TypeError, match="unsupported script step"):
        FakeEngine().push_script("not a step")


def test_stopping_the_engine_is_not_a_client_cancellation() -> None:
    engine = FakeEngine(script=[_token(0, 9), Hang()])

    async def main() -> None:
        await engine.start()
        client = await OpenEngineClient.connect(engine.target, ready_timeout=WAIT_SECONDS)
        try:
            async with client.generate(request()) as stream:
                await stream.__anext__()
                await engine.stop()
                with pytest.raises(OpenEngineRpcError):
                    [response async for response in stream]
        finally:
            await client.close()
        assert engine.cancelled == [] and not engine.cancel_seen.is_set()

    run(main())


def test_abort_after_stop_reports_nothing_running() -> None:
    engine = FakeEngine()

    async def main() -> None:
        async with engine:
            pass

    run(main())
    assert engine.abort("gone") is False


def test_tokens_that_may_start_a_stop_string_are_held_back() -> None:
    message = request(
        [5, 6, 8, 9], stopping={"maxTokens": 9, "conditions": [{"stopText": "<6><7>"}]}
    )

    async def main() -> list[generation_pb2.GenerateResponse]:
        async with connected(FakeEngine()) as client:
            return await collect(client, message)

    responses = run(main())
    token_messages = [
        [info.token_id for info in response.token.tokens]
        for response in responses
        if response.WhichOneof("event") == "token"
    ]
    # "<6>" could start "<6><7>", so token 6 waits until token 8 rules that out.
    assert token_messages[:2] == [[5], [6, 8]]
    assert tokens_by_output(responses)[0] == [5, 6, 8, 9, 5, 6, 8, 9, 5]


def test_script_function_may_fall_back_to_the_generator() -> None:
    engine = FakeEngine(script=lambda message: None)

    async def main() -> None:
        async with connected(engine) as client:
            responses = await collect(client, request(stopping={"maxTokens": 2}))
        assert tokens_by_output(responses) == {0: [5, 6]}

    run(main())


# -- Ways to run the engine ------------------------------------------------------------------


def test_engine_serves_from_a_thread_for_synchronous_callers() -> None:
    with running_in_thread(FakeEngine("glm-like")) as engine:
        with grpc.insecure_channel(engine.target) as channel:
            stub = openengine_pb2_grpc.InferenceStub(channel)
            message = request(requestId="sync", stopping={"maxTokens": 2})
            responses = list(stub.Generate(message, timeout=WAIT_SECONDS))
            info = openengine_pb2_grpc.ControlStub(channel).GetServerInfo(
                server_pb2.GetServerInfoRequest(), timeout=WAIT_SECONDS
            )
        assert tokens_by_output(responses) == {0: [5, 6]}
        assert info.engine_name == "glm5_serve"
        assert len(engine.received) == 1


def test_command_line_engine_serves_until_terminated(tmp_path) -> None:
    port_file = tmp_path / "port"
    script = os.path.join(os.path.dirname(__file__), "fake_openengine.py")
    process = subprocess.Popen(
        [
            sys.executable,
            script,
            "--profile",
            "glm-like",
            "--port-file",
            str(port_file),
            "--profile-json",
            '{"model": "cli-model", "max_context_length": 128}',
        ],
    )
    try:

        async def main() -> None:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 300
            while not port_file.exists():
                assert process.poll() is None, "the engine exited"
                assert loop.time() < deadline, "the engine did not start"
                await asyncio.sleep(0.1)
            target = f"127.0.0.1:{port_file.read_text().strip()}"
            client = await OpenEngineClient.connect(target, ready_timeout=WAIT_SECONDS)
            try:
                assert client.model_info.served_model_name == "cli-model"
                assert client.model_info.max_context_length == 128
                responses = await collect(
                    client, request(model="cli-model", stopping={"maxTokens": 2})
                )
                assert tokens_by_output(responses) == {0: [5, 6]}
            finally:
                await client.close()

        asyncio.run(main())
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``OpenEngineEngineClient`` and the client's metadata, LoRA and KV-source calls on the fake."""

from __future__ import annotations

import asyncio
from typing import Any

import grpc
import pytest
from fake_openengine import PROFILES, FakeEngine, FakeProfile, Hang
from openengine_client_helpers import WAIT_SECONDS, token

from tensorrt_llm.grpc.openengine.bindings import generation_pb2, kv_pb2, lifecycle_pb2, lora_pb2
from tensorrt_llm.grpc.openengine_client import (
    OpenEngineClient,
    OpenEngineEngineError,
    OpenEngineRpcError,
)
from tensorrt_llm.grpc.openengine_conventions import (
    CAP_STOP_TEXT,
    COUNTER_GENERATE_ACCEPTED,
    COUNTER_GENERATE_RECEIVED,
    META_PRIORITY,
    META_TRACEPARENT,
)
from tensorrt_llm.serve.openengine_poc.engine_client import (
    EngineClient,
    EngineRequest,
    GuidedSpec,
    MediaSpec,
    OutputFinished,
    PromptInfo,
    TokenDelta,
)
from tensorrt_llm.serve.openengine_poc.engine_client_impl import (
    OpenEngineEngineClient,
    to_generate_request,
)

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

PROMPT = [5, 6, 7, 8]


def _run(engine: FakeEngine, body) -> Any:
    async def main() -> Any:
        async with engine:
            client = await OpenEngineEngineClient.connect(engine.target, ready_timeout=WAIT_SECONDS)
            try:
                return await asyncio.wait_for(body(engine, client), WAIT_SECONDS * 2)
            finally:
                await client.close()

    return asyncio.run(main())


def _request(**fields: Any) -> EngineRequest:
    fields.setdefault("request_id", "trtllm-0123")
    fields.setdefault("prompt_token_ids", list(PROMPT))
    fields.setdefault("max_tokens", 3)
    fields.setdefault("model", "fake-model")
    return EngineRequest(**fields)


async def _events(client: OpenEngineEngineClient, request: EngineRequest) -> list[Any]:
    stream = client.generate(request)
    try:
        return [event async for event in stream]
    finally:
        await stream.aclose()


# -- Request building ------------------------------------------------------------


def test_request_carries_every_set_field_and_leaves_the_rest_unset() -> None:
    message = to_generate_request(
        _request(
            sampling={"temperature": 0.0, "top_k": 5, "seed": 7},
            min_tokens=2,
            stop_token_ids=[9],
            stop_texts=["##"],
            ignore_eos=True,
            include_stop_in_output=False,
            num_sequences=2,
            output_logprobs_top_n=0,
            prompt_logprobs_top_n=3,
            guided=GuidedSpec("regex", "a+"),
            cache_salt="salt",
            media=[MediaSpec("image", url="http://x/y.png", uuid="h")],
            lora_name="ada",
            extra={"detokenize": False},
        )
    )
    assert message.request_id == "trtllm-0123"
    assert list(message.token_ids.ids) == PROMPT
    # temperature 0 is set, not left out: it means greedy.
    assert message.sampling.HasField("temperature") and message.sampling.temperature == 0.0
    assert message.sampling.top_k == 5 and message.sampling.seed == 7
    assert not message.sampling.HasField("top_p")
    assert message.sampling.num_sequences == 2
    assert message.stopping.max_tokens == 3 and message.stopping.min_tokens == 2
    assert [c.WhichOneof("condition") for c in message.stopping.conditions] == [
        "stop_token_id",
        "stop_text",
    ]
    assert message.stopping.ignore_eos is True
    assert message.stopping.HasField("include_stop_in_output")
    assert message.response.return_output_logprobs and message.response.output_candidates.top_n == 0
    assert message.response.return_prompt_logprobs and message.response.prompt_candidates.top_n == 3
    assert message.guided.regex == "a+"
    assert message.kv.cache_salt == "salt"
    assert message.media[0].url == "http://x/y.png" and message.media[0].uuid == "h"
    assert message.lora_name == "ada"
    assert message.extra.fields["detokenize"].bool_value is False


def test_request_leaves_optional_sections_unset_by_default() -> None:
    message = to_generate_request(_request(max_tokens=None))
    assert not message.HasField("guided")
    assert not message.HasField("kv")
    assert not message.response.HasField("return_output_logprobs")
    assert not message.stopping.HasField("max_tokens")
    assert not message.sampling.HasField("num_sequences")
    assert not message.HasField("extra")


def test_json_object_guide() -> None:
    message = to_generate_request(_request(guided=GuidedSpec("json_object")))
    assert message.guided.WhichOneof("guide") == "json_object"


def test_unknown_sampling_field_is_refused() -> None:
    with pytest.raises(ValueError, match="best_of"):
        to_generate_request(_request(sampling={"best_of": 2}))


# -- Generate --------------------------------------------------------------------


def test_client_satisfies_the_engine_client_protocol() -> None:
    async def body(engine, client):
        return isinstance(client, EngineClient)

    assert _run(FakeEngine(), body)


def test_generate_yields_tokens_then_finished_with_usage() -> None:
    async def body(engine, client):
        return await _events(client, _request())

    events = _run(FakeEngine(), body)
    tokens = [t for e in events if isinstance(e, TokenDelta) for t in e.token_ids]
    assert tokens == [5, 6, 7]
    finished = events[-1]
    assert isinstance(finished, OutputFinished)
    assert finished.reason == "length" and finished.output_index == 0
    assert finished.usage is not None and finished.usage.completion_tokens == 3
    assert all(e.logprobs is None for e in events if isinstance(e, TokenDelta))


def test_generate_reports_logprobs_and_prompt_logprobs() -> None:
    async def body(engine, client):
        return await _events(client, _request(output_logprobs_top_n=2, prompt_logprobs_top_n=1))

    events = _run(FakeEngine(), body)
    assert isinstance(events[0], PromptInfo)
    delta = next(e for e in events if isinstance(e, TokenDelta))
    assert delta.logprobs is not None and delta.logprobs[0].logprob == -0.5
    assert len(delta.logprobs[0].top) == 2


def test_generate_reports_what_stopped_an_output() -> None:
    async def body(engine, client):
        return await _events(client, _request(max_tokens=10, stop_token_ids=[7]))

    finished = _run(FakeEngine(), body)[-1]
    assert finished.reason == "stop" and finished.stop_token_id == 7
    assert finished.eos_token_id is None and finished.stop_text is None


def test_generate_sends_metadata() -> None:
    profile = PROFILES["trtllm-like"].replace(supports_priority=True)
    metadata = {META_TRACEPARENT: "00-abc-def-01", META_PRIORITY: "250"}

    async def body(engine, client):
        await _events(client, _request(metadata=metadata))
        return engine.received[-1]

    received = _run(FakeEngine(profile), body)
    assert received.admitted is not None
    assert received.metadata_value(META_TRACEPARENT) == "00-abc-def-01"
    assert received.metadata_value(META_PRIORITY) == "250"


def test_undeclared_protocol_metadata_is_refused_before_acceptance() -> None:
    async def body(engine, client):
        with pytest.raises(OpenEngineRpcError) as raised:
            await _events(client, _request(metadata={META_PRIORITY: "250"}))
        return raised.value, engine.counters

    error, counters = _run(FakeEngine(), body)
    assert error.code == grpc.StatusCode.UNIMPLEMENTED
    assert counters[COUNTER_GENERATE_RECEIVED] == 1
    assert counters[COUNTER_GENERATE_ACCEPTED] == 0


@pytest.mark.parametrize("key", ["Bad Key", "trace-bin", "grpc-timeout"])
def test_invalid_metadata_key_is_refused_locally(key: str) -> None:
    async def body(engine, client):
        with pytest.raises(ValueError, match="metadata key"):
            client.generate(_request(metadata={key: "x"}))
        return engine.counters[COUNTER_GENERATE_RECEIVED]

    assert _run(FakeEngine(), body) == 0


def test_engine_error_after_acceptance_is_raised_while_iterating() -> None:
    error = generation_pb2.GenerateResponse()
    error.error.code = 1
    error.error.message = "boom"

    async def body(engine, client):
        engine.push_script(token(0, 5), error)
        stream = client.generate(_request())
        first = await stream.__anext__()
        with pytest.raises(OpenEngineEngineError, match="boom"):
            await stream.__anext__()
        await stream.aclose()
        return first

    assert isinstance(_run(FakeEngine(), body), TokenDelta)


def test_cancel_aborts_the_request_on_the_engine() -> None:
    async def body(engine, client):
        engine.push_script(token(0, 5), Hang())
        stream = client.generate(_request())
        await stream.__anext__()
        stream.cancel()
        rest = [event async for event in stream]
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        return rest, stream.cancelled

    rest, cancelled = _run(FakeEngine(), body)
    assert rest == [] and cancelled


# -- Control ---------------------------------------------------------------------


def test_identity_and_capabilities_are_cached() -> None:
    async def body(engine, client):
        return client.server_info(), client.model_info()

    info, caps = _run(FakeEngine(instance_id="fake-1"), body)
    assert info.engine_name == "tensorrt_llm" and info.instance_id == "fake-1"
    assert info.schema_revision == 1 and info.max_running_requests == 64
    assert caps.served_model_name == "fake-model" and caps.supports(CAP_STOP_TEXT)
    assert caps.tokenizer_fingerprint is not None


def test_refresh_reads_a_restarted_engine() -> None:
    async def body(engine, client):
        engine.instance_id = "fake-2"
        await client.refresh()
        return client.server_info().instance_id

    assert _run(FakeEngine(instance_id="fake-1"), body) == "fake-2"


def test_health_maps_states() -> None:
    async def body(engine, client):
        ready = await client.health()
        engine.set_health(lifecycle_pb2.HEALTH_STATE_NOT_READY)
        not_ready = await client.health()
        return ready, not_ready

    ready, not_ready = _run(FakeEngine(), body)
    assert ready.state == "ready" and ready.serving
    assert not_ready.state == "not_ready" and not not_ready.serving


def test_load_reports_the_counters() -> None:
    async def body(engine, client):
        await _events(client, _request())
        return await client.load()

    load = _run(FakeEngine(), body)
    assert load is not None and load.instance_id
    assert load.attributes[COUNTER_GENERATE_RECEIVED] == 1
    assert load.attributes[COUNTER_GENERATE_ACCEPTED] == 1


def test_abort_of_an_unknown_request_is_not_an_error() -> None:
    async def body(engine, client):
        await client.abort("trtllm-unknown")
        return True

    assert _run(FakeEngine(), body)


def _lora_profile() -> FakeProfile:
    return PROFILES["trtllm-like"].replace(supports_lora=True, max_loras=4)


def test_load_and_list_loras() -> None:
    async def body(engine, client):
        first = await client.load_lora("ada", 1, "/loras/ada")
        again = await client.load_lora("ada", 1, "/loras/ada")
        return first, again, await client.list_loras()

    first, again, loras = _run(FakeEngine(_lora_profile()), body)
    assert (first, again) == (False, True)
    assert [(a.name, a.int_id, a.path) for a in loras] == [("ada", 1, "/loras/ada")]


def test_lora_on_an_engine_without_lora_fails() -> None:
    async def body(engine, client):
        with pytest.raises(OpenEngineRpcError) as raised:
            await client.load_lora("ada", 1, "/loras/ada")
        return raised.value.code

    assert _run(FakeEngine(), body) == grpc.StatusCode.UNIMPLEMENTED


def test_kv_event_sources() -> None:
    source = kv_pb2.KvEventSource(
        transport="zmq",
        endpoint_addr=kv_pb2.KvEndpoint(host="127.0.0.1", port=5557, protocol="tcp"),
        topic="kv",
        data_parallel_rank=0,
        encoding="msgpack",
    )
    profile = PROFILES["trtllm-like"].replace(kv_event_sources=(source,))

    async def body(engine, client):
        return await client.kv_event_sources()

    (found,) = _run(FakeEngine(profile), body)
    assert (found.transport, found.host, found.port, found.topic) == (
        "zmq",
        "127.0.0.1",
        5557,
        "kv",
    )
    assert found.data_parallel_rank == 0 and found.encoding == "msgpack"
    assert _run(FakeEngine(), body) == []


def test_client_library_lora_calls() -> None:
    async def main() -> Any:
        async with FakeEngine(_lora_profile()) as engine:
            async with await OpenEngineClient.connect(
                engine.target, ready_timeout=WAIT_SECONDS
            ) as client:
                response = await client.load_lora(
                    lora_pb2.LoraAdapter(lora_id=3, lora_name="b", source_path="/p")
                )
                return response.already_loaded, await client.list_loras()

    already, adapters = asyncio.run(main())
    assert already is False and adapters[0].lora_name == "b"

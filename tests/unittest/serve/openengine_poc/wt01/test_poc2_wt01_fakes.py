# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of the in-process FakeEngineClient."""

import asyncio
import dataclasses

import pytest

from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.serve.openengine_poc.engine_client import (
    EngineClient,
    EngineRequest,
    MediaSpec,
    OutputFinished,
    TokenDelta,
)

__extra_import_path__ = [".."]
from fakes import (  # noqa: E402
    DEFAULT_TOKENS,
    Delay,
    Fail,
    FakeEngineClient,
    Reject,
    glm_like,
    trtllm_like,
)

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def _request(request_id="trtllm-1", **fields):
    return EngineRequest(request_id=request_id, prompt_token_ids=[1, 2], max_tokens=8, **fields)


async def _collect(client, request):
    return [event async for event in client.generate(request)]


def test_fake_satisfies_the_protocol():
    assert isinstance(FakeEngineClient(), EngineClient)


def test_default_script_and_records():
    client = FakeEngineClient()
    events = asyncio.run(_collect(client, _request(num_sequences=2)))
    assert events[:2] == [TokenDelta(0, DEFAULT_TOKENS), TokenDelta(1, DEFAULT_TOKENS)]
    assert [e.output_index for e in events[2:]] == [0, 1]
    assert events[2].usage is None
    assert events[3].usage.completion_tokens == 2 * len(DEFAULT_TOKENS)
    assert [r.request_id for r in client.requests] == ["trtllm-1"]
    assert (client.generate_received, client.generate_accepted) == (1, 1)
    assert client.active == set() and client.cancelled == []


def test_max_tokens_caps_the_default_tokens():
    client = FakeEngineClient()
    events = asyncio.run(
        _collect(client, EngineRequest(request_id="r", prompt_token_ids=[1], max_tokens=2))
    )
    assert events[0] == TokenDelta(0, DEFAULT_TOKENS[:2])


def test_reject_before_acceptance_is_counted_as_received_only():
    client = FakeEngineClient(script=lambda request: [Reject(ValueError("no"))])
    stream = client.generate(_request())
    assert client.generate_received == 1
    with pytest.raises(ValueError, match="no"):
        asyncio.run(_collect_stream(stream))
    assert (client.generate_received, client.generate_accepted) == (1, 0)


async def _collect_stream(stream):
    return [event async for event in stream]


def test_fail_after_acceptance():
    client = FakeEngineClient(
        script=lambda request: [TokenDelta(0, (1,)), Fail(RuntimeError("engine died"))]
    )
    seen = []

    async def run():
        async for event in client.generate(_request()):
            seen.append(event)

    with pytest.raises(RuntimeError, match="engine died"):
        asyncio.run(run())
    assert seen == [TokenDelta(0, (1,))]
    assert client.generate_accepted == 1
    assert client.cancelled == ["trtllm-1"]


def test_reject_is_only_allowed_first():
    client = FakeEngineClient(script=lambda request: [TokenDelta(0, (1,)), Reject(ValueError())])
    with pytest.raises(ValueError, match="first step"):
        asyncio.run(_collect(client, _request()))


def test_abort_ends_the_unfinished_outputs_as_cancelled():
    client = FakeEngineClient(
        script=lambda request: [
            TokenDelta(0, (1,)),
            Delay(0.01),
            TokenDelta(0, (2,)),
            OutputFinished(0, "length"),
        ]
    )

    async def run():
        events = []
        async for event in client.generate(_request()):
            events.append(event)
            if len(events) == 1:
                assert client.active == {"trtllm-1"}
                await client.abort("trtllm-1")
        return events

    events = asyncio.run(run())
    assert events == [TokenDelta(0, (1,)), OutputFinished(0, "cancelled")]
    assert client.aborted == ["trtllm-1"]
    assert client.active == set()
    # Aborting an unknown or finished request is not an error.
    asyncio.run(client.abort("trtllm-1"))
    asyncio.run(client.abort("unknown"))


def test_closing_the_iterator_cancels_the_request():
    client = FakeEngineClient()

    async def run():
        stream = client.generate(_request())
        await stream.__anext__()
        await stream.aclose()

    asyncio.run(run())
    assert client.cancelled == ["trtllm-1"]
    assert client.active == set()


def test_counters_grow_monotonically_and_show_in_load():
    script_calls = iter([[Reject(ValueError())], None, None])

    def script(request):
        steps = next(script_calls)
        return steps if steps is not None else [OutputFinished(0, "length")]

    client = FakeEngineClient(script=script)
    counters = []
    for i in range(3):
        try:
            asyncio.run(_collect(client, _request(f"r{i}")))
        except ValueError:
            pass
        load = asyncio.run(client.load())
        counters.append(conv.engine_counters(load.attributes))
    assert counters == [(1, 0), (2, 1), (3, 2)]
    assert asyncio.run(FakeEngineClient(report_load=False).load()) is None


def test_health_info_and_presets():
    client = FakeEngineClient.like("glm", health_state="not_ready")
    health = asyncio.run(client.health(deep=True))
    assert health.state == "not_ready" and not health.serving
    assert client.server_info().engine_name == "glm5"
    caps = client.model_info()
    assert caps.max_num_sequences == 1
    assert caps.supports(conv.CAP_STOP_TEXT) and caps.supports(conv.CAP_MIN_P)
    assert not caps.supports(conv.CAP_FREQUENCY_PENALTY)
    assert not any(caps.supports(key) for key in conv.REQUEST_CAPABILITY_KEYS)

    info, trtllm = trtllm_like()
    assert info.extra[conv.CONVENTIONS_VERSION_KEY] == conv.CONVENTIONS_VERSION
    assert trtllm.max_num_sequences == 16
    assert all(trtllm.supports(key) for key in conv.SAMPLING_CAPABILITY_KEYS)
    assert not trtllm.supports(conv.CAP_REQ_BEST_OF)
    assert glm_like()[1].output_logprobs_max_top_n == 20

    client.with_capabilities(max_num_sequences=4)
    assert client.model_info().max_num_sequences == 4


def test_lora_registration():
    client = FakeEngineClient()
    with pytest.raises(NotImplementedError):
        asyncio.run(client.load_lora("a", 1, "/a"))
    client.with_capabilities(supports_lora=True)
    assert asyncio.run(client.load_lora("a", 1, "/a")) is False
    assert asyncio.run(client.load_lora("a", 1, "/a")) is True
    with pytest.raises(ValueError):
        asyncio.run(client.load_lora("a", 2, "/b"))
    assert [a.name for a in asyncio.run(client.list_loras())] == ["a"]


def test_kv_event_sources_default_empty():
    assert asyncio.run(FakeEngineClient().kv_event_sources()) == []


def test_engine_request_defaults_are_not_shared():
    a, b = _request("a"), _request("b")
    a.extra["k"] = 1
    a.stop_token_ids.append(3)
    assert b.extra == {} and b.stop_token_ids == []
    assert dataclasses.replace(a, request_id="c").extra == {"k": 1}


def test_media_spec_needs_exactly_one_source():
    MediaSpec("image", url="http://x")
    with pytest.raises(ValueError):
        MediaSpec("image")
    with pytest.raises(ValueError):
        MediaSpec("image", url="http://x", data_uri="data:,")

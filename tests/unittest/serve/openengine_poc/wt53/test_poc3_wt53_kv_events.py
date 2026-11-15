# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a KV-cache-aware router reads from an OpenEngine frontend (SV-DSG-11, CS-10, CS-11).

- ``/server_info`` states the engine's ``kv_block_size`` as ``tokens_per_block``,
  over the frontend's configured value; without one it keeps that value.
- The frontend subscribes to the engine's KV-cache events when it starts, so
  events published before the first ``/kv_cache_events`` call reach it, and the
  events start with a ``created`` event stating the engine's
  ``total_kv_blocks`` (from its ``ServerInfo``, else its load), which a router
  reads as in-process.
- The block hashes the engine publishes as signed 64-bit integers reach a
  router as the unsigned ones it computes for the same tokens.
"""

from __future__ import annotations

import asyncio
import socket
import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import zmq
from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback, Fuse
from tokenizers.decoders import Sequence as DecoderSequence
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from tensorrt_llm.grpc.openengine.bindings import kv_pb2, server_pb2
from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.openengine_poc import OpenEngineLLM, contracts
from tensorrt_llm.serve.openengine_poc import metrics as metrics_module
from tensorrt_llm.serve.openengine_poc.engine_client import EngineLoad, KvEventSource
from tensorrt_llm.serve.openengine_poc.engine_client_impl import _engine_info
from tensorrt_llm.serve.openengine_poc.facade import adapt_server
from tensorrt_llm.serve.router import KvCacheAwareServerState
from tensorrt_llm.serve.router_utils import BlockHashMixin
from tensorrt_llm.tokenizer import TransformersTokenizer

__extra_import_path__ = ["~/tests/unittest/grpc/openengine_client"]
import fake_openengine  # noqa: E402
from fake_openengine import PROFILES, FakeEngine  # noqa: E402
from openengine_client_helpers import WAIT_SECONDS  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

MODEL = "stub-model"
_VOCAB = {"<unk>": 0, "</s>": 1, "a": 2, "b": 3}
HASH_ALGO = "v2_sha256_64"
ENGINE_BLOCK_SIZE = 64
YAML_BLOCK_SIZE = 32
TOTAL_KV_BLOCKS = 1234


@pytest.fixture(autouse=True)
def _fresh_slots():
    contracts.reset_slot_cache()
    yield
    contracts.reset_slot_cache()


class _SizedEngine(FakeEngine):
    """A fake engine that reports its KV-cache block size and block count, as the TensorRT-LLM server does."""

    def server_info(self) -> server_pb2.ServerInfo:
        info = super().server_info()
        info.capacity.kv_block_size = ENGINE_BLOCK_SIZE
        info.capacity.total_kv_blocks = TOTAL_KV_BLOCKS
        return info


def _load_reports_total_kv_blocks(monkeypatch) -> None:
    """Make every fake engine's GetLoad report ``total_kv_blocks``, as the TensorRT-LLM server can."""
    original = fake_openengine._Control.GetLoad

    async def get_load(self, request, context):
        load = await original(self, request, context)
        load.total_kv_blocks = TOTAL_KV_BLOCKS
        return load

    monkeypatch.setattr(fake_openengine._Control, "GetLoad", get_load)


def _profile(**overrides: Any):
    profile = PROFILES["trtllm-like"]
    return profile.replace(
        model=MODEL,
        max_context_length=64,
        vocab_size=len(_VOCAB),
        eos_token_ids=(1,),
        bos_token_id=None,
        server_info_extra={
            **profile.server_info_extra,
            metrics_module.KV_EVENT_HASH_ALGO_KEY: HASH_ALGO,
        },
        **overrides,
    )


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        backend="pytorch",
        reasoning_parser=None,
        gather_generation_logits=False,
        guided_decoding_backend=None,
        num_postprocess_workers=0,
        cache_transceiver_config=None,
        trust_remote_code=False,
        checkpoint_format=None,
        return_perf_metrics=False,
        perf_metrics_output_dir=None,
        speculative_config=None,
        post_processor_hook=None,
        sleep_config=None,
        num_serve_frontends=1,
        enable_iter_perf_stats=False,
        max_beam_width=1,
        custom_tokenizer=None,
        max_batch_size=None,
        kv_cache_config=KvCacheConfig(tokens_per_block=YAML_BLOCK_SIZE),
        otlp_traces_endpoint=None,
    )


def _tokenizer() -> TransformersTokenizer:
    backend = Tokenizer(WordLevel(_VOCAB, unk_token="<unk>"))
    backend.decoder = DecoderSequence([ByteFallback(), Fuse()])
    backend.add_special_tokens(["</s>"])
    hf = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>")
    hf.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    return TransformersTokenizer(hf)


class _InputProcessor:
    def __call__(self, prompt, sampling_params):
        return [2, 3], None


def _serve(engine: FakeEngine, tmp_path, body) -> Any:
    async def main() -> Any:
        async with engine:
            facade = OpenEngineLLM(
                args=_args(),
                tokenizer=_tokenizer(),
                input_processor=_InputProcessor(),
                hf_model_dir=str(tmp_path),
                endpoint=engine.target,
                ready_timeout=WAIT_SECONDS,
            )
            await facade.start()
            previous_media_executor = BaseMediaIO._executor
            server = OpenAIServer(
                generator=facade,
                model=MODEL,
                tool_parser=None,
                server_role=None,
                metadata_server_cfg=None,
            )
            adapt_server(server, facade)
            BaseMediaIO.set_executor(previous_media_executor)
            server._media_load_executor.shutdown(wait=False)
            try:
                transport = httpx.ASGITransport(app=server.app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                    return await asyncio.wait_for(body(engine, facade, http), WAIT_SECONDS * 4)
            finally:
                await facade.aclose()
                server._input_proc_executor.shutdown(wait=False)
                server._metrics_stats_executor.shutdown(wait=False)

    return asyncio.run(main())


# --- tokens_per_block -------------------------------------------------------------------


def test_engine_info_reads_the_kv_block_size_and_count():
    info = server_pb2.ServerInfo(engine_name="e", instance_id="i", schema_revision=1)
    assert _engine_info(info).kv_block_size is None
    assert _engine_info(info).total_kv_blocks is None
    info.capacity.kv_block_size = ENGINE_BLOCK_SIZE
    info.capacity.total_kv_blocks = TOTAL_KV_BLOCKS
    assert _engine_info(info).kv_block_size == ENGINE_BLOCK_SIZE
    assert _engine_info(info).total_kv_blocks == TOTAL_KV_BLOCKS


@pytest.mark.parametrize(
    "engine_type, expected",
    [(_SizedEngine, ENGINE_BLOCK_SIZE), (FakeEngine, YAML_BLOCK_SIZE)],
    ids=["engine-reports-it", "engine-does-not"],
)
def test_server_info_takes_tokens_per_block_from_the_engine(tmp_path, engine_type, expected):
    async def body(engine, facade, http):
        response = await http.get("/server_info")
        assert response.status_code == 200, response.text
        return response.json()

    content = _serve(engine_type(_profile()), tmp_path, body)
    assert content["tokens_per_block"] == expected


# --- /kv_cache_events ---------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _source(port: int) -> kv_pb2.KvEventSource:
    return kv_pb2.KvEventSource(
        transport="zmq",
        endpoint_addr=kv_pb2.KvEndpoint(host="127.0.0.1", port=port, protocol="tcp"),
        data_parallel_rank=0,
        encoding="msgpack",
    )


@pytest.mark.parametrize("source", ["server-info", "load"])
def test_events_published_before_the_first_call_arrive_after_a_created_event(
    tmp_path, monkeypatch, source
):
    from tensorrt_llm._torch.pyexecutor.kv_cache_events import (
        BlockStored,
        KVEventBatch,
        ZmqEventPublisher,
    )

    if source == "load":
        _load_reports_total_kv_blocks(monkeypatch)
    engine_type = _SizedEngine if source == "server-info" else FakeEngine
    port = _free_port()
    publisher = ZmqEventPublisher(0, endpoint=f"tcp://127.0.0.1:{port}")
    publisher.start()
    engine = engine_type(_profile(kv_event_sources=(_source(port),)))

    async def body(engine, facade, http):
        # Subscribed at start, before any /kv_cache_events call.
        calls = [rpc for rpc, _ in engine.control_calls]
        assert calls.count("GetKvEventSources") == 1
        # The load is read only when the ServerInfo does not report the block count.
        assert ("GetLoad" in calls) == (source == "load")
        (subscriber,) = facade._metrics._kv_events._subscribers
        deadline = time.monotonic() + WAIT_SECONDS
        # The subscription connects in the background: publish until a batch is held.
        while not subscriber.socket.poll(0, zmq.POLLIN):
            assert time.monotonic() < deadline, "the subscription did not connect"
            publisher.publish(
                KVEventBatch(
                    ts=time.time(),
                    events=[
                        BlockStored(
                            block_hashes=[11],
                            parent_block_hash=None,
                            token_ids=[2, 3],
                            block_size=2,
                            lora_id=None,
                            medium="GPU",
                            lora_name=None,
                        )
                    ],
                )
            )
            await asyncio.sleep(0.1)
        first = await http.post("/kv_cache_events")
        second = await http.post("/kv_cache_events")
        assert first.status_code == second.status_code == 200, first.text
        return first.json(), second.json()

    try:
        first, second = _serve(engine, tmp_path, body)
    finally:
        publisher.shutdown()
    created, stored = first[:2]
    assert created == {
        "event_id": 0,
        "data": {"type": "created", "num_blocks_per_cache_level": [TOTAL_KV_BLOCKS]},
        "window_size": None,
        "hash_algo": HASH_ALGO,
    }
    assert stored["event_id"] == 1 and stored["data"]["type"] == "stored"
    assert all(event["data"]["type"] != "created" for event in first[1:] + second)
    event_ids = [event["event_id"] for event in first + second]
    assert event_ids == list(range(len(event_ids)))
    # A KV-cache-aware router takes the events as an in-process worker's.
    state = KvCacheAwareServerState("worker:1")
    state.update_with_events(first)
    assert state.hash_algo == HASH_ALGO
    assert 11 in state._block_table(HASH_ALGO)


def test_an_engine_without_events_answers_no_events_and_is_asked_once(tmp_path):
    async def body(engine, facade, http):
        responses = [await http.post("/kv_cache_events") for _ in range(2)]
        return responses, [rpc for rpc, _ in engine.control_calls]

    responses, calls = _serve(FakeEngine(_profile()), tmp_path, body)
    assert [(r.status_code, r.json()) for r in responses] == [(200, []), (200, [])]
    # Asked at start; the answer is kept.
    assert calls.count("GetKvEventSources") == 1


class _Client:
    def __init__(
        self,
        sources: list[KvEventSource] | None = None,
        sources_error: Exception | None = None,
        load: Any = None,
    ) -> None:
        self.sources = sources or []
        self.sources_error = sources_error
        self.load_result = load
        self.source_calls = 0
        self.load_calls = 0

    def server_info(self):
        return SimpleNamespace(engine_name="e", extra={}, total_kv_blocks=None)

    async def load(self):
        self.load_calls += 1
        if isinstance(self.load_result, BaseException):
            raise self.load_result
        return self.load_result

    async def kv_event_sources(self):
        self.source_calls += 1
        if self.sources_error is not None:
            raise self.sources_error
        return self.sources


def _close(subscription) -> None:
    for subscriber in subscription._subscribers or []:
        subscriber.socket.close()


def _unconnected_source() -> KvEventSource:
    # Nothing publishes there; ZeroMQ connects in the background and keeps trying.
    return KvEventSource(transport="zmq", host="127.0.0.1", port=_free_port(), encoding="msgpack")


def test_a_subscription_that_failed_at_start_is_made_at_a_call_and_starts_with_created():
    from tensorrt_llm.grpc.openengine_client import OpenEngineClientError

    client = _Client(
        sources=[_unconnected_source()],
        sources_error=OpenEngineClientError("down", target="x"),
        load=EngineLoad(instance_id="i", total_kv_blocks=TOTAL_KV_BLOCKS),
    )
    subscription = metrics_module._KvEventSubscription()

    async def run():
        await subscription.subscribe(client)
        client.sources_error = None
        return await subscription.events(client, 0), await subscription.events(client, 0)

    try:
        first, second = asyncio.run(run())
    finally:
        _close(subscription)
    assert [event["data"] for event in first] == [
        {"type": "created", "num_blocks_per_cache_level": [TOTAL_KV_BLOCKS]}
    ]
    assert first[0]["event_id"] == 0 and second == []
    # At start, then at the first call; the load once, when subscribed.
    assert (client.source_calls, client.load_calls) == (2, 1)


def test_a_load_that_cannot_be_read_gives_no_created_event():
    from tensorrt_llm.grpc.openengine_client import OpenEngineClientError

    client = _Client(
        sources=[_unconnected_source()], load=OpenEngineClientError("down", target="x")
    )
    subscription = metrics_module._KvEventSubscription()

    async def run():
        await subscription.subscribe(client)
        return await subscription.events(client, 0)

    try:
        assert asyncio.run(run()) == []
    finally:
        _close(subscription)
    # Subscribed all the same; the next event is numbered from 0.
    assert subscription._subscribers is not None and subscription._next_event_id == 0
    assert client.load_calls == 1


def test_a_call_interrupted_while_reading_the_load_leaves_the_created_event_to_the_next():
    client = _Client(sources=[_unconnected_source()], load=asyncio.CancelledError())
    subscription = metrics_module._KvEventSubscription()

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await subscription.events(client, 0)
        client.load_result = EngineLoad(instance_id="i", total_kv_blocks=TOTAL_KV_BLOCKS)
        return await subscription.events(client, 0)

    try:
        events = asyncio.run(run())
    finally:
        _close(subscription)
    assert [event["data"] for event in events] == [
        {"type": "created", "num_blocks_per_cache_level": [TOTAL_KV_BLOCKS]}
    ]
    assert events[0]["event_id"] == 0
    # Subscribed once; the load read again.
    assert (client.source_calls, client.load_calls) == (1, 2)


def test_a_failed_subscription_at_start_closes_the_connection(tmp_path, monkeypatch):
    from tensorrt_llm.serve.openengine_poc import engine_client_impl

    async def fails(self, client):
        raise RuntimeError("subscription failed")

    monkeypatch.setattr(
        metrics_module.OpenEngineMetricsRecorder, "subscribe_kv_cache_events", fails
    )
    closes = []
    original_close = engine_client_impl.OpenEngineEngineClient.close

    async def close(self):
        closes.append(self)
        await original_close(self)

    monkeypatch.setattr(engine_client_impl.OpenEngineEngineClient, "close", close)
    engine = FakeEngine(_profile())

    async def main():
        async with engine:
            facade = OpenEngineLLM(
                args=_args(),
                tokenizer=_tokenizer(),
                input_processor=_InputProcessor(),
                hf_model_dir=str(tmp_path),
                endpoint=engine.target,
                ready_timeout=WAIT_SECONDS,
            )
            with pytest.raises(RuntimeError, match="subscription failed"):
                await facade.start()
            # As after a failed startup check: the connection is closed, no health watch runs.
            assert len(closes) == 1 and facade._health_task is None
            await facade.aclose()

    asyncio.run(main())


def _signed(block_hash: int) -> int:
    return block_hash - 2**64 if block_hash >= 2**63 else block_hash


def test_published_signed_block_hashes_match_the_ones_a_router_computes():
    tokens_per_block = 4
    tokens = list(range(100, 100 + 8 * tokens_per_block + 1))
    router = SimpleNamespace(_tokens_per_block=tokens_per_block)
    (hashes,) = BlockHashMixin._compute_block_hashes(router, [tokens], HASH_ALGO)
    published = [_signed(h) for h in hashes]
    # The check needs hashes that differ when read as signed: some do, at least one negative.
    assert any(h < 0 for h in published) and published != hashes, hashes
    block_size = tokens_per_block
    batch = [
        1.0,
        [
            {
                "type": "BlockStored",
                "block_hashes": published[1:],
                "parent_block_hash": published[0],
                "token_ids": tokens[block_size : len(hashes) * block_size],
                "block_size": block_size,
                "medium": "GPU",
            },
            {"type": "BlockRemoved", "block_hashes": published[-1:]},
        ],
    ]
    stored, removed = metrics_module.to_in_process_events(
        batch, first_event_id=0, hash_algo=HASH_ALGO, attention_dp_rank=None
    )
    assert [b["block_hash"] for b in stored["data"]["blocks"]] == hashes[1:]
    assert stored["data"]["parent_hash"] == hashes[0]
    assert removed["data"]["block_hashes"] == hashes[-1:]

    state = KvCacheAwareServerState("worker:1", tokens_per_block=tokens_per_block)
    state.update_with_events([stored])
    matched = asyncio.run(state.matched_tokens([hashes[1:]], HASH_ALGO))
    assert matched == (len(hashes) - 1) * tokens_per_block
    state.update_with_events([removed])
    matched = asyncio.run(state.matched_tokens([hashes[1:]], HASH_ALGO))
    assert matched == (len(hashes) - 2) * tokens_per_block

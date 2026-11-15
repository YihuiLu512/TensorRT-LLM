# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the orchestrator does with the workers' KV caches (F-43, DG-19, DG-24, DG-25).

Each test has clusters of its own, which run until the test has ended.

- NT-073: after a request through the orchestrator, the context worker's
  ``/kv_cache_events``, read directly, start with a ``created`` event and
  store the request's blocks under the hashes a router computes for them
  with the algorithm the events name.
- NT-081: two context and two generation workers behind ``kv_cache_aware``
  routers that learn the cached blocks from the workers' events only
  (``track_routed_blocks: false``); after two warm-up requests, requests
  sharing their prefix land on one context worker. A router reads a
  worker's events when a request it routed there ends; the second warm-up
  request caches the prefix on whichever worker it lands on and is read
  there, in case the first read came before the first worker's events.
- NT-086: with ``conditional_disagg_config``, a request whose uncached part
  on the generation worker is short is prefilled there: the context worker
  gets no call and the generation worker serves the whole request. The
  generation router learns the cached blocks from the worker's events only.
- NT-087: the ``kv_hints`` of a request are stripped by default and reach
  the context engine, integers beyond double precision intact, once the
  orchestrator and the workers trust them (``TRTLLM_TRUST_KV_HINTS=1``).

Where the requests went is read from the engines (their Generate counts and
the GenerateRequests their frontends sent), so it is checked on the chain
only. Needs one GPU of Hopper or later and
``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
import requests

from tensorrt_llm.grpc.openengine_conventions import REQ_KV_HINTS
from tensorrt_llm.runtime.kv_cache_hash import KV_CACHE_HASH_ALGO_V1
from tensorrt_llm.serve.router import KvCacheAwareServerState
from tensorrt_llm.serve.router_utils import BlockHashMixin

__extra_import_path__ = ["~/tests/unittest/serve/openengine_poc/wt53"]
from wt53_gpu import (  # noqa: E402
    CHAIN,
    REQUEST_TIMEOUT_S,
    cluster,
    engine_counts,
    model_dir,
    sent_names,
    sent_since,
    server_info,
    worker_config,
)

pytestmark = [pytest.mark.threadleak(enabled=False)]

KV_EVENTS = {
    "free_gpu_memory_fraction": 0.1,
    "enable_block_reuse": True,
    "enable_partial_reuse": True,
    "event_buffer_max_size": 1024,
}
# Long enough for several blocks, and for an uncached part beyond max_local_prefill_length.
PREFIX = "The history of the printing press begins with movable type. " * 30
MAX_LOCAL_PREFILL_LENGTH = 100
NUM_REQUESTS = 6
# Time for a router to read the events of a finished request.
ROUTER_POLL_S = 2
EVENTS_TIMEOUT_S = 60
U64_BLOCK_HASHES = [2**64 - 1, 2**53 + 1, 11]


def _kv_aware() -> dict:
    """A kv_cache_aware router that tokenizes with the model's tokenizer and learns blocks from events only."""
    return {
        "router": {
            "type": "kv_cache_aware",
            "tokenizer_dir": model_dir(),
            "track_routed_blocks": False,
        }
    }


def _events(server) -> list[dict]:
    response = requests.post(server.url_for("kv_cache_events"), timeout=REQUEST_TIMEOUT_S)
    assert response.status_code == 200, response.text
    return response.json()


def _signed(value):
    """A 64-bit hash read as signed; other hashes as they are."""
    if isinstance(value, int) and value >= 2**63:
        return value - 2**64
    return value


@pytest.fixture
def events_cluster(tmp_path):
    with cluster(tmp_path, config=worker_config(kv_cache_config=KV_EVENTS)) as served:
        yield served


def test_kv_cache_events_direct(events_cluster, record_property):
    """NT-073 (F-43): the context worker's events store the request's blocks under a router's hashes."""
    served = events_cluster
    tokens_per_block = server_info(served.ctx)["tokens_per_block"]
    # The engine keys no block with the last token: these are exactly three full blocks.
    full_blocks = 3
    tokens = list(range(1000, 1000 + full_blocks * tokens_per_block + 1))
    response = served.post("v1/completions", {"prompt": tokens, "max_tokens": 4})
    assert response.status_code == 200, response.text

    # The events reach the frontend after the request; read until the request's blocks are stored.
    router = SimpleNamespace(_tokens_per_block=tokens_per_block)
    events: list[dict] = []
    deadline = time.monotonic() + EVENTS_TIMEOUT_S
    while True:
        events += _events(served.ctx)
        state = KvCacheAwareServerState(served.ctx.url_root, tokens_per_block=tokens_per_block)
        state.update_with_events(events)
        algo = state.hash_algo
        (expected,) = BlockHashMixin._compute_block_hashes(router, [tokens], algo)
        stored = {
            block["block_hash"]: block
            for event in events
            if event["data"]["type"] == "stored"
            for block in event["data"]["blocks"]
        }
        if set(expected) <= stored.keys() or time.monotonic() > deadline:
            break
        time.sleep(1)

    assert events and events[0]["data"]["type"] == "created", events[:1]
    assert len(expected) == full_blocks, expected
    matched = asyncio.run(state.matched_tokens([expected], algo))
    # What a reader assuming the default algorithm, or signed hashes, would match.
    (v1_hashes,) = BlockHashMixin._compute_block_hashes(router, [tokens], KV_CACHE_HASH_ALGO_V1)
    readings = {
        "hash_algo": algo,
        "matched_tokens": matched,
        "matched_blocks_as_v1": sum(h in stored for h in v1_hashes),
        "matched_blocks_as_signed": sum(_signed(h) in stored for h in expected),
    }
    for name, value in readings.items():
        record_property(name, value)
    assert matched == full_blocks * tokens_per_block, readings
    stored_tokens = [t["token_id"] for h in expected for t in stored[h]["tokens"]]
    assert stored_tokens == tokens[: full_blocks * tokens_per_block], readings


def _complete(served, prompt, **body):
    response = served.post("v1/completions", {"prompt": prompt, "max_tokens": 4, **body})
    assert response.status_code == 200, response.text
    return response


@pytest.fixture
def routing_cluster(tmp_path):
    with cluster(
        tmp_path,
        counts={"context": 2, "generation": 2},
        config=worker_config(kv_cache_config=KV_EVENTS),
        servers={"context": _kv_aware(), "generation": _kv_aware()},
    ) as served:
        yield served


def test_kv_cache_aware_routing_hits_through_the_orchestrator(routing_cluster):
    """NT-081 (DG-19): requests sharing a prefix land on one context worker."""
    served = routing_cluster
    for i in range(2):
        _complete(served, f"{PREFIX} Warm-up {i}:")
        time.sleep(ROUTER_POLL_S)
    before = engine_counts(served, "context") if CHAIN else None
    for i in range(NUM_REQUESTS):
        _complete(served, f"{PREFIX} Question {i}:")
    if CHAIN:
        landed = [after - b for after, b in zip(engine_counts(served, "context"), before)]
        assert sorted(landed) == [0, NUM_REQUESTS], landed


@pytest.fixture
def conditional_cluster(tmp_path):
    with cluster(
        tmp_path,
        config=worker_config(kv_cache_config=KV_EVENTS),
        orchestrator={
            "conditional_disagg_config": {"max_local_prefill_length": MAX_LOCAL_PREFILL_LENGTH}
        },
        servers={"generation": _kv_aware()},
    ) as served:
        yield served


def test_conditional_disagg_local_prefill_skips_the_ctx_leg(conditional_cluster):
    """NT-086 (DG-24): a request mostly cached on the generation worker is prefilled there."""
    served = conditional_cluster
    ctx_before = engine_counts(served, "context") if CHAIN else None
    _complete(served, f"{PREFIX} Question 0:")
    if not CHAIN:
        _complete(served, f"{PREFIX} Question 1:")
        return
    assert engine_counts(served, "context")[0] - ctx_before[0] == 1
    local = None
    for i in range(1, NUM_REQUESTS):
        time.sleep(ROUTER_POLL_S)
        ctx_before = engine_counts(served, "context")
        gen_before = sent_names(served, "generation")
        _complete(served, f"{PREFIX} Question {i}:")
        if engine_counts(served, "context") == ctx_before:
            (local,) = sent_since(served, "generation", gen_before)
            break
    assert local is not None, "every request went through the context worker"
    # The whole request, not the generation leg of a context worker's prefill.
    assert "kv" not in local, local


def _hints() -> dict:
    return {
        "protocol_version": "0.1",
        "message_id": "nt087-hints",
        "actions": [
            {
                "action_id": "a-1",
                "action_type": "kv.fetch",
                "action_version": "1.0",
                "payload": {"block_hashes": U64_BLOCK_HASHES},
            }
        ],
    }


def _sent_hints(served) -> dict[str, list]:
    """The kv_hints in the GenerateRequests each role's frontend sends for one request, parsed."""
    before = {role: sent_names(served, role) for role in ("context", "generation")}
    _complete(served, "The capital of France is", kv_hints=_hints())
    sent = {}
    for role in ("context", "generation"):
        calls = sent_since(served, role, before[role])
        texts = [call.get("extra", {}).get(REQ_KV_HINTS) for call in calls]
        sent[role] = [None if text is None else json.loads(text) for text in texts]
    return sent


@pytest.fixture
def default_cluster(tmp_path):
    with cluster(tmp_path / "default") as served:
        yield served


@pytest.fixture
def trusted_cluster(tmp_path):
    with cluster(tmp_path / "trusted", env={"TRTLLM_TRUST_KV_HINTS": "1"}) as served:
        yield served


def test_kv_hints_are_stripped_by_default_and_passed_when_trusted(default_cluster, trusted_cluster):
    """NT-087 (DG-25): kv_hints reach no engine by default and the context engine, exact, when trusted."""
    stripped = _sent_hints(default_cluster)
    trusted = _sent_hints(trusted_cluster)
    if not CHAIN:
        return
    assert stripped == {"context": [None], "generation": [None]}, stripped
    assert trusted["context"] == [_hints()], trusted
    assert trusted["generation"] in ([None], [_hints()]), trusted

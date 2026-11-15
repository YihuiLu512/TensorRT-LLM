# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the orchestrator does with the workers' KV caches (F-43, DG-19, DG-24, DG-25).

Each test starts its own cluster and stops it before the next starts.

- NT-073: after a request through the orchestrator, the context worker's
  ``/kv_cache_events``, read directly, start with a ``created`` event and
  store the request's blocks under the hashes a router computes for them
  with the algorithm the events name.
- NT-081: two context and two generation workers behind ``kv_cache_aware``
  routers; requests sharing a prefix land on one context worker.
- NT-086: with ``conditional_disagg_config``, a request whose uncached part
  is short is prefilled on the generation worker: the context worker gets
  no call and the generation worker serves the whole request.
- NT-087: the ``kv_hints`` of a request are stripped by default and reach
  the engine, integers beyond double precision intact, once the
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

__extra_import_path__ = ["~/tests/integration/defs", "~/tests/unittest/serve/openengine_poc/wt53"]
from wt53_gpu import (  # noqa: E402
    CHAIN,
    REQUEST_TIMEOUT_S,
    cluster,
    engine_counts,
    sent_requests,
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
KV_AWARE = {"router": {"type": "kv_cache_aware"}}
# Long enough for several blocks, and for an uncached part beyond max_local_prefill_length.
PREFIX = "The history of the printing press begins with movable type. " * 30
MAX_LOCAL_PREFILL_LENGTH = 100
NUM_REQUESTS = 6
ROUTER_POLL_S = 2
U64_BLOCK_HASHES = [2**64 - 1, 2**53 + 1, 11]


def _events(server) -> list[dict]:
    response = requests.post(server.url_for("kv_cache_events"), timeout=REQUEST_TIMEOUT_S)
    assert response.status_code == 200, response.text
    return response.json()


def _signed(value: int) -> int:
    return value - 2**64 if value >= 2**63 else value


def test_kv_cache_events_direct(tmp_path, record_property):
    """NT-073 (F-43): the context worker's events store the request's blocks under a router's hashes."""
    with cluster(tmp_path, config=worker_config(kv_cache_config=KV_EVENTS)) as served:
        tokens_per_block = server_info(served.ctx)["tokens_per_block"]
        tokens = list(range(1000, 1000 + 3 * tokens_per_block + 5))
        response = served.post("v1/completions", {"prompt": tokens, "max_tokens": 4})
        assert response.status_code == 200, response.text
        events = _events(served.ctx)

    assert events and events[0]["data"]["type"] == "created", events[:1]
    state = KvCacheAwareServerState(served.ctx.url_root, tokens_per_block=tokens_per_block)
    state.update_with_events(events)
    algo = state.hash_algo
    # A router hashes a request as the engine keys its blocks: the last token is in none.
    router = SimpleNamespace(_tokens_per_block=tokens_per_block)
    (expected,) = BlockHashMixin._compute_block_hashes(router, [tokens], algo)
    full_blocks = (len(tokens) - 1) // tokens_per_block
    assert len(expected) == full_blocks, expected
    matched = asyncio.run(state.matched_tokens([expected], algo))
    stored = [b for e in events if e["data"]["type"] == "stored" for b in e["data"]["blocks"]]
    stored_hashes = {block["block_hash"] for block in stored}
    # What a reader assuming the default algorithm, or signed hashes, would match.
    (v1_hashes,) = BlockHashMixin._compute_block_hashes(router, [tokens], KV_CACHE_HASH_ALGO_V1)
    readings = {
        "hash_algo": algo,
        "matched_tokens": matched,
        "matched_blocks_as_v1": sum(h in stored_hashes for h in v1_hashes),
        "matched_blocks_as_signed": sum(_signed(h) in stored_hashes for h in expected),
    }
    for name, value in readings.items():
        record_property(name, value)
    assert matched == full_blocks * tokens_per_block, readings
    stored_tokens = [t["token_id"] for block in stored for t in block["tokens"]]
    assert (
        stored_tokens[: full_blocks * tokens_per_block] == tokens[: full_blocks * tokens_per_block]
    )


def _complete(served, prompt, **body):
    response = served.post("v1/completions", {"prompt": prompt, "max_tokens": 4, **body})
    assert response.status_code == 200, response.text
    return response


def test_kv_cache_aware_routing_hits_through_the_orchestrator(tmp_path):
    """NT-081 (DG-19): requests sharing a prefix land on one context worker."""
    with cluster(
        tmp_path,
        counts={"context": 2, "generation": 2},
        config=worker_config(kv_cache_config=KV_EVENTS),
        servers={"context": KV_AWARE, "generation": KV_AWARE},
    ) as served:
        before = engine_counts(served, "context") if CHAIN else None
        _complete(served, f"{PREFIX} Question 0:")
        # The routers learn the first request's blocks from the workers' events.
        time.sleep(ROUTER_POLL_S)
        for i in range(1, NUM_REQUESTS):
            _complete(served, f"{PREFIX} Question {i}:")
        if CHAIN:
            landed = [after - b for after, b in zip(engine_counts(served, "context"), before)]
            assert sorted(landed) == [0, NUM_REQUESTS], landed


def test_conditional_disagg_local_prefill_skips_the_ctx_leg(tmp_path):
    """NT-086 (DG-24): a request mostly cached on the generation worker is prefilled there."""
    with cluster(
        tmp_path,
        config=worker_config(kv_cache_config=KV_EVENTS),
        orchestrator={
            "conditional_disagg_config": {"max_local_prefill_length": MAX_LOCAL_PREFILL_LENGTH}
        },
        servers={"generation": KV_AWARE},
    ) as served:
        ctx_before = engine_counts(served, "context") if CHAIN else None
        _complete(served, f"{PREFIX} Question 0:")
        if not CHAIN:
            _complete(served, f"{PREFIX} Question 1:")
            return
        assert engine_counts(served, "context")[0] - ctx_before[0] == 1
        # The generation router learns the cached blocks from the worker's events.
        local = None
        for i in range(1, NUM_REQUESTS):
            time.sleep(ROUTER_POLL_S)
            ctx_before = engine_counts(served, "context")
            gen_before = len(sent_requests(served, "generation"))
            _complete(served, f"{PREFIX} Question {i}:")
            if engine_counts(served, "context") == ctx_before:
                (local,) = sent_requests(served, "generation")[gen_before:]
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


def _sent_hints(served) -> dict[str, dict | None]:
    """The kv_hints in the last GenerateRequest each role's frontend sent, parsed, or None."""
    sent = {}
    for role in ("context", "generation"):
        last = sent_requests(served, role)[-1]
        text = last.get("extra", {}).get(REQ_KV_HINTS)
        sent[role] = None if text is None else json.loads(text)
    return sent


def test_kv_hints_are_stripped_by_default_and_passed_when_trusted(tmp_path):
    """NT-087 (DG-25): kv_hints reach no engine by default and the context engine, exact, when trusted."""
    with cluster(tmp_path / "default") as served:
        _complete(served, "The capital of France is", kv_hints=_hints())
        if CHAIN:
            assert _sent_hints(served) == {"context": None, "generation": None}
    with cluster(tmp_path / "trusted", env={"TRTLLM_TRUST_KV_HINTS": "1"}) as served:
        _complete(served, "The capital of France is", kv_hints=_hints())
        if CHAIN:
            sent = _sent_hints(served)
            assert sent["context"] == _hints(), sent
            assert sent["generation"] in (None, _hints()), sent

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A context worker whose KV cache is full waits for the generation worker (EG-30).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU), in-process or on the
OpenEngine chain, with a KV cache of 128 tokens on each worker, as in
``disaggregated/test_disaggregated_single_gpu.py::test_disaggregated_llama_context_capacity``:
the context engine holds the KV cache of a few prompts at most, and a prompt's
KV cache is freed only once the generation worker has taken it.

- NT-206 (EG-30): 32 requests sent at once through the orchestrator all
  complete with every token they asked for. On the chain the context engine
  accepted 32 context-only calls and sent 32 handoffs, and the generation
  engine accepted 32 generation-only calls: no handoff was left unused.
"""

from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest
import requests

from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.grpc.openengine.bindings import openengine_pb2_grpc, server_pb2

from ._test_openengine_disagg_common import CHAIN, REQUEST_TIMEOUT_S, disagg_deployment

pytestmark = pytest.mark.threadleak(enabled=False)

NUM_REQUESTS = 32
MAX_TOKENS = 25
KV_CACHE_TOKENS = 128
# The prompt of that test: the context worker's KV cache holds only a few of them.
PROMPT = (
    "European Union is a political and economic union of 27 countries. The European Union "
    "is headquartered in Brussels, Belgium. The first president of the European Union was "
    "Jean-Claude Juncker. The current president is Ursula von der Leyen. The European Union "
    "is a major economic and political entity."
)
CTX_COUNTERS = (
    conv.COUNTER_GENERATE_ACCEPTED_BUCKETS[conv.BUCKET_CONTEXT_ONLY],
    conv.COUNTER_PREFILL_READY_SENT,
)
GEN_COUNTERS = (conv.COUNTER_GENERATE_ACCEPTED_BUCKETS[conv.BUCKET_GENERATION_ONLY],)


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    with disagg_deployment(
        tmp_path_factory.mktemp("disagg_capacity"),
        worker_config={
            "kv_cache_config": {
                "max_tokens": KV_CACHE_TOKENS,
                "free_gpu_memory_fraction": 0.15,
                "enable_block_reuse": False,
            }
        },
    ) as deployment:
        yield deployment


def _counters(worker, names) -> dict[str, int]:
    with grpc.insecure_channel(worker._chain_handle.engine_endpoint) as channel:
        load = openengine_pb2_grpc.ControlStub(channel).GetLoad(
            server_pb2.GetLoadRequest(), timeout=30
        )
    attributes = dict(load.attributes)
    values = {name: conv.as_int(attributes.get(name)) for name in names}
    assert None not in values.values(), values
    return values


def _engine_counts(deployment) -> dict[str, int]:
    return {**_counters(deployment.ctx, CTX_COUNTERS), **_counters(deployment.gen, GEN_COUNTERS)}


def _complete(deployment) -> requests.Response:
    body = {
        "model": deployment.model,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,
        "temperature": 0.0,
    }
    return requests.post(
        f"{deployment.url}/v1/completions", json=body, timeout=REQUEST_TIMEOUT_S * 4
    )


def test_context_capacity_waits_for_generation(deployment):
    before = _engine_counts(deployment) if CHAIN else None
    with ThreadPoolExecutor(max_workers=NUM_REQUESTS) as pool:
        responses = list(pool.map(lambda _: _complete(deployment), range(NUM_REQUESTS)))
    for response in responses:
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["completion_tokens"] == MAX_TOKENS, response.text
    if CHAIN:
        after = _engine_counts(deployment)
        assert {name: after[name] - before[name] for name in after} == dict.fromkeys(
            after, NUM_REQUESTS
        ), (before, after)

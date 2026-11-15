# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A disaggregated worker's ``/server_info``, read directly from each worker (F-06, DG-89).

A context and a generation worker serve KV-cache events and are started with
``--max_batch_size``, which on the chain only the engine gets; neither test
sends a generation request.

- NT-055: each worker answers ``/server_info`` with the fields a router reads.
- NT-186 (chain only): the fields a router sizes and hashes blocks with come
  from the engine's ``GetServerInfo``, not from the frontend's own
  configuration: ``max_batch_size`` is the engine's (the frontend was not
  given the option), ``kv_cache_hash_algo`` the algorithm the engine declares
  for its KV-cache events (the frontend's own configuration gives the default
  one), ``tokens_per_block`` the engine's block size, and the context worker's
  ``ctx_info_endpoint`` the first endpoint of the engine's KV connector.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import pytest

from tensorrt_llm.runtime.kv_cache_hash import KV_CACHE_HASH_ALGO_DEFAULT

__extra_import_path__ = ["~/tests/integration/defs", "~/tests/unittest/serve/openengine_poc/wt53"]
from wt53_gpu import (  # noqa: E402
    chain_only,
    cluster,
    engine_server_info,
    server_info,
    worker_config,
)

pytestmark = [pytest.mark.threadleak(enabled=False)]

MAX_BATCH_SIZE = 8
# The engine's ServerInfo.extra key naming the algorithm of its KV-cache event hashes.
HASH_ALGO_KEY = "trtllm_kv_event_hash_algo"
# What a router reads from a worker's /server_info.
ROUTER_FIELDS = {"disaggregated_params", "max_batch_size", "kv_cache_hash_algo", "tokens_per_block"}


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    config = worker_config(
        kv_cache_config={
            "free_gpu_memory_fraction": 0.1,
            "enable_block_reuse": True,
            "event_buffer_max_size": 1024,
        }
    )
    with cluster(
        tmp_path_factory.mktemp("server_info"),
        config=config,
        args=["--max_batch_size", str(MAX_BATCH_SIZE)],
    ) as result:
        yield result


def test_server_info_direct(served):
    """NT-055 (F-06): each worker answers /server_info with the fields a router reads."""
    for role, servers in served.workers.items():
        info = server_info(servers[0])
        assert ROUTER_FIELDS <= info.keys(), (role, info)
        assert info["max_batch_size"] == MAX_BATCH_SIZE, (role, info)
        assert info["tokens_per_block"] > 0, (role, info)
        if role == "context":
            assert info["disaggregated_params"]["ctx_info_endpoint"], info


def _endpoint_text(endpoint, transfer_backend: str) -> str:
    """``host:port`` of a KV connector endpoint, with its scheme unless that is the backend's name."""
    host = f"[{endpoint.host}]" if ":" in endpoint.host else endpoint.host
    address = f"{host}:{endpoint.port}"
    if endpoint.protocol and endpoint.protocol != transfer_backend:
        return f"{endpoint.protocol}://{address}"
    return address


@chain_only
def test_disagg_fields_come_from_the_engine(served):
    """NT-186 (DG-89): /server_info takes the router's fields from the engine's ServerInfo."""
    for role, servers in served.workers.items():
        info = server_info(servers[0])
        engine = engine_server_info(servers[0])
        assert engine.capacity.max_running_requests == MAX_BATCH_SIZE, engine
        assert info["max_batch_size"] == engine.capacity.max_running_requests, (role, info)
        assert info["tokens_per_block"] == engine.capacity.kv_block_size, (role, info, engine)
        engine_algo = engine.extra[HASH_ALGO_KEY]
        # The frontend's own configuration would give the default algorithm.
        assert engine_algo != KV_CACHE_HASH_ALGO_DEFAULT, engine
        assert info["kv_cache_hash_algo"] == engine_algo, (role, info)
        if role == "context":
            connector = engine.kv_connector
            assert connector.local_endpoints, engine
            expected = _endpoint_text(connector.local_endpoints[0], connector.transfer_backend)
            assert info["disaggregated_params"]["ctx_info_endpoint"] == [expected], info

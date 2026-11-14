# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A real engine without a KV cache transceiver refuses disaggregated requests and lives on.

One OpenEngine server (``trtllm-serve --grpc --grpc-protocol openengine``) without
``cache_transceiver_config`` gets, over gRPC and with no front end, a ``context_only``
request and a request with a ``kv.session``. Each is refused with ``FAILED_PRECONDITION``
before it reaches the engine; afterwards the server still reports ``SERVING`` and serves
an aggregated request.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import os

import grpc
import pytest
import torch
from grpc_health.v1 import health_pb2, health_pb2_grpc

__extra_import_path__ = ["../..", "~/tests/integration/defs"]
from openengine_chain import direct_replay as dr
from utils.llm_data import llm_models_root

from tensorrt_llm.grpc.openengine.bindings import kv_pb2
from tensorrt_llm.grpc.openengine_conventions import (
    COUNTER_GENERATE_ACCEPTED,
    COUNTER_GENERATE_RECEIVED,
)

pytestmark = [pytest.mark.threadleak(enabled=False)]

MODEL_SUBDIR = os.path.join("Qwen3", "Qwen3-0.6B")
MODEL_NAME = "Qwen3/Qwen3-0.6B"
PROMPT = "The capital of France is Paris. The capital of Germany is"
ENGINE_READY_TIMEOUT_S = 1200
REQUEST_TIMEOUT_S = 120
CONFIG = {"kv_cache_config": {"free_gpu_memory_fraction": 0.2}}


def _model_dir() -> str:
    root = llm_models_root()
    path = None if root is None else os.path.join(str(root), MODEL_SUBDIR)
    assert path is not None and os.path.isdir(path), f"{MODEL_SUBDIR} is not under LLM_MODELS_ROOT"
    return path


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (9, 0):
        pytest.skip("needs a GPU of Hopper or later")
    workdir = str(tmp_path_factory.mktemp("no_transceiver"))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    started = dr.start_engine("plain", _model_dir(), workdir, config=CONFIG, env=env)
    try:
        started.wait_serving(ENGINE_READY_TIMEOUT_S)
        yield started
    finally:
        started.stop()


def _health(endpoint: str) -> int:
    with grpc.insecure_channel(endpoint) as channel:
        reply = health_pb2_grpc.HealthStub(channel).Check(
            health_pb2.HealthCheckRequest(service="inference"), timeout=30
        )
    return reply.status


def _counters(endpoint: str) -> dict[str, float]:
    return dr.get_load(endpoint).get("attributes", {})


def _session() -> kv_pb2.KvSessionRef:
    session = kv_pb2.KvSessionRef(session_id="42")
    session.attributes_struct.update(
        {"ctx_info_endpoint": "tcp://127.0.0.1:1", "first_gen_tokens": [7]}
    )
    return session


@pytest.mark.parametrize(
    "fields",
    [{"extra": dr.CONTEXT_ONLY_EXTRA}, {"session": _session()}],
    ids=["context_only", "kv_session"],
)
def test_disagg_request_to_an_engine_without_a_transceiver_is_failed_precondition(engine, fields):
    assert dr.server_info(engine.endpoint).get("kv_connector") is None
    before = _counters(engine.endpoint)

    leg = dr.generate(
        engine.endpoint,
        dr.build_request(MODEL_NAME, PROMPT, max_tokens=4, **fields),
        timeout=REQUEST_TIMEOUT_S,
    )

    assert leg.status == "FAILED_PRECONDITION", leg.to_json()
    assert "transceiver" in leg.details, leg.to_json()
    assert leg.events == [], leg.to_json()
    after = _counters(engine.endpoint)
    assert after[COUNTER_GENERATE_RECEIVED] - before[COUNTER_GENERATE_RECEIVED] == 1
    assert after[COUNTER_GENERATE_ACCEPTED] == before[COUNTER_GENERATE_ACCEPTED]

    # The engine is untouched: still serving, and an aggregated request completes.
    assert engine.poll() is None
    assert _health(engine.endpoint) == health_pb2.HealthCheckResponse.SERVING
    aggregated = dr.generate(
        engine.endpoint,
        dr.build_request(MODEL_NAME, PROMPT, max_tokens=4),
        timeout=REQUEST_TIMEOUT_S,
    )
    assert aggregated.status == "OK", aggregated.to_json()
    assert aggregated.terminal() == "finished", aggregated.to_json()
    assert len(aggregated.token_ids()) == 4, aggregated.to_json()

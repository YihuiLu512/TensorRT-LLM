# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-level scheduling options of a disaggregated request reach both engines (F-44a to F-44c).

The deployments are ``_test_openengine_disagg_common.disagg_deployment``
(Qwen3-0.6B, one context and one generation worker on one GPU), in-process or
on the OpenEngine chain. Each request goes through the orchestrator, so the
context and the generation leg each make one Generate call. On the chain the
tests read the two legs' GenerateRequests
(``_test_openengine_disagg_common.legs``).

- NT-074 (F-44a): a request with ``priority`` is served by engines that order
  their waiting requests by priority; on the chain both engines declare
  priority scheduling and accept their leg. ``priority`` travels as gRPC
  metadata, which the dumps do not hold.
- NT-075 (F-44b): a chat request with ``agent_hierarchy`` is served; on the
  chain both legs' GenerateRequests carry it in ``trtllm.scheduling``. Its
  workers run the agent-tree reorder policy with attention DP and the V1 KV
  cache manager, which the engine needs before it declares the option.
- NT-076 (F-44c): a request with ``conversation_params.conversation_id`` is
  served; on the chain both legs' GenerateRequests carry the same id.
"""

from pathlib import Path

import grpc
import pytest
import requests

from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.grpc.openengine.bindings import model_pb2, openengine_pb2_grpc, server_pb2

from ._test_openengine_disagg_common import (
    CHAIN,
    DUMP_ENV,
    REQUEST_TIMEOUT_S,
    ROLES,
    disagg_deployment,
    dump_names,
    legs,
)

pytestmark = pytest.mark.threadleak(enabled=False)

PROMPT = "The capital of France is"
QUESTION = [{"role": "user", "content": "What is 1 + 1?"}]
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
# The Generate calls each role's engine accepts for one disaggregated request.
BUCKETS = {"context": conv.BUCKET_CONTEXT_ONLY, "generation": conv.BUCKET_GENERATION_ONLY}
# One of the node types the C++ agent tree registers; an unknown name stops the workers
# from starting.
AGENT_TYPE = "AgentDeepResearch"
# Read only by the agent-tree policy, which the C++ capacity scheduler installs for
# a V1 KV cache manager; the engine declares trtllm.request.scheduling only with
# attention DP as well. A V1 KV cache manager needs the C++ cache transceiver.
AGENT_TREE_CONFIG = {
    "enable_attention_dp": True,
    "cache_transceiver_config": {"backend": "DEFAULT", "transceiver_runtime": "CPP"},
    "reorder_policy_config": {
        "policy_name": "AgentTree",
        "policy_args": {"agent_types": [AGENT_TYPE]},
    },
    "kv_cache_config": {
        "free_gpu_memory_fraction": 0.15,
        "enable_block_reuse": False,
        "use_kv_cache_manager_v2": False,
    },
}


@pytest.fixture(scope="module")
def dump_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("disagg_scheduling_dumps")


@pytest.fixture(scope="module")
def deployment(tmp_path_factory, dump_dir):
    with disagg_deployment(
        tmp_path_factory.mktemp("disagg_scheduling"),
        worker_config={"scheduler_config": {"waiting_queue_policy": "priority"}},
        env={DUMP_ENV: str(dump_dir)},
    ) as deployment:
        yield deployment


# Function-scoped, and its test runs first: it stops before the module's deployment starts.
@pytest.fixture
def agent_tree_deployment(tmp_path_factory, dump_dir):
    with disagg_deployment(
        tmp_path_factory.mktemp("disagg_agent_tree"),
        worker_config=AGENT_TREE_CONFIG,
        env={DUMP_ENV: str(dump_dir)},
    ) as deployment:
        yield deployment


def _engine_endpoints(deployment) -> dict[str, str]:
    """Each role's engine endpoint on the chain."""
    return {role: deployment.workers[role][0]._chain_handle.engine_endpoint for role in ROLES}


def _capabilities(endpoint: str) -> conv.ModelCapabilities:
    with grpc.insecure_channel(endpoint) as channel:
        control = openengine_pb2_grpc.ControlStub(channel)
        return conv.model_capabilities_from_proto(
            control.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=30),
            control.GetModelInfo(model_pb2.GetModelInfoRequest(), timeout=30),
        )


def _accepted(endpoint: str, bucket: str) -> int:
    """The engine's count of accepted Generate calls of ``bucket``."""
    with grpc.insecure_channel(endpoint) as channel:
        load = openengine_pb2_grpc.ControlStub(channel).GetLoad(
            server_pb2.GetLoadRequest(), timeout=30
        )
    value = conv.as_int(dict(load.attributes).get(conv.COUNTER_GENERATE_ACCEPTED_BUCKETS[bucket]))
    assert value is not None, f"the engine at {endpoint} reports no {bucket} count"
    return value


def _accepted_by_role(deployment) -> dict[str, int]:
    endpoints = _engine_endpoints(deployment)
    return {role: _accepted(endpoints[role], BUCKETS[role]) for role in ROLES}


def _post(deployment, path: str, body: dict) -> requests.Response:
    return requests.post(
        f"{deployment.url}/{path}",
        json={"model": deployment.model, **body},
        timeout=REQUEST_TIMEOUT_S,
    )


def _send(deployment, dump_dir: Path, path: str, body: dict) -> tuple[dict, dict] | None:
    """Send ``body`` through the orchestrator; on the chain, the two legs' GenerateRequests."""
    before = dump_names(dump_dir)
    accepted = _accepted_by_role(deployment) if CHAIN else None
    response = _post(deployment, path, body)
    assert response.status_code == 200, response.text
    assert response.json()["choices"], response.text
    if not CHAIN:
        return None
    after = _accepted_by_role(deployment)
    for role in ROLES:
        assert after[role] == accepted[role] + 1, (role, accepted, after)
    return legs(dump_dir, before)


def test_agent_hierarchy_reaches_both_engines(agent_tree_deployment, dump_dir):
    deployment = agent_tree_deployment
    if CHAIN:
        for role, endpoint in _engine_endpoints(deployment).items():
            assert _capabilities(endpoint).supports(conv.CAP_REQ_SCHEDULING), role
    sent_legs = _send(
        deployment,
        dump_dir,
        "v1/chat/completions",
        {
            "messages": QUESTION,
            "max_completion_tokens": 8,
            "agent_hierarchy": [[AGENT_TYPE, 1]],
            **NO_THINKING,
        },
    )
    if sent_legs is not None:
        for leg in sent_legs:
            scheduling = leg["extra"].get(conv.REQ_SCHEDULING, {})
            hierarchy = [
                (kind, conv.as_int(node)) for kind, node in scheduling.get("agent_hierarchy", [])
            ]
            assert hierarchy == [(AGENT_TYPE, 1)], leg


def test_priority_reaches_both_engines(deployment, dump_dir):
    if CHAIN:
        for role, endpoint in _engine_endpoints(deployment).items():
            assert _capabilities(endpoint).supports_priority, role
    _send(
        deployment, dump_dir, "v1/completions", {"prompt": PROMPT, "max_tokens": 4, "priority": 0.9}
    )


def test_conversation_id_reaches_both_engines(deployment, dump_dir):
    conversation_id = "nt076-conversation"
    sent_legs = _send(
        deployment,
        dump_dir,
        "v1/completions",
        {
            "prompt": PROMPT,
            "max_tokens": 4,
            "conversation_params": {"conversation_id": conversation_id},
        },
    )
    if sent_legs is not None:
        for leg in sent_legs:
            assert leg["extra"].get(conv.REQ_CONVERSATION_ID) == conversation_id, leg

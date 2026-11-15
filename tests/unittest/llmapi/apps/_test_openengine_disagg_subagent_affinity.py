# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sub-agent requests follow their parent's placement through the orchestrator (DG-22, DG-35).

Two context workers and a generation worker behind an orchestrator that
routes by conversation and reads the parent session from
``X-Dynamo-Parent-Session-ID``. The context workers run conversation-aware
attention DP, so on the chain their engines check the signed parent id
(``tensorrt_llm/grpc/openengine/affinity.py``); the key is the cluster's
``internal_request_auth_key``, which every worker frontend and engine gets.

- NT-083: with ``subagent_affinity_scope: context`` the sub-agents' requests
  succeed and their context legs land on the parent's context worker; on the
  chain the context engines declare that they check the hint and none
  refuses it.
- NT-084: with ``subagent_affinity_scope: both`` (a second orchestrator) the
  requests succeed; on the chain the generation frontend accepts the signed
  hint and does not send it to its engine, which serves context requests
  only with one.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import pytest
import requests

__extra_import_path__ = ["~/tests/integration/defs", "~/tests/unittest/serve/openengine_poc/wt53"]
from wt53_gpu import (  # noqa: E402
    CHAIN,
    MODEL_NAME,
    REQUEST_TIMEOUT_S,
    another_orchestrator,
    chain_log,
    cluster,
    engine_counts,
    engine_server_info,
)

pytestmark = [pytest.mark.threadleak(enabled=False)]

PARENT_HEADER = "X-Dynamo-Parent-Session-ID"
NUM_CHILDREN = 6
SUPPORTS_AFFINITY_KEY = "trtllm_supports_subagent_affinity"
# What a frontend or an engine logs when it does not take a hint.
NOT_DECLARED_LOG = f"does not declare {SUPPORTS_AFFINITY_KEY}"
REFUSED_LOGS = ("Subagent affinity", "subagent affinity authentication")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    conversation_router = {"router": {"type": "conversation"}}
    with cluster(
        tmp_path_factory.mktemp("subagent_affinity"),
        counts={"context": 2},
        role_config={
            "context": {
                "enable_attention_dp": True,
                "attention_dp_config": {"kv_cache_routing_conversation_affinity": True},
            }
        },
        orchestrator={
            "conversation_affinity_header_for_subagents": PARENT_HEADER,
            "subagent_affinity_scope": "context",
        },
        servers={"context": conversation_router, "generation": conversation_router},
    ) as result:
        yield result


def _chat(url_root: str, session: str, parent: str | None = None) -> None:
    headers = {"x-session-id": session}
    if parent is not None:
        headers[PARENT_HEADER] = parent
    response = requests.post(
        f"{url_root}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": f"Agent {session}: name a colour."}],
            "max_tokens": 4,
            "temperature": 0.0,
        },
        headers=headers,
        timeout=REQUEST_TIMEOUT_S,
    )
    assert response.status_code == 200, response.text


def _family(url_root: str, parent: str) -> None:
    """A parent's request, then a request of each of its sub-agents."""
    _chat(url_root, parent)
    for i in range(NUM_CHILDREN):
        _chat(url_root, f"{parent}-child-{i}", parent)


def _assert_no_refusal(servers, sides=("frontend", "engine")) -> None:
    for server in servers:
        for side in sides:
            log = chain_log(server, side)
            for text in REFUSED_LOGS:
                assert text not in log, (side, text)


def test_subagent_affinity_scope_context(served):
    """NT-083 (DG-22, DG-35): sub-agents' context legs land on the parent's context worker."""
    before = engine_counts(served, "context") if CHAIN else None
    _family(served.orchestrator.url_root, "parent-context")
    if not CHAIN:
        return
    landed = [after - b for after, b in zip(engine_counts(served, "context"), before)]
    assert sorted(landed) == [0, 1 + NUM_CHILDREN], landed
    for server in served.workers["context"]:
        assert engine_server_info(server).extra[SUPPORTS_AFFINITY_KEY] is True
        assert NOT_DECLARED_LOG not in chain_log(server, "frontend")
    _assert_no_refusal(served.workers["context"])


def test_subagent_affinity_scope_both_is_not_forwarded_to_the_gen_engine(served, tmp_path):
    """NT-084 (DG-22, DG-35): with scope both the generation leg succeeds and its engine gets no hint."""
    log_path = tmp_path / "output_orchestrator_scope_both.log"
    with another_orchestrator(
        served, log_path, orchestrator={"subagent_affinity_scope": "both"}
    ) as orchestrator:
        _family(orchestrator.url_root, "parent-both")
    if CHAIN:
        _assert_no_refusal(served.workers["generation"])
        _assert_no_refusal(served.workers["context"])

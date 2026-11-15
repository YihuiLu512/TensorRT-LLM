# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sub-agent requests follow their parent's placement through the orchestrator (DG-22, DG-35).

Two context workers and a generation worker behind an orchestrator that
routes by conversation and reads the parent session from
``X-Dynamo-Parent-Session-ID``. The workers run conversation-aware attention
DP, so on the chain their engines check the signed parent id
(``tensorrt_llm/grpc/openengine/affinity.py``); the key is the cluster's
``internal_request_auth_key``, which every worker frontend and engine gets.

- NT-083: with ``subagent_affinity_scope: context`` the sub-agents' requests
  succeed. On the chain their context legs land on the parent's context
  worker, the context engines declare that they check the hint, the context
  calls carry the conversation id the frontend signs the hint for, and no
  engine refuses it. That the frontend sent the hint is inferred from these:
  it leaves it out only when the engine does not declare the check, when the
  call has no conversation id, or when the parent id is not printable ASCII.
- NT-084: with ``subagent_affinity_scope: both`` (a second orchestrator) the
  requests succeed. On the chain, where the generation engine also declares
  that it checks the hint, this is the evidence that the generation
  frontend did not send the hint to its engine: the engine refuses a
  generation call that carries one ("supported on context requests only"),
  which would fail the request; the hint travels as gRPC metadata, which the
  frontend's request dumps do not hold. Neither engine logs a refusal.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import pytest
import requests

from tensorrt_llm.grpc.openengine_conventions import REQ_CONVERSATION_ID

__extra_import_path__ = ["~/tests/unittest/serve/openengine_poc/wt53"]
from wt53_gpu import (  # noqa: E402
    CHAIN,
    MODEL_NAME,
    REQUEST_TIMEOUT_S,
    another_orchestrator,
    chain_log,
    chain_log_size,
    cluster,
    engine_counts,
    engine_server_info,
    sent_names,
    sent_since,
)

pytestmark = [pytest.mark.threadleak(enabled=False)]

PARENT_HEADER = "X-Dynamo-Parent-Session-ID"
NUM_CHILDREN = 6
SUPPORTS_AFFINITY_KEY = "trtllm_supports_subagent_affinity"
# What a frontend logs when its engine does not declare that it checks the hint.
NOT_DECLARED_LOG = f"does not declare {SUPPORTS_AFFINITY_KEY}"
REFUSED_LOGS = ("Subagent affinity", "subagent affinity authentication")
CONVERSATION_ADP = {
    "enable_attention_dp": True,
    "attention_dp_config": {"kv_cache_routing_conversation_affinity": True},
}


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    with cluster(
        tmp_path_factory.mktemp("subagent_affinity"),
        counts={"context": 2},
        role_config={role: CONVERSATION_ADP for role in ("context", "generation")},
        orchestrator={
            "conversation_affinity_header_for_subagents": PARENT_HEADER,
            "subagent_affinity_scope": "context",
        },
        servers={
            "context": {"router": {"type": "conversation"}},
            "generation": {"router": {"type": "conversation"}},
        },
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


def _log_sizes(served) -> dict:
    """Where each chain worker's logs end now, so a test reads only what it caused."""
    return {
        (id(server), side): chain_log_size(server, side)
        for servers in served.workers.values()
        for server in servers
        for side in ("frontend", "engine")
    }


def _assert_no_refusal(servers, sizes) -> None:
    for server in servers:
        for side in ("frontend", "engine"):
            log = chain_log(server, side, sizes[(id(server), side)])
            for text in REFUSED_LOGS:
                assert text not in log, (side, text)


def test_subagent_affinity_scope_context(served):
    """NT-083 (DG-22, DG-35): sub-agents' context legs land on the parent's context worker."""
    if CHAIN:
        sizes = _log_sizes(served)
        before = engine_counts(served, "context")
        names = [sent_names(served, "context", i) for i in range(len(served.workers["context"]))]
    _family(served.orchestrator.url_root, "parent-context")
    if not CHAIN:
        return
    landed = [after - b for after, b in zip(engine_counts(served, "context"), before)]
    assert sorted(landed) == [0, 1 + NUM_CHILDREN], landed
    for i, server in enumerate(served.workers["context"]):
        assert engine_server_info(server).extra[SUPPORTS_AFFINITY_KEY] is True
        assert NOT_DECLARED_LOG not in chain_log(server, "frontend")
        for call in sent_since(served, "context", names[i], i):
            assert call.get("extra", {}).get(REQ_CONVERSATION_ID), call
    _assert_no_refusal(served.workers["context"], sizes)


@pytest.fixture
def scope_both(served, tmp_path):
    log_path = tmp_path / "output_orchestrator_scope_both.log"
    with another_orchestrator(
        served, log_path, orchestrator={"subagent_affinity_scope": "both"}
    ) as orchestrator:
        yield orchestrator


def test_subagent_affinity_scope_both_is_not_forwarded_to_the_gen_engine(served, scope_both):
    """NT-084 (DG-22, DG-35): with scope both the generation leg succeeds and its engine gets no hint."""
    sizes = _log_sizes(served) if CHAIN else None
    _family(scope_both.url_root, "parent-both")
    if CHAIN:
        (gen,) = served.workers["generation"]
        assert engine_server_info(gen).extra[SUPPORTS_AFFINITY_KEY] is True
        _assert_no_refusal(served.workers["generation"], sizes)
        _assert_no_refusal(served.workers["context"], sizes)

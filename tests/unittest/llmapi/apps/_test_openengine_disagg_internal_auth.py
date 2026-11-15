# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A generation worker refuses a generation-only request whose handoff is not signed (DG-34, DG-37).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU), in-process or on the
OpenEngine chain; it gives the workers and the orchestrator one
``internal_request_auth_key``. The test sends both legs to the workers itself,
as the orchestrator would, not through the orchestrator.

- NT-088 (DG-34, DG-37): the handoff of a real context-only response, sent to
  the generation worker with a signature that does not match it or with none,
  is refused with 400 before any Generate call (on the chain the generation
  engine receives none). Sent with the signature the orchestrator would add,
  the same handoff is served.
"""

import grpc
import pytest
import requests

from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.grpc.openengine.bindings import openengine_pb2_grpc, server_pb2
from tensorrt_llm.llmapi.disagg_utils import get_global_disagg_request_id
from tensorrt_llm.serve.disagg_auth import (
    INTERNAL_DISAGG_AUTH_HEADER,
    build_internal_disagg_auth_headers,
)
from tensorrt_llm.serve.openai_protocol import CompletionRequest

from ._test_openengine_disagg_common import CHAIN, REQUEST_TIMEOUT_S, disagg_deployment

pytestmark = pytest.mark.threadleak(enabled=False)

PROMPT = "The capital of France is"
MAX_TOKENS = 8
FORGED_SIGNATURE = "sha256=" + "0" * 64


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    with disagg_deployment(tmp_path_factory.mktemp("disagg_internal_auth")) as deployment:
        yield deployment


def _generate_received(endpoint: str) -> int:
    with grpc.insecure_channel(endpoint) as channel:
        load = openengine_pb2_grpc.ControlStub(channel).GetLoad(
            server_pb2.GetLoadRequest(), timeout=30
        )
    counters = conv.engine_counters(dict(load.attributes))
    assert counters is not None, f"the engine at {endpoint} reports no Generate counters"
    return counters[0]


def _post(worker, body: dict, headers: dict | None = None) -> requests.Response:
    return requests.post(
        worker.url_for("v1/completions"), json=body, headers=headers, timeout=REQUEST_TIMEOUT_S
    )


def test_gen_worker_frontend_rejects_a_forged_generation_only_request(deployment):
    request = {"model": deployment.model, "prompt": PROMPT, "max_tokens": MAX_TOKENS}
    ctx_response = _post(
        deployment.ctx,
        {
            **request,
            "disaggregated_params": {
                "request_type": "context_only",
                "disagg_request_id": get_global_disagg_request_id(node_id=0),
            },
        },
    )
    assert ctx_response.status_code == 200, ctx_response.text
    [choice] = ctx_response.json()["choices"]
    handoff = {**choice["disaggregated_params"], "request_type": "generation_only"}
    gen_request = {**request, "disaggregated_params": handoff}
    auth_key = deployment.orchestrator.disagg_config["internal_request_auth_key"]
    signed = build_internal_disagg_auth_headers(
        auth_key, CompletionRequest.model_validate(gen_request)
    )
    assert INTERNAL_DISAGG_AUTH_HEADER in signed, "the handoff has no field the key protects"

    gen_engine = deployment.gen._chain_handle.engine_endpoint if CHAIN else None
    received = _generate_received(gen_engine) if CHAIN else None
    for name, headers in (
        ("forged", {INTERNAL_DISAGG_AUTH_HEADER: FORGED_SIGNATURE}),
        ("unsigned", None),
    ):
        response = _post(deployment.gen, gen_request, headers)
        assert response.status_code == 400, (name, response.status_code, response.text)
        assert "authentication" in response.text, (name, response.text)
    if CHAIN:
        assert _generate_received(gen_engine) == received, "a refused handoff reached the engine"

    # The same handoff with the orchestrator's signature is served, and frees the
    # context worker's KV cache.
    response = _post(deployment.gen, gen_request, signed)
    assert response.status_code == 200, response.text
    assert response.json()["usage"]["completion_tokens"] > 0, response.text
    if CHAIN:
        assert _generate_received(gen_engine) == received + 1

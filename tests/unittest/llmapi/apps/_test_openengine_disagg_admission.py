# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How long a disaggregated request has waited reaches the generation engine (EG-21).

NT-201 (chain only): the orchestrator stamps the generation leg of a request
with ``orig_elapsed_s``, the seconds since the request arrived; the
generation worker frontend sends it to its engine in the GenerateRequest's
``extra`` (EX-DSG-3), which the engine can order its admission by. The
context leg carries none. The evidence is the GenerateRequests the worker
frontends wrote as they sent them (``TRTLLM_OPENENGINE_REQUEST_DUMP_DIR``);
the chain proof shows each one reached its engine. The orchestrator's own
value is not readable, so the value is checked against the time the request
took as the test saw it.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import time

import pytest

from tensorrt_llm.grpc.openengine_conventions import REQ_ORIG_ELAPSED_S

__extra_import_path__ = ["~/tests/unittest/serve/openengine_poc/wt53"]
from wt53_gpu import chain_only, cluster, sent_names, sent_since  # noqa: E402

pytestmark = [pytest.mark.threadleak(enabled=False)]


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    with cluster(tmp_path_factory.mktemp("admission")) as result:
        yield result


@chain_only
def test_orig_elapsed_s_reaches_the_generation_engine(served):
    """NT-201 (EG-21): the generation leg's GenerateRequest carries orig_elapsed_s, the context leg's none."""
    ctx_before = sent_names(served, "context")
    gen_before = sent_names(served, "generation")
    started = time.monotonic()
    response = served.post(
        "v1/completions", {"prompt": "The capital of France is", "max_tokens": 4}
    )
    took = time.monotonic() - started
    assert response.status_code == 200, response.text

    (ctx_call,) = sent_since(served, "context", ctx_before)
    (gen_call,) = sent_since(served, "generation", gen_before)
    assert REQ_ORIG_ELAPSED_S not in ctx_call.get("extra", {}), ctx_call
    assert "session" in gen_call.get("kv", {}), gen_call
    elapsed = gen_call.get("extra", {}).get(REQ_ORIG_ELAPSED_S)
    assert isinstance(elapsed, float), gen_call
    # Stamped after the context leg ended, within the time the request took.
    assert 0 < elapsed <= took, (elapsed, took)

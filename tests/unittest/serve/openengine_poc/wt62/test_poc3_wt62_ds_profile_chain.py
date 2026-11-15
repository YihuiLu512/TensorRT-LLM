# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CPU chain with engines that declare what the ds-engine declares (case-03), on CPU.

The chain (``cpu_disagg_chain``) runs the orchestrator and two real worker
frontends in front of two fake engines with the ``ds`` handoff shape and the
``ds-like`` profile: the capabilities the ds-engine's OpenEngine server
reports.

- NT-177 (DG-45): a context leg that fails once is sent again by the
  orchestrator with a new ``disagg_request_id``, and the context worker passes
  the new id on to its engine; the generation leg pulls with that id.
- NT-178 (DG-88): each request parameter the case-03 module expects the ds
  profile to refuse (``DS_REFUSED``) is refused with HTTP 400 before any
  Generate call, through the orchestrator, with the same error body shape.
"""

from __future__ import annotations

import grpc
import pytest
import yaml

from tensorrt_llm.grpc.openengine_conventions import (
    SERVE_COUNTER_GENERATE_SENT,
    SERVE_COUNTER_HTTP_REJECTED,
    struct_to_dict,
)

__extra_import_path__ = ["..", "~/tests/unittest/grpc/openengine_client"]
from cpu_disagg_chain import CpuDisaggChain  # noqa: E402
from fake_openengine import Fail  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

CTX_LEG = "the context leg (context_only) of a disaggregated request"
PROMPT = "He llo"


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    directory = tmp_path_factory.mktemp("wt62-ds-chain")
    # SV-DSG-1: a worker frontend without cache_transceiver_config refuses the handoff fields.
    config = directory / "frontend.yaml"
    config.write_text(yaml.safe_dump({"cache_transceiver_config": {"backend": "DEFAULT"}}))
    args = ["--config", str(config)]
    with CpuDisaggChain(
        directory, "ds", frontend_args={"context": args, "generation": args}
    ) as served:
        yield served


def _delta(before: dict, after: dict, role: str, key: str) -> int:
    return after[role]["frontend"].get(key, 0) - before[role]["frontend"].get(key, 0)


def _disagg_request_id(request) -> str:
    return str(struct_to_dict(request.extra)["disagg_request_id"])


def test_ctx_retry_uses_new_disagg_request_id(chain):
    """NT-177 (DG-45): the resent context leg carries a new id down to the engine."""
    ctx_engine, gen_engine = chain.ctx.engine, chain.gen.engine
    ctx_seen, gen_seen = len(ctx_engine.received), len(gen_engine.received)
    ctx_engine.push_script(Fail(grpc.StatusCode.INTERNAL, "scripted context leg failure"))

    response = chain.complete(prompt=PROMPT, max_tokens=4)

    assert response.status_code == 200, response.text
    ctx_calls = [record.request for record in ctx_engine.received[ctx_seen:]]
    assert len(ctx_calls) == 2, [_disagg_request_id(r) for r in ctx_calls]
    failed, resent = (_disagg_request_id(r) for r in ctx_calls)
    assert failed != resent
    gen_calls = [record.request for record in gen_engine.received[gen_seen:]]
    assert len(gen_calls) == 1
    # The generation leg pulls the KV cache the resent context leg parked.
    assert gen_calls[0].kv.session.session_id == resent


# What the case-03 module's NT-119 and NT-120 send through the orchestrator, as completions or
# chat request fields, except the image input: the CPU chain's model has no image modality, so
# the front end refuses it before the engine's profile applies.
DS_REFUSED = {
    "logprobs": ("/v1/completions", {"logprobs": 1}),
    "repetition_penalty": ("/v1/completions", {"repetition_penalty": 1.2}),
    "presence_penalty": ("/v1/completions", {"presence_penalty": 0.5}),
    "frequency_penalty": ("/v1/completions", {"frequency_penalty": 0.5}),
    "min_tokens": ("/v1/completions", {"min_tokens": 1}),
    "logit_bias": ("/v1/completions", {"logit_bias": {"5": 1.0}}),
    "thinking_token_budget": ("/v1/completions", {"thinking_token_budget": 8}),
    "guided_json": (
        "/v1/completions",
        {"response_format": {"type": "json", "schema": {"type": "object"}}},
    ),
    "priority": ("/v1/completions", {"priority": 0.5}),
    "cache_salt": ("/v1/chat/completions", {"cache_salt": "salt"}),
    "return_context_logits": ("/v1/completions", {"return_context_logits": True}),
    "scheduling_params": ("/v1/chat/completions", {"agent_hierarchy": [["planner", 1]]}),
    "lora": (
        "/v1/completions",
        {"lora_request": {"lora_name": "adapter", "lora_int_id": 1, "lora_path": "/nonexistent"}},
    ),
    "n2": ("/v1/completions", {"n": 2}),
    "best_of2": ("/v1/completions", {"n": 1, "best_of": 2}),
}


def _body(path: str, fields: dict) -> dict:
    if path == "/v1/chat/completions":
        messages = [{"role": "user", "content": PROMPT}]
        return {"messages": messages, "max_tokens": 4, **fields}
    return {"prompt": PROMPT, "max_tokens": 4, **fields}


@pytest.fixture(scope="module")
def refusals(chain):
    """Each refused request's response, by parameter, with the counter deltas it caused."""
    seen = {}
    for name, (path, fields) in DS_REFUSED.items():
        before = chain.snapshot()
        response = chain.post(path, _body(path, fields))
        seen[name] = (response, before, chain.snapshot())
    return seen


@pytest.mark.parametrize("name", list(DS_REFUSED))
def test_ds_profile_rejects(refusals, name):
    """NT-178 (DG-88): refused with 400 at the context worker, before any Generate call."""
    response, before, after = refusals[name]
    assert response.status_code == 400, response.text
    assert CTX_LEG in response.text, response.text
    for role in ("context", "generation"):
        assert _delta(before, after, role, SERVE_COUNTER_GENERATE_SENT) == 0
    # The orchestrator resends a context leg it got an error for once (max_retries=1).
    assert 1 <= _delta(before, after, "context", SERVE_COUNTER_HTTP_REJECTED) <= 2
    assert _delta(before, after, "generation", SERVE_COUNTER_HTTP_REJECTED) == 0


def test_ds_profile_rejects_have_one_body_shape(refusals):
    """NT-178 (DG-88): every refusal has the same error body keys and names the context leg."""
    shapes = {name: sorted(response.json()) for name, (response, _, _) in refusals.items()}
    assert len({tuple(keys) for keys in shapes.values()}) == 1, shapes
    texts = {name: response.text for name, (response, _, _) in refusals.items()}
    assert all(CTX_LEG in text for text in texts.values()), texts


def test_ds_profile_accepts_supported_sampling(chain):
    """NT-178 (DG-88) control: sampling the profile supports reaches both engines."""
    before = chain.snapshot()
    response = chain.post(
        "/v1/completions", _body("/v1/completions", {"temperature": 0.5, "top_p": 0.9, "seed": 3})
    )
    after = chain.snapshot()
    assert response.status_code == 200, response.text
    for role in ("context", "generation"):
        assert _delta(before, after, role, SERVE_COUNTER_GENERATE_SENT) == 1, role
        assert _delta(before, after, role, SERVE_COUNTER_HTTP_REJECTED) == 0, role

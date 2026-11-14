# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subagent affinity through the worker frontends (SV-DSG-14, OE-DSG-9; CS-09).

NT-004 runs on the CPU chain (``cpu_disagg_chain``): the orchestrator takes the
parent session from a gateway header and signs it for both legs
(``subagent_affinity_scope: both``); the context worker frontend checks that
signature, signs the parent id again for the Generate call's ``request_id``
and sends it as metadata, which the fake engine checks as the TensorRT-LLM
server does; the generation worker frontend checks it and sends nothing. A
signature that does not check is refused before any Generate call. A LoRA
request, whose call is renamed after it is mapped, is signed for its final
``request_id``. The unit tests check the metadata the facade builds against
the engine's own check.
"""

from __future__ import annotations

import dataclasses
import functools
from types import SimpleNamespace

import pytest
import requests
import yaml

from tensorrt_llm.grpc.openengine.affinity import (
    AFFINITY_AUTH_HEADER,
    AFFINITY_ID_HEADER,
    validate_parent_affinity,
)
from tensorrt_llm.grpc.openengine_conventions import (
    BUCKET_CONTEXT_ONLY,
    BUCKET_GENERATION_ONLY,
    DISAGG_REQUEST_ID_MIN,
    REQ_CONVERSATION_ID,
    REQUEST_TYPE_CONTEXT_ONLY,
    SERVE_COUNTER_GENERATE_SENT,
    SERVE_COUNTER_HTTP_REJECTED,
)
from tensorrt_llm.serve.openengine_poc.engine_client import EngineRequest
from tensorrt_llm.serve.openengine_poc.engine_client_impl import to_generate_request
from tensorrt_llm.serve.openengine_poc.facade import OpenEngineLLM

__extra_import_path__ = ["..", "~/tests/unittest/grpc/openengine_client"]
from cpu_disagg_chain import MODEL, CpuDisaggChain  # noqa: E402
from fake_openengine import disagg_profile  # noqa: E402
from fakes import trtllm_like  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

KEY = "wt53-affinity-key"
PARENT_HEADER = "X-Parent-Session-ID"
PARENT = "parent-53"
CHILD = "child-53"


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    directory = tmp_path_factory.mktemp("affinity")
    # The key signs the orchestrator's headers and the engine's metadata alike:
    # every worker frontend YAML and every engine has it (TH-DSG-2).
    config = directory / "frontend.yaml"
    config.write_text(
        yaml.safe_dump(
            {"cache_transceiver_config": {"backend": "DEFAULT"}, "internal_request_auth_key": KEY}
        )
    )
    args = ["--config", str(config)]
    profile = disagg_profile("trtllm-like", subagent_affinity_key=KEY, supports_lora=True)
    with CpuDisaggChain(
        directory,
        "trtllm",
        profiles={"context": profile, "generation": profile},
        frontend_args={"context": args, "generation": args},
        orchestrator_config={
            "internal_request_auth_key": KEY,
            "conversation_affinity_header_for_subagents": PARENT_HEADER,
            "subagent_affinity_scope": "both",
        },
    ) as chain:
        yield chain


def _chat(url: str, headers: dict[str, str], **fields) -> requests.Response:
    return requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "He llo"}],
            "max_tokens": 3,
            **fields,
        },
        headers={"Connection": "close", **headers},
        timeout=60,
    )


def _calls(engine, bucket: str) -> list:
    return [r for r in engine.received if r.admitted is not None and r.admitted.bucket == bucket]


def _header(record: dict, name: str) -> str | None:
    return {key.lower(): value for key, value in record["headers"].items()}.get(name.lower())


def test_ctx_leg_resigns_affinity_and_gen_leg_drops_it(chain):
    """NT-004 (DG-22 with ``scope: both``, DG-35)."""
    ctx_before = len(_calls(chain.ctx.engine, BUCKET_CONTEXT_ONLY))
    gen_before = len(_calls(chain.gen.engine, BUCKET_GENERATION_ONLY))
    response = _chat(chain.url, {PARENT_HEADER: PARENT, "x-session-id": CHILD})
    assert response.status_code == 200, chain.orchestrator_log()[-4000:]
    assert response.json()["choices"][0]["message"]["content"]

    (prefill,) = _calls(chain.ctx.engine, BUCKET_CONTEXT_ONLY)[ctx_before:]
    # The fake engine admitted it with the TensorRT-LLM server's check, which
    # binds the Generate call's request_id: the frontend signed it again.
    assert prefill.admitted.subagent_affinity_id == PARENT
    assert prefill.metadata_value(AFFINITY_ID_HEADER) == PARENT
    grpc_signature = prefill.metadata_value(AFFINITY_AUTH_HEADER)
    assert prefill.request.extra.fields[REQ_CONVERSATION_ID].string_value == CHILD
    ctx_http = [r for r in chain.ctx.http_requests() if r["path"] == "/v1/chat/completions"][-1]
    assert _header(ctx_http, AFFINITY_ID_HEADER) == PARENT
    # The orchestrator's signature binds the disagg_request_id instead.
    assert _header(ctx_http, AFFINITY_AUTH_HEADER) not in (None, grpc_signature)

    (decode,) = _calls(chain.gen.engine, BUCKET_GENERATION_ONLY)[gen_before:]
    # With scope both the generation worker frontend was sent the affinity
    # headers and checked them; the generation engine is sent none.
    gen_http = [r for r in chain.gen.http_requests() if r["path"] == "/v1/chat/completions"][-1]
    assert _header(gen_http, AFFINITY_ID_HEADER) == PARENT
    assert decode.metadata_value(AFFINITY_ID_HEADER) is None
    assert decode.metadata_value(AFFINITY_AUTH_HEADER) is None
    assert decode.admitted.subagent_affinity_id is None


def test_a_lora_request_is_signed_for_its_final_request_id(chain, tmp_path):
    """The LoRA path maps the call before the adapter is registered and renames it afterwards."""
    before = len(_calls(chain.ctx.engine, BUCKET_CONTEXT_ONLY))
    lora_request = {"lora_name": "adapter-53", "lora_int_id": 53, "lora_path": str(tmp_path)}
    response = _chat(
        chain.url, {PARENT_HEADER: PARENT, "x-session-id": CHILD}, lora_request=lora_request
    )
    assert response.status_code == 200, chain.orchestrator_log()[-4000:]
    (prefill,) = _calls(chain.ctx.engine, BUCKET_CONTEXT_ONLY)[before:]
    assert prefill.request.lora_name == "adapter-53"
    assert prefill.admitted.subagent_affinity_id == PARENT


def test_a_request_without_a_parent_sends_no_affinity_metadata(chain):
    before = len(_calls(chain.ctx.engine, BUCKET_CONTEXT_ONLY))
    response = _chat(chain.url, {"x-session-id": CHILD})
    assert response.status_code == 200, response.text
    (prefill,) = _calls(chain.ctx.engine, BUCKET_CONTEXT_ONLY)[before:]
    assert prefill.metadata_value(AFFINITY_ID_HEADER) is None
    assert prefill.admitted.subagent_affinity_id is None


def test_an_affinity_signature_that_does_not_check_sends_no_generate(chain):
    """SV-DSG-14: refused at the HTTP layer of the worker frontend, before any Generate call."""
    received = len(chain.ctx.engine.received)
    before = chain.snapshot()
    response = _chat(
        chain.ctx.url,
        {AFFINITY_ID_HEADER: PARENT, AFFINITY_AUTH_HEADER: "sha256=" + "0" * 64},
        disaggregated_params={
            "request_type": "context_only",
            "disagg_request_id": DISAGG_REQUEST_ID_MIN + 53,
        },
        conversation_params={"conversation_id": CHILD},
    )
    assert 400 <= response.status_code < 500, response.text
    assert "subagent affinity" in response.text
    assert len(chain.ctx.engine.received) == received
    after = chain.snapshot()
    frontend_before, frontend_after = before["context"]["frontend"], after["context"]["frontend"]
    assert frontend_after.get(SERVE_COUNTER_GENERATE_SENT, 0) == frontend_before.get(
        SERVE_COUNTER_GENERATE_SENT, 0
    )
    assert (
        frontend_after.get(SERVE_COUNTER_HTTP_REJECTED, 0)
        == frontend_before.get(SERVE_COUNTER_HTTP_REJECTED, 0) + 1
    )


# -- The facade's metadata, without a chain ----------------------------------------


def _info(declared: bool = True):
    info = trtllm_like()[0]
    return dataclasses.replace(
        info, extra={**info.extra, "trtllm_supports_subagent_affinity": declared}
    )


def _request(**extra) -> EngineRequest:
    return EngineRequest(
        request_id="call-1",
        prompt_token_ids=[1, 2, 3],
        max_tokens=4,
        model="fake-model",
        extra={REQ_CONVERSATION_ID: CHILD, **extra},
        request_type=REQUEST_TYPE_CONTEXT_ONLY,
    )


def _metadata(key: str | None, request: EngineRequest, info=None) -> dict[str, str]:
    facade = SimpleNamespace(_subagent_affinity_key=key)
    return OpenEngineLLM._affinity_metadata(facade, request, PARENT, info or _info())


def test_the_metadata_passes_the_engines_check():
    """DG-35: the signature binds the call's request_id, model, conversation id and type."""
    request = _request()
    metadata = _metadata(KEY, request)
    assert set(metadata) == {AFFINITY_ID_HEADER, AFFINITY_AUTH_HEADER}
    message = to_generate_request(request)
    assert (
        validate_parent_affinity(KEY, message, CHILD, REQUEST_TYPE_CONTEXT_ONLY, metadata) == PARENT
    )
    # Another call of the same request does not take it.
    other = to_generate_request(dataclasses.replace(request, request_id="call-2"))
    with pytest.raises(ValueError, match="Invalid internal subagent affinity"):
        validate_parent_affinity(KEY, other, CHILD, REQUEST_TYPE_CONTEXT_ONLY, metadata)


def test_a_frontend_without_a_key_refuses_affinity():
    with pytest.raises(ValueError, match="requires internal_request_auth_key"):
        _metadata(None, _request())


@pytest.mark.parametrize(
    "request_extra, parent, declared",
    [({}, PARENT, False), ({REQ_CONVERSATION_ID: " "}, PARENT, True), ({}, "parenté", True)],
    ids=["engine-does-not-check", "no-conversation", "parent-not-ascii"],
)
def test_affinity_the_engine_cannot_take_is_not_sent(request_extra, parent, declared):
    """OE-DSG-9: the call is served without the hint, as in-process without conversation-aware ADP."""
    facade = SimpleNamespace(_subagent_affinity_key=KEY)
    request = _request(**request_extra)
    assert OpenEngineLLM._affinity_metadata(facade, request, parent, _info(declared)) == {}


def test_a_renamed_call_is_signed_for_its_new_request_id():
    """The LoRA path renames the first call after mapping it; the metadata follows."""
    info = _info()
    facade = SimpleNamespace(
        _subagent_affinity_key=KEY, client=SimpleNamespace(server_info=lambda: info)
    )
    facade._affinity_metadata = functools.partial(OpenEngineLLM._affinity_metadata, facade)
    request = _request()
    request.metadata = {"other": "kept", **_metadata(KEY, request, info)}
    renamed = OpenEngineLLM._with_request_id(facade, request, "call-9")
    assert renamed.request_id == "call-9" and renamed.metadata["other"] == "kept"
    message = to_generate_request(renamed)
    assert (
        validate_parent_affinity(KEY, message, CHILD, REQUEST_TYPE_CONTEXT_ONLY, renamed.metadata)
        == PARENT
    )
    # The original call keeps its own metadata.
    assert request.metadata[AFFINITY_AUTH_HEADER] != renamed.metadata[AFFINITY_AUTH_HEADER]
    # A call without affinity metadata is only renamed.
    plain = OpenEngineLLM._with_request_id(facade, _request(), "call-10")
    assert plain.request_id == "call-10" and AFFINITY_ID_HEADER not in plain.metadata
    # An engine that restarted without the declaration is sent none: the old signature goes.
    facade.client = SimpleNamespace(server_info=lambda: _info(declared=False))
    dropped = OpenEngineLLM._with_request_id(facade, request, "call-11")
    assert dropped.metadata == {"other": "kept"}

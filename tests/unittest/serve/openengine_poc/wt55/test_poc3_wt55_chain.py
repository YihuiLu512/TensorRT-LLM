# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation-first requests through real worker frontends on the CPU chain (``cpu_disagg_chain``).

The orchestrator runs with ``schedule_style: generation_first``: it sends both
legs at once, the generation leg without a context response. The prefill call
carries ``extra.schedule_style``; the decode call carries a ``kv.session`` the
generation worker frontend built from the HTTP fields, with the context
endpoint the orchestrator read from the context worker's ``/server_info``.
A context engine that does not declare ``trtllm.disagg.generation_first``, or
that names no ``kv_connector.local_endpoints``, is never sent the prefill call.
"""

from __future__ import annotations

import json

import pytest
import requests
import yaml
from google.protobuf import json_format

from tensorrt_llm.grpc.openengine_conventions import (
    BUCKET_CONTEXT_ONLY,
    BUCKET_GENERATION_ONLY,
    CAP_DISAGG_GENERATION_FIRST,
    DISAGG_REQUEST_ID_MIN,
    REQ_DISAGG_REQUEST_ID,
    REQ_SCHEDULE_STYLE,
    SCHEDULE_STYLE_GENERATION_FIRST,
    SERVE_COUNTER_HTTP_RECEIVED_BUCKETS,
    SERVE_COUNTER_HTTP_REJECTED,
)

__extra_import_path__ = ["..", "~/tests/unittest/grpc/openengine_client"]
from cpu_disagg_chain import MODEL, CpuDisaggChain  # noqa: E402
from fake_openengine import FakeKvConnector, disagg_profile  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

CTX_ENDPOINT = ("10.1.2.3", 4567, "tcp")
CTX_ENDPOINT_TEXT = "tcp://10.1.2.3:4567"
PROMPT = "He llo"
CTX_RECEIVED = SERVE_COUNTER_HTTP_RECEIVED_BUCKETS[BUCKET_CONTEXT_ONLY]


def _profiles(ctx_generation_first: bool, ctx_endpoints: tuple) -> dict:
    """The generation engine always declares generation-first; the context engine as given."""
    profiles = {}
    for role in ("context", "generation"):
        endpoints = ctx_endpoints if role == "context" else ()
        declared = ctx_generation_first if role == "context" else True
        profiles[role] = disagg_profile(
            "trtllm-like",
            kv_connector=FakeKvConnector(local_endpoints=endpoints),
            capabilities={CAP_DISAGG_GENERATION_FIRST: declared},
        )
    return profiles


def _chain(
    directory, ctx_generation_first: bool, ctx_endpoints: tuple = (CTX_ENDPOINT,)
) -> CpuDisaggChain:
    config = directory / "frontend.yaml"
    config.write_text(yaml.safe_dump({"cache_transceiver_config": {"backend": "DEFAULT"}}))
    args = ["--config", str(config)]
    return CpuDisaggChain(
        directory,
        "trtllm",
        profiles=_profiles(ctx_generation_first, ctx_endpoints),
        frontend_args={"context": args, "generation": args},
        orchestrator_config={"schedule_style": "generation_first"},
    )


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    with _chain(tmp_path_factory.mktemp("chain-gen-first"), ctx_generation_first=True) as chain:
        yield chain


def _calls(engine, bucket: str) -> list:
    return [r for r in engine.received if r.admitted is not None and r.admitted.bucket == bucket]


def _body(path: str) -> dict:
    if path == "/v1/completions":
        return {"prompt": PROMPT, "max_tokens": 4}
    return {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 4}


def _post(chain, path: str, stream: bool) -> tuple[int, str]:
    if not stream:
        response = chain.post(path, _body(path))
        if response.status_code != 200:
            return response.status_code, response.text
        choice = response.json()["choices"][0]
        return 200, choice["text"] if path == "/v1/completions" else choice["message"]["content"]
    with requests.post(
        f"{chain.url}{path}",
        json={"model": MODEL, **_body(path), "stream": True},
        headers={"Connection": "close"},
        timeout=60,
        stream=True,
    ) as response:
        text = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choice = json.loads(line[len("data: ") :])["choices"][0]
            text += choice.get("text") or (choice.get("delta") or {}).get("content") or ""
        return response.status_code, text


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("path", ["/v1/completions", "/v1/chat/completions"])
def test_a_generation_first_request_goes_through_both_legs(chain, path, stream):
    ctx_before = len(_calls(chain.ctx.engine, BUCKET_CONTEXT_ONLY))
    gen_before = len(_calls(chain.gen.engine, BUCKET_GENERATION_ONLY))
    status, text = _post(chain, path, stream)
    assert status == 200, (text, chain.orchestrator_log()[-4000:])
    assert text
    (prefill,) = _calls(chain.ctx.engine, BUCKET_CONTEXT_ONLY)[ctx_before:]
    (decode,) = _calls(chain.gen.engine, BUCKET_GENERATION_ONLY)[gen_before:]

    prefill_extra = json_format.MessageToDict(prefill.request.extra)
    assert prefill_extra[REQ_SCHEDULE_STYLE] == SCHEDULE_STYLE_GENERATION_FIRST
    disagg_request_id = prefill_extra[REQ_DISAGG_REQUEST_ID]

    decode_extra = json_format.MessageToDict(decode.request.extra)
    assert REQ_SCHEDULE_STYLE not in decode_extra
    assert REQ_DISAGG_REQUEST_ID not in decode_extra
    session = decode.request.kv.session
    # The orchestrator names the context request by its own id in generation-first.
    assert session.session_id == disagg_request_id
    assert json_format.MessageToDict(session.attributes_struct) == {
        "ctx_info_endpoint": CTX_ENDPOINT_TEXT,
        "disagg_request_id": disagg_request_id,
        "schedule_style": SCHEDULE_STYLE_GENERATION_FIRST,
    }
    assert [(e.host, e.port, e.protocol) for e in session.endpoints] == [CTX_ENDPOINT]
    # Each worker frontend renders the prompt itself, to the same tokens.
    assert prefill.request.HasField("token_ids")
    assert decode.request.token_ids == prefill.request.token_ids


def _through_orchestrator(chain) -> tuple[int, int, str]:
    """One request through the orchestrator: the context legs the context frontend refused, and the reply."""
    before = chain.snapshot()["context"]["frontend"]
    status, text = _post(chain, "/v1/completions", stream=False)
    after = chain.snapshot()["context"]["frontend"]
    received = after.get(CTX_RECEIVED, 0) - before.get(CTX_RECEIVED, 0)
    rejected = after.get(SERVE_COUNTER_HTTP_REJECTED, 0) - before.get(
        SERVE_COUNTER_HTTP_REJECTED, 0
    )
    assert received == rejected, (received, rejected, status, text)
    return rejected, status, text


def _context_leg(chain) -> requests.Response:
    """A generation-first context leg sent straight to the context worker frontend."""
    return chain.post(
        "/v1/completions",
        {
            "prompt": PROMPT,
            "max_tokens": 1,
            "disaggregated_params": {
                "request_type": "context_only",
                "schedule_style": SCHEDULE_STYLE_GENERATION_FIRST,
                "disagg_request_id": DISAGG_REQUEST_ID_MIN,
            },
        },
        url=chain.ctx.url,
    )


def test_a_context_engine_without_generation_first_is_not_sent_the_prefill(tmp_path):
    with _chain(tmp_path, ctx_generation_first=False) as chain:
        rejected, _, _ = _through_orchestrator(chain)
        # The orchestrator retries the context leg; the frontend refuses every attempt.
        assert rejected >= 1
        assert _calls(chain.ctx.engine, BUCKET_CONTEXT_ONLY) == []
        response = _context_leg(chain)
        assert response.status_code == 400, response.text
        assert CAP_DISAGG_GENERATION_FIRST in response.text


def test_a_context_engine_naming_no_local_endpoint_is_not_sent_the_prefill(tmp_path):
    with _chain(tmp_path, ctx_generation_first=True, ctx_endpoints=()) as chain:
        rejected, _, _ = _through_orchestrator(chain)
        assert rejected >= 1
        assert _calls(chain.ctx.engine, BUCKET_CONTEXT_ONLY) == []
        response = _context_leg(chain)
        assert response.status_code == 400, response.text
        assert "local_endpoints" in response.text

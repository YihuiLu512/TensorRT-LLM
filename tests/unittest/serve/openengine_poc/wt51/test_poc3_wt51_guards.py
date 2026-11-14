# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The refusals around a handoff that do not need two legs to reach them (SV-DSG-2, SV-DSG-3).

A disaggregated request to an engine without KV transfer is refused by the
worker frontend, before any Generate call. A handoff the engine could not read
back, and a call that sets the prefill or decode fields where the client does
not take them, are refused before they are sent. The request counters bucket a
body as the server reads its request type.
"""

from __future__ import annotations

import json

import pytest
import requests
import yaml

from tensorrt_llm.grpc.openengine_conventions import (
    BUCKET_AGGREGATED,
    BUCKET_CONTEXT_ONLY,
    REQ_REQUEST_TYPE,
)
from tensorrt_llm.serve.openengine_poc import facade, handoff
from tensorrt_llm.serve.openengine_poc.engine_client import EngineRequest, KvSessionSpec
from tensorrt_llm.serve.openengine_poc.engine_client_impl import to_generate_request

__extra_import_path__ = [".."]
from cpu_disagg_chain import MODEL, CpuDisaggChain, disagg_profile  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def _request(**fields) -> EngineRequest:
    return EngineRequest(request_id="r", prompt_token_ids=[1, 2], max_tokens=2, **fields)


@pytest.mark.parametrize(
    "fields, match",
    [
        ({"extra": {REQ_REQUEST_TYPE: "context_only"}}, "request_type"),
        (
            {"request_type": "context_only", "kv_session": KvSessionSpec(session_id="1")},
            "kv_session",
        ),
        (
            {"kv_session": KvSessionSpec(session_id="1", attributes={"vendor": object()})},
            "cannot be sent",
        ),
    ],
    ids=["request-type-in-extra", "prefill-with-session", "unsendable-attribute"],
)
def test_a_call_the_client_cannot_send_is_refused(fields, match):
    with pytest.raises(ValueError, match=match):
        to_generate_request(_request(**fields))


@pytest.mark.parametrize(
    "attributes",
    [json.dumps({"vendor": [1, 1 << 60]}), '{"vendor": ' + "[" * 5000 + "]" * 5000 + "}"],
    ids=["beyond-2^53", "nested-too-deeply"],
)
def test_pass_through_attributes_a_struct_cannot_hold_are_refused(attributes):
    params, _ = handoff.decode_kv_session(KvSessionSpec(session_id="12"))
    with pytest.raises(ValueError):
        handoff.encode_kv_session(params, attributes, "")


@pytest.mark.parametrize(
    "body, bucket",
    [
        (b'{"disaggregated_params": {"request_type": "CONTEXT_ONLY"}}', BUCKET_CONTEXT_ONLY),
        (b'{"prompt": ' + b"[" * 5000 + b"]" * 5000 + b"}", BUCKET_AGGREGATED),
    ],
    ids=["upper-case-type", "nested-too-deeply"],
)
def test_a_body_is_bucketed_as_the_server_reads_it(body, bucket):
    assert facade._http_bucket(body, msgpack=False) == bucket


@pytest.fixture(scope="module")
def chain_without_kv_transfer(tmp_path_factory):
    directory = tmp_path_factory.mktemp("chain-no-kv")
    config = directory / "frontend.yaml"
    config.write_text(yaml.safe_dump({"cache_transceiver_config": {"backend": "DEFAULT"}}))
    profile = disagg_profile("trtllm-like").replace(kv_connector=None)
    args = ["--config", str(config)]
    with CpuDisaggChain(
        directory,
        "trtllm",
        profiles={"context": profile, "generation": profile},
        frontend_args={"context": args, "generation": args},
    ) as chain:
        yield chain


@pytest.mark.parametrize(
    "params",
    [
        {"request_type": "context_only", "disagg_request_id": (1 << 41) + 1},
        {"request_type": "generation_only", "ctx_request_id": 1, "disagg_request_id": 2},
    ],
    ids=["context-leg", "generation-leg"],
)
def test_a_leg_to_an_engine_without_kv_transfer_is_refused(chain_without_kv_transfer, params):
    worker = chain_without_kv_transfer.ctx
    before = len(worker.engine.requests)
    response = requests.post(
        f"{worker.url}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "He llo",
            "max_tokens": 2,
            "disaggregated_params": params,
        },
        timeout=60,
    )
    assert response.status_code == 400, response.text
    assert "has no KV transfer enabled" in response.text
    assert len(worker.engine.requests) == before

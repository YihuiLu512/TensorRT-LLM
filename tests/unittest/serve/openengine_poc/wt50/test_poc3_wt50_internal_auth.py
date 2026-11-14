# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A gen worker frontend checks the orchestrator's signature on a handoff (SV-DSG-1), on CPU.

A worker frontend with the orchestrator's key refuses a handoff whose
signature does not match; one without a key refuses a signed handoff, whose
signature it cannot verify, and serves an unsigned one.

The worker frontends are the real ``trtllm-serve`` command line on the gRPC
fake engine (see ``wt50_chain``); the requests are sent to them directly, as
the orchestrator would send the generation leg.
"""

from __future__ import annotations

import secrets
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wt50_chain as chain  # noqa: E402

from tensorrt_llm.serve.disagg_auth import build_internal_disagg_auth_headers  # noqa: E402
from tensorrt_llm.serve.openai_protocol import CompletionRequest  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False), pytest.mark.timeout(900)]

AUTH_KEY = secrets.token_hex(16)
AUTH_FAILED = "Invalid internal disaggregated request authentication"
NO_TRANSCEIVER = "require cache_transceiver_config"
UNVERIFIABLE = "no internal_request_auth_key to verify it"


def _handoff(opaque_state: str) -> dict:
    return {
        "model": chain.MODEL,
        "prompt": "He",
        "max_tokens": 2,
        "disaggregated_params": {
            "request_type": "generation_only",
            "ctx_request_id": 1,
            "disagg_request_id": 2,
            "encoded_opaque_state": opaque_state,
        },
    }


def _chat_handoff(opaque_state: str) -> dict:
    body = _handoff(opaque_state)
    del body["prompt"]
    return {**body, "messages": [{"role": "user", "content": "He"}]}


def _signed(body: dict, key: str = AUTH_KEY) -> dict[str, str]:
    # The signature covers the disaggregated fields only, so a chat body signs as a completion.
    request = CompletionRequest(
        model=body["model"], prompt="He", disaggregated_params=body["disaggregated_params"]
    )
    return build_internal_disagg_auth_headers(key, request)


def _gen_worker(
    tmp_path_factory, name: str, key: str | None
) -> Iterator[tuple[str, chain.FakeEngine]]:
    """A gen worker frontend with ``key`` (None for none) in its YAML."""
    tmp_path = tmp_path_factory.mktemp(f"wt50_auth_{name}")
    model = chain.model_dir(tmp_path / "model")
    config: dict = {"cache_transceiver_config": {"backend": "DEFAULT"}}
    if key is not None:
        config["internal_request_auth_key"] = key
    with chain.running_in_thread(chain.engine(model)) as fake:
        port = chain.free_port()
        worker = chain.start_worker(
            tmp_path, name, model, fake, role="generation", config=config, port=port
        )
        with chain.stopping(worker):
            url = f"http://127.0.0.1:{port}"
            chain.wait(
                lambda: chain.status(f"{url}/health") == 200,
                "the worker to be healthy",
                processes=[worker],
            )
            yield url, fake


@pytest.fixture(scope="module")
def gen_worker(tmp_path_factory) -> Iterator[tuple[str, chain.FakeEngine]]:
    """A gen worker frontend with the orchestrator's key in its YAML."""
    yield from _gen_worker(tmp_path_factory, "keyed", AUTH_KEY)


@pytest.fixture(scope="module")
def keyless_gen_worker(tmp_path_factory) -> Iterator[tuple[str, chain.FakeEngine]]:
    """A gen worker frontend without a key."""
    yield from _gen_worker(tmp_path_factory, "keyless", None)


def _post(
    url: str, body: dict, headers: dict[str, str], path: str = "/v1/completions"
) -> requests.Response:
    return requests.post(f"{url}{path}", json=body, headers=headers, timeout=60)


def test_gen_worker_frontend_rejects_a_tampered_handoff_signature(gen_worker, keyless_gen_worker):
    """NT-008 (DG-34, DG-37): a worker frontend refuses a signature it does not accept.

    A worker frontend with the key refuses a handoff whose signature does not
    match; one without a key refuses every signed handoff.
    """
    keyed, keyed_engine = gen_worker
    keyless, keyless_engine = keyless_gen_worker
    body = _handoff("b3BhcXVl")
    headers = _signed(body)
    tampered = _handoff("dGFtcGVyZWQ=")
    before = chain.generate_calls(keyed_engine)

    refused = [
        (tampered, headers),  # the opaque state changed after signing
        (body, {}),  # no signature
        (body, _signed(body, "another-key")),  # signed with another key
    ]
    for sent_body, sent_headers in refused:
        response = _post(keyed, sent_body, sent_headers)
        assert 400 <= response.status_code < 500, response.text
        assert AUTH_FAILED in response.text
    assert chain.generate_calls(keyed_engine) == before

    # Only the orchestrator has the key: the worker frontend cannot verify the signature, so it
    # refuses the handoff, tampered or not.
    keyless_before = chain.generate_calls(keyless_engine)
    for sent_body in (tampered, body):
        response = _post(keyless, sent_body, headers)
        assert response.status_code == 400, response.text
        assert AUTH_FAILED in response.text and UNVERIFIABLE in response.text
    assert chain.generate_calls(keyless_engine) == keyless_before


def test_a_keyless_worker_frontend_refuses_a_signed_chat_handoff(keyless_gen_worker):
    """As a signed completion handoff, a signed chat completion handoff is refused."""
    keyless, keyless_engine = keyless_gen_worker
    body = _chat_handoff("b3BhcXVl")
    before = chain.generate_calls(keyless_engine)
    response = _post(keyless, body, _signed(body), "/v1/chat/completions")
    assert response.status_code == 400, response.text
    assert AUTH_FAILED in response.text and UNVERIFIABLE in response.text
    assert chain.generate_calls(keyless_engine) == before


def test_a_keyless_worker_frontend_serves_what_carries_no_signature_to_verify(keyless_gen_worker):
    """A worker frontend without a key serves an unsigned handoff.

    It also serves a request without handoff fields that carries the signature
    header: the header alone makes no handoff.
    """
    keyless, keyless_engine = keyless_gen_worker
    body = _handoff("b3BhcXVl")
    before = chain.generate_calls(keyless_engine)
    response = _post(keyless, body, {})
    assert response.status_code == 200, response.text
    assert chain.generate_calls(keyless_engine) == before + 1

    plain = {"model": chain.MODEL, "prompt": "He", "max_tokens": 2}
    response = _post(keyless, plain, _signed(body))
    assert response.status_code == 200, response.text
    assert chain.generate_calls(keyless_engine) == before + 2


def test_a_signed_handoff_is_not_refused_for_a_missing_transceiver(gen_worker):
    """SV-DSG-1 (UV-43): the worker frontend's YAML keeps ``cache_transceiver_config``, so the check passes.

    The request passes the signature and transceiver checks and reaches the
    engine.
    """
    keyed, keyed_engine = gen_worker
    body = _handoff("b3BhcXVl")
    before = chain.generate_calls(keyed_engine)
    response = _post(keyed, body, _signed(body))
    assert response.status_code == 200, response.text
    assert chain.generate_calls(keyed_engine) == before + 1


def test_a_worker_frontend_without_a_transceiver_refuses_a_handoff(tmp_path):
    """The other side of UV-43: without ``cache_transceiver_config`` the handoff fields are refused."""
    model = chain.model_dir(tmp_path / "model")
    with chain.running_in_thread(chain.engine(model)) as fake:
        port = chain.free_port()
        worker = chain.start_worker(tmp_path, "bare", model, fake, role="generation", port=port)
        with chain.stopping(worker):
            url = f"http://127.0.0.1:{port}"
            chain.wait(
                lambda: chain.status(f"{url}/health") == 200,
                "the worker to be healthy",
                processes=[worker],
            )
            body = _handoff("b3BhcXVl")
            # Signed or not: the transceiver check comes before the signature checks.
            for headers in ({}, _signed(body)):
                response = _post(url, body, headers)
                assert response.status_code == 400 and NO_TRANSCEIVER in response.text
            assert chain.generate_calls(fake) == 0

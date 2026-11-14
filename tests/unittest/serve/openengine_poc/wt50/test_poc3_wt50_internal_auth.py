# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A gen worker frontend checks the orchestrator's signature on a handoff (SV-DSG-1), on CPU.

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


def _refused_by_the_facade(response: requests.Response) -> bool:
    """A 400 about ``disaggregated_params``, past the signature and transceiver checks."""
    text = response.text
    return (
        response.status_code == 400
        and "disaggregated_params" in text
        and AUTH_FAILED not in text
        and NO_TRANSCEIVER not in text
    )


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


def _signed(body: dict, key: str = AUTH_KEY) -> dict[str, str]:
    return build_internal_disagg_auth_headers(key, CompletionRequest(**body))


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


def _post(url: str, body: dict, headers: dict[str, str]) -> requests.Response:
    return requests.post(f"{url}/v1/completions", json=body, headers=headers, timeout=60)


def test_gen_worker_frontend_rejects_a_tampered_handoff_signature(gen_worker, keyless_gen_worker):
    """NT-008 (DG-34, DG-37): a handoff whose signature does not match is refused before the engine."""
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

    # Only one side has the key: the worker frontend warns and lets it through, as in-process,
    # to the facade's refusal of the handoff fields.
    response = _post(keyless, tampered, headers)
    assert _refused_by_the_facade(response), (response.status_code, response.text)
    assert chain.generate_calls(keyless_engine) == 0


def test_a_signed_handoff_is_not_refused_for_a_missing_transceiver(gen_worker):
    """SV-DSG-1 (UV-43): the worker frontend's YAML keeps ``cache_transceiver_config``, so the check passes.

    The request passes the signature and transceiver checks and reaches the
    facade, which refuses the handoff fields it does not map.
    """
    keyed, keyed_engine = gen_worker
    body = _handoff("b3BhcXVl")
    before = chain.generate_calls(keyed_engine)
    response = _post(keyed, body, _signed(body))
    assert _refused_by_the_facade(response), (response.status_code, response.text)
    assert chain.generate_calls(keyed_engine) == before


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
            response = _post(url, _handoff("b3BhcXVl"), {})
            assert response.status_code == 400 and NO_TRANSCEIVER in response.text
            assert chain.generate_calls(fake) == 0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Errors of a disaggregated request, through the orchestrator (DG-41, DG-42).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU), in-process or on the
OpenEngine chain, with ``max_retries: 0``: the orchestrator sends a refused
request once, not again. Whether the orchestrator sent a request to a worker
is read from its own per-role counters (``ctx_total_requests``,
``gen_total_requests`` on ``/prometheus/metrics``), which both forms have.

- NT-093 (DG-41): a request the context worker refuses comes back from the
  orchestrator with the worker's status and message, as ``{"detail": ...}``,
  and without a worker's address; the generation worker is not asked. The
  request's ``logit_bias`` names a token id outside the vocabulary, which the
  worker refuses while it builds the sampling parameters, before any engine
  call, in-process and as a worker frontend alike.
- NT-094 (DG-42): a ``/v1/completions`` or ``/v1/chat/completions`` body that
  fails the request schema is answered 400 ``{"error": <validation text>}`` by
  the orchestrator, which sends nothing to any worker.
"""

import json
import os
import re

import pytest
import requests

from ._test_openengine_disagg_common import REQUEST_TIMEOUT_S, ROLES, disagg_deployment, model_path

pytestmark = pytest.mark.threadleak(enabled=False)

PROMPT = "The capital of France is"
QUESTION = [{"role": "user", "content": "What is 1 + 1?"}]
# The orchestrator's count of requests it sent to each role's workers.
_SENT = re.compile(r"^(ctx|gen)_total_requests_total(?:\{[^}]*\})?\s+(\S+)\s*$", re.MULTILINE)


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    with disagg_deployment(
        tmp_path_factory.mktemp("disagg_errors"), orchestrator_config={"max_retries": 0}
    ) as deployment:
        yield deployment


def _sent(deployment) -> dict[str, int]:
    """Requests the orchestrator has sent to the context (``ctx``) and generation (``gen``) workers."""
    response = requests.get(f"{deployment.url}/prometheus/metrics", timeout=REQUEST_TIMEOUT_S)
    assert response.status_code == 200, response.text
    counts = {"ctx": 0, "gen": 0}
    for role, value in _SENT.findall(response.text):
        counts[role] += int(float(value))
    return counts


def _bodies(deployment, **fields) -> list[tuple[str, dict]]:
    """A completion and a chat request with ``fields`` added."""
    common = {"model": deployment.model, "max_tokens": 4, **fields}
    return [
        ("/v1/completions", {"prompt": PROMPT, **common}),
        ("/v1/chat/completions", {"messages": QUESTION, **common}),
    ]


def _post(deployment, path: str, body: dict) -> requests.Response:
    return requests.post(f"{deployment.url}{path}", json=body, timeout=REQUEST_TIMEOUT_S)


def _vocab_size(deployment) -> int:
    with open(os.path.join(model_path(deployment.model), "config.json"), encoding="utf-8") as f:
        return int(json.load(f)["vocab_size"])


def test_worker_frontend_4xx_passes_through_the_orchestrator(deployment):
    vocab_size = _vocab_size(deployment)
    expected = f"Token ID {vocab_size} out of vocabulary range [0, {vocab_size})"
    worker_ports = [str(w.port) for role in ROLES for w in deployment.workers[role]]
    bodies = _bodies(deployment, logit_bias={str(vocab_size): 1.0})
    before = _sent(deployment)
    for path, body in bodies:
        response = _post(deployment, path, body)
        assert response.status_code == 400, (path, response.text)
        assert response.json() == {"detail": expected}, (path, response.text)
        assert "http://" not in response.text, (path, response.text)
        for port in worker_ports:
            assert f":{port}" not in response.text, (path, port, response.text)
    after = _sent(deployment)
    assert after["ctx"] - before["ctx"] == len(bodies), (before, after)
    assert after["gen"] == before["gen"], (before, after)


def test_orchestrator_rejects_an_invalid_body_before_any_worker(deployment):
    before = _sent(deployment)
    for path, body in _bodies(deployment, no_such_field=1):
        response = _post(deployment, path, body)
        assert response.status_code == 400, (path, response.text)
        payload = response.json()
        assert list(payload) == ["error"], (path, response.text)
        assert "no_such_field" in payload["error"], (path, response.text)
    assert _sent(deployment) == before

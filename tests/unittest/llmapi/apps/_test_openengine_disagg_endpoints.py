# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endpoints of a disaggregated deployment, through the orchestrator and on one worker (F-02 to F-05d).

The deployment is ``_test_openengine_disagg_common.disagg_deployment``: one
context and one generation worker of Qwen3-0.6B on one GPU, behind the
orchestrator, in-process or on the OpenEngine chain.

- NT-050 (F-02): with ``echo`` the whole and the streamed completion return
  the same text, which starts with the prompt. It fails in-process already
  (UV-21), so it is marked as an expected failure in both forms.
- NT-051 (F-02, F-03, F-04): what the orchestrator does not serve is refused
  the same way in both forms: several prompts in one completion request,
  ``previous_response_id``, retrieving or deleting a stored response, and
  Anthropic message batches; ``store`` is accepted and the request served.
- NT-052 to NT-054 (F-05b to F-05d): ``/version``, ``/health`` and
  ``/health_generate``, which the orchestrator does not serve for its
  workers, answered by each worker itself: the package version, 200 once the
  orchestrator lists both workers, and 200 after a generation of the worker's own.
"""

import json
from urllib.parse import urlsplit

import pytest
import requests

from tensorrt_llm.version import __version__

from ._test_openengine_disagg_common import REQUEST_TIMEOUT_S, ROLES, disagg_deployment

pytestmark = pytest.mark.threadleak(enabled=False)

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    with disagg_deployment(tmp_path_factory.mktemp("disagg_endpoints")) as deployment:
        yield deployment


def _streamed_text(response: requests.Response) -> str:
    """The text of a streamed completion's server-sent events."""
    text = []
    for line in response.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line.removeprefix("data: "))
            text.extend(choice["text"] for choice in chunk.get("choices", []))
    return "".join(text)


@pytest.mark.xfail(
    strict=True,
    reason="UV-21: the orchestrator sends the generation leg the prompt as token ids, "
    "to which the generation worker's echo cannot add the completion's text",
)
def test_completion_echo(deployment):
    body = {
        "model": deployment.model,
        "prompt": PROMPT,
        "max_tokens": 8,
        "echo": True,
        "temperature": 0,
    }
    whole = _post(deployment, "/v1/completions", body)
    assert whole.status_code == 200, whole.text
    text = whole.json()["choices"][0]["text"]
    assert text.startswith(PROMPT) and len(text) > len(PROMPT), text
    streamed = _post(deployment, "/v1/completions", {**body, "stream": True})
    assert streamed.status_code == 200, streamed.text
    assert _streamed_text(streamed) == text, streamed.text


def _post(deployment, path: str, body: dict) -> requests.Response:
    return requests.post(f"{deployment.url}{path}", json=body, timeout=REQUEST_TIMEOUT_S)


@pytest.mark.parametrize(
    "case",
    [
        "multi_prompt",
        "previous_response_id",
        "store",
        "responses_retrieve_delete",
        "messages_batches",
    ],
)
def test_orchestrator_limits_same_as_baseline(deployment, case):
    model = deployment.model
    if case == "multi_prompt":
        # LIM-01: one prompt per request; a list of prompts fails on the orchestrator.
        response = _post(
            deployment, "/v1/completions", {"model": model, "prompt": [PROMPT, PROMPT]}
        )
        assert response.status_code == 500, response.text
        assert "single string prompt" in response.text, response.text
    elif case == "previous_response_id":
        # LIM-02: a stored response may be on another worker.
        body = {"model": model, "input": PROMPT, "previous_response_id": "resp_1"}
        response = _post(deployment, "/v1/responses", body)
        assert response.status_code == 400, response.text
        assert "previous_response_id" in response.text, response.text
    elif case == "store":
        # LIM-03: store is accepted and the request served; the orchestrator has no
        # route to read a stored response back (the responses_retrieve_delete case).
        body = {"model": model, "input": PROMPT, "max_output_tokens": 8, "store": True}
        response = _post(deployment, "/v1/responses", body)
        assert response.status_code == 200, response.text
        assert response.json()["output"], response.text
    elif case == "responses_retrieve_delete":
        # LIM-03: the orchestrator has no route to stored responses.
        url = f"{deployment.url}/v1/responses/resp_1"
        for method in (requests.get, requests.delete):
            response = method(url, timeout=REQUEST_TIMEOUT_S)
            assert response.status_code == 404, (method.__name__, response.text)
    else:
        # LIM-05: Anthropic message batches are not served.
        body = {"requests": [{"custom_id": "a", "params": {"model": model, "max_tokens": 4}}]}
        response = _post(deployment, "/v1/messages/batches", body)
        assert response.status_code == 404, response.text


def _worker_get(deployment, role: str, path: str) -> requests.Response:
    return requests.get(deployment.workers[role][0].url_for(path), timeout=REQUEST_TIMEOUT_S)


def test_version_direct(deployment):
    for role in ROLES:
        response = _worker_get(deployment, role, "version")
        assert response.status_code == 200, (role, response.text)
        assert response.json()["version"] == __version__, (role, response.text)


def test_health_direct(deployment):
    info = requests.get(f"{deployment.url}/cluster_info", timeout=REQUEST_TIMEOUT_S).json()
    for role in ROLES:
        # The orchestrator routes to "host:port" of each worker it lists.
        listed = [int(server.rsplit(":", 1)[1]) for server in info["server_lists"][role]]
        port = urlsplit(deployment.workers[role][0].url_for("health")).port
        assert listed == [port], (role, info)
        response = _worker_get(deployment, role, "health")
        assert response.status_code == 200, (role, response.text)


def test_health_generate_direct(deployment):
    for role in ROLES:
        response = _worker_get(deployment, role, "health_generate")
        assert response.status_code == 200, (role, response.text)

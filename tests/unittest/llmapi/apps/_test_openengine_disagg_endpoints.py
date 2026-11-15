# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endpoints of a disaggregated deployment, through the orchestrator and on one worker (F-02 to F-05d).

The deployment is ``_test_openengine_disagg_common.disagg_deployment``: one
context and one generation worker of Qwen3-0.6B on one GPU, behind the
orchestrator, in-process or on the OpenEngine chain.

- NT-050 (F-02): ``echo`` returns the prompt before the completion.
- NT-051 (F-02, F-03, F-04): what the orchestrator does not serve is refused
  the same way in both forms: several prompts in one completion request,
  ``previous_response_id``, stored responses and their retrieval, and
  Anthropic message batches.
- NT-052 to NT-054 (F-05b to F-05d): ``/version``, ``/health`` and
  ``/health_generate``, which the orchestrator does not serve for its
  workers, answered by a worker itself (``leg_expect: direct_worker`` on the
  chain).
"""

import pytest
import requests

from ._test_openengine_disagg_common import REQUEST_TIMEOUT_S, disagg_deployment

pytestmark = pytest.mark.threadleak(enabled=False)

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    with disagg_deployment(tmp_path_factory.mktemp("disagg_endpoints")) as deployment:
        yield deployment


def _direct_completion(deployment, role: str) -> None:
    """One aggregated request straight to a worker, so that its engine has served a call."""
    response = deployment.worker_client(role).completions.create(
        model=deployment.model, prompt=PROMPT, max_tokens=4
    )
    assert response.choices[0].text


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
def test_completion_echo(deployment, stream):
    client = deployment.client()
    kwargs = dict(model=deployment.model, prompt=PROMPT, max_tokens=8, echo=True, temperature=0)
    if stream:
        text = "".join(
            chunk.choices[0].text
            for chunk in client.completions.create(stream=True, **kwargs)
            if chunk.choices
        )
    else:
        text = client.completions.create(**kwargs).choices[0].text
    assert text.startswith(PROMPT), text
    assert len(text) > len(PROMPT), text


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
        # LIM-03: store is ignored; the response is served and not kept.
        body = {"model": model, "input": PROMPT, "max_output_tokens": 8, "store": True}
        response = _post(deployment, "/v1/responses", body)
        assert response.status_code == 200, response.text
        response_id = response.json()["id"]
        stored = requests.get(
            f"{deployment.url}/v1/responses/{response_id}", timeout=REQUEST_TIMEOUT_S
        )
        assert stored.status_code == 404, stored.text
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


def test_version_direct(deployment):
    _direct_completion(deployment, "generation")
    response = requests.get(deployment.gen.url_for("version"), timeout=REQUEST_TIMEOUT_S)
    assert response.status_code == 200, response.text
    assert response.json().get("version"), response.text


def test_health_direct(deployment):
    _direct_completion(deployment, "generation")
    for role in ("context", "generation"):
        worker = deployment.workers[role][0]
        response = requests.get(worker.url_for("health"), timeout=REQUEST_TIMEOUT_S)
        assert response.status_code == 200, (role, response.text)


def test_health_generate_direct(deployment):
    _direct_completion(deployment, "generation")
    response = requests.get(deployment.gen.url_for("health_generate"), timeout=REQUEST_TIMEOUT_S)
    assert response.status_code == 200, response.text

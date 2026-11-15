# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on case-03, the ds-engine: what the chain refuses, and what form K does instead.

The chain's worker front ends refuse, with HTTP 400 and before any Generate call, the request
parameters the ds engine does not declare; through the orchestrator the error names the context
leg, and the client sees the context worker's status and message unchanged. Form K, the
ds-engine's own deployment, accepts some of them and ignores them, and refuses others; its ds
proxy answers a worker's refusal with HTTP 500 that quotes the worker's status. Each test
checks the form its session serves. The forms, fixtures and environment are those of
``_test_openengine_ds_disagg_basic.py``.

:func:`test_startup_rejects` starts workers of its own, so it runs last, after the module's
engine group is stopped.
"""

from __future__ import annotations

import itertools
import os
import secrets
import subprocess  # nosec B404
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import requests

from tensorrt_llm.grpc.openengine_conventions import (
    COUNTER_GENERATE_RECEIVED,
    DISAGG_REQUEST_ID_MIN,
)

from . import _test_openengine_ds_disagg_basic as basic

MODEL_NAME = basic.MODEL_NAME
# The module's fixtures.
model_path = basic.model_path
form = basic.form
groups = basic.groups
served = basic.served

pytestmark = [pytest.mark.threadleak(enabled=False)]

COMPLETIONS = "/v1/completions"
CHAT = "/v1/chat/completions"
REQUEST_TIMEOUT_S = 300
FRONTEND_EXIT_TIMEOUT_S = 600
CTX_LEG = "the context leg (context_only) of a disaggregated request"
# Form K's per-token logprob when the engine returns none.
PLACEHOLDER_LOGPROB = -9999.0
# A 1x1 PNG, inline, so that no side fetches anything.
PNG_DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "2mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
IMAGE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
            {"type": "text", "text": "What is in this image?"},
        ],
    }
]
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

# What form K does with a parameter the chain refuses.
K_ACCEPTS = "accepts"
K_PLACEHOLDER = "placeholder logprobs"
K_REFUSES = "refuses"
# Parameter -> (path, request fields, text the chain's refusal names, what form K does).
UNSUPPORTED = {
    "image_url": (CHAT, {"messages": IMAGE_MESSAGES}, "image inputs", K_REFUSES),
    "repetition_penalty": (
        COMPLETIONS,
        {"repetition_penalty": 1.2},
        "repetition_penalty",
        K_ACCEPTS,
    ),
    "presence_penalty": (COMPLETIONS, {"presence_penalty": 0.5}, "presence_penalty", K_ACCEPTS),
    "frequency_penalty": (COMPLETIONS, {"frequency_penalty": 0.5}, "frequency_penalty", K_ACCEPTS),
    "min_tokens": (COMPLETIONS, {"min_tokens": 1}, "min_tokens", K_ACCEPTS),
    "logprobs": (COMPLETIONS, {"logprobs": 1}, "logprobs", K_PLACEHOLDER),
    "logit_bias": (COMPLETIONS, {"logit_bias": {"5": 1.0}}, "logit_bias", K_REFUSES),
    "guided_json": (
        COMPLETIONS,
        {"response_format": {"type": "json", "schema": {"type": "object"}}},
        "guided_decoding",
        K_ACCEPTS,
    ),
    "return_context_logits": (
        COMPLETIONS,
        {"return_context_logits": True},
        "return_context_logits",
        K_REFUSES,
    ),
    # Form K's workers run without a reasoning parser, which the budget needs there.
    "thinking_token_budget": (
        COMPLETIONS,
        {"thinking_token_budget": 8},
        "thinking_token_budget",
        K_REFUSES,
    ),
    "cache_salt": (CHAT, {"cache_salt": "salt"}, "cache_salt", K_ACCEPTS),
    "priority": (COMPLETIONS, {"priority": 0.5}, "priority", K_ACCEPTS),
    "scheduling_params": (
        CHAT,
        {"agent_hierarchy": [["planner", 1]]},
        "scheduling_params.agent_hierarchy",
        K_ACCEPTS,
    ),
    "lora": (
        COMPLETIONS,
        {"lora_request": {"lora_name": "adapter", "lora_int_id": 1, "lora_path": "/nonexistent"}},
        "lora_request",
        K_REFUSES,
    ),
}
# The ids of the context legs a test sends to a worker itself.
_DISAGG_REQUEST_IDS = itertools.count(DISAGG_REQUEST_ID_MIN + secrets.randbelow(1 << 30) * 1024)


def _body(path: str, fields: dict) -> dict:
    """A request of ``path`` with ``fields``, short and greedy unless ``fields`` says otherwise."""
    body = {"model": MODEL_NAME, "max_tokens": 4, "temperature": 0.0}
    if path == CHAT:
        body["messages"] = basic.CHAT_MESSAGES
        body.update(basic.NO_THINKING)
    else:
        body["prompt"] = basic.COMPLETION_PROMPT
    return {**body, **fields}


def _post(url: str, path: str, body: dict) -> requests.Response:
    return requests.post(f"{url}{path}", json=body, timeout=REQUEST_TIMEOUT_S)


def _message(response: requests.Response) -> str:
    """The error message of a worker front end, the orchestrator or a ds server."""
    try:
        body = response.json()
    except ValueError:
        return response.text
    if not isinstance(body, dict):
        return response.text
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")
    return str(body.get("message") or body.get("detail") or error or "")


def _context_leg(body: dict) -> dict:
    """``body`` as the context leg of a disaggregated request, under an id of its own."""
    params = {"request_type": "context_only", "disagg_request_id": next(_DISAGG_REQUEST_IDS)}
    return {**body, "disaggregated_params": params}


def _generate_calls(served: basic.Served) -> dict[str, int]:
    """The Generate calls each of the chain's engines has received; empty in form K."""
    if not served.chain:
        return {}
    return {
        role: basic.engine_counter(endpoint, COUNTER_GENERATE_RECEIVED)
        for role, endpoint in served.engines.items()
    }


def _k_relays_refusal(response: requests.Response) -> None:
    """Form K's ds proxy answers a worker's HTTP 400 with HTTP 500 that quotes the 400."""
    assert response.status_code == 500, (response.status_code, response.text[:2000])
    assert "HTTP 400" in _message(response), response.text[:2000]


def _chain_refuses(served: basic.Served, response: requests.Response, *named: str) -> str:
    """``response`` through the orchestrator is the context worker's 400 naming ``named``."""
    assert response.status_code == 400, response.text[:2000]
    message = _message(response)
    assert message.startswith(f"{CTX_LEG}: "), response.text[:2000]
    for word in named:
        assert word in message, response.text[:2000]
    return message


@pytest.mark.parametrize("name", list(UNSUPPORTED))
def test_engine_unsupported_params_rejected(served, name):
    """NT-119 (F-10, F-13, F-15, F-20, F-22, F-23, F-26, F-29, F-35, F-44a, F-44b, F-46, DS-S09,
    DS-I01, DS-I03, DS-I04, DG-41, DG-88): a parameter the ds engine does not serve.

    The chain refuses it with HTTP 400 naming the context leg and the parameter, before any
    Generate call; the client sees the context worker's own status and message unchanged.
    Form K accepts it (ignored; logprobs come back as placeholders) or refuses it.
    """
    path, fields, named, k_does = UNSUPPORTED[name]
    body = _body(path, fields)
    before = _generate_calls(served)
    response = _post(served.url, path, body)
    if not served.chain:
        if k_does == K_REFUSES:
            _k_relays_refusal(response)
            return
        assert response.status_code == 200, response.text[:2000]
        if k_does == K_PLACEHOLDER:
            logprobs = response.json()["choices"][0]["logprobs"]["token_logprobs"]
            assert logprobs and set(logprobs) == {PLACEHOLDER_LOGPROB}, logprobs
        return
    message = _chain_refuses(served, response, named)
    direct = _post(served.workers["context"], path, _context_leg(body))
    assert (direct.status_code, _message(direct)) == (400, message), direct.text[:2000]
    assert _generate_calls(served) == before


# Parameter -> request fields; "direct_" ones go to a worker itself as an aggregated request.
N_BEST_OF = {
    "n2": {"n": 2},
    "best_of2": {"n": 1, "best_of": 2},
    "direct_best_of_gt_n": {"n": 1, "best_of": 2},
}
# Sampled: form K refuses best_of > 1 with greedy decoding on its own.
SAMPLED = {"temperature": 0.8, "top_p": 0.95}


@pytest.mark.parametrize("name", list(N_BEST_OF))
def test_n_best_of_rejected(served, name):
    """NT-120 (F-19, DS-U01, DS-I02, DG-88): more than one sequence per request.

    The chain answers 400 before any Generate call: through the orchestrator naming the
    context leg, and sent to a worker itself as an aggregated request. Form K fails the
    two-leg request on its generation hop, and serves the aggregated ``best_of > n`` request
    with ``n`` choices.
    """
    direct = name.startswith("direct_")
    url = served.workers["context"] if direct else served.url
    body = _body(COMPLETIONS, {**SAMPLED, "max_tokens": 8, **N_BEST_OF[name]})
    before = _generate_calls(served)
    response = _post(url, COMPLETIONS, body)
    if not served.chain:
        if direct:
            assert response.status_code == 200, response.text[:2000]
            assert len(response.json()["choices"]) == 1, response.text[:2000]
        else:
            # The ds proxy names the worker whose call failed: the generation worker.
            gen_port = urlsplit(served.workers["generation"]).port
            assert response.status_code == 500, response.text[:2000]
            assert f":{gen_port}/" in _message(response), response.text[:2000]
        return
    if direct:
        assert response.status_code == 400, response.text[:2000]
        assert "best_of" in _message(response), response.text[:2000]
    else:
        fields = N_BEST_OF[name]
        named = f"best_of={fields['best_of']}" if "best_of" in fields else f"n={fields['n']}"
        _chain_refuses(served, response, named)
    assert _generate_calls(served) == before


def test_profile_endpoints_rejected_direct(served):
    """NT-122 (F-47c): ``/start_profile`` and ``/stop_profile`` sent to each worker itself.

    A worker front end answers 501: the profiler runs where the engine runs. Form K's ds
    workers answer 200.
    """
    expected = 501 if served.chain else 200
    before = _generate_calls(served)
    for role, url in served.workers.items():
        for path in ("/start_profile", "/stop_profile"):
            response = requests.post(f"{url}{path}", timeout=60)
            assert response.status_code == expected, (role, path, response.text[:2000])
    assert _generate_calls(served) == before


def test_tool_choice_required_rejected(served):
    """NT-123 (DG-63): ``tool_choice: required`` is refused by both forms, before any Generate call.

    The chain answers 4xx; form K's worker answers 400, which its ds proxy relays as 500.
    """
    body = _body(
        CHAT,
        {
            "messages": [
                {"role": "user", "content": "What is the weather in Paris? Use the tool."}
            ],
            "tools": [WEATHER_TOOL],
            "tool_choice": "required",
            "max_tokens": 64,
        },
    )
    before = _generate_calls(served)
    response = _post(served.url, CHAT, body)
    if not served.chain:
        _k_relays_refusal(response)
        assert "tool_choice" in _message(response), response.text[:2000]
        return
    assert 400 <= response.status_code < 500, (response.status_code, response.text[:2000])
    assert _generate_calls(served) == before


# Parameter -> what the context worker is started with: configuration keys, or options.
STARTUP = {
    "beam": {"config": {"max_beam_width": 2}},
    "resource_governor": {"config": {"enable_resource_governor": True}},
    "sleep_mode": {"args": ["--enable_sleep_mode"]},
    "otlp": {"args": ["--otlp_traces_endpoint"]},
}
K_BEAM_WARNING = "max_beam_width > 1 (beam search) is not supported by dsv4-serve; ignored."
SLEEP_MODE_REFUSAL = "--openengine-endpoint cannot be combined with --enable_sleep_mode"


def _serve_command() -> list[str]:
    entry = Path(sys.executable).parent / "trtllm-serve"
    if entry.is_file():
        return [str(entry)]
    return [sys.executable, "-m", "tensorrt_llm.commands.serve"]


def _context_worker(model: str, workdir: Path, config: dict, args=()):
    """The chain's context worker of the module's default group, with ``config`` and ``args`` added."""
    from .openai_server import RemoteOpenAIServer

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": basic.ROLE_GPUS["context"]}
    return RemoteOpenAIServer(
        model,
        env=env,
        cli_args=[*basic.worker_serve_args("context"), *args],
        extra_config={**basic.DEFAULT_GROUP.config("context"), **config},
        log_path=str(workdir / "output_context.log"),
    )


K_BEAM_GROUP = basic.Group(context=STARTUP["beam"]["config"])


@pytest.fixture
def startup_served(groups, request) -> basic.Served | None:
    """What :func:`test_startup_rejects` runs on, set up before the test starts.

    The chain: nothing; the module's engine group is stopped, so that the test's own workers
    have the GPUs. Form K: the engine group whose context worker has ``max_beam_width > 1``
    (the other parameters are skipped).
    """
    name = request.node.callspec.params["name"]
    if groups.form == basic.FORM_CHAIN:
        groups.close()
        return None
    if name != "beam":
        pytest.skip("form K: only the beam width is compared")
    return groups.get(K_BEAM_GROUP)


@pytest.mark.parametrize("name", list(STARTUP))
def test_startup_rejects(startup_served, model_path, tmp_path, name):
    """NT-121 (F-25, F-45, F-47a, F-47b): worker options the chain does not serve.

    A context worker started with ``max_beam_width > 1`` or the resource governor does not
    start: its front end refuses the option, and the chain's expectations table reads the
    reason from its log. A worker front end with ``--enable_sleep_mode`` exits at its option
    check. With an OTLP endpoint the worker starts and serves: tracing covers the front end
    only. Form K is compared on the beam width only: its workers start, warn and ignore it.
    The workers under test are started by the test itself, after the fixture stopped the
    module's engine group.
    """
    if startup_served is not None:
        log = startup_served.native.log("context").read_text(errors="replace")
        assert K_BEAM_WARNING in log
        text, _, _ = basic.generate(startup_served, "completions", False, max_tokens=4)
        assert text, text
        return
    if name == "sleep_mode":
        # The option check runs before the front end contacts the engine at the endpoint.
        endpoint = f"127.0.0.1:{basic.free_ports(1)[0]}"
        result = subprocess.run(  # nosec B603
            [
                *_serve_command(),
                model_path,
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--openengine-endpoint",
                endpoint,
                "--server_role",
                "context",
                *basic.TOKENIZER_ARGS,
                *STARTUP[name]["args"],
            ],
            capture_output=True,
            text=True,
            timeout=FRONTEND_EXIT_TIMEOUT_S,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TRTLLM_RUNTIME_CONTROL_API_KEY": "key"},
        )
        output = result.stdout + result.stderr
        assert result.returncode != 0, output[-4000:]
        assert SLEEP_MODE_REFUSAL in output, output[-4000:]
        return
    if name == "otlp":
        sink = f"http://127.0.0.1:{basic.free_ports(1)[0]}"
        with _context_worker(model_path, tmp_path, {}, [*STARTUP[name]["args"], sink]) as worker:
            response = _post(worker.url_root, COMPLETIONS, _body(COMPLETIONS, {}))
            assert response.status_code == 200, response.text[:2000]
        return
    # The harness raises when the front end refuses to start.
    with _context_worker(model_path, tmp_path, STARTUP[name]["config"]):
        pytest.fail(f"a context worker started with {STARTUP[name]}")

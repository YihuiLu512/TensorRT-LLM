# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The trtllm-serve frontend in front of an OpenEngine engine, end to end.

Most tests here check behaviour that only the OpenEngine frontend has and run
only when ``TRTLLM_TEST_OPENENGINE_CHAIN`` selects the chain harness
(``tests/integration/defs/openengine_chain``). ``test_chat_template_kwargs``
also runs on the in-process server.
"""

import os
import signal
import time

import grpc
import pytest
import requests
from transformers import AutoTokenizer

from tensorrt_llm.grpc.openengine.bindings import model_pb2, openengine_pb2_grpc, server_pb2

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

# "none" runs the in-process server through the harness.
CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
chain_only = pytest.mark.skipif(
    not CHAIN, reason="checks the OpenEngine frontend; runs on the chain only"
)

# How long a frontend takes to stop when its engine stops answering: the first
# failed poll ends after the 5 s poll interval and the 5 s Health RPC timeout,
# the engine is declared lost 30 s later, and shutdown gets the rest.
ENGINE_LOST_STOP_S = 5 + 5 + 30 + 90
# Endpoints that need an engine in the frontend's process; the frontend answers 501.
UNSUPPORTED_ENDPOINTS = (
    ("POST", "/start_profile"),
    ("POST", "/stop_profile"),
    ("POST", "/release_memory"),
    ("POST", "/resume_memory"),
    ("GET", "/memory_status"),
    ("POST", "/update_weights"),
)


@pytest.fixture(scope="module")
def model_name() -> str:
    return "Qwen3/Qwen3-0.6B"


def _server_args() -> list[str]:
    return ["--backend", "pytorch", "--kv_cache_free_gpu_memory_fraction", "0.2"]


@pytest.fixture(scope="module")
def server(model_name: str):
    with RemoteOpenAIServer(get_model_path(model_name), _server_args()) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


def _engine_control(server: RemoteOpenAIServer):
    """A Control stub on the engine of ``server``'s chain."""
    channel = grpc.insecure_channel(server._chain_handle.engine_endpoint)
    return openengine_pb2_grpc.ControlStub(channel)


def _server_info(server: RemoteOpenAIServer) -> dict:
    response = requests.get(server.url_for("server_info"), timeout=30)
    assert response.status_code == 200, response.text
    return response.json()


@chain_only
def test_server_info_engine_block(server: RemoteOpenAIServer, client, model_name: str):
    engine = _engine_control(server).GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=30)
    before = _server_info(server)["engine"]
    assert before["kind"] == "openengine"
    assert before["endpoint"] == server._chain_handle.engine_endpoint
    assert before["engine_name"] == engine.engine_name
    assert before["engine_version"] == engine.engine_version
    assert before["instance_id"] == engine.instance_id

    client.completions.create(model=model_name, prompt="Hello", max_tokens=2, temperature=0.0)

    after = _server_info(server)["engine"]
    assert after["generate_sent"] == before["generate_sent"] + 1
    assert after["instance_id"] == before["instance_id"]


@chain_only
@pytest.mark.parametrize("method, path", UNSUPPORTED_ENDPOINTS)
def test_unsupported_endpoints(server: RemoteOpenAIServer, method: str, path: str):
    response = requests.request(method, server.url_for(path.lstrip("/")), timeout=30)
    assert response.status_code == 501, response.text
    assert "OpenEngine" in response.json()["message"]


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_chat_template_kwargs(
    server: RemoteOpenAIServer, client, model_name: str, enable_thinking: bool
):
    """``chat_template_kwargs`` reach the template; on the chain, both sides use one tokenizer."""
    model_path = get_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    messages = [{"role": "user", "content": "What is 1 + 1?"}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
    )
    expected_prompt_tokens = len(tokenizer.encode(text, add_special_tokens=False))

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=4,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    assert response.usage.prompt_tokens == expected_prompt_tokens

    if CHAIN:
        from tensorrt_llm.grpc.openengine_conventions import (
            compare_tokenizer_fingerprints,
            compute_tokenizer_fingerprint,
            model_capabilities_from_proto,
        )
        from tensorrt_llm.tokenizer import TransformersTokenizer

        control = _engine_control(server)
        server_info = control.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=30)
        model_info = control.GetModelInfo(model_pb2.GetModelInfoRequest(), timeout=30)
        engine_fingerprint = model_capabilities_from_proto(
            server_info, model_info
        ).tokenizer_fingerprint
        assert engine_fingerprint is not None
        local = compute_tokenizer_fingerprint(TransformersTokenizer(tokenizer), model_path)
        assert compare_tokenizer_fingerprints(local, engine_fingerprint) == []


def _running_requests(server: RemoteOpenAIServer) -> int:
    load = _engine_control(server).GetLoad(server_pb2.GetLoadRequest(), timeout=30)
    return load.running_requests


@chain_only
def test_disconnect_ends_the_engine_request(server: RemoteOpenAIServer, model_name: str):
    """A client that goes away mid-stream leaves no request running on the engine."""
    deadline = time.monotonic() + 60
    while _running_requests(server) != 0:
        assert time.monotonic() < deadline, "the engine never became idle"
        time.sleep(0.5)

    body = {
        "model": model_name,
        "prompt": "Write a long story about a lighthouse.",
        "max_tokens": 4000,
        "ignore_eos": True,
        "stream": True,
    }
    with requests.post(
        server.url_for("v1/completions"), json=body, stream=True, timeout=60
    ) as response:
        assert response.status_code == 200
        chunks = 0
        for line in response.iter_lines():
            if line.startswith(b"data: "):
                chunks += 1
                if chunks == 3:
                    break
        assert chunks == 3
        assert _running_requests(server) >= 1
    # Leaving the block closes the connection with the stream unread.

    deadline = time.monotonic() + 30
    while _running_requests(server) != 0:
        assert time.monotonic() < deadline, "the engine request outlived the client"
        time.sleep(0.2)


@chain_only
def test_frontend_stops_itself_when_the_engine_is_lost(model_name: str):
    """The frontend stops on its own once the engine stops answering.

    The engine is frozen rather than killed, so its processes can still be
    stopped normally afterwards.
    """
    with RemoteOpenAIServer(get_model_path(model_name), _server_args()) as remote_server:
        handle = remote_server._chain_handle
        client = remote_server.get_client()
        client.completions.create(model=model_name, prompt="Hello", max_tokens=2, temperature=0.0)

        engine_group = handle.engine.pid
        os.killpg(engine_group, signal.SIGSTOP)
        try:
            start = time.monotonic()
            saw_unhealthy = False
            while handle.frontend.poll() is None:
                assert time.monotonic() - start < ENGINE_LOST_STOP_S, (
                    "the frontend kept running without its engine"
                )
                try:
                    health = requests.get(remote_server.url_for("health"), timeout=5)
                    saw_unhealthy |= health.status_code == 503
                except requests.RequestException:
                    pass
                time.sleep(1)
        finally:
            os.killpg(engine_group, signal.SIGCONT)
        assert saw_unhealthy
        with open(handle.frontend_log, errors="replace") as log:
            assert "stopping this frontend" in log.read()

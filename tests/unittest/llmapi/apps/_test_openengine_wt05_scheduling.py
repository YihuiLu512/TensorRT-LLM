# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-level options end to end: priority, agent_hierarchy, kv_hints and a scan of more options.

The tests run on the in-process server and, when ``TRTLLM_TEST_OPENENGINE_CHAIN``
selects the chain harness (``tests/integration/defs/openengine_chain``), through
the OpenEngine frontend to the engine. There, a request with an option the
engine declares must reach the engine, and one with an option it does not
declare must be refused with 400 before any engine call. The engine's request
counter shows whether a request arrived, not whether each option did.
"""

import contextlib
import os
import threading
import time

import grpc
import pytest
import requests
import yaml

from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.grpc.openengine.bindings import model_pb2, openengine_pb2_grpc, server_pb2

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

# "none" runs the in-process server through the harness.
CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
TRTLLM_CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN") == "trtllm"
MODEL = "Qwen3/Qwen3-0.6B"
TIMEOUT_S = 120

# Ids above 2**53 lose precision as doubles; the HTTP path admits up to 2**64 - 1.
U64_BLOCK_HASHES = [2**64 - 1, 2**53 + 1, 11]


def _envelope(payload: dict) -> dict:
    return {
        "protocol_version": "0.1",
        "message_id": "wt05-hints",
        "actions": [
            {
                "action_id": "a-1",
                "action_type": "kv.fetch",
                "action_version": "1.0",
                "payload": payload,
            }
        ],
    }


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    # One batch slot keeps later requests waiting, so the priority queue orders them.
    options = tmp_path_factory.mktemp("wt05") / "options.yaml"
    options.write_text(
        yaml.safe_dump(
            {
                "max_batch_size": 1,
                "scheduler_config": {"waiting_queue_policy": "priority"},
                "cuda_graph_config": None,
            }
        )
    )
    args = [
        "--backend",
        "pytorch",
        "--kv_cache_free_gpu_memory_fraction",
        "0.2",
        "--reasoning_parser",
        "qwen3",
        "--extra_llm_api_options",
        str(options),
    ]
    # Without it the HTTP ingress strips kv_hints before the request is mapped.
    env = {**os.environ, "TRTLLM_TRUST_KV_HINTS": "1"}
    with RemoteOpenAIServer(get_model_path(MODEL), args, env=env) as remote_server:
        yield remote_server


def _post(server: RemoteOpenAIServer, path: str, body: dict) -> requests.Response:
    return requests.post(server.url_for(path), json={"model": MODEL, **body}, timeout=TIMEOUT_S)


def _completion(server: RemoteOpenAIServer, **body) -> requests.Response:
    return _post(
        server, "v1/completions", {"prompt": "The capital of France is", "max_tokens": 4, **body}
    )


def _chat(server: RemoteOpenAIServer, **body) -> requests.Response:
    messages = [{"role": "user", "content": "What is 1 + 1?"}]
    return _post(
        server, "v1/chat/completions", {"messages": messages, "max_completion_tokens": 8, **body}
    )


@contextlib.contextmanager
def _control(server: RemoteOpenAIServer):
    with grpc.insecure_channel(server._chain_handle.engine_endpoint) as channel:
        yield openengine_pb2_grpc.ControlStub(channel)


def _capabilities(server: RemoteOpenAIServer) -> conv.ModelCapabilities:
    with _control(server) as control:
        return conv.model_capabilities_from_proto(
            control.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=30),
            control.GetModelInfo(model_pb2.GetModelInfoRequest(), timeout=30),
        )


def _generate_received(server: RemoteOpenAIServer) -> int:
    with _control(server) as control:
        load = control.GetLoad(server_pb2.GetLoadRequest(), timeout=30)
    counters = conv.engine_counters(dict(load.attributes))
    assert counters is not None, "the engine reports no Generate counters"
    return counters[0]


def _assert_taken_or_refused(server: RemoteOpenAIServer, send, declared: bool, name: str) -> None:
    """On the chain, a request with a declared option reaches the engine; else it is a 400."""
    if not CHAIN:
        response = send()
        assert response.status_code == 200, (name, response.text)
        return
    before = _generate_received(server)
    response = send()
    after = _generate_received(server)
    if declared:
        assert response.status_code == 200, (name, response.text)
        assert after > before, f"{name}: the request never reached the engine"
    else:
        assert response.status_code == 400, (name, response.status_code, response.text)
        assert after == before, f"{name}: refused, yet the engine received a request"


def test_priority_reorders_waiting_requests(server: RemoteOpenAIServer):
    """With one batch slot busy, a later high-priority request overtakes a low-priority one."""
    blocker_started = threading.Event()
    finished: list[str] = []
    errors: list[Exception] = []

    def blocker():
        body = {
            "model": MODEL,
            "prompt": "Write a long story about a lighthouse.",
            "max_tokens": 2000,
            "ignore_eos": True,
            "stream": True,
        }
        try:
            with requests.post(
                server.url_for("v1/completions"), json=body, stream=True, timeout=TIMEOUT_S
            ) as response:
                assert response.status_code == 200, response.text
                for line in response.iter_lines():
                    if line.startswith(b"data: "):
                        blocker_started.set()
        except Exception as error:  # reported by the main thread
            errors.append(error)
        finally:
            blocker_started.set()

    def queued(name: str, priority: float):
        try:
            # Long enough that the later of the two finishes many steps after the other.
            response = _completion(server, priority=priority, max_tokens=64, ignore_eos=True)
            assert response.status_code == 200, response.text
            finished.append(name)
        except Exception as error:  # reported by the main thread
            errors.append(error)

    threads = [threading.Thread(target=blocker)]
    threads[0].start()
    assert blocker_started.wait(TIMEOUT_S), "the blocking request never started"
    assert not errors, errors
    threads.append(threading.Thread(target=queued, args=("low", 0.1)))
    threads[-1].start()
    # Let the low-priority request reach the waiting queue first.
    time.sleep(0.5)
    threads.append(threading.Thread(target=queued, args=("high", 0.9)))
    threads[-1].start()
    for thread in threads:
        thread.join(TIMEOUT_S)
        assert not thread.is_alive()

    assert not errors, errors
    assert finished == ["high", "low"]


def test_agent_hierarchy_reaches_the_engine_or_is_refused(server: RemoteOpenAIServer):
    """A request with ``agent_hierarchy`` reaches an engine that declares scheduling hints."""
    declared = CHAIN and _capabilities(server).supports(conv.CAP_REQ_SCHEDULING)
    _assert_taken_or_refused(
        server,
        lambda: _chat(server, agent_hierarchy=[["planner", 1]]),
        declared,
        "agent_hierarchy",
    )


def test_kv_hints_with_integers_beyond_double_precision(server: RemoteOpenAIServer):
    """Hints with u64 block hashes are accepted, as they are over HTTP in process.

    The engine refuses hints in the object form once a number reaches 2**53,
    so the chain cannot pass with rounded block hashes.
    """
    envelope = _envelope({"block_hashes": U64_BLOCK_HASHES, "ratio": 2.0})
    declared = CHAIN and _capabilities(server).supports(conv.CAP_REQ_KV_HINTS)
    if TRTLLM_CHAIN:
        assert declared, "the TRT-LLM engine declares trtllm.request.kv_hints"
    _assert_taken_or_refused(
        server, lambda: _completion(server, kv_hints=envelope), declared, "kv_hints"
    )


# name -> (endpoint, request fields, whether the engine declares the option)
PROBES = {
    "priority": ("completions", {"priority": 0.9}, lambda caps: caps.supports_priority),
    "agent_hierarchy": (
        "chat",
        {"agent_hierarchy": [["planner", 1]]},
        lambda caps: caps.supports(conv.CAP_REQ_SCHEDULING),
    ),
    "kv_hints": (
        "completions",
        {"kv_hints": _envelope({"block_hashes": [11, 22]})},
        lambda caps: caps.supports(conv.CAP_REQ_KV_HINTS),
    ),
    "logit_bias": (
        "completions",
        {"logit_bias": {"100": 5}},
        lambda caps: caps.supports(conv.CAP_REQ_LOGIT_BIAS),
    ),
    "thinking_token_budget": (
        "chat",
        {"thinking_token_budget": 4},
        lambda caps: caps.supports(conv.CAP_REQ_THINKING_TOKEN_BUDGET),
    ),
    "best_of": (
        "completions",
        {"n": 1, "best_of": 2, "temperature": 1.0},
        lambda caps: caps.supports(conv.CAP_REQ_BEST_OF),
    ),
    "cache_salt": ("chat", {"cache_salt": "tenant-a"}, lambda caps: caps.supports_cache_salt),
}


@pytest.mark.parametrize("name", sorted(PROBES))
def test_option_reaches_the_engine_or_is_refused(server: RemoteOpenAIServer, name: str):
    """A request with each probed option reaches the engine or is refused with 400."""
    if name == "best_of" and not CHAIN:
        pytest.skip("best_of=2 needs two batch slots; this server keeps one for the priority test")
    endpoint, fields, supported = PROBES[name]
    declared = CHAIN and supported(_capabilities(server))
    send = _chat if endpoint == "chat" else _completion
    _assert_taken_or_refused(server, lambda: send(server, **fields), declared, name)

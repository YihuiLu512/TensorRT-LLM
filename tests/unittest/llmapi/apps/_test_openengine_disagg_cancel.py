# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A client that leaves cancels the Generate call of its leg on that leg's engine (SV-DSG-10).

An orchestrator and a context and a generation worker, each started with
``--server_role``, register through the orchestrator's cluster; on the
TensorRT-LLM OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN=trtllm``,
``tests/integration/defs/openengine_chain``) each worker is a frontend and a
TensorRT-LLM OpenEngine server. The tests watch the engines over
``Control.GetLoad``: ``running_requests`` (the Generate calls the engine has
accepted and not ended, waiting ones included), the Generate calls it
received and the PrefillReady events it sent. The context engine takes one
request at a time and prefills a long prompt in small chunks, so that a
context leg can be left while it waits or while it prefills; a test checks
that the leg still waits or prefills when its client leaves.

- NT-089 (DG-38): a streamed request left at the orchestrator, once its
  generation leg decodes, ends that leg's call well before its last token.
- NT-090 (DG-39): a request left at the orchestrator unstreamed runs to its
  end: the orchestrator does not notice, and records both legs as completed.
  This test also runs on the in-process workers, where only the
  orchestrator's counters are read.
- NT-092 (DG-39c): a context leg left while it prefills ends its call before
  the engine hands off.
- NT-197 (EG-16a): the cancel reaches the engine in each stage: a context leg
  waiting behind another, a context leg prefilling, and a generation leg
  decoding.

After their cancels, NT-089 and NT-197 check that the engines serve a new
request. The chain is this module's own, and stops with it.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import contextlib
import http.client
import itertools
import json
import os
import time
import urllib.parse
from dataclasses import dataclass

import pytest
import requests
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._utils import get_free_ports
from tensorrt_llm.grpc.openengine_conventions import (
    COUNTER_GENERATE_RECEIVED,
    COUNTER_PREFILL_READY_SENT,
    DISAGG_REQUEST_ID_MIN,
)

from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

__extra_import_path__ = ["~/tests/integration/defs"]
from openengine_chain import direct_replay as dr  # noqa: E402
from openengine_chain import launch  # noqa: E402
from openengine_chain.conventions import CHAIN_TRTLLM  # noqa: E402

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN") == CHAIN_TRTLLM
chain_only = pytest.mark.skipif(
    not CHAIN, reason="the test watches the engines of the TensorRT-LLM OpenEngine chain"
)
pytestmark = [pytest.mark.threadleak(enabled=False)]

MODEL_SUBDIR = os.path.join("Qwen3", "Qwen3-0.6B")
PROMPT = "The capital of France is Paris. The capital of Germany is"
ENGINE_READY_TIMEOUT_S = 1200
REQUEST_TIMEOUT_S = 300
REGISTER_TIMEOUT_S = 120
WORKER_CONFIG = {
    "cache_transceiver_config": {"backend": "DEFAULT"},
    "kv_cache_config": {"free_gpu_memory_fraction": 0.2, "enable_block_reuse": False},
    "disable_overlap_scheduler": True,
    "cuda_graph_config": None,
}
# The context engine takes one request at a time and prefills in chunks of 32
# tokens (one KV-cache block), so that a LONG_PROMPT keeps it busy for seconds.
CTX_CONFIG = {"max_batch_size": 1, "enable_chunked_prefill": True, "max_num_tokens": 32}
LONG_PROMPT = [1000 + i % 20000 for i in range(32000)]
# A generation leg this long decodes for far longer than a cancel takes to arrive.
LONG_DECODE = 8000
# How long an engine may take to end a call whose client left (the worker
# frontend checks for a gone client about once a second), and how long NT-090
# waits before it checks that its call still runs.
CANCEL_ARRIVES_S = 10.0
STILL_RUNNING_S = 3.0
_DISAGG_REQUEST_IDS = itertools.count(DISAGG_REQUEST_ID_MIN + 1)


def _model_dir() -> str:
    root = llm_models_root()
    path = None if root is None else os.path.join(str(root), MODEL_SUBDIR)
    assert path is not None and os.path.isdir(path), f"{MODEL_SUBDIR} is not under LLM_MODELS_ROOT"
    return path


def _needs_gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (9, 0):
        pytest.skip("needs a GPU of Hopper or later")


@dataclass
class Chain:
    model: str
    orchestrator: RemoteDisaggOpenAIServer
    ctx: RemoteOpenAIServer
    gen: RemoteOpenAIServer


def _wait_registered(orchestrator: RemoteDisaggOpenAIServer) -> None:
    """Wait until the orchestrator lists a context and a generation worker."""
    deadline = time.monotonic() + REGISTER_TIMEOUT_S
    while True:
        try:
            lists = requests.get(orchestrator.url_for("cluster_info"), timeout=10).json()
            lists = lists["server_lists"]
            if lists["context"] and lists["generation"]:
                return
        except (requests.RequestException, ValueError, KeyError):
            pass
        assert time.monotonic() < deadline, "the workers did not register with the orchestrator"
        time.sleep(1)


@contextlib.contextmanager
def _chain(directory):
    """An orchestrator, a context worker with CTX_CONFIG and a generation worker, registered."""
    _needs_gpu()
    model = _model_dir()
    disagg_port, ctx_port, gen_port = get_free_ports(3)
    # On the chain the harness picks the engines' ports as the workers start: not these three.
    launch.reserve_ports(disagg_port, ctx_port, gen_port)
    cluster = {"cluster_uri": f"http://localhost:{disagg_port}", "cluster_name": ""}
    # Both workers run on this host: loopback is the only TCP device UCX needs.
    env = {**os.environ, "UCX_NET_DEVICES": "lo"}

    def worker(role: str, port: int, overrides: dict) -> RemoteOpenAIServer:
        return RemoteOpenAIServer(
            model,
            port=port,
            env=env,
            cli_args=["--server_role", role],
            extra_config={**WORKER_CONFIG, "disagg_cluster": cluster, **overrides},
            log_path=str(directory / f"output_{role}.log"),
            wait=False,
        )

    disagg_config = {"hostname": "localhost", "port": disagg_port, "disagg_cluster": cluster}
    with worker("context", ctx_port, CTX_CONFIG) as ctx, worker("generation", gen_port, {}) as gen:
        with RemoteDisaggOpenAIServer(
            ctx_servers=[], gen_servers=[], port=disagg_port, disagg_config=disagg_config
        ) as orchestrator:
            for server in (ctx, gen):
                server.wait_for_server(timeout=ENGINE_READY_TIMEOUT_S)
            _wait_registered(orchestrator)
            yield Chain(model, orchestrator, ctx, gen)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    with _chain(tmp_path_factory.mktemp("disagg_cancel")) as served:
        yield served


def _load(server: RemoteOpenAIServer) -> dict:
    return dr.get_load(server._chain_handle.engine_endpoint)


def _running(server: RemoteOpenAIServer) -> int:
    """The Generate calls the worker's engine has accepted and not ended, waiting ones included."""
    return int(_load(server).get("running_requests", 0))


def _counter(server: RemoteOpenAIServer, key: str) -> int:
    return int(_load(server).get("attributes", {}).get(key, 0))


def _wait(condition, what: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"timed out after {timeout} s waiting for {what}"
        time.sleep(0.2)


def _orchestrator_counts(chain: Chain) -> dict[str, int]:
    """The orchestrator's ``<role>_<name>_total`` request counters."""
    text = requests.get(chain.orchestrator.url_for("prometheus", "metrics"), timeout=10).text
    counts: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith(("ctx_", "gen_")) and "_requests_total" in line:
            name, _, value = line.partition(" ")
            name = name.split("{", 1)[0].removesuffix("_total")
            counts[name] = counts.get(name, 0) + int(float(value))
    return counts


def _context_leg_body(chain: Chain, prompt) -> dict:
    return {
        "model": chain.model,
        "prompt": prompt,
        "max_tokens": 4,
        "disaggregated_params": {
            "request_type": "context_only",
            "disagg_request_id": next(_DISAGG_REQUEST_IDS),
        },
    }


def _send(url: str, body: dict) -> http.client.HTTPConnection:
    """Send ``body`` to ``url`` and return the connection unread; closing it leaves the request."""
    parts = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=REQUEST_TIMEOUT_S)
    connection.request(
        "POST", parts.path, body=json.dumps(body), headers={"Content-Type": "application/json"}
    )
    return connection


def _leave_while_prefilling(
    chain: Chain, connection: http.client.HTTPConnection, handed_off: int
) -> None:
    """Leave the context leg on ``connection`` while it prefills; it ends without a handoff.

    The context engine runs only that leg; ``handed_off`` is its PrefillReady count before it.
    """
    _wait(lambda: _running(chain.ctx) == 1, "the prefill to run alone", REQUEST_TIMEOUT_S)
    assert _counter(chain.ctx, COUNTER_PREFILL_READY_SENT) == handed_off, (
        "the prefill ended before the client left; the prompt is too short for this GPU"
    )
    connection.close()
    _wait(lambda: _running(chain.ctx) == 0, "the prefill call to end", CANCEL_ARRIVES_S)
    assert _counter(chain.ctx, COUNTER_PREFILL_READY_SENT) == handed_off


def _leave_once_decoding(url: str, body: dict, gen: RemoteOpenAIServer) -> None:
    """Stream ``body`` from ``url``; leave once text arrived and ``gen``'s engine decodes it."""
    with requests.post(
        url,
        json={**body, "stream": True},
        headers={"Connection": "close"},
        timeout=REQUEST_TIMEOUT_S,
        stream=True,
    ) as response:
        assert response.status_code == 200, response.text
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: ") and line != "data: [DONE]":
                if json.loads(line[len("data: ") :])["choices"][0].get("text"):
                    break
        else:
            raise AssertionError("the stream ended before any text")
        _wait(lambda: _running(gen) == 1, "the decode call to run", REQUEST_TIMEOUT_S)


def _context_leg(chain: Chain) -> dict:
    response = requests.post(
        chain.ctx.url_for("v1", "completions"),
        json=_context_leg_body(chain, PROMPT),
        timeout=REQUEST_TIMEOUT_S,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _generation_leg_body(chain: Chain, ctx: dict, max_tokens: int) -> dict:
    """A generation leg for the handoff of the context leg answer ``ctx``."""
    handoff = {**ctx["choices"][0]["disaggregated_params"], "request_type": "generation_only"}
    return {
        "model": chain.model,
        "prompt": ctx["prompt_token_ids"],
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "disaggregated_params": handoff,
    }


def _serves(chain: Chain) -> None:
    """A request through the orchestrator still completes."""
    response = requests.post(
        chain.orchestrator.url_for("v1", "completions"),
        json={"model": chain.model, "prompt": PROMPT, "max_tokens": 4},
        timeout=REQUEST_TIMEOUT_S,
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["text"], response.text


@chain_only
def test_streaming_disconnect_aborts_the_gen_leg(chain):
    """NT-089 (DG-38): the orchestrator passes a streamed client's leaving on to the generation leg."""
    assert _running(chain.gen) == 0
    received = _counter(chain.gen, COUNTER_GENERATE_RECEIVED)
    body = {"model": chain.model, "prompt": PROMPT, "max_tokens": LONG_DECODE, "ignore_eos": True}
    _leave_once_decoding(chain.orchestrator.url_for("v1", "completions"), body, chain.gen)
    _wait(lambda: _running(chain.gen) == 0, "the decode call to end", CANCEL_ARRIVES_S)
    assert _counter(chain.gen, COUNTER_GENERATE_RECEIVED) == received + 1
    _serves(chain)


def test_non_streaming_disconnect_runs_both_legs_to_completion(chain):
    """NT-090 (DG-39): an unstreamed request whose client left runs to its end on both legs."""
    before = _orchestrator_counts(chain)

    def counts_grew(name: str) -> bool:
        return _orchestrator_counts(chain).get(name, 0) > before.get(name, 0)

    body = {
        "model": chain.model,
        "prompt": PROMPT,
        "max_tokens": LONG_DECODE,
        "ignore_eos": True,
    }
    connection = _send(chain.orchestrator.url_for("v1", "completions"), body)
    _wait(lambda: counts_grew("gen_total_requests"), "the generation leg", REQUEST_TIMEOUT_S)
    if CHAIN:
        _wait(lambda: _running(chain.gen) == 1, "the decode call to run", REQUEST_TIMEOUT_S)
    connection.close()
    if CHAIN:
        # Past the time a cancel takes to arrive, the generation leg still decodes.
        time.sleep(STILL_RUNNING_S)
        assert _running(chain.gen) == 1
    for role in ("ctx", "gen"):
        _wait(
            lambda: counts_grew(f"{role}_completed_requests"),
            f"the orchestrator to record the {role} leg as completed",
            REQUEST_TIMEOUT_S,
        )
    after = _orchestrator_counts(chain)
    for role in ("ctx", "gen"):
        name = f"{role}_error_requests"
        assert after.get(name, 0) == before.get(name, 0), (before, after)


@chain_only
def test_disconnect_during_prefill_aborts_the_ctx_leg(chain):
    """NT-092 (DG-39c): a context leg left while it prefills ends on the context engine."""
    handed_off = _counter(chain.ctx, COUNTER_PREFILL_READY_SENT)
    received = _counter(chain.ctx, COUNTER_GENERATE_RECEIVED)
    url = chain.ctx.url_for("v1", "completions")
    _leave_while_prefilling(chain, _send(url, _context_leg_body(chain, LONG_PROMPT)), handed_off)
    assert _counter(chain.ctx, COUNTER_GENERATE_RECEIVED) == received + 1


@chain_only
def test_abort_reaches_the_engine_at_each_stage(chain):
    """NT-197 (EG-16a): waiting, prefilling or decoding, a call whose client left ends."""
    url = chain.ctx.url_for("v1", "completions")
    handed_off = _counter(chain.ctx, COUNTER_PREFILL_READY_SENT)
    # Waiting: a short context leg behind a long one, on an engine that takes one at a time.
    long_leg = _send(url, _context_leg_body(chain, LONG_PROMPT))
    _wait(lambda: _running(chain.ctx) == 1, "the long prefill to start", REQUEST_TIMEOUT_S)
    short_leg = _send(url, _context_leg_body(chain, PROMPT))
    _wait(lambda: _running(chain.ctx) == 2, "the short leg to wait", REQUEST_TIMEOUT_S)
    assert _counter(chain.ctx, COUNTER_PREFILL_READY_SENT) == handed_off, (
        "the long prefill ended before the short leg's client left"
    )
    short_leg.close()
    _wait(lambda: _running(chain.ctx) == 1, "the waiting call to end", CANCEL_ARRIVES_S)

    # Prefilling: the long leg, which neither leg handed off before.
    _leave_while_prefilling(chain, long_leg, handed_off)

    # Decoding: a generation leg streamed and left once it decodes.
    body = _generation_leg_body(chain, _context_leg(chain), LONG_DECODE)
    _leave_once_decoding(chain.gen.url_for("v1", "completions"), body, chain.gen)
    _wait(lambda: _running(chain.gen) == 0, "the decode call to end", CANCEL_ARRIVES_S)

    # The engines serve the legs of a new request.
    body = _generation_leg_body(chain, _context_leg(chain), 4)
    response = requests.post(
        chain.gen.url_for("v1", "completions"), json=body, timeout=REQUEST_TIMEOUT_S
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["text"], response.text

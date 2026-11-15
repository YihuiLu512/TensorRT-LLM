# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A client that leaves cancels the Generate call of its leg on that leg's engine (SV-DSG-10).

An orchestrator and a context and a generation worker, each started with
``--server_role``, register through the orchestrator's cluster; on the
OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN``,
``tests/integration/defs/openengine_chain``) each worker is a frontend and a
TensorRT-LLM OpenEngine server. The tests watch the engines over
``Control.GetLoad``: ``running_requests`` (the Generate calls the engine has
accepted and not ended) and the PrefillReady events it sent, so they run on
the chain only. The context engine takes one request at a time and prefills
a long prompt in small chunks, so that a context leg can be left while it
waits or while it prefills.

- NT-089 (DG-38): a streamed request left at the orchestrator after its first
  text ends its generation leg's call well before its last token.
- NT-090 (DG-39): a request left at the orchestrator unstreamed runs to its
  end: the orchestrator does not notice, and records the generation leg as
  completed.
- NT-092 (DG-39c): a context leg left while it prefills ends its call before
  the engine hands off.
- NT-197 (EG-16a): the cancel reaches the engine in each stage: a context leg
  waiting behind another, a context leg prefilling, and a generation leg
  decoding.

The tests that send legs to a worker themselves leave some handoffs
unclaimed; the expectations table counts them
(``openengine_chain/expectations/WT-52.py``). The chain is this module's own,
and stops with it.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import contextlib
import itertools
import json
import os
import threading
import time
from dataclasses import dataclass

import pytest
import requests
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._utils import get_free_ports
from tensorrt_llm.grpc.openengine_conventions import (
    COUNTER_PREFILL_READY_SENT,
    DISAGG_REQUEST_ID_MIN,
)

from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

__extra_import_path__ = ["~/tests/integration/defs"]
from openengine_chain import direct_replay as dr  # noqa: E402

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not CHAIN, reason="the tests watch the engines of the OpenEngine chain"),
]

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
# The context engine takes one request at a time and prefills in chunks of
# CHUNK tokens, so that a LONG_PROMPT keeps it busy for seconds.
CTX_CONFIG = {"max_batch_size": 1, "enable_chunked_prefill": True, "max_num_tokens": 32}
CHUNK = 32
LONG_PROMPT = [1000 + i % 20000 for i in range(32000)]
# A generation leg this long decodes for far longer than a cancel takes to arrive.
LONG_DECODE = 8000
# How long a client waits before it leaves, and how long an engine may take to end a call
# whose client left (the worker frontend checks for a gone client about once a second).
LEAVE_AFTER_S = 1.0
CANCEL_ARRIVES_S = 10.0
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
def _chain(directory, ctx_config=None, gen_config=None, role_env=None, cluster_config=None):
    """An orchestrator and two workers with roles, registered through its cluster.

    ``ctx_config`` and ``gen_config`` add to a worker's configuration,
    ``role_env`` (role name to variables) to its environment, and
    ``cluster_config`` to the cluster settings of all three.
    """
    _needs_gpu()
    model = _model_dir()
    disagg_port, ctx_port, gen_port = get_free_ports(3)
    cluster = {
        "cluster_uri": f"http://localhost:{disagg_port}",
        "cluster_name": "",
        **(cluster_config or {}),
    }
    # Both workers run on this host: loopback is the only TCP device UCX needs.
    env = {**os.environ, "UCX_NET_DEVICES": "lo"}

    def worker(role: str, port: int, overrides) -> RemoteOpenAIServer:
        return RemoteOpenAIServer(
            model,
            port=port,
            env={**env, **(role_env or {}).get(role, {})},
            cli_args=["--server_role", role],
            extra_config={**WORKER_CONFIG, "disagg_cluster": cluster, **(overrides or {})},
            log_path=str(directory / f"output_{role}.log"),
            wait=False,
        )

    disagg_config = {"hostname": "localhost", "port": disagg_port, "disagg_cluster": cluster}
    with (
        worker("context", ctx_port, ctx_config) as ctx,
        worker("generation", gen_port, gen_config) as gen,
    ):
        with RemoteDisaggOpenAIServer(
            ctx_servers=[], gen_servers=[], port=disagg_port, disagg_config=disagg_config
        ) as orchestrator:
            for server in (ctx, gen):
                server.wait_for_server(timeout=ENGINE_READY_TIMEOUT_S)
            _wait_registered(orchestrator)
            yield Chain(model, orchestrator, ctx, gen)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    with _chain(tmp_path_factory.mktemp("disagg_cancel"), ctx_config=CTX_CONFIG) as served:
        yield served


def _load(server: RemoteOpenAIServer) -> dict:
    return dr.get_load(server._chain_handle.engine_endpoint)


def _running(server: RemoteOpenAIServer) -> int:
    """The Generate calls the worker's engine has accepted and not ended."""
    return int(_load(server).get("running_requests", 0))


def _prefill_ready_sent(server: RemoteOpenAIServer) -> int:
    return int(_load(server).get("attributes", {}).get(COUNTER_PREFILL_READY_SENT, 0))


def _wait(condition, what: str, timeout: float) -> float:
    """Wait until ``condition()`` holds; return the seconds it took."""
    start = time.monotonic()
    while not condition():
        assert time.monotonic() - start < timeout, f"timed out after {timeout} s waiting for {what}"
        time.sleep(0.2)
    return time.monotonic() - start


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


def _context_leg_body(chain: Chain, prompt, max_tokens: int = 4) -> dict:
    return {
        "model": chain.model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "disaggregated_params": {
            "request_type": "context_only",
            "disagg_request_id": next(_DISAGG_REQUEST_IDS),
        },
    }


def _leave_after(url: str, body: dict, seconds: float) -> None:
    """Send ``body`` to ``url`` and close the connection after ``seconds`` without an answer."""
    with pytest.raises(requests.exceptions.ReadTimeout):
        requests.post(url, json=body, headers={"Connection": "close"}, timeout=(5, seconds))


def _leave_after_first_text(url: str, body: dict) -> None:
    """Stream ``body`` from ``url`` and close the connection once some text arrived."""
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
                    return
    raise AssertionError("the stream ended before any text")


def _in_background(send) -> threading.Thread:
    thread = threading.Thread(target=send, daemon=True)
    thread.start()
    return thread


def _left_while_prefilling(chain: Chain) -> None:
    """A context leg with a long prompt, left while it prefills, ends before its handoff."""
    handed_off = _prefill_ready_sent(chain.ctx)
    url = chain.ctx.url_for("v1", "completions")
    _leave_after(url, _context_leg_body(chain, LONG_PROMPT), LEAVE_AFTER_S)
    _wait(lambda: _running(chain.ctx) == 0, "the prefill call to end", CANCEL_ARRIVES_S)
    assert _prefill_ready_sent(chain.ctx) == handed_off


def _decode_leg_body(chain: Chain) -> dict:
    """A generation leg for a fresh context leg's handoff, decoding LONG_DECODE tokens."""
    response = requests.post(
        chain.ctx.url_for("v1", "completions"),
        json=_context_leg_body(chain, PROMPT),
        timeout=REQUEST_TIMEOUT_S,
    )
    assert response.status_code == 200, response.text
    ctx = response.json()
    handoff = {**ctx["choices"][0]["disaggregated_params"], "request_type": "generation_only"}
    return {
        "model": chain.model,
        "prompt": ctx["prompt_token_ids"],
        "max_tokens": LONG_DECODE,
        "ignore_eos": True,
        "disaggregated_params": handoff,
    }


def test_streaming_disconnect_aborts_the_gen_leg(chain):
    """NT-089 (DG-38): the orchestrator passes a streamed client's leaving on to the generation leg."""
    assert _running(chain.gen) == 0
    body = {"model": chain.model, "prompt": PROMPT, "max_tokens": LONG_DECODE, "ignore_eos": True}
    _leave_after_first_text(chain.orchestrator.url_for("v1", "completions"), body)
    _wait(lambda: _running(chain.gen) == 0, "the decode call to end", CANCEL_ARRIVES_S)


def test_non_streaming_disconnect_runs_both_legs_to_completion(chain):
    """NT-090 (DG-39): an unstreamed request whose client left runs to its end on both legs."""
    before = _orchestrator_counts(chain)
    body = {
        "model": chain.model,
        "prompt": PROMPT,
        "max_tokens": LONG_DECODE // 2,
        "ignore_eos": True,
    }
    _leave_after(chain.orchestrator.url_for("v1", "completions"), body, LEAVE_AFTER_S)
    _wait(lambda: _running(chain.gen) == 1, "the decode call to start", REQUEST_TIMEOUT_S)
    # Long after a cancel would have arrived, the generation leg still decodes.
    time.sleep(2 * LEAVE_AFTER_S)
    assert _running(chain.gen) == 1
    _wait(lambda: _running(chain.gen) == 0, "the decode call to end", REQUEST_TIMEOUT_S)
    _wait(
        lambda: (
            _orchestrator_counts(chain).get("gen_completed_requests", 0)
            > before.get("gen_completed_requests", 0)
        ),
        "the orchestrator to record the generation leg as completed",
        30,
    )
    after = _orchestrator_counts(chain)
    for role in ("ctx", "gen"):
        name = f"{role}_error_requests"
        assert after.get(name, 0) == before.get(name, 0), (before, after)


def test_disconnect_during_prefill_aborts_the_ctx_leg(chain):
    """NT-092 (DG-39c): a context leg left while it prefills ends on the context engine."""
    _left_while_prefilling(chain)


def test_abort_reaches_the_engine_at_each_stage(chain):
    """NT-197 (EG-16a): waiting, prefilling or decoding, a call whose client left ends."""
    url = chain.ctx.url_for("v1", "completions")
    # Waiting: a short context leg behind a long one, on an engine that takes one at a time.
    handed_off = _prefill_ready_sent(chain.ctx)
    first = _in_background(
        lambda: requests.post(
            url, json=_context_leg_body(chain, LONG_PROMPT), timeout=REQUEST_TIMEOUT_S
        )
    )
    _wait(lambda: _running(chain.ctx) == 1, "the long prefill to start", REQUEST_TIMEOUT_S)
    waiting = _in_background(
        lambda: _leave_after(url, _context_leg_body(chain, PROMPT), 2 * LEAVE_AFTER_S)
    )
    _wait(lambda: _running(chain.ctx) == 2, "the short leg to wait", REQUEST_TIMEOUT_S)
    waiting.join()
    _wait(lambda: _running(chain.ctx) == 1, "the waiting call to end", CANCEL_ARRIVES_S)
    first.join(REQUEST_TIMEOUT_S)
    assert not first.is_alive()
    _wait(lambda: _running(chain.ctx) == 0, "the long prefill to end", REQUEST_TIMEOUT_S)
    # Only the long leg handed off.
    assert _prefill_ready_sent(chain.ctx) == handed_off + 1

    # Prefilling.
    _left_while_prefilling(chain)

    # Decoding: a generation leg streamed and left after its first text.
    _leave_after_first_text(chain.gen.url_for("v1", "completions"), _decode_leg_body(chain))
    _wait(lambda: _running(chain.gen) == 0, "the decode call to end", CANCEL_ARRIVES_S)

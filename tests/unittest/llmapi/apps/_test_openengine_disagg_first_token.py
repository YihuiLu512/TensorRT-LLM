# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A disaggregated request's first token reaches the client once (SV-DSG-5, SV-DSG-6).

An orchestrator and a context and a generation worker, each started with
``--server_role``, register through the orchestrator's cluster. The context
leg makes the first token and the generation leg sends it on.

NT-182 (DG-86): with greedy sampling, a streamed and an unstreamed request
through the orchestrator give the same text and the same usage; the stream
carries the first token once, and ``completion_tokens`` counts it once.

The test runs on the in-process workers as well as on the OpenEngine chain
(``TRTLLM_TEST_OPENENGINE_CHAIN``, ``tests/integration/defs/openengine_chain``),
where each worker is a frontend and a TensorRT-LLM OpenEngine server. It
records the text in the junit property ``text``, so that a run on the chain
can be compared with a run on the in-process workers.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import contextlib
import json
import os
import time
from dataclasses import dataclass

import pytest
import requests
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._utils import get_free_ports

from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

__extra_import_path__ = ["~/tests/integration/defs"]
from openengine_chain import launch  # noqa: E402

pytestmark = [pytest.mark.threadleak(enabled=False)]

MODEL_SUBDIR = os.path.join("Qwen3", "Qwen3-0.6B")
PROMPT = "The capital of France is Paris. The capital of Germany is"
ENGINE_READY_TIMEOUT_S = 1200
REQUEST_TIMEOUT_S = 120
REGISTER_TIMEOUT_S = 120
WORKER_CONFIG = {
    "cache_transceiver_config": {"backend": "DEFAULT"},
    "kv_cache_config": {"free_gpu_memory_fraction": 0.2, "enable_block_reuse": False},
    "disable_overlap_scheduler": True,
    "cuda_graph_config": None,
}
MAX_TOKENS = 16


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
    """An orchestrator and two workers with roles, registered through its cluster."""
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
    with worker("context", ctx_port, {}) as ctx, worker("generation", gen_port, {}) as gen:
        with RemoteDisaggOpenAIServer(
            ctx_servers=[], gen_servers=[], port=disagg_port, disagg_config=disagg_config
        ) as orchestrator:
            for server in (ctx, gen):
                server.wait_for_server(timeout=ENGINE_READY_TIMEOUT_S)
            _wait_registered(orchestrator)
            yield Chain(model, orchestrator, ctx, gen)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    with _chain(tmp_path_factory.mktemp("disagg_first_token")) as served:
        yield served


def _whole(chain: Chain, path: str, body: dict) -> tuple[str, dict]:
    """The text and usage of an unstreamed response."""
    response = requests.post(
        chain.orchestrator.url_for(*path.split("/")), json=body, timeout=REQUEST_TIMEOUT_S
    )
    assert response.status_code == 200, response.text
    data = response.json()
    choice = data["choices"][0]
    text = choice["text"] if "text" in choice else choice["message"]["content"]
    return text, data["usage"]


def _streamed(chain: Chain, path: str, body: dict) -> tuple[list[str], dict]:
    """The pieces of text a streamed response carries, and its usage."""
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    pieces: list[str] = []
    usage = None
    with requests.post(
        chain.orchestrator.url_for(*path.split("/")),
        json=body,
        timeout=REQUEST_TIMEOUT_S,
        stream=True,
    ) as response:
        assert response.status_code == 200, response.text
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: ") :])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                piece = choice["text"] if "text" in choice else choice["delta"].get("content")
                if piece:
                    pieces.append(piece)
    assert usage is not None, "the stream carried no usage"
    return pieces, usage


@pytest.mark.parametrize(
    "path, body",
    [
        ("v1/completions", {"prompt": PROMPT}),
        ("v1/chat/completions", {"messages": [{"role": "user", "content": PROMPT}]}),
    ],
    ids=["completions", "chat"],
)
def test_stream_and_non_stream_token_sequences_agree(chain, path, body, record_property):
    """NT-182 (DG-86): streamed and unstreamed, the same tokens; the first one comes once."""
    body = {
        "model": chain.model,
        **body,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    text, usage = _whole(chain, path, body)
    pieces, streamed_usage = _streamed(chain, path, body)
    record_property("text", text)
    assert pieces, "the stream carried no text"
    assert "".join(pieces) == text, (pieces, text)
    assert usage["completion_tokens"] == streamed_usage["completion_tokens"] == MAX_TOKENS
    assert usage["prompt_tokens"] == streamed_usage["prompt_tokens"] > 0

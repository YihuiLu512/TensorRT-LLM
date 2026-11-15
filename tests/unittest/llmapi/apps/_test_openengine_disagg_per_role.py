# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A disaggregated deployment whose roles differ: their own configuration, GPUs and environment.

No tests here. :func:`deploy` starts an orchestrator and a context and a
generation worker, each a ``trtllm-serve`` with ``--server_role``, that
register with it through its own cluster, and stops them at the test's
teardown. Each role gets the shared
configuration merged with its own (top level only), its own GPUs
(``CUDA_VISIBLE_DEVICES``; ``None`` shares the visible ones) and its own
environment. Without ``TRTLLM_TEST_OPENENGINE_CHAIN`` each worker runs its
engine in-process (the baseline form); with it, the chain harness
(``tests/integration/defs/openengine_chain``) starts each worker as a worker
frontend and a TensorRT-LLM OpenEngine server, which takes the worker's
configuration file and environment.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import secrets
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import requests
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._utils import get_free_port

from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

__extra_import_path__ = ["~/tests/integration/defs"]
from openengine_chain import direct_replay as dr  # noqa: E402
from openengine_chain import launch  # noqa: E402

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
ENGINE_READY_TIMEOUT_S = 1800
REQUEST_TIMEOUT_S = 300
REGISTER_TIMEOUT_S = 120

BASE_CONFIG: Mapping[str, Any] = {
    "cache_transceiver_config": {"backend": "DEFAULT"},
    "kv_cache_config": {"free_gpu_memory_fraction": 0.2, "enable_block_reuse": False},
    "disable_overlap_scheduler": True,
    "cuda_graph_config": None,
}
# For a worker with a GPU of its own.
OWN_GPU_KV_CACHE: Mapping[str, Any] = {"free_gpu_memory_fraction": 0.5, "enable_block_reuse": False}


@dataclasses.dataclass(frozen=True)
class Role:
    """What one role's worker gets on top of the shared settings.

    ``gpus`` lists indices into the GPUs this process sees, comma-separated.
    """

    config: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    gpus: str | None = None
    env: Mapping[str, str] = dataclasses.field(default_factory=dict)


def model_dir(subdir: str) -> str:
    root = llm_models_root()
    path = None if root is None else os.path.join(str(root), subdir)
    assert path is not None and os.path.isdir(path), f"{subdir} is not under LLM_MODELS_ROOT"
    return path


def tiktoken_env() -> dict[str, str]:
    """Where gpt-oss's tokenizer finds its encodings without the network."""
    vocab = model_dir(os.path.join("datasets", "tiktoken_vocab"))
    return {"TIKTOKEN_RS_CACHE_DIR": vocab, "TIKTOKEN_ENCODINGS_BASE": vocab}


def needs_gpus(count: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < count:
        pytest.skip(f"needs {count} GPUs")
    if any(torch.cuda.get_device_capability(i) < (9, 0) for i in range(count)):
        pytest.skip(f"needs {count} GPUs of Hopper or later")


TWO_GPUS = {"context": Role(gpus="0"), "generation": Role(gpus="1")}


def attention_dp_roles(
    generation_dp_config: Mapping[str, Any], context_dp_config: Mapping[str, Any] | None = None
) -> dict[str, Role]:
    """Two GPUs per role with attention DP, and each role's ``attention_dp_config``."""
    adp = {"tensor_parallel_size": 2, "enable_attention_dp": True}

    def config(dp_config: Mapping[str, Any] | None) -> dict[str, Any]:
        return {**adp, "attention_dp_config": dict(dp_config)} if dp_config else adp

    return {
        "context": Role(config=config(context_dp_config), gpus="0,1"),
        "generation": Role(config=config(generation_dp_config), gpus="2,3"),
    }


@dataclasses.dataclass
class Deployment:
    model: str
    orchestrator: RemoteDisaggOpenAIServer
    workers: dict[str, RemoteOpenAIServer]

    @property
    def ctx(self) -> RemoteOpenAIServer:
        return self.workers["context"]

    def url(self, *parts: str) -> str:
        return self.orchestrator.url_for(*parts)


def complete(served: Deployment, prompt, max_tokens: int = 16, **extra) -> dict:
    """A greedy completion through the orchestrator; its response body."""
    response = requests.post(
        served.url("v1", "completions"),
        json={
            "model": served.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            **extra,
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    assert response.status_code == 200, response.text
    return response.json()


def load(server: RemoteOpenAIServer) -> dict:
    """``Control.GetLoad`` of the worker's engine (chain only)."""
    return dr.get_load(server._chain_handle.engine_endpoint)


def wait_for(condition, what: str, timeout: float, interval: float = 0.2) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"timed out after {timeout} s waiting for {what}"
        time.sleep(interval)


def _physical_gpus(gpus: str, env: Mapping[str, str]) -> str:
    """``gpus`` (indices into the GPUs ``env`` makes visible) as ``CUDA_VISIBLE_DEVICES``."""
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return gpus
    listed = [gpu.strip() for gpu in visible.split(",") if gpu.strip()]
    return ",".join(listed[int(index)] for index in gpus.split(","))


def _ready(orchestrator: RemoteDisaggOpenAIServer) -> bool:
    """Whether the orchestrator's ``/health`` is 200: both workers have registered with it."""
    code = orchestrator.proc.poll()
    assert code is None, f"the orchestrator exited with code {code}; log {orchestrator.log_path}"
    try:
        return requests.get(orchestrator.url_for("health"), timeout=10).status_code == 200
    except requests.RequestException:
        return False


@contextlib.contextmanager
def _deployment(
    directory: Path,
    model: str,
    *,
    config: Mapping[str, Any] | None = None,
    roles: Mapping[str, Role] | None = None,
    env: Mapping[str, str] | None = None,
) -> Iterator[Deployment]:
    """Start the orchestrator and the two workers; ``config`` is merged over :data:`BASE_CONFIG`."""
    disagg_port = get_free_port()
    # On the chain the harness picks the engines' ports as the workers start: not this one.
    launch.reserve_ports(disagg_port)
    cluster = {"cluster_uri": f"http://localhost:{disagg_port}", "cluster_name": ""}
    auth_key = secrets.token_hex(32)
    shared = {**BASE_CONFIG, **(config or {})}
    # The workers run on this host: loopback is the only TCP device UCX needs.
    base_env = {**os.environ, "UCX_NET_DEVICES": "lo", **(env or {})}

    def worker(role: str) -> RemoteOpenAIServer:
        own = (roles or {}).get(role, Role())
        worker_env = {**base_env, **own.env}
        if own.gpus is not None:
            worker_env["CUDA_VISIBLE_DEVICES"] = _physical_gpus(own.gpus, base_env)
        worker_config = {
            **shared,
            **own.config,
            "disagg_cluster": cluster,
            "internal_request_auth_key": auth_key,
        }
        # No port given: the worker registers its own address with the orchestrator, and a
        # chain front end whose port was taken moves to another.
        return RemoteOpenAIServer(
            model,
            env=worker_env,
            cli_args=["--server_role", role],
            llmapi_launch=False,
            extra_config=worker_config,
            log_path=str(directory / f"output_{role}.log"),
            wait=False,
        )

    disagg_config = {
        "hostname": "localhost",
        "port": disagg_port,
        "disagg_cluster": cluster,
        "internal_request_auth_key": auth_key,
    }
    with RemoteDisaggOpenAIServer(
        ctx_servers=[],
        gen_servers=[],
        port=disagg_port,
        env=base_env,
        llmapi_launch=False,
        disagg_config=disagg_config,
        log_path=str(directory / "orchestrator.log"),
        # Its /health stays 503 until the workers register; the wait below has a timeout.
        wait_ready=False,
    ) as orchestrator:
        with worker("context") as ctx, worker("generation") as gen:
            for server in (ctx, gen):
                server.wait_for_server(timeout=ENGINE_READY_TIMEOUT_S)
            wait_for(lambda: _ready(orchestrator), "the workers to register", REGISTER_TIMEOUT_S)
            yield Deployment(model, orchestrator, {"context": ctx, "generation": gen})


def deploy(
    request: pytest.FixtureRequest,
    directory: Path,
    model: str,
    *,
    config: Mapping[str, Any] | None = None,
    roles: Mapping[str, Role] | None = None,
    env: Mapping[str, str] | None = None,
) -> Deployment:
    """Start a deployment that stops at the test's teardown; its logs go to ``directory``.

    On the chain the harness reads the orchestrator's counters when the test
    body ends, so the deployment must still run then.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stack = contextlib.ExitStack()
    request.addfinalizer(stack.close)
    return stack.enter_context(_deployment(directory, model, config=config, roles=roles, env=env))

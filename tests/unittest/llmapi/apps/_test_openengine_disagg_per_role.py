# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A disaggregated deployment whose roles differ: their own configuration, GPUs and environment.

No tests here. :func:`deployment` starts a context and a generation worker,
each a ``trtllm-serve`` with ``--server_role``, and an orchestrator they
register with through its own cluster. Each role gets the shared
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

from tensorrt_llm._utils import get_free_ports

from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

__extra_import_path__ = ["~/tests/integration/defs"]
from openengine_chain import direct_replay as dr  # noqa: E402
from openengine_chain import launch  # noqa: E402

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
chain_only = pytest.mark.skipif(
    not CHAIN, reason="reads the OpenEngine engines; set TRTLLM_TEST_OPENENGINE_CHAIN"
)
ENGINE_READY_TIMEOUT_S = 1800
REQUEST_TIMEOUT_S = 300
REGISTER_TIMEOUT_S = 120
ROLES = ("context", "generation")

BASE_CONFIG: Mapping[str, Any] = {
    "cache_transceiver_config": {"backend": "DEFAULT"},
    "kv_cache_config": {"free_gpu_memory_fraction": 0.2, "enable_block_reuse": False},
    "disable_overlap_scheduler": True,
    "cuda_graph_config": None,
}


@dataclasses.dataclass(frozen=True)
class Role:
    """What one role's worker gets on top of the shared settings."""

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
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (9, 0):
        pytest.skip("needs a GPU of Hopper or later")
    if torch.cuda.device_count() < count:
        pytest.skip(f"needs {count} GPUs")


def attention_dp_roles(generation_dp_config: Mapping[str, Any]) -> dict[str, Role]:
    """Two GPUs per role with attention DP; the generation worker also gets ``generation_dp_config``."""
    adp = {"tensor_parallel_size": 2, "enable_attention_dp": True}
    return {
        "context": Role(config=adp, gpus="0,1"),
        "generation": Role(
            config={**adp, "attention_dp_config": dict(generation_dp_config)}, gpus="2,3"
        ),
    }


@dataclasses.dataclass
class Deployment:
    model: str
    orchestrator: RemoteDisaggOpenAIServer
    workers: dict[str, RemoteOpenAIServer]
    logs: dict[str, Path]

    @property
    def ctx(self) -> RemoteOpenAIServer:
        return self.workers["context"]

    @property
    def gen(self) -> RemoteOpenAIServer:
        return self.workers["generation"]

    def url(self, *parts: str) -> str:
        return self.orchestrator.url_for(*parts)

    def log(self, role: str) -> str:
        path = self.logs[role]
        return path.read_text(errors="replace") if path.exists() else ""


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


def counter(server: RemoteOpenAIServer, key: str) -> int:
    return int(load(server).get("attributes", {}).get(key, 0))


def wait_for(condition, what: str, timeout: float, interval: float = 0.2) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"timed out after {timeout} s waiting for {what}"
        time.sleep(interval)


def _registered(orchestrator: RemoteDisaggOpenAIServer) -> bool:
    try:
        lists = requests.get(orchestrator.url_for("cluster_info"), timeout=10).json()
        return bool(lists["server_lists"]["context"] and lists["server_lists"]["generation"])
    except (requests.RequestException, ValueError, KeyError):
        return False


@contextlib.contextmanager
def deployment(
    directory: Path,
    model: str,
    *,
    config: Mapping[str, Any] | None = None,
    roles: Mapping[str, Role] | None = None,
    orchestrator_config: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> Iterator[Deployment]:
    """Start the two workers and the orchestrator; ``config`` is merged over :data:`BASE_CONFIG`."""
    disagg_port, ctx_port, gen_port = get_free_ports(3)
    # On the chain the harness picks the engines' ports as the workers start: not these three.
    launch.reserve_ports(disagg_port, ctx_port, gen_port)
    cluster = {"cluster_uri": f"http://localhost:{disagg_port}", "cluster_name": ""}
    auth_key = secrets.token_hex(32)
    shared = {**BASE_CONFIG, **(config or {})}
    # The workers run on this host: loopback is the only TCP device UCX needs.
    base_env = {**os.environ, "UCX_NET_DEVICES": "lo", **(env or {})}
    logs = {role: directory / f"output_{role}.log" for role in ROLES}

    def worker(role: str, port: int) -> RemoteOpenAIServer:
        own = (roles or {}).get(role, Role())
        worker_env = {**base_env, **own.env}
        if own.gpus is not None:
            worker_env["CUDA_VISIBLE_DEVICES"] = own.gpus
        worker_config = {
            **shared,
            **own.config,
            "disagg_cluster": cluster,
            "internal_request_auth_key": auth_key,
        }
        return RemoteOpenAIServer(
            model,
            port=port,
            env=worker_env,
            cli_args=["--server_role", role],
            llmapi_launch=False,
            extra_config=worker_config,
            log_path=str(logs[role]),
            wait=False,
        )

    disagg_config = {
        "hostname": "localhost",
        "port": disagg_port,
        "disagg_cluster": cluster,
        "internal_request_auth_key": auth_key,
        **(orchestrator_config or {}),
    }
    with worker("context", ctx_port) as ctx, worker("generation", gen_port) as gen:
        with RemoteDisaggOpenAIServer(
            ctx_servers=[],
            gen_servers=[],
            port=disagg_port,
            env=base_env,
            llmapi_launch=False,
            disagg_config=disagg_config,
            log_path=str(directory / "orchestrator.log"),
        ) as orchestrator:
            for server in (ctx, gen):
                server.wait_for_server(timeout=ENGINE_READY_TIMEOUT_S)
            wait_for(
                lambda: _registered(orchestrator), "the workers to register", REGISTER_TIMEOUT_S
            )
            yield Deployment(model, orchestrator, {"context": ctx, "generation": gen}, logs)

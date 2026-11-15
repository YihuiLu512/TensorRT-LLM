# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The disaggregated deployment the WT-57 ``_test_openengine_disagg_*`` files start (no tests here).

:func:`disagg_deployment` starts context and generation workers, each a
``trtllm-serve`` with ``--server_role``, and an orchestrator that they
register with through its own cluster, as ``_test_openengine_disagg_perf_metrics.py``
does. Without ``TRTLLM_TEST_OPENENGINE_CHAIN`` every worker runs its engine
in-process (the baseline form); with it, the chain harness
(``tests/integration/defs/openengine_chain``) starts each worker as a worker
frontend and a TensorRT-LLM OpenEngine server. The workers share the visible
GPUs, so one GPU serves both legs of the default small model.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import secrets
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import openai
import pytest
import torch
from test_common.http_utils import wait_for_endpoint_ready

from tensorrt_llm._utils import get_free_ports

from ..test_llm import get_model_path
from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
chain_only = pytest.mark.skipif(
    not CHAIN, reason="checks the OpenEngine chain; set TRTLLM_TEST_OPENENGINE_CHAIN"
)
MODEL_NAME = "Qwen3/Qwen3-0.6B"
ENGINE_READY_TIMEOUT_S = 1200
REQUEST_TIMEOUT_S = 120
ROLES = ("context", "generation")

# What every worker of a deployment gets unless a test changes it: a small
# share of the GPU each, so that several workers fit on one GPU.
BASE_WORKER_CONFIG: Mapping[str, Any] = {
    "cache_transceiver_config": {"backend": "DEFAULT"},
    "kv_cache_config": {"free_gpu_memory_fraction": 0.15, "enable_block_reuse": False},
    "disable_overlap_scheduler": True,
    "cuda_graph_config": None,
}


def needs_gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (9, 0):
        pytest.skip("needs a GPU of Hopper or later")


def model_path(name: str = MODEL_NAME) -> str:
    path = get_model_path(name)
    if not os.path.isdir(path):
        pytest.skip(f"{name} is not under LLM_MODELS_ROOT")
    return path


@dataclasses.dataclass
class Deployment:
    """A running orchestrator and its workers, by role."""

    orchestrator: RemoteDisaggOpenAIServer
    workers: dict[str, list[RemoteOpenAIServer]]
    logs: dict[str, list[str]]
    model: str

    @property
    def ctx(self) -> RemoteOpenAIServer:
        return self.workers["context"][0]

    @property
    def gen(self) -> RemoteOpenAIServer:
        return self.workers["generation"][0]

    @property
    def url(self) -> str:
        return self.orchestrator.url_root

    def client(self) -> openai.OpenAI:
        return openai.OpenAI(
            base_url=self.orchestrator.url_for("v1"),
            api_key="tensorrt_llm",
            timeout=REQUEST_TIMEOUT_S,
        )

    def worker_client(self, role: str, index: int = 0) -> openai.OpenAI:
        return openai.OpenAI(
            base_url=self.workers[role][index].url_for("v1"),
            api_key="tensorrt_llm",
            timeout=REQUEST_TIMEOUT_S,
        )

    def log(self, role: str, index: int = 0) -> str:
        path = Path(self.logs[role][index])
        return path.read_text(errors="replace") if path.exists() else ""


@contextlib.contextmanager
def disagg_deployment(
    directory: Path,
    *,
    model: str = MODEL_NAME,
    worker_args: Sequence[str] = (),
    role_args: Mapping[str, Sequence[str]] | None = None,
    worker_config: Mapping[str, Any] | None = None,
    counts: Mapping[str, int] | None = None,
    orchestrator_config: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> Iterator[Deployment]:
    """Start the workers (one per role unless ``counts`` says otherwise) and the orchestrator.

    ``worker_args`` go to every worker's command line and ``role_args[role]``
    to that role's; ``worker_config`` is merged over :data:`BASE_WORKER_CONFIG`
    (top level only) as each worker's ``--extra_llm_api_options``.
    """
    needs_gpu()
    path = model_path(model)
    counts = {role: (counts or {}).get(role, 1) for role in ROLES}
    ports = get_free_ports(1 + sum(counts.values()))
    disagg_port, worker_ports = ports[0], iter(ports[1:])
    cluster = {"cluster_uri": f"http://localhost:{disagg_port}", "cluster_name": ""}
    auth_key = secrets.token_hex(32)
    config = {
        **BASE_WORKER_CONFIG,
        "model": model,
        "disagg_cluster": cluster,
        "internal_request_auth_key": auth_key,
        **(worker_config or {}),
    }
    # The workers run on this host: loopback is the only TCP device UCX needs.
    process_env = {**os.environ, "UCX_NET_DEVICES": "lo", **(env or {})}
    logs: dict[str, list[str]] = {role: [] for role in ROLES}
    with contextlib.ExitStack() as stack:
        workers: dict[str, list[RemoteOpenAIServer]] = {role: [] for role in ROLES}
        for role in ROLES:
            for index in range(counts[role]):
                log = str(directory / f"{role}_{index}.log")
                logs[role].append(log)
                args = ["--server_role", role, *worker_args, *(role_args or {}).get(role, ())]
                workers[role].append(
                    stack.enter_context(
                        RemoteOpenAIServer(
                            path,
                            port=next(worker_ports),
                            env=process_env,
                            cli_args=args,
                            extra_config=config,
                            log_path=log,
                            wait=False,
                        )
                    )
                )
        orchestrator = stack.enter_context(
            RemoteDisaggOpenAIServer(
                ctx_servers=[],
                gen_servers=[],
                port=disagg_port,
                env=process_env,
                disagg_config={
                    "hostname": "localhost",
                    "port": disagg_port,
                    "disagg_cluster": cluster,
                    "internal_request_auth_key": auth_key,
                    **(orchestrator_config or {}),
                },
                log_path=str(directory / "orchestrator.log"),
            )
        )
        for role in ROLES:
            for worker in workers[role]:
                worker.wait_for_server(timeout=ENGINE_READY_TIMEOUT_S)
        wait_for_endpoint_ready(orchestrator.url_root + "/health")
        yield Deployment(orchestrator, workers, logs, model)

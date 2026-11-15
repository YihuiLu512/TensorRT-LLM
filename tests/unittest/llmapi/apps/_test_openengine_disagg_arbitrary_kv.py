# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arbitrary KV transfer is not served on the disaggregated OpenEngine chain (EG-29, DQ-43).

An orchestrator and a context and a generation worker, each started with
``--server_role``, register through the orchestrator's cluster; on the
TensorRT-LLM OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN=trtllm``,
``tests/integration/defs/openengine_chain``) each worker is a frontend and a
TensorRT-LLM OpenEngine server. In-process a worker answers with its KV cache
transceiver's state, so the test runs on the chain only.

NT-204 (EG-29): each worker frontend answers ``GET /v1/data_transceiver_state``
with 501: there is no transceiver state to hand to a client for a transfer
from another instance. The test sends no generation request.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import contextlib
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
from openengine_chain.conventions import CHAIN_TRTLLM  # noqa: E402

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN") == CHAIN_TRTLLM
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not CHAIN, reason="the 501 is the TensorRT-LLM OpenEngine worker frontend's"
    ),
]

MODEL_SUBDIR = os.path.join("Qwen3", "Qwen3-0.6B")
ENGINE_READY_TIMEOUT_S = 1200
REQUEST_TIMEOUT_S = 120
REGISTER_TIMEOUT_S = 120
WORKER_CONFIG = {
    "cache_transceiver_config": {"backend": "DEFAULT"},
    "kv_cache_config": {"free_gpu_memory_fraction": 0.2, "enable_block_reuse": False},
    "disable_overlap_scheduler": True,
    "cuda_graph_config": None,
}


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

    def worker(role: str, port: int) -> RemoteOpenAIServer:
        return RemoteOpenAIServer(
            model,
            port=port,
            env=env,
            cli_args=["--server_role", role],
            extra_config={**WORKER_CONFIG, "disagg_cluster": cluster},
            log_path=str(directory / f"output_{role}.log"),
            wait=False,
        )

    disagg_config = {"hostname": "localhost", "port": disagg_port, "disagg_cluster": cluster}
    with worker("context", ctx_port) as ctx, worker("generation", gen_port) as gen:
        with RemoteDisaggOpenAIServer(
            ctx_servers=[], gen_servers=[], port=disagg_port, disagg_config=disagg_config
        ) as orchestrator:
            for server in (ctx, gen):
                server.wait_for_server(timeout=ENGINE_READY_TIMEOUT_S)
            _wait_registered(orchestrator)
            yield Chain(model, orchestrator, ctx, gen)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    with _chain(tmp_path_factory.mktemp("disagg_arbitrary_kv")) as served:
        yield served


def test_data_transceiver_state_endpoint_unsupported(chain):
    """NT-204 (EG-29): both worker frontends answer 501 for the transceiver state."""
    for server in (chain.ctx, chain.gen):
        response = requests.get(
            server.url_for("v1", "data_transceiver_state"), timeout=REQUEST_TIMEOUT_S
        )
        assert response.status_code == 501, response.text
        message = response.json()["message"]
        assert "/v1/data_transceiver_state" in message and "OpenEngine" in message, message

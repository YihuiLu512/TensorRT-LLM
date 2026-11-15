# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A worker frontend whose engine exits stops itself, and the orchestrator drops it (SV-DSG-10).

An orchestrator and a context and a generation worker, each started with
``--server_role``, register through the orchestrator's cluster; on the
OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN``,
``tests/integration/defs/openengine_chain``) each worker is a frontend and a
TensorRT-LLM OpenEngine server. Each test starts a chain of its own, sends one
request through it, and kills one worker's engine process; the test runs on
the chain only.

- NT-183 (DG-87): the context engine exits; its frontend reports the engine
  lost and exits, and the orchestrator no longer lists the worker.
- NT-184 (DG-87): the same for the generation engine.

The expectations table marks the killed role (``leg_expect: killed``,
``openengine_chain/expectations/WT-52.py``).

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._utils import get_free_ports

from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not CHAIN, reason="the engine is a process of its own on the chain only"),
]

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
# Workers beat every second and drop out of the cluster after five seconds without one.
CLUSTER_CONFIG = {"heartbeat_interval_sec": 1, "inactive_timeout_sec": 5}
# The frontend gives up on its engine after its engine-lost timeout (30 s by default).
FRONTEND_STOP_S = 180
DROP_S = 60
# What a frontend logs when it gives up on its engine.
LOST_LOG = "stopping this frontend"


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


@pytest.fixture
def chain(tmp_path):
    """A chain for one test: the test kills one of its engines."""
    with _chain(tmp_path, cluster_config=CLUSTER_CONFIG) as served:
        yield served


def _wait(condition, what: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"timed out after {timeout} s waiting for {what}"
        time.sleep(0.5)


def _listed(chain: Chain, role: str, port: int) -> bool:
    """Whether the orchestrator lists a ``role`` worker on ``port``."""
    lists = requests.get(chain.orchestrator.url_for("cluster_info"), timeout=10).json()
    return any(str(address).endswith(f":{port}") for address in lists["server_lists"][role])


def _engine_exit_stops_the_worker(chain: Chain, role: str) -> None:
    server = chain.ctx if role == "context" else chain.gen
    handle = server._chain_handle
    response = requests.post(
        chain.orchestrator.url_for("v1", "completions"),
        json={"model": chain.model, "prompt": PROMPT, "max_tokens": 4},
        timeout=REQUEST_TIMEOUT_S,
    )
    assert response.status_code == 200, response.text
    assert _listed(chain, role, server.port)

    handle.engine.kill()
    _wait(lambda: handle.frontend.poll() is not None, "the frontend to stop", FRONTEND_STOP_S)
    assert LOST_LOG in Path(handle.frontend_log).read_text(errors="replace")
    _wait(lambda: not _listed(chain, role, server.port), "the worker to be dropped", DROP_S)


def test_ctx_engine_exit_stops_the_frontend_and_drops_the_worker(chain):
    """NT-183 (DG-87): the context engine exits."""
    _engine_exit_stops_the_worker(chain, "context")


def test_gen_engine_exit_stops_the_frontend_and_drops_the_worker(chain):
    """NT-184 (DG-87): the generation engine exits."""
    _engine_exit_stops_the_worker(chain, "generation")

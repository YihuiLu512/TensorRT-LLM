# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A handoff no generation leg claims can be counted from the engines (DG-90).

An orchestrator and a context and a generation worker, each started with
``--server_role``, register through the orchestrator's cluster; on the
OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN``,
``tests/integration/defs/openengine_chain``) each worker is a frontend and a
TensorRT-LLM OpenEngine server, whose ``Control.GetLoad`` counters the test
reads, so it runs on the chain only.

NT-187 (DG-90): the test sends context legs to the context worker itself and
no generation leg; the orphan handoffs, the PrefillReady events the context
engine sent less the generation_only calls the generation engine accepted,
are as many as the context legs. The expectations table says how many
handoffs the test leaves unclaimed (``leg_expect: direct_ctx_leg``,
``openengine_chain/expectations/WT-52.py``).

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import contextlib
import itertools
import os
import time
from dataclasses import dataclass

import pytest
import requests
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._utils import get_free_ports
from tensorrt_llm.grpc.openengine_conventions import (
    BUCKET_GENERATION_ONLY,
    COUNTER_GENERATE_ACCEPTED_BUCKETS,
    COUNTER_PREFILL_READY_SENT,
    DISAGG_REQUEST_ID_MIN,
)

from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

__extra_import_path__ = ["~/tests/integration/defs"]
from openengine_chain import direct_replay as dr  # noqa: E402

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not CHAIN, reason="the counters are the OpenEngine engines'"),
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
# The context legs NT-187 leaves unclaimed; the expectations table says the same number.
UNCLAIMED = 2
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
    with _chain(tmp_path_factory.mktemp("disagg_orphans")) as served:
        yield served


def _counter(server: RemoteOpenAIServer, key: str) -> int:
    load = dr.get_load(server._chain_handle.engine_endpoint)
    return int(load.get("attributes", {}).get(key, 0))


def _orphans(chain: Chain) -> int:
    """Handoffs the context engine sent that the generation engine has not taken."""
    taken = COUNTER_GENERATE_ACCEPTED_BUCKETS[BUCKET_GENERATION_ONLY]
    return _counter(chain.ctx, COUNTER_PREFILL_READY_SENT) - _counter(chain.gen, taken)


def test_an_unclaimed_handoff_is_reported_as_an_orphan(chain):
    """NT-187 (DG-90): each context leg without a generation leg is one more orphan handoff."""
    before = _orphans(chain)
    for _ in range(UNCLAIMED):
        body = {
            "model": chain.model,
            "prompt": PROMPT,
            "max_tokens": 4,
            "disaggregated_params": {
                "request_type": "context_only",
                "disagg_request_id": next(_DISAGG_REQUEST_IDS),
            },
        }
        response = requests.post(
            chain.ctx.url_for("v1", "completions"), json=body, timeout=REQUEST_TIMEOUT_S
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0].get("disaggregated_params"), response.text
    assert _orphans(chain) - before == UNCLAIMED

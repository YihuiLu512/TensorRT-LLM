# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A worker frontend with a role refuses a real engine without KV transfer (SV-DSG-9).

One OpenEngine server (``trtllm-serve --grpc --grpc-protocol openengine``) runs
without ``cache_transceiver_config``: it declares no ``kv_connector``, and
without KV transfer it does not declare ``trtllm.request.disagg_request_id``
either. The chain harness always gives a worker's engine a transceiver, so
the test starts the engine itself, and the worker frontends
(``trtllm-serve --openengine-endpoint``) in front of it; it runs with the
TensorRT-LLM OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN=trtllm``) only.

NT-180 (DG-84): a frontend started with ``--server_role context`` or
``--server_role generation`` exits at its startup check of the engine's
capabilities, naming the engine and the capability it lacks; the engine is
left serving. The test sends no Generate call.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import grpc
import pytest
import torch
from grpc_health.v1 import health_pb2, health_pb2_grpc
from utils.llm_data import llm_models_root

from tensorrt_llm.grpc.openengine_conventions import CAP_REQ_DISAGG_REQUEST_ID

__extra_import_path__ = ["~/tests/integration/defs"]
from openengine_chain import direct_replay as dr  # noqa: E402
from openengine_chain.conventions import CHAIN_TRTLLM  # noqa: E402

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN") == CHAIN_TRTLLM
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not CHAIN, reason="the refusal is the TensorRT-LLM OpenEngine frontend's"),
]

MODEL_SUBDIR = os.path.join("Qwen3", "Qwen3-0.6B")
ENGINE_READY_TIMEOUT_S = 1200
FRONTEND_EXIT_TIMEOUT_S = 600
CONFIG = {"kv_cache_config": {"free_gpu_memory_fraction": 0.2}}


def _model_dir() -> str:
    root = llm_models_root()
    path = None if root is None else os.path.join(str(root), MODEL_SUBDIR)
    assert path is not None and os.path.isdir(path), f"{MODEL_SUBDIR} is not under LLM_MODELS_ROOT"
    return path


def _serve_command() -> list[str]:
    entry = Path(sys.executable).parent / "trtllm-serve"
    if entry.is_file():
        return [str(entry)]
    return [sys.executable, "-m", "tensorrt_llm.commands.serve"]


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (9, 0):
        pytest.skip("needs a GPU of Hopper or later")
    workdir = str(tmp_path_factory.mktemp("no_kv_connector"))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    started = dr.start_engine("plain", _model_dir(), workdir, config=CONFIG, env=env)
    try:
        started.wait_serving(ENGINE_READY_TIMEOUT_S)
        yield started
    finally:
        started.stop()


def _serving(endpoint: str) -> bool:
    with grpc.insecure_channel(endpoint) as channel:
        reply = health_pb2_grpc.HealthStub(channel).Check(
            health_pb2.HealthCheckRequest(service="inference"), timeout=30
        )
    return reply.status == health_pb2.HealthCheckResponse.SERVING


def test_role_frontend_refuses_an_engine_without_kv_connector(engine, tmp_path):
    """NT-180 (DG-84): either role's frontend exits naming the capability the engine lacks."""
    assert dr.server_info(engine.endpoint).get("kv_connector") is None
    name = dr.server_info(engine.endpoint)["engine_name"]
    for role in ("context", "generation"):
        result = subprocess.run(  # nosec B603
            [
                *_serve_command(),
                _model_dir(),
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--report_addr",
                str(tmp_path / f"addr-{role}"),
                "--openengine-endpoint",
                engine.endpoint,
                "--server_role",
                role,
            ],
            capture_output=True,
            text=True,
            timeout=FRONTEND_EXIT_TIMEOUT_S,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        output = result.stdout + result.stderr
        assert result.returncode != 0, output[-4000:]
        assert "OpenEngine startup failed at step 'capabilities'" in output, output[-4000:]
        lacks = f"engine {name!r} does not declare {CAP_REQ_DISAGG_REQUEST_ID}"
        assert f"{lacks}; a {role.upper()} worker needs it" in output, output[-4000:]
    assert engine.poll() is None
    assert _serving(engine.endpoint)

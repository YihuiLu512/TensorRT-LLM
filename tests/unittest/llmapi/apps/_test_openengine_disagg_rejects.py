# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the disaggregated OpenEngine chain refuses, on real engines (SV-DSG-8, SV-DSG-9).

An orchestrator and a context and a generation worker, each started with
``--server_role``, register through the orchestrator's cluster. On the
OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN``,
``tests/integration/defs/openengine_chain``) each worker is a frontend and a
TensorRT-LLM OpenEngine server; the refusals are the frontend's, so the tests
run on the chain only.

- NT-058 (F-10), NT-064 (F-19), NT-066 (F-26), NT-191 (EG-07): a request
  through the orchestrator with media, ``n > 1`` or ``best_of > 1``, or
  ``return_context_logits`` is answered 400 naming the context leg, and no
  Generate call reaches either engine.
- NT-185 (DG-88): each refusal of a leg is HTTP 400 whose message starts with
  the leg it refuses: the context leg's own refusals, and a generation leg
  with a handoff field it cannot carry.
- NT-190 (EG-05a, EG-05b), NT-192 (EG-10b): a generation leg with multimodal
  positions, with ``trtllm_decode_preparation`` in its
  ``openengine_handoff_attributes``, or with ``first_gen_logits`` is refused
  before its Generate call.
- NT-065 (F-25), NT-216 (EG-19): a worker started with ``max_beam_width > 1``
  or with the remote tail (``bounded_replay_on_generation``) does not start.
  The harness reads the refusal from the frontend's log
  (``openengine_chain/expectations/WT-52.py``).

The tests that send legs to a worker themselves leave the context legs'
handoffs unclaimed; the expectations table counts them.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import contextlib
import itertools
import json
import os
import time
from dataclasses import dataclass

import pytest
import requests
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._utils import get_free_ports
from tensorrt_llm.grpc.openengine_conventions import (
    COUNTER_GENERATE_RECEIVED,
    DISAGG_REQUEST_ID_MIN,
    HANDOFF_DECODE_PREPARATION,
)

from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

__extra_import_path__ = ["~/tests/integration/defs"]
from openengine_chain import direct_replay as dr  # noqa: E402

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not CHAIN, reason="the refusals are the OpenEngine worker frontend's"),
]

MODEL_SUBDIR = os.path.join("Qwen3", "Qwen3-0.6B")
PROMPT = "The capital of France is Paris. The capital of Germany is"
ENGINE_READY_TIMEOUT_S = 1200
REQUEST_TIMEOUT_S = 120
REGISTER_TIMEOUT_S = 120
CTX_LEG = "the context leg (context_only) of a disaggregated request"
GEN_LEG = "the generation leg (generation_only) of a disaggregated request"
WORKER_CONFIG = {
    "cache_transceiver_config": {"backend": "DEFAULT"},
    "kv_cache_config": {"free_gpu_memory_fraction": 0.2, "enable_block_reuse": False},
    "disable_overlap_scheduler": True,
    "cuda_graph_config": None,
}
# A 1x1 PNG, so that no test needs the network.
PNG_DATA_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
# The ids of the context legs a test sends to a worker itself.
_DISAGG_REQUEST_IDS = itertools.count(DISAGG_REQUEST_ID_MIN + 1)
# One float32 0.5, as the HTTP field carries a tensor.
FIRST_GEN_LOGITS = [{"data": "AAAAPw==", "shape": [1], "dtype": "float32"}]


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
    with _chain(tmp_path_factory.mktemp("disagg_rejects")) as served:
        yield served


def _received(server: RemoteOpenAIServer) -> int:
    """The Generate calls the worker's engine has received."""
    load = dr.get_load(server._chain_handle.engine_endpoint)
    return int(load.get("attributes", {}).get(COUNTER_GENERATE_RECEIVED, 0))


def _post(server: RemoteOpenAIServer, path: str, body: dict) -> requests.Response:
    return requests.post(server.url_for(*path.split("/")), json=body, timeout=REQUEST_TIMEOUT_S)


def _refused(chain: Chain, server, path: str, body: dict, leg: str, *named: str) -> str:
    """``body`` sent to ``server`` is answered 400 naming ``leg`` and ``named``, before Generate."""
    before = (_received(chain.ctx), _received(chain.gen))
    response = _post(server, path, {"model": chain.model, **body})
    assert response.status_code == 400, response.text
    text = response.text
    assert leg in text, text
    for word in named:
        assert word in text, text
    assert (_received(chain.ctx), _received(chain.gen)) == before, text
    return text


def _context_leg(chain: Chain) -> dict:
    """A context leg sent to the context worker; its handoff is left unclaimed."""
    body = {
        "model": chain.model,
        "prompt": PROMPT,
        "max_tokens": 4,
        "disaggregated_params": {
            "request_type": "context_only",
            "disagg_request_id": next(_DISAGG_REQUEST_IDS),
        },
    }
    response = _post(chain.ctx, "v1/completions", body)
    assert response.status_code == 200, response.text
    return response.json()


def _generation_leg(ctx: dict, **params) -> dict:
    handoff = {**ctx["choices"][0]["disaggregated_params"], "request_type": "generation_only"}
    handoff.update(params)
    return {"prompt": ctx["prompt_token_ids"], "max_tokens": 4, "disaggregated_params": handoff}


def _image_chat() -> dict:
    content = [
        {"type": "image_url", "image_url": {"url": PNG_DATA_URI}},
        {"type": "text", "text": "What is in this image?"},
    ]
    return {"messages": [{"role": "user", "content": content}], "max_tokens": 4}


def _start_refused(directory, role: str, overrides: dict) -> None:
    """Start a worker with ``overrides``; it is expected to fail to start.

    The harness raises when the frontend refuses to start; the expectations
    table makes that refusal, with its log line, the test's pass.
    """
    _needs_gpu()
    config = {**WORKER_CONFIG, **overrides}
    config["kv_cache_config"] = {**config["kv_cache_config"], "free_gpu_memory_fraction": 0.1}
    with RemoteOpenAIServer(
        _model_dir(),
        env={**os.environ, "UCX_NET_DEVICES": "lo"},
        cli_args=["--server_role", role],
        extra_config=config,
        log_path=str(directory / f"output_{role}.log"),
    ):
        pytest.fail(f"a {role} worker started with {overrides}")


def test_multimodal_rejected(chain):
    """NT-058 (F-10): a chat request with an image is refused through the orchestrator."""
    _refused(chain, chain.orchestrator, "v1/chat/completions", _image_chat(), CTX_LEG)


def test_n_gt_1_rejected(chain):
    """NT-064 (F-19): ``n > 1`` and ``best_of > 1`` are refused through the orchestrator."""
    for fields, named in (({"n": 2}, "n=2"), ({"best_of": 2}, "best_of=2")):
        body = {"prompt": PROMPT, "max_tokens": 4, **fields}
        _refused(chain, chain.orchestrator, "v1/completions", body, CTX_LEG, named)


def test_logits_fields_rejected(chain):
    """NT-066 (F-26): ``return_context_logits`` is refused through the orchestrator."""
    body = {"prompt": PROMPT, "max_tokens": 4, "return_context_logits": True}
    _refused(chain, chain.orchestrator, "v1/completions", body, CTX_LEG, "return_context_logits")


def test_two_leg_rejections_return_400_naming_the_leg(chain):
    """NT-185 (DG-88): each refusal is HTTP 400 whose message starts with the leg it refuses."""
    ctx_leg = {
        "prompt": PROMPT,
        "max_tokens": 4,
        "n": 2,
        "disaggregated_params": {
            "request_type": "context_only",
            "disagg_request_id": next(_DISAGG_REQUEST_IDS),
        },
    }
    text = _refused(chain, chain.ctx, "v1/completions", ctx_leg, CTX_LEG, "n=2")
    assert json.loads(text)["message"].startswith(f"{CTX_LEG}: "), text
    ctx = _context_leg(chain)
    for params, named in (
        ({"first_gen_logits": FIRST_GEN_LOGITS}, "first_gen_logits"),
        ({"remote_tail_start": 1}, "remote_tail_start"),
        ({"multimodal_positions": [0], "multimodal_lengths": [1]}, "multimodal"),
    ):
        body = _generation_leg(ctx, **params)
        text = _refused(chain, chain.gen, "v1/completions", body, GEN_LEG, named)
        assert json.loads(text)["message"].startswith(f"{GEN_LEG}: "), text


def test_multimodal_request_is_rejected(chain):
    """NT-190 (EG-05a, EG-05b): no multimodal data reaches a generation leg's engine."""
    ctx = _context_leg(chain)
    body = _generation_leg(ctx, multimodal_positions=[0], multimodal_lengths=[1])
    _refused(chain, chain.gen, "v1/completions", body, GEN_LEG, "multimodal")
    preparation = json.dumps({HANDOFF_DECODE_PREPARATION: {"mrope": [1, 2]}})
    body = _generation_leg(ctx, openengine_handoff_attributes=preparation)
    _refused(chain, chain.gen, "v1/completions", body, GEN_LEG, HANDOFF_DECODE_PREPARATION)


def test_n_above_one_is_rejected_through_the_orchestrator(chain):
    """NT-191 (EG-07): a chat request with ``n > 1`` reaches neither engine."""
    body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 4, "n": 2}
    _refused(chain, chain.orchestrator, "v1/chat/completions", body, CTX_LEG, "n=2")


def test_generation_logits_request_is_rejected(chain):
    """NT-192 (EG-10b): a generation leg with ``first_gen_logits`` is refused."""
    ctx = _context_leg(chain)
    body = _generation_leg(ctx, first_gen_logits=FIRST_GEN_LOGITS)
    _refused(chain, chain.gen, "v1/completions", body, GEN_LEG, "first_gen_logits")


def test_beam_width_startup_rejected(tmp_path):
    """NT-065 (F-25): a worker with ``max_beam_width > 1`` does not start."""
    _start_refused(tmp_path, "context", {"max_beam_width": 2})


def test_remote_tail_config_refused_at_startup(tmp_path):
    """NT-216 (EG-19): a worker with the remote tail does not start."""
    _start_refused(tmp_path, "generation", {"bounded_replay_on_generation": True})

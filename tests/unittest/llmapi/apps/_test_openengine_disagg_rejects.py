# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the disaggregated OpenEngine chain refuses, on real engines (SV-DSG-8, SV-DSG-9).

An orchestrator and a context and a generation worker, each started with
``--server_role``, register through the orchestrator's cluster. On the
TensorRT-LLM OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN=trtllm``,
``tests/integration/defs/openengine_chain``) each worker is a frontend and a
TensorRT-LLM OpenEngine server; the refusals are the frontend's, so the tests
run on that chain only. A refused request sends no Generate call.

- NT-065 (F-25), NT-216 (EG-19): a worker started with ``max_beam_width > 1``
  or with the remote tail (``bounded_replay_on_generation``) does not start.
  The harness reads the refusal from the frontend's log
  (``openengine_chain/expectations/WT-52.py``).
- NT-058 (F-10), NT-190 (EG-05a, EG-05b): a chat request with an image, or
  with a video, through the orchestrator is answered 400: the context leg's
  frontend refuses a disaggregated request with media.
- NT-185 (DG-88): through the orchestrator, a request with ``n > 1`` and one
  with an image are answered 400 with a message that starts with the leg
  that refuses them.
- NT-064 (F-19), NT-066 (F-26), NT-191 (EG-07): ``n > 1``, ``best_of > 1`` or
  ``return_context_logits`` through the orchestrator is answered 400 naming
  the context leg.
- NT-192 (EG-10b): a generation leg with ``first_gen_logits`` is answered 400
  naming the generation leg; the test sends the context leg it hands off
  from to the context worker itself.

NT-058, NT-185 and NT-190 each start a chain of a model whose engines take
images and videos, so that the refusal they see is the one of a
disaggregated request and not the one of a model without media. The tests
run in the order they are written: the workers that do not start, those
three, then the tests on the module's chain, so that no two chains run at
once.

Needs one GPU of Hopper or later, and ``Qwen3/Qwen3-0.6B``,
``Qwen3/Qwen3-VL-2B-Instruct`` and ``multimodals/test_data`` under
``${LLM_MODELS_ROOT}``.
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
    COUNTER_GENERATE_RECEIVED,
    DISAGG_REQUEST_ID_MIN,
)

from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

__extra_import_path__ = ["~/tests/integration/defs"]
from openengine_chain import direct_replay as dr  # noqa: E402
from openengine_chain import launch  # noqa: E402
from openengine_chain.conventions import CHAIN_TRTLLM  # noqa: E402

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN") == CHAIN_TRTLLM
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not CHAIN, reason="the refusals are the TensorRT-LLM OpenEngine worker frontend's"
    ),
]

MODEL_SUBDIR = os.path.join("Qwen3", "Qwen3-0.6B")
MEDIA_MODEL_SUBDIR = os.path.join("Qwen3", "Qwen3-VL-2B-Instruct")
MEDIA_SUBDIR = os.path.join("multimodals", "test_data")
PROMPT = "The capital of France is Paris. The capital of Germany is"
ENGINE_READY_TIMEOUT_S = 1200
REQUEST_TIMEOUT_S = 120
REGISTER_TIMEOUT_S = 120
CTX_LEG = "the context leg (context_only) of a disaggregated request"
GEN_LEG = "the generation leg (generation_only) of a disaggregated request"
MEDIA_REFUSAL = "a disaggregated request with media is not supported"
WORKER_CONFIG = {
    "cache_transceiver_config": {"backend": "DEFAULT"},
    "kv_cache_config": {"free_gpu_memory_fraction": 0.2, "enable_block_reuse": False},
    "disable_overlap_scheduler": True,
    "cuda_graph_config": None,
}
# One float32 0.5, as the HTTP field carries a tensor.
FIRST_GEN_LOGITS = [{"data": "AAAAPw==", "shape": [1], "dtype": "float32"}]
# The ids of the context legs a test sends to a worker itself.
_DISAGG_REQUEST_IDS = itertools.count(DISAGG_REQUEST_ID_MIN + 1)


def _under_models_root(subdir: str) -> str:
    root = llm_models_root()
    path = None if root is None else os.path.join(str(root), subdir)
    assert path is not None and os.path.exists(path), f"{subdir} is not under LLM_MODELS_ROOT"
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
def _chain(directory, model_subdir: str = MODEL_SUBDIR, worker_env=None):
    """An orchestrator and two workers of ``model_subdir`` with roles, registered through it.

    ``worker_env`` adds to the workers' environment.
    """
    _needs_gpu()
    model = _under_models_root(model_subdir)
    disagg_port, ctx_port, gen_port = get_free_ports(3)
    # On the chain the harness picks the engines' ports as the workers start: not these three.
    launch.reserve_ports(disagg_port, ctx_port, gen_port)
    cluster = {"cluster_uri": f"http://localhost:{disagg_port}", "cluster_name": ""}
    # Both workers run on this host: loopback is the only TCP device UCX needs.
    env = {**os.environ, "UCX_NET_DEVICES": "lo", **(worker_env or {})}

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
    with _chain(tmp_path_factory.mktemp("disagg_rejects")) as served:
        yield served


@pytest.fixture
def media_chain(tmp_path):
    """A chain of a model that takes media, for one test; the media it reads are local files."""
    media = _under_models_root(MEDIA_SUBDIR)
    env = {"TRTLLM_MEDIA_ALLOWED_LOCAL_PATH": media}
    with _chain(tmp_path, MEDIA_MODEL_SUBDIR, env) as served:
        yield served


def _received(server: RemoteOpenAIServer) -> int:
    """The Generate calls the worker's engine has received."""
    load = dr.get_load(server._chain_handle.engine_endpoint)
    return int(load.get("attributes", {}).get(COUNTER_GENERATE_RECEIVED, 0))


def _post(server, path: str, body: dict) -> requests.Response:
    return requests.post(server.url_for(*path.split("/")), json=body, timeout=REQUEST_TIMEOUT_S)


def _message(response: requests.Response) -> str:
    """The error message: a worker frontend's ``message``, or the orchestrator's ``detail``."""
    body = response.json()
    return body.get("message") or body.get("detail") or ""


def _refused(chain: Chain, server, path: str, body: dict, leg: str, *named: str) -> str:
    """``body`` sent to ``server`` is answered 400 naming ``leg`` and ``named``, before Generate.

    Returns the error message.
    """
    before = (_received(chain.ctx), _received(chain.gen))
    response = _post(server, path, {"model": chain.model, **body})
    assert response.status_code == 400, response.text
    message = _message(response)
    for word in (leg, *named):
        assert word in message, response.text
    assert (_received(chain.ctx), _received(chain.gen)) == before, response.text
    return message


def _media_chat(kind: str, file_name: str) -> dict:
    """A chat request with one ``kind`` (image or video) item, a file of the test data."""
    url = os.path.join(_under_models_root(MEDIA_SUBDIR), file_name)
    content = [
        {"type": f"{kind}_url", f"{kind}_url": {"url": url}},
        {"type": "text", "text": f"What is in this {kind}?"},
    ]
    return {"messages": [{"role": "user", "content": content}], "max_tokens": 4}


IMAGE_CHAT = ("image", "seashore.png")
VIDEO_CHAT = ("video", "OAI-sora-tokyo-walk.mp4")


def _start_refused(directory, role: str, overrides: dict) -> None:
    """Start a worker with ``overrides``; it is expected to fail to start.

    The harness raises when the frontend refuses to start; the expectations
    table makes that refusal, with its log line, the test's pass.
    """
    _needs_gpu()
    config = {**WORKER_CONFIG, **overrides}
    config["kv_cache_config"] = {**config["kv_cache_config"], "free_gpu_memory_fraction": 0.1}
    with RemoteOpenAIServer(
        _under_models_root(MODEL_SUBDIR),
        env={**os.environ, "UCX_NET_DEVICES": "lo"},
        cli_args=["--server_role", role],
        extra_config=config,
        log_path=str(directory / f"output_{role}.log"),
    ):
        pytest.fail(f"a {role} worker started with {overrides}")


def test_beam_width_startup_rejected(tmp_path):
    """NT-065 (F-25): a worker with ``max_beam_width > 1`` does not start."""
    _start_refused(tmp_path, "context", {"max_beam_width": 2})


def test_remote_tail_config_refused_at_startup(tmp_path):
    """NT-216 (EG-19): a worker with the remote tail does not start."""
    _start_refused(tmp_path, "generation", {"bounded_replay_on_generation": True})


def test_multimodal_rejected(media_chain):
    """NT-058 (F-10): a chat request with an image or a video is refused through the orchestrator."""
    for media in (IMAGE_CHAT, VIDEO_CHAT):
        body = _media_chat(*media)
        _refused(
            media_chain,
            media_chain.orchestrator,
            "v1/chat/completions",
            body,
            CTX_LEG,
            MEDIA_REFUSAL,
        )


def test_two_leg_rejections_return_400_naming_the_leg(media_chain):
    """NT-185 (DG-88): each refusal is HTTP 400 whose message starts with the leg it refuses."""
    text_chat = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 4, "n": 2}
    for body, named in ((text_chat, "n=2"), (_media_chat(*IMAGE_CHAT), MEDIA_REFUSAL)):
        orchestrator = media_chain.orchestrator
        message = _refused(media_chain, orchestrator, "v1/chat/completions", body, CTX_LEG, named)
        assert message.startswith(f"{CTX_LEG}: "), message


def test_multimodal_request_is_rejected(media_chain):
    """NT-190 (EG-05a, EG-05b): a chat request with an image reaches neither engine."""
    body = _media_chat(*IMAGE_CHAT)
    orchestrator = media_chain.orchestrator
    _refused(media_chain, orchestrator, "v1/chat/completions", body, CTX_LEG, MEDIA_REFUSAL)


def test_n_gt_1_rejected(chain):
    """NT-064 (F-19): ``n > 1`` and ``best_of > 1`` are refused through the orchestrator."""
    for fields, named in (({"n": 2}, "n=2"), ({"best_of": 2}, "best_of=2")):
        body = {"prompt": PROMPT, "max_tokens": 4, **fields}
        _refused(chain, chain.orchestrator, "v1/completions", body, CTX_LEG, named)


def test_logits_fields_rejected(chain):
    """NT-066 (F-26): ``return_context_logits`` is refused through the orchestrator."""
    body = {"prompt": PROMPT, "max_tokens": 4, "return_context_logits": True}
    _refused(chain, chain.orchestrator, "v1/completions", body, CTX_LEG, "return_context_logits")


def test_n_above_one_is_rejected_through_the_orchestrator(chain):
    """NT-191 (EG-07): a chat request with ``n > 1`` reaches neither engine."""
    body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 4, "n": 2}
    _refused(chain, chain.orchestrator, "v1/chat/completions", body, CTX_LEG, "n=2")


def test_generation_logits_request_is_rejected(chain):
    """NT-192 (EG-10b): a generation leg with ``first_gen_logits`` is refused, naming its leg."""
    context_leg = {
        "model": chain.model,
        "prompt": PROMPT,
        "max_tokens": 4,
        "disaggregated_params": {
            "request_type": "context_only",
            "disagg_request_id": next(_DISAGG_REQUEST_IDS),
        },
    }
    response = _post(chain.ctx, "v1/completions", context_leg)
    assert response.status_code == 200, response.text
    ctx = response.json()
    handoff = {
        **ctx["choices"][0]["disaggregated_params"],
        "request_type": "generation_only",
        "first_gen_logits": FIRST_GEN_LOGITS,
    }
    body = {"prompt": ctx["prompt_token_ids"], "max_tokens": 4, "disaggregated_params": handoff}
    message = _refused(chain, chain.gen, "v1/completions", body, GEN_LEG, "first_gen_logits")
    assert message.startswith(f"{GEN_LEG}: "), message

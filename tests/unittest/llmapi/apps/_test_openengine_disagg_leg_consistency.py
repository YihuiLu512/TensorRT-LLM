# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The two legs of a disaggregated request carry the same request, on real engines (SV-DSG-7, CH-7).

An orchestrator and a context and a generation worker, each started with
``--server_role``, register through the orchestrator's cluster; on the
OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN``,
``tests/integration/defs/openengine_chain``) each worker is a frontend and a
TensorRT-LLM OpenEngine server. Each worker frontend writes the
GenerateRequests it sends to a directory of its own
(``TRTLLM_OPENENGINE_REQUEST_DUMP_DIR``), so the test runs on the chain only.

NT-181 (DG-85): for each request sent through the orchestrator, the context
leg's and the generation leg's GenerateRequest pair up by
``disagg_request_id`` and are the same but for ``request_id`` and the keys only
one leg carries (EX-DSG-7); the context leg's ``max_tokens`` is the client's.
The two engines declare different length limits; a chat request without
``max_tokens`` makes each frontend fill in its own default, and the test
records the two values (UV-67) without requiring them to agree.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import contextlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._utils import get_free_ports
from tensorrt_llm.grpc.openengine_conventions import (
    REQ_DISAGG_REQUEST_ID,
    REQ_ORIG_ELAPSED_S,
    REQ_REQUEST_TYPE,
    REQ_SCHEDULE_STYLE,
    SERVE_REQUEST_DUMP_DIR_ENV,
)

from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not CHAIN, reason="the OpenEngine worker frontends write the requests"),
]

MODEL_SUBDIR = os.path.join("Qwen3", "Qwen3-0.6B")
PROMPT = "The capital of France is Paris. The capital of Germany is"
ENGINE_READY_TIMEOUT_S = 1200
# A chat request without max_tokens may decode up to the generation engine's limit.
REQUEST_TIMEOUT_S = 300
REGISTER_TIMEOUT_S = 120
WORKER_CONFIG = {
    "cache_transceiver_config": {"backend": "DEFAULT"},
    "kv_cache_config": {"free_gpu_memory_fraction": 0.2, "enable_block_reuse": False},
    "disable_overlap_scheduler": True,
    "cuda_graph_config": None,
}
# Different length limits for the two engines (UV-67).
CTX_CONFIG = {"max_seq_len": 4096}
GEN_CONFIG = {"max_seq_len": 8192}
# EX-DSG-7: the keys only one leg carries.
PREFILL_ONLY_EXTRA = (REQ_REQUEST_TYPE, REQ_DISAGG_REQUEST_ID, REQ_SCHEDULE_STYLE)
DECODE_ONLY_EXTRA = (REQ_ORIG_ELAPSED_S,)
DUMP_WAIT_S = 30


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
    directory = tmp_path_factory.mktemp("disagg_leg_consistency")
    dump_dirs = {role: directory / f"requests_{role}" for role in ("context", "generation")}
    for path in dump_dirs.values():
        path.mkdir()
    role_env = {role: {SERVE_REQUEST_DUMP_DIR_ENV: str(path)} for role, path in dump_dirs.items()}
    with _chain(directory, CTX_CONFIG, GEN_CONFIG, role_env) as served:
        served.dump_dirs = dump_dirs
        yield served


def _dumps(directory: Path) -> dict[str, dict]:
    """The GenerateRequests written to ``directory``, by file name."""
    return {path.name: json.loads(path.read_text()) for path in directory.glob("*.json")}


def _new_dump(directory: Path, seen: dict[str, dict]) -> dict:
    """The one GenerateRequest written to ``directory`` since ``seen`` was read."""
    deadline = time.monotonic() + DUMP_WAIT_S
    while True:
        new = {name: dump for name, dump in _dumps(directory).items() if name not in seen}
        if new:
            assert len(new) == 1, sorted(new)
            return next(iter(new.values()))
        assert time.monotonic() < deadline, f"no GenerateRequest was written to {directory}"
        time.sleep(0.2)


def _comparable(dump: dict, leg_keys: tuple[str, ...]) -> dict:
    """``dump`` without ``request_id``, ``kv.session`` and the keys only its leg carries."""
    dump = json.loads(json.dumps(dump))
    dump.pop("request_id")
    extra = dump.get("extra", {})
    for key in leg_keys:
        extra.pop(key, None)
    if not extra:
        dump.pop("extra", None)
    kv = dump.get("kv", {})
    kv.pop("session", None)
    if not kv:
        dump.pop("kv", None)
    return dump


def _legs(chain: Chain, path: str, body: dict) -> tuple[dict, dict]:
    """Send ``body`` through the orchestrator; the two legs' GenerateRequests, paired."""
    seen = {role: _dumps(chain.dump_dirs[role]) for role in chain.dump_dirs}
    response = requests.post(
        chain.orchestrator.url_for(*path.split("/")),
        json={"model": chain.model, **body},
        timeout=REQUEST_TIMEOUT_S,
    )
    assert response.status_code == 200, (body, response.text)
    prefill = _new_dump(chain.dump_dirs["context"], seen["context"])
    decode = _new_dump(chain.dump_dirs["generation"], seen["generation"])
    assert prefill["extra"][REQ_REQUEST_TYPE] == "context_only", prefill
    session = decode["kv"]["session"]["attributes_struct"]
    assert str(session["disagg_request_id"]) == str(prefill["extra"][REQ_DISAGG_REQUEST_ID])
    assert prefill["request_id"] != decode["request_id"]
    return prefill, decode


BODIES = [
    {"prompt": PROMPT, "max_tokens": 3},
    {"prompt": PROMPT, "max_tokens": 5, "temperature": 0.5, "top_p": 0.9, "seed": 7, "stop": ["!"]},
    {"prompt": PROMPT, "max_tokens": 4, "stop": ["x"], "ignore_eos": True},
    {"prompt": PROMPT, "max_tokens": 4, "logprobs": 2, "top_k": 3, "min_p": 0.1},
    {"prompt": PROMPT, "max_tokens": 4, "frequency_penalty": 0.5, "presence_penalty": 0.5},
]


def test_ctx_and_gen_generate_requests_match_on_real_engines(chain, record_property):
    """NT-181 (DG-85): both legs carry the same request; only the single-leg keys differ."""
    for body in BODIES:
        prefill, decode = _legs(chain, "v1/completions", body)
        # The context leg's max_tokens is the client's: the engine enforces one token itself.
        assert int(prefill["stopping"]["max_tokens"]) == body["max_tokens"], prefill
        assert _comparable(prefill, PREFILL_ONLY_EXTRA) == _comparable(decode, DECODE_ONLY_EXTRA)

    # UV-67: without max_tokens each frontend fills in a default from its engine's limit.
    chat = {"messages": [{"role": "user", "content": PROMPT}], "temperature": 0.0}
    prefill, decode = _legs(chain, "v1/chat/completions", chat)
    limits = {
        leg: dump.get("stopping", {}).get("max_tokens")
        for leg, dump in (("ctx", prefill), ("gen", decode))
    }
    record_property("uv67_default_max_tokens", json.dumps(limits))
    for dump in (prefill, decode):
        dump.get("stopping", {}).pop("max_tokens", None)
    assert _comparable(prefill, PREFILL_ONLY_EXTRA) == _comparable(decode, DECODE_ONLY_EXTRA)

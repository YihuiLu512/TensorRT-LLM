# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The disaggregated deployment of the ``_test_openengine_disagg_*`` test files (no tests here).

:func:`disagg_deployment` starts context and generation workers, each a
``trtllm-serve`` with ``--server_role``, and an orchestrator that they
register with through its own cluster, as ``_test_openengine_disagg_perf_metrics.py``
does. Without ``TRTLLM_TEST_OPENENGINE_CHAIN`` every worker runs its engine
in-process (the baseline form); with it, the chain harness
(``tests/integration/defs/openengine_chain``) starts each worker as a worker
frontend and a TensorRT-LLM OpenEngine server. The workers share the visible
GPUs, so one GPU serves both legs of the default small model.

:func:`legs` reads back, on the chain, the two GenerateRequests of a request
from the directory a deployment started with :data:`DUMP_ENV` names.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import re
import secrets
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import openai
import pytest
import requests
import torch
from test_common.http_utils import wait_for_endpoint_ready

from tensorrt_llm._utils import get_free_ports
from tensorrt_llm.grpc import openengine_conventions as conv

from ..test_llm import get_model_path
from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
# Set in a deployment's ``env``: on the chain each worker frontend writes every
# GenerateRequest it sends to this directory.
DUMP_ENV = conv.SERVE_REQUEST_DUMP_DIR_ENV
# The orchestrator's count of requests it sent to each role's workers.
_SENT = re.compile(r"^(ctx|gen)_total_requests_total(?:\{[^}]*\})?\s+(\S+)\s*$", re.MULTILINE)
WORKERS_LISTED_TIMEOUT_S = 300
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
    assert os.path.isdir(path), f"{name} is not under LLM_MODELS_ROOT"
    return path


@dataclasses.dataclass
class Deployment:
    """A running orchestrator and its workers, by role."""

    orchestrator: RemoteDisaggOpenAIServer
    workers: dict[str, list[RemoteOpenAIServer]]
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
    [disagg_port] = get_free_ports(1)
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
    with contextlib.ExitStack() as stack:
        try:
            # The orchestrator starts first, so its port is taken while the workers start.
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
                    wait_ready=False,
                )
            )
            # Each worker picks its own port: on the chain, a worker frontend whose port
            # is taken before it binds then restarts on another one.
            workers: dict[str, list[RemoteOpenAIServer]] = {role: [] for role in ROLES}
            for role in ROLES:
                for index in range(counts[role]):
                    args = ["--server_role", role, *worker_args, *(role_args or {}).get(role, ())]
                    workers[role].append(
                        stack.enter_context(
                            RemoteOpenAIServer(
                                path,
                                env=process_env,
                                cli_args=args,
                                extra_config=config,
                                log_path=str(directory / f"{role}_{index}.log"),
                                wait=False,
                            )
                        )
                    )
            # A worker that exits makes its wait raise at once. The orchestrator's /health
            # answers 200 only once workers have registered.
            for role in ROLES:
                for worker in workers[role]:
                    worker.wait_for_server(timeout=ENGINE_READY_TIMEOUT_S)
            wait_for_endpoint_ready(
                orchestrator.url_root + "/health", server_proc=orchestrator.proc
            )
            _wait_until_listed(orchestrator.url_root, counts)
        except Exception:
            _print_log_tails(directory)
            raise
        yield Deployment(orchestrator, workers, model)


def _print_log_tails(directory: Path, lines: int = 60) -> None:
    """Print the end of each log in ``directory``, for a deployment that did not start."""
    for log in sorted(directory.glob("*.log")):
        tail = log.read_text(errors="replace").splitlines()[-lines:]
        print(f"----- {log} (last {len(tail)} lines) -----", *tail, sep="\n")


def _wait_until_listed(url: str, counts: Mapping[str, int]) -> None:
    """Wait until the orchestrator routes to every worker; /health needs only one per role."""
    deadline = time.monotonic() + WORKERS_LISTED_TIMEOUT_S
    listed = {}
    while time.monotonic() < deadline:
        response = requests.get(f"{url}/cluster_info", timeout=REQUEST_TIMEOUT_S)
        if response.ok:
            listed = response.json().get("server_lists", {})
            if all(len(listed.get(role, [])) >= count for role, count in counts.items()):
                return
        time.sleep(1)
    raise AssertionError(f"the orchestrator lists workers {listed}, not {dict(counts)}")


def orchestrator_sent(deployment: Deployment) -> dict[str, int]:
    """Requests the orchestrator has sent to the context (``ctx``) and generation (``gen``) workers.

    Read from its own ``/prometheus/metrics``, which both forms have; a request it sends
    again after an error counts once.
    """
    response = requests.get(f"{deployment.url}/prometheus/metrics", timeout=REQUEST_TIMEOUT_S)
    assert response.status_code == 200, response.text
    counts = {"ctx": 0, "gen": 0}
    for role, value in _SENT.findall(response.text):
        counts[role] += int(float(value))
    return counts


def dump_names(dump_dir: Path) -> set[str]:
    """The GenerateRequests written to ``dump_dir`` so far; pass it to :func:`legs`."""
    return {path.name for path in dump_dir.glob("*.json")}


def legs(dump_dir: Path, before: set[str]) -> tuple[dict, dict]:
    """The context and generation GenerateRequests of the one request sent since ``before``.

    The context leg is the one whose ``extra.request_type`` is ``context_only``,
    the generation leg the one with ``kv.session``; both name one ``disagg_request_id``.
    """
    dumps = [
        json.loads(path.read_text())
        for path in sorted(dump_dir.glob("*.json"))
        if path.name not in before
    ]
    ctx = [
        d
        for d in dumps
        if d.get("extra", {}).get(conv.REQ_REQUEST_TYPE) == conv.REQUEST_TYPE_CONTEXT_ONLY
    ]
    gen = [d for d in dumps if "session" in d.get("kv", {})]
    assert len(ctx) == 1 and len(gen) == 1 and len(dumps) == 2, dumps
    [prefill], [decode] = ctx, gen
    ctx_id = conv.as_int(prefill["extra"][conv.REQ_DISAGG_REQUEST_ID])
    gen_id = conv.as_int(decode["kv"]["session"]["attributes_struct"]["disagg_request_id"])
    assert ctx_id is not None and ctx_id == gen_id, (prefill, decode)
    return prefill, decode

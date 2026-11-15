# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The handoff between the two legs of a disaggregated request, on real engines (DG-16, DG-82).

A context and a generation worker, each with ``--server_role``, register
through the orchestrator's cluster; the orchestrator and both workers write
their perf records to one directory. The tests run on the in-process workers as well
as on the OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN``,
``tests/integration/defs/openengine_chain``), where each worker is a frontend
and a TensorRT-LLM OpenEngine server:

- NT-080: a request whose first token is the end-of-sequence token ends on
  the context leg: the context worker serves it and the generation worker is
  sent nothing.
- NT-179: the three perf records of a request carry one ``disagg_request_id``,
  the orchestrator's; on the chain each engine's leg record carries it too.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import json
import os
import secrets
import time
from pathlib import Path

import pytest
import requests
import torch
from test_common.http_utils import wait_for_endpoint_ready
from transformers import AutoTokenizer
from utils.llm_data import llm_models_root

from tensorrt_llm._utils import get_free_ports
from tensorrt_llm.grpc.openengine.streaming import LEG_DURATIONS_FILE_PREFIX

from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

pytestmark = [pytest.mark.threadleak(enabled=False), pytest.mark.timeout(3600)]

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
MODEL_SUBDIR = os.path.join("Qwen3", "Qwen3-0.6B")
MODEL_NAME = "Qwen3/Qwen3-0.6B"
PROMPT = "The capital of France is Paris. The capital of Germany is"
REQUEST_TIMEOUT_S = 120
RECORD_WAIT_S = 30
ENGINE_READY_TIMEOUT_S = 1200
# The kinds of perf record files: perf_metrics-<kind>-*.jsonl.
KINDS = ("disagg", "context", "generation")
# How long the generation worker must stay without a record once the context worker has written its.
QUIET_S = 5


def _model_dir() -> str:
    root = llm_models_root()
    path = None if root is None else os.path.join(str(root), MODEL_SUBDIR)
    assert path is not None and os.path.isdir(path), f"{MODEL_SUBDIR} is not under LLM_MODELS_ROOT"
    return path


def _needs_gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (9, 0):
        pytest.skip("needs a GPU of Hopper or later")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """An orchestrator and two workers with roles, registered through the orchestrator's cluster."""
    _needs_gpu()
    model = _model_dir()
    directory = tmp_path_factory.mktemp("handoff_chain")
    perf_dir = directory / "perf"
    disagg_port, ctx_port, gen_port = get_free_ports(3)
    cluster = {"cluster_uri": f"http://localhost:{disagg_port}", "cluster_name": ""}
    auth_key = secrets.token_hex(32)
    perf_config = {
        "return_perf_metrics": True,
        "perf_metrics_output_dir": str(perf_dir),
        "perf_metrics_max_requests": 1000,
    }
    worker_config = {
        "model": MODEL_NAME,
        "disagg_cluster": cluster,
        "internal_request_auth_key": auth_key,
        "cache_transceiver_config": {"backend": "DEFAULT"},
        "kv_cache_config": {"free_gpu_memory_fraction": 0.2, "enable_block_reuse": False},
        "disable_overlap_scheduler": True,
        "cuda_graph_config": None,
        **perf_config,
    }
    # Both workers run on this host: loopback is the only TCP device UCX needs.
    env = {**os.environ, "UCX_NET_DEVICES": "lo"}

    def worker(role: str, port: int) -> RemoteOpenAIServer:
        return RemoteOpenAIServer(
            model,
            port=port,
            env=env,
            cli_args=["--server_role", role],
            llmapi_launch=False,
            extra_config=worker_config,
            log_path=str(directory / f"output_{role}.log"),
            wait=False,
        )

    disagg_config = {
        "hostname": "localhost",
        "port": disagg_port,
        "disagg_cluster": cluster,
        "internal_request_auth_key": auth_key,
        **perf_config,
    }
    with worker("context", ctx_port) as ctx, worker("generation", gen_port) as gen:
        with RemoteDisaggOpenAIServer(
            ctx_servers=[],
            gen_servers=[],
            port=disagg_port,
            llmapi_launch=False,
            disagg_config=disagg_config,
        ) as orchestrator:
            for server in (ctx, gen):
                server.wait_for_server(timeout=ENGINE_READY_TIMEOUT_S)
            wait_for_endpoint_ready(orchestrator.url_root + "/health")
            yield {"orchestrator": orchestrator, "perf_dir": perf_dir, "model": model}


def _jsonl(path: Path) -> list[dict]:
    """The records of a JSONL file another process may still be appending to.

    A last line without its newline is still being written and is left out.
    """
    text = path.read_text()
    lines = text.splitlines()
    if lines and not text.endswith("\n"):
        lines.pop()
    return [json.loads(line) for line in lines if line]


def _records(perf_dir: Path) -> dict[str, list[dict]]:
    """The perf records in ``perf_dir`` by the kind of server that wrote them.

    A line still being written is left out until it is complete.
    """
    return {
        kind: [
            record
            for path in sorted(perf_dir.glob(f"perf_metrics-{kind}-*.jsonl"))
            for record in _jsonl(path)
        ]
        for kind in KINDS
    }


def _wait_for_records(perf_dir: Path, counts: dict[str, int]) -> dict[str, list[dict]]:
    """The records once each kind in ``counts`` has at least that many."""
    deadline = time.monotonic() + RECORD_WAIT_S
    while True:
        records = _records(perf_dir)
        if all(len(records[kind]) >= count for kind, count in counts.items()):
            return records
        assert time.monotonic() < deadline, (
            f"wanted {counts}, have { {kind: len(found) for kind, found in records.items()} }"
        )
        time.sleep(0.5)


def _leg_records(perf_dir: Path, disagg_request_id: int) -> list[dict]:
    """The engines' leg records of one request, once both legs have one or the wait has ended."""
    deadline = time.monotonic() + RECORD_WAIT_S
    while True:
        records = [
            record
            for path in perf_dir.glob(f"{LEG_DURATIONS_FILE_PREFIX}-*.jsonl")
            for record in _jsonl(path)
            if record.get("disagg_request_id") == disagg_request_id
        ]
        legs = {record.get("leg") for record in records}
        if {"context", "generation"} <= legs or time.monotonic() > deadline:
            return records
        time.sleep(0.5)


def _completion(served, **fields) -> requests.Response:
    payload = {"model": MODEL_NAME, "prompt": PROMPT, "max_tokens": 8, "temperature": 0.0}
    return requests.post(
        f"{served['orchestrator'].url_root}/v1/completions",
        json={**payload, **fields},
        timeout=REQUEST_TIMEOUT_S,
    )


def test_ctx_finished_on_eos_skips_the_gen_leg(served):
    """NT-080 (DG-16): a first token that ends the sequence ends the request on the context leg."""
    perf_dir = served["perf_dir"]
    eos = AutoTokenizer.from_pretrained(served["model"]).eos_token_id
    assert isinstance(eos, int), eos
    # The largest bias makes the end-of-sequence token the first token of a greedy request.
    forced_eos = {"logit_bias": {str(eos): 100}}
    before = {kind: len(found) for kind, found in _records(perf_dir).items()}

    response = _completion(served, **forced_eos)
    assert response.status_code == 200, response.text
    (choice,) = response.json()["choices"]
    assert choice["finish_reason"] == "stop", choice

    # Streamed, the request still ends on the context leg: the stream carries no text.
    streamed = _completion(served, stream=True, **forced_eos)
    assert streamed.status_code == 200, streamed.text
    events = [line for line in streamed.text.splitlines() if line.startswith("data: ")]
    assert events and events[-1] == "data: [DONE]", streamed.text
    for event in events[:-1]:
        for chunk in json.loads(event.removeprefix("data: "))["choices"]:
            assert not chunk.get("text"), streamed.text

    # The context worker recorded both requests; the generation worker records neither, even
    # after a quiet spell.
    _wait_for_records(perf_dir, {"context": before["context"] + 2})
    time.sleep(QUIET_S)
    assert len(_records(perf_dir)["generation"]) == before["generation"]


def _ids(records: dict[str, list[dict]]) -> dict[str, list]:
    """The disagg_request_id of each record, by kind."""
    return {kind: [r.get("disagg_request_id") for r in found] for kind, found in records.items()}


def test_one_disagg_request_id_across_orchestrator_frontends_and_engines(served):
    """NT-179 (DG-82): every record of a request carries the orchestrator's disagg_request_id."""
    perf_dir = served["perf_dir"]
    seen = {i for ids in _ids(_records(perf_dir)).values() for i in ids}

    response = _completion(served)
    assert response.status_code == 200, response.text
    (choice,) = response.json()["choices"]
    assert choice["text"], choice

    # The request is the one new id that the orchestrator and both workers each record once.
    # Records of an earlier test may still be landing; they carry ids of their own.
    deadline = time.monotonic() + RECORD_WAIT_S
    while True:
        ids = _ids(_records(perf_dir))
        new = {i for i in ids["disagg"] if i not in seen}
        mine = [i for i in new if all(i in ids[kind] for kind in ("context", "generation"))]
        if mine or time.monotonic() > deadline:
            break
        time.sleep(0.5)
    assert len(mine) == 1, (
        mine,
        {kind: [i for i in found if i not in seen] for kind, found in ids.items()},
    )
    (disagg_request_id,) = mine
    assert isinstance(disagg_request_id, int), disagg_request_id
    for kind in KINDS:
        assert ids[kind].count(disagg_request_id) == 1, (kind, disagg_request_id, ids[kind])

    if CHAIN:
        legs = _leg_records(perf_dir, disagg_request_id)
        assert sorted(record["leg"] for record in legs) == ["context", "generation"], legs

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation-first serving through the disaggregated server (DG-39b, EG-20).

With ``schedule_style: generation_first`` the disaggregated server sends the
context leg and the generation leg at once; the generation worker waits for the
KV cache the context worker hands over. Context and generation workers, each
with ``--server_role``, share one GPU behind disaggregated servers. The tests
run on the in-process workers as well as on the TensorRT-LLM OpenEngine chain
(``TRTLLM_TEST_OPENENGINE_CHAIN=trtllm``, ``tests/integration/defs/openengine_chain``),
where each worker is a frontend and a TensorRT-LLM OpenEngine server:

- NT-091: a client that goes away mid-stream leaves no request running on
  either worker (on the chain, the generation engine holds no more KV-cache
  blocks than before), and both serve the next request in full. A streamed
  generation-first response starts only once the context leg has returned,
  so by then the context worker has finished its part. The context engine's
  KV-cache blocks are not checked: a context request in transfer has left the
  executor's active requests, and an idle executor stops refreshing the load it
  reports, so until the next request the reading can stay at a sample taken
  while the transfer still held the blocks.
- NT-200: with chunked prefill and ``enable_pipelined_transfer``, a prompt of
  several chunks gets, generation-first, the greedy output that context-first
  gives on workers without pipelined transfer, streamed or not; the pipelined
  context worker refuses a context-first request.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

import grpc
import pytest
import requests
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm.grpc.openengine.bindings import openengine_pb2_grpc, server_pb2
from tensorrt_llm.grpc.openengine.streaming import LEG_DURATIONS_FILE_PREFIX
from tensorrt_llm.grpc.openengine_conventions import (
    BUCKET_GENERATION_ONLY,
    COUNTER_GENERATE_ACCEPTED_BUCKETS,
    COUNTER_PREFILL_READY_SENT,
    struct_to_dict,
)

from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

pytestmark = [pytest.mark.threadleak(enabled=False)]

MODEL_SUBDIR = os.path.join("Qwen3", "Qwen3-0.6B")
MODEL_NAME = "Qwen3/Qwen3-0.6B"
REQUEST_TIMEOUT_S = 300
READY_TIMEOUT_S = 1200
# A generation-first transfer needs the Python transceiver (the default runtime), which
# supports the NIXL backend only.
TRANSCEIVER = {"backend": "NIXL"}
KV_CACHE = {"free_gpu_memory_fraction": 0.2, "enable_block_reuse": False}
# All workers run on this host: no InfiniBand or GPUDirect copy, and loopback is the
# only TCP device UCX needs.
UCX_ENV = {"UCX_TLS": "^ib,gdr_copy", "UCX_NET_DEVICES": "lo"}


def _model_dir() -> str:
    root = llm_models_root()
    path = None if root is None else os.path.join(str(root), MODEL_SUBDIR)
    assert path is not None and os.path.isdir(path), f"{MODEL_SUBDIR} is not under LLM_MODELS_ROOT"
    return path


def _needs_gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (9, 0):
        pytest.skip("needs a GPU of Hopper or later")


def _worker(role: str, config: dict, log_dir: Path) -> RemoteOpenAIServer:
    return RemoteOpenAIServer(
        _model_dir(),
        env={**os.environ, **UCX_ENV},
        cli_args=["--server_role", role],
        extra_config=config,
        log_path=str(log_dir / f"output_{role}.log"),
        wait=False,
    )


def _pair(stack: ExitStack, config: dict, log_dir: Path) -> dict:
    """A context worker and a generation worker.

    In process the two load at the same time; on the chain one after the other.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    workers = {
        role: stack.enter_context(_worker(role, config, log_dir))
        for role in ("context", "generation")
    }
    for worker in workers.values():
        worker.wait_for_server(timeout=READY_TIMEOUT_S)
    return workers


def _orchestrator(workers: dict, style: str) -> RemoteDisaggOpenAIServer:
    """A disaggregated server in front of ``workers``; it returns once it is healthy."""
    # The workers' ports are final once they are ready: a chain frontend may move.
    ctx, gen = workers["context"], workers["generation"]
    return RemoteDisaggOpenAIServer(
        ctx_servers=[f"{ctx.host}:{ctx.port}"],
        gen_servers=[f"{gen.host}:{gen.port}"],
        disagg_config={"hostname": "localhost", "schedule_style": style},
    )


def _post(orchestrator: RemoteDisaggOpenAIServer, body: dict, stream: bool) -> requests.Response:
    return requests.post(
        orchestrator.url_for("v1", "completions"),
        json={"model": MODEL_NAME, "temperature": 0.0, **body, "stream": stream},
        stream=stream,
        timeout=REQUEST_TIMEOUT_S,
    )


def _complete(orchestrator: RemoteDisaggOpenAIServer, body: dict, stream: bool = False) -> dict:
    """POST ``body`` to ``/v1/completions``; the text, finish reason and usage of choice 0."""
    response = _post(orchestrator, body, stream)
    assert response.status_code == 200, response.text
    if not stream:
        payload = response.json()
        choice = payload["choices"][0]
        return {
            "text": choice["text"],
            "finish_reason": choice["finish_reason"],
            "usage": payload["usage"],
        }
    text, finish_reason, done = "", None, False
    for line in response.iter_lines():
        if not line.startswith(b"data: "):
            continue
        data = line[len(b"data: ") :]
        if data == b"[DONE]":
            done = True
            continue
        chunk = json.loads(data)
        for choice in chunk.get("choices", []):
            text += choice.get("text") or ""
            finish_reason = choice.get("finish_reason") or finish_reason
    assert done, "the stream ended without [DONE]"
    return {"text": text, "finish_reason": finish_reason, "usage": None}


# -- NT-091: a client that goes away -------------------------------------------------

CANCEL_CONFIG = {
    "kv_cache_config": KV_CACHE,
    "cache_transceiver_config": TRANSCEIVER,
    "disable_overlap_scheduler": True,
    "cuda_graph_config": None,
    # In process a worker reports its running requests in its iteration stats; on the
    # chain the stats also give each engine's load its KV-cache blocks.
    "enable_iter_perf_stats": True,
}
# Long enough that a request left running is still running when the check gives up.
LONG_GENERATION = {
    "prompt": "Write a long story about a lighthouse.",
    "max_tokens": 16000,
    "ignore_eos": True,
}
IDLE_WAIT_S = 15


@pytest.fixture
def cancel_servers(tmp_path):
    """A generation-first disaggregated server, stopped right after its one test."""
    _needs_gpu()
    with ExitStack() as stack:
        workers = _pair(stack, CANCEL_CONFIG, tmp_path)
        orchestrator = stack.enter_context(_orchestrator(workers, "generation_first"))
        yield orchestrator, workers


def _engine_load(worker: RemoteOpenAIServer):
    with grpc.insecure_channel(worker._chain_handle.engine_endpoint) as channel:
        return openengine_pb2_grpc.ControlStub(channel).GetLoad(
            server_pb2.GetLoadRequest(), timeout=30
        )


def _running_requests(worker: RemoteOpenAIServer) -> int:
    """The requests ``worker`` runs: its engine's load on the chain, else its iteration stats."""
    if worker._chain_handle is not None:
        return _engine_load(worker).running_requests
    # The iteration stats since the last read; an empty list reads as no running request.
    response = requests.get(worker.url_for("metrics"), timeout=30)
    assert response.status_code == 200, response.text
    stats = response.json()
    return stats[-1]["numActiveRequests"] if stats else 0


def _wait_idle(worker: RemoteOpenAIServer, role: str, since: str) -> None:
    deadline = time.monotonic() + IDLE_WAIT_S
    # Two idle readings in a row.
    idle = 0
    while idle < 2:
        assert time.monotonic() < deadline, f"the {role} worker still runs a request {since}"
        idle = idle + 1 if _running_requests(worker) == 0 else 0
        time.sleep(0.5)


def _handoffs(workers: dict) -> tuple[int, int]:
    """Hand-offs the context engine sent and generation-only calls the generation engine
    accepted, so far (the chain)."""
    sent = struct_to_dict(_engine_load(workers["context"]).attributes)
    accepted = struct_to_dict(_engine_load(workers["generation"]).attributes)
    return (
        int(sent.get(COUNTER_PREFILL_READY_SENT, 0)),
        int(accepted.get(COUNTER_GENERATE_ACCEPTED_BUCKETS[BUCKET_GENERATION_ONLY], 0)),
    )


def _used_kv_blocks(worker: RemoteOpenAIServer) -> int:
    load = _engine_load(worker)
    assert load.HasField("used_kv_blocks"), f"the engine reports no KV-cache load: {load}"
    return load.used_kv_blocks


def _wait_kv_released(worker: RemoteOpenAIServer, role: str, baseline: int) -> None:
    """On the chain: the engine holds no more KV-cache blocks than before the request."""
    deadline = time.monotonic() + IDLE_WAIT_S
    while (used := _used_kv_blocks(worker)) > baseline:
        assert time.monotonic() < deadline, (
            f"the {role} engine still holds {used} KV-cache blocks, {baseline} before"
        )
        time.sleep(0.5)


def test_gen_first_cancel_closes_both_legs(cancel_servers):
    """NT-091 (DG-39b): a client gone mid-stream leaves both workers idle; both serve on."""
    orchestrator, workers = cancel_servers
    chain = all(worker._chain_handle is not None for worker in workers.values())
    for role, worker in workers.items():
        _wait_idle(worker, role, "before the test")
    if chain:
        kv_before = _used_kv_blocks(workers["generation"])
        sent_before, accepted_before = _handoffs(workers)

    with _post(orchestrator, LONG_GENERATION, stream=True) as response:
        assert response.status_code == 200, response.text
        chunks = 0
        for line in response.iter_lines():
            if line.startswith(b"data: "):
                chunks += 1
                if chunks == 3:
                    break
        assert chunks == 3
        assert _running_requests(workers["generation"]) >= 1
    # Leaving the block closes the connection with the stream unread.
    for role, worker in workers.items():
        _wait_idle(worker, role, "after its client left")
    if chain:
        _wait_kv_released(workers["generation"], "generation", kv_before)

    # Both workers serve the next request in full.
    result = _complete(
        orchestrator, {"prompt": "The capital of France is", "max_tokens": 8, "ignore_eos": True}
    )
    assert result["usage"]["completion_tokens"] == 8, result
    if chain:
        sent, accepted = _handoffs(workers)
        # Two requests: the context engine sent two hand-offs, and the generation engine
        # accepted two generation-only calls.
        assert (sent - sent_before, accepted - accepted_before) == (2, 2), (
            (sent_before, accepted_before),
            (sent, accepted),
        )


# -- NT-200: pipelined transfer with chunked prefill ---------------------------------

MAX_NUM_TOKENS = 256
CHUNKED_CONFIG = {
    "kv_cache_config": KV_CACHE,
    "cache_transceiver_config": TRANSCEIVER,
    "enable_chunked_prefill": True,
    "max_num_tokens": MAX_NUM_TOKENS,
    "disable_overlap_scheduler": True,
    "cuda_graph_config": None,
}
PIPELINED_TRANSCEIVER = {**TRANSCEIVER, "enable_pipelined_transfer": True}
# Several chunks of MAX_NUM_TOKENS each.
LONG_PROMPT = (
    " ".join(f"Item {i}: the lighthouse keeper logs ship number {i * 7}." for i in range(80))
    + " Summarize the log in one sentence:"
)
OUTPUT_TOKENS = 32


@pytest.fixture
def pipelined_servers(tmp_path):
    """Pipelined workers behind generation-first and context-first disaggregated servers.

    The reference is context-first on another pair of workers, alike but without
    pipelined transfer: a context worker with pipelined transfer serves
    generation-first requests only.
    """
    _needs_gpu()
    perf_dir = tmp_path / "perf"
    pipelined_config = {
        **CHUNKED_CONFIG,
        "cache_transceiver_config": PIPELINED_TRANSCEIVER,
        "perf_metrics_output_dir": str(perf_dir),
    }
    with ExitStack() as stack:
        pipelined = _pair(stack, pipelined_config, tmp_path / "pipelined")
        plain = _pair(stack, CHUNKED_CONFIG, tmp_path / "plain")
        yield {
            "generation_first": stack.enter_context(_orchestrator(pipelined, "generation_first")),
            "pipelined_context_first": stack.enter_context(
                _orchestrator(pipelined, "context_first")
            ),
            "reference": stack.enter_context(_orchestrator(plain, "context_first")),
            "perf": perf_dir,
        }


def _kv_transfers(perf_dir: Path) -> list[dict]:
    """The KV transfers the generation engines recorded, from the complete lines written so far."""
    transfers = []
    for path in sorted(perf_dir.glob(f"{LEG_DURATIONS_FILE_PREFIX}-*.jsonl")):
        text = path.read_text()
        # A writer thread appends the records: the text after the last newline may be partial.
        for line in text[: text.rfind("\n") + 1].splitlines():
            record = json.loads(line)
            if record.get("leg") == "generation":
                transfers.append(
                    {key: record.get(key) for key in ("kv_cache_size", "durations_ms")}
                )
    return transfers


def test_pipelined_transfer_with_chunked_prefill(pipelined_servers, record_property):
    """NT-200 (EG-20): chunked prefill and pipelined transfer give the reference output."""
    body = {"prompt": LONG_PROMPT, "max_tokens": OUTPUT_TOKENS, "ignore_eos": True}
    try:
        reference = _complete(pipelined_servers["reference"], body)
        # The prompt needs several chunks.
        assert reference["usage"]["prompt_tokens"] > 2 * MAX_NUM_TOKENS, reference["usage"]
        assert reference["usage"]["completion_tokens"] == OUTPUT_TOKENS, reference["usage"]

        gen_first = pipelined_servers["generation_first"]
        whole = _complete(gen_first, body)
        assert whole["usage"] == reference["usage"], (whole, reference)
        assert whole["text"] == reference["text"], (whole, reference)
        streamed = _complete(gen_first, body, stream=True)
        assert streamed["text"] == reference["text"], (streamed, reference)
        assert streamed["finish_reason"] == whole["finish_reason"] == reference["finish_reason"]

        # The context worker runs with pipelined transfer: it refuses a context-first request.
        refused = _post(pipelined_servers["pipelined_context_first"], body, stream=False)
        assert refused.status_code >= 400, refused.text
        assert "enable_pipelined_transfer" in refused.text, refused.text
    finally:
        # Supplementary evidence of the KV transfers (CH-8), not a check of its own. On the
        # chain the pipelined generation engine writes these records; in process nothing does.
        record_property("ch8_kv_transfers", json.dumps(_kv_transfers(pipelined_servers["perf"])))

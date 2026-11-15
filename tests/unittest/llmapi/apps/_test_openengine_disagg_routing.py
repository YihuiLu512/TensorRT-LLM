# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The orchestrator's conversation router keeps a conversation on one context worker (DG-20, DG-21).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` with two
context workers and one generation worker of Qwen3-0.6B on one GPU, in-process
or on the OpenEngine chain, behind an orchestrator whose context router is
``conversation``. The orchestrator writes a perf record of each request, which
names the context worker it was sent to; on the chain each context engine's
count of accepted context-only calls shows which engine served it as well.

- NT-082 (DG-20, DG-21): the turns of a conversation whose id is in the body
  (``conversation_params.conversation_id``) all go to one context worker, and
  so do those of a conversation whose id is in the ``X-Correlation-ID`` header.
  With ids in both places, the body's decides.
"""

import json
import time
from pathlib import Path

import grpc
import pytest
import requests

from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.grpc.openengine.bindings import openengine_pb2_grpc, server_pb2

from ._test_openengine_disagg_common import CHAIN, REQUEST_TIMEOUT_S, disagg_deployment

pytestmark = pytest.mark.threadleak(enabled=False)

NUM_CTX = 2
TURNS = 4
# New conversations tried before two of them are on different context workers.
MAX_NEW_CONVERSATIONS = 8
RECORD_WAIT_S = 30
PROMPT = "Hello, this is a test prompt for conversation routing"
HEADER = "X-Correlation-ID"


@pytest.fixture(scope="module")
def perf_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("disagg_routing_perf")


@pytest.fixture(scope="module")
def deployment(tmp_path_factory, perf_dir):
    with disagg_deployment(
        tmp_path_factory.mktemp("disagg_routing"),
        counts={"context": NUM_CTX},
        orchestrator_config={
            "context_servers": {
                "num_instances": 0,
                "urls": [],
                "router": {"type": "conversation"},
            },
            "perf_metrics_output_dir": str(perf_dir),
        },
    ) as deployment:
        _wait_for_context_workers(deployment, NUM_CTX)
        yield deployment


def _wait_for_context_workers(deployment, count: int, timeout: float = 300) -> None:
    """Wait until the orchestrator routes to ``count`` context workers."""
    deadline = time.monotonic() + timeout
    servers = []
    while time.monotonic() < deadline:
        response = requests.get(f"{deployment.url}/cluster_info", timeout=REQUEST_TIMEOUT_S)
        if response.ok:
            servers = response.json().get("server_lists", {}).get("context", [])
            if len(servers) >= count:
                return
        time.sleep(1)
    raise AssertionError(f"the orchestrator lists context workers {servers}, not {count}")


def _records(perf_dir: Path) -> list[dict]:
    """The orchestrator's perf records so far, in the order it wrote them."""
    records = []
    for path in sorted(perf_dir.glob("perf_metrics-disagg-*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _context_only_accepted(endpoint: str) -> int:
    with grpc.insecure_channel(endpoint) as channel:
        load = openengine_pb2_grpc.ControlStub(channel).GetLoad(
            server_pb2.GetLoadRequest(), timeout=30
        )
    key = conv.COUNTER_GENERATE_ACCEPTED_BUCKETS[conv.BUCKET_CONTEXT_ONLY]
    value = conv.as_int(dict(load.attributes).get(key))
    assert value is not None, f"the engine at {endpoint} reports no {key}"
    return value


def _engine_counts(deployment) -> list[int]:
    """Each context engine's accepted context-only calls, by worker index (chain only)."""
    return [
        _context_only_accepted(worker._chain_handle.engine_endpoint)
        for worker in deployment.workers["context"]
    ]


def _ctx_index(deployment, ctx_server: str) -> int:
    port = int(ctx_server.rsplit(":", 1)[1])
    ports = [worker.port for worker in deployment.workers["context"]]
    assert port in ports, (ctx_server, ports)
    return ports.index(port)


def _turn(deployment, perf_dir, prompt, *, body_id=None, header_id=None) -> tuple[int, str]:
    """Send one turn through the orchestrator; the index of its context worker, and its text."""
    seen = len(_records(perf_dir))
    counts = _engine_counts(deployment) if CHAIN else None
    body = {
        "model": deployment.model,
        "prompt": prompt,
        "max_tokens": 8,
        "ignore_eos": True,
        "temperature": 0.0,
    }
    if body_id is not None:
        body["conversation_params"] = {"conversation_id": body_id}
    headers = {HEADER: header_id} if header_id is not None else None
    response = requests.post(
        f"{deployment.url}/v1/completions", json=body, headers=headers, timeout=REQUEST_TIMEOUT_S
    )
    assert response.status_code == 200, response.text
    text = response.json()["choices"][0]["text"]
    deadline = time.monotonic() + RECORD_WAIT_S
    while len(records := _records(perf_dir)) <= seen:
        assert time.monotonic() < deadline, "the orchestrator wrote no perf record of the request"
        time.sleep(0.1)
    new = records[seen:]
    assert len(new) == 1, new
    index = _ctx_index(deployment, new[0]["ctx_server"])
    if CHAIN:
        after = _engine_counts(deployment)
        expected = [count + (i == index) for i, count in enumerate(counts)]
        assert after == expected, (index, counts, after)
    return index, text


def _conversation(deployment, perf_dir, conversation_id: str, *, in_body: bool) -> int:
    """Send ``TURNS`` turns of one conversation; the context worker all of them went to."""
    prompt, seen = PROMPT, []
    ids = {"body_id": conversation_id} if in_body else {"header_id": conversation_id}
    for _ in range(TURNS):
        index, text = _turn(deployment, perf_dir, prompt, **ids)
        seen.append(index)
        prompt += text
    assert len(set(seen)) == 1, (conversation_id, seen)
    return seen[0]


def test_conversation_id_from_body_and_headers_sticks_to_one_ctx_worker(deployment, perf_dir):
    homes = {
        "nt082-body": _conversation(deployment, perf_dir, "nt082-body", in_body=True),
        "nt082-header": _conversation(deployment, perf_dir, "nt082-header", in_body=False),
    }

    # Two conversations on different context workers, to tell which id decides.
    first = "nt082-body"
    other = next((cid for cid, home in homes.items() if home != homes[first]), None)
    for i in range(MAX_NEW_CONVERSATIONS):
        if other is not None:
            break
        cid = f"nt082-new-{i}"
        homes[cid], _ = _turn(deployment, perf_dir, f"Unique prompt number {i}", header_id=cid)
        if homes[cid] != homes[first]:
            other = cid
    assert other is not None, f"new conversations all went to one context worker: {homes}"

    for body_id, header_id in ((first, other), (other, first)):
        index, _ = _turn(deployment, perf_dir, PROMPT, body_id=body_id, header_id=header_id)
        assert index == homes[body_id], (body_id, header_id, homes)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated workers with several serving frontends (F-39, DG-23).

- NT-069: context and generation workers each run two frontends and register
  through the orchestrator's cluster; two-stage requests complete and one
  process of each worker, the launcher frontend, registered (the cluster
  storage keys a registration by the registering process). On the chain, where
  every frontend process counts the Generate calls it sent its engine, the
  two frontends of a worker are both connected, and the calls they sent
  together are each leg once, all received by the one engine.
- NT-085: the context worker's two frontends also listen on affinity ports
  (``serve_frontend_affinity_port_stride``), which answer ``/health``, and
  the orchestrator is given them; requests with a conversation id complete,
  and an orchestrator given a stride the worker does not listen at logs the
  refused affinity port and sends the context leg to the shared port. On the
  chain the context legs of a conversation are all sent by one frontend, and
  two conversations picked for different frontends are sent by different
  ones, the attached frontend serving its legs as the launcher does.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import zlib

import pytest
import requests

__extra_import_path__ = ["~/tests/unittest/serve/openengine_poc/wt53"]
from wt53_gpu import (  # noqa: E402
    CHAIN,
    MODEL_NAME,
    REQUEST_TIMEOUT_S,
    another_orchestrator,
    chain_log,
    cluster,
    engine_counts,
    members,
    port_block,
    registered_workers,
    sent_by_frontend,
)

pytestmark = [pytest.mark.threadleak(enabled=False)]

NUM_FRONTENDS = 2
NUM_REQUESTS = 8
AFFINITY_STRIDE = 7
# A stride the context worker does not listen at, for an orchestrator whose affinity ports refuse.
WRONG_STRIDE = 11
REFUSED_LOG = "refused the connection"


def _affinity(stride: int) -> dict:
    return {"num_serve_frontends": NUM_FRONTENDS, "serve_frontend_affinity_port_stride": stride}


def _conversations() -> dict[int, str]:
    """A conversation id for each frontend index, as the orchestrator picks it (crc32 % K)."""
    picked: dict[int, str] = {}
    i = 0
    while len(picked) < NUM_FRONTENDS:
        conversation_id = f"conversation-{i}"
        picked.setdefault(zlib.crc32(conversation_id.encode()) % NUM_FRONTENDS, conversation_id)
        i += 1
    return picked


CONVERSATIONS = _conversations()


def _complete(orchestrator, index: int, conversation_id: str | None = None) -> None:
    body = {
        "model": MODEL_NAME,
        "prompt": f"Request {index}: the capital of France is",
        "max_tokens": 4,
        "temperature": 0.0,
    }
    if conversation_id is not None:
        body["conversation_params"] = {"conversation_id": conversation_id}
    response = requests.post(
        f"{orchestrator.url_root}/v1/completions", json=body, timeout=REQUEST_TIMEOUT_S
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["text"], response.text


def _delta(after: dict[int, int], before: dict[int, int]) -> dict[int, int]:
    return {pid: count - before.get(pid, 0) for pid, count in after.items()}


@pytest.fixture
def two_frontend_workers(tmp_path):
    with cluster(
        tmp_path, register=True, args=["--num_serve_frontends", str(NUM_FRONTENDS)]
    ) as served:
        yield served


def test_multi_frontend_worker_two_stage(two_frontend_workers):
    """NT-069 (F-39): two-frontend workers register once each and serve two-stage requests."""
    served = two_frontend_workers
    if CHAIN:
        for role, servers in served.workers.items():
            log = chain_log(servers[0], "frontend")
            assert f"All {NUM_FRONTENDS} serving frontends are connected" in log, role
        before = {role: sent_by_frontend(s[0]) for role, s in served.workers.items()}
        engine_before = {role: engine_counts(served, role) for role in served.workers}
    for i in range(NUM_REQUESTS):
        _complete(served.orchestrator, i)
    # One process of each worker registered, under the worker's port; the routers list it once.
    registered = registered_workers(served)
    listed = members(served.orchestrator)
    for role, servers in served.workers.items():
        keys = [key for key in registered if f"/{role.upper()}-" in key]
        assert len(keys) == 1, registered
        assert f":{servers[0].port}-" in keys[0], (keys, servers[0].port)
        assert listed[role] and all(a.endswith(f":{servers[0].port}") for a in listed[role]), listed
    if not CHAIN:
        return
    for role, servers in served.workers.items():
        sent = {
            pid: n for pid, n in _delta(sent_by_frontend(servers[0]), before[role]).items() if n
        }
        # Each frontend process that sent a call counts its own; together they sent each leg once.
        assert 1 <= len(sent) <= NUM_FRONTENDS, (role, sent)
        assert sum(sent.values()) == NUM_REQUESTS, (role, sent)
        (engine_after,) = engine_counts(served, role)
        assert engine_after - engine_before[role][0] == NUM_REQUESTS, role


@pytest.fixture
def affinity_cluster(tmp_path):
    # The worker's port and the affinity ports of both strides, kept off the harness's picks.
    port = port_block(WRONG_STRIDE * NUM_FRONTENDS + 1)
    with cluster(
        tmp_path,
        ports={"context": port},
        role_config={"context": _affinity(AFFINITY_STRIDE)},
        servers={"context": _affinity(AFFINITY_STRIDE)},
    ) as served:
        yield served


@pytest.fixture
def wrong_stride_orchestrator(affinity_cluster, tmp_path):
    log_path = tmp_path / "output_orchestrator_wrong_stride.log"
    with another_orchestrator(
        affinity_cluster, log_path, servers={"context": _affinity(WRONG_STRIDE)}
    ) as orchestrator:
        yield orchestrator, log_path


def test_ctx_frontend_affinity_port_is_used_and_falls_back(
    affinity_cluster, wrong_stride_orchestrator
):
    """NT-085 (DG-23): a conversation's context leg goes to its frontend's affinity port, else the shared one."""
    served = affinity_cluster
    ctx = served.ctx
    for i in range(NUM_FRONTENDS):
        affinity_port = ctx.port + AFFINITY_STRIDE * (i + 1)
        health = requests.get(f"http://{ctx.host}:{affinity_port}/health", timeout=30)
        assert health.status_code == 200, (i, health.text)

    # Each conversation's legs land on one frontend; the two conversations on two.
    landed = []
    for frontend, conversation_id in sorted(CONVERSATIONS.items()):
        before = sent_by_frontend(ctx) if CHAIN else None
        for i in range(NUM_REQUESTS):
            _complete(served.orchestrator, i, conversation_id)
        if CHAIN:
            sent = {pid: n for pid, n in _delta(sent_by_frontend(ctx), before).items() if n}
            assert list(sent.values()) == [NUM_REQUESTS], (frontend, conversation_id, sent)
            landed.append(next(iter(sent)))
    if CHAIN:
        assert len(set(landed)) == NUM_FRONTENDS, landed

    # An orchestrator given a stride the worker does not listen at falls back to the shared port.
    orchestrator, log_path = wrong_stride_orchestrator
    before = sum(sent_by_frontend(ctx).values()) if CHAIN else None
    for i in range(NUM_REQUESTS):
        _complete(orchestrator, i, CONVERSATIONS[0])
    if CHAIN:
        assert sum(sent_by_frontend(ctx).values()) - before == NUM_REQUESTS
    assert REFUSED_LOG in log_path.read_text(errors="replace")

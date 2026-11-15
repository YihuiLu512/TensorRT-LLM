# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated workers with several serving frontends in front of one engine (F-39, DG-23).

Runs only on the OpenEngine chain, where every frontend process of a worker
keeps its own count of the Generate calls it sent its engine.

- NT-069: context and generation workers each run two frontends and register
  through the orchestrator's cluster; two-stage requests complete, only the
  launcher frontend of each worker is registered, and the frontends of a
  worker share its engine.
- NT-085: the context worker's two frontends also listen on affinity ports
  (``serve_frontend_affinity_port_stride``) and the orchestrator is given
  them; the context leg of a request with a conversation id goes to the
  affinity port of the frontend picked for the conversation, the attached
  frontend serves its leg as the launcher does, and an orchestrator whose
  affinity ports refuse connections sends the leg to the shared port.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import zlib

import pytest
import requests

__extra_import_path__ = ["~/tests/integration/defs", "~/tests/unittest/serve/openengine_poc/wt53"]
from wt53_gpu import (  # noqa: E402
    CHAIN,
    REQUEST_TIMEOUT_S,
    another_orchestrator,
    chain_log,
    cluster,
    engine_counts,
    members,
    port_with_free_strides,
    sent_by_frontend,
)

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not CHAIN, reason="counts per frontend; runs on the chain only"),
]

NUM_FRONTENDS = 2
NUM_REQUESTS = 8
AFFINITY_STRIDE = 7
# A stride the context worker does not listen at, for an orchestrator whose affinity ports refuse.
WRONG_STRIDE = 11
REFUSED_LOG = "refused the connection"


def _complete(orchestrator, index: int, conversation_id: str | None = None):
    body = {
        "model": "Qwen3/Qwen3-0.6B",
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


def test_multi_frontend_worker_two_stage(tmp_path):
    """NT-069 (F-39): two-frontend workers register once each and serve two-stage requests."""
    with cluster(
        tmp_path, register=True, args=["--num_serve_frontends", str(NUM_FRONTENDS)]
    ) as served:
        for role, servers in served.workers.items():
            log = chain_log(servers[0], "frontend")
            assert f"All {NUM_FRONTENDS} serving frontends are connected" in log, role
        before = {role: sent_by_frontend(s[0]) for role, s in served.workers.items()}
        engine_before = {role: engine_counts(served, role) for role in served.workers}
        for i in range(NUM_REQUESTS):
            _complete(served.orchestrator, i)
        for role, servers in served.workers.items():
            sent = _delta(sent_by_frontend(servers[0]), before[role])
            # Every frontend process of the worker counts; together they sent each leg once.
            assert len(sent) == NUM_FRONTENDS, (role, sent)
            assert sum(sent.values()) == NUM_REQUESTS, (role, sent)
            (engine_after,) = engine_counts(served, role)
            assert engine_after - engine_before[role][0] == NUM_REQUESTS, role
        # Only the launcher of each worker registered, under the worker's port.
        listed = members(served.orchestrator)
        for role, servers in served.workers.items():
            assert len(listed[role]) == 1, listed
            assert listed[role][0].endswith(f":{servers[0].port}"), (listed, servers[0].port)


def _conversations() -> dict[int, str]:
    """A conversation id for each frontend index, as the orchestrator picks it (crc32 % K)."""
    picked: dict[int, str] = {}
    i = 0
    while len(picked) < NUM_FRONTENDS:
        conversation_id = f"conversation-{i}"
        picked.setdefault(zlib.crc32(conversation_id.encode()) % NUM_FRONTENDS, conversation_id)
        i += 1
    return picked


def test_ctx_frontend_affinity_port_is_used_and_falls_back(tmp_path):
    """NT-085 (DG-23): a conversation's context leg goes to its frontend's affinity port, else the shared one."""
    port = port_with_free_strides((AFFINITY_STRIDE, WRONG_STRIDE), NUM_FRONTENDS)
    affinity = {
        "num_serve_frontends": NUM_FRONTENDS,
        "serve_frontend_affinity_port_stride": AFFINITY_STRIDE,
    }
    with cluster(
        tmp_path,
        ports={"context": port},
        role_config={"context": affinity},
        servers={"context": affinity},
    ) as served:
        ctx = served.ctx
        for i in range(NUM_FRONTENDS):
            health = requests.get(
                f"http://{ctx.host}:{port + AFFINITY_STRIDE * (i + 1)}/health", timeout=30
            )
            assert health.status_code == 200, (i, health.text)

        # Each conversation's legs land on one frontend; the two conversations on two.
        landed = []
        for frontend, conversation_id in sorted(_conversations().items()):
            before = sent_by_frontend(ctx)
            for i in range(NUM_REQUESTS):
                _complete(served.orchestrator, i, conversation_id)
            sent = {pid: n for pid, n in _delta(sent_by_frontend(ctx), before).items() if n}
            assert list(sent.values()) == [NUM_REQUESTS], (frontend, conversation_id, sent)
            landed.append(next(iter(sent)))
        assert len(set(landed)) == NUM_FRONTENDS, landed

        # An orchestrator given a stride the worker does not listen at falls back to the shared port.
        log_path = tmp_path / "output_orchestrator_wrong_stride.log"
        wrong = {
            "num_serve_frontends": NUM_FRONTENDS,
            "serve_frontend_affinity_port_stride": WRONG_STRIDE,
        }
        with another_orchestrator(served, log_path, servers={"context": wrong}) as orchestrator:
            before = sum(sent_by_frontend(ctx).values())
            for i in range(NUM_REQUESTS):
                _complete(orchestrator, i, _conversations()[0])
            assert sum(sent_by_frontend(ctx).values()) - before == NUM_REQUESTS
        assert REFUSED_LOG in log_path.read_text(errors="replace")

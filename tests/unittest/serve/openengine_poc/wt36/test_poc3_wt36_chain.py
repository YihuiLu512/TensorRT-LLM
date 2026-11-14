# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-test of the CPU disaggregated chain (``cpu_disagg_chain``), for each handoff shape.

It checks what the chain itself provides: every server starts and stops,
each worker frontend serves through its own fake engine, and a request sent
to the orchestrator reaches the context worker frontend as the context leg
of a disaggregated request. What the worker frontends then do with the two
legs is tested where they are implemented.
"""

from __future__ import annotations

import time

import psutil
import pytest
import requests

from tensorrt_llm.grpc.openengine_conventions import (
    BUCKET_AGGREGATED,
    COUNTER_GENERATE_ACCEPTED_BUCKETS,
    COUNTER_GENERATE_RECEIVED,
    SERVE_COUNTER_GENERATE_SENT,
)

__extra_import_path__ = [".."]
from cpu_disagg_chain import MODEL, SHAPES, CpuDisaggChain, Hang  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


@pytest.fixture(scope="module", params=SHAPES)
def chain(request, tmp_path_factory):
    with CpuDisaggChain(tmp_path_factory.mktemp(f"chain-{request.param}"), request.param) as chain:
        yield chain


def _delta(before: dict, after: dict, role: str, side: str, key: str) -> int:
    return after[role][side].get(key, 0) - before[role][side].get(key, 0)


def test_every_server_of_the_chain_is_up(chain):
    for worker in chain.workers:
        assert worker.engine.port and worker.url.startswith("http://127.0.0.1:")
        assert worker.process.popen.poll() is None, worker.log()
        assert "--openengine-endpoint" in worker.process.command
        info = worker.engine.server_info()
        assert info.kv_connector.enabled
    assert chain.ctx.engine.instance_id != chain.gen.engine.instance_id
    assert requests.get(f"{chain.url}/health", timeout=10).status_code == 200
    # Each frontend serves the chain's model from its engine.
    for worker in chain.workers:
        models = requests.get(f"{worker.url}/v1/models", timeout=10).json()
        assert [model["id"] for model in models["data"]] == [MODEL]


def test_each_worker_frontend_serves_through_its_own_engine(chain):
    for worker in chain.workers:
        before = chain.snapshot()
        response = chain.complete(url=worker.url)
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["text"]
        after = chain.snapshot()
        for role in ("context", "generation"):
            expected = 1 if role == worker.role else 0
            sent = _delta(before, after, role, "frontend", SERVE_COUNTER_GENERATE_SENT)
            received = _delta(before, after, role, "engine", COUNTER_GENERATE_RECEIVED)
            assert (sent, received) == (expected, expected), (role, before, after)
        bucket = COUNTER_GENERATE_ACCEPTED_BUCKETS[BUCKET_AGGREGATED]
        assert _delta(before, after, worker.role, "engine", bucket) == 1
        recorded = worker.http_requests()[-1]
        assert (recorded["method"], recorded["path"], recorded["status"]) == (
            "POST",
            "/v1/completions",
            200,
        )


def _completion_legs(worker, seen: int) -> list[dict]:
    return [
        record for record in worker.http_requests()[seen:] if record["path"] == "/v1/completions"
    ]


def test_a_request_to_the_orchestrator_reaches_the_context_worker_as_its_context_leg(chain):
    seen = {worker.role: len(worker.http_requests()) for worker in chain.workers}
    response = chain.complete(prompt="He llo", max_tokens=3)
    legs = _completion_legs(chain.ctx, seen["context"])
    assert legs, chain.orchestrator_log()[-4000:]
    assert legs[0]["body"]["disaggregated_params"]["request_type"] == "context_only"
    if response.status_code == 200:
        # A request answered in full went through the generation worker too.
        assert _completion_legs(chain.gen, seen["generation"]), chain.orchestrator_log()[-4000:]


def test_a_worker_frontend_keeps_serving_after_its_engine_leaves_a_call_unfinished(chain):
    # An engine can leave a call without a terminal event, as the
    # TensorRT-LLM server does with a failed decode call. The client gives
    # up, and the worker frontend keeps serving.
    engine = chain.gen.engine
    engine.push_script(Hang())
    with pytest.raises(requests.Timeout):
        chain.complete(url=chain.gen.url, timeout=3)
    deadline = time.monotonic() + 30
    while not engine.running_request_ids:
        assert time.monotonic() < deadline, "the call never reached the engine"
        time.sleep(0.1)
    assert sum(engine.abort(request_id) for request_id in engine.running_request_ids) == 1
    response = chain.complete(url=chain.gen.url)
    assert response.status_code == 200, response.text


def _gone(process: psutil.Process) -> bool:
    try:
        return process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_stopping_the_chain_leaves_no_process(tmp_path):
    chain = CpuDisaggChain(tmp_path, "trtllm").start()
    try:
        leaders = [worker.process.popen.pid for worker in chain.workers]
        leaders.append(chain._orchestrator.popen.pid)
        processes = []
        for pid in leaders:
            leader = psutil.Process(pid)
            processes += [leader, *leader.children(recursive=True)]
    finally:
        chain.stop()
    assert [process.pid for process in processes if not _gone(process)] == []
    with pytest.raises(RuntimeError, match="only once"):
        chain.start()

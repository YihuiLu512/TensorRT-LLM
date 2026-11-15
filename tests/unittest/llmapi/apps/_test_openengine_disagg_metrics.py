# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``/metrics`` and ``/prometheus/metrics`` of one worker of a disaggregated deployment (F-42a, F-42b).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU, both with
``return_perf_metrics`` and ``enable_iter_perf_stats``), in-process or on the
OpenEngine chain. Each test sends one aggregated completion straight to the
generation worker (``leg_expect: direct_worker`` on the chain) and reads that
worker's endpoint.

- NT-071 (F-42a, DV-18): ``/metrics`` answers 200 after the request.
  In-process it gives the engine's iteration stats; on the chain the worker
  frontend gives a snapshot of the engine's ``GetLoad``, with only the load
  fields and no per-iteration field made up.
- NT-072 (F-42b): ``/prometheus/metrics`` can be read, and its request metrics
  grow with the request.
"""

import time

import pytest
import requests

from ._test_openengine_disagg_common import CHAIN, REQUEST_TIMEOUT_S, disagg_deployment

pytestmark = pytest.mark.threadleak(enabled=False)

WAIT_S = 30
# What a /metrics entry of the chain's worker frontend may hold: the GetLoad snapshot.
SNAPSHOT_KEYS = {"source", "timestamp", "numActiveRequests", "numQueuedRequests", "kvCacheStats"}
SNAPSHOT_KV_KEYS = {"usedNumBlocks", "maxNumBlocks", "freeNumBlocks"}
GETLOAD_SOURCE = "openengine_getload"
REQUEST_METRICS = (
    "trtllm_request_success_total",
    "trtllm_e2e_request_latency_seconds_count",
    "trtllm_time_to_first_token_seconds_count",
    "trtllm_time_per_output_token_seconds_count",
    "trtllm_prompt_tokens_total",
    "trtllm_generation_tokens_total",
)


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    with disagg_deployment(
        tmp_path_factory.mktemp("disagg_metrics"),
        worker_config={"return_perf_metrics": True, "enable_iter_perf_stats": True},
    ) as deployment:
        yield deployment


def _direct_completion(deployment) -> None:
    """One aggregated completion straight to the generation worker."""
    body = {"model": deployment.model, "prompt": "Hello, my name is", "max_tokens": 4}
    response = requests.post(
        deployment.gen.url_for("v1", "completions"), json=body, timeout=REQUEST_TIMEOUT_S
    )
    assert response.status_code == 200, response.text


def _iteration_stats(deployment) -> list[dict]:
    response = requests.get(deployment.gen.url_for("metrics"), timeout=REQUEST_TIMEOUT_S)
    assert response.status_code == 200, response.text
    return response.json()


def _prometheus_counts(deployment) -> dict[str, float]:
    response = requests.get(
        deployment.gen.url_for("prometheus", "metrics"), timeout=REQUEST_TIMEOUT_S
    )
    assert response.status_code == 200, response.text
    counts: dict[str, float] = {}
    for line in response.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name.endswith(("_count", "_total")):
            counts[name] = counts.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return counts


def test_metrics_snapshot_direct(deployment):
    # Take what earlier reads left, so the entries below come after the request.
    _iteration_stats(deployment)
    _direct_completion(deployment)
    deadline = time.monotonic() + WAIT_S
    stats = _iteration_stats(deployment)
    while not stats:
        assert time.monotonic() < deadline, "no /metrics entry"
        time.sleep(0.5)
        stats = _iteration_stats(deployment)
    for stat in stats:
        assert "numActiveRequests" in stat, stat
        if CHAIN:
            assert stat["source"] == GETLOAD_SOURCE, stat
            assert set(stat) <= SNAPSHOT_KEYS, stat
            assert set(stat.get("kvCacheStats", {})) <= SNAPSHOT_KV_KEYS, stat
        else:
            assert "iter" in stat, stat


def test_prometheus_direct(deployment):
    before = _prometheus_counts(deployment)
    _direct_completion(deployment)
    deadline = time.monotonic() + WAIT_S
    after = _prometheus_counts(deployment)
    name = "trtllm_e2e_request_latency_seconds_count"
    while after.get(name, 0) <= before.get(name, 0):
        assert time.monotonic() < deadline, (before, after)
        time.sleep(0.5)
        after = _prometheus_counts(deployment)
    for name in REQUEST_METRICS:
        assert after.get(name, 0) > before.get(name, 0), (name, before, after)

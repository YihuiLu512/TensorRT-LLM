# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request metrics, ``/metrics``, KV-cache events and tracing, end to end.

Every test runs on the in-process server as well as on the OpenEngine chain
(``TRTLLM_TEST_OPENENGINE_CHAIN``, ``tests/integration/defs/openengine_chain``).
On the chain the frontend measures a request from the events it receives, so
what only the engine measures (time in queue, first scheduled time, iteration
counts, KV-cache reuse per request) is expected to be absent there rather
than reported as 0.
"""

import os
import socket
import tempfile
import threading
import time
from concurrent import futures

import pytest
import requests
from test_common.perf_metrics_utils import wait_for_perf_metrics_jsonl

from tensorrt_llm.serve import perf_metrics

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
PARENT_SPAN_ID = "00f067aa0ba902b7"
TRACEPARENT = f"00-{TRACE_ID}-{PARENT_SPAN_ID}-01"
WAIT_S = 30


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _TraceCollector:
    """An OTLP trace collector on gRPC that keeps every exported span."""

    def __init__(self) -> None:
        import grpc
        from opentelemetry.proto.collector.trace.v1 import trace_service_pb2, trace_service_pb2_grpc

        collector = self

        class Service(trace_service_pb2_grpc.TraceServiceServicer):
            def Export(self, request, context):
                with collector.lock:
                    for resource_spans in request.resource_spans:
                        for scope_spans in resource_spans.scope_spans:
                            collector.spans.extend(scope_spans.spans)
                return trace_service_pb2.ExportTraceServiceResponse()

        self.spans = []
        self.lock = threading.Lock()
        self._executor = futures.ThreadPoolExecutor(max_workers=1)
        self._server = grpc.server(self._executor)
        trace_service_pb2_grpc.add_TraceServiceServicer_to_server(Service(), self._server)
        self.address = f"localhost:{self._server.add_insecure_port('localhost:0')}"
        self._server.start()

    def wait_for(self, trace_id: str):
        deadline = time.monotonic() + WAIT_S
        while time.monotonic() < deadline:
            with self.lock:
                found = [s for s in self.spans if s.trace_id.hex() == trace_id]
            if found:
                return found
            time.sleep(0.2)
        raise AssertionError(f"no span of trace {trace_id} within {WAIT_S} s")

    def stop(self) -> None:
        self._server.stop(None)
        self._executor.shutdown(wait=True)


@pytest.fixture(scope="module")
def model_name() -> str:
    return "Qwen3/Qwen3-0.6B"


@pytest.fixture(scope="module")
def trace_collector():
    pytest.importorskip("opentelemetry.proto.collector.trace.v1")
    collector = _TraceCollector()
    yield collector
    collector.stop()


@pytest.fixture(scope="module")
def perf_dir():
    with tempfile.TemporaryDirectory(prefix="wt15-perf-") as directory:
        yield directory


@pytest.fixture(scope="module")
def server(model_name: str, trace_collector: _TraceCollector, perf_dir: str):
    kv_cache_config = {
        "free_gpu_memory_fraction": 0.2,
        "enable_block_reuse": True,
        "use_kv_cache_manager_v2": True,
    }
    if CHAIN:
        # The chain's engine publishes the events to the frontend over ZeroMQ.
        kv_cache_config["kv_events_config"] = {
            "enable_kv_cache_events": True,
            "endpoint": f"tcp://127.0.0.1:{_free_port()}",
        }
    else:
        # In-process, /kv_cache_events reads the event buffer, which publishing
        # would replace.
        kv_cache_config["event_buffer_max_size"] = 1024
    extra_config = {
        "return_perf_metrics": True,
        "perf_metrics_output_dir": perf_dir,
        "enable_iter_perf_stats": True,
        "kv_cache_config": kv_cache_config,
    }
    env = os.environ.copy()
    env["OTEL_EXPORTER_OTLP_TRACES_INSECURE"] = "true"
    args = ["--backend", "pytorch", "--otlp_traces_endpoint", trace_collector.address]
    with RemoteOpenAIServer(
        get_model_path(model_name), args, extra_config=extra_config, env=env
    ) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


def _complete(server: RemoteOpenAIServer, model_name: str, headers=None, **fields):
    body = {"model": model_name, "prompt": "Hello, my name is", "max_tokens": 4, **fields}
    response = requests.post(
        server.url_for("v1", "completions"), json=body, headers=headers or {}, timeout=120
    )
    assert response.status_code == 200, response.text
    return response


def _prometheus_counts(server: RemoteOpenAIServer) -> dict[str, float]:
    response = requests.get(server.url_for("prometheus", "metrics"), timeout=30)
    assert response.status_code == 200, response.text
    counts: dict[str, float] = {}
    for line in response.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name.endswith(("_count", "_total")):
            counts[name] = counts.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return counts


def test_perf_metrics_over_chain(server: RemoteOpenAIServer, model_name: str, perf_dir: str):
    response = _complete(server, model_name, headers={perf_metrics.RETURN_METRICS_HEADER: "1"})
    timing = response.headers[perf_metrics.SERVER_TIMING_HEADER]
    assert "server_ttft;dur=" in timing
    assert "server_e2e;dur=" in timing
    assert response.headers.get(perf_metrics.START_END_TIME_HEADER)
    records = wait_for_perf_metrics_jsonl(perf_dir, expected_count=1)
    metrics = records[-1]["perf_metrics"]
    times = metrics["timing_metrics"]
    assert times["arrival_time"] < times["first_token_time"] <= times["last_token_time"]
    if CHAIN:
        # Measured inside the engine: not reported by the frontend, rather than 0.
        assert "server_queue" not in timing
        assert times.get("first_scheduled_time") is None
        assert metrics.get("first_iter") is None
        assert not response.headers.get(perf_metrics.STEP_METRICS_HEADER)
    else:
        assert "server_queue;dur=" in timing
        assert times["arrival_time"] < times["first_scheduled_time"] < times["first_token_time"]


def test_prometheus_over_chain(server: RemoteOpenAIServer, model_name: str):
    _complete(server, model_name)
    deadline = time.monotonic() + WAIT_S
    counts = _prometheus_counts(server)
    while counts.get("trtllm_e2e_request_latency_seconds_count", 0) == 0:
        assert time.monotonic() < deadline, counts
        time.sleep(0.5)
        counts = _prometheus_counts(server)
    for name in (
        "trtllm_request_success_total",
        "trtllm_e2e_request_latency_seconds_count",
        "trtllm_time_to_first_token_seconds_count",
        "trtllm_time_per_output_token_seconds_count",
        "trtllm_prompt_tokens_total",
        "trtllm_generation_tokens_total",
    ):
        assert counts.get(name, 0) > 0, name
    queued = counts.get("trtllm_request_queue_time_seconds_count", 0)
    if CHAIN:
        assert queued == 0
    else:
        assert queued > 0


def test_iteration_stats_over_chain(server: RemoteOpenAIServer, model_name: str):
    # Take what earlier reads left, so the entries below come after the request.
    assert requests.get(server.url_for("metrics"), timeout=30).status_code == 200
    _complete(server, model_name)
    deadline = time.monotonic() + WAIT_S
    stats = []
    while not stats:
        assert time.monotonic() < deadline, "no /metrics entry"
        response = requests.get(server.url_for("metrics"), timeout=30)
        assert response.status_code == 200, response.text
        stats = response.json()
        if not stats:
            time.sleep(0.5)
    for stat in stats:
        assert "numActiveRequests" in stat
        if CHAIN:
            assert stat["source"] == "openengine_getload"
        else:
            assert "iter" in stat


def test_tracing_over_chain(
    server: RemoteOpenAIServer, model_name: str, trace_collector: _TraceCollector
):
    response = _complete(
        server,
        model_name,
        headers={"traceparent": TRACEPARENT},
        temperature=0.9,
        top_p=0.9,
    )
    usage = response.json()["usage"]
    (span,) = [s for s in trace_collector.wait_for(TRACE_ID) if s.name == "llm_request"]
    assert span.parent_span_id.hex() == PARENT_SPAN_ID
    attributes = {kv.key: kv.value for kv in span.attributes}
    assert attributes["gen_ai.request.temperature"].double_value == pytest.approx(0.9)
    assert attributes["gen_ai.request.top_p"].double_value == pytest.approx(0.9)
    assert attributes["gen_ai.request.max_tokens"].int_value == 4
    assert attributes["gen_ai.usage.prompt_tokens"].int_value == usage["prompt_tokens"]
    assert attributes["gen_ai.usage.completion_tokens"].int_value == usage["completion_tokens"]
    assert attributes["gen_ai.latency.time_to_first_token"].double_value > 0
    assert attributes["gen_ai.latency.e2e"].double_value > 0
    if CHAIN:
        assert "gen_ai.latency.time_in_queue" not in attributes
    else:
        assert attributes["gen_ai.latency.time_in_queue"].double_value > 0


def _kv_events(server: RemoteOpenAIServer) -> list:
    response = requests.post(server.url_for("kv_cache_events"), timeout=30)
    assert response.status_code == 200, response.text
    return response.json()


def test_kv_cache_events_over_chain(server: RemoteOpenAIServer, client, model_name: str):
    # Drains what earlier requests left; on the chain, subscribes to the engine's publisher.
    _kv_events(server)
    time.sleep(1)
    prompt = list(range(1000, 1130))
    client.completions.create(model=model_name, prompt=prompt, max_tokens=4, temperature=0.0)
    stored: list[tuple[int, ...]] = []
    event_ids: list[int] = []
    block_size = None
    deadline = time.monotonic() + WAIT_S
    while True:
        for event in _kv_events(server):
            event_ids.append(event["event_id"])
            if event["data"]["type"] != "stored":
                continue
            for block in event["data"]["blocks"]:
                tokens = tuple(token["token_id"] for token in block["tokens"])
                stored.append(tokens)
                block_size = block_size or len(tokens)
        if block_size and all(
            tuple(prompt[i : i + block_size]) in stored
            for i in range(0, len(prompt) - block_size + 1, block_size)
        ):
            break
        assert time.monotonic() < deadline, f"prompt blocks not stored: {stored}"
        time.sleep(0.5)
    assert event_ids == sorted(event_ids) and len(set(event_ids)) == len(event_ids)

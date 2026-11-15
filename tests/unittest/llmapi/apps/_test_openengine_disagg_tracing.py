# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OTLP tracing of a disaggregated request (F-45).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU, with ``return_perf_metrics``),
in-process or on the OpenEngine chain. The orchestrator and each worker export
their spans to a trace collector of their own, so every span is known by the
process that made it. On the chain a worker's spans come from its worker
frontend.

- NT-077 (F-45): a completion sent to the orchestrator with a ``traceparent``
  gives one ``disaggregated_request`` span of the orchestrator, in that trace
  and under that parent, and ``llm_request`` spans of the context and the
  generation worker that carry the request's sampling options and prompt
  length. The orchestrator and both workers each make spans, so the count of
  all spans of the request is not 1 as for an aggregated server; how many each
  process makes is not asserted. The orchestrator does not send the trace
  context on to the workers, in-process as on the chain, so the workers'
  spans are not asserted to be in the request's trace.
"""

import threading
import time
from concurrent import futures

import pytest
import requests

from ._test_openengine_disagg_common import REQUEST_TIMEOUT_S, disagg_deployment

pytestmark = pytest.mark.threadleak(enabled=False)

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
PARENT_SPAN_ID = "00f067aa0ba902b7"
TRACEPARENT = f"00-{TRACE_ID}-{PARENT_SPAN_ID}-01"
WAIT_S = 30
TEMPERATURE = 0.9
TOP_P = 0.9
MAX_TOKENS = 8


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

    def wait_for(self, name: str) -> list:
        """The spans named ``name``, once there is one."""
        deadline = time.monotonic() + WAIT_S
        while time.monotonic() < deadline:
            with self.lock:
                found = [span for span in self.spans if span.name == name]
            if found:
                return found
            time.sleep(0.2)
        raise AssertionError(f"no {name} span at {self.address} within {WAIT_S} s")

    def stop(self) -> None:
        self._server.stop(None)
        self._executor.shutdown(wait=True)


@pytest.fixture(scope="module")
def collectors():
    started = {}
    try:
        for where in ("orchestrator", "context", "generation"):
            started[where] = _TraceCollector()
        yield started
    finally:
        for collector in started.values():
            collector.stop()


@pytest.fixture(scope="module")
def deployment(tmp_path_factory, collectors):
    with disagg_deployment(
        tmp_path_factory.mktemp("disagg_tracing"),
        role_args={
            role: ["--otlp_traces_endpoint", collectors[role].address]
            for role in ("context", "generation")
        },
        worker_config={"return_perf_metrics": True},
        orchestrator_config={
            "otlp_config": {"otlp_traces_endpoint": collectors["orchestrator"].address}
        },
        env={"OTEL_EXPORTER_OTLP_TRACES_INSECURE": "true"},
    ) as deployment:
        yield deployment


def _attributes(span) -> dict:
    return {kv.key: kv.value for kv in span.attributes}


def test_tracing_spans(deployment, collectors):
    body = {
        "model": deployment.model,
        "prompt": "Hello, my name is",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    response = requests.post(
        f"{deployment.url}/v1/completions",
        json=body,
        headers={"traceparent": TRACEPARENT},
        timeout=REQUEST_TIMEOUT_S,
    )
    assert response.status_code == 200, response.text
    usage = response.json()["usage"]

    orchestrator = [
        span
        for span in collectors["orchestrator"].wait_for("disaggregated_request")
        if span.trace_id.hex() == TRACE_ID
    ]
    assert len(orchestrator) == 1, orchestrator
    assert orchestrator[0].parent_span_id.hex() == PARENT_SPAN_ID

    for role in ("context", "generation"):
        spans = collectors[role].wait_for("llm_request")
        for span in spans:
            attributes = _attributes(span)
            assert attributes["gen_ai.request.temperature"].double_value == pytest.approx(
                TEMPERATURE
            ), (role, attributes)
            assert attributes["gen_ai.request.top_p"].double_value == pytest.approx(TOP_P), (
                role,
                attributes,
            )
            assert attributes["gen_ai.usage.prompt_tokens"].int_value == usage["prompt_tokens"], (
                role,
                attributes,
            )
            assert attributes["gen_ai.latency.e2e"].double_value > 0, (role, attributes)

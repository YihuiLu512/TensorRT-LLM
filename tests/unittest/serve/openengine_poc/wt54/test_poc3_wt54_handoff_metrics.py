# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The request metrics of a context leg that hands off its KV cache (SV-DSG-12).

The context leg's tokens before ``PrefillReady`` are dropped from the response,
so the metrics leave them out too, with the postprocessing pool off (the
assembled output has no token) and on (the events tell): no generated token,
first and last token time when the handoff arrived, finish reason
``length`` (the pool's outputs carry none, or ``cancelled`` after an abort). The handoff's usage is the
engine's count of the prompt. A context leg that finishes on the context
engine keeps its tokens.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import pytest

from tensorrt_llm.metrics.collector import MetricsCollector
from tensorrt_llm.metrics.enums import MetricNames
from tensorrt_llm.serve.openengine_poc.engine_client import (
    KvSessionSpec,
    OutputFinished,
    PrefillReady,
    TokenDelta,
    Usage,
)
from tensorrt_llm.serve.openengine_poc.metrics import OpenEngineRequestMetrics

pytestmark = pytest.mark.cpu_only

ARRIVAL = 1000.0
TOKEN_AT = 1000.2
HANDOFF_AT = 1000.5
FINISH_AT = 1000.6
OFFSET = 50.0


def _metrics() -> OpenEngineRequestMetrics:
    return OpenEngineRequestMetrics("r-1", ARRIVAL, steady_offset=OFFSET)


def _output(token_ids=(), finish_reason="length") -> SimpleNamespace:
    return SimpleNamespace(index=0, token_ids=list(token_ids), finish_reason=finish_reason)


def _handed_off(assembled: bool) -> tuple[OpenEngineRequestMetrics, SimpleNamespace]:
    metrics = _metrics()
    metrics.on_event(TokenDelta(output_index=0, token_ids=(7,)), TOKEN_AT)
    metrics.on_event(
        PrefillReady(
            output_index=0,
            kv_session=KvSessionSpec(session_id="5"),
            usage=Usage(prompt_tokens=11, completion_tokens=1, cached_prompt_tokens=3),
        ),
        HANDOFF_AT,
    )
    # The response ends the assembled output with ``length``; the pool's outputs carry no reason.
    output = _output(finish_reason="length" if assembled else None)
    metrics.finish([output], FINISH_AT, prompt_tokens=2, assembled=assembled)
    return metrics, output


@pytest.mark.parametrize("assembled", [True, False], ids=["pool-off", "pool-on"])
def test_a_handed_off_context_leg_counts_no_token_and_ends_at_the_handoff(assembled):
    metrics, output = _handed_off(assembled)

    timing = output.request_perf_metrics.timing_metrics
    handoff = datetime.timedelta(seconds=HANDOFF_AT + OFFSET)
    assert timing.first_token_time == handoff
    assert timing.last_token_time == handoff
    assert timing.arrival_time == datetime.timedelta(seconds=ARRIVAL + OFFSET)
    stats = metrics.metrics_dict()
    assert stats[MetricNames.TTFT] == pytest.approx(HANDOFF_AT - ARRIVAL)
    assert MetricNames.TPOT not in stats
    assert MetricNames.GENERATION_TOKENS not in stats


def test_pool_off_and_on_record_the_same_metrics():
    off, _ = _handed_off(assembled=True)
    on, _ = _handed_off(assembled=False)

    assert off.metrics_dict() == on.metrics_dict()
    assert off.candidate_metrics() == on.candidate_metrics()
    (stats,) = on.candidate_metrics()
    assert stats[MetricsCollector.labelname_finish_reason] == "length"


def test_a_handed_off_context_leg_cancelled_after_the_handoff_still_ends_with_length():
    # With the pool, an abort after the handoff marks the output cancelled; without it the
    # output already ended with length.
    metrics = _metrics()
    metrics.on_event(
        PrefillReady(output_index=0, kv_session=KvSessionSpec(session_id="5")), HANDOFF_AT
    )
    metrics.finish(
        [_output(finish_reason="cancelled")], FINISH_AT, prompt_tokens=2, assembled=False
    )

    (stats,) = metrics.candidate_metrics()
    assert stats[MetricsCollector.labelname_finish_reason] == "length"


def test_the_handoffs_usage_counts_the_prompt():
    metrics, _ = _handed_off(assembled=True)

    stats = metrics.metrics_dict()
    assert stats[MetricNames.PROMPT_TOKENS] == 11
    assert stats[MetricNames.PROMPT_CACHE_CACHED_TOKENS] == 3


def test_a_context_leg_that_finishes_on_the_context_engine_keeps_its_tokens():
    metrics = _metrics()
    metrics.on_event(TokenDelta(output_index=0, token_ids=(7,)), TOKEN_AT)
    metrics.on_event(OutputFinished(output_index=0, reason="stop", usage=Usage(2, 1)), FINISH_AT)
    output = _output([7])
    metrics.finish([output], FINISH_AT, prompt_tokens=2, assembled=True)

    timing = output.request_perf_metrics.timing_metrics
    assert timing.first_token_time == datetime.timedelta(seconds=TOKEN_AT + OFFSET)


def test_a_request_without_tokens_or_handoff_has_no_token_times():
    metrics = _metrics()
    output = _output()
    metrics.finish([output], FINISH_AT, prompt_tokens=2, assembled=True)

    timing = output.request_perf_metrics.timing_metrics
    assert timing.first_token_time is None
    assert timing.last_token_time is None

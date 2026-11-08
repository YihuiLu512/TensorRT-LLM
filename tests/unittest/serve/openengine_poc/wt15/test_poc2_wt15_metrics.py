# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenEngine metrics recorder on CPU: request metrics, ``/metrics``, KV-cache events, tracing."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import math
import socket
import time
from types import SimpleNamespace
from typing import Any

import httpx
import msgspec
import pytest
import zmq
from prometheus_client import CollectorRegistry
from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback, Fuse
from tokenizers.decoders import Sequence as DecoderSequence
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from tensorrt_llm.executor.result import CompletionOutput, GenerationResultBase
from tensorrt_llm.grpc.openengine.bindings import generation_pb2, kv_pb2
from tensorrt_llm.grpc.openengine_client import OpenEngineClientError
from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.llmapi import tracing
from tensorrt_llm.metrics.collector import MetricsCollector
from tensorrt_llm.metrics.enums import MetricNames, RequestEventTiming
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve import openai_server as openai_server_module
from tensorrt_llm.serve import perf_metrics
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.openengine_poc import OpenEngineLLM, contracts
from tensorrt_llm.serve.openengine_poc import metrics as metrics_module
from tensorrt_llm.serve.openengine_poc.engine_client import (
    EngineLoad,
    KvEventSource,
    OutputFinished,
    PromptInfo,
    TokenDelta,
    TokenLogprob,
    Usage,
)
from tensorrt_llm.serve.openengine_poc.facade import adapt_server, check_frontend_options
from tensorrt_llm.tokenizer import TransformersTokenizer

__extra_import_path__ = ["~/tests/unittest/grpc/openengine_client"]
from fake_openengine import PROFILES, FakeEngine, Hang  # noqa: E402
from openengine_client_helpers import WAIT_SECONDS, token  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

MODEL = "stub-model"
MAX_CONTEXT = 64
_VOCAB = {"<unk>": 0, "</s>": 1, "He": 2, "llo": 3, " world": 4, "!": 5, " again": 6}
UNK, EOS, HE, LLO, WORLD, BANG, AGAIN = range(7)
PROMPT_IDS = [2, 3, 5]
HASH_ALGO = "v2_sha256_64"
# The engine-internal phases a serving-side measurement leaves out.
ENGINE_PHASES = {
    MetricNames.REQUEST_QUEUE_TIME,
    MetricNames.PREFILL_TIME,
    MetricNames.INFERENCE_TIME,
}


@pytest.fixture(autouse=True)
def _fresh_slots():
    contracts.reset_slot_cache()
    yield
    contracts.reset_slot_cache()


def _args(**overrides: Any) -> SimpleNamespace:
    args = SimpleNamespace(
        backend="pytorch",
        reasoning_parser=None,
        gather_generation_logits=False,
        guided_decoding_backend=None,
        num_postprocess_workers=0,
        cache_transceiver_config=None,
        trust_remote_code=False,
        checkpoint_format=None,
        return_perf_metrics=False,
        perf_metrics_output_dir=None,
        speculative_config=None,
        post_processor_hook=None,
        sleep_config=None,
        num_serve_frontends=1,
        enable_iter_perf_stats=False,
        max_beam_width=1,
        custom_tokenizer=None,
        max_batch_size=None,
        kv_cache_config=None,
        otlp_traces_endpoint=None,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


# --- Request metrics from a synthetic event stream ----------------------------------


def _recorder(steady_at_start: float = 1000.0, **args: Any):
    return metrics_module.OpenEngineMetricsRecorder(
        _args(**args), steady_clock=lambda: steady_at_start
    )


def _stream(request) -> None:
    """Arrival at 10.0; tokens at 10.5 (two), 10.7 and 11.0; finished at 11.0."""
    request.on_event(TokenDelta(0, (HE, LLO)), 10.5)
    request.on_event(TokenDelta(0, (WORLD,)), 10.7)
    request.on_event(TokenDelta(0, (BANG,)), 11.0)
    request.on_event(OutputFinished(0, "stop", eos_token_id=EOS, usage=Usage(7, 4, 3)), 11.0)


def _output(index: int = 0, length: int = 4, finish_reason: str = "stop") -> CompletionOutput:
    return CompletionOutput(index=index, token_ids=[HE] * length, finish_reason=finish_reason)


def _finish(request, outputs, now: float, *, prompt_tokens: int = 7, assembled=True) -> None:
    request.finish(outputs, now, prompt_tokens=prompt_tokens, assembled=assembled)


def test_request_metrics_follow_the_event_times():
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    _stream(request)
    assert request.metrics_dict() == {}
    output = _output()
    _finish(request, [output], 11.1)
    got = request.metrics_dict()
    assert got[MetricsCollector.labelname_finish_reason] == "stop"
    assert got[MetricNames.TTFT] == pytest.approx(0.5)
    assert got[MetricNames.E2E] == pytest.approx(1.0)
    assert got[MetricNames.DECODE_TIME] == pytest.approx(0.5)
    assert got[MetricNames.TPOT] == pytest.approx(0.5 / 3)
    assert got[MetricNames.GENERATION_TOKENS] == 4
    assert got[MetricNames.PROMPT_TOKENS] == 7
    assert got[MetricNames.PROMPT_CACHE_CACHED_TOKENS] == 3
    # The engine's phases are not measured here, and not reported as 0.
    assert not ENGINE_PHASES & got.keys()


def _in_process_metrics(arrival, first_scheduled, first_token, last_token, output, cached):
    result = GenerationResultBase(id=1, sampling_params=SamplingParams())
    result.prompt_token_ids = list(range(7))
    result.cached_tokens = cached
    result.record_stats(
        output,
        {
            RequestEventTiming.ARRIVAL_TIME: arrival,
            RequestEventTiming.FIRST_SCHEDULED_TIME: first_scheduled,
            RequestEventTiming.FIRST_TOKEN_TIME: first_token,
            RequestEventTiming.LAST_TOKEN_TIME: last_token,
        },
    )
    return result.metrics_dict


def test_metrics_dict_matches_the_in_process_one_but_the_engine_phases():
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    _stream(request)
    _finish(request, [_output()], 11.1)
    in_process = _in_process_metrics(10.0, 10.2, 10.5, 11.0, _output(), cached=3)
    assert ENGINE_PHASES <= in_process.keys()
    expected = {k: v for k, v in in_process.items() if k not in ENGINE_PHASES}
    got = request.metrics_dict()
    assert got.keys() == expected.keys()
    for key, value in expected.items():
        assert got[key] == pytest.approx(value), key


def _observed(registry: CollectorRegistry) -> set[str]:
    """Names of the series that got an observation or an increment."""
    names = set()
    for family in registry.collect():
        for sample in family.samples:
            if sample.name.endswith(("_count", "_total")) and sample.value > 0:
                names.add(sample.name)
    return names


def test_prometheus_series_are_the_in_process_ones_but_the_engine_phases():
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    _stream(request)
    _finish(request, [_output()], 11.1)
    in_process = _in_process_metrics(10.0, 10.2, 10.5, 11.0, _output(), cached=3)
    chain_registry, in_process_registry = CollectorRegistry(), CollectorRegistry()
    labels = {"model_name": MODEL, "engine_type": "pytorch"}
    MetricsCollector(labels, registry=chain_registry).log_request_metrics_dict(
        request.metrics_dict()
    )
    MetricsCollector(labels, registry=in_process_registry).log_request_metrics_dict(in_process)
    engine_phase_series = {
        "trtllm_request_queue_time_seconds_count",
        "trtllm_request_prefill_time_seconds_count",
        "trtllm_request_inference_time_seconds_count",
    }
    in_process_names = _observed(in_process_registry)
    assert engine_phase_series <= in_process_names
    assert _observed(chain_registry) == in_process_names - engine_phase_series


def test_nothing_is_recorded_with_perf_metrics_off():
    request = _recorder().start_request("r", 10.0)
    _stream(request)
    output = _output()
    _finish(request, [output], 11.1)
    assert request.metrics_dict() == {}
    assert request.candidate_metrics() == []
    assert output.request_perf_metrics is None


def test_a_request_can_turn_perf_metrics_on():
    request = _recorder().start_request(
        "r", 10.0, sampling_params=SamplingParams(return_perf_metrics=True)
    )
    _stream(request)
    _finish(request, [_output()], 11.1)
    assert request.metrics_dict()[MetricNames.E2E] == pytest.approx(1.0)


def test_the_server_turning_perf_metrics_on_after_the_recorder_is_built_counts():
    args = _args()
    recorder = metrics_module.OpenEngineMetricsRecorder(args, steady_clock=lambda: 0.0)
    # OpenAIServer does this after the frontend is built, when it writes perf records.
    args.return_perf_metrics = True
    request = recorder.start_request("r", 10.0)
    _stream(request)
    _finish(request, [_output()], 11.1)
    assert request.metrics_dict()[MetricNames.TTFT] == pytest.approx(0.5)


def test_candidates_are_recorded_in_the_order_they_finish():
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    request.on_event(TokenDelta(0, (HE,)), 10.1)
    request.on_event(TokenDelta(1, (HE,)), 10.3)
    request.on_event(TokenDelta(1, (LLO,)), 10.4)
    request.on_event(OutputFinished(1, "length"), 10.4)
    request.on_event(TokenDelta(0, (LLO, WORLD)), 10.9)
    request.on_event(OutputFinished(0, "stop", usage=Usage(7, 5, 2)), 10.9)
    outputs = [_output(0, 3, "stop"), _output(1, 2, "length")]
    _finish(request, outputs, 11.0)
    got = request.metrics_dict()
    # As in-process, the last candidate to finish leaves its own entries ...
    assert got[MetricsCollector.labelname_finish_reason] == "stop"
    assert got[MetricNames.TTFT] == pytest.approx(0.1)
    assert got[MetricNames.E2E] == pytest.approx(0.9)
    assert got[MetricNames.GENERATION_TOKENS] == 3
    # ... and output 0 the prompt's.
    assert got[MetricNames.PROMPT_TOKENS] == 7
    assert got[MetricNames.PROMPT_CACHE_CACHED_TOKENS] == 2
    first = outputs[1].request_perf_metrics.timing_metrics
    assert first.last_token_time == datetime.timedelta(seconds=10.4 + 990.0)
    # One candidate entry per output, in the order they finished.
    candidates = request.candidate_metrics()
    assert [c[MetricsCollector.labelname_finish_reason] for c in candidates] == ["length", "stop"]
    assert MetricNames.PROMPT_TOKENS not in candidates[0]
    assert MetricNames.PROMPT_TOKENS in candidates[1]


def test_every_candidate_is_counted_as_in_process():
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    request.on_event(TokenDelta(0, (HE, LLO)), 10.3)
    request.on_event(OutputFinished(0, "length"), 10.3)
    request.on_event(TokenDelta(1, (HE, LLO, WORLD)), 10.6)
    request.on_event(OutputFinished(1, "stop", usage=Usage(7, 5, 3)), 10.6)
    outputs = [_output(0, 2, "length"), _output(1, 3, "stop")]
    _finish(request, outputs, 10.7)
    in_process = GenerationResultBase(id=1, sampling_params=SamplingParams(n=2, temperature=1.0))
    in_process.prompt_token_ids = list(range(7))
    in_process.cached_tokens = 3
    for index, last in ((0, 10.3), (1, 10.6)):
        in_process.record_stats(
            outputs[index],
            {
                RequestEventTiming.ARRIVAL_TIME: 10.0,
                RequestEventTiming.FIRST_TOKEN_TIME: last,
                RequestEventTiming.LAST_TOKEN_TIME: last,
            },
            sequence_index=index,
        )
    totals = []
    labels = {"model_name": MODEL, "engine_type": "pytorch"}
    for candidates in (request.candidate_metrics(), in_process.candidate_metrics):
        registry = CollectorRegistry()
        collector = MetricsCollector(labels, registry=registry)
        for candidate in candidates:
            collector.log_request_metrics_dict(candidate)
        total: dict[str, float] = {}
        for family in registry.collect():
            for sample in family.samples:
                if sample.name.endswith("_total"):
                    # Summed over the labels, such as the finish reason.
                    total[sample.name] = total.get(sample.name, 0) + sample.value
        totals.append(total)
    chain, expected = totals
    assert chain["trtllm_request_success_total"] == 2
    assert chain["trtllm_generation_tokens_total"] == 5
    assert chain == expected


def test_without_usage_the_prompt_count_is_the_facade_one_and_cached_tokens_are_unknown():
    # A serving-side stop ends the request before the engine's usage arrives.
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    request.on_event(TokenDelta(0, (HE, LLO)), 10.5)
    _finish(request, [_output(0, 2)], 10.6, prompt_tokens=3)
    got = request.metrics_dict()
    assert got[MetricNames.PROMPT_TOKENS] == 3
    assert MetricNames.PROMPT_CACHE_CACHED_TOKENS not in got


def test_the_engine_prompt_count_wins_when_its_usage_arrived():
    # With media the facade's prompt holds the unexpanded placeholders.
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    request.on_event(TokenDelta(0, (HE,)), 10.5)
    request.on_event(OutputFinished(0, "stop", usage=Usage(260, 1, 0)), 10.5)
    _finish(request, [_output(0, 1)], 10.6, prompt_tokens=5)
    assert request.metrics_dict()[MetricNames.PROMPT_TOKENS] == 260


def test_cached_tokens_come_from_the_first_usage():
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    request.on_event(TokenDelta(0, (HE,)), 10.5)
    request.on_event(OutputFinished(0, "stop", usage=Usage(7, 1, 1)), 10.5)
    request.on_event(TokenDelta(1, (HE,)), 10.6)
    request.on_event(OutputFinished(1, "stop", usage=Usage(7, 1, 4)), 10.6)
    _finish(request, [_output(0, 1), _output(1, 1)], 10.7)
    # As in the response's usage: a later call can find the prompt the first one cached.
    assert request.candidate_metrics()[0][MetricNames.PROMPT_CACHE_CACHED_TOKENS] == 1


def test_tokens_past_a_serving_side_stop_do_not_count():
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    for now, token_id, logprob in ((10.5, HE, -1.0), (10.6, LLO, -1.0), (10.7, WORLD, -1.0)):
        request.on_event(TokenDelta(0, (token_id,), (TokenLogprob(token_id, logprob),)), now)
    # The token completing the stop string, then one that arrived before the call was cancelled.
    request.on_event(TokenDelta(0, (BANG,), (TokenLogprob(BANG, -5.0),)), 11.0)
    request.on_event(TokenDelta(0, (AGAIN,), (TokenLogprob(AGAIN, -9.0),)), 12.0)
    # An output the response ends at the stop string's last token.
    output = CompletionOutput(index=0, token_ids=[HE, LLO, WORLD, BANG], finish_reason="stop")
    _finish(request, [output], 12.1)
    got = request.metrics_dict()
    assert got[MetricNames.GENERATION_TOKENS] == 4
    assert got[MetricNames.E2E] == pytest.approx(1.0)
    assert got[MetricNames.TPOT] == pytest.approx(0.5 / 3)
    assert got[MetricNames.GENERATION_PERPLEXITY] == pytest.approx(math.exp(2.0))
    timing = output.request_perf_metrics.timing_metrics
    assert timing.last_token_time == datetime.timedelta(seconds=1001.0)


def test_an_output_ended_in_the_pool_by_a_stop_string_is_counted():
    # With the postprocessing pool a serving-side stop ends the request before
    # the engine's finish event, and the facade does not assemble the outputs.
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    request.on_event(TokenDelta(0, (HE, LLO)), 10.5)
    _finish(request, [CompletionOutput(index=0)], 10.6, assembled=False)
    registry = CollectorRegistry()
    collector = MetricsCollector({"model_name": MODEL, "engine_type": "pytorch"}, registry=registry)
    for candidate in request.candidate_metrics():
        collector.log_request_metrics_dict(candidate)
    assert "trtllm_request_success_total" in _observed(registry)


def test_an_output_trimmed_to_no_tokens_counts_none():
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    _stream(request)
    _finish(request, [CompletionOutput(index=0, token_ids=[], finish_reason="stop")], 11.1)
    # As in-process, a count of 0 is left out.
    assert MetricNames.GENERATION_TOKENS not in request.metrics_dict()


def test_without_output_tokens_the_events_give_the_length():
    # With a postprocessing pool the facade does not assemble the outputs.
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    _stream(request)
    _finish(request, [CompletionOutput(index=0)], 11.1, assembled=False)
    got = request.metrics_dict()
    assert got[MetricNames.GENERATION_TOKENS] == 4
    assert got[MetricsCollector.labelname_finish_reason] == "stop"


def test_an_aborted_request_with_the_pool_counts_the_tokens_it_received():
    # The pool assembled the response; the facade's outputs end as cancelled, without tokens.
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    request.on_event(TokenDelta(0, (HE, LLO)), 10.5)
    request.on_event(TokenDelta(0, (WORLD,)), 11.0)
    _finish(request, [CompletionOutput(index=0, finish_reason="cancelled")], 11.1, assembled=False)
    got = request.metrics_dict()
    assert got[MetricNames.GENERATION_TOKENS] == 3
    assert got[MetricNames.E2E] == pytest.approx(1.0)
    assert got[MetricsCollector.labelname_finish_reason] == "cancelled"


def test_perplexities_come_from_the_logprobs_the_engine_sent():
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    # The first prompt token has no logprob: the engine sends a placeholder for it.
    prompt = (TokenLogprob(1, 0.0), TokenLogprob(2, -2.0), TokenLogprob(3, -4.0))
    request.on_event(PromptInfo(prompt), 10.1)
    request.on_event(TokenDelta(0, (HE,), (TokenLogprob(HE, -1.0),)), 10.5)
    request.on_event(TokenDelta(0, (LLO,), (TokenLogprob(LLO, -3.0),)), 10.6)
    request.on_event(OutputFinished(0, "length", usage=Usage(2, 2)), 10.6)
    _finish(request, [_output(0, 2, "length")], 10.7)
    got = request.metrics_dict()
    assert got[MetricNames.PREFILL_PERPLEXITY] == pytest.approx(2.718281828**3)
    assert got[MetricNames.GENERATION_PERPLEXITY] == pytest.approx(2.718281828**2)


def test_perf_record_holds_the_serving_side_times_only():
    request = _recorder(steady_at_start=1000.0, return_perf_metrics=True).start_request("r", 10.0)
    _stream(request)
    output = _output()
    _finish(request, [output], 11.1)
    timing = output.request_perf_metrics.timing_metrics
    # The steady clock read 1000.0 when the request arrived at wall time 10.0.
    assert timing.arrival_time == datetime.timedelta(seconds=1000.0)
    assert timing.first_token_time == datetime.timedelta(seconds=1000.5)
    assert timing.last_token_time == datetime.timedelta(seconds=1001.0)
    result = SimpleNamespace(outputs=[output], request_id="r")
    record = perf_metrics.build_request_metrics_record(result)
    phase = record["phases"]["server"]
    assert phase["timing_metrics"]["arrival_time"] == pytest.approx(1000.0)
    assert phase["timing_metrics"]["last_token_time"] == pytest.approx(1001.0)
    # Measured inside the engine: unknown, not 0.
    assert phase["timing_metrics"]["first_scheduled_time"] is None
    assert phase["first_iter"] is None and phase["last_iter"] is None
    assert set(phase["kv_cache_metrics"].values()) == {None}
    assert "speculative_decoding" not in phase
    headers = perf_metrics.build_metrics_headers([record])
    timing_header = headers[perf_metrics.SERVER_TIMING_HEADER]
    assert "server_ttft;dur=500.000000" in timing_header
    assert "server_e2e;dur=1000.000000" in timing_header
    assert "server_queue" not in timing_header


def test_finish_is_recorded_once():
    request = _recorder(return_perf_metrics=True).start_request("r", 10.0)
    _stream(request)
    _finish(request, [_output()], 11.1)
    first = dict(request.metrics_dict())
    request.on_event(TokenDelta(0, (AGAIN,)), 12.0)
    # What the engine sends after the end is not read, even when it is malformed.
    request.on_event(TokenDelta(0, (AGAIN,), logprobs=(None,)), 12.0)
    _finish(request, [_output(length=5)], 12.1)
    assert request.metrics_dict() == first


# --- /metrics from GetLoad -----------------------------------------------------------


def test_iteration_stats_take_what_the_load_reports():
    stats = metrics_module.iteration_stats_from_load(
        EngineLoad(instance_id="i", running_requests=3, used_kv_blocks=10, total_kv_blocks=40)
    )
    assert stats["source"] == metrics_module.GETLOAD_SOURCE
    assert stats["numActiveRequests"] == 3
    assert stats["kvCacheStats"] == {"usedNumBlocks": 10, "maxNumBlocks": 40, "freeNumBlocks": 30}
    # Not in the snapshot: left out rather than 0.
    assert "numQueuedRequests" not in stats
    bare = metrics_module.iteration_stats_from_load(EngineLoad(instance_id="i"))
    assert set(bare) == {"source", "timestamp"}
    assert _recorder(enable_iter_perf_stats=True).iteration_stats(None) is None
    # As in-process, no entries without enable_iter_perf_stats.
    assert _recorder().iteration_stats(EngineLoad(instance_id="i", running_requests=3)) is None


def test_an_idle_engine_whose_load_has_not_changed_gives_no_entry():
    recorder = _recorder(enable_iter_perf_stats=True)
    idle = EngineLoad(instance_id="i", running_requests=0, used_kv_blocks=2, total_kv_blocks=40)
    assert recorder.iteration_stats(idle) is not None
    # As in-process, an idle engine adds nothing, however often it is read.
    assert recorder.iteration_stats(idle) is None
    assert recorder.iteration_stats(idle) is None
    # A request started here since the last entry.
    recorder.start_request("r", 10.0)
    assert recorder.iteration_stats(idle) is not None
    assert recorder.iteration_stats(idle) is None
    # The load changed (another frontend's request, a restarted engine).
    assert recorder.iteration_stats(dataclasses.replace(idle, used_kv_blocks=3)) is not None
    assert recorder.iteration_stats(dataclasses.replace(idle, used_kv_blocks=3)) is None
    restarted = dataclasses.replace(idle, used_kv_blocks=3, instance_id="j")
    assert recorder.iteration_stats(restarted) is not None
    # A busy engine gives an entry at every read.
    busy = dataclasses.replace(idle, running_requests=1)
    assert recorder.iteration_stats(busy) is not None
    assert recorder.iteration_stats(busy) is not None
    queued = dataclasses.replace(idle, queued_requests=1)
    assert recorder.iteration_stats(queued) is not None
    assert recorder.iteration_stats(queued) is not None
    # An engine that does not report its running requests is not taken as idle.
    unknown = dataclasses.replace(idle, running_requests=None)
    assert recorder.iteration_stats(unknown) is not None
    assert recorder.iteration_stats(unknown) is not None


def test_iteration_stats_feed_the_in_process_gauges():
    registry = CollectorRegistry()
    collector = MetricsCollector({"model_name": MODEL, "engine_type": "pytorch"}, registry=registry)
    collector.log_iteration_stats(
        metrics_module.iteration_stats_from_load(
            EngineLoad(instance_id="i", running_requests=3, used_kv_blocks=10, total_kv_blocks=40)
        )
    )
    values = {
        sample.name: sample.value for family in registry.collect() for sample in family.samples
    }
    assert values["trtllm_num_requests_running"] == 3
    assert values["trtllm_kv_cache_utilization"] == pytest.approx(0.25)


# --- Through OpenAIServer and the gRPC fake ------------------------------------------


def _tokenizer() -> TransformersTokenizer:
    backend = Tokenizer(WordLevel(_VOCAB, unk_token="<unk>"))
    backend.decoder = DecoderSequence([ByteFallback(), Fuse()])
    backend.add_special_tokens(["</s>"])
    hf = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>")
    hf.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    return TransformersTokenizer(hf)


class _InputProcessor:
    def __call__(self, prompt: dict[str, Any], sampling_params: SamplingParams):
        return list(PROMPT_IDS), None


def _profile(**overrides: Any):
    profile = PROFILES["trtllm-like"]
    return profile.replace(
        model=MODEL,
        max_context_length=MAX_CONTEXT,
        vocab_size=len(_VOCAB),
        eos_token_ids=(EOS,),
        bos_token_id=None,
        server_info_extra={
            **profile.server_info_extra,
            metrics_module.KV_EVENT_HASH_ALGO_KEY: HASH_ALGO,
        },
        **overrides,
    )


def _finished_with_usage() -> generation_pb2.GenerateResponse:
    event = generation_pb2.GenerationFinished(
        output_index=0, reason=generation_pb2.FINISH_REASON_STOP
    )
    event.stop_match.eos_token_id = EOS
    response = generation_pb2.GenerateResponse(finished=event)
    response.usage.CopyFrom(
        generation_pb2.Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    )
    return response


def _serve(engine: FakeEngine, tmp_path, body, args: SimpleNamespace) -> Any:
    async def main() -> Any:
        async with engine:
            facade = OpenEngineLLM(
                args=args,
                tokenizer=_tokenizer(),
                input_processor=_InputProcessor(),
                hf_model_dir=str(tmp_path),
                endpoint=engine.target,
                ready_timeout=WAIT_SECONDS,
            )
            await facade.start()
            previous_media_executor = BaseMediaIO._executor
            server = OpenAIServer(
                generator=facade,
                model=MODEL,
                tool_parser=None,
                server_role=None,
                metadata_server_cfg=None,
            )
            adapt_server(server, facade)
            BaseMediaIO.set_executor(previous_media_executor)
            server._media_load_executor.shutdown(wait=False)
            try:
                transport = httpx.ASGITransport(app=server.app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                    return await asyncio.wait_for(body(engine, facade, http), WAIT_SECONDS * 4)
            finally:
                await facade.aclose()
                server._input_proc_executor.shutdown(wait=False)
                server._metrics_stats_executor.shutdown(wait=False)

    return asyncio.run(main())


@pytest.fixture
def private_prometheus(monkeypatch):
    """Give OpenAIServer's collector a registry of its own, outside multiprocess mode."""
    registries: list[CollectorRegistry] = []

    def from_args(args, model, **kwargs):
        registries.append(CollectorRegistry())
        return MetricsCollector(
            {"model_name": model, "engine_type": "pytorch"}, registry=registries[-1]
        )

    monkeypatch.setattr(openai_server_module.MetricsCollector, "from_args", from_args)
    # The config series read model fields these test arguments do not have.
    monkeypatch.setattr(MetricsCollector, "log_config_from_args", lambda self, args, model: None)
    monkeypatch.setattr(openai_server_module, "set_prometheus_multiproc_dir", lambda: None)
    monkeypatch.setattr(OpenAIServer, "mount_metrics", lambda self: None)
    return registries


def test_the_metrics_options_are_served_once_the_recorder_is_present():
    check_frontend_options(
        _args(
            return_perf_metrics=True,
            perf_metrics_output_dir="/tmp/x",
            enable_iter_perf_stats=True,
            otlp_traces_endpoint="localhost:1",
        )
    )


def test_a_completion_gets_perf_headers_and_prometheus_series(tmp_path, private_prometheus):
    engine = FakeEngine(_profile(), script=[token(0, HE), token(0, LLO), _finished_with_usage()])

    async def body(engine, facade, http):
        response = await http.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": PROMPT_IDS, "max_tokens": 8},
            headers={perf_metrics.RETURN_METRICS_HEADER: "1"},
        )
        assert response.status_code == 200, response.text
        return response

    response = _serve(engine, tmp_path, body, _args(return_perf_metrics=True))
    timing_header = response.headers[perf_metrics.SERVER_TIMING_HEADER]
    assert "server_ttft;dur=" in timing_header and "server_e2e;dur=" in timing_header
    assert "server_queue" not in timing_header
    (registry,) = private_prometheus
    observed = _observed(registry)
    for name in (
        "trtllm_request_success_total",
        "trtllm_e2e_request_latency_seconds_count",
        "trtllm_time_to_first_token_seconds_count",
        "trtllm_time_per_output_token_seconds_count",
        "trtllm_prompt_tokens_total",
        "trtllm_generation_tokens_total",
    ):
        assert name in observed, name
    assert "trtllm_request_queue_time_seconds_count" not in observed


def test_metrics_endpoint_reports_the_engine_load(tmp_path):
    engine = FakeEngine(_profile())

    async def body(engine, facade, http):
        response = await http.get("/metrics")
        assert response.status_code == 200, response.text
        return response.json()

    stats = _serve(engine, tmp_path, body, _args(enable_iter_perf_stats=True))
    assert len(stats) == 1
    assert stats[0]["source"] == metrics_module.GETLOAD_SOURCE
    assert stats[0]["numActiveRequests"] == 0
    assert [rpc for rpc, _ in engine.control_calls].count("GetLoad") >= 1

    async def twice(engine, facade, http):
        responses = [await http.get("/metrics") for _ in range(2)]
        return [response.json() for response in responses]

    # The idle engine's load has not changed since the first read.
    first, second = _serve(
        FakeEngine(_profile()), tmp_path, twice, _args(enable_iter_perf_stats=True)
    )
    assert len(first) == 1 and second == []


@pytest.fixture
def in_memory_tracer(monkeypatch):
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_global_tracer_", provider.get_tracer("trt.llm"))
    return exporter


TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
PARENT_SPAN_ID = "00f067aa0ba902b7"
TRACEPARENT = f"00-{TRACE_ID}-{PARENT_SPAN_ID}-01"


def test_trace_context_reaches_the_engine_and_parents_the_span(
    tmp_path, in_memory_tracer, private_prometheus
):
    engine = FakeEngine(_profile(), script=[token(0, HE), token(0, LLO), _finished_with_usage()])

    async def body(engine, facade, http):
        response = await http.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_completion_tokens": 8,
                "temperature": 0.9,
                "top_p": 0.9,
            },
            headers={"traceparent": TRACEPARENT},
        )
        assert response.status_code == 200, response.text
        return response.json()

    completion = _serve(engine, tmp_path, body, _args(return_perf_metrics=True))
    assert engine.received[0].metadata_value("traceparent") == TRACEPARENT
    (span,) = in_memory_tracer.get_finished_spans()
    assert span.name == "llm_request"
    assert format(span.context.trace_id, "032x") == TRACE_ID
    assert format(span.parent.span_id, "016x") == PARENT_SPAN_ID
    attributes = dict(span.attributes)
    assert attributes[tracing.SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] == pytest.approx(0.9)
    assert attributes[tracing.SpanAttributes.GEN_AI_REQUEST_TOP_P] == pytest.approx(0.9)
    assert attributes[tracing.SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 8
    assert (
        attributes[tracing.SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS]
        == completion["usage"]["completion_tokens"]
    )
    assert (
        attributes[tracing.SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS]
        == completion["usage"]["prompt_tokens"]
    )
    assert attributes[tracing.SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN] > 0
    assert attributes[tracing.SpanAttributes.GEN_AI_LATENCY_E2E] > 0
    # Measured inside the engine: not on the frontend's span.
    assert tracing.SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE not in attributes


def test_event_times_are_taken_as_the_events_arrive(tmp_path, private_prometheus):
    engine = FakeEngine(
        _profile(), script=[token(0, HE), token(0, LLO), token(0, WORLD), _finished_with_usage()]
    )

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
        await output.__anext__()
        # A slow reader: the remaining events arrive while it is away.
        await asyncio.sleep(2.0)
        await output.aresult()
        return output.metrics_dict

    got = _serve(engine, tmp_path, body, _args(return_perf_metrics=True))
    assert got[MetricNames.E2E] < 1.5


def test_an_aborted_request_is_recorded_as_cancelled(tmp_path, private_prometheus):
    engine = FakeEngine(_profile(), script=[token(0, HE), Hang()])

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
        await output.__anext__()
        output.abort()
        await asyncio.wait_for(output.aresult(), WAIT_SECONDS)
        return output.candidate_metrics

    (got,) = _serve(engine, tmp_path, body, _args(return_perf_metrics=True))
    assert got[MetricsCollector.labelname_finish_reason] == "cancelled"
    assert got[MetricNames.GENERATION_TOKENS] == 1


def test_a_recorder_failure_does_not_fail_the_request(tmp_path, monkeypatch, private_prometheus):
    def broken(self, event, now):
        raise RuntimeError("broken recorder")

    monkeypatch.setattr(metrics_module.OpenEngineRequestMetrics, "on_event", broken)
    engine = FakeEngine(_profile(), script=[token(0, HE), token(0, LLO), _finished_with_usage()])

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
        await output.aresult()
        return output

    output = _serve(engine, tmp_path, body, _args(return_perf_metrics=True))
    assert output.outputs[0].token_ids == [HE, LLO]
    assert output.metrics_dict == {}


@pytest.mark.parametrize("broken_method", ["start_request", "metrics_dict", "candidate_metrics"])
def test_a_recorder_that_cannot_start_or_report_does_not_fail_the_request(
    tmp_path, monkeypatch, broken_method, private_prometheus
):
    def broken(*args, **kwargs):
        raise RuntimeError("broken recorder")

    owner = (
        metrics_module.OpenEngineMetricsRecorder
        if broken_method == "start_request"
        else metrics_module.OpenEngineRequestMetrics
    )
    monkeypatch.setattr(owner, broken_method, broken)
    engine = FakeEngine(_profile(), script=[token(0, HE), token(0, LLO), _finished_with_usage()])

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
        await output.aresult()
        return output

    output = _serve(engine, tmp_path, body, _args(return_perf_metrics=True))
    assert output.outputs[0].token_ids == [HE, LLO]
    if broken_method == "start_request":
        assert output.metrics_dict == {}
        assert output.candidate_metrics == []
    else:
        assert not getattr(output, broken_method)


def test_no_span_without_perf_metrics(tmp_path, in_memory_tracer):
    engine = FakeEngine(_profile(), script=[token(0, HE), _finished_with_usage()])

    async def body(engine, facade, http):
        response = await http.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": PROMPT_IDS, "max_tokens": 8},
            headers={"traceparent": TRACEPARENT},
        )
        assert response.status_code == 200, response.text

    _serve(engine, tmp_path, body, _args())
    # The trace context is still forwarded; as in-process, no span without perf metrics.
    assert engine.received[0].metadata_value("traceparent") == TRACEPARENT
    assert in_memory_tracer.get_finished_spans() == ()


def test_otlp_endpoint_sets_up_the_tracer_once(monkeypatch):
    calls = []
    monkeypatch.setattr(tracing, "_global_tracer_", None)

    def init_tracer(name, endpoint):
        calls.append((name, endpoint))
        monkeypatch.setattr(tracing, "_global_tracer_", object())

    monkeypatch.setattr(tracing, "init_tracer", init_tracer)
    metrics_module.create_metrics_recorder(_args(otlp_traces_endpoint="collector:4317"))
    metrics_module.create_metrics_recorder(_args(otlp_traces_endpoint="collector:4317"))
    assert calls == [("trt.llm", "collector:4317")]


# --- KV-cache events -------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _source(port: int, **fields: Any) -> kv_pb2.KvEventSource:
    values = dict(
        transport="zmq",
        endpoint_addr=kv_pb2.KvEndpoint(host="127.0.0.1", port=port, protocol="tcp"),
        topic="",
        data_parallel_rank=0,
        encoding="msgpack",
    )
    values.update(fields)
    return kv_pb2.KvEventSource(**values)


def test_kv_cache_events_arrive_through_get_kv_event_sources(tmp_path):
    from tensorrt_llm._torch.pyexecutor.kv_cache_events import (
        AllBlocksCleared,
        BlockRemoved,
        BlockStored,
        KVEventBatch,
        ZmqEventPublisher,
    )

    port = _free_port()
    publisher = ZmqEventPublisher(0, endpoint=f"tcp://127.0.0.1:{port}")
    publisher.start()
    engine = FakeEngine(_profile(kv_event_sources=(_source(port),)))

    async def body(engine, facade, http):
        # The first call subscribes; nothing was received before it.
        first = await http.post("/kv_cache_events")
        assert first.status_code == 200, first.text
        assert first.json() == []
        events: list = []
        deadline = time.monotonic() + WAIT_SECONDS
        while not events and time.monotonic() < deadline:
            # A subscription takes a moment to connect; publish until one gets through.
            publisher.publish(
                KVEventBatch(
                    ts=time.time(),
                    events=[
                        BlockStored(
                            block_hashes=[11, 12],
                            parent_block_hash=None,
                            token_ids=list(range(1, 9)),
                            block_size=4,
                            lora_id=None,
                            medium="GPU",
                            lora_name=None,
                        ),
                        BlockRemoved(block_hashes=[11], medium="GPU"),
                        AllBlocksCleared(),
                    ],
                )
            )
            await asyncio.sleep(0.1)
            response = await http.post("/kv_cache_events")
            assert response.status_code == 200, response.text
            events = response.json()
        return events

    try:
        events = _serve(engine, tmp_path, body, _args())
    finally:
        publisher.shutdown()
    stored, removed, cleared = events[:3]
    assert [e["event_id"] for e in events[:3]] == [0, 1, 2]
    assert all(e["hash_algo"] == HASH_ALGO for e in events)
    assert all("attention_dp_rank" not in e for e in events)
    assert stored["data"]["type"] == "stored"
    assert stored["data"]["parent_hash"] is None
    blocks = stored["data"]["blocks"]
    assert [b["block_hash"] for b in blocks] == [11, 12]
    assert [[t["token_id"] for t in b["tokens"]] for b in blocks] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert {b["cache_level"] for b in blocks} == {0}
    assert removed["data"] == {"type": "removed", "block_hashes": [11]}
    assert cleared["data"] == {"type": "all_blocks_cleared"}
    assert [rpc for rpc, _ in engine.control_calls].count("GetKvEventSources") == 1


@pytest.mark.parametrize(
    "sources",
    [(), (_source(1, encoding="protobuf"),), (_source(1, transport="grpc"),)],
    ids=["none", "protobuf", "grpc"],
)
def test_kv_cache_events_the_frontend_cannot_read_answer_501(tmp_path, sources):
    engine = FakeEngine(_profile(kv_event_sources=sources))

    async def body(engine, facade, http):
        return [await http.post("/kv_cache_events") for _ in range(2)]

    responses = _serve(engine, tmp_path, body, _args())
    assert [r.status_code for r in responses] == [501, 501], responses[0].text
    # The engine's answer is kept: asked once.
    assert [rpc for rpc, _ in engine.control_calls].count("GetKvEventSources") == 1


def test_published_events_convert_to_the_in_process_format():
    batch = [
        123.0,
        [
            {
                "type": "BlockStored",
                "block_hashes": [7],
                "parent_block_hash": 3,
                "token_ids": [1, 2],
                "block_size": 2,
                "lora_id": None,
                "medium": "CPU",
                "lora_name": None,
                "group_idx": 1,
                "kv_cache_spec_sliding_window": 128,
                "mm_keys": [[{"type": "mm_key", "hash": "ab", "start_offset": 1, "uuid": "u"}]],
            },
            {"type": "SomethingNew"},
        ],
        1,
    ]
    (entry,) = metrics_module.to_in_process_events(
        batch, first_event_id=5, hash_algo=None, attention_dp_rank=1
    )
    assert entry["event_id"] == 5
    assert entry["window_size"] == 128
    assert entry["layer_group_id"] == 1
    assert entry["attention_dp_rank"] == 1
    assert "hash_algo" not in entry
    (block,) = entry["data"]["blocks"]
    assert entry["data"]["parent_hash"] == 3
    assert block["cache_level"] == 1
    # Not in the published events; present as in-process, as None.
    assert block["priority"] is None and block["cache_salt"] is None
    assert block["mm_keys"] == [{"type": "mm_key", "hash": "ab", "start_offset": 1, "uuid": "u"}]
    assert block["tokens"] == [
        {"type": "unique_token", "token_id": 1, "token_extra_id": 0},
        {"type": "unique_token", "token_id": 2, "token_extra_id": 0},
    ]


class _Socket:
    """A SUB socket stand-in holding the messages to receive."""

    def __init__(self, messages: list[list[bytes]]) -> None:
        self.messages = list(messages)

    def recv_multipart(self, flags: int) -> list[bytes]:
        if not self.messages:
            raise zmq.Again()
        return self.messages.pop(0)


def _message(seq: int, batch: Any) -> list[bytes]:
    return [b"", seq.to_bytes(8, "big"), msgspec.msgpack.encode(batch)]


def _subscription(*messages: list[bytes]):
    subscription = metrics_module._KvEventSubscription()
    source = KvEventSource(transport="zmq", host="127.0.0.1", port=1)
    subscription._subscribers = [metrics_module._Subscriber(source, _Socket(list(messages)))]
    return subscription


def _removed(block_hash: int) -> list[Any]:
    return [1.0, [{"type": "BlockRemoved", "block_hashes": [block_hash]}]]


def test_a_malformed_batch_is_skipped_and_the_others_kept():
    endless_block = [
        1.0,
        [
            {
                "type": "BlockStored",
                "block_hashes": [1],
                "token_ids": [1],
                "block_size": float("inf"),
            }
        ],
    ]
    # Values a JSON response cannot carry.
    byte_hash = [1.0, [{"type": "BlockRemoved", "block_hashes": [b"\x00"]}]]
    nan_hash = [1.0, [{"type": "BlockRemoved", "block_hashes": [float("nan")]}]]
    subscription = _subscription(
        _message(0, 5),
        _message(1, _removed(9)),
        _message(2, [1.0, 7]),
        _message(3, endless_block),
        _message(4, byte_hash),
        _message(5, nan_hash),
    )
    (entry,) = subscription._drain()
    assert entry["event_id"] == 0
    assert entry["data"] == {"type": "removed", "block_hashes": [9]}


def test_one_call_converts_a_bounded_number_of_batches(monkeypatch):
    monkeypatch.setattr(metrics_module, "KV_EVENT_MAX_BATCHES", 2)
    subscription = _subscription(*(_message(seq, _removed(seq)) for seq in range(3)))
    assert [e["data"]["block_hashes"] for e in subscription._drain()] == [[0], [1]]
    assert [e["event_id"] for e in subscription._drain()] == [2]


def test_every_publisher_gets_its_share_of_a_call(monkeypatch):
    monkeypatch.setattr(metrics_module, "KV_EVENT_MAX_BATCHES", 2)
    subscription = _subscription(*(_message(seq, _removed(seq)) for seq in range(3)))
    busy = subscription._subscribers[0]
    quiet = metrics_module._Subscriber(busy.source, _Socket([_message(0, _removed(100))]))
    subscription._subscribers.append(quiet)
    assert [e["data"]["block_hashes"] for e in subscription._drain()] == [[0], [100]]


class _Client:
    def __init__(self, sources=None, error: Exception | None = None) -> None:
        self.sources = sources or []
        self.error = error
        self.calls = 0

    def server_info(self):
        return SimpleNamespace(engine_name="e", extra={})

    async def kv_event_sources(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.sources


def test_kv_cache_events_are_empty_while_the_sources_cannot_be_read():
    client = _Client(error=OpenEngineClientError("down", target="x"))
    subscription = metrics_module._KvEventSubscription()

    async def twice():
        return [await subscription.events(client, 0) for _ in range(2)]

    assert asyncio.run(twice()) == [[], []]
    # Not subscribed: the next call asks again.
    assert client.calls == 2


def test_kv_cache_events_unsupported_raise_a_new_error_each_call():
    client = _Client(sources=[])
    subscription = metrics_module._KvEventSubscription()

    async def raised():
        try:
            await subscription.events(client, 0)
        except contracts.OpenEngineUnsupportedError as error:
            return error
        raise AssertionError("not raised")

    async def twice():
        return [await raised() for _ in range(2)]

    first, second = asyncio.run(twice())
    # Raising the same error again would lengthen its traceback at every call.
    assert first is not second
    assert str(first) == str(second)
    assert client.calls == 1


def test_kv_cache_events_arrive_from_an_ipv6_publisher():
    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.setsockopt(zmq.IPV6, 1)
    try:
        port = publisher.bind_to_random_port("tcp://[::1]")
    except zmq.ZMQError as error:
        publisher.close()
        pytest.skip(f"no IPv6 loopback: {error}")
    source = KvEventSource(transport="zmq", host="::1", port=port, encoding="msgpack")
    client = _Client(sources=[source])
    subscription = metrics_module._KvEventSubscription()

    async def receive():
        assert await subscription.events(client, 0) == []
        # The subscription connects in the background: publish until it is.
        for seq in range(200):
            publisher.send_multipart(_message(seq, _removed(seq)))
            entries = await subscription.events(client, 0.01)
            if entries:
                return entries
        return []

    try:
        entries = asyncio.run(receive())
    finally:
        publisher.close()
        for subscriber in subscription._subscribers or []:
            subscriber.socket.close()
    assert entries and entries[0]["data"]["type"] == "removed"


def test_kv_cache_events_are_empty_while_a_publisher_cannot_be_connected(monkeypatch):
    monkeypatch.setattr(metrics_module, "_tcp_endpoint", lambda host, port: "nope://x")
    source = KvEventSource(transport="zmq", host="127.0.0.1", port=1, encoding="msgpack")
    client = _Client(sources=[source, source])
    subscription = metrics_module._KvEventSubscription()

    async def twice():
        return [await subscription.events(client, 0) for _ in range(2)]

    assert asyncio.run(twice()) == [[], []]
    assert client.calls == 2

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request metrics, iteration stats, KV-cache events and tracing of an OpenEngine frontend.

The ``create_metrics_recorder`` slot of :mod:`.contracts` (SV-MET):

- Request metrics (SV-MET-1) are measured on the serving side from the arrival
  time and the times the engine events arrive: time to first token, end-to-end
  latency, time per output token and decode time, with the token counts, in
  the ``metrics_dict`` keys the in-process ``RequestOutput`` uses, so
  ``/prometheus/metrics`` keeps its metric names. The phases inside the engine
  (queueing, prefill, KV-cache transfer) cannot be measured here and are left
  out, not reported as 0. As in-process, each output gives one candidate
  entry, and ``metrics_dict`` merges them, a later entry's values replacing
  an earlier one's. When a request ends, completed or aborted, each output
  gets a ``request_perf_metrics`` holding the same serving-side times, None
  for the engine's own times and 0 draft tokens, which the per-request perf
  records and headers are built from.
- ``/metrics`` (SV-MET-2): with ``enable_iter_perf_stats`` on, as
  in-process, a read of the iteration stats gives an entry derived from the
  engine's ``GetLoad``, with ``"source": "openengine_getload"``. It gives none
  when the load cannot be read, or when the engine reports no running or
  queued requests, its load is the one the last entry reported and no request
  has started on this frontend since: in-process an idle engine produces no
  iteration stats, and a multi-frontend launcher reads them every second.
- ``/kv_cache_events`` (SV-MET-4) subscribes, at its first call, to the ZeroMQ
  publishers ``GetKvEventSources`` names, and returns the events received since
  the last call (at most :data:`KV_EVENT_MAX_BATCHES` batches per call; the
  rest wait for the next), converted to the in-process format; see
  :func:`to_in_process_events` for what the conversion cannot carry over.
  Events published before the subscription is connected, which happens in
  the background after the first call, are not received. While the engine
  cannot be reached, or names a publisher address ZeroMQ rejects, it answers
  with no events and tries to subscribe again at the next call.
- Tracing (SV-MET-5): with ``otlp_traces_endpoint`` the frontend sets up the
  OTLP tracer, and each request that completes or is aborted records one
  ``llm_request`` span, named as in-process, with the in-process attributes
  the frontend measures, whose parent is the trace context the request
  arrived with. The facade forwards that context to the engine as
  ``traceparent``/``tracestate``.

As in-process, request metrics and spans are recorded only when perf metrics
are on, server-wide (``return_perf_metrics``) or for the request
(``SamplingParams.return_perf_metrics``), and only once the request has been
read to its end: a streaming request whose client went away before its end
is not recorded.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tensorrt_llm.grpc.openengine_client import OpenEngineClientError
from tensorrt_llm.logger import logger
from tensorrt_llm.metrics.collector import MetricsCollector
from tensorrt_llm.metrics.enums import MetricNames, RequestEventTiming
from tensorrt_llm.metrics.perf_utils import process_req_perf_metrics

from .contracts import OpenEngineUnsupportedError
from .engine_client import (
    EngineClient,
    EngineEvent,
    EngineLoad,
    KvEventSource,
    OutputFinished,
    PromptInfo,
    TokenDelta,
)

if TYPE_CHECKING:
    from tensorrt_llm.executor.result import CompletionOutput
    from tensorrt_llm.sampling_params import SamplingParams

__all__ = [
    "GETLOAD_SOURCE",
    "KV_EVENT_HASH_ALGO_KEY",
    "OpenEngineMetricsRecorder",
    "OpenEngineRequestMetrics",
    "ServingKvCacheMetrics",
    "ServingPerfMetrics",
    "ServingSpeculativeDecodingMetrics",
    "ServingTimingMetrics",
    "create_metrics_recorder",
    "iteration_stats_from_load",
    "to_in_process_events",
]

# The value of "source" in the /metrics entries derived from GetLoad.
GETLOAD_SOURCE = "openengine_getload"
# ServerInfo.extra key naming the hash algorithm of the engine's KV-cache event block hashes.
KV_EVENT_HASH_ALGO_KEY = "trtllm_kv_event_hash_algo"
# About how many event batches the subscription to one publisher holds between
# two /kv_cache_events calls (a ZeroMQ high-water mark); ZeroMQ drops what comes
# beyond it.
KV_EVENT_RECEIVE_HWM = 100_000
# How many event batches one /kv_cache_events call converts, shared among the
# publishers, so that a backlog does not hold up the event loop for long.
KV_EVENT_MAX_BATCHES = 1_000
_KV_EVENT_POLL_SECONDS = 0.01
# The storage medium of a block in the engine's events, as the in-process cache level.
_CACHE_LEVELS = {"GPU": 0, "CPU": 1}


def create_metrics_recorder(args: Any) -> OpenEngineMetricsRecorder:
    """The ``create_metrics_recorder`` slot: a recorder for a frontend serving with ``args``."""
    return OpenEngineMetricsRecorder(args)


def _steady_clock() -> float:
    """The runtime's steady clock, which the in-process perf records and the server use."""
    from tensorrt_llm._utils import get_global_steady_clock_now_in_seconds

    return get_global_steady_clock_now_in_seconds()


class OpenEngineMetricsRecorder:
    """The metrics recorder of one frontend.

    ``steady_clock`` gives the time the per-request perf records are stated in;
    it defaults to the clock ``OpenAIServer`` stamps its own times with.
    """

    def __init__(self, args: Any, *, steady_clock: Callable[[], float] = _steady_clock) -> None:
        self._args = args
        self._steady_clock = steady_clock
        self._kv_events = _KvEventSubscription()
        # The load the last /metrics entry reported, and whether a request
        # has started here since.
        self._last_load: tuple[Any, ...] | None = None
        self._started_since_entry = False
        self._init_tracing(getattr(args, "otlp_traces_endpoint", None))

    @staticmethod
    def _init_tracing(endpoint: str | None) -> None:
        if not endpoint:
            return
        from tensorrt_llm.llmapi import tracing

        if tracing.is_tracing_enabled():
            # One tracer per process; another frontend object here has set it up.
            return
        try:
            tracing.init_tracer("trt.llm", endpoint)
            logger.info(f"Initialized OTLP tracer successfully, endpoint: {endpoint}")
        except Exception as error:  # noqa: BLE001 - as in-process, serving goes on without traces
            logger.error(f"Failed to initialize OTLP tracer: {error}")

    def start_request(
        self,
        request_id: str,
        arrival_time: float,
        *,
        sampling_params: SamplingParams | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> OpenEngineRequestMetrics | _UnmeasuredRequest:
        self._started_since_entry = True
        if not _perf_metrics_on(self._args, sampling_params):
            return _UnmeasuredRequest()
        return OpenEngineRequestMetrics(
            request_id,
            arrival_time,
            steady_offset=self._steady_clock() - arrival_time,
            sampling_params=sampling_params,
            trace_headers=trace_headers,
        )

    def iteration_stats(self, load: EngineLoad | None) -> dict[str, Any] | None:
        # As in-process, /metrics has entries only with enable_iter_perf_stats.
        if load is None or not getattr(self._args, "enable_iter_perf_stats", False):
            return None
        snapshot = (
            load.instance_id,
            load.running_requests,
            load.queued_requests,
            load.used_kv_blocks,
            load.total_kv_blocks,
        )
        # An engine that does not report its running requests is never taken as idle.
        idle = load.running_requests == 0 and not load.queued_requests
        if idle and snapshot == self._last_load and not self._started_since_entry:
            return None
        self._last_load = snapshot
        self._started_since_entry = False
        return iteration_stats_from_load(load)

    async def kv_cache_events(self, client: EngineClient, timeout: float | None) -> list[dict]:
        return await self._kv_events.events(client, timeout)


def _perf_metrics_on(args: Any, sampling_params: SamplingParams | None) -> bool:
    # Read per request: OpenAIServer turns return_perf_metrics on in the args
    # after the frontend is built, when it writes perf records to files.
    return bool(getattr(args, "return_perf_metrics", False)) or bool(
        getattr(sampling_params, "return_perf_metrics", False)
    )


class _UnmeasuredRequest:
    """A request with perf metrics off: nothing is measured or recorded."""

    def on_event(self, event: EngineEvent, now: float) -> None:
        pass

    def metrics_dict(self) -> dict[Any, Any]:
        return {}

    def candidate_metrics(self) -> list[dict[Any, Any]]:
        return []

    def finish(
        self,
        outputs: Sequence[CompletionOutput],
        now: float,
        *,
        prompt_tokens: int,
        assembled: bool,
    ) -> None:
        from tensorrt_llm.llmapi import tracing

        if tracing.is_tracing_enabled():
            # As in-process: a span needs the request's perf metrics.
            tracing.insufficient_request_metrics_warning()


# --- Request metrics -------------------------------------------------------------


@dataclass
class _OutputTiming:
    """What the events of one output said; times are the facade's (wall clock)."""

    # When each token arrived.
    token_times: list[float] = field(default_factory=list)
    # Each token's logprob, None where the engine sent none or a non-finite one.
    token_logprobs: list[float | None] = field(default_factory=list)
    finish_reason: str | None = None

    def kept(self, length: int) -> tuple[float | None, float | None, list[float]]:
        """The first and last token times and the logprobs of an output of ``length`` tokens.

        The output keeps its first ``length`` tokens; those past them, which the
        response left out after a serving-side stop, do not count.
        """
        times = self.token_times
        if not times:
            return None, None, []
        logprobs = [lp for lp in self.token_logprobs[:length] if lp is not None]
        return times[0], times[max(min(length, len(times)), 1) - 1], logprobs


@dataclass(frozen=True)
class ServingTimingMetrics:
    """``RequestPerfMetrics.timing_metrics`` as the serving side measures it.

    Times are steady-clock offsets, as in-process; the engine's own phases are None.
    """

    arrival_time: datetime.timedelta | None = None
    first_token_time: datetime.timedelta | None = None
    last_token_time: datetime.timedelta | None = None
    first_scheduled_time: None = None
    kv_cache_transfer_start: None = None
    kv_cache_transfer_end: None = None
    kv_cache_size: None = None


@dataclass(frozen=True)
class ServingKvCacheMetrics:
    """``RequestPerfMetrics.kv_cache_metrics``: measured in the engine, so unknown here."""

    num_total_allocated_blocks: None = None
    num_new_allocated_blocks: None = None
    num_reused_blocks: None = None
    num_missed_blocks: None = None
    kv_cache_hit_rate: None = None


@dataclass(frozen=True)
class ServingSpeculativeDecodingMetrics:
    """``RequestPerfMetrics.speculative_decoding``: the engine reports no drafts to the frontend."""

    acceptance_rate: None = None
    total_accepted_draft_tokens: int = 0
    total_draft_tokens: int = 0


@dataclass(frozen=True)
class ServingPerfMetrics:
    """The ``request_perf_metrics`` of an output served through an OpenEngine engine."""

    timing_metrics: ServingTimingMetrics
    first_iter: None = None
    last_iter: None = None
    kv_cache_metrics: ServingKvCacheMetrics = field(default_factory=ServingKvCacheMetrics)
    speculative_decoding: ServingSpeculativeDecodingMetrics = field(
        default_factory=ServingSpeculativeDecodingMetrics
    )


class OpenEngineRequestMetrics:
    """The serving-side timing of one request with perf metrics on.

    ``steady_offset`` turns the facade's wall-clock times into steady-clock
    times for the perf records.
    """

    def __init__(
        self,
        request_id: str,
        arrival_time: float,
        *,
        steady_offset: float,
        sampling_params: SamplingParams | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.request_id = request_id
        self.arrival_time = arrival_time
        self._steady_offset = steady_offset
        self._sampling_params = sampling_params
        self._trace_headers = dict(trace_headers) if trace_headers else None
        self._outputs: dict[int, _OutputTiming] = {}
        # Output indices in the order their OutputFinished events arrived.
        self._finish_order: list[int] = []
        # The engine's counts, from the first usage received.
        self._usage_seen = False
        self._usage_prompt_tokens: int | None = None
        self._usage_cached_tokens: int | None = None
        self._prompt_tokens = 0
        self._prompt_logprob_sum = 0.0
        self._prompt_logprob_count = 0
        self._metrics: dict[Any, Any] = {}
        self._candidates: list[dict[Any, Any]] = []
        self._finished = False

    def _output(self, index: int) -> _OutputTiming:
        timing = self._outputs.get(index)
        if timing is None:
            timing = self._outputs[index] = _OutputTiming()
        return timing

    def on_event(self, event: EngineEvent, now: float) -> None:
        if self._finished:
            # The metrics are recorded; what the engine sends after the end changes nothing.
            return
        if isinstance(event, TokenDelta):
            if not event.token_ids:
                return
            timing = self._output(event.output_index)
            timing.token_times.extend([now] * len(event.token_ids))
            logprobs = list(event.logprobs or ())
            for i in range(len(event.token_ids)):
                value = logprobs[i].logprob if i < len(logprobs) else None
                timing.token_logprobs.append(
                    value if value is not None and math.isfinite(value) else None
                )
        elif isinstance(event, OutputFinished):
            timing = self._output(event.output_index)
            if timing.finish_reason is None:
                self._finish_order.append(event.output_index)
            timing.finish_reason = event.reason
            if event.usage is not None and not self._usage_seen:
                # The first one counts, as in the response's usage.
                self._usage_seen = True
                self._usage_prompt_tokens = event.usage.prompt_tokens
                self._usage_cached_tokens = event.usage.cached_prompt_tokens
        elif isinstance(event, PromptInfo):
            # The first prompt token has no logprob; the engine sends a
            # placeholder entry for it, which the response leaves out too.
            for logprob in event.token_logprobs[1:]:
                if math.isfinite(logprob.logprob):
                    self._prompt_logprob_sum += logprob.logprob
                    self._prompt_logprob_count += 1

    def metrics_dict(self) -> dict[Any, Any]:
        return self._metrics

    def candidate_metrics(self) -> list[dict[Any, Any]]:
        return self._candidates

    def finish(
        self,
        outputs: Sequence[CompletionOutput],
        now: float,
        *,
        prompt_tokens: int,
        assembled: bool,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        # The engine counts the prompt as it ran it (with any media tokens
        # expanded, as in-process); the facade's count stands in when no usage
        # arrived, as when a serving-side stop ended the request first. That
        # count leaves media placeholders unexpanded.
        self._prompt_tokens = (
            self._usage_prompt_tokens if self._usage_prompt_tokens else prompt_tokens
        )
        by_index = {getattr(output, "index", i): output for i, output in enumerate(outputs)}
        # In-process records candidates as they finish: those the engine ended
        # first, in that order, then the rest (a serving-side stop or an abort
        # ended them), by index.
        order = [i for i in self._finish_order if i in by_index]
        order += [i for i in sorted(by_index) if i not in order]
        # Everything is computed before anything is recorded, so a failure
        # leaves no candidate recorded.
        recorded = []
        for index in order:
            output = by_index[index]
            timing = self._outputs.get(index, _OutputTiming())
            # The tokens the response serves; with the postprocessing pool the
            # outputs hold none, and the events tell.
            length = (
                len(getattr(output, "token_ids", None) or ())
                if assembled
                else len(timing.token_times)
            )
            stats = self._candidate_metrics(index, output, timing, length)
            recorded.append((output, stats, length, self._perf_metrics(timing, length)))
        for output, stats, _, perf_metrics in recorded:
            self._candidates.append(stats)
            self._metrics.update(stats)
            output.request_perf_metrics = perf_metrics
        if recorded:
            # As in-process, the span takes the times of the request's metrics
            # and the last candidate's output.
            output, _, length, _ = recorded[-1]
            self._trace(output, self._metrics, length, now)

    def _candidate_metrics(
        self,
        index: int,
        output: Any,
        timing: _OutputTiming,
        length: int,
    ) -> dict[Any, Any]:
        """One candidate's entries, as ``GenerationResultBase.record_stats`` makes them in-process."""
        stats: dict[Any, Any] = {}
        # An output that has no finish reason of its own and that the engine
        # did not end was ended in the postprocessing pool; it is recorded as
        # stopped.
        finish_reason = getattr(output, "finish_reason", None) or timing.finish_reason or "stop"
        stats[MetricsCollector.labelname_finish_reason] = finish_reason
        first_token_time, last_token_time, logprobs = timing.kept(length)
        times = {RequestEventTiming.ARRIVAL_TIME: self.arrival_time}
        if first_token_time is not None:
            times[RequestEventTiming.FIRST_TOKEN_TIME] = first_token_time
        if last_token_time is not None:
            times[RequestEventTiming.LAST_TOKEN_TIME] = last_token_time
        # No first scheduled time: the queue, prefill and inference times stay out.
        stats.update(process_req_perf_metrics(times, length))
        if index == 0:
            if self._prompt_tokens > 0:
                stats[MetricNames.PROMPT_TOKENS] = self._prompt_tokens
            # Unknown, not 0, when no usage arrived (a serving-side stop ended
            # the request first).
            if self._usage_cached_tokens is not None:
                stats[MetricNames.PROMPT_CACHE_CACHED_TOKENS] = self._usage_cached_tokens
            if self._prompt_logprob_count:
                ppl = math.exp(-self._prompt_logprob_sum / self._prompt_logprob_count)
                if math.isfinite(ppl):
                    stats[MetricNames.PREFILL_PERPLEXITY] = ppl
        if length > 0 and logprobs:
            ppl = math.exp(-sum(logprobs) / len(logprobs))
            if math.isfinite(ppl):
                stats[MetricNames.GENERATION_PERPLEXITY] = ppl
        return stats

    def _steady(self, wall: float | None) -> datetime.timedelta | None:
        if wall is None:
            return None
        return datetime.timedelta(seconds=wall + self._steady_offset)

    def _perf_metrics(self, timing: _OutputTiming, length: int) -> ServingPerfMetrics:
        first_token_time, last_token_time, _ = timing.kept(length)
        return ServingPerfMetrics(
            timing_metrics=ServingTimingMetrics(
                arrival_time=self._steady(self.arrival_time),
                first_token_time=self._steady(first_token_time),
                last_token_time=self._steady(last_token_time),
            )
        )

    def _trace(self, output: Any, stats: Mapping[Any, Any], length: int, now: float) -> None:
        """Record the request's span, as ``GenerationResultBase.do_tracing`` does in-process."""
        from tensorrt_llm.llmapi import tracing

        tracer = tracing.global_otlp_tracer()
        if not tracer:
            return
        attributes: dict[str, Any] = {}
        params = self._sampling_params
        for attribute, name in (
            (tracing.SpanAttributes.GEN_AI_REQUEST_TEMPERATURE, "temperature"),
            (tracing.SpanAttributes.GEN_AI_REQUEST_TOP_P, "top_p"),
            (tracing.SpanAttributes.GEN_AI_REQUEST_TOP_K, "top_k"),
            (tracing.SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS, "max_tokens"),
            (tracing.SpanAttributes.GEN_AI_REQUEST_N, "n"),
        ):
            attributes[attribute] = getattr(params, name, None)
        attributes[tracing.SpanAttributes.GEN_AI_REQUEST_ID] = self.request_id
        attributes[tracing.SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] = self._prompt_tokens or None
        attributes[tracing.SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] = length
        attributes[tracing.SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN] = stats.get(
            MetricNames.TTFT
        )
        attributes[tracing.SpanAttributes.GEN_AI_LATENCY_E2E] = stats.get(MetricNames.E2E)
        # Unlike in-process, no time in queue or KV-cache transfer time: the engine measures them.
        finish_reason = stats.get(MetricsCollector.labelname_finish_reason)
        attributes[tracing.SpanAttributes.GEN_AI_RESPONSE_FINISH_REASONS] = (
            json.dumps([finish_reason]) if finish_reason else None
        )
        span = tracer.start_span(
            "llm_request",
            kind=tracing.SpanKind.SERVER,
            context=tracing.extract_trace_context(self._trace_headers),
            start_time=int(self.arrival_time * 1e9),
        )
        for name, value in attributes.items():
            if value is not None:
                span.set_attribute(name, value)
        span.end(end_time=int(now * 1e9))


# --- Iteration stats ---------------------------------------------------------------


def iteration_stats_from_load(load: EngineLoad) -> dict[str, Any]:
    """The ``/metrics`` entry for an engine load snapshot.

    Field names follow the in-process iteration stats; a field the snapshot
    lacks is left out rather than reported as 0.
    """
    stats: dict[str, Any] = {
        "source": GETLOAD_SOURCE,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if load.running_requests is not None:
        stats["numActiveRequests"] = load.running_requests
    if load.queued_requests is not None:
        stats["numQueuedRequests"] = load.queued_requests
    kv: dict[str, int] = {}
    if load.used_kv_blocks is not None:
        kv["usedNumBlocks"] = load.used_kv_blocks
    if load.total_kv_blocks is not None:
        kv["maxNumBlocks"] = load.total_kv_blocks
        if load.used_kv_blocks is not None:
            kv["freeNumBlocks"] = max(load.total_kv_blocks - load.used_kv_blocks, 0)
    if kv:
        stats["kvCacheStats"] = kv
    return stats


# --- KV-cache events ---------------------------------------------------------------


def to_in_process_events(
    batch: Sequence[Any],
    *,
    first_event_id: int,
    hash_algo: str | None,
    attention_dp_rank: int | None,
) -> list[dict[str, Any]]:
    """Convert one published event batch to ``/kv_cache_events`` entries as in-process.

    ``batch`` is a decoded ``(ts, events[, data_parallel_rank])`` batch of the
    engine's ZeroMQ publisher. ``event_id`` numbers the entries from
    ``first_event_id``; ``attention_dp_rank``, when not None, is stated on
    each. What the published events do not carry is None (a block's
    ``priority`` and ``cache_salt``, the ``window_size`` of a full-attention
    group), 0 (a token's ``token_extra_id``) or absent (the engine's
    ``created`` and ``updated`` events); a block's ``cache_level`` follows its
    medium. An ``mm_key``'s ``hash`` is the published digest, with its
    ``uuid`` beside it when it has one. All blocks cleared, which has no
    in-process counterpart, becomes ``{"type": "all_blocks_cleared"}``.

    Raises:
        TypeError, ValueError, KeyError, AttributeError, IndexError, OverflowError:
            The batch does not have the published shape.
    """
    events = batch[1] if len(batch) > 1 else []
    entries: list[dict[str, Any]] = []
    for event in events:
        data = _event_data(event)
        if data is None:
            continue
        entry: dict[str, Any] = {
            "event_id": first_event_id + len(entries),
            "data": data,
            "window_size": event.get("kv_cache_spec_sliding_window"),
        }
        if hash_algo is not None:
            entry["hash_algo"] = hash_algo
        if event.get("group_idx") is not None:
            entry["layer_group_id"] = event["group_idx"]
        if attention_dp_rank is not None:
            entry["attention_dp_rank"] = attention_dp_rank
        entries.append(entry)
    return entries


def _event_data(event: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = event.get("type")
    if kind == "BlockStored":
        return _stored(event)
    if kind == "BlockRemoved":
        return {"type": "removed", "block_hashes": list(event.get("block_hashes") or [])}
    if kind == "AllBlocksCleared":
        return {"type": "all_blocks_cleared"}
    logger.warning(f"Ignoring a KV-cache event of unknown type {kind!r}")
    return None


def _stored(event: Mapping[str, Any]) -> dict[str, Any]:
    hashes = list(event.get("block_hashes") or [])
    tokens = list(event.get("token_ids") or [])
    block_size = int(event.get("block_size") or 0)
    mm_keys = event.get("mm_keys") or []
    level = _CACHE_LEVELS.get(event.get("medium"))
    blocks = []
    for i, block_hash in enumerate(hashes):
        block_tokens = tokens[i * block_size : (i + 1) * block_size] if block_size else []
        blocks.append(
            {
                "type": "stored_block",
                "block_hash": block_hash,
                "tokens": [
                    {"type": "unique_token", "token_id": token, "token_extra_id": 0}
                    for token in block_tokens
                ],
                "cache_salt": None,
                "cache_level": level,
                "priority": None,
                "mm_keys": [
                    _mm_key(key) for key in (mm_keys[i] if i < len(mm_keys) else None) or []
                ],
            }
        )
    return {"type": "stored", "parent_hash": event.get("parent_block_hash"), "blocks": blocks}


def _mm_key(key: Mapping[str, Any]) -> dict[str, Any]:
    entry = {"type": "mm_key", "hash": key.get("hash"), "start_offset": key.get("start_offset")}
    if key.get("uuid") is not None:
        entry["uuid"] = key["uuid"]
    return entry


def _tcp_endpoint(host: str, port: int) -> str:
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"tcp://{host}:{port}"


@dataclass
class _Subscriber:
    """One ZeroMQ SUB socket on one publisher, and where its sequence numbers got to."""

    source: KvEventSource
    socket: Any
    last_seq: int | None = None


class _KvEventSubscription:
    """The frontend's subscription to the engine's KV-cache event publishers.

    Subscribes at the first call and keeps the received batches in the sockets
    (up to :data:`KV_EVENT_RECEIVE_HWM` per publisher) until a call reads them.
    The subscription lasts as long as the frontend process.
    """

    def __init__(self) -> None:
        self._subscribers: list[_Subscriber] | None = None
        self._lock: asyncio.Lock | None = None
        self._next_event_id = 0
        self._hash_algo: str | None = None
        self._dp_rank_known = False
        # Why the engine's events cannot be read, once that is known.
        self._unsupported: str | None = None

    async def events(self, client: EngineClient, timeout: float | None) -> list[dict]:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._unsupported is not None:
                # A new error each time: raising one again would grow its traceback.
                raise OpenEngineUnsupportedError(self._unsupported)
            if self._subscribers is None:
                self._subscribers = await self._subscribe(client)
                if self._subscribers is None:
                    return []
            entries = self._drain()
            if not entries and timeout:
                deadline = time.monotonic() + timeout
                while not entries and time.monotonic() < deadline:
                    await asyncio.sleep(_KV_EVENT_POLL_SECONDS)
                    entries = self._drain()
            return entries

    async def _subscribe(self, client: EngineClient) -> list[_Subscriber] | None:
        """Subscribe to every publisher; None when that cannot be done now.

        Raises:
            OpenEngineUnsupportedError: The engine publishes no KV-cache events
                in a form the frontend can read.
        """
        engine = client.server_info()
        try:
            sources = list(await client.kv_event_sources())
        except OpenEngineClientError as error:
            # As /metrics: no entries rather than an error while the engine cannot answer.
            logger.warning(
                f"Could not read the KV-cache event sources of the OpenEngine engine "
                f"{engine.engine_name!r}: {error}"
            )
            return None
        reason = None
        if not sources:
            reason = f"the OpenEngine engine {engine.engine_name!r} publishes no KV-cache events"
        for source in sources:
            unsupported = (
                source.transport != "zmq"
                or source.encoding not in ("", "msgpack")
                or source.protocol not in ("", "tcp")
            )
            if unsupported:
                reason = (
                    f"the OpenEngine engine {engine.engine_name!r} publishes KV-cache events over "
                    f"{source.transport}/{source.protocol or '-'} in {source.encoding or '-'}; "
                    "only zmq/tcp in msgpack is supported"
                )
                break
        if reason is not None:
            # The 501 response names only the endpoint: the reason goes to the log, once.
            logger.warning(f"/kv_cache_events is not supported: {reason}")
            self._unsupported = reason
            raise OpenEngineUnsupportedError(reason)
        import zmq

        self._hash_algo = engine.extra.get(KV_EVENT_HASH_ALGO_KEY)
        # In-process, an event names its attention-DP rank when attention DP is
        # on; the sources tell only whether there are several ranks.
        self._dp_rank_known = len(sources) > 1
        context = zmq.Context.instance()
        subscribers: list[_Subscriber] = []
        try:
            for source in sources:
                sock = context.socket(zmq.SUB)
                subscribers.append(_Subscriber(source, sock))
                sock.setsockopt(zmq.LINGER, 0)
                # A publisher may name an IPv6 address.
                sock.setsockopt(zmq.IPV6, 1)
                sock.setsockopt(zmq.RCVHWM, KV_EVENT_RECEIVE_HWM)
                sock.setsockopt(zmq.SUBSCRIBE, source.topic.encode("utf-8"))
                sock.connect(_tcp_endpoint(source.host, source.port))
        except zmq.ZMQError as error:
            for subscriber in subscribers:
                subscriber.socket.close()
            logger.warning(
                f"Could not subscribe to the KV-cache events of the OpenEngine engine "
                f"{engine.engine_name!r}: {error}"
            )
            return None
        logger.info(
            f"Subscribed to the KV-cache events of the OpenEngine engine {engine.engine_name!r} "
            f"({len(subscribers)} publisher(s))"
        )
        return subscribers

    def _drain(self) -> list[dict]:
        import msgspec
        import zmq

        entries: list[dict] = []
        assert self._subscribers is not None
        # Each publisher gets its share, so a busy one does not hold up the others.
        share = max(1, KV_EVENT_MAX_BATCHES // len(self._subscribers))
        for subscriber in self._subscribers:
            for _ in range(share):
                try:
                    frames = subscriber.socket.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if len(frames) != 3:
                    logger.warning(f"Ignoring a KV-cache event message of {len(frames)} frames")
                    continue
                seq = int.from_bytes(frames[1], "big")
                if subscriber.last_seq is not None and seq != subscriber.last_seq + 1:
                    logger.warning(
                        f"KV-cache events of data-parallel rank "
                        f"{subscriber.source.data_parallel_rank} skipped from batch "
                        f"{subscriber.last_seq + 1} to {seq}"
                    )
                subscriber.last_seq = seq
                try:
                    batch = msgspec.msgpack.decode(frames[2])
                except (msgspec.DecodeError, RecursionError) as error:
                    logger.warning(f"Ignoring an undecodable KV-cache event batch: {error}")
                    continue
                try:
                    rank = None
                    if self._dp_rank_known:
                        rank = batch[2] if len(batch) > 2 else subscriber.source.data_parallel_rank
                    converted = to_in_process_events(
                        batch,
                        first_event_id=self._next_event_id,
                        hash_algo=self._hash_algo,
                        attention_dp_rank=rank,
                    )
                    # The response renders the entries as strict JSON: a batch
                    # holding bytes or a non-finite number would fail all of them.
                    json.dumps(converted, allow_nan=False)
                except (
                    TypeError,
                    ValueError,
                    KeyError,
                    AttributeError,
                    IndexError,
                    OverflowError,
                    RecursionError,
                ) as error:
                    logger.warning(f"Ignoring a malformed KV-cache event batch: {error}")
                    continue
                self._next_event_id += len(converted)
                entries.extend(converted)
        return entries

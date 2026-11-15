# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay context-first handoffs directly between two OpenEngine engines, many at a time.

The load version of :mod:`openengine_chain.direct_replay`: ``concurrency``
requests are in flight at any time until ``num_requests`` have run. Each sends a
``context_only`` Generate (one output token) to the context engine and, with the
``KvSessionRef`` of its ``PrefillReady``, a Generate to the generation engine;
no ``trtllm-serve`` front end and no orchestrator are involved. The prompts are
random token ids of ``input_len`` tokens; the generation leg asks for
``output_len`` tokens and ignores EOS.

The latencies are taken as a client of the disaggregated server would see them:
TTFT from sending the context leg to the generation leg's first token event,
ITL between the generation leg's token events, TPOT the mean time per output
token after the first, E2EL from sending the context leg to the end of the
generation leg. Both engines' ``trtllm.counters.*`` are read before and after
the measured requests. :func:`build_result` makes the result document of the
``E4-direct`` arm (``perf/openengine_arms.py``, TH-DSG-6): the latencies, the
engine-side counts of each leg (the front-end counts are null) and per-request
records next to it.

Import it as ``openengine_chain.direct_replay_load`` with
``tests/integration/defs`` on the import path; run it as
``python -m openengine_chain.direct_replay_load --help`` from that directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import statistics
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict

from tensorrt_llm.grpc.openengine.bindings import kv_pb2, openengine_pb2_grpc

from . import direct_replay as dr

ARM = "E4-direct"
COUNTER_PREFIX = "trtllm.counters."
# The engine counters each leg's proof reads (tensorrt_llm/grpc/openengine_conventions.py).
GENERATE_RECEIVED = "trtllm.counters.generate_received"
ACCEPTED_CONTEXT_ONLY = "trtllm.counters.generate_accepted.context_only"
ACCEPTED_GENERATION_ONLY = "trtllm.counters.generate_accepted.generation_only"
ACCEPTED_AGGREGATED = "trtllm.counters.generate_accepted.aggregated"
PREFILL_READY_SENT = "trtllm.counters.prefill_ready_sent"


@dataclass(frozen=True)
class LoadConfig:
    """What to replay: the model name sent, prompt and output lengths, and how many requests."""

    model: str
    input_len: int
    output_len: int
    concurrency: int
    num_requests: int
    seed: int = 0
    max_token_id: int = 10000
    timeout: float = 600.0

    def __post_init__(self) -> None:
        for name in ("input_len", "output_len", "concurrency", "num_requests"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.max_token_id < 2:
            raise ValueError("max_token_id must be at least 2")


@dataclass
class RequestRecord:
    """One replayed request; times are seconds since the load started."""

    index: int
    request_id: str
    sent_s: float
    ok: bool = False
    error: str = ""
    ctx_status: str = ""
    gen_status: str = ""
    ctx_first_token_s: float | None = None
    prefill_ready_s: float | None = None
    token_event_s: list[float] = field(default_factory=list)
    output_tokens: int = 0
    end_s: float | None = None

    @property
    def ttft_s(self) -> float | None:
        return self.token_event_s[0] - self.sent_s if self.token_event_s else None

    @property
    def e2el_s(self) -> float | None:
        return None if self.end_s is None else self.end_s - self.sent_s

    def itl_s(self) -> list[float]:
        times = self.token_event_s
        return [later - earlier for earlier, later in zip(times, times[1:])]

    def tpot_s(self) -> float | None:
        if self.output_tokens < 2 or self.ttft_s is None or self.e2el_s is None:
            return None
        return (self.e2el_s - self.ttft_s) / (self.output_tokens - 1)

    def to_json(self) -> dict[str, Any]:
        doc = asdict(self)
        doc.update(ttft_s=self.ttft_s, e2el_s=self.e2el_s, tpot_s=self.tpot_s())
        return doc


def random_prompts(config: LoadConfig) -> list[list[int]]:
    """``num_requests`` prompts of ``input_len`` token ids in ``[1, max_token_id)``, from ``seed``."""
    rng = random.Random(config.seed)
    return [
        [rng.randrange(1, config.max_token_id) for _ in range(config.input_len)]
        for _ in range(config.num_requests)
    ]


def replay_one(
    ctx_stub: Any, gen_stub: Any, config: LoadConfig, prompt: Sequence[int], index: int, t0: float
) -> RequestRecord:
    """Send one request through both legs; never raises for what the engines answer."""
    request_id = uuid.uuid4().hex
    record = RequestRecord(index, request_id, time.monotonic() - t0)
    ctx_request = dr.build_request(
        config.model,
        token_ids=prompt,
        max_tokens=1,
        request_id=f"{request_id}-ctx",
        extra=dr.CONTEXT_ONLY_EXTRA,
        ignore_eos=True,
    )
    session = None
    try:
        for response in ctx_stub.Generate(ctx_request, timeout=config.timeout):
            now = time.monotonic() - t0
            kind = response.WhichOneof("event")
            if kind == "token" and record.ctx_first_token_s is None:
                record.ctx_first_token_s = now
            elif kind == "prefill_ready" and session is None:
                session = kv_pb2.KvSessionRef()
                session.CopyFrom(response.prefill_ready.kv_session)
                record.prefill_ready_s = now
            elif kind == "error":
                record.error = f"context leg: engine error {response.error.message!r}"
        record.ctx_status = "OK"
    except grpc.RpcError as error:
        record.ctx_status = error.code().name
        record.error = f"context leg: {error.details() or record.ctx_status}"
    if session is None:
        record.error = record.error or "the context leg returned no PrefillReady"
        record.end_s = time.monotonic() - t0
        return record
    gen_request = dr.build_request(
        config.model,
        token_ids=prompt,
        max_tokens=config.output_len,
        request_id=f"{request_id}-gen",
        session=session,
        ignore_eos=True,
    )
    finished = False
    try:
        for response in gen_stub.Generate(gen_request, timeout=config.timeout):
            now = time.monotonic() - t0
            kind = response.WhichOneof("event")
            if kind == "token":
                record.token_event_s.append(now)
                record.output_tokens += len(response.token.tokens)
            elif kind == "finished":
                reason = MessageToDict(response.finished, preserving_proto_field_name=True).get(
                    "reason"
                )
                finished = reason in dr._NORMAL_FINISH_REASONS
                if not finished:
                    record.error = record.error or f"generation leg: finished with reason {reason}"
            elif kind == "error":
                record.error = f"generation leg: engine error {response.error.message!r}"
        record.gen_status = "OK"
    except grpc.RpcError as error:
        record.gen_status = error.code().name
        record.error = f"generation leg: {error.details() or record.gen_status}"
    record.end_s = time.monotonic() - t0
    record.ok = finished and not record.error
    return record


def run_load(ctx: str, gen: str, config: LoadConfig) -> tuple[list[RequestRecord], float]:
    """Replay ``num_requests`` requests, ``concurrency`` at a time.

    Returns the records in sent order and the seconds from the first send to the last end.
    """
    prompts = random_prompts(config)
    with grpc.insecure_channel(ctx) as ctx_channel, grpc.insecure_channel(gen) as gen_channel:
        ctx_stub = openengine_pb2_grpc.InferenceStub(ctx_channel)
        gen_stub = openengine_pb2_grpc.InferenceStub(gen_channel)
        t0 = time.monotonic()
        with ThreadPoolExecutor(config.concurrency, thread_name_prefix="direct-replay") as pool:
            futures = [
                pool.submit(replay_one, ctx_stub, gen_stub, config, prompt, i, t0)
                for i, prompt in enumerate(prompts)
            ]
            records = [future.result() for future in futures]
    duration = max((r.end_s for r in records if r.end_s is not None), default=0.0)
    return sorted(records, key=lambda r: r.sent_s), duration


def read_counters(endpoint: str) -> dict[str, int]:
    """The engine's ``trtllm.counters.*`` of ``GetLoad``."""
    attributes = dr.get_load(endpoint).get("attributes") or {}
    return {
        key: int(value)
        for key, value in attributes.items()
        if key.startswith(COUNTER_PREFIX) and isinstance(value, (int, float))
    }


def counter_delta(before: Mapping[str, int], after: Mapping[str, int], key: str) -> int | None:
    if key not in before or key not in after:
        return None
    return after[key] - before[key]


def _stats_ms(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p99": None}
    ordered = sorted(v * 1000 for v in values)
    p99 = ordered[min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))]
    return {
        "mean": round(statistics.fmean(ordered), 3),
        "median": round(statistics.median(ordered), 3),
        "p99": round(p99, 3),
    }


def summarize(records: Sequence[RequestRecord], duration_s: float) -> tuple[dict[str, Any], int]:
    """The result's ``metrics`` over the requests that finished, and their output tokens."""
    done = [r for r in records if r.ok]
    output_tokens = sum(r.output_tokens for r in done)
    metrics = {
        "ttft_ms": _stats_ms([r.ttft_s for r in done if r.ttft_s is not None]),
        "tpot_ms": _stats_ms([t for r in done if (t := r.tpot_s()) is not None]),
        "itl_ms": _stats_ms([gap for r in done for gap in r.itl_s()]),
        "e2el_ms": _stats_ms([r.e2el_s for r in done if r.e2el_s is not None]),
        "output_tok_per_s": round(output_tokens / duration_s, 3) if duration_s > 0 else None,
        # The engines' step times are in their logs; the replay does not read them.
        "gen_device_step_ms": {"mean": None, "median": None},
    }
    return metrics, output_tokens


def leg_counts(
    before: Mapping[str, Mapping[str, int]], after: Mapping[str, Mapping[str, int]]
) -> dict[str, dict[str, int | None]]:
    """The engine-side counts of each leg over the measured requests; the front-end ones are null."""

    def delta(role: str, key: str) -> int | None:
        return counter_delta(before[role], after[role], key)

    return {
        "ctx": {
            "generate_sent": None,
            "generate_received": delta("ctx", GENERATE_RECEIVED),
            "accepted_context_only": delta("ctx", ACCEPTED_CONTEXT_ONLY),
            "prefill_ready_sent": delta("ctx", PREFILL_READY_SENT),
            "prefill_ready_received": None,
            "http_received": None,
            "http_rejected": None,
        },
        "gen": {
            "generate_sent": None,
            "generate_received": delta("gen", GENERATE_RECEIVED),
            "accepted_generation_only": delta("gen", ACCEPTED_GENERATION_ONLY),
            "accepted_aggregated": delta("gen", ACCEPTED_AGGREGATED),
            "http_received": None,
            "http_rejected": None,
        },
    }


def judge_counts(
    counts: Mapping[str, Mapping[str, int | None]], sent: int, handoffs: int
) -> tuple[dict, list]:
    """``status`` and ``details`` of the direct arm's proof: each leg's engine took every call sent.

    ``sent`` context legs were sent and ``handoffs`` of them returned a
    ``PrefillReady`` and sent a generation leg. A count the engine does not
    report leaves its leg ``incomplete``.
    """
    expected = {
        "ctx": {
            "generate_received": sent,
            "accepted_context_only": sent,
            "prefill_ready_sent": handoffs,
        },
        "gen": {
            "generate_received": handoffs,
            "accepted_generation_only": handoffs,
            "accepted_aggregated": 0,
        },
    }
    status: dict[str, str] = {}
    details: list[str] = []
    for role, wanted in expected.items():
        role_status = "ok"
        for key, value in wanted.items():
            got = counts[role][key]
            if got is None:
                role_status = "incomplete" if role_status == "ok" else role_status
                details.append(f"{role}.{key}: the engine does not report it")
            elif got != value:
                role_status = "fail"
                details.append(f"{role}.{key} = {got}, expected {value}")
        status[role] = role_status
    order = ("ok", "incomplete", "fail")
    status["overall"] = max(status.values(), key=order.index)
    return status, details


def _role_block(gpus: int | None, startup_s: float | None) -> dict[str, Any]:
    return {
        "chain": "trtllm",
        "instances": 1,
        "gpus_per_instance": gpus,
        "frontends": {"configured": 0, "started": 0},
        "postproc_workers": {"configured": 0, "started": 0},
        "stream_interval": None,
        "frontend_cpu_s": None,
        "frontend_cpu_us_per_token": None,
        "busiest_thread_cores": None,
        "startup_s": startup_s,
    }


def _arms() -> Any:
    """``perf/openengine_arms.py``: the arm, the schema and the writer of the result."""
    import importlib

    return importlib.import_module("perf.openengine_arms")


@dataclass
class LoadRun:
    """What one measured load left: its records and both engines' counters and server info."""

    config: LoadConfig
    records: list[RequestRecord]
    duration_s: float
    before: dict[str, dict[str, int]]
    after: dict[str, dict[str, int]]
    server_info: dict[str, dict[str, Any]]
    client_cpu_s: float | None = None


# How often, and how far apart, the counters are read after the load until two reads agree.
_SETTLE_READS = 5
_SETTLE_GAP_S = 0.5


def settled_counters(ctx: str, gen: str) -> dict[str, dict[str, int]]:
    """Both engines' counters, read until two reads in a row agree (or the reads run out)."""
    last = {"ctx": read_counters(ctx), "gen": read_counters(gen)}
    for _ in range(_SETTLE_READS - 1):
        time.sleep(_SETTLE_GAP_S)
        now = {"ctx": read_counters(ctx), "gen": read_counters(gen)}
        if now == last:
            break
        last = now
    return last


def measure(ctx: str, gen: str, config: LoadConfig) -> LoadRun:
    """Read both engines, run the load, read them again once their counters settle."""
    info = {"ctx": dr.server_info(ctx), "gen": dr.server_info(gen)}
    before = {"ctx": read_counters(ctx), "gen": read_counters(gen)}
    cpu = os.times()
    records, duration = run_load(ctx, gen, config)
    used = os.times()
    after = settled_counters(ctx, gen)
    client_cpu = round((used.user - cpu.user) + (used.system - cpu.system), 3)
    return LoadRun(config, records, duration, before, after, info, client_cpu)


def build_result(
    run: LoadRun,
    *,
    test_id: str,
    gpus: Mapping[str, int | None] | None = None,
    startup_s: Mapping[str, float | None] | None = None,
    kv_runtime: str | None = None,
    alignment: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The ``E4-direct`` result document of ``run`` (``perf/openengine_arms.py`` checks it).

    ``alignment`` sets the alignment items the replay cannot read from the
    engines; the rest are null, except ``input_form`` (token ids).
    """
    arms = _arms()
    env = os.environ if environ is None else environ
    arm = arms.DISAGG_ARMS[ARM]
    metrics, output_tokens = summarize(run.records, run.duration_s)
    handoffs = sum(1 for r in run.records if r.prefill_ready_s is not None)
    counts = leg_counts(run.before, run.after)
    status, details = judge_counts(counts, len(run.records), handoffs)
    failed = [r for r in run.records if not r.ok]
    notes = [f"{len(failed)} of {len(run.records)} requests did not finish"] if failed else []
    notes += sorted({r.error for r in failed if r.error})[:5]
    backend = {
        role: (info.get("kv_connector") or {}).get("transfer_backend")
        for role, info in run.server_info.items()
    }
    aligned = {key: None for key in arms.DISAGG_ALIGNMENT_KEYS}
    aligned["input_form"] = "token_ids"
    aligned.update(alignment or {})
    return {
        "case": arm.case,
        "arm": arm.name,
        "topology": arm.topology,
        "test_id": test_id,
        "benchmark_mode": "e2e",
        "concurrency": run.config.concurrency,
        "dataset": "token_ids",
        "run": arms.run_number(env),
        "metrics": metrics,
        "output_tokens": output_tokens,
        "roles": {
            role: _role_block((gpus or {}).get(role), (startup_s or {}).get(role))
            for role in ("ctx", "gen")
        },
        "orch": {
            "kind": "none",
            "schedule_style": "context_first",
            "router": None,
            "num_workers": None,
            "cpu_s": None,
            "config": {},
        },
        "client": {"cpu_s": run.client_cpu_s},
        "kv_transfer": {"backend": backend, "runtime": kv_runtime, "env": None},
        "chain_proof": {
            **counts,
            "orch_requests": {"ctx": None, "gen": None},
            "handoffs": handoffs,
            # A context leg without a generation leg leaves its KV cache held.
            "orphan_handoffs": sum(
                1 for r in run.records if r.prefill_ready_s is not None and not r.token_event_s
            ),
            "leg_expectation": None,
            "status": status,
            "details": details,
        },
        "first_output": None,
        "alignment": {key: aligned[key] for key in arms.DISAGG_ALIGNMENT_KEYS},
        "versions": {"trtllm": arms.trtllm_sha(env), "ds": None},
        "nodes": [socket.gethostname()],
        "job_id": env.get("SLURM_JOB_ID") or None,
        "notes": notes,
    }


def write_records(run: LoadRun, path: str) -> None:
    """The per-request records of ``run`` as JSON lines."""
    with open(path, "w", encoding="utf-8") as f:
        for record in run.records:
            f.write(json.dumps(record.to_json()) + "\n")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ctx", required=True, help="context engine host:port")
    parser.add_argument("--gen", required=True, help="generation engine host:port")
    parser.add_argument("--model", required=True, help="model name sent in the requests")
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--num-requests", type=int, default=160)
    parser.add_argument("--warmup-requests", type=int, default=0, help="run first, not measured")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-token-id", type=int, default=10000)
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per Generate call")
    parser.add_argument("--test-id", default="direct", help="test_id written to the result")
    parser.add_argument("--kv-runtime", default=None, help="the engines' transceiver_runtime")
    parser.add_argument("--ctx-gpus", type=int, default=None)
    parser.add_argument("--gen-gpus", type=int, default=None)
    parser.add_argument("--out-dir", required=True, help="where the result and the records go")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the load and write ``<out-dir>/<result file>`` and its ``.requests.jsonl``.

    Exit 0 if every measured request finished and the engines' counts match, else 1;
    invalid arguments exit 2.
    """
    args = _parse_args(argv)
    try:
        config = LoadConfig(
            args.model,
            args.input_len,
            args.output_len,
            args.concurrency,
            args.num_requests,
            seed=args.seed,
            max_token_id=args.max_token_id,
            timeout=args.timeout,
        )
    except ValueError as error:
        print(f"direct_replay_load: {error}", file=sys.stderr)
        return 2
    if args.warmup_requests > 0:
        warmup = LoadConfig(
            **{**asdict(config), "num_requests": args.warmup_requests, "seed": args.seed + 1}
        )
        run_load(args.ctx, args.gen, warmup)
    run = measure(args.ctx, args.gen, config)
    doc = build_result(
        run,
        test_id=args.test_id,
        gpus={"ctx": args.ctx_gpus, "gen": args.gen_gpus},
        kv_runtime=args.kv_runtime,
    )
    path = _arms().write_disagg_result(doc, args.out_dir)
    write_records(run, f"{os.path.splitext(path)[0]}.requests.jsonl")
    ok = all(r.ok for r in run.records) and doc["chain_proof"]["status"]["overall"] == "ok"
    print(json.dumps({"ok": ok, "result": path, "metrics": doc["metrics"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mechanical proof that requests went through the chain.

Each check returns ``ok``, ``fail`` or ``incomplete``. ``incomplete`` means the
check could not be made (for example a process does not report its counter
yet) and never fails a test by itself.

A server whose handle has a role (``"ctx"`` or ``"gen"``) is a worker of a
disaggregated server; :func:`judge_disagg` proves each leg of such a test with
the checks CH-1 to CH-6 from the per-bucket counters of the conventions'
version 4, the disaggregated server's metrics and its worker list.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
import weakref
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import yaml

from . import conventions as cv
from .launch import _descendants, _live_processes, _sockets_of, live_descendants, live_process_tree

OK = "ok"
FAIL = "fail"
INCOMPLETE = "incomplete"

_RPC_TIMEOUT_S = 5.0
_COUNTER_FILE_TRIES = 5
_COUNTER_FILE_RETRY_S = 0.1
# Set by trtllm-serve's launcher in the environment of each further front end it starts.
_FRONTEND_ID_ENV = b"TLLM_EXECUTOR_FRONTEND_ID="

# Version 4 counters. They repeat tensorrt_llm/grpc/openengine_conventions.py, so the
# harness imports without TensorRT-LLM; a harness self-test checks that they match.
CTX = "ctx"
GEN = "gen"
ROLES = (CTX, GEN)
BUCKET_CONTEXT_ONLY = "context_only"
BUCKET_GENERATION_ONLY = "generation_only"
BUCKET_AGGREGATED = "aggregated"
BUCKETS = (BUCKET_CONTEXT_ONLY, BUCKET_GENERATION_ONLY, BUCKET_AGGREGATED)
# The bucket of the calls each role's engine takes.
ROLE_BUCKET = {CTX: BUCKET_CONTEXT_ONLY, GEN: BUCKET_GENERATION_ONLY}
ENGINE_ACCEPTED = "trtllm.counters.generate_accepted"
ENGINE_ACCEPTED_BUCKETS = {b: f"{ENGINE_ACCEPTED}.{b}" for b in BUCKETS}
ENGINE_PREFILL_READY_SENT = "trtllm.counters.prefill_ready_sent"
ENGINE_V4_COUNTERS = (*ENGINE_ACCEPTED_BUCKETS.values(), ENGINE_PREFILL_READY_SENT)
SERVE_SENT_BUCKETS = {b: f"{cv.FRONTEND_COUNTER_FIELD}.{b}" for b in BUCKETS}
SERVE_PREFILL_READY_RECEIVED = "prefill_ready_received"
SERVE_HTTP_RECEIVED = {b: f"http_received.{b}" for b in BUCKETS}
SERVE_HTTP_REJECTED = "http_rejected"
SERVE_HANDOFF_UNREGISTERED = "handoff_unregistered_attributes"
SERVE_V4_COUNTERS = (
    *SERVE_SENT_BUCKETS.values(),
    SERVE_PREFILL_READY_RECEIVED,
    *SERVE_HTTP_RECEIVED.values(),
    SERVE_HTTP_REJECTED,
    SERVE_HANDOFF_UNREGISTERED,
)


@dataclass(frozen=True)
class Counters:
    """Generate calls counted so far; ``None`` where a side does not report its counter."""

    frontend: int | None
    engine: int | None
    # No front end had written a counter file: none had started serving yet, or
    # the front ends do not count.
    frontend_unwritten: bool = False
    # A worker's version 4 counters (read for handles with a role only): the front
    # ends' SERVE_V4_COUNTERS, summed, and the engine's ENGINE_V4_COUNTERS plus
    # ENGINE_ACCEPTED. None where a side does not report them.
    serve: Mapping[str, int] | None = None
    engine_v4: Mapping[str, int] | None = None


@dataclass(frozen=True)
class Check:
    status: str
    detail: str = ""


@dataclass(frozen=True)
class CountCheck(Check):
    # The engine's increase, or None if the engine does not report its counter.
    engine_calls: int | None = None
    # The two sides disagree; a call counted late can still settle it.
    mismatch: bool = False


# ----------------------------------------------------------------------------- counters


def read_frontend_counter(handle) -> int | None:
    """Sum of the front ends' ``generate_sent``; ``None`` if no front end has a counter file.

    A front end writes its file when it starts serving, so no file means that
    none has started yet or that the front ends do not count;
    :func:`counting_start` tells the two apart once a file appears. A running
    front end is asked for its current count at the address its file names; one
    that has closed its counter left its final count in the file. A missing
    directory reads as "no file"; any other read that fails raises, as does the
    read of a front end that exited without closing its counter.
    """
    files = _counter_files(handle)
    if files is None:
        return None
    return sum(_read_counter_file(path) for path in files)


def read_frontend_counts(handle) -> dict[str, int] | None:
    """The front ends' ``generate_sent`` and SERVE_V4_COUNTERS, each summed; as :func:`read_frontend_counter`.

    A front end publishes a counter once it has counted something, so a key
    missing from a front end's record counts 0 for it.
    """
    files = _counter_files(handle)
    if files is None:
        return None
    totals = dict.fromkeys((cv.FRONTEND_COUNTER_FIELD, *SERVE_V4_COUNTERS), 0)
    for path in files:
        record = _read_live_record(path)
        for key in totals:
            value = record.get(key, 0)
            if type(value) is not int or value < 0:
                raise ValueError(f"{path}: {key} is not a count: {value!r}")
            totals[key] += value
    return totals


def _counter_files(handle) -> list[str] | None:
    """The front ends' counter files, or ``None`` if there are none (yet)."""
    try:
        names = os.listdir(handle.counter_dir)
    except FileNotFoundError:
        # Created just before the front end starts: no front end has sent anything.
        return None
    files = sorted(os.path.join(handle.counter_dir, n) for n in names if n.endswith(".json"))
    return files or None


def _read_counter_file(path: str) -> int:
    """One front end's count."""
    record = _read_live_record(
        path, lambda address: {cv.FRONTEND_COUNTER_FIELD: _ask_frontend(address)}
    )
    return record[cv.FRONTEND_COUNTER_FIELD]


def _read_live_record(path: str, ask=None) -> dict:
    """One front end's counters: its answer while it runs, else the final ones in its file.

    ``ask`` asks the running front end; by default :func:`_ask_frontend_record`.
    """
    record = _read_counter_record(path)
    address = record.get(cv.FRONTEND_COUNTER_ADDRESS_FIELD)
    if address is None:
        return record
    try:
        return (ask or _ask_frontend_record)(address)
    except OSError as error:
        # A front end that closes its counter writes its final count before it stops answering.
        record = _read_counter_record(path)
        if cv.FRONTEND_COUNTER_ADDRESS_FIELD in record:
            raise OSError(
                f"front end pid {record['pid']} did not answer for its counter ({path}): {error!r}"
            ) from error
        return record


def _read_counter_record(path: str) -> dict:
    """The record in a counter file; a file caught while being rewritten is read again."""

    def read() -> dict:
        with open(path, encoding="utf-8") as f:
            return _counter_record(f.read())

    for _ in range(_COUNTER_FILE_TRIES - 1):
        try:
            return read()
        except ValueError:
            time.sleep(_COUNTER_FILE_RETRY_S)
    return read()


def _ask_frontend(address: str) -> int:
    return _ask_frontend_record(address)[cv.FRONTEND_COUNTER_FIELD]


def _ask_frontend_record(address: str) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(_RPC_TIMEOUT_S)
        connection.connect("\0" + address)
        chunks = []
        while chunk := connection.recv(4096):
            chunks.append(chunk)
    if not chunks:
        # The front end exited, or closed its counter, before it answered.
        raise ConnectionResetError(f"no answer from {address}")
    return _counter_record(b"".join(chunks).decode())


def _counter_record(text: str) -> dict:
    """``text`` as a counter record; ``ValueError`` if it is not one."""
    record = json.loads(text)
    if not (
        isinstance(record, dict)
        and type(record.get("pid")) is int
        and type(record.get(cv.FRONTEND_COUNTER_FIELD)) is int
        and isinstance(record.get(cv.FRONTEND_COUNTER_ADDRESS_FIELD, ""), str)
    ):
        raise ValueError(f"not a counter record: {text!r}")
    return record


def _control_stub(endpoint: str):
    import grpc

    from tensorrt_llm.grpc.openengine.bindings import openengine_pb2_grpc

    channel = grpc.insecure_channel(endpoint)
    return channel, openengine_pb2_grpc.ControlStub(channel)


def read_engine_counter(handle) -> int | None:
    """``GetLoad().attributes[trtllm.counters.generate_received]``, or ``None``."""
    import grpc

    from tensorrt_llm.grpc.openengine.bindings import server_pb2

    channel, stub = _control_stub(handle.engine_endpoint)
    with channel:
        try:
            load = stub.GetLoad(server_pb2.GetLoadRequest(), timeout=_RPC_TIMEOUT_S)
        except grpc.RpcError as error:
            if error.code() == grpc.StatusCode.UNIMPLEMENTED:
                return None
            raise
    fields = load.attributes.fields
    if cv.ENGINE_COUNTER_KEY not in fields:
        return None
    return int(fields[cv.ENGINE_COUNTER_KEY].number_value)


def engine_received(handle) -> int:
    """Generate calls the engine has received, for tests that count them; fails if not reported."""
    received = read_engine_counter(handle)
    if received is None:
        raise AssertionError("the engine does not report the Generate calls it received")
    return received


def _environ(pid: int) -> list[bytes]:
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            return f.read().split(b"\0")
    except OSError:
        return []


def frontend_count(handle, *, descendants_reader=live_descendants, environ_reader=_environ) -> int:
    """How many front ends serve the chain, from its running front-end processes.

    The server's front-end process is one; trtllm-serve's launcher starts each
    further front end below it with its own ``TLLM_EXECUTOR_FRONTEND_ID``, which
    that front end's own child processes inherit. A value the server's
    front-end process has itself came from its own environment and names no
    further front end. The command line does not settle the count, as a
    ``--set`` override wins over it.
    """
    if handle.frontend is None or handle.frontend.poll() is not None:
        raise AssertionError("the front end is not running, so its front ends cannot be counted")

    def ids_of(pid: int) -> set[bytes]:
        return {e for e in environ_reader(pid) if e.startswith(_FRONTEND_ID_ENV)}

    ids = set()
    for pid in descendants_reader(handle.frontend.pid):
        ids |= ids_of(pid)
    return 1 + len(ids - ids_of(handle.frontend.pid))


@dataclass(frozen=True)
class EngineInfo:
    """What CH-1 reads from an engine's ``GetServerInfo``."""

    instance_id: str | None
    # kv_connector.enabled; None if the engine reports no kv_connector.
    kv_connector_enabled: bool | None


# GetServerInfo results by handle, kept with the engine name read in the same call.
_ENGINE_INFO: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def read_engine_name(handle) -> str | None:
    """The engine's name; the first call also keeps its :class:`EngineInfo` (:func:`engine_info`)."""
    if handle._engine_name is not None and handle in _ENGINE_INFO:
        return handle._engine_name
    from tensorrt_llm.grpc.openengine.bindings import server_pb2

    channel, stub = _control_stub(handle.engine_endpoint)
    with channel:
        info = stub.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=_RPC_TIMEOUT_S)
    connector = info.kv_connector if info.HasField("kv_connector") else None
    _ENGINE_INFO[handle] = EngineInfo(
        instance_id=info.instance_id or None,
        kv_connector_enabled=(
            connector.enabled if connector is not None and connector.HasField("enabled") else None
        ),
    )
    handle._engine_name = info.engine_name or None
    return handle._engine_name


def engine_info(handle) -> EngineInfo | None:
    """The engine's :class:`EngineInfo` as read before, else read now if it runs; ``None`` if not read."""
    if handle not in _ENGINE_INFO and handle.alive:
        read_engine_name(handle)
    return _ENGINE_INFO.get(handle)


def read_engine_load(handle) -> dict[str, int] | None:
    """The engine's ``trtllm.counters.*`` attributes of ``GetLoad``; ``None`` if it has no ``GetLoad``."""
    import grpc

    from tensorrt_llm.grpc.openengine.bindings import server_pb2

    channel, stub = _control_stub(handle.engine_endpoint)
    with channel:
        try:
            load = stub.GetLoad(server_pb2.GetLoadRequest(), timeout=_RPC_TIMEOUT_S)
        except grpc.RpcError as error:
            if error.code() == grpc.StatusCode.UNIMPLEMENTED:
                return None
            raise
    return {
        key: int(value.number_value)
        for key, value in load.attributes.fields.items()
        if key.startswith("trtllm.counters.") and value.WhichOneof("kind") == "number_value"
    }


def read_counters(handle) -> Counters:
    """Both counters of a running chain; the values read at shutdown once it has stopped.

    For a handle with a role, also the version 4 counters of both sides
    (``serve``, ``engine_v4``). A front end that has sent calls but published
    no per-bucket count does not count them; neither does an engine whose
    ``GetLoad`` has none of ENGINE_V4_COUNTERS.
    """
    if not handle.alive:
        return handle.final_counters or Counters(None, None)
    if getattr(handle, "role", None) is None:
        frontend = read_frontend_counter(handle)
        return Counters(
            frontend=frontend,
            engine=read_engine_counter(handle),
            frontend_unwritten=frontend is None,
        )
    counts = read_frontend_counts(handle)
    load = read_engine_load(handle)
    serve = None
    if counts is not None and (
        counts[cv.FRONTEND_COUNTER_FIELD] == 0
        or any(counts[k] for k in SERVE_SENT_BUCKETS.values())
    ):
        serve = {k: counts[k] for k in SERVE_V4_COUNTERS}
    engine_v4 = None
    if load is not None and any(k in load for k in ENGINE_V4_COUNTERS):
        engine_v4 = {k: load.get(k, 0) for k in (ENGINE_ACCEPTED, *ENGINE_V4_COUNTERS)}
    return Counters(
        frontend=None if counts is None else counts[cv.FRONTEND_COUNTER_FIELD],
        engine=None if load is None else load.get(cv.ENGINE_COUNTER_KEY),
        frontend_unwritten=counts is None,
        serve=serve,
        engine_v4=engine_v4,
    )


def counting_start(before: Counters, after: Counters) -> Counters:
    """``before`` of one server, with an unwritten front-end counter read as 0 once ``after`` has one.

    A front end writes its counter file when it starts serving, before its first
    Generate call, so front ends that have written one since ``before`` count and
    had sent nothing then.
    """
    if before.frontend_unwritten and after.frontend is not None:
        serve = before.serve
        if serve is None and after.serve is not None:
            serve = dict.fromkeys(after.serve, 0)
        return dataclasses.replace(before, frontend=0, frontend_unwritten=False, serve=serve)
    return before


ZERO = Counters(0, 0)


def zero_like(after: Counters) -> Counters:
    """The counters of a server that started during the test: 0 on each side ``after`` reports."""
    return Counters(
        0,
        0,
        serve=None if after.serve is None else dict.fromkeys(after.serve, 0),
        engine_v4=None if after.engine_v4 is None else dict.fromkeys(after.engine_v4, 0),
    )


def check_counts(
    before: Counters, after: Counters, *, no_engine_traffic: bool, cancels_calls: bool = False
) -> CountCheck:
    """The front end's and the engine's increases must be equal, and non-zero unless allowed.

    With ``cancels_calls`` (a test listed as ``CANCELS_CALLS`` in an expectation
    table) the front end's increase may also exceed the engine's: the front end
    counts a call once it has started it, and a call cancelled before it reaches
    the engine never runs there. The engine's increase still may not exceed the
    front end's, and must still be non-zero unless allowed.
    """
    d_front = (
        None if None in (before.frontend, after.frontend) else after.frontend - before.frontend
    )
    d_engine = None if None in (before.engine, after.engine) else after.engine - before.engine
    calls = d_engine
    missing = [side for side, d in (("front end", d_front), ("engine", d_engine)) if d is None]
    if missing:
        return CountCheck(INCOMPLETE, f"no counter from the {' and the '.join(missing)}", calls)
    sides = f"front end sent {d_front} Generate calls, engine received {d_engine}"
    if d_front != d_engine and not (cancels_calls and d_front > d_engine):
        return CountCheck(FAIL, sides, calls, mismatch=True)
    if d_engine == 0 and not no_engine_traffic:
        return CountCheck(
            FAIL,
            "no Generate call reached the engine, and the test's expectations do not say "
            "it sends none",
            calls,
        )
    if d_front != d_engine:
        return CountCheck(OK, f"{sides}; accepted, as the test cancels calls", calls)
    return CountCheck(OK, f"{d_engine} Generate calls on both sides", calls)


def check_server_counts(
    starts: Sequence[Counters],
    ends: Sequence[Counters],
    *,
    no_engine_traffic: bool,
    cancels_calls: bool = False,
) -> CountCheck:
    """:func:`check_counts` over all servers, failing if any one server's two sides disagree.

    The total is incomplete as soon as one server lacks a counter; checking each
    server on its own keeps that from hiding a disagreement on another.
    """
    total = check_counts(
        add_counters(starts),
        add_counters(ends),
        no_engine_traffic=no_engine_traffic,
        cancels_calls=cancels_calls,
    )
    if total.mismatch:
        return total
    disagree = [
        check.detail
        for check in (
            check_counts(s, e, no_engine_traffic=True, cancels_calls=cancels_calls)
            for s, e in zip(starts, ends, strict=True)
        )
        if check.mismatch
    ]
    if disagree:
        return CountCheck(FAIL, "; ".join(disagree), total.engine_calls, mismatch=True)
    return total


def add_counters(items: Iterable[Counters]) -> Counters:
    front = engine = 0
    for c in items:
        front = None if front is None or c.frontend is None else front + c.frontend
        engine = None if engine is None or c.engine is None else engine + c.engine
    return Counters(front, engine)


# ----------------------------------------------------------------------------- topology


def _cmdline(pid: int) -> list[str] | None:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except OSError:
        return None
    # A zombie, or a process that is exiting, has an empty command line.
    return [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a] or None


def gpu_pids() -> set[int] | None:
    """PIDs nvidia-smi lists as compute apps, or ``None`` if it cannot be run."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {int(line) for line in out.split() if line.strip().isdigit()}


def check_topology(handle, *, gpu_pid_reader=gpu_pids, tree_reader=live_process_tree) -> Check:
    """The front end was started for the chain and none of its processes holds a GPU."""
    if handle.frontend is None:
        return Check(INCOMPLETE, "front end not started")
    args = _cmdline(handle.frontend.pid)
    if args is None:
        return Check(INCOMPLETE, "front end has exited; cannot read its command line")
    if "--openengine-endpoint" not in args:
        return Check(FAIL, f"front end command line lacks --openengine-endpoint: {args}")
    on_gpu = gpu_pid_reader()
    if on_gpu is None:
        return Check(INCOMPLETE, "nvidia-smi unavailable; GPU use of the front end not checked")
    frontend = tree_reader(handle.frontend.pid)
    if frontend & on_gpu:
        return Check(FAIL, f"front-end processes {sorted(frontend & on_gpu)} hold a GPU")
    # Positive control: an empty list would also show no front-end process.
    if not tree_reader(handle.engine.pid) & on_gpu:
        return Check(
            INCOMPLETE,
            "nvidia-smi lists no engine process either (other PID namespace?); "
            "GPU use of the front end not checked",
        )
    return Check(OK, "front end started with --openengine-endpoint and holds no GPU")


# ----------------------------------------------------------------------------- disaggregated servers

# Prometheus counters of the disaggregated server's client for each role
# (tensorrt_llm/serve/perf_metrics.py: "<ctx|gen>_<name>", exposed with "_total").
# total_requests counts each request once, when it is first sent; retry_requests
# counts the further attempts after a failure (not the resend of a request a worker's
# affinity port refused); requests forwarded by the control plane count in neither.
ORCH_METRICS = ("total_requests", "retry_requests", "error_requests", "completed_requests")
_ORCH_METRIC_LINE = re.compile(
    r"^(ctx|gen)_(" + "|".join(ORCH_METRICS) + r")_total(?:\{[^}]*\})?\s+(\S+)\s*$", re.MULTILINE
)
_HTTP_TIMEOUT_S = 5.0
# Serving programs by executable name, and by module name (``python -m``).
SERVE_PROGRAMS = ("trtllm-serve", "dsv4-serve")
SERVE_MODULES = ("tensorrt_llm.commands.serve", "dsv4_serve")


@dataclass(frozen=True)
class Orchestrator:
    """A running ``trtllm-serve disaggregated`` process and where it serves."""

    pid: int
    url: str
    config: str


@dataclass(frozen=True)
class OrchestratorReading:
    """The disaggregated servers' client counters (``"<role>.<name>"``) and worker lists.

    ``members`` maps a role to the worker addresses the servers route to, or is
    ``None`` if it could not be read. ``stale`` means a server stopped during
    the test, so its counters are those of its last reading before it stopped:
    a lower bound of what it sent.
    """

    counts: Mapping[str, int]
    members: Mapping[str, tuple[str, ...]] | None
    stale: bool = False
    note: str = ""


def _serve_program(args: Sequence[str]) -> int | None:
    """Index of the serving program (SERVE_PROGRAMS, SERVE_MODULES) in ``args``, if it runs one."""
    for i, token in enumerate(args):
        if os.path.basename(token) in SERVE_PROGRAMS:
            return i
        if token == "-m" and i + 1 < len(args) and args[i + 1] in SERVE_MODULES:
            return i + 1
    return None


def _option(args: Sequence[str], *names: str) -> str | None:
    """The value of the last of ``names`` in ``args`` (``--name value`` or ``--name=value``)."""
    value = None
    for i, token in enumerate(args):
        name, eq, rest = token.partition("=")
        if name in names:
            value = rest if eq else (args[i + 1] if i + 1 < len(args) else None)
    return value


def find_orchestrators(root: int, *, cmdline_reader=None) -> list[Orchestrator]:
    """The ``trtllm-serve disaggregated`` processes below ``root``, one per address."""
    cmdline_reader = cmdline_reader or _cmdline
    found: dict[str, Orchestrator] = {}
    for pid in sorted(live_descendants(root)):
        args = cmdline_reader(pid)
        if not args:
            continue
        program = _serve_program(args)
        if program is None or "disaggregated" not in args[program + 1 : program + 2]:
            continue
        path = _option(args, "-c", "--config_file")
        if not path:
            continue
        if not os.path.isabs(path):
            try:
                path = os.path.join(os.readlink(f"/proc/{pid}/cwd"), path)
            except OSError:
                continue
        try:
            with open(path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            continue
        host = str(config.get("hostname") or "localhost")
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
        port = config.get("port", 8000)
        url = f"http://{'[' + host + ']' if ':' in host else host}:{port}"
        found.setdefault(url, Orchestrator(pid, url, path))
    return list(found.values())


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_S) as response:
        return response.read()


def read_orchestrator(orchestrator: Orchestrator, *, getter=_get) -> OrchestratorReading:
    """One disaggregated server's counters and worker lists; raises if its metrics cannot be read."""
    text = getter(f"{orchestrator.url}/prometheus/metrics").decode("utf-8", "replace")
    counts = {f"{r}.{n}": 0 for r in ROLES for n in ORCH_METRICS}
    for role, name, value in _ORCH_METRIC_LINE.findall(text):
        counts[f"{role}.{name}"] += int(float(value))
    members = None
    try:
        lists = json.loads(getter(f"{orchestrator.url}/cluster_info"))["server_lists"]
        members = {
            CTX: tuple(str(a) for a in lists["context"]),
            GEN: tuple(str(a) for a in lists["generation"]),
        }
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return OrchestratorReading(counts, members)


def add_orchestrator_readings(readings: Iterable[OrchestratorReading]) -> OrchestratorReading:
    counts = {f"{r}.{n}": 0 for r in ROLES for n in ORCH_METRICS}
    members: dict[str, tuple[str, ...]] | None = {CTX: (), GEN: ()}
    stale = False
    notes = []
    for reading in readings:
        for key, value in reading.counts.items():
            counts[key] += value
        if reading.members is None or members is None:
            members = None
        else:
            members = {r: members[r] + tuple(reading.members.get(r, ())) for r in ROLES}
        stale = stale or reading.stale
        if reading.note:
            notes.append(reading.note)
    return OrchestratorReading(counts, members, stale, "; ".join(notes))


def inprocess_workers(root: int, *, gpu_pid_reader=None, table_reader=None, cmdline_reader=None):
    """Serving processes below ``root`` that hold a GPU without being a chain engine.

    For each process ``nvidia-smi`` lists, the nearest serving process
    (:func:`_serve_program`) at or above it in the process tree that is below
    ``root`` is the server it belongs to; that server must have been started
    with ``--grpc``. Returns a description of each server that was not, or
    ``None`` if ``nvidia-smi`` cannot be run or lists no process below ``root``
    (then it cannot see this test's processes, for example from another PID
    namespace).
    """
    on_gpu = (gpu_pid_reader or gpu_pids)()
    if on_gpu is None:
        return None
    table = (table_reader or _live_processes)()
    cmdline_reader = cmdline_reader or _cmdline
    below = set(_descendants(root, table))
    if not on_gpu & below:
        return None
    found = {}
    for pid in sorted(on_gpu):
        current = pid
        while current in below:
            args = cmdline_reader(current)
            if args and _serve_program(args) is not None:
                if "--grpc" not in args:
                    found[current] = f"pid {current} ({' '.join(args)[:300]}) holds a GPU"
                break
            current = table[current][0]
    return [found[p] for p in sorted(found)]


@functools.lru_cache(maxsize=1)
def _local_names() -> tuple[str, frozenset[str]]:
    """This host's short name, and every name and address that means this host."""
    short = socket.gethostname().lower().split(".")[0]
    names = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "::", short, socket.getfqdn().lower()}
    try:
        _, aliases, addresses = socket.gethostbyname_ex(socket.gethostname())
        names |= {a.lower() for a in aliases} | set(addresses)
    except OSError:
        pass
    return short, frozenset(names)


def listening_ports(pids: Iterable[int]) -> set[int]:
    """The TCP ports the processes ``pids`` listen on (those whose sockets this process can read)."""
    held = _sockets_of(set(pids))
    ports = set()
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(name, encoding="ascii") as f:
                lines = f.read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            # fields: sl, local address, remote address, state (0A = LISTEN), ..., inode.
            if len(fields) > 9 and fields[3] == "0A" and int(fields[9]) in held:
                ports.add(int(fields[1].rsplit(":", 1)[1], 16))
    return ports


def node_names() -> list[str]:
    """The names a worker address on this node may use, as :func:`normalize_address` writes hosts."""
    short, local = _local_names()
    return sorted(
        {short}
        | {
            n
            for n in local
            if re.fullmatch(r"[0-9.]+", n) and n != "0.0.0.0" and not n.startswith("127.")
        }
    )


def normalize_address(address: str) -> str:
    """``host:port`` of a worker, comparable across processes and nodes.

    A URL scheme and path are dropped; a name or address of this host becomes
    this host's short name, and any other host name its short name (an IP
    address of another host stays as it is).
    """
    address = re.sub(r"^[a-z]+://", "", address.strip().lower()).split("/", 1)[0]
    host, _, port = address.rpartition(":")
    host = host.strip("[]")
    short, local = _local_names()
    if host in local:
        host = short
    elif not re.fullmatch(r"[0-9.]+|[0-9a-f:]+", host):
        host = host.split(".")[0]
    return f"{host}:{port}"


# ----------------------------------------------------------------------------- per-leg proof

# A check that could not be made because the test killed the worker: explained, so it
# does not keep the test from counting as proven.
KILLED_INCOMPLETE = "incomplete(killed)"
_RANK = {OK: 0, KILLED_INCOMPLETE: 1, INCOMPLETE: 2, FAIL: 3}


@dataclass(frozen=True)
class Leg:
    """One worker of a disaggregated server during a test: its counters at both ends and more."""

    role: str
    # The worker front end's host:port as normalize_address writes it.
    address: str | None
    start: Counters
    end: Counters
    info: EngineInfo | None
    topology: Check


@dataclass(frozen=True)
class LegRule:
    """How the test's expectations change the checks (see expectations.LEG_VALUES)."""

    leg: str | None = None
    killed: tuple[str, ...] = ()
    orphans: int | None = None
    direct_requests: int | None = None
    no_engine_traffic: bool = False
    cancels_calls: bool = False


@dataclass
class DisaggProof:
    status: str
    # The status of the checks of each role's legs, and of the checks of both roles.
    roles: dict[str, str]
    # (check, role or None for both, result)
    checks: list[tuple[str, str | None, Check]]
    # Per-role sums and the disaggregated server's reading, for the junit file.
    counts: dict
    # The cross_node checks that need the other node's readings were left to the runner.
    deferred: bool = False

    def details(self) -> list[str]:
        return [
            f"{name}{'' if role is None else ' ' + role}: {c.status}: {c.detail}"
            for name, role, c in self.checks
        ]


def worst(statuses: Iterable[str]) -> str:
    statuses = list(statuses)
    return max(statuses, key=_RANK.__getitem__) if statuses else INCOMPLETE


def _diff(start: Mapping[str, int] | None, end: Mapping[str, int] | None, key: str) -> int | None:
    if start is None or end is None or key not in start or key not in end:
        return None
    return end[key] - start[key]


def _sub(a: int | None, b: int | None) -> int | None:
    return None if a is None or b is None else a - b


def leg_deltas(leg: Leg) -> dict[str, int | None]:
    """The increases of one leg's counters during the test.

    ``sent`` and ``received`` are the Generate calls on the two sides; every
    version 4 counter is under its key.
    """
    deltas = {
        "sent": _sub(leg.end.frontend, leg.start.frontend),
        "received": _sub(leg.end.engine, leg.start.engine),
    }
    for key in SERVE_V4_COUNTERS:
        deltas[key] = _diff(leg.start.serve, leg.end.serve, key)
    for key in (ENGINE_ACCEPTED, *ENGINE_V4_COUNTERS):
        deltas[key] = _diff(leg.start.engine_v4, leg.end.engine_v4, key)
    return deltas


def role_sums(legs: Sequence[Leg]) -> dict:
    """Per role: the number of legs, their summed increases and their addresses.

    A sum is ``None`` if the increase of one of the legs is unknown.
    """
    sums: dict = {}
    for role in ROLES:
        mine = [leg_deltas(leg) for leg in legs if leg.role == role]
        totals: dict[str, int | None] = {}
        for key in ("sent", "received", *SERVE_V4_COUNTERS, ENGINE_ACCEPTED, *ENGINE_V4_COUNTERS):
            values = [d[key] for d in mine]
            totals[key] = None if None in values else sum(values)
        sums[role] = {
            "legs": len(mine),
            "addresses": sorted(leg.address for leg in legs if leg.role == role and leg.address),
            "unaddressed": sum(1 for leg in legs if leg.role == role and not leg.address),
            "counts": totals,
        }
    return sums


def _role_buckets(role: str, rule: LegRule) -> tuple[str, ...]:
    """The engine buckets a ``role`` worker may take calls in under ``rule``.

    A directly addressed worker may take aggregated calls; so may the generation
    workers under ``ctx_zero_ok`` and the context side of a ``single_side``
    (``ctx_only``) benchmark, which is judged as aggregated.
    """
    if rule.leg == "direct_worker" or (role, rule.leg) in (
        (GEN, "ctx_zero_ok"),
        (CTX, "single_side"),
    ):
        return (ROLE_BUCKET[role], BUCKET_AGGREGATED)
    return (ROLE_BUCKET[role],)


def _needed_roles(rule: LegRule, present: set[str]) -> tuple[str, ...]:
    """The roles whose engines must take at least one call under ``rule``."""
    if rule.no_engine_traffic or rule.leg == "direct_worker":
        return ()
    needed = {
        "gen_zero_ok": {CTX},
        "ctx_zero_ok": {GEN},
        "single_side": present,
        "direct_ctx_leg": {CTX},
    }.get(rule.leg, set(ROLES))
    return tuple(r for r in ROLES if r in needed and r not in rule.killed)


def _judged_legs(legs: Sequence[Leg], rule: LegRule) -> list[Leg]:
    """The legs whose own checks are made.

    They are CH-1, CH-2, a leg's CH-3, a context leg's CH-4 and CH-5's second half.
    """
    if rule.leg == "direct_ctx_leg":
        return [leg for leg in legs if leg.role == CTX]
    if rule.leg == "direct_worker":
        sent_to = [
            leg
            for leg in legs
            if any(leg_deltas(leg)[k] for k in ("sent", "received", *SERVE_HTTP_RECEIVED.values()))
        ]
        return sent_to or list(legs)
    return list(legs)


def judge_disagg(
    legs: Sequence[Leg],
    orchestrator: OrchestratorReading | None,
    inprocess: Sequence[str] | None,
    rule: LegRule,
    last: bool = True,
) -> DisaggProof:
    """CH-1 to CH-6 for the workers of a disaggregated server during one test.

    ``orchestrator`` is the increase of the disaggregated servers' counters
    during the test with their worker lists (``None`` if none was found);
    ``inprocess`` is :func:`inprocess_workers` (``None`` if not checked). With
    ``rule.leg == "cross_node"`` the checks that need both nodes are left to
    :func:`judge_cross_node`. A test that cancels calls is held to equal
    Generate counts (CH-2) until the ``last`` read, so that an engine count
    arriving during the re-reads is still this test's.
    """
    checks: list[tuple[str, str | None, Check]] = []

    def add(name: str, role: str | None, status: str, detail: str) -> None:
        checks.append((name, role, Check(status, detail)))

    judged = _judged_legs(legs, rule)
    killed = set(rule.killed)
    cross = rule.leg == "cross_node"
    # CH-1: each worker is a front end without a GPU and an engine with one, with KV transfer.
    ids: dict[str, str] = {}
    for leg in judged:
        where = f"{leg.address or '?'}"
        statuses = [(leg.topology.status, leg.topology.detail)]
        if leg.info is None:
            statuses.append((INCOMPLETE, "the engine's GetServerInfo was not read"))
        else:
            if leg.info.kv_connector_enabled is not True:
                statuses.append((FAIL, "the engine reports no enabled kv_connector"))
            if leg.info.instance_id is None:
                statuses.append((INCOMPLETE, "the engine reports no instance_id"))
            elif leg.info.instance_id in ids:
                statuses.append(
                    (
                        FAIL,
                        f"instance_id {leg.info.instance_id} also at {ids[leg.info.instance_id]}",
                    )
                )
            else:
                ids[leg.info.instance_id] = where
        status = worst(s for s, _ in statuses)
        add(
            "CH-1",
            leg.role,
            status,
            f"{where}: " + "; ".join(d for s, d in statuses if s == status),
        )
    for leg in judged:
        deltas = leg_deltas(leg)
        where = f"{leg.address or '?'}"
        if leg.role in killed:
            add("CH-2", leg.role, KILLED_INCOMPLETE, f"{where}: the test killed this role")
            add("CH-5", leg.role, KILLED_INCOMPLETE, f"{where}: the test killed this role")
            continue
        # CH-2: the front end's Generate calls all reached its own engine.
        sent, received = deltas["sent"], deltas["received"]
        if sent is None or received is None:
            add("CH-2", leg.role, INCOMPLETE, f"{where}: a side reports no Generate counter")
        elif sent == received or (rule.cancels_calls and last and sent > received):
            add("CH-2", leg.role, OK, f"{where}: front end sent {sent}, engine received {received}")
        else:
            add(
                "CH-2",
                leg.role,
                FAIL,
                f"{where}: front end sent {sent}, engine received {received}",
            )
        # CH-3 (this leg): the engine took only calls of its role's buckets.
        _check_leg_buckets(add, leg, deltas, rule)
        # CH-4 (a context leg): every PrefillReady the engine sent reached the front end.
        if leg.role == CTX and rule.leg not in ("direct_worker",):
            pair = (deltas[ENGINE_PREFILL_READY_SENT], deltas[SERVE_PREFILL_READY_RECEIVED])
            if None in pair:
                add("CH-4", CTX, INCOMPLETE, f"{where}: a side reports no PrefillReady counter")
            else:
                status = OK if pair[0] == pair[1] else FAIL
                add(
                    "CH-4",
                    CTX,
                    status,
                    f"{where}: engine sent {pair[0]} PrefillReady, front end received {pair[1]}",
                )
        # CH-5 (second half): every HTTP request the front end received became a call or a refusal.
        _check_frontend_requests(add, leg, deltas, rule)
    sums = role_sums(legs)
    if rule.leg not in ("direct_worker", "direct_two_leg", "direct_ctx_leg", "single_side"):
        # CH-6 (reverse): no server holds a GPU outside the chain.
        if inprocess is None:
            add(
                "CH-6",
                None,
                INCOMPLETE,
                "nvidia-smi unavailable; servers holding a GPU not checked",
            )
        elif inprocess:
            add("CH-6", None, FAIL, "servers outside the chain hold a GPU: " + "; ".join(inprocess))
        else:
            add("CH-6", None, OK, "every serving process holding a GPU is a chain engine")
    if not cross:
        checks.extend(judge_roles(sums, orchestrator, rule))
    else:
        # This node's role must still carry traffic.
        for role in sorted({leg.role for leg in legs}):
            _check_role_traffic(add, sums, role, rule)
    status = worst(c.status for _, _, c in checks)
    roles = {role: worst(c.status for _, r, c in checks if r in (role, None)) for role in ROLES}
    counts = {
        "roles": sums,
        # What judge_cross_node needs of the test's expectations.
        "rule": {"cancels_calls": rule.cancels_calls, "no_engine_traffic": rule.no_engine_traffic},
        "orchestrator": None
        if orchestrator is None
        else {
            "counts": dict(orchestrator.counts),
            "members": None
            if orchestrator.members is None
            else {r: list(m) for r, m in orchestrator.members.items()},
            "stale": orchestrator.stale,
        },
    }
    return DisaggProof(_settled(status), roles, checks, counts, deferred=cross)


def _settled(status: str) -> str:
    """The test's status: explained gaps alone leave it proven."""
    return OK if status == KILLED_INCOMPLETE else status


def _check_leg_buckets(add, leg: Leg, deltas: Mapping[str, int | None], rule: LegRule) -> None:
    where = f"{leg.address or '?'}"
    accepted = deltas[ENGINE_ACCEPTED]
    by_bucket = {b: deltas[ENGINE_ACCEPTED_BUCKETS[b]] for b in BUCKETS}
    if accepted is None or None in by_bucket.values():
        add("CH-3", leg.role, INCOMPLETE, f"{where}: the engine reports no per-bucket counter")
        return
    allowed = _role_buckets(leg.role, rule)
    others = {b: n for b, n in by_bucket.items() if b not in allowed and n}
    if others or sum(by_bucket[b] for b in allowed) != accepted:
        add(
            "CH-3",
            leg.role,
            FAIL,
            f"{where}: the {leg.role} engine accepted {accepted} calls: {by_bucket}",
        )
        return
    if rule.leg == "direct_worker":
        # The aggregated calls the test's requests became: as registered, else as the front end
        # sent them, provided each request it received became a call or a refusal.
        expected = rule.direct_requests
        if expected is None:
            sent_agg = deltas[SERVE_SENT_BUCKETS[BUCKET_AGGREGATED]]
            http_agg = deltas[SERVE_HTTP_RECEIVED[BUCKET_AGGREGATED]]
            rejected = deltas[SERVE_HTTP_REJECTED]
            if None not in (sent_agg, http_agg, rejected) and sent_agg < http_agg - rejected:
                add(
                    "CH-3",
                    leg.role,
                    FAIL,
                    f"{where}: the front end received {http_agg} aggregated requests, refused "
                    f"{rejected} and sent {sent_agg} aggregated calls",
                )
                return
            expected = sent_agg
        if expected is None:
            add(
                "CH-3",
                leg.role,
                INCOMPLETE,
                f"{where}: the front end reports no per-bucket counter",
            )
            return
        if by_bucket[BUCKET_AGGREGATED] != expected:
            add(
                "CH-3",
                leg.role,
                FAIL,
                f"{where}: the engine accepted {by_bucket[BUCKET_AGGREGATED]} aggregated calls, "
                f"the test sent {expected}",
            )
            return
    add("CH-3", leg.role, OK, f"{where}: {by_bucket}")


def _check_frontend_requests(
    add, leg: Leg, deltas: Mapping[str, int | None], rule: LegRule
) -> None:
    """CH-5's second half: received HTTP requests = Generate calls sent + requests refused."""
    where = f"{leg.address or '?'}"
    keys = (
        *SERVE_HTTP_RECEIVED.values(),
        SERVE_HTTP_REJECTED,
        SERVE_SENT_BUCKETS[BUCKET_AGGREGATED],
    )
    if deltas["sent"] is None or any(deltas[k] is None for k in keys):
        add("CH-5", leg.role, INCOMPLETE, f"{where}: the front end reports no per-bucket counter")
        return
    received = sum(deltas[k] for k in SERVE_HTTP_RECEIVED.values())
    rejected = deltas[SERVE_HTTP_REJECTED]
    handled = deltas["sent"] + rejected
    sides = (
        f"{where}: {received} HTTP requests received, {deltas['sent']} calls sent, "
        f"{rejected} refused"
    )
    # An aggregated request may be split into one call per sequence (UV-52): then the other
    # buckets must still balance, each of their requests one call or one refusal, and each
    # aggregated request at least one call unless refused.
    http_agg = deltas[SERVE_HTTP_RECEIVED[BUCKET_AGGREGATED]]
    sent_agg = deltas[SERVE_SENT_BUCKETS[BUCKET_AGGREGATED]]
    split = sent_agg > http_agg
    if split and rule.cancels_calls:
        add("CH-5", leg.role, INCOMPLETE, f"{sides}; split requests and cancelled calls together")
        return
    if split:
        refused_other = (received - http_agg) - (deltas["sent"] - sent_agg)
        ok = 0 <= refused_other <= rejected and sent_agg >= http_agg - (rejected - refused_other)
    else:
        ok = received == handled or (rule.cancels_calls and received > handled)
    if ok:
        add("CH-5", leg.role, OK, sides + ("; aggregated requests were split" if split else ""))
    else:
        add("CH-5", leg.role, FAIL, sides)


def _check_role_traffic(add, sums: Mapping, role: str, rule: LegRule) -> None:
    """CH-3's sum: ``role``'s engines took at least one call of its buckets."""
    if role not in _needed_roles(rule, {r for r in ROLES if sums[r]["legs"]}):
        return
    if not sums[role]["legs"]:
        add("CH-3", role, FAIL, f"no {role} worker ran in the chain")
        return
    counts = sums[role]["counts"]
    values = [counts[ENGINE_ACCEPTED_BUCKETS[b]] for b in _role_buckets(role, rule)]
    if None in values:
        add("CH-3", role, INCOMPLETE, f"a {role} engine reports no per-bucket counter")
    elif sum(values) == 0:
        add(
            "CH-3",
            role,
            FAIL,
            f"no call reached a {role} engine, and the test's expectations do not allow it",
        )
    else:
        add("CH-3", role, OK, f"{sum(values)} calls reached the {role} engines")


def judge_roles(sums: Mapping, orchestrator: OrchestratorReading | None, rule: LegRule):
    """The checks over all legs of each role, as ``(check, role, result)`` items.

    They are CH-3's sums, CH-4's hand-offs, CH-5's first half and CH-6's worker list.
    """
    checks: list[tuple[str, str | None, Check]] = []

    def add(name: str, role: str | None, status: str, detail: str) -> None:
        checks.append((name, role, Check(status, detail)))

    for role in ROLES:
        _check_role_traffic(add, sums, role, rule)
    direct = rule.leg in ("direct_worker", "direct_two_leg", "direct_ctx_leg")
    ctx, gen = sums[CTX]["counts"], sums[GEN]["counts"]
    retries = {
        r: (orchestrator.counts.get(f"{r}.retry_requests", 0) if orchestrator else 0) for r in ROLES
    }
    # CH-4: the generation engines took no more hand-offs than the context engines sent.
    if rule.leg not in ("direct_worker", "single_side", "direct_two_leg") and not rule.killed:
        handed = ctx[ENGINE_PREFILL_READY_SENT]
        taken = gen[ENGINE_ACCEPTED_BUCKETS[BUCKET_GENERATION_ONLY]]
        if handed is None or taken is None:
            add("CH-4", None, INCOMPLETE, "an engine reports no hand-off counter")
        else:
            orphans = handed - taken
            sides = f"{handed} hand-offs sent, {taken} taken: {orphans} orphan hand-offs"
            if rule.leg == "direct_ctx_leg":
                expected = rule.orphans
                if expected is None:
                    expected = ctx[SERVE_HTTP_RECEIVED[BUCKET_CONTEXT_ONLY]]
                status = OK if expected is not None and orphans == expected else FAIL
                add("CH-4", None, status, f"{sides}; expected {expected}")
            elif orphans < 0:
                add("CH-4", None, FAIL, sides)
            elif orphans == 0:
                add("CH-4", None, OK, sides)
            elif rule.cancels_calls:
                add("CH-4", None, OK, f"{sides}; the test cancels calls")
            elif orphans <= retries[CTX]:
                add(
                    "CH-4",
                    None,
                    OK,
                    f"{sides}; the disaggregated server retried {retries[CTX]} times",
                )
            else:
                add("CH-4", None, FAIL, sides)
    # CH-5 (first half): the workers received what the disaggregated server sent each role.
    if not direct:
        for role in ROLES:
            if role in rule.killed or (rule.leg == "single_side" and not sums[role]["legs"]):
                continue
            buckets = (ROLE_BUCKET[role],) + ((BUCKET_AGGREGATED,) if role == GEN else ())
            got = [sums[role]["counts"][SERVE_HTTP_RECEIVED[b]] for b in buckets]
            if orchestrator is None:
                if rule.leg != "single_side":
                    # A ctx_only benchmark has no disaggregated server: only the second half holds.
                    add(
                        "CH-5",
                        role,
                        INCOMPLETE,
                        "no disaggregated server was found to compare with",
                    )
                continue
            if None in got:
                add("CH-5", role, INCOMPLETE, f"a {role} front end reports no per-bucket counter")
                continue
            got_total = sum(got)
            sent = orchestrator.counts.get(f"{role}.total_requests", 0)
            sides = (
                f"the disaggregated server sent {sent} {role} requests "
                f"(+{retries[role]} retries), the {role} workers received {got_total}"
            )
            if got_total < sent:
                add("CH-5", role, FAIL, sides)
            elif got_total <= sent + retries[role]:
                add("CH-5", role, OK, sides)
            elif orchestrator.stale:
                add(
                    "CH-5",
                    role,
                    INCOMPLETE,
                    f"{sides}; it stopped during the test, so the count is a lower bound",
                )
            else:
                add("CH-5", role, FAIL, sides)
    # CH-6 (members): the disaggregated server routes only to chain workers.
    if not direct and rule.leg != "single_side":
        if orchestrator is None or orchestrator.members is None:
            add("CH-6", None, INCOMPLETE, "the disaggregated server's worker list was not read")
        elif not any(orchestrator.members[r] for r in ROLES):
            add("CH-6", None, INCOMPLETE, "the disaggregated server lists no worker")
        else:
            chain = {a for r in ROLES for a in sums[r]["addresses"]}
            outside = sorted(
                {normalize_address(a) for r in ROLES for a in orchestrator.members[r]} - chain
            )
            unaddressed = sum(sums[r].get("unaddressed", 0) for r in ROLES)
            if outside and unaddressed:
                add(
                    "CH-6",
                    None,
                    INCOMPLETE,
                    f"the disaggregated server routes to {outside}; the address of {unaddressed} "
                    "chain worker(s) is unknown",
                )
            elif outside:
                add(
                    "CH-6",
                    None,
                    FAIL,
                    f"the disaggregated server routes to workers outside the chain: {outside}",
                )
            else:
                add(
                    "CH-6",
                    None,
                    OK,
                    f"the disaggregated server routes to chain workers only ({sorted(chain)})",
                )
    return checks


def judge_cross_node(nodes: Sequence[Mapping | None]) -> DisaggProof:
    """The cross_node checks over the per-node ``DisaggProof.counts`` of one test.

    Each node's counts carry ``node_names`` (:func:`node_names`), so that a
    worker address the disaggregated server lists by any of them is that
    node's worker, and the test's ``rule``. A missing node (``None``) leaves the
    result ``incomplete``. The result's ``counts`` holds the merged per-role sums.
    """
    if len(nodes) < 2 or any(n is None for n in nodes):
        missing = Check(INCOMPLETE, "the readings of a node are missing")
        return DisaggProof(
            INCOMPLETE, dict.fromkeys(ROLES, INCOMPLETE), [("cross_node", None, missing)], {}
        )
    sums = {}
    for role in ROLES:
        parts = [n["roles"][role] for n in nodes]
        counts = {}
        for key in parts[0]["counts"]:
            values = [p["counts"].get(key) for p in parts]
            counts[key] = None if None in values else sum(values)
        addresses = set()
        for node, part in zip(nodes, parts):
            for address in part["addresses"]:
                port = address.rpartition(":")[2]
                addresses |= {f"{name}:{port}" for name in node.get("node_names") or ()}
                addresses.add(address)
        sums[role] = {
            "legs": sum(p["legs"] for p in parts),
            "addresses": sorted(addresses),
            "unaddressed": sum(p.get("unaddressed", 0) for p in parts),
            "counts": counts,
        }
    readings = [n["orchestrator"] for n in nodes if n.get("orchestrator") is not None]
    orchestrator = None
    if readings:
        orchestrator = add_orchestrator_readings(
            OrchestratorReading(
                r["counts"],
                None
                if r["members"] is None
                else {role: tuple(m) for role, m in r["members"].items()},
                r["stale"],
            )
            for r in readings
        )
    rules = [n.get("rule") or {} for n in nodes]
    rule = LegRule(
        cancels_calls=any(r.get("cancels_calls", False) for r in rules),
        no_engine_traffic=all(r.get("no_engine_traffic", False) for r in rules),
    )
    checks = judge_roles(sums, orchestrator, rule)
    roles = {role: worst(c.status for _, r, c in checks if r in (role, None)) for role in ROLES}
    status = worst(c.status for _, _, c in checks)
    merged = {
        "roles": sums,
        "orchestrator": None if orchestrator is None else {"counts": dict(orchestrator.counts)},
    }
    return DisaggProof(_settled(status), roles, checks, merged)


NOT_HERE = "n/a"


def disagg_props(
    counts: Mapping, roles: Mapping[str, str], partial: bool = False
) -> dict[str, str]:
    """The per-role junit properties of a disaggregated test from ``DisaggProof.counts``.

    ``partial``: the counts are one node's of a cross_node test; a value that needs
    a role this node does not run is ``n/a``.
    """
    sums = counts["roles"]
    present = {r for r in ROLES if sums[r]["legs"]}
    orch = counts.get("orchestrator")

    def text(value) -> str:
        return INCOMPLETE if value is None else str(value)

    def of(needed: set[str], value) -> str:
        return NOT_HERE if partial and not needed <= present else text(value)

    handed = sums[CTX]["counts"][ENGINE_PREFILL_READY_SENT]
    taken = sums[GEN]["counts"][ENGINE_ACCEPTED_BUCKETS[BUCKET_GENERATION_ONLY]]
    props = {
        "chain_topology": "disagg",
        "chain_roles": ",".join(f"{r}:{sums[r]['legs']}" for r in ROLES),
        "handoffs": of({CTX}, handed),
        "orphan_handoffs": of(set(ROLES), _sub(handed, taken)),
    }
    for role in ROLES:
        props[f"engine_calls_{role}"] = of({role}, sums[role]["counts"]["received"])
        props[f"orch_requests_{role}"] = (
            NOT_HERE
            if partial and orch is None
            else text(orch and orch["counts"].get(f"{role}.total_requests"))
        )
        props[f"chain_proof_{role}"] = NOT_HERE if partial and role not in present else roles[role]
    return props

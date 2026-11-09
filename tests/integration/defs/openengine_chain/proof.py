# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mechanical proof that requests went through the chain.

Each check returns ``ok``, ``fail`` or ``incomplete``. ``incomplete`` means the
check could not be made (for example a process does not report its counter
yet) and never fails a test by itself.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from . import conventions as cv
from .launch import live_descendants, live_process_tree

OK = "ok"
FAIL = "fail"
INCOMPLETE = "incomplete"

_RPC_TIMEOUT_S = 5.0
_COUNTER_FILE_TRIES = 5
_COUNTER_FILE_RETRY_S = 0.1
# Set by trtllm-serve's launcher in the environment of each further front end it starts.
_FRONTEND_ID_ENV = b"TLLM_EXECUTOR_FRONTEND_ID="


@dataclass(frozen=True)
class Counters:
    """Generate calls counted so far; ``None`` where a side does not report its counter."""

    frontend: int | None
    engine: int | None
    # No front end had written a counter file: none had sent a Generate call yet,
    # or the front ends do not count.
    frontend_unwritten: bool = False


@dataclass(frozen=True)
class Check:
    status: str
    detail: str = ""


@dataclass(frozen=True)
class CountCheck(Check):
    # The engine's increase, or None if the engine does not report its counter.
    engine_calls: int | None = None
    # The two sides disagree; a counter written late can still settle it.
    mismatch: bool = False


# ----------------------------------------------------------------------------- counters


def read_frontend_counter(handle) -> int | None:
    """Sum of ``generate_sent`` over the front ends' counter files; ``None`` if there is none.

    A front end writes its file after each Generate call it sends, so no file
    means that none has sent one yet or that the front ends do not count;
    :func:`counting_start` tells the two apart once a file appears. A missing
    directory reads as "no file"; one that exists but cannot be listed raises.
    """
    try:
        names = os.listdir(handle.counter_dir)
    except FileNotFoundError:
        # Created just before the front end starts: no front end has sent anything.
        return None
    files = sorted(os.path.join(handle.counter_dir, n) for n in names if n.endswith(".json"))
    if not files:
        return None
    return sum(_read_counter_file(path) for path in files)


def _read_counter_file(path: str) -> int:
    """One front end's count; a file caught while being rewritten is read again."""

    def read() -> int:
        with open(path, encoding="utf-8") as f:
            return int(json.load(f)[cv.FRONTEND_COUNTER_FIELD])

    for _ in range(_COUNTER_FILE_TRIES - 1):
        try:
            return read()
        except ValueError:
            time.sleep(_COUNTER_FILE_RETRY_S)
    return read()


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


def read_engine_name(handle) -> str | None:
    if handle._engine_name is not None:
        return handle._engine_name
    from tensorrt_llm.grpc.openengine.bindings import server_pb2

    channel, stub = _control_stub(handle.engine_endpoint)
    with channel:
        info = stub.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=_RPC_TIMEOUT_S)
    handle._engine_name = info.engine_name or None
    return handle._engine_name


def read_counters(handle) -> Counters:
    """Both counters of a running chain; the values read at shutdown once it has stopped."""
    if not handle.alive:
        return handle.final_counters or Counters(None, None)
    frontend = read_frontend_counter(handle)
    return Counters(
        frontend=frontend,
        engine=read_engine_counter(handle),
        frontend_unwritten=frontend is None,
    )


def counting_start(before: Counters, after: Counters) -> Counters:
    """``before`` of one server, with an unwritten front-end counter read as 0 once ``after`` has one.

    Counter files appear with a front end's first Generate call, so front ends
    that have written one since ``before`` count and had sent nothing then.
    """
    if before.frontend_unwritten and after.frontend is not None:
        return dataclasses.replace(before, frontend=0, frontend_unwritten=False)
    return before


ZERO = Counters(0, 0)


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

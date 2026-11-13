# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay one context-first handoff directly between two OpenEngine engines.

A ``context_only`` Generate goes to the context engine. The ``KvSessionRef`` in its
``PrefillReady`` is then sent, unchanged unless the caller rewrites it, as
``kv.session`` of a Generate to the generation engine. No ``trtllm-serve`` front
end and no orchestrator are involved. Every event of both legs is kept, with its
arrival time, so a caller can check the first token, the usage and the KV transfer.

This is the single-request version of the tool. It also starts engines
(``trtllm-serve <model> --grpc --grpc-protocol openengine``), each in its own
session, so that stopping one ends its worker processes as well.

Import it as ``openengine_chain.direct_replay`` with ``tests/integration/defs`` on
the import path. Besides ``grpc``, ``grpc_health`` and ``protobuf`` it imports
TensorRT-LLM's OpenEngine bindings, which loads the ``tensorrt_llm`` package, and
the port check of ``openengine_chain.launch``; it starts and stops engines with its
own helpers, not with ``launch.py``'s server registry. Run it as
``python -m openengine_chain.direct_replay --help`` from that directory.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import grpc
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict
from grpc_health.v1 import health_pb2, health_pb2_grpc

from tensorrt_llm.grpc.openengine.bindings import (
    generation_pb2,
    kv_pb2,
    openengine_pb2_grpc,
    server_pb2,
)

from .launch import foreign_listeners

ENGINE_HOST = "127.0.0.1"
DEFAULT_LAUNCHER = ("trtllm-serve",)
# An engine with Prometheus enabled binds 8001 unless given a port; 0 is an ephemeral one,
# so that two engines on one host do not collide.
METRICS_PORT_OPTION = "--openengine-metrics-port"
# extra["request_type"] selects the context phase on a TensorRT-LLM OpenEngine server.
CONTEXT_ONLY_EXTRA = {"request_type": "context_only"}
# build_request arguments run_handoff sets the same on both legs.
_PROMPT_KEYS = frozenset({"model", "prompt", "token_ids"})
# The KV transfer CSV files written under TRTLLM_KVCACHE_TIME_OUTPUT_PATH.
KV_TASK_SEND = "KVSendTask"
KV_TASK_RECV = "KVRecvTask"
KV_SUMMARY_SUFFIX = "_gen_transfer_summary.csv"


class EngineStartError(RuntimeError):
    """An engine exited, or did not report SERVING, before the deadline."""


_HANDED_OUT_PORTS: set[int] = set()


def free_port() -> int:
    """A TCP port on ``ENGINE_HOST`` that was free when this returned.

    Never the same port twice in one process, so two engines started back to back
    get different ports. Another process may still take the port before the engine
    binds it; :meth:`Engine.wait_serving` fails if it sees such a listener.
    """
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((ENGINE_HOST, 0))
            port = sock.getsockname()[1]
        if port not in _HANDED_OUT_PORTS:
            _HANDED_OUT_PORTS.add(port)
            return port


@dataclass
class Engine:
    """One OpenEngine server process, the leader of its own session and process group."""

    name: str
    endpoint: str
    process: subprocess.Popen
    log_path: str
    config_path: str
    command: list[str]
    # (pid, start time) of every process seen under the leader, so that workers
    # that leave its session or process group are still stopped.
    tree: set[tuple[int, int]] = field(default_factory=set)

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def wait_serving(self, timeout: float, interval: float = 2.0) -> float:
        """Wait until this engine's gRPC health service ``inference`` reports SERVING.

        Each poll records the engine's processes (:meth:`remember_tree`) and uses a
        new channel. Returns the seconds waited. Raises :class:`EngineStartError` if
        the process exits first, the deadline passes, or, once SERVING is seen, the
        port also has a listener that is not one of the engine's processes.
        """
        start = time.monotonic()
        deadline = start + timeout
        port = int(self.endpoint.rsplit(":", 1)[1])
        while True:
            self.remember_tree()
            code = self.process.poll()
            if code is not None:
                raise EngineStartError(
                    f"engine {self.name} exited with code {code}; log {self.log_path}"
                )
            try:
                with grpc.insecure_channel(self.endpoint) as channel:
                    reply = health_pb2_grpc.HealthStub(channel).Check(
                        health_pb2.HealthCheckRequest(service="inference"), timeout=5
                    )
            except grpc.RpcError:
                reply = None
            if reply is not None and reply.status == health_pb2.HealthCheckResponse.SERVING:
                self.remember_tree()
                pids = {self.process.pid} | {pid for pid, _ in self.tree}
                foreign = foreign_listeners(port, pids)
                if foreign:
                    raise EngineStartError(
                        f"engine {self.name}: port {port} also has listening sockets "
                        f"{sorted(foreign)} of other processes; log {self.log_path}"
                    )
                return time.monotonic() - start
            if time.monotonic() > deadline:
                raise EngineStartError(
                    f"engine {self.name} at {self.endpoint} not SERVING within "
                    f"{timeout:.0f} s; log {self.log_path}"
                )
            time.sleep(interval)

    def remember_tree(self) -> None:
        """Record the processes under the leader while it runs.

        These are the leader's descendants and the members of its session. Nothing
        is recorded once the leader has exited, because its pid may then belong to
        another process; :meth:`stop` finds what is left of its session and process
        group then.
        """
        if self.process.poll() is not None:
            return
        table = _process_table()
        found = _descendants(self.process.pid, table) | {
            pid
            for pid, (_, _, session, _) in table.items()
            if session == self.process.pid and pid != self.process.pid
        }
        self.tree.update((pid, table[pid][3]) for pid in found if pid in table)

    def stop(self, grace: float = 30.0) -> int | None:
        """Stop the engine and everything it started; never raises.

        While the leader runs, SIGTERM goes to its process group, SIGKILL after
        ``grace`` seconds. Then, if the leader is reaped and no other process has
        taken its pid, every process still in its process group or session gets
        SIGKILL, and so does every recorded process (:meth:`remember_tree`) that is
        still the same process. Returns the leader's exit code (negative for a
        signal), or None if it could not be reaped.
        """
        self.remember_tree()
        leader = self.process.pid
        if self.process.poll() is None:
            _signal_group(leader, signal.SIGTERM)
            try:
                self.process.wait(grace)
            except subprocess.TimeoutExpired:
                _signal_group(leader, signal.SIGKILL)
                try:
                    self.process.wait(30)
                except subprocess.TimeoutExpired:
                    pass
        table = _process_table()
        # Linux does not give a new process a pid that is still some process's group
        # or session id. So once the leader is reaped, processes whose group or session
        # is the leader's pid are the engine's; if the leader is not reaped, or its pid
        # is taken, nothing is killed by that id.
        if self.process.returncode is not None and leader not in table:
            members = {
                pid for pid, (_, group, session, _) in table.items() if leader in (group, session)
            }
            if any(table[pid][1] == leader for pid in members):
                _signal_group(leader, signal.SIGKILL)
            for pid in members:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        for pid, started in self.tree:
            # A recorded pid is killed only while it is still the same process.
            if pid in table and table[pid][3] == started:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        return self.process.returncode


def _signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except (ProcessLookupError, PermissionError):
        pass


def _process_table() -> dict[int, tuple[int, int, int, int]]:
    """``pid -> (parent pid, process group, session, start time)`` for every live process."""
    table: dict[int, tuple[int, int, int, int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8", errors="replace") as handle:
                fields = handle.read().rsplit(")", 1)[1].split()
            # After the command name: state, ppid, pgrp, session, ..., starttime (field
            # 22 of the whole line).
            table[int(entry)] = (int(fields[1]), int(fields[2]), int(fields[3]), int(fields[19]))
        except (OSError, IndexError, ValueError):
            continue
    return table


def _descendants(root: int, table: Mapping[int, tuple[int, int, int, int]]) -> set[int]:
    children: dict[int, list[int]] = {}
    for pid, (parent, _, _, _) in table.items():
        children.setdefault(parent, []).append(pid)
    found: set[int] = set()
    pending = [root]
    while pending:
        for child in children.get(pending.pop(), []):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def start_engine(
    name: str,
    model: str,
    workdir: str,
    *,
    config: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
    extra_args: Sequence[str] = (),
    launcher: Sequence[str] = DEFAULT_LAUNCHER,
) -> Engine:
    """Start ``launcher <model> --grpc --grpc-protocol openengine`` on a free local port.

    ``config`` is written to ``<workdir>/<name>.yaml`` (as JSON, which YAML accepts)
    and passed with ``--config``. Unless ``extra_args`` gives
    ``--openengine-metrics-port``, the engine gets ``--openengine-metrics-port 0``.
    Output goes to ``<workdir>/<name>.log``. ``env`` defaults to this process's
    environment. The caller waits with :meth:`Engine.wait_serving` and must call
    :meth:`Engine.stop`.
    """
    os.makedirs(workdir, exist_ok=True)
    config_path = os.path.join(workdir, f"{name}.yaml")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(dict(config), handle, indent=2)
    port = free_port()
    command = [
        *launcher,
        model,
        "--grpc",
        "--grpc-protocol",
        "openengine",
        "--host",
        ENGINE_HOST,
        "--port",
        str(port),
        "--config",
        config_path,
        *extra_args,
    ]
    if not any(arg.split("=", 1)[0] == METRICS_PORT_OPTION for arg in extra_args):
        command += [METRICS_PORT_OPTION, "0"]
    log_path = os.path.join(workdir, f"{name}.log")
    with open(log_path, "ab") as log:
        process = subprocess.Popen(
            command,
            env=dict(os.environ if env is None else env),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return Engine(name, f"{ENGINE_HOST}:{port}", process, log_path, config_path, command)


@dataclass
class Leg:
    """Everything one Generate stream returned.

    ``events`` holds one dict per response, in arrival order: ``t`` (seconds since
    the call started, including connecting a new channel), ``kind`` (the response's
    ``event`` field, or ``None`` when it carries only usage) and ``body`` (the
    response as JSON). ``status`` is ``"OK"`` or the gRPC status code name the
    stream ended with.
    """

    request_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    details: str = ""
    elapsed_s: float = 0.0
    session: kv_pb2.KvSessionRef | None = None

    def record(self, response: generation_pb2.GenerateResponse, t: float) -> None:
        kind = response.WhichOneof("event")
        if kind == "prefill_ready" and self.session is None:
            self.session = kv_pb2.KvSessionRef()
            self.session.CopyFrom(response.prefill_ready.kv_session)
        body = MessageToDict(response, preserving_proto_field_name=True)
        self.events.append({"t": round(t, 6), "kind": kind, "body": body})

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [event["body"][kind] for event in self.events if event["kind"] == kind]

    def token_ids(self, output_index: int = 0) -> list[int]:
        """Token ids of one output sequence, concatenated over its ``token`` events."""
        ids: list[int] = []
        for token in self.of_kind("token"):
            if int(token.get("output_index", 0)) == output_index:
                ids.extend(int(info.get("token_id", 0)) for info in token.get("tokens", []))
        return ids

    def text(self, output_index: int = 0) -> str:
        return "".join(
            token.get("text", "")
            for token in self.of_kind("token")
            if int(token.get("output_index", 0)) == output_index
        )

    def first_event_s(self, kind: str) -> float | None:
        return next((event["t"] for event in self.events if event["kind"] == kind), None)

    def usage(self) -> dict[str, int] | None:
        """The last usage the stream reported, with integer values."""
        usages = [event["body"]["usage"] for event in self.events if "usage" in event["body"]]
        return {key: int(value) for key, value in usages[-1].items()} if usages else None

    def terminal(self) -> str:
        """How the stream ended: the last event's kind, or ``none`` if it sent no event."""
        kinds = [event["kind"] for event in self.events if event["kind"] is not None]
        return kinds[-1] if kinds else "none"

    def to_json(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "details": self.details,
            "elapsed_s": round(self.elapsed_s, 6),
            "terminal": self.terminal(),
            "token_ids": self.token_ids(),
            "text": self.text(),
            "usage": self.usage(),
            "events": self.events,
        }


def build_request(
    model: str,
    prompt: str | None = None,
    *,
    token_ids: Sequence[int] | None = None,
    max_tokens: int = 16,
    request_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
    session: kv_pb2.KvSessionRef | None = None,
    ignore_eos: bool = False,
    stop_token_ids: Sequence[int] = (),
    prompt_logprobs_top_n: int | None = None,
) -> generation_pb2.GenerateRequest:
    """A greedy GenerateRequest; give exactly one of ``prompt`` and non-empty ``token_ids``."""
    if (prompt is None) == (token_ids is None):
        raise ValueError("give exactly one of prompt and token_ids")
    if token_ids is not None and not token_ids:
        raise ValueError("token_ids must not be empty")
    request = generation_pb2.GenerateRequest(
        request_id=request_id or uuid.uuid4().hex,
        model=model,
        sampling=generation_pb2.SamplingParams(temperature=0.0),
        stopping=generation_pb2.StoppingOptions(
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            conditions=[generation_pb2.StopCondition(stop_token_id=t) for t in stop_token_ids],
        ),
    )
    if prompt is not None:
        request.prompt = prompt
    else:
        request.token_ids.ids.extend(token_ids)
    if prompt_logprobs_top_n is not None:
        request.response.return_prompt_logprobs = True
        request.response.prompt_candidates.top_n = prompt_logprobs_top_n
    if extra:
        struct = struct_pb2.Struct()
        struct.update(dict(extra))
        request.extra.CopyFrom(struct)
    if session is not None:
        request.kv.session.CopyFrom(session)
    return request


def generate(
    endpoint: str, request: generation_pb2.GenerateRequest, *, timeout: float = 120.0
) -> Leg:
    """Run one Generate stream to its end, or until ``timeout`` seconds pass."""
    leg = Leg(request.request_id)
    start = time.monotonic()
    with grpc.insecure_channel(endpoint) as channel:
        stub = openengine_pb2_grpc.InferenceStub(channel)
        try:
            for response in stub.Generate(request, timeout=timeout):
                leg.record(response, time.monotonic() - start)
        except grpc.RpcError as error:
            leg.status = error.code().name
            leg.details = error.details() or ""
    leg.elapsed_s = time.monotonic() - start
    return leg


def session_attributes(session: kv_pb2.KvSessionRef) -> dict[str, Any]:
    """``attributes_struct`` of a session as plain JSON values."""
    return MessageToDict(session.attributes_struct)


@dataclass
class Handoff:
    """The two legs of one replayed request.

    ``gen`` is ``None`` when the context leg returned no ``PrefillReady``.
    ``received`` is the session the context engine returned, ``sent`` the one the
    generation request carried (``None`` if it carried none).
    """

    ctx: Leg
    gen: Leg | None
    received: kv_pb2.KvSessionRef | None
    sent: kv_pb2.KvSessionRef | None

    def first_gen_tokens(self) -> list[int]:
        """The first token(s) the context engine handed over, from the session attributes."""
        if self.received is None:
            return []
        return [int(t) for t in session_attributes(self.received).get("first_gen_tokens", [])]

    def to_json(self) -> dict[str, Any]:
        def as_dict(session: kv_pb2.KvSessionRef | None) -> dict[str, Any] | None:
            if session is None:
                return None
            return MessageToDict(session, preserving_proto_field_name=True)

        return {
            "ctx": self.ctx.to_json(),
            "gen": None if self.gen is None else self.gen.to_json(),
            "session_received": as_dict(self.received),
            "session_sent": as_dict(self.sent),
            "first_gen_tokens": self.first_gen_tokens(),
        }


def run_handoff(
    ctx_endpoint: str,
    gen_endpoint: str,
    model: str,
    prompt: str | None = None,
    *,
    token_ids: Sequence[int] | None = None,
    max_tokens: int = 16,
    ctx_max_tokens: int = 1,
    timeout: float = 120.0,
    rewrite_session: Callable[[kv_pb2.KvSessionRef], None] | None = None,
    ctx_options: Mapping[str, Any] | None = None,
    gen_options: Mapping[str, Any] | None = None,
) -> Handoff:
    """Send one request through both engines and return both legs.

    The context leg asks for ``ctx_max_tokens`` tokens (default 1: a context-only
    request produces the first token only). ``rewrite_session`` may edit a copy of
    the returned session in place before it is sent. ``ctx_options`` /
    ``gen_options`` are further :func:`build_request` keyword arguments for one leg
    except ``model``, ``prompt`` and ``token_ids`` (both legs send the same ones);
    they override ``max_tokens``, ``request_id`` and, on the generation leg,
    ``session``. An ``extra`` on the context leg is merged over
    :data:`CONTEXT_ONLY_EXTRA`.
    """
    for options in (ctx_options, gen_options):
        if options and _PROMPT_KEYS & set(options):
            raise ValueError(f"leg options may not contain {sorted(_PROMPT_KEYS)}")
    request_id = uuid.uuid4().hex
    ctx_kwargs: dict[str, Any] = {
        "token_ids": token_ids,
        "max_tokens": ctx_max_tokens,
        "request_id": f"{request_id}-ctx",
    }
    ctx_kwargs.update(ctx_options or {})
    ctx_kwargs["extra"] = {**CONTEXT_ONLY_EXTRA, **(ctx_kwargs.get("extra") or {})}
    ctx_request = build_request(model, prompt, **ctx_kwargs)
    ctx = generate(ctx_endpoint, ctx_request, timeout=timeout)
    if ctx.session is None:
        return Handoff(ctx, None, None, None)
    sent = kv_pb2.KvSessionRef()
    sent.CopyFrom(ctx.session)
    if rewrite_session is not None:
        rewrite_session(sent)
    gen_kwargs: dict[str, Any] = {
        "token_ids": token_ids,
        "max_tokens": max_tokens,
        "request_id": f"{request_id}-gen",
        "session": sent,
    }
    gen_kwargs.update(gen_options or {})
    gen_request = build_request(model, prompt, **gen_kwargs)
    carried = None
    if gen_request.HasField("kv") and gen_request.kv.HasField("session"):
        carried = kv_pb2.KvSessionRef()
        carried.CopyFrom(gen_request.kv.session)
    gen = generate(gen_endpoint, gen_request, timeout=timeout)
    return Handoff(ctx, gen, ctx.session, carried)


def server_info(endpoint: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """``Control.GetServerInfo`` as JSON."""
    with grpc.insecure_channel(endpoint) as channel:
        stub = openengine_pb2_grpc.ControlStub(channel)
        reply = stub.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=timeout)
    return MessageToDict(reply, preserving_proto_field_name=True)


def get_load(endpoint: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """``Control.GetLoad`` (per rank) as JSON."""
    with grpc.insecure_channel(endpoint) as channel:
        stub = openengine_pb2_grpc.ControlStub(channel)
        reply = stub.GetLoad(server_pb2.GetLoadRequest(include_per_rank=True), timeout=timeout)
    return MessageToDict(reply, preserving_proto_field_name=True)


def kv_transfer_rows(directory: str) -> dict[str, list[dict[str, str]]]:
    """The rows of the KV transfer CSV files an engine wrote under ``directory``.

    Keys: ``summary`` (rows of ``*_gen_transfer_summary.csv``; the Python
    transceiver, ``transceiver_runtime: PYTHON``, puts the received size in
    ``kv_cache_size``), and from the other files ``send`` (``task_type``
    ``KVSendTask``, sizes in ``transfer_size_bytes``) and ``recv`` (``KVRecvTask``,
    size columns empty). Other rows are left out, and so is a row the engine is
    still writing (fewer fields than the header). Missing files give empty lists.
    """
    rows: dict[str, list[dict[str, str]]] = {"send": [], "recv": [], "summary": []}
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        with open(path, newline="", encoding="utf-8") as handle:
            reader = [row for row in csv.DictReader(handle) if None not in row.values()]
        if path.endswith(KV_SUMMARY_SUFFIX):
            rows["summary"].extend(reader)
            continue
        for row in reader:
            if row.get("task_type") == KV_TASK_SEND:
                rows["send"].append(row)
            elif row.get("task_type") == KV_TASK_RECV:
                rows["recv"].append(row)
    return rows


_NORMAL_FINISH_REASONS = ("FINISH_REASON_STOP", "FINISH_REASON_LENGTH")


def _token_ids(value: str) -> list[int]:
    try:
        ids = [int(item) for item in value.split(",") if item.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a list of integers: {value!r}") from None
    if not ids:
        raise argparse.ArgumentTypeError("no token ids given")
    return ids


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ctx", required=True, help="context engine host:port")
    parser.add_argument("--gen", required=True, help="generation engine host:port")
    parser.add_argument("--model", required=True, help="model name sent in the requests")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--token-ids", type=_token_ids, help="comma-separated prompt token ids")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", help="write the full result as JSON to this file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Replay one request.

    Exit 0 if both streams ended with status OK and the generation leg finished by
    stop or length, else 1.
    """
    args = _parse_args(argv)
    handoff = run_handoff(
        args.ctx,
        args.gen,
        args.model,
        args.prompt,
        token_ids=args.token_ids,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    result = handoff.to_json()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    gen = handoff.gen
    ok = (
        handoff.ctx.status == "OK"
        and gen is not None
        and gen.status == "OK"
        and gen.terminal() == "finished"
        and gen.of_kind("finished")[-1].get("reason") in _NORMAL_FINISH_REASONS
    )
    summary = {
        "ok": ok,
        "ctx": {"status": handoff.ctx.status, "terminal": handoff.ctx.terminal()},
        "gen": None if gen is None else {"status": gen.status, "terminal": gen.terminal()},
        "first_gen_tokens": handoff.first_gen_tokens(),
        "gen_token_ids": [] if gen is None else gen.token_ids(),
    }
    print(json.dumps(summary))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

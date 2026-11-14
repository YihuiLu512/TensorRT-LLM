# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start and stop an engine plus a ``trtllm-serve --openengine-endpoint`` front end.

The harness keeps no Python threads alive between calls: process output goes
through ``tee`` and ``tail`` processes, and the engine is polled with
synchronous gRPC.
Test configurations check for leaked threads after each test.

The harness's own settings (the chain, the results directory, the glm
command) come from ``os.environ``; an ``env`` argument is only the
environment of the processes it starts.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import importlib.util
import itertools
import os
import re
import select
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import IO, Any

import yaml

from . import conventions as cv
from .split import (
    ServeCommand,
    engine_yaml,
    frontend_env,
    iteration_stats_enabled,
    option_value,
    parse_serve_command,
    split_args,
)

ENGINE_HOST = "127.0.0.1"
# Functional tests' server-ready limit (RemoteOpenAIServer.MAX_SERVER_START_WAIT_S).
DEFAULT_ENGINE_READY_TIMEOUT_S = 7200.0
FRONTEND_READY_TIMEOUT_S = 60
TERMINATE_GRACE_S = 30.0
# How long processes killed at shutdown may take to disappear before they count as survivors.
SURVIVOR_WAIT_S = 5.0
LOG_TAIL_BYTES = 16384
# Engine starts per server: a start that lost the gRPC port the harness picked is retried on
# a new port.
ENGINE_PORT_ATTEMPTS = 3
# A line of a log saying that taking a port failed because the port is in use, such as
# gRPC's "Error in bind for address '[::ffff:127.0.0.1]:<port>': Address already in use",
# uvicorn's "[Errno 98] error while attempting to bind on address ('127.0.0.1', <port>): ..."
# or trtllm-serve's "Port <port> is already reserved by another multi-frontend trtllm-serve:
# [Errno 98] Address already in use".
_PORT_IN_USE_LINE = (
    r"^(?:(?=.*\bbind\b)(?=.*(?::|,\s*){port}\b)|(?=.*\bport {port} is already reserved\b))"
    r".*(?:address already in use|eaddrinuse|errno 98)"
)
# Given to a TensorRT-LLM engine whose configuration turns on iteration stats: the front
# end's iteration stats come from the engine's load, which reports KV-cache blocks only then.
ENGINE_LOAD_METRICS_FLAG = "--openengine-enable-load-metrics"
# The port of a TensorRT-LLM engine's Prometheus listener. Without it the engine binds
# 8001 once the model is loaded, whenever its configuration enables Prometheus, so two
# such engines in one network namespace collide. The harness gives 0, an ephemeral port
# the engine binds itself (launch_serve_command says when). An engine with Prometheus
# off does not bind it.
ENGINE_METRICS_PORT_OPTION = "--openengine-metrics-port"
# Set in the environment of a front end that only runs its startup checks (FrontendCheck).
# trtllm-serve does not read it; it tells the people and test doubles reading the records.
FRONTEND_CHECK_ENV = "TRTLLM_OPENENGINE_CHAIN_FRONTEND_CHECK"
# A line of a log saying that taking some port failed because the port is in use.
_ANY_PORT_IN_USE = re.compile(
    r"address already in use|eaddrinuse|errno 98|is already reserved", re.IGNORECASE
)


class ChainError(RuntimeError):
    """The harness could not set up the chain."""


class ChainStartupError(ChainError):
    """A chain process exited or did not become ready."""


class EnginePortTaken(ChainStartupError):
    """Another process listens on the engine's port too."""


def resolve_chain() -> str:
    """Return ``"none"``, ``"trtllm"`` or ``"glm"`` from ``TRTLLM_TEST_OPENENGINE_CHAIN``."""
    value = os.environ.get(cv.ENV_CHAIN, "")
    if value == "":
        return cv.CHAIN_NONE
    if value not in cv.CHAINS:
        raise ChainError(
            f"{cv.ENV_CHAIN}={value!r}; expected unset, {' or '.join(repr(c) for c in cv.CHAINS)}"
        )
    return value


def require_plugin() -> None:
    """Refuse to run a chain without the plugin that proves each test went through it."""
    if cv.PLUGIN_MODULE not in sys.modules:
        raise ChainError(
            f"{cv.ENV_CHAIN} is set but the pytest plugin {cv.PLUGIN_MODULE} is not loaded; run "
            "through openengine_chain/runner.py, or set PYTEST_ADDOPTS='-p "
            f"{cv.PLUGIN_MODULE}' and put a directory holding openengine_chain on PYTHONPATH"
        )


def free_port() -> int:
    """A port that was free a moment ago, as ``tensorrt_llm._utils.get_free_port`` picks one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((ENGINE_HOST, 0))
        return s.getsockname()[1]


def tail(path: str | None, limit: int = LOG_TAIL_BYTES) -> str:
    if not path:
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - limit))
            return f.read().decode("utf-8", "replace")
    except OSError as error:
        return f"<cannot read {path}: {error}>"


def _stop_waits(proc: subprocess.Popen, grace: float) -> None:
    """SIGTERM, wait ``grace`` seconds, SIGKILL, wait a bounded time; does not raise on a timeout."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            # A process stuck in the kernel can outlive SIGKILL; the caller checks poll().
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=SURVIVOR_WAIT_S)


@functools.lru_cache(maxsize=1)
def _serve_source() -> str:
    """The source of ``tensorrt_llm.commands.serve`` in this interpreter's installation.

    Read without importing that module (its parent packages are imported), or
    empty if ``tensorrt_llm`` has no such module; the chain's ``trtllm-serve``
    commands are taken to run from the same installation.
    """
    spec = importlib.util.find_spec("tensorrt_llm.commands.serve")
    if spec is None or spec.origin is None:
        return ""
    with open(spec.origin, encoding="utf-8") as f:
        return f.read()


def frontend_supports_ready_timeout() -> bool:
    """Whether the installed ``trtllm-serve`` declares ``--openengine-ready-timeout``."""
    return '"--openengine-ready-timeout"' in _serve_source()


def engine_supports_metrics_port() -> bool:
    """Whether the installed ``trtllm-serve``'s metrics port option has a lower bound of 0.

    Reads the lower bound of the ``IntRange`` the option is declared with from
    the parsed source, so line breaks and comments do not matter. An
    installation whose option takes only 1-65535 does not. A declaration it
    cannot read that way, such as one whose option name or type is held in a
    variable, counts as not having that bound, so the engine keeps its default port.
    """
    try:
        tree = ast.parse(_serve_source())
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == ENGINE_METRICS_PORT_OPTION
        ):
            continue
        bounds = {k.arg: k.value for k in node.keywords}.get("type")
        if isinstance(bounds, ast.Call):
            low = (
                bounds.args[0]
                if bounds.args
                else {k.arg: k.value for k in bounds.keywords}.get("min")
            )
            return isinstance(low, ast.Constant) and low.value == 0
    return False


def _live_processes() -> dict[int, tuple[int, int, int]]:
    """``{pid: (ppid, session, start time)}`` of the non-zombie processes.

    The start time (in clock ticks since boot) tells a process from a later one
    that reuses its PID.
    """
    table = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as f:
                stat = f.read()
        except OSError:
            continue
        # After the parenthesised command: fields 3 (state) onwards of proc(5).
        fields = stat[stat.rfind(b")") + 2 :].split()
        if len(fields) > 19 and fields[0] != b"Z":
            table[int(entry)] = (int(fields[1]), int(fields[3]), int(fields[19]))
    return table


def live_session_members(session: int) -> set[int]:
    """PIDs of the non-zombie processes in session ``session``."""
    return {pid for pid, (_, sid, _) in _live_processes().items() if sid == session}


def live_process_tree(root: int) -> set[int]:
    """PIDs of ``root``'s session and of the processes below ``root``."""
    return live_session_members(root) | live_descendants(root)


def live_descendants(root: int) -> set[int]:
    """PIDs of the non-zombie processes below ``root`` in the process tree."""
    return set(_descendants(root, _live_processes()))


def _descendants(root: int, table: dict[int, tuple[int, int, int]]) -> dict[int, int]:
    """``{pid: start time}`` of the processes below ``root`` in ``table``."""
    children: dict[int, list[int]] = {}
    for pid, (ppid, _, _) in table.items():
        children.setdefault(ppid, []).append(pid)
    found: set[int] = set()
    stack = [root]
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in found:
                found.add(child)
                stack.append(child)
    return {pid: table[pid][2] for pid in found}


def _listening_sockets(port: int) -> set[int]:
    """Inodes of the TCP sockets listening on ``port`` in this network namespace."""
    inodes = set()
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(name, encoding="ascii") as f:
                lines = f.read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            # fields: sl, local address, remote address, state (0A = LISTEN), ..., inode.
            if len(fields) > 9 and fields[3] == "0A":
                if int(fields[1].rsplit(":", 1)[1], 16) == port:
                    inodes.add(int(fields[9]))
    return inodes


def _sockets_of(pids: set[int]) -> set[int]:
    """Inodes of the sockets the processes ``pids`` hold open (those it can read)."""
    inodes = set()
    for pid in pids:
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if target.startswith("socket:["):
                inodes.add(int(target[len("socket:[") : -1]))
    return inodes


def foreign_listeners(port: int, pids: set[int]) -> set[int] | None:
    """Sockets listening on ``port`` that none of ``pids`` holds; ``None`` if none is seen at all.

    Two servers that both bind with SO_REUSEPORT (gRPC's default, kept by some
    engines) share a port without a bind error; a listener on another address
    of the same port does not stop a bind either.
    """
    listening = _listening_sockets(port)
    if not listening:
        return None
    return listening - _sockets_of(pids)


class LoggedProcess:
    """A process in its own session; stdout and stderr go to ``<role>-<pid>.log`` and to a sink.

    With ``defer_sink`` the output reaches the sink only from :meth:`attach_sink`
    (everything logged so far, then the rest as it comes) or :meth:`copy_to_sink`,
    so a start that is given up and retried leaves nothing in the sink.
    """

    def __init__(
        self,
        role: str,
        cmd: Sequence[str],
        env: Mapping[str, str],
        log_dir: str,
        sink: IO[Any] | int | None,
        *,
        defer_sink: bool = False,
    ) -> None:
        self.role = role
        self.cmd = list(cmd)
        self._sink = sink
        self.follower: subprocess.Popen | None = None
        # Things :meth:`stop` saw but did not act on: reported, not failed.
        self.notes: list[str] = []
        self._stopped = False
        os.makedirs(log_dir, exist_ok=True)
        pending = os.path.join(log_dir, f"{role}-starting-{os.getpid()}-{time.monotonic_ns()}.log")
        log_fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            # tee opens the log through its inherited descriptor, so renaming the path is safe.
            self.tee = subprocess.Popen(
                ["tee", "-a", f"/dev/fd/{log_fd}"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL if defer_sink else sink,
                pass_fds=(log_fd,),
                start_new_session=True,
            )
        finally:
            os.close(log_fd)
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=self.tee.stdin,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            self.tee.stdin.close()
            self.tee.wait()
            raise
        finally:
            # tee sees EOF once the process exits.
            self.tee.stdin.close()
        self.log_path = os.path.join(log_dir, f"{role}-{self.proc.pid}.log")
        # {pid: start time} of processes seen below this one.
        self._known: dict[int, int] = {}
        os.rename(pending, self.log_path)

    @property
    def pid(self) -> int:
        return self.proc.pid

    def poll(self) -> int | None:
        return self.proc.poll()

    def attach_sink(self) -> None:
        """Send the deferred output to the sink: the log so far, then the rest until tee exits."""
        if self.follower is None:
            self.follower = subprocess.Popen(
                ["tail", "-c", "+1", "-f", f"--pid={self.tee.pid}", self.log_path],
                stdin=subprocess.DEVNULL,
                stdout=self._sink,
                start_new_session=True,
            )

    def copy_to_sink(self) -> None:
        """Copy the whole log to the sink, for deferred output that was never attached."""
        if self.follower is None:
            if hasattr(self._sink, "flush"):
                self._sink.flush()
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                subprocess.run(["cat", self.log_path], stdout=self._sink, timeout=60, check=False)

    def remember_tree(self) -> None:
        """Record the processes below this one, for :meth:`stop`.

        Once the process exits, its descendants are reparented and no longer
        below it; workers that started a session of their own are then found
        only through this record.
        """
        if self.proc.poll() is None:
            self._known.update(_descendants(self.proc.pid, _live_processes()))

    def stop(self, grace: float) -> list[str]:
        """SIGTERM, wait ``grace`` seconds, SIGKILL; then SIGKILL the rest of its process tree.

        The rest is every process left in its session, every process recorded
        below it (by :meth:`remember_tree`, also called here first) that is
        still the same process, and the sessions those lead and the processes
        below them. Returns problems: processes still running afterwards.
        Processes left in the sessions of recorded workers that have exited are
        not killed but reported in :attr:`notes`. Once a stop has completed,
        further calls do nothing and return no problems.
        """
        if self._stopped:
            return []
        self.remember_tree()

        def left() -> set[int]:
            table = _live_processes()
            known = {
                pid for pid, start in self._known.items() if table.get(pid, ())[2:] == (start,)
            }
            # Recorded processes still running are what they were, so the sessions they
            # lead and the processes below them, started later, are theirs too.
            below = {pid for root in known for pid in _descendants(root, table)}
            sessions = {self.proc.pid} | known
            return {pid for pid, (_, sid, _) in table.items() if sid in sessions} | known | below

        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.proc.wait(timeout=SURVIVOR_WAIT_S)
        # Workers stay in its session even if they changed process group; those that
        # started a session of their own are killed, with their session and the
        # processes below them, if they were recorded below it.
        for pid in left():
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        for helper in (self.tee, self.follower):
            if helper is None:
                continue
            try:
                # The follower stops within a second of tee.
                helper.wait(timeout=10)
            except subprocess.TimeoutExpired:
                helper.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    helper.wait(timeout=SURVIVOR_WAIT_S)
        problems = []
        if self.proc.poll() is None:
            problems.append(f"{self.role} pid {self.pid} still running after SIGKILL")
        deadline = time.monotonic() + SURVIVOR_WAIT_S
        survivors = left()
        while survivors and time.monotonic() < deadline:
            # Processes started since the SIGKILL above get theirs here.
            for pid in survivors:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)
            time.sleep(0.1)
            survivors = left()
        if survivors:
            problems.append(
                f"{self.role} pid {self.pid}: processes {sorted(survivors)} survived SIGKILL"
            )
        table = _live_processes()
        exited = {pid for pid, start in self._known.items() if table.get(pid, ())[2:] != (start,)}
        stray = sorted(pid for pid, (_, sid, _) in table.items() if sid in exited)
        if stray:
            # Their session ids are those of recorded workers that exited, but such an id
            # may since belong to an unrelated process, so they are only reported.
            self.notes.append(
                f"{self.role} pid {self.pid}: processes {stray} left in the sessions of "
                f"exited workers {sorted(exited)}; not killed"
            )
        self._stopped = True
        return problems


_SEQ = itertools.count(1)
_HANDLES: list["ServeHandle"] = []


def next_seq() -> int:
    """Return an event number: handles record the numbers at which they start and end."""
    return next(_SEQ)


def handles_since(seq: int) -> list["ServeHandle"]:
    """Handles alive at event ``seq`` or started after it."""
    return [h for h in _HANDLES if h.ended_seq is None or h.ended_seq > seq]


def handles_ended_after(seq: int) -> list["ServeHandle"]:
    """Handles that ended after event ``seq``."""
    return [h for h in _HANDLES if h.ended_seq is not None and h.ended_seq > seq]


class ServeHandle:
    """The processes behind one served endpoint.

    In a chain the handle exists from the moment the engine starts, so a
    startup failure still leaves its logs reachable; ``frontend`` is ``None``
    until the front end starts. When the front end refuses to start while the
    engine loads, ``frontend`` is the exited process of its startup check.
    """

    def __init__(
        self,
        *,
        chain: str,
        model: str,
        host: str,
        port: int | None,
        frontend: LoggedProcess | subprocess.Popen | None,
        engine: LoggedProcess | None,
        engine_endpoint: str | None,
        workdir: str | None,
        counter_dir: str | None,
    ) -> None:
        self.chain = chain
        self.model = model
        self.host = host
        self.port = port
        self._frontend = frontend
        self._engine = engine
        self.engine_endpoint = engine_endpoint
        self.workdir = workdir
        self.counter_dir = counter_dir
        # Starts a chain's front end on the given port (None: no --port), for restarts.
        self._start_frontend: Callable[[str | None], LoggedProcess] | None = None
        # Counters and topology read just before shutdown, for tests that start and stop the server.
        self.final_counters = None
        self.final_topology = None
        # The first topology check that came out ``ok`` (set by the pytest plugin).
        self.proven_topology = None
        # The engine's GetServerInfo name, once read.
        self._engine_name: str | None = None
        # Processes left running by shutdown: the test that stopped the server fails.
        self.teardown_errors: list[str] = []
        # Checks that could not be made before shutdown: reported, not failed.
        self.teardown_notes: list[str] = []
        # Files the harness wrote for this server, removed at shutdown.
        self.cleanup_paths: list[str] = []
        self.started_seq = next_seq()
        self.ended_seq: int | None = None
        _HANDLES.append(self)

    @property
    def frontend(self) -> subprocess.Popen | None:
        f = self._frontend
        return f.proc if isinstance(f, LoggedProcess) else f

    @property
    def engine(self) -> subprocess.Popen | None:
        return self._engine.proc if self._engine is not None else None

    @property
    def frontend_log(self) -> str | None:
        f = self._frontend
        return f.log_path if isinstance(f, LoggedProcess) else None

    @property
    def engine_log(self) -> str | None:
        return self._engine.log_path if self._engine is not None else None

    @property
    def alive(self) -> bool:
        return self.ended_seq is None

    def poll(self) -> int | None:
        """The front end's exit code; if only the engine has exited, its code, or 1 for 0."""
        if self.frontend is not None:
            rc = self.frontend.poll()
            if rc is not None or self.engine is None:
                return rc
        engine_rc = self.engine.poll()
        if engine_rc is None:
            return None
        return engine_rc or 1

    def url_for(self, *parts: str) -> str:
        return f"http://{self.host}:{self.port}/" + "/".join(parts)

    def frontend_port_problem(self) -> str | None:
        """Why the front end cannot serve on its port, or ``None``.

        Either it exited after failing to take the port because another process
        holds it, or it runs while another process listens on the port too (a
        bind with SO_REUSEPORT, which several front ends of one server use, does
        not fail). A front end that exited for another reason has no port problem.
        """
        frontend = self._frontend
        if not isinstance(frontend, LoggedProcess) or self.port is None:
            return None
        if frontend.poll() is not None:
            if _lost_its_port(frontend.log_path, self.port):
                return f"the front end exited: port {self.port} is in use; log {frontend.log_path}"
            return None
        foreign = foreign_listeners(self.port, live_process_tree(frontend.pid))
        if foreign:
            return f"another process also listens on port {self.port} (sockets {sorted(foreign)})"
        return None

    def restart_frontend(self, port: int) -> None:
        """Stop the front end and start it again on ``port``, keeping the engine.

        For a front end with a :meth:`frontend_port_problem`. Shutdown problems of
        the stopped front end are reported with this server.
        """
        old = self._frontend
        if self._start_frontend is None or not isinstance(old, LoggedProcess):
            raise ChainError("this server's front end cannot be restarted")
        if self.engine is None or self.engine.poll() is not None:
            raise ChainStartupError("the engine has exited; the front end is not restarted")
        self.teardown_errors.extend(old.stop(TERMINATE_GRACE_S))
        self.teardown_notes.extend(old.notes)
        self.teardown_notes.append(
            f"front end pid {old.pid} could not serve on port {self.port}; restarted on port {port}"
        )
        self._frontend = None
        self.port = port
        with self.stop_on_error():
            self._frontend = self._start_frontend(str(port))

    @contextlib.contextmanager
    def stop_on_error(self) -> Iterator[None]:
        """Stop both processes if the body raises, for callers that would leave the engine running."""
        try:
            yield
        except BaseException:
            self.terminate()
            raise

    def terminate(self, grace: float = TERMINATE_GRACE_S) -> None:
        """Stop the front end, then the engine.

        Shutdown problems do not raise: processes that survive, or a stop that
        fails, go to ``teardown_errors``, which the pytest plugin turns into a
        failure of the test that stopped the server. An interrupt, such as
        KeyboardInterrupt, is raised again once both processes are stopped.
        """
        if not self.alive:
            return
        interrupted: BaseException | None = None
        try:
            try:
                self._read_before_shutdown()
            except BaseException as error:  # noqa: BLE001 - the processes must still stop
                self.teardown_notes.append(f"reading the chain before shutdown failed: {error!r}")
                interrupted = error
            for proc in (self._frontend, self._engine):
                if proc is None:
                    continue
                try:
                    if isinstance(proc, LoggedProcess):
                        self.teardown_errors.extend(proc.stop(grace))
                        self.teardown_notes.extend(proc.notes)
                    else:
                        _stop_waits(proc, grace)
                        if proc.poll() is None:
                            self.teardown_errors.append(
                                f"pid {proc.pid} still running after SIGKILL"
                            )
                except BaseException as error:  # noqa: BLE001 - the other process must still stop
                    self.teardown_errors.append(f"stopping pid {proc.pid} failed: {error!r}")
                    if not isinstance(error, Exception):
                        interrupted = interrupted or error
            for path in self.cleanup_paths:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError as error:
                    self.teardown_notes.append(f"cannot remove {path}: {error}")
            if interrupted is not None:
                raise interrupted
        finally:
            self.ended_seq = next_seq()

    def _read_before_shutdown(self) -> None:
        """Read the counters, the topology and the engine name while both processes run."""
        if self.chain == cv.CHAIN_NONE:
            return
        if not self._both_running():
            why = "the front end never started" if self.frontend is None else "a process had exited"
            self.teardown_notes.append(f"counters and topology not read: {why}")
            return
        from .proof import check_topology, read_counters, read_engine_name

        try:
            self.final_counters = read_counters(self)
        except Exception as error:  # noqa: BLE001 - recorded, shutdown must go on
            self.teardown_notes.append(f"reading counters before shutdown failed: {error!r}")
        try:
            self.final_topology = self.proven_topology or check_topology(self)
        except Exception as error:  # noqa: BLE001 - recorded, shutdown must go on
            self.teardown_notes.append(f"topology check before shutdown failed: {error!r}")
        try:
            read_engine_name(self)
        except Exception as error:  # noqa: BLE001 - recorded, shutdown must go on
            self.teardown_notes.append(f"reading the engine name before shutdown failed: {error!r}")

    def _both_running(self) -> bool:
        return (
            self.frontend is not None
            and self.frontend.poll() is None
            and (self.engine is None or self.engine.poll() is None)
        )


class FrontendCheck:
    """The front end's own startup checks, run while the engine loads.

    The front end checks its configuration, such as refusing a beam width
    above 1, before it contacts the engine. This runs the server's front-end
    command line against an engine address where only the harness listens, on
    a port of its own, and reads the outcome from what the process does, never
    from the options: once it connects to that address, its configuration
    checks have passed and it is stopped. If it exits first with an error code
    and its log does not say that a port was in use, the front end refused to
    start; any other exit (a port in use, a signal, code 0) drops the check.
    The process of a refusal then stands for the server's front end, with its
    log and exit code.
    """

    PASSED = "passed"
    REFUSED = "refused"
    DROPPED = "dropped"

    def __init__(
        self,
        command: Callable[[str], Sequence[str]],
        env: Mapping[str, str],
        log_dir: str,
        sink: IO[Any] | int | None,
    ) -> None:
        """Start ``command(endpoint)``, a front-end command line for the engine at ``endpoint``."""
        self._address = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._address.bind((ENGINE_HOST, 0))
            self._address.listen(1)
            endpoint = f"{ENGINE_HOST}:{self._address.getsockname()[1]}"
            self.process = LoggedProcess(
                "frontend-check", command(endpoint), env, log_dir, sink, defer_sink=True
            )
        except BaseException:
            self._address.close()
            raise
        # None until the outcome is known.
        self.outcome: str | None = None
        # Processes the stop left running, and what it saw but did not act on.
        self.problems: list[str] = []
        self.notes: list[str] = []
        self._stopped = False

    def refused(self) -> bool:
        """Whether the front end refused to start; a check that passed or was dropped is stopped."""
        if self.outcome is None:
            self.outcome = self._read_outcome()
            if self.outcome in (self.PASSED, self.DROPPED):
                self.stop()
        return self.outcome == self.REFUSED

    def _read_outcome(self) -> str | None:
        if self._contacted():
            return self.PASSED
        rc = self.process.poll()
        if rc is None:
            return None
        # It may have connected just before it exited.
        if self._contacted():
            return self.PASSED
        if rc <= 0:
            return self.DROPPED
        # The whole log is written once tee has seen the end of the output.
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.process.tee.wait(timeout=SURVIVOR_WAIT_S)
        try:
            with open(self.process.log_path, "rb") as f:
                text = f.read().decode("utf-8", "replace")
        except OSError:
            text = ""
        return self.DROPPED if _ANY_PORT_IN_USE.search(text) else self.REFUSED

    def _contacted(self) -> bool:
        if self._address.fileno() < 0:
            return False
        # poll, unlike select, takes descriptors above FD_SETSIZE.
        poller = select.poll()
        poller.register(self._address, select.POLLIN)
        return bool(poller.poll(0))

    def adopt(self) -> LoggedProcess:
        """The process of a refusal, to stand for the front end; its output then reaches the sink."""
        self._stopped = True
        self._address.close()
        self.process.attach_sink()
        return self.process

    def stop(self) -> None:
        """Stop the process and close the address; does nothing after :meth:`adopt`."""
        if self._stopped:
            return
        self._stopped = True
        try:
            self.problems.extend(self.process.stop(TERMINATE_GRACE_S))
            self.notes.extend(self.process.notes)
        finally:
            self._address.close()


def wait_engine_ready(
    endpoint: str,
    engine: LoggedProcess,
    timeout: float,
    interval: float = 2.0,
    give_up: Callable[[], bool] | None = None,
) -> bool:
    """Poll gRPC health service ``inference`` until it reports SERVING, from ``engine`` alone.

    Returns True then, or False as soon as ``give_up`` returns True.
    """
    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    port = int(endpoint.rsplit(":", 1)[1])
    deadline = time.monotonic() + timeout
    with grpc.insecure_channel(endpoint) as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        while True:
            rc = engine.poll()
            if rc is not None:
                raise ChainStartupError(
                    f"engine exited with code {rc} before it was ready; log {engine.log_path}:\n"
                    f"{tail(engine.log_path)}"
                )
            if give_up is not None and give_up():
                return False
            try:
                response = stub.Check(health_pb2.HealthCheckRequest(service="inference"), timeout=5)
                if response.status == health_pb2.HealthCheckResponse.SERVING:
                    # The answer may come from another server on the same port.
                    foreign = foreign_listeners(port, live_process_tree(engine.pid))
                    if foreign:
                        raise EnginePortTaken(
                            f"another process also listens on {endpoint} (sockets "
                            f"{sorted(foreign)}); log {engine.log_path}"
                        )
                    return True
            except grpc.RpcError:
                pass
            if time.monotonic() > deadline:
                raise ChainStartupError(
                    f"engine at {endpoint} not SERVING within {timeout:.0f} s; log "
                    f"{engine.log_path}:\n{tail(engine.log_path)}"
                )
            time.sleep(interval)


def _results_dir() -> str:
    """A new directory for one server's logs and files, kept after it stops.

    Under ``TRTLLM_OPENENGINE_RESULTS_DIR`` if set, else in the system temp directory.
    """
    base = os.environ.get(cv.ENV_RESULTS_DIR)
    if base:
        os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix="chain-", dir=base or None)


def _engine_prefix(chain: str, command: ServeCommand) -> list[str]:
    if chain == cv.CHAIN_TRTLLM:
        return list(command.prefix)
    if chain == cv.CHAIN_DS:
        ds = os.environ.get(cv.ENV_DS_SERVE_CMD, "")
        if not ds:
            raise ChainError(f"{cv.ENV_CHAIN}=ds needs {cv.ENV_DS_SERVE_CMD}")
        return shlex.split(ds)
    glm = os.environ.get(cv.ENV_GLM_SERVE_CMD, "")
    if not glm:
        raise ChainError(f"{cv.ENV_CHAIN}=glm needs {cv.ENV_GLM_SERVE_CMD}")
    return shlex.split(glm)


def launch_serve_command(
    cmd: Sequence[str],
    *,
    chain: str | None = None,
    env: Mapping[str, str] | None = None,
    sink: IO[Any] | int | None = None,
    engine_ready_timeout: float = DEFAULT_ENGINE_READY_TIMEOUT_S,
    check_plugin: bool = True,
) -> ServeHandle:
    """Start what ``cmd`` (a ``trtllm-serve`` command line) starts, through the chain if one is selected.

    ``chain`` defaults to ``TRTLLM_TEST_OPENENGINE_CHAIN``; ``env`` (default
    ``os.environ``) is the processes' environment. Returns once the engine is
    ready and the front end is started; the caller then waits for the front
    end's ``/health`` itself. While the engine loads, a :class:`FrontendCheck`
    runs the front end's startup checks; if the front end refuses to start, the
    engine is stopped and the call returns at once, with the check's exited
    process as the front end (its output reaches ``sink``, the engine's does
    not). Otherwise the engine's output reaches ``sink`` from the attempt that
    becomes ready, or from the last one if none does. An attempt retried on a new
    port leaves its log files; its teardown errors, and a note saying why it was
    replaced, are reported with the server that replaces it (its other teardown
    notes are not). A TensorRT-LLM engine whose command line gives no
    :data:`ENGINE_METRICS_PORT_OPTION` gets it with 0 (an ephemeral port) if the
    installed ``trtllm-serve`` declares it with a lower bound of 0. A command
    line that gives the option is refused for a glm chain, and for a TensorRT-LLM
    chain if that port is nonzero and equals the ``--port`` it gives. The gRPC
    port the harness picks for the engine differs from the front end's ``--port``
    and the metrics port ``cmd`` gives, and from the front-end check's port. On a startup
    failure the started processes are stopped before the error propagates, and
    their handle stays registered with its logs.
    """
    env = dict(os.environ if env is None else env)
    chain = resolve_chain() if chain is None else chain
    if chain == cv.CHAIN_NONE:
        # Unchanged command line: the split table does not apply.
        proc = subprocess.Popen(list(cmd), env=env, stdout=sink, stderr=sink)
        host, port = _address(cmd)
        return ServeHandle(
            chain=chain,
            model="",
            host=host,
            port=port,
            frontend=proc,
            engine=None,
            engine_endpoint=None,
            workdir=None,
            counter_dir=None,
        )
    command = parse_serve_command(cmd)
    if check_plugin:
        require_plugin()
    engine_prefix = _engine_prefix(chain, command)
    given_metrics_port = option_value(command.args, ENGINE_METRICS_PORT_OPTION)
    if given_metrics_port is not None and chain != cv.CHAIN_TRTLLM:
        raise ChainError(
            f"{ENGINE_METRICS_PORT_OPTION} is a TensorRT-LLM engine option; the harness does "
            f"not pass it to the {chain} chain's engine"
        )
    host, port = _address(command.args)
    # trtllm-serve refuses a metrics port equal to its --port, but the engine gets a
    # --port the harness picks, so that check does not see a clash with the front end's.
    if _as_port(given_metrics_port) and _as_port(given_metrics_port) == port:
        raise ChainError(f"{ENGINE_METRICS_PORT_OPTION} must differ from --port")
    workdir = _results_dir()
    split = split_args(command.args, rewrite_yaml=lambda path: engine_yaml(path, workdir))
    engine_env = {k: v for k, v in env.items() if k != cv.ENV_COUNTER_DIR}
    engine_flags = []
    if chain == cv.CHAIN_TRTLLM and iteration_stats_enabled(command.args):
        engine_flags.append(ENGINE_LOAD_METRICS_FLAG)
    if chain == cv.CHAIN_TRTLLM and given_metrics_port is None and engine_supports_metrics_port():
        engine_flags += [ENGINE_METRICS_PORT_OPTION, "0"]
    counter_dir = os.path.join(workdir, "counters")
    fe_env = frontend_env(env)
    check_env = {k: v for k, v in fe_env.items() if k != cv.ENV_COUNTER_DIR}
    check_env[FRONTEND_CHECK_ENV] = "1"
    fe_env[cv.ENV_COUNTER_DIR] = counter_dir

    def frontend_cmd(engine_endpoint: str, frontend_port: str | None) -> list[str]:
        cmd = list(command.prefix) + [command.model, "--openengine-endpoint", engine_endpoint]
        if frontend_supports_ready_timeout():
            cmd += ["--openengine-ready-timeout", str(FRONTEND_READY_TIMEOUT_S)]
        if split.host is not None:
            cmd += ["--host", split.host]
        if frontend_port is not None:
            cmd += ["--port", frontend_port]
        return cmd + list(split.frontend)

    # Runs while the engine loads, so that a configuration the front end refuses
    # does not wait for the engine.
    check_port = free_port()
    check = FrontendCheck(
        lambda check_endpoint: frontend_cmd(check_endpoint, str(check_port)),
        check_env,
        workdir,
        sink,
    )
    handle: ServeHandle | None = None
    try:
        replaced_errors: list[str] = []
        replaced_notes: list[str] = []
        for attempt in range(1, ENGINE_PORT_ATTEMPTS + 1):
            # The engine binds its gRPC port only after loading the model, so another
            # process may take it in between. The front end's port, the front-end check's
            # port and a metrics port the command line gives may be unbound at this point
            # too, so free_port() may return them.
            engine_port = _free_port_other_than(_as_port(given_metrics_port), port, check_port)
            endpoint = f"{ENGINE_HOST}:{engine_port}"
            engine_cmd = (
                engine_prefix
                + [
                    command.model,
                    "--grpc",
                    "--grpc-protocol",
                    "openengine",
                    "--host",
                    ENGINE_HOST,
                    "--port",
                    str(engine_port),
                ]
                + list(split.engine)
                + engine_flags
            )
            handle = ServeHandle(
                chain=chain,
                model=command.model,
                host=host,
                port=port,
                frontend=None,
                engine=LoggedProcess(
                    "engine", engine_cmd, engine_env, workdir, sink, defer_sink=True
                ),
                engine_endpoint=endpoint,
                workdir=workdir,
                counter_dir=counter_dir,
            )
            handle.teardown_errors.extend(replaced_errors)
            handle.teardown_notes.extend(replaced_notes)
            try:
                with handle.stop_on_error():
                    ready = wait_engine_ready(
                        endpoint, handle._engine, engine_ready_timeout, give_up=check.refused
                    )
                    if ready:
                        # The engine has started its workers by now.
                        handle._engine.remember_tree()
                        handle._engine.attach_sink()
            except BaseException as error:
                if (
                    isinstance(error, ChainStartupError)
                    and attempt < ENGINE_PORT_ATTEMPTS
                    and (
                        isinstance(error, EnginePortTaken)
                        or _lost_its_port(handle.engine_log, engine_port)
                    )
                ):
                    # The next attempt replaces this server; it is no test's server.
                    _HANDLES.remove(handle)
                    replaced_errors = list(handle.teardown_errors)
                    replaced_notes = replaced_notes + [
                        f"engine pid {handle._engine.pid} lost its gRPC port {engine_port} to "
                        f"another process (log {handle.engine_log}); started again on a new port"
                    ]
                    continue
                handle._engine.copy_to_sink()
                raise
            break
        if not ready:
            # The front end refused to start: the engine is stopped without waiting for
            # it to load, and the check's exited process stands for the front end.
            handle.teardown_errors.extend(handle._engine.stop(TERMINATE_GRACE_S))
            handle._frontend = check.adopt()
            return handle
        check.stop()

        def start_frontend(frontend_port: str | None) -> LoggedProcess:
            return LoggedProcess(
                "frontend", frontend_cmd(endpoint, frontend_port), fe_env, workdir, sink
            )

        handle._start_frontend = start_frontend
        with handle.stop_on_error():
            os.makedirs(counter_dir, exist_ok=True)
            handle._frontend = start_frontend(split.port)
        return handle
    finally:
        check.stop()
        if handle is not None and handle in _HANDLES:
            handle.teardown_errors.extend(check.problems)
            handle.teardown_notes.extend(check.notes)
        else:
            # The start failed before a server that reports them was registered.
            for line in check.problems + check.notes:
                print(f"front-end check of a failed start: {line}", file=sys.stderr, flush=True)
        check.problems.clear()
        check.notes.clear()


def _as_port(value: str | None) -> int | None:
    """``value`` as a port number, or ``None`` if it is not one."""
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _free_port_other_than(*taken: int | None) -> int:
    """A :func:`free_port` that is none of ``taken``.

    ``taken`` are ports that another process or option of the server uses.
    """
    for _ in range(100):
        port = free_port()
        if port not in taken:
            return port
    raise ChainError(f"no free port other than {taken} found")


def _lost_its_port(log_path: str | None, port: int) -> bool:
    """Whether a process's log reports that taking ``port`` failed because it is in use.

    Reads the whole log: shutdown output can push the error out of the tail.
    The error, the port and the reason must be on one line, so that errors
    about other ports (MPI, metrics) do not count.
    """
    if not log_path:
        return False
    try:
        with open(log_path, "rb") as f:
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return False
    pattern = re.compile(_PORT_IN_USE_LINE.format(port=port), re.IGNORECASE | re.MULTILINE)
    return pattern.search(text) is not None


def _address(args: Sequence[str]) -> tuple[str, int | None]:
    """The front end's address for clients, from ``--host`` and ``--port`` in ``args``."""
    found: dict[str, str] = {}
    for i, token in enumerate(args):
        name, eq, value = token.partition("=")
        if name in ("--host", "--port"):
            if eq:
                found[name] = value
            elif i + 1 < len(args):
                found[name] = args[i + 1]
    host = found.get("--host", "localhost")
    if host in ("0.0.0.0", "::"):
        host = "localhost"
    port = found.get("--port")
    return host, int(port) if port is not None else None


def launch_serve(
    model: str,
    args: Sequence[str],
    extra_config: Mapping[str, Any] | None = None,
    *,
    chain: str,
    prefix: Sequence[str] = ("trtllm-serve",),
    env: Mapping[str, str] | None = None,
    sink: IO[Any] | int | None = None,
    engine_ready_timeout: float = DEFAULT_ENGINE_READY_TIMEOUT_S,
    check_plugin: bool = True,
) -> ServeHandle:
    """Start ``trtllm-serve MODEL ARGS`` with ``extra_config`` as an extra YAML file.

    The entry point for callers that build the command themselves, such as
    benchmark drivers. The YAML file is removed when the server stops.
    """
    args = list(args)
    path = None
    if extra_config:
        results = os.environ.get(cv.ENV_RESULTS_DIR)
        if results:
            os.makedirs(results, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="extra-llm-api-", suffix=".yaml", dir=results or None)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(extra_config), f)
        args += ["--extra_llm_api_options", path]
    try:
        handle = launch_serve_command(
            list(prefix) + [model] + args,
            chain=chain,
            env=env,
            sink=sink,
            engine_ready_timeout=engine_ready_timeout,
            check_plugin=check_plugin,
        )
    except BaseException:
        if path:
            os.remove(path)
        raise
    if path:
        handle.cleanup_paths.append(path)
    return handle


class ChainProcess:
    """Stands in for the ``subprocess.Popen`` of a test that starts ``trtllm-serve`` itself."""

    def __init__(self, handle: ServeHandle) -> None:
        self.handle = handle
        self.args = handle.frontend.args

    @property
    def pid(self) -> int:
        return self.handle.frontend.pid

    @property
    def returncode(self) -> int | None:
        return self.poll()

    def poll(self) -> int | None:
        return self.handle.poll()

    def wait(self, timeout: float | None = None) -> int:
        """Wait until :meth:`poll` reports an exit: the front end's, or the engine's."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while (rc := self.poll()) is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(0.1)
        return rc

    def terminate(self) -> None:
        self.handle.terminate()

    def kill(self) -> None:
        self.handle.terminate(grace=0)


@contextlib.contextmanager
def chain_popen(
    cmd: Sequence[str],
    *,
    stdout: IO[Any] | int | None = None,
    stderr: IO[Any] | int | None = None,
    env: Mapping[str, str] | None = None,
) -> Iterator[ChainProcess]:
    """Stands in for ``popen(cmd, stdout=<file>, stderr=..., env=...)`` around ``trtllm-serve``."""
    if stdout == subprocess.PIPE:
        # Nothing would read the pipe: the server would block once it fills.
        raise ChainError("chain_popen writes the output to a file or to stdout, not to a pipe")
    if stderr not in (None, stdout, subprocess.STDOUT):
        raise ChainError("chain_popen sends stdout and stderr to one sink; got different ones")
    handle = launch_serve_command(cmd, env=env, sink=stdout)
    try:
        yield ChainProcess(handle)
    finally:
        handle.terminate()

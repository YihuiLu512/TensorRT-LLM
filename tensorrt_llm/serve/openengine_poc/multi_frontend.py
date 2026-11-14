# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Several serving frontends in front of one OpenEngine engine (SV-MF).

With ``num_serve_frontends`` N > 1, N frontend processes share the serving port
through ``SO_REUSEPORT``; each builds its own facade, and so opens its own
connection to the engine (SV-MF-2). The process ``trtllm-serve`` started is
frontend 0, the launcher: it connects to the engine, then re-execs its command
line N - 1 times (``serve/_frontend_processes.py``). No frontend listens before
every frontend has connected to the engine (SV-MF-3): the launcher listens once
every attached frontend has reported READY, and an attached frontend once the
launcher listens.

- The attached frontends forward the launcher-owned routes to the launcher
  (``serve/multi_frontend.py``), and stop when the launcher process is gone or
  no longer answers. Each frontend watches the engine itself.
- An attached frontend that exits while serving is logged, and the others keep
  serving.
- SIGTERM stops a frontend after its cleanup: the launcher stops the attached
  frontends, and every frontend closes its facade.
- The Responses API store is off, since a follow-up request may reach another
  frontend (SV-MF-4).
- ``--port 0`` and ``--report_addr`` need a single frontend.
- ``num_postprocess_workers`` is split between the frontends (SV-PP-1).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import gc
import os
import shutil
import signal
import socket
import sys
import tempfile
import threading
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import click
import uvloop

from tensorrt_llm.logger import logger
from tensorrt_llm.serve._frontend_processes import (
    MultiFrontendMode,
    _frontend_affinity_sockets,
    _signal_frontend_ready,
    _spawn_frontend_group,
    _terminate_attached_frontends,
    _watch_attached_frontends,
)
from tensorrt_llm.serve.multi_frontend import (
    LAUNCHER_UDS_ENV,
    LAUNCHER_UDS_NAME,
    MultiFrontendServing,
)

from . import contracts
from .contracts import FrontendLaunchConfig

if TYPE_CHECKING:
    from tensorrt_llm.serve.openai_server import OpenAIServer

    from .facade import OpenEngineLLM

# Set for each frontend the launcher spawns: its id, from 1.
_FRONTEND_ID_ENV = "TLLM_EXECUTOR_FRONTEND_ID"

# Longest path a Unix socket address holds, without its terminating NUL.
_MAX_UNIX_PATH = 107


def postproc_share(total: int, num_frontends: int, frontend_id: int) -> int:
    """Frontend ``frontend_id``'s share of ``total`` postprocessing workers (SV-PP-1).

    The workers are split evenly, the first frontends taking one more each
    when ``total`` does not divide. Every frontend gets at least one unless
    ``total`` is 0, so with fewer workers than frontends the frontends run
    ``num_frontends`` workers in all, more than ``total``.
    """
    if total <= 0:
        return 0
    base, extra = divmod(total, num_frontends)
    return max(1, base + (1 if frontend_id < extra else 0))


def launch_frontends(config: FrontendLaunchConfig) -> None:
    """Start ``num_serve_frontends`` frontends for the engine and serve until they stop.

    A single frontend is served as by
    :func:`~tensorrt_llm.serve.openengine_poc.contracts.default_launch_frontends`,
    or, as a disaggregated worker, by :func:`_serve_one_worker`. With several
    frontends, only the launcher registers with the disaggregated cluster.

    Raises:
        click.UsageError: The configuration asks for something these frontends cannot do.
        OpenEngineStartupError: The engine is not usable; the message names the step.
        RuntimeError: The port cannot be bound, the launcher's socket path is too
            long, a frontend failed to start, or (in an attached frontend) the
            launcher exited before it started serving.
        TypeError: The facade refuses the tokenizer (not a Hugging Face one).
        ValueError: ``OpenAIServer`` refuses the serving configuration.
        OSError: The temporary directory or a socket could not be created, or
            (in an attached frontend) the launcher was gone when it reported ready.
    """
    launcher_uds = os.environ.pop(LAUNCHER_UDS_ENV, None)
    num_frontends = _num_frontends(config.llm_args)
    if launcher_uds is not None:
        _serve_attached(config, num_frontends, launcher_uds)
    elif num_frontends == 1:
        if _is_worker(config):
            _serve_one_worker(config)
        else:
            contracts.default_launch_frontends(config)
    else:
        _serve_launcher(config, num_frontends)


def _is_worker(config: FrontendLaunchConfig) -> bool:
    """Whether the frontend serves as a disaggregated worker."""
    # As OpenAIServer reads them: an empty cluster configuration or key is none.
    return (
        config.server_role is not None
        or bool(config.disagg_cluster_config)
        or bool(config.internal_disagg_auth_key)
    )


def _serve_one_worker(config: FrontendLaunchConfig) -> None:
    """One frontend, in this process, as a disaggregated worker.

    ``contracts.default_launch_frontends`` does not pass the worker fields on.
    """
    from . import launch

    # Passed only when set, so the frontend's own defaults apply otherwise.
    kwargs: dict[str, Any] = {}
    if config.ready_timeout is not None:
        kwargs["ready_timeout"] = config.ready_timeout
    if config.control_timeout is not None:
        kwargs["control_timeout"] = config.control_timeout
    launch.launch_openengine_server(
        config.host,
        config.port,
        config.llm_args,
        config.endpoint,
        tool_parser=config.tool_parser,
        middleware=config.middleware,
        chat_template=config.chat_template,
        served_model_name=config.served_model_name,
        allow_request_chat_template=config.allow_request_chat_template,
        multimodal_server_config=config.multimodal_server_config,
        num_input_processor_workers=config.num_input_processor_workers,
        num_media_load_workers=config.num_media_load_workers,
        report_addr=config.report_addr,
        allow_newer_schema=config.allow_newer_schema,
        server_role=config.server_role,
        disagg_cluster_config=config.disagg_cluster_config,
        internal_disagg_auth_key=config.internal_disagg_auth_key,
        **kwargs,
    )


def _num_frontends(llm_args: dict[str, Any]) -> int:
    value = llm_args.get("num_serve_frontends")
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise click.UsageError(f"num_serve_frontends must be a positive integer, got {value!r}")
    return value


def _postproc_total(llm_args: dict[str, Any]) -> int:
    value = llm_args.get("num_postprocess_workers")
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise click.UsageError(
            f"num_postprocess_workers must be a non-negative integer, got {value!r}"
        )
    return value


def _attached_frontend_id(num_frontends: int) -> int:
    value = os.environ.get(_FRONTEND_ID_ENV)
    try:
        frontend_id = int(value) if value is not None else None
    except ValueError:
        frontend_id = None
    if frontend_id is None or not 1 <= frontend_id < num_frontends:
        raise click.UsageError(
            f"{LAUNCHER_UDS_ENV} is set, so {_FRONTEND_ID_ENV} must be a frontend id "
            f"from 1 to {num_frontends - 1}, got {value!r}"
        )
    return frontend_id


class _Sigterm:
    """Turns SIGTERM into an orderly stop of this frontend.

    Until uvicorn takes SIGTERM over, the handler cancels the frontend's main
    task, also for a SIGTERM received before the task starts. Once uvicorn has
    shut down, it raises SIGTERM again into this handler. The handler does not
    raise: under uvloop an exception raised by a signal handler is logged and
    dropped while the loop is idle. A process forked from this one inherits
    the handler and dies of SIGTERM as by default.
    """

    def __init__(self) -> None:
        self.received = False
        self._pid = os.getpid()
        self._task: asyncio.Task | None = None
        self._previous = signal.signal(signal.SIGTERM, self._handle)

    def cancel_on_signal(self, task: asyncio.Task) -> None:
        self._task = task
        if self.received:
            task.cancel()

    def stop_cancelling(self) -> None:
        """From now on only record SIGTERM: uvicorn handles it, or the cleanup runs."""
        self._task = None

    def installed(self) -> bool:
        return signal.getsignal(signal.SIGTERM) == self._handle

    def _handle(self, signum: int, frame: Any) -> None:
        if os.getpid() != self._pid:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.raise_signal(signal.SIGTERM)
            return
        self.received = True
        if self._task is not None and not self._task.done():
            self._task.cancel()

    def restore(self) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_DFL if self._previous is None else self._previous)


@contextlib.contextmanager
def _stopped_by_sigterm():
    """Handle SIGTERM in the block; after a SIGTERM, end the process by it once the block is done.

    The cancellation a SIGTERM causes ends the block quietly. Any other
    exception the block raises propagates instead.
    """
    sigterm = _Sigterm()
    try:
        yield sigterm
    except asyncio.CancelledError:
        if not sigterm.received:
            raise
    finally:
        sigterm.restore()
    if sigterm.received:
        # Ending by the signal skips the buffered output otherwise.
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(AttributeError, OSError, ValueError):
                stream.flush()
        signal.raise_signal(signal.SIGTERM)


async def _serve(
    server: OpenAIServer,
    config: FrontendLaunchConfig,
    sockets: list[socket.socket],
    sigterm: _Sigterm,
) -> None:
    """Serve until uvicorn stops, cancelled by a SIGTERM that comes before uvicorn takes it over."""
    serving = asyncio.ensure_future(server(config.host, config.port, sockets=sockets))
    try:
        # uvicorn replaces the SIGTERM handler in the task's first step.
        while not serving.done() and sigterm.installed():
            await asyncio.sleep(0)
    except BaseException:
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving
        raise
    sigterm.stop_cancelling()
    await serving


async def _finish(cleanup: Awaitable[None]) -> None:
    """Await ``cleanup`` to its end even if this task is cancelled meanwhile, then pass that on.

    Once uvicorn has shut down after a SIGINT, it raises SIGINT again, and the
    event loop's runner cancels the main task while it cleans up.
    """
    task = asyncio.ensure_future(cleanup)
    cancelled = False
    while True:
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                raise
            cancelled = True
        else:
            break
    if cancelled:
        raise asyncio.CancelledError


def _disable_responses_store(*, warn: bool) -> None:
    # As launch_server does: the per-process store cannot follow a request
    # that reaches another frontend.
    if warn and not os.getenv("TRTLLM_RESPONSES_API_DISABLE_STORE"):
        logger.warning(
            "num_serve_frontends > 1: stateful Responses API storage "
            "(store/previous_response_id) is disabled; the per-frontend "
            "in-memory store cannot be shared across frontends. Pass "
            "--num_serve_frontends 1 to keep it."
        )
    os.environ["TRTLLM_RESPONSES_API_DISABLE_STORE"] = "1"


def _address_family(host: str, port: int) -> socket.AddressFamily:
    addr_info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    if all(info[0] == socket.AF_INET6 for info in addr_info):
        return socket.AF_INET6
    return socket.AF_INET


def _bind_shared_port(host: str, port: int, family: socket.AddressFamily) -> socket.socket:
    """This frontend's listening socket on the port all the frontends share.

    Raises:
        RuntimeError: The address cannot be bound; the message names the port's
            holder when it can.
    """
    from tensorrt_llm.commands.serve import _diagnose_port_in_use

    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        # As launch_server does: a restart is not refused during TIME_WAIT,
        # and the kernel spreads the connections over every frontend's socket.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind((host, port))
    except OSError as error:
        sock.close()
        raise RuntimeError(
            f"Failed to bind socket to {host}:{port}: {error}. "
            f"Port holder(s): {_diagnose_port_in_use(port)}"
        ) from error
    except BaseException:
        sock.close()
        raise
    return sock


def _bind_launcher_socket(ipc_dir: str) -> tuple[socket.socket, str]:
    """The Unix socket the attached frontends forward the launcher-owned routes to."""
    path = os.path.join(ipc_dir, LAUNCHER_UDS_NAME)
    if len(os.fsencode(path)) > _MAX_UNIX_PATH:
        raise RuntimeError(
            f"The launcher's Unix socket path {path} is longer than {_MAX_UNIX_PATH} bytes; "
            "set TMPDIR to a shorter directory."
        )
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(path)
    except BaseException:
        sock.close()
        raise
    return sock, path


def _close_facade(facade: OpenEngineLLM) -> None:
    """Close a facade that never served, without hiding the error that stopped it."""
    try:
        asyncio.run(facade.aclose())
    # Broad on purpose: this runs while another error propagates, which must win.
    except Exception as error:
        logger.warning(f"Closing the OpenEngine facade failed: {error!r}")


def _build_frontend(
    config: FrontendLaunchConfig,
    *,
    postproc_workers: int,
    serving: MultiFrontendServing,
    before_server: Callable[[OpenEngineLLM], None] | None = None,
    after_server: Callable[[OpenEngineLLM, OpenAIServer], None] | None = None,
) -> tuple[OpenEngineLLM, OpenAIServer]:
    """This process's facade and the ``OpenAIServer`` on it.

    Built with the arguments ``launch_openengine_server`` uses, plus
    ``postproc_workers`` and ``serving``. ``before_server`` runs on the facade
    before the server is built, ``after_server`` on both once it is.
    """
    from tensorrt_llm.commands.serve import _apply_fastapi_middlewares
    from tensorrt_llm.serve.openai_server import OpenAIServer

    from .facade import (
        DEFAULT_CONTROL_TIMEOUT_S,
        DEFAULT_READY_TIMEOUT_S,
        OpenEngineLLM,
        adapt_server,
    )

    try:
        facade = OpenEngineLLM.from_llm_args(
            config.llm_args,
            config.endpoint,
            ready_timeout=(
                DEFAULT_READY_TIMEOUT_S if config.ready_timeout is None else config.ready_timeout
            ),
            control_timeout=(
                DEFAULT_CONTROL_TIMEOUT_S
                if config.control_timeout is None
                else config.control_timeout
            ),
            allow_newer_schema=config.allow_newer_schema,
            postproc_workers=postproc_workers,
            server_role=config.server_role,
        )
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    try:
        if before_server is not None:
            before_server(facade)
        server = OpenAIServer(
            generator=facade,
            model=config.served_model_name or config.llm_args["model"],
            tool_parser=config.tool_parser,
            server_role=config.server_role,
            metadata_server_cfg=None,
            # OpenAIServer registers only from the launcher.
            disagg_cluster_config=config.disagg_cluster_config,
            multimodal_server_config=config.multimodal_server_config,
            chat_template=config.chat_template,
            allow_request_chat_template=config.allow_request_chat_template,
            input_processor_workers=config.num_input_processor_workers,
            media_load_workers=config.num_media_load_workers,
            internal_disagg_auth_key=config.internal_disagg_auth_key,
            multi_frontend_serving=serving,
        )
        adapt_server(server, facade)
        _apply_fastapi_middlewares(server.app, config.middleware)
        if after_server is not None:
            after_server(facade, server)
    except BaseException:
        # The facade may already hold a postprocessing pool.
        _close_facade(facade)
        raise
    return facade, server


def _maybe_disable_gc() -> None:
    # As launch_server does.
    if os.getenv("TRTLLM_SERVER_DISABLE_GC", "0") == "1":
        gc.disable()


def _serve_launcher(config: FrontendLaunchConfig, num_frontends: int) -> None:
    """Serve as frontend 0 and run the other ``num_frontends - 1`` frontends."""
    from tensorrt_llm.commands.serve import (
        _diagnose_port_in_use,
        _reserve_multi_frontend_port,
        _share_prometheus_multiproc_dir,
    )

    host, port = config.host, config.port
    if port == 0 or config.report_addr:
        raise click.UsageError(
            "port 0 and --report_addr are only supported with a single serving "
            f"frontend, but num_serve_frontends={num_frontends}."
        )
    postproc_total = _postproc_total(config.llm_args)
    if 0 < postproc_total < num_frontends:
        logger.warning(
            f"num_postprocess_workers={postproc_total} is less than "
            f"num_serve_frontends={num_frontends}: each frontend's share is one "
            f"postprocessing worker, {num_frontends} in all."
        )
    _disable_responses_store(warn=True)
    family = _address_family(host, port)
    mode = MultiFrontendMode(num_frontends, is_attached_frontend=False)
    with _stopped_by_sigterm() as sigterm, contextlib.ExitStack() as stack:
        # Private to the user (mkdtemp creates it 0700); holds the launcher's
        # socket and the shared Prometheus directory.
        ipc_dir = tempfile.mkdtemp(prefix="tllm-oe-frontends-")
        stack.callback(shutil.rmtree, ipc_dir, ignore_errors=True)
        stack.enter_context(_reserve_multi_frontend_port(host, port, family))
        sock = stack.enter_context(_bind_shared_port(host, port, family))
        affinity_socket, attached_affinity_sockets = stack.enter_context(
            _frontend_affinity_sockets(
                host,
                port,
                family,
                mode,
                config.llm_args.get("serve_frontend_affinity_port_stride"),
                _diagnose_port_in_use,
            )
        )
        uds_sock, uds_path = _bind_launcher_socket(ipc_dir)
        stack.enter_context(uds_sock)
        facade, server = _build_frontend(
            config,
            postproc_workers=postproc_share(postproc_total, num_frontends, 0),
            serving=MultiFrontendServing(launcher_uds=uds_path, is_launcher=True),
            before_server=lambda facade: _share_prometheus_multiproc_dir(facade, ipc_dir),
        )
        sockets = [sock, uds_sock]
        if affinity_socket is not None:
            sockets.append(affinity_socket)
        uvloop.run(
            _run_launcher(
                config,
                num_frontends,
                facade,
                server,
                sockets,
                uds_path,
                attached_affinity_sockets,
                sigterm,
            )
        )


async def _run_launcher(
    config: FrontendLaunchConfig,
    num_frontends: int,
    facade: OpenEngineLLM,
    server: OpenAIServer,
    sockets: list[socket.socket],
    uds_path: str,
    attached_affinity_sockets: dict[int, socket.socket],
    sigterm: _Sigterm,
) -> None:
    sigterm.cancel_on_signal(asyncio.current_task())
    _maybe_disable_gc()
    cancelled = threading.Event()
    spawn: asyncio.Future | None = None
    children: list = []
    stop_watching: threading.Event | None = None
    try:
        await facade.start()
        spawn = asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                _spawn_frontend_group,
                num_frontends,
                extra_env={LAUNCHER_UDS_ENV: uds_path},
                cancelled=cancelled,
                affinity_sockets=attached_affinity_sockets,
            ),
        )
        # Shielded: if this task is cancelled, the spawn still finishes, and
        # the cleanup below reaps whatever it started.
        children = await asyncio.shield(spawn)
        stop_watching = _watch_attached_frontends(children)
        logger.info(
            f"All {num_frontends} serving frontends are connected to the OpenEngine engine "
            f"at {config.endpoint}; serving on {config.host}:{config.port}"
        )
        await _serve(server, config, sockets, sigterm)
    finally:
        sigterm.stop_cancelling()
        # Stops a spawn still waiting for READY; it then reaps its own children.
        cancelled.set()
        if stop_watching is not None:
            stop_watching.set()
        await _finish(_stop_launcher(facade, spawn, children))


async def _stop_launcher(
    facade: OpenEngineLLM, spawn: asyncio.Future | None, children: list
) -> None:
    """Stop the attached frontends, then the facade.

    The frontends include those of a spawn that finished while this task was
    being cancelled; a spawn stopped by ``cancelled`` reaps its own.
    """
    try:
        if spawn is not None and not children:
            await asyncio.wait([spawn])
            if spawn.exception() is None:
                children = spawn.result()
        if children:
            await asyncio.to_thread(_terminate_attached_frontends, children)
    finally:
        await facade.aclose()


def _serve_attached(config: FrontendLaunchConfig, num_frontends: int, launcher_uds: str) -> None:
    """Serve as one of the frontends the launcher spawned."""
    from tensorrt_llm.commands.serve import _diagnose_port_in_use

    if num_frontends == 1:
        raise click.UsageError(
            f"{LAUNCHER_UDS_ENV} is set, but num_serve_frontends is 1: "
            "only a multi-frontend launcher starts attached frontends"
        )
    frontend_id = _attached_frontend_id(num_frontends)
    postproc_total = _postproc_total(config.llm_args)
    host, port = config.host, config.port
    launcher_pid = os.getppid()
    _disable_responses_store(warn=False)
    family = _address_family(host, port)
    mode = MultiFrontendMode(num_frontends, is_attached_frontend=True)
    with _stopped_by_sigterm() as sigterm, contextlib.ExitStack() as stack:
        sock = stack.enter_context(_bind_shared_port(host, port, family))
        affinity_socket, _ = stack.enter_context(
            _frontend_affinity_sockets(
                host,
                port,
                family,
                mode,
                config.llm_args.get("serve_frontend_affinity_port_stride"),
                _diagnose_port_in_use,
            )
        )
        facade, server = _build_frontend(
            config,
            postproc_workers=postproc_share(postproc_total, num_frontends, frontend_id),
            serving=MultiFrontendServing(
                launcher_uds=launcher_uds, is_launcher=False, launcher_pid=launcher_pid
            ),
            after_server=_follow_launcher,
        )
        sockets = [sock]
        if affinity_socket is not None:
            sockets.append(affinity_socket)
        uvloop.run(
            _run_attached(
                config, mode, facade, server, sockets, launcher_uds, launcher_pid, sigterm
            )
        )


def _follow_launcher(facade: OpenEngineLLM, server: OpenAIServer) -> None:
    """Adapt the attached frontend's launcher watchdog to this path.

    The watchdog stops this frontend after the launcher's ``/health`` fails
    several times in a row. On this path ``/health`` reports the launcher's
    connection to the engine, which this frontend watches itself, so any
    answer counts as alive and only an unreachable launcher as a failure.
    When the launcher is lost, the frontend stops as the facade stops it on a
    lost engine: one SIGINT, with ``doing_shutdown`` set so that ``/health``
    sends no second one.
    """
    from .facade import _stop_frontend

    forwarder = server._launcher_forwarder
    get_status = forwarder.get_status

    async def answered(path: str, timeout_s: float = 5.0) -> int | None:
        status = await get_status(path, timeout_s)
        return None if status is None else 200

    def launcher_lost(error: BaseException) -> None:
        facade._executor._set_fatal_error(error)
        _stop_frontend(facade)

    forwarder.get_status = answered
    server._on_launcher_lost = launcher_lost


async def _wait_for_launcher(launcher_uds: str, launcher_pid: int, interval_s: float = 0.2) -> None:
    """Wait until the launcher listens, which it does once every frontend is READY.

    Raises:
        RuntimeError: The launcher exited first.
    """
    while True:
        if os.getppid() != launcher_pid:
            raise RuntimeError(
                f"The launcher frontend (pid {launcher_pid}) exited before it started serving"
            )
        try:
            _, writer = await asyncio.open_unix_connection(launcher_uds)
        except OSError:
            # Refused while the socket is bound but not listening yet.
            await asyncio.sleep(interval_s)
            continue
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()
        return


async def _run_attached(
    config: FrontendLaunchConfig,
    mode: MultiFrontendMode,
    facade: OpenEngineLLM,
    server: OpenAIServer,
    sockets: list[socket.socket],
    launcher_uds: str,
    launcher_pid: int,
    sigterm: _Sigterm,
) -> None:
    sigterm.cancel_on_signal(asyncio.current_task())
    _maybe_disable_gc()
    try:
        await facade.start()
        # Only a frontend connected to the engine counts as started.
        _signal_frontend_ready(mode)
        await _wait_for_launcher(launcher_uds, launcher_pid)
        await _serve(server, config, sockets, sigterm)
    finally:
        sigterm.stop_cancelling()
        await _finish(facade.aclose())

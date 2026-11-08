# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Several serving frontends in front of one OpenEngine engine (SV-MF).

With ``num_serve_frontends`` N > 1, N frontend processes share the serving port
through ``SO_REUSEPORT``; each builds its own facade, and so opens its own
connection to the engine (SV-MF-2). The process ``trtllm-serve`` started is
frontend 0, the launcher. It connects to the engine, then re-execs its command
line N - 1 times, as the in-process server does (``serve/_frontend_processes.py``),
and starts listening only once every attached frontend has connected to the
engine and reported READY (SV-MF-3). As with the in-process server:

- the attached frontends forward the launcher-owned routes to the launcher
  (``serve/multi_frontend.py``), and stop when the launcher is gone;
- a frontend that exits while serving is logged, and the others keep serving;
- the Responses API store is off, since a follow-up request may reach another
  frontend (SV-MF-4);
- ``--port 0`` and ``--report_addr`` need a single frontend.

``num_postprocess_workers`` is split between the frontends (SV-PP-1).
"""

from __future__ import annotations

import asyncio
import gc
import os
import shutil
import socket
import tempfile
import threading
from collections.abc import Callable
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
from tensorrt_llm.serve.multi_frontend import LAUNCHER_UDS_NAME, MultiFrontendServing

from . import contracts
from .contracts import FrontendLaunchConfig

if TYPE_CHECKING:
    from tensorrt_llm.serve.openai_server import OpenAIServer

    from .facade import OpenEngineLLM

# Set by the launcher for the frontends it spawns: the path of its Unix socket.
LAUNCHER_UDS_ENV = "TLLM_OPENENGINE_LAUNCHER_UDS"

# Longest path a Unix socket address holds, without its terminating NUL.
_MAX_UNIX_PATH = 107


def postproc_share(total: int, num_frontends: int, frontend_id: int) -> int:
    """Frontend ``frontend_id``'s share of ``total`` postprocessing workers (SV-PP-1).

    The workers are split evenly, the first frontends taking one more each
    when ``total`` does not divide; every frontend gets at least one unless
    ``total`` is 0.
    """
    if total <= 0:
        return 0
    base, extra = divmod(total, num_frontends)
    return max(1, base + (1 if frontend_id < extra else 0))


def launch_frontends(config: FrontendLaunchConfig) -> None:
    """Start ``num_serve_frontends`` frontends for the engine and serve until they stop.

    A single frontend is served as by
    :func:`~tensorrt_llm.serve.openengine_poc.contracts.default_launch_frontends`.

    Raises:
        click.UsageError: The configuration asks for something these frontends cannot do.
        OpenEngineStartupError: The engine is not usable; the message names the step.
        RuntimeError: The port is taken, or a frontend failed to start.
    """
    launcher_uds = os.environ.pop(LAUNCHER_UDS_ENV, None)
    num_frontends = _num_frontends(config.llm_args)
    if launcher_uds is not None:
        _serve_attached(config, num_frontends, launcher_uds)
    elif num_frontends == 1:
        contracts.default_launch_frontends(config)
    else:
        _serve_launcher(config, num_frontends)


def _num_frontends(llm_args: dict[str, Any]) -> int:
    value = llm_args.get("num_serve_frontends")
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise click.UsageError(f"num_serve_frontends must be a positive integer, got {value!r}")
    return value


def _postproc_total(llm_args: dict[str, Any]) -> int:
    value = llm_args.get("num_postprocess_workers") or 0
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise click.UsageError(
            f"num_postprocess_workers must be an integer, got {value!r}"
        ) from error


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
        RuntimeError: The address is taken; the message names the holder when it can.
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
    except Exception as error:
        logger.warning(f"Closing the OpenEngine facade failed: {error!r}")


def _build_frontend(
    config: FrontendLaunchConfig,
    *,
    postproc_workers: int,
    serving: MultiFrontendServing,
    before_server: Callable[[OpenEngineLLM], None] | None = None,
) -> tuple[OpenEngineLLM, OpenAIServer]:
    """This process's facade and the ``OpenAIServer`` on it, as ``launch_openengine_server`` builds them.

    ``before_server`` runs on the facade before the server is built.
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
        )
    except (ValueError, TypeError) as error:
        raise click.UsageError(str(error)) from error
    try:
        if before_server is not None:
            before_server(facade)
        server = OpenAIServer(
            generator=facade,
            model=config.served_model_name or config.llm_args["model"],
            tool_parser=config.tool_parser,
            server_role=None,
            metadata_server_cfg=None,
            multimodal_server_config=config.multimodal_server_config,
            chat_template=config.chat_template,
            allow_request_chat_template=config.allow_request_chat_template,
            input_processor_workers=config.num_input_processor_workers,
            media_load_workers=config.num_media_load_workers,
            multi_frontend_serving=serving,
        )
        adapt_server(server, facade)
        _apply_fastapi_middlewares(server.app, config.middleware)
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
    _disable_responses_store(warn=True)
    family = _address_family(host, port)
    mode = MultiFrontendMode(num_frontends, is_attached_frontend=False)
    # Private to the user (mkdtemp creates it 0700); holds the launcher's
    # socket and the shared Prometheus directory.
    ipc_dir = tempfile.mkdtemp(prefix="tllm-oe-frontends-")
    try:
        with (
            _reserve_multi_frontend_port(host, port, family),
            _bind_shared_port(host, port, family) as sock,
            _frontend_affinity_sockets(
                host,
                port,
                family,
                mode,
                config.llm_args.get("serve_frontend_affinity_port_stride"),
                _diagnose_port_in_use,
            ) as (affinity_socket, attached_affinity_sockets),
        ):
            uds_sock, uds_path = _bind_launcher_socket(ipc_dir)
            with uds_sock:
                facade, server = _build_frontend(
                    config,
                    postproc_workers=postproc_share(
                        _postproc_total(config.llm_args), num_frontends, 0
                    ),
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
                    )
                )
    finally:
        shutil.rmtree(ipc_dir, ignore_errors=True)


async def _run_launcher(
    config: FrontendLaunchConfig,
    num_frontends: int,
    facade: OpenEngineLLM,
    server: OpenAIServer,
    sockets: list[socket.socket],
    uds_path: str,
    attached_affinity_sockets: dict[int, socket.socket],
) -> None:
    _maybe_disable_gc()
    cancelled = threading.Event()
    children: list = []
    stop_watching: threading.Event | None = None
    try:
        await facade.start()
        children = await asyncio.to_thread(
            _spawn_frontend_group,
            num_frontends,
            extra_env={LAUNCHER_UDS_ENV: uds_path},
            cancelled=cancelled,
            affinity_sockets=attached_affinity_sockets,
        )
        stop_watching = _watch_attached_frontends(children)
        logger.info(
            f"All {num_frontends} serving frontends are connected to the OpenEngine engine "
            f"at {config.endpoint}; serving on {config.host}:{config.port}"
        )
        await server(config.host, config.port, sockets=sockets)
    finally:
        # Stops a spawn still waiting for READY; it then reaps its own children.
        cancelled.set()
        if stop_watching is not None:
            stop_watching.set()
        try:
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
    frontend_id = int(os.environ["TLLM_EXECUTOR_FRONTEND_ID"])
    host, port = config.host, config.port
    _disable_responses_store(warn=False)
    family = _address_family(host, port)
    mode = MultiFrontendMode(num_frontends, is_attached_frontend=True)
    with (
        _bind_shared_port(host, port, family) as sock,
        _frontend_affinity_sockets(
            host,
            port,
            family,
            mode,
            config.llm_args.get("serve_frontend_affinity_port_stride"),
            _diagnose_port_in_use,
        ) as (affinity_socket, _),
    ):
        facade, server = _build_frontend(
            config,
            postproc_workers=postproc_share(
                _postproc_total(config.llm_args), num_frontends, frontend_id
            ),
            serving=MultiFrontendServing(
                launcher_uds=launcher_uds, is_launcher=False, launcher_pid=os.getppid()
            ),
        )
        sockets = [sock]
        if affinity_socket is not None:
            sockets.append(affinity_socket)
        uvloop.run(_run_attached(config, mode, facade, server, sockets))


async def _run_attached(
    config: FrontendLaunchConfig,
    mode: MultiFrontendMode,
    facade: OpenEngineLLM,
    server: OpenAIServer,
    sockets: list[socket.socket],
) -> None:
    _maybe_disable_gc()
    try:
        await facade.start()
        # Only a frontend connected to the engine counts as started.
        _signal_frontend_ready(mode)
        await server(config.host, config.port, sockets=sockets)
    finally:
        await facade.aclose()

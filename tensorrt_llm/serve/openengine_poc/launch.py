# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the OpenAI HTTP server in front of an external OpenEngine engine."""

from __future__ import annotations

import asyncio
import gc
import os
import socket
from collections.abc import Sequence
from typing import Any

import uvloop

from tensorrt_llm.logger import logger

from .facade import DEFAULT_CONTROL_TIMEOUT_S, DEFAULT_READY_TIMEOUT_S, OpenEngineLLM, adapt_server


def _bind(host: str, port: int) -> socket.socket:
    """Bind the listening socket now, so ``port=0`` resolves to the port the kernel picked.

    Raises:
        RuntimeError: The address is taken; the message names the holder when it can.
    """
    from tensorrt_llm.commands.serve import _diagnose_port_in_use

    addr_info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    family = (
        socket.AF_INET6 if all(info[0] == socket.AF_INET6 for info in addr_info) else socket.AF_INET
    )
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        # As launch_server does: a restart is not refused during TIME_WAIT.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as error:
        sock.close()
        holder = _diagnose_port_in_use(port)
        raise RuntimeError(
            f"Failed to bind socket to {host}:{port}: {error}. Port holder(s): {holder}"
        ) from error
    except BaseException:
        sock.close()
        raise
    return sock


def launch_openengine_server(
    host: str,
    port: int,
    llm_args: dict[str, Any],
    endpoint: str,
    *,
    tool_parser: str | None = None,
    middleware: Sequence[str] = (),
    chat_template: str | None = None,
    served_model_name: str | None = None,
    allow_request_chat_template: bool = False,
    multimodal_server_config: Any = None,
    num_input_processor_workers: int = 8,
    num_media_load_workers: int = 8,
    ready_timeout: float = DEFAULT_READY_TIMEOUT_S,
    control_timeout: float = DEFAULT_CONTROL_TIMEOUT_S,
    allow_newer_schema: bool = False,
    report_addr: str | None = None,
    server_role: Any = None,
    disagg_cluster_config: Any = None,
    internal_disagg_auth_key: str | None = None,
) -> None:
    """Serve the OpenAI API from the engine at ``endpoint`` in this process.

    No model weights are loaded and no executor is started. The tokenizer is
    the one in ``llm_args``, else the model directory's. ``/health`` follows the
    engine; endpoints that need an in-process engine answer 501. With
    ``report_addr``, the socket is bound before the engine is awaited and the
    bound address is written to ``report_addr``, as the in-process server
    does. ``port=0`` (the kernel picks the port) needs ``report_addr`` or
    ``disagg_cluster_config`` to tell clients the bound port: it is written to
    ``report_addr`` when one is given and registered with the cluster when
    there is one.

    ``server_role`` (a ``ServerRole``), ``disagg_cluster_config`` and
    ``internal_disagg_auth_key`` make the frontend a disaggregated worker, as
    for the in-process server: ``OpenAIServer`` registers with the cluster once
    it serves, which is after the engine checks pass, and checks the
    orchestrator's signatures with the key.

    Raises:
        click.UsageError: The configuration asks for something this frontend cannot do.
        OpenEngineStartupError: The engine is not usable; the message names the step.
    """
    import click

    from tensorrt_llm.commands.serve import _apply_fastapi_middlewares, _publish_bound_address
    from tensorrt_llm.serve.openai_server import OpenAIServer

    if port == 0 and not report_addr and not disagg_cluster_config:
        raise click.UsageError(
            "--port 0 needs --report_addr or a disagg_cluster to tell clients the port"
        )
    try:
        facade = OpenEngineLLM.from_llm_args(
            llm_args,
            endpoint,
            ready_timeout=ready_timeout,
            control_timeout=control_timeout,
            allow_newer_schema=allow_newer_schema,
            server_role=server_role,
        )
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    try:
        server = OpenAIServer(
            generator=facade,
            model=served_model_name or llm_args["model"],
            tool_parser=tool_parser,
            server_role=server_role,
            metadata_server_cfg=None,
            disagg_cluster_config=disagg_cluster_config,
            multimodal_server_config=multimodal_server_config,
            chat_template=chat_template,
            allow_request_chat_template=allow_request_chat_template,
            input_processor_workers=num_input_processor_workers,
            media_load_workers=num_media_load_workers,
            internal_disagg_auth_key=internal_disagg_auth_key,
        )
        adapt_server(server, facade)
        _apply_fastapi_middlewares(server.app, middleware)

        sockets = None
        if report_addr or port == 0:
            sock = _bind(host, port)
            try:
                port = sock.getsockname()[1]
                if report_addr:
                    _publish_bound_address(report_addr, host, port)
            except BaseException:
                sock.close()
                raise
            sockets = [sock]
    except BaseException:
        # The facade may already hold a postprocessing pool.
        asyncio.run(facade.aclose())
        raise

    async def run() -> None:
        # As launch_server does.
        if os.getenv("TRTLLM_SERVER_DISABLE_GC", "0") == "1":
            gc.disable()
        try:
            await facade.start()
            logger.info(f"Serving the OpenEngine engine at {endpoint} on {host}:{port}")
            await server(host, port, sockets=sockets)
        finally:
            await facade.aclose()

    try:
        uvloop.run(run())
    finally:
        for sock in sockets or ():
            sock.close()

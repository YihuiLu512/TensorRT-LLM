# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the OpenAI HTTP server in front of an external OpenEngine engine."""

from __future__ import annotations

import gc
import os
from collections.abc import Sequence
from typing import Any

import uvloop

from tensorrt_llm.logger import logger

from .adapter import OpenEngineLLMAdapter


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
    ready_timeout: float = 600.0,
) -> None:
    """Serve ``/v1/chat/completions`` and ``/v1/completions`` from the engine at ``endpoint``.

    No model weights are loaded and no executor is started. The tokenizer is
    the one in ``llm_args``, else the model directory's.
    ``/health`` follows the engine. Other ``OpenAIServer`` endpoints are
    registered but not adapted to this path.
    """
    from tensorrt_llm.commands.serve import _apply_fastapi_middlewares
    from tensorrt_llm.serve.openai_server import OpenAIServer

    adapter = OpenEngineLLMAdapter.from_llm_args(llm_args, endpoint)
    server = OpenAIServer(
        generator=adapter,
        model=served_model_name or llm_args["model"],
        tool_parser=tool_parser,
        server_role=None,
        metadata_server_cfg=None,
        multimodal_server_config=multimodal_server_config,
        chat_template=chat_template,
        allow_request_chat_template=allow_request_chat_template,
        input_processor_workers=num_input_processor_workers,
        media_load_workers=num_media_load_workers,
    )
    _apply_fastapi_middlewares(server.app, middleware)

    async def run() -> None:
        # As launch_server does.
        if os.getenv("TRTLLM_SERVER_DISABLE_GC", "0") == "1":
            gc.disable()
        try:
            await adapter.start(ready_timeout=ready_timeout)
            logger.info(f"Serving the OpenEngine engine at {endpoint} on {host}:{port}")
            await server(host, port)
        finally:
            await adapter.aclose()

    uvloop.run(run())

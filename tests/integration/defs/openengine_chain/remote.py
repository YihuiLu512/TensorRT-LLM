# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RemoteOpenAIServer``'s entry into the chain."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import IO, Any

from .launch import ChainError, ServeHandle, launch_serve_command


def launch_for_remote_server(
    launch_cmd: Sequence[str],
    *,
    env: Mapping[str, str],
    llmapi_launch: bool,
    role: Any,
    sink: IO[Any] | int | None,
) -> ServeHandle:
    """Start ``launch_cmd`` as an engine plus a front end; the caller waits for ``/health``."""
    if llmapi_launch:
        raise ChainError("the OpenEngine chain does not support trtllm-llmapi-launch")
    if role is not None:
        raise ChainError(f"the OpenEngine chain does not support --server_role ({role})")
    return launch_serve_command(launch_cmd, env=env, sink=sink)

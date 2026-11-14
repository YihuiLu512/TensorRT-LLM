# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ASGI middleware that records each HTTP request a server answers, for the CPU disaggregated chain.

A worker frontend of :mod:`cpu_disagg_chain` loads it with
``--middleware cpu_disagg_chain_recorder.RecordHttpRequests``. When the
environment variable :data:`HTTP_LOG_ENV` names a file, each HTTP request
appends one JSON line to it just before the last part of its response is
sent, so a client that has the whole response finds the line (or, without
a complete response, once the application ends): the pid, method, path,
request headers, response status (None when no response was started) and
the request body. The body
is decoded as the orchestrator encodes it: msgpack when the request carries
:data:`MSGPACK_HEADER`, else JSON; a body that does not decode is kept as
text. Without the variable it passes requests through untouched.

It needs only the standard library, and ``msgspec`` for msgpack bodies.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

HTTP_LOG_ENV = "TRTLLM_CPU_CHAIN_HTTP_LOG"
# The header with which the orchestrator marks a msgpack body.
MSGPACK_HEADER = "x-trtllm-msgpack"


def _body(raw: bytes, headers: dict[str, str]) -> Any:
    if not raw:
        return None
    if headers.get(MSGPACK_HEADER) == "1":
        try:
            import msgspec
        except ImportError:
            msgspec = None
        if msgspec is not None:
            try:
                return msgspec.msgpack.decode(raw)
            except msgspec.DecodeError:
                pass
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return raw.decode("utf-8", errors="replace")


def read_http_log(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """The records in ``path``, oldest first; none if it does not exist yet."""
    try:
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except FileNotFoundError:
        return []


class RecordHttpRequests:
    """Pure ASGI middleware; see the module docstring."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.path = os.environ.get(HTTP_LOG_ENV) or None

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or self.path is None:
            await self.app(scope, receive, send)
            return
        chunks: list[bytes] = []
        status: int | None = None
        recorded = False

        def record() -> None:
            nonlocal recorded
            if recorded:
                return
            recorded = True
            headers = {
                name.decode("latin-1").lower(): value.decode("latin-1")
                for name, value in scope.get("headers", ())
            }
            self._write(
                {
                    "pid": os.getpid(),
                    "time": time.time(),
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "headers": headers,
                    "status": status,
                    "body": _body(b"".join(chunks), headers),
                }
            )

        async def receive_and_keep() -> dict[str, Any]:
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(message.get("body", b""))
            return message

        async def send_and_keep(message: dict[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body" and not message.get("more_body"):
                record()
            await send(message)

        try:
            await self.app(scope, receive_and_keep, send_and_keep)
        finally:
            record()

    def _write(self, record: dict[str, Any]) -> None:
        # A decoded msgpack body may hold bytes, which JSON has no type for.
        line = (json.dumps(record, ensure_ascii=False, default=repr) + "\n").encode("utf-8")
        # One write on an O_APPEND descriptor, so the lines of several processes do not interleave.
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(descriptor, line)
        finally:
            os.close(descriptor)

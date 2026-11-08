# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``create_lora_registry`` slot: register a request's LoRA adapter with the engine (OE-LORA).

A frontend keeps one registry per engine instance. The first request naming an
adapter registers it with ``Control.LoadLora`` (``lora_id`` and ``source_path``
taken from the request's ``lora_request``); later requests for the same name
reuse the registration. The id is the client's ``lora_int_id``, never one this
registry assigns, so frontends agree on an adapter's id whenever their clients
do; the engine refuses a name or an id already bound to another adapter.

:meth:`check` refuses, without a call, what the registry can tell is wrong, so
the facade reports it before a streamed response starts.

``lora_request.lora_ckpt_source`` is not sent: ``LoadLora`` has no field for it,
and the engine reads every adapter in the format it was started with.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .engine_client import EngineClient, ModelCapabilities

__all__ = ["create_lora_registry"]


def _fields(lora_request: Any) -> tuple[str, int, str]:
    """The adapter's name, id and path, checked.

    Raises:
        ValueError: A field is missing, empty, of the wrong type or out of range.
    """
    name = getattr(lora_request, "lora_name", None)
    int_id = getattr(lora_request, "lora_int_id", None)
    path = getattr(lora_request, "lora_path", None)
    if not isinstance(name, str) or not name:
        raise ValueError("lora_request.lora_name must be a non-empty string")
    if isinstance(int_id, bool) or not isinstance(int_id, int) or not 0 <= int_id < 1 << 63:
        raise ValueError(
            f"lora_request.lora_int_id must be an integer in [0, 2**63), got {int_id!r}"
        )
    if not isinstance(path, str) or not path:
        raise ValueError(
            "lora_request.lora_path must name the adapter's directory on the engine's machine"
        )
    return name, int_id, path


def _conflict(name: str, int_id: int, path: str, other: tuple[str, int, str]) -> ValueError:
    other_name, other_id, other_path = other
    return ValueError(
        f"LoRA adapter {name!r} (id {int_id}, path {path!r}) conflicts with the registered "
        f"adapter {other_name!r} (id {other_id}, path {other_path!r}); a name and an id each "
        "stay bound to one adapter"
    )


class _EngineLoraRegistry:
    """Registers each adapter with the engine the first time a request names it."""

    def __init__(self, client: EngineClient, capabilities: ModelCapabilities) -> None:
        self._client = client
        self._supported = capabilities.supports_lora
        self._engine_name = client.server_info().engine_name
        # name -> (name, id, path) of the adapters the engine accepted.
        self._by_name: dict[str, tuple[str, int, str]] = {}
        self._by_id: dict[int, tuple[str, int, str]] = {}
        # name -> (fields, task) of a LoadLora in flight, shared by the requests that wait on it.
        self._loading: dict[str, tuple[tuple[str, int, str], asyncio.Task[None]]] = {}

    def check(self, lora_request: Any) -> None:
        """Refuse ``lora_request`` if it cannot be registered, without asking the engine.

        Raises:
            ValueError: The engine does not declare LoRA support (OE-LORA-1), a
                field is malformed, or the name or id is bound, or being
                bound, to another adapter (OE-LORA-3).
        """
        self._checked(lora_request)

    def _checked(self, lora_request: Any) -> tuple[str, int, str]:
        if not self._supported:
            raise ValueError(
                "lora_request is not supported: the OpenEngine engine "
                f"{self._engine_name!r} does not declare LoRA support"
            )
        fields = _fields(lora_request)
        name, int_id, _ = fields
        for known in (self._by_name.get(name), self._by_id.get(int_id)):
            if known is not None and known != fields:
                # The registration stays as it is.
                raise _conflict(*fields, known)
        for other, _task in self._loading.values():
            if other != fields and (other[0] == name or other[1] == int_id):
                raise _conflict(*fields, other)
        return fields

    async def resolve(self, lora_request: Any) -> str | None:
        if lora_request is None:
            return None
        fields = self._checked(lora_request)
        name = fields[0]
        if self._by_name.get(name) == fields:
            return name
        loading = self._loading.get(name)
        if loading is None:
            task = asyncio.ensure_future(self._load(fields))
            self._loading[name] = (fields, task)
            task.add_done_callback(self._loaded)
        else:
            task = loading[1]
        # One request giving up does not cancel the registration others wait on.
        await asyncio.shield(task)
        return name

    def _loaded(self, task: asyncio.Task[None]) -> None:
        for name, (_, pending) in list(self._loading.items()):
            if pending is task:
                del self._loading[name]
        # Retrieved here so a failure no request waited for is not logged as unhandled;
        # the requests that wait on the task still get it.
        if not task.cancelled():
            task.exception()

    async def _load(self, fields: tuple[str, int, str]) -> None:
        from tensorrt_llm.grpc.openengine_client import OpenEngineRpcError

        name, int_id, path = fields
        try:
            await self._client.load_lora(name, int_id, path)
        except OpenEngineRpcError as error:
            import grpc

            # The engine refused this adapter: malformed fields, a path without
            # an adapter, a name or id bound to another adapter, more ids than
            # it keeps, or no LoRA support after all. Other failures
            # (unavailable, timed out) keep their own status.
            if error.code in (
                grpc.StatusCode.INVALID_ARGUMENT,
                grpc.StatusCode.FAILED_PRECONDITION,
                grpc.StatusCode.ALREADY_EXISTS,
                grpc.StatusCode.UNIMPLEMENTED,
                grpc.StatusCode.RESOURCE_EXHAUSTED,
            ):
                raise ValueError(
                    f"the OpenEngine engine {self._engine_name!r} did not register LoRA adapter "
                    f"{name!r} ({error.code.name}): {error.details}"
                ) from error
            raise
        self._by_name[name] = fields
        self._by_id[int_id] = fields


def create_lora_registry(client: EngineClient, capabilities: ModelCapabilities) -> Any:
    """The LoRA registry of a frontend connected to ``client`` (OE-LORA-2, OE-LORA-3)."""
    return _EngineLoraRegistry(client, capabilities)

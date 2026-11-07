# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoRA adapter registry behind the OpenEngine ``LoadLora`` family of RPCs.

The LLM API loads an adapter's weights when the first request that names its
id runs, and keeps them under that id for the life of the engine. ``LoadLora``
therefore records a name, an id and a path instead of loading weights,
``Generate`` resolves ``lora_name`` to a ``LoRARequest`` through this registry,
and an id stays bound to its first path even after ``UnloadLora``.

The registry lives in the frontend process, so LoRA is served only when the
engine has one frontend.
"""

import asyncio
import os
from dataclasses import dataclass

import grpc

from tensorrt_llm.executor.request import LoRARequest

from .bindings import lora_pb2

# Most distinct adapter ids one process accepts; each id's weights stay cached
# in the engine once a request has used it.
MAX_REGISTERED_LORAS = 4096

# Each frontend process would keep its own registry, so ids could collide
# across frontends; LoRA is served only with one.
LORA_UNSUPPORTED_MESSAGE = (
    "LoRA is not supported: the engine was started without lora_config or with "
    "several serve frontends"
)


class LoraNotRegisteredError(Exception):
    """``Generate`` named an adapter that this process has not registered."""


class LoraConflictError(Exception):
    """A registration reuses the name or the id of a different adapter."""


def check_fields(name: str, lora_id: int, source_path: str) -> None:
    """Check a ``LoraAdapter``'s fields.

    Raises:
        ValueError: A field is empty or out of range.
    """
    if not name:
        raise ValueError("adapter.lora_name must be non-empty")
    if lora_id < 0:
        raise ValueError(f"adapter.lora_id must be non-negative, got {lora_id}")
    if not source_path:
        raise ValueError("adapter.source_path must be non-empty")


def check_adapter_path(ckpt_source: str, source_path: str) -> None:
    """Check that ``source_path`` holds an adapter in the ``ckpt_source`` format.

    Raises:
        FileNotFoundError: An ``hf`` path is not a directory with
            ``adapter_config.json``, or a ``nemo`` path does not exist.
    """
    if ckpt_source == "hf":
        if not os.path.isfile(os.path.join(source_path, "adapter_config.json")):
            raise FileNotFoundError(
                f"adapter.source_path '{source_path}' is not a directory with adapter_config.json"
            )
    elif not os.path.exists(source_path):
        raise FileNotFoundError(f"adapter.source_path '{source_path}' does not exist")


@dataclass(frozen=True)
class RegisteredLora:
    """One registered adapter."""

    name: str
    lora_id: int
    source_path: str

    def to_proto(self) -> lora_pb2.LoraAdapter:
        return lora_pb2.LoraAdapter(
            lora_id=self.lora_id, lora_name=self.name, source_path=self.source_path
        )


class LoraRegistry:
    """Map adapter names to the ids and paths a ``LoRARequest`` needs.

    Args:
        ckpt_source: Checkpoint format of every registered adapter, taken from
            the engine's ``lora_config.lora_ckpt_source``.
        max_entries: Most distinct adapter ids this registry accepts over its
            lifetime.
    """

    def __init__(self, ckpt_source: str, max_entries: int = MAX_REGISTERED_LORAS) -> None:
        self._ckpt_source = ckpt_source
        self._max_entries = max_entries
        self._by_name: dict[str, RegisteredLora] = {}
        self._by_id: dict[int, RegisteredLora] = {}
        # Never shrinks: the engine keeps serving an id's first weights.
        self._path_by_id: dict[int, str] = {}

    @property
    def ckpt_source(self) -> str:
        return self._ckpt_source

    def register(self, name: str, lora_id: int, source_path: str) -> tuple[RegisteredLora, bool]:
        """Register an adapter; returns it and whether it was already registered.

        The caller checks ``source_path`` with ``check_adapter_path`` first.

        Raises:
            ValueError: A field is empty or out of range.
            LoraConflictError: The name or the id belongs to another adapter,
                or the id was bound to another path before.
            OverflowError: The registry has accepted ``max_entries`` ids.
        """
        check_fields(name, lora_id, source_path)
        adapter = RegisteredLora(name, lora_id, source_path)
        existing = self._by_name.get(name)
        if existing == adapter:
            return existing, True
        if existing is not None:
            raise LoraConflictError(
                f"LoRA '{name}' is registered with id {existing.lora_id} and path "
                f"'{existing.source_path}', not id {lora_id} and path '{source_path}'"
            )
        owner = self._by_id.get(lora_id)
        if owner is not None:
            raise LoraConflictError(f"LoRA id {lora_id} is registered to '{owner.name}'")
        bound_path = self._path_by_id.get(lora_id)
        if bound_path is None:
            if len(self._path_by_id) >= self._max_entries:
                raise OverflowError(f"at most {self._max_entries} LoRA ids can be registered")
        elif bound_path != source_path:
            raise LoraConflictError(
                f"LoRA id {lora_id} was used with path '{bound_path}' and the engine keeps "
                "its weights; register this adapter with a new id"
            )
        self._path_by_id[lora_id] = source_path
        self._by_name[name] = adapter
        self._by_id[lora_id] = adapter
        return adapter, False

    def find(self, name: str, lora_id: int, source_path: str) -> RegisteredLora | None:
        """The adapter registered with exactly these fields, or None."""
        existing = self._by_name.get(name)
        if existing == RegisteredLora(name, lora_id, source_path):
            return existing
        return None

    def unregister(self, name: str) -> RegisteredLora:
        """Remove an adapter by name; its id stays bound to its path.

        Raises:
            KeyError: No adapter has that name.
        """
        adapter = self._by_name.pop(name)
        del self._by_id[adapter.lora_id]
        return adapter

    def adapters(self) -> list[RegisteredLora]:
        """Registered adapters ordered by name."""
        return [self._by_name[name] for name in sorted(self._by_name)]

    def lora_request(self, name: str) -> LoRARequest:
        """The ``LoRARequest`` for a registered adapter name.

        Raises:
            LoraNotRegisteredError: No adapter has that name.
        """
        adapter = self._by_name.get(name)
        if adapter is None:
            raise LoraNotRegisteredError(
                f"LoRA '{name}' is not registered on this engine; call LoadLora first"
            )
        return LoRARequest(
            lora_name=adapter.name,
            lora_int_id=adapter.lora_id,
            lora_path=adapter.source_path,
            lora_ckpt_source=self._ckpt_source,
        )


async def load_lora(
    registry: LoraRegistry | None,
    request: lora_pb2.LoadLoraRequest,
    context: grpc.aio.ServicerContext,
) -> lora_pb2.LoadLoraResponse:
    """Serve ``Control.LoadLora``; ``registry`` is None when LoRA is not supported."""
    if registry is None:
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, LORA_UNSUPPORTED_MESSAGE)
    if not request.HasField("adapter"):
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "adapter must be set")
    adapter = request.adapter
    try:
        check_fields(adapter.lora_name, adapter.lora_id, adapter.source_path)
        # A repeated LoadLora succeeds even after the adapter files are gone.
        existing = registry.find(adapter.lora_name, adapter.lora_id, adapter.source_path)
        if existing is not None:
            return lora_pb2.LoadLoraResponse(adapter=existing.to_proto(), already_loaded=True)
        # Off the event loop: the path may be on a slow or remote filesystem.
        await asyncio.to_thread(check_adapter_path, registry.ckpt_source, adapter.source_path)
        registered, already_loaded = registry.register(
            adapter.lora_name, adapter.lora_id, adapter.source_path
        )
    except ValueError as error:
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
    except FileNotFoundError as error:
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
    except LoraConflictError as error:
        await context.abort(grpc.StatusCode.ALREADY_EXISTS, str(error))
    except OverflowError as error:
        await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))
    return lora_pb2.LoadLoraResponse(adapter=registered.to_proto(), already_loaded=already_loaded)


async def unload_lora(
    registry: LoraRegistry | None,
    request: lora_pb2.UnloadLoraRequest,
    context: grpc.aio.ServicerContext,
) -> lora_pb2.UnloadLoraResponse:
    """Serve ``Control.UnloadLora``.

    Only the registration is removed: the engine keeps the weights, and later
    requests for the name fail until it is loaded again.
    """
    if registry is None:
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, LORA_UNSUPPORTED_MESSAGE)
    try:
        adapter = registry.unregister(request.lora_name)
    except KeyError:
        await context.abort(
            grpc.StatusCode.NOT_FOUND, f"LoRA '{request.lora_name}' is not registered"
        )
    return lora_pb2.UnloadLoraResponse(adapter=adapter.to_proto())


async def list_loras(
    registry: LoraRegistry | None,
    request: lora_pb2.ListLorasRequest,
    context: grpc.aio.ServicerContext,
) -> lora_pb2.ListLorasResponse:
    """Serve ``Control.ListLoras``."""
    if registry is None:
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, LORA_UNSUPPORTED_MESSAGE)
    return lora_pb2.ListLorasResponse(
        adapters=[adapter.to_proto() for adapter in registry.adapters()]
    )


__all__ = [
    "LORA_UNSUPPORTED_MESSAGE",
    "LoraConflictError",
    "LoraNotRegisteredError",
    "LoraRegistry",
    "RegisteredLora",
    "check_adapter_path",
    "check_fields",
    "list_loras",
    "load_lora",
    "unload_lora",
]

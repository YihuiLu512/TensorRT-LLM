# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The handoff of a disaggregated request, between ``KvSessionSpec`` and ``DisaggregatedParams``.

This module implements the two functions :mod:`.contracts` names in
``HANDOFF_MODULE``:

- :func:`decode_kv_session`, on a context worker: the ``kv_session`` of the
  engine's ``PrefillReady`` becomes the LLM-level ``DisaggregatedParams`` of
  the HTTP response, plus the ``openengine_handoff_attributes`` text;
- :func:`encode_kv_session`, on a generation worker: the request's LLM-level
  ``DisaggregatedParams`` and its ``openengine_handoff_attributes`` become
  the ``kv_session`` of the decode call.

They mirror the TensorRT-LLM OpenEngine server, which packs the handoff into a
``KvSessionRef`` on the context engine and unpacks it on the generation
engine. Each mapped attribute (``HANDOFF_MAPPED_ATTRIBUTES``) is converted
field by field without changing its value: ``ctx_info_endpoint`` stays the
string the engine wrote, ``opaque_state`` crosses as the same base64 text,
ids are decimal strings and numbers that are integers come back as ints.
Every other attribute crosses unread in ``openengine_handoff_attributes``.
Encoding a session decoded from the server's packing gives the session back,
for its ``session_id``, its ``dp_rank`` and its attributes; a session the
server does not pack (a ``session_id`` with leading zeros, a ``dp_rank``
without the ``ctx_dp_rank`` attribute) comes back in the server's form.

The module is imported in postprocessing processes as well, so it does not
import the gRPC stack or the facade.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import math
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.grpc.openengine_conventions import (
    HANDOFF_DECODE_PREPARATION,
    HANDOFF_MAPPED_ATTRIBUTES,
    HANDOFF_REGISTERED_PASSTHROUGH_ATTRIBUTES,
    REQUEST_TYPE_CONTEXT_ONLY,
    parse_disagg_request_id,
)

from . import contracts
from .engine_client import KvEndpoint, KvSessionSpec

__all__ = [
    "OpenEngineDisaggregatedParams",
    "decode_kv_session",
    "encode_kv_session",
    "handoff_attributes_of",
    "unregistered_attributes",
    "with_handoff_attributes",
]

# Integers a Struct number holds exactly.
_STRUCT_INT_LIMIT = 1 << 53
_MAPPED = frozenset(HANDOFF_MAPPED_ATTRIBUTES)
_REGISTERED = _MAPPED | frozenset(HANDOFF_REGISTERED_PASSTHROUGH_ATTRIBUTES)


@dataclasses.dataclass(slots=True, kw_only=True)
class OpenEngineDisaggregatedParams(DisaggregatedParams):
    """The LLM-level params of a context worker's handoff, with the attributes they have no field for.

    Attributes:
        openengine_handoff_attributes: The value of the HTTP field of that
            name, or None; the response handlers copy it there.
    """

    openengine_handoff_attributes: str | None = None


def with_handoff_attributes(
    params: DisaggregatedParams, handoff_attributes: str | None
) -> OpenEngineDisaggregatedParams:
    """A copy of ``params`` carrying ``handoff_attributes`` as its ``openengine_handoff_attributes``."""
    fields = {
        field.name: getattr(params, field.name) for field in dataclasses.fields(DisaggregatedParams)
    }
    return OpenEngineDisaggregatedParams(**fields, openengine_handoff_attributes=handoff_attributes)


def handoff_attributes_of(params: Any) -> str | None:
    """The ``openengine_handoff_attributes`` an LLM-level params object carries; None for none."""
    return getattr(params, "openengine_handoff_attributes", None)


def unregistered_attributes(session: KvSessionSpec) -> list[str]:
    """The attributes of ``session`` that the conventions neither map nor register, sorted."""
    return sorted(key for key in session.attributes if key not in _REGISTERED)


# --- context worker: KvSessionSpec -> DisaggregatedParams ---------------------


def _int(value: Any, name: str) -> int:
    """``value`` as an int: an int, or a float holding an integer as a Struct number does."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"handoff attribute {name} is not a number: {value!r}")
    if isinstance(value, float) and not (math.isfinite(value) and value.is_integer()):
        raise ValueError(f"handoff attribute {name} is not an integer: {value!r}")
    return int(value)


def _ints(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"handoff attribute {name} is not a list: {value!r}")
    return [_int(item, name) for item in value]


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"handoff attribute {name} is not a string: {value!r}")
    return value


def _restore_counts(value: Any) -> Any:
    """``value`` with every float holding an integer turned back into an int, at any depth."""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, Mapping):
        return {key: _restore_counts(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_restore_counts(item) for item in value]
    return value


def _decode_log_probs(positions: Any) -> list[Any]:
    """``first_gen_log_probs`` as the engine holds it, from the engine's Struct form.

    Each position is a float (the simple format) or a list of
    ``[token_id, logprob, rank]`` candidates.
    """
    from tensorrt_llm.executor.result import Logprob

    if not isinstance(positions, (list, tuple)):
        raise ValueError(f"handoff attribute first_gen_log_probs is not a list: {positions!r}")
    result: list[Any] = []
    for position in positions:
        if isinstance(position, (list, tuple)):
            entry = {}
            for candidate in position:
                if not isinstance(candidate, (list, tuple)) or len(candidate) != 3:
                    raise ValueError(
                        "handoff attribute first_gen_log_probs holds a candidate that is not "
                        f"[token_id, logprob, rank]: {candidate!r}"
                    )
                token_id, logprob, rank = candidate
                entry[_int(token_id, "first_gen_log_probs")] = Logprob(
                    logprob=logprob,
                    rank=None if rank is None else _int(rank, "first_gen_log_probs"),
                )
            result.append(entry)
        elif isinstance(position, (int, float)) and not isinstance(position, bool):
            result.append(position)
        else:
            raise ValueError(
                f"handoff attribute first_gen_log_probs holds {position!r}, "
                "neither a logprob nor a list of candidates"
            )
    return result


def decode_kv_session(session: KvSessionSpec) -> tuple[DisaggregatedParams, str | None]:
    """The LLM-level params and the ``openengine_handoff_attributes`` of a context worker's handoff.

    The params are those of a ``context_only`` response. ``transfer_backend``
    and ``endpoints`` are not read: the endpoint is the
    ``ctx_info_endpoint`` attribute, as the engine wrote it.

    Raises:
        ValueError: An attribute or the ``session_id`` does not have the form
            the conventions give it.
    """
    attributes = dict(session.attributes)
    params = DisaggregatedParams(request_type=REQUEST_TYPE_CONTEXT_ONLY)
    if session.session_id:
        if not session.session_id.isascii() or not session.session_id.isdigit():
            raise ValueError(f"kv_session session_id is not a decimal id: {session.session_id!r}")
        params.ctx_request_id = int(session.session_id)
    if "ctx_info_endpoint" in attributes:
        params.ctx_info_endpoint = _string(attributes["ctx_info_endpoint"], "ctx_info_endpoint")
    if "opaque_state" in attributes:
        text = _string(attributes["opaque_state"], "opaque_state")
        try:
            params.opaque_state = base64.b64decode(text, validate=True)
        except binascii.Error as error:
            raise ValueError(f"handoff attribute opaque_state is not base64: {error}") from None
    if "first_gen_tokens" in attributes:
        params.first_gen_tokens = _ints(attributes["first_gen_tokens"], "first_gen_tokens")
    if "draft_tokens" in attributes:
        params.draft_tokens = _ints(attributes["draft_tokens"], "draft_tokens")
    if "disagg_request_id" in attributes:
        disagg_request_id = parse_disagg_request_id(attributes["disagg_request_id"])
        if disagg_request_id is None:
            raise ValueError(
                "handoff attribute disagg_request_id is not a valid decimal id: "
                f"{attributes['disagg_request_id']!r}"
            )
        params.disagg_request_id = disagg_request_id
    # The attribute keeps an explicit 0 apart from none; dp_rank is the fallback.
    if "ctx_dp_rank" in attributes:
        params.ctx_dp_rank = _int(attributes["ctx_dp_rank"], "ctx_dp_rank")
    elif session.dp_rank:
        params.ctx_dp_rank = session.dp_rank
    if "schedule_style" in attributes:
        params.schedule_style = DisaggScheduleStyle(
            _int(attributes["schedule_style"], "schedule_style")
        )
    if "ctx_usage" in attributes:
        ctx_usage = attributes["ctx_usage"]
        if not isinstance(ctx_usage, Mapping):
            raise ValueError(f"handoff attribute ctx_usage is not an object: {ctx_usage!r}")
        params.ctx_usage = _restore_counts(ctx_usage)
    if "first_gen_log_probs" in attributes:
        params.first_gen_log_probs = _decode_log_probs(attributes["first_gen_log_probs"])
    passthrough = {key: value for key, value in attributes.items() if key not in _MAPPED}
    return params, contracts.dump_handoff_attributes(passthrough)


# --- generation worker: DisaggregatedParams -> KvSessionSpec -------------------


def _check_struct_numbers(value: Any, name: str) -> None:
    """Refuse an integer a Struct number cannot hold exactly, at any depth."""
    if isinstance(value, bool):
        return
    if isinstance(value, int) and abs(value) > _STRUCT_INT_LIMIT:
        raise ValueError(
            f"openengine_handoff_attributes {name} holds {value}, an integer too large for a "
            "Struct number; larger integers travel as decimal strings"
        )
    if isinstance(value, Mapping):
        for item in value.values():
            _check_struct_numbers(item, name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_struct_numbers(item, name)


def _encode_log_probs(positions: Any) -> list[Any]:
    """``first_gen_log_probs`` in the engine's Struct form; the inverse of :func:`_decode_log_probs`."""
    result: list[Any] = []
    for position in positions:
        if isinstance(position, Mapping):
            result.append(
                [
                    [int(token_id), logprob.logprob, logprob.rank]
                    for token_id, logprob in position.items()
                ]
            )
        else:
            result.append(position)
    return result


def _endpoint(value: str, transfer_backend: str) -> tuple[KvEndpoint, ...]:
    """``ctx_info_endpoint`` split as the engine splits it; empty when it does not split."""
    try:
        parts = urlsplit(value if "://" in value else f"//{value}")
        host = parts.hostname or ""
        port = parts.port or 0
    except ValueError:
        return ()
    if not host:
        return ()
    return (KvEndpoint(host=host, port=port, protocol=parts.scheme or transfer_backend or ""),)


def _plain_usage(ctx_usage: Any) -> Any:
    """``ctx_usage`` as plain data: the HTTP layer may hand over a pydantic model."""
    dump = getattr(ctx_usage, "model_dump", None)
    return dump() if callable(dump) else dict(ctx_usage)


def encode_kv_session(
    params: Any, handoff_attributes: str | None, transfer_backend: str
) -> KvSessionSpec:
    """The ``kv_session`` of a generation worker's decode call.

    ``session_id`` is ``ctx_request_id`` in decimal; ``dp_rank`` is
    ``ctx_dp_rank`` (0 when unset); ``endpoints`` is ``ctx_info_endpoint``
    split as the engine splits it; the attributes are the mapped fields that
    are set, plus the attributes ``handoff_attributes`` holds.

    Raises:
        ValueError: The request cannot be expressed as a handoff, reported as
            HTTP 400: ``handoff_attributes`` is malformed, names a mapped
            attribute, holds ``trtllm_decode_preparation`` (multimodal requests
            are not served disaggregated) or an integer beyond 2^53; or the
            params set ``first_gen_logits``, ``remote_tail_start`` or
            multimodal positions, which a handoff does not carry.
    """
    if params.first_gen_logits:
        raise ValueError(
            "disaggregated_params.first_gen_logits is not supported when serving through an "
            "OpenEngine engine: a handoff does not carry generation logits"
        )
    if params.remote_tail_start is not None:
        raise ValueError(
            "disaggregated_params.remote_tail_start is not supported when serving through an "
            "OpenEngine engine"
        )
    if params.multimodal_positions is not None or params.multimodal_lengths is not None:
        raise ValueError(
            "a multimodal disaggregated request is not supported when serving through an "
            "OpenEngine engine"
        )
    try:
        passthrough = contracts.load_handoff_attributes(handoff_attributes)
    except RecursionError:
        raise ValueError("openengine_handoff_attributes is nested too deeply") from None
    mapped = sorted(_MAPPED.intersection(passthrough))
    if mapped:
        raise ValueError(
            "openengine_handoff_attributes names attributes that disaggregated_params "
            f"carries: {', '.join(mapped)}"
        )
    if HANDOFF_DECODE_PREPARATION in passthrough:
        raise ValueError(
            f"openengine_handoff_attributes holds {HANDOFF_DECODE_PREPARATION}: a multimodal "
            "disaggregated request is not supported when serving through an OpenEngine engine"
        )
    try:
        for key, value in passthrough.items():
            _check_struct_numbers(value, key)
    except RecursionError:
        raise ValueError("openengine_handoff_attributes is nested too deeply") from None

    attributes: dict[str, Any] = {}
    if params.ctx_info_endpoint is not None:
        attributes["ctx_info_endpoint"] = params.ctx_info_endpoint
    if params.opaque_state is not None:
        attributes["opaque_state"] = base64.b64encode(params.opaque_state).decode("ascii")
    if params.first_gen_tokens is not None:
        attributes["first_gen_tokens"] = [int(token) for token in params.first_gen_tokens]
    if params.draft_tokens is not None:
        attributes["draft_tokens"] = [int(token) for token in params.draft_tokens]
    if params.disagg_request_id is not None:
        attributes["disagg_request_id"] = str(int(params.disagg_request_id))
    if params.ctx_dp_rank is not None:
        attributes["ctx_dp_rank"] = int(params.ctx_dp_rank)
    if params.schedule_style is not None:
        attributes["schedule_style"] = int(params.schedule_style)
    if params.ctx_usage is not None:
        attributes["ctx_usage"] = _plain_usage(params.ctx_usage)
    if params.first_gen_log_probs is not None:
        attributes["first_gen_log_probs"] = _encode_log_probs(params.first_gen_log_probs)
    attributes.update(passthrough)
    endpoints = (
        _endpoint(params.ctx_info_endpoint, transfer_backend) if params.ctx_info_endpoint else ()
    )
    return KvSessionSpec(
        session_id="" if params.ctx_request_id is None else str(int(params.ctx_request_id)),
        transfer_backend=transfer_backend,
        dp_rank=int(params.ctx_dp_rank or 0),
        endpoints=endpoints,
        attributes=attributes,
    )

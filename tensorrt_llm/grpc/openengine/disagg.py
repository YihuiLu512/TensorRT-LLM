# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The disaggregated-serving handoff: KvSessionRef <-> DisaggregatedParams.

A context worker returns its handoff as a PrefillReady carrying a KvSessionRef;
a generation worker resumes it by echoing that session back on its request.
"""

import base64
import math
import os
import secrets
from collections.abc import Mapping, Sequence
from typing import Any, Optional
from urllib.parse import urlsplit

from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict

from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.grpc.openengine_conventions import (
    CAP_DISAGG_GENERATION_FIRST,
    CAP_DISAGG_ORIG_ELAPSED_S,
    CAP_REQ_DISAGG_REQUEST_ID,
    DISAGG_REQUEST_ID_LIMIT,
    DISAGG_REQUEST_ID_MIN,
    DISAGG_REQUEST_KEY_CAPABILITY,
    REQ_DISAGG_REQUEST_ID,
    REQ_ORIG_ELAPSED_S,
    REQ_REQUEST_TYPE,
    REQUEST_TYPE_CONTEXT_ONLY,
    REQUEST_TYPE_GENERATION_ONLY,
    parse_disagg_request_id,
    request_bucket,
)

from .bindings import generation_pb2, kv_pb2
from .errors import UnsupportedFeatureError
from .formatting import clamp_logprob

# OpenEngine has no native request-type field, so the phase of a disaggregated
# request is carried in the request's `extra` Struct under this key. A request
# that carries a `kv.session` handle is always treated as generation_only.
_REQUEST_TYPE_KEY = REQ_REQUEST_TYPE

# Set to "1" in the generation engine's environment, the engine serves a
# kv.session without any context worker to fetch from (the KV is not
# transferred); the same variable puts the executor in that mode.
GEN_ONLY_BENCHMARK_ENV = "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY"


class DisaggUnavailableError(Exception):
    """A disaggregated request reached a server that has no KV cache transceiver."""


def disagg_capabilities(kv_transfer_enabled: bool) -> dict[str, bool]:
    """The disaggregated-serving capability keys of conventions version 4, as this server reports them.

    The request keys work only on a server with a KV cache transceiver. Generation-first
    scheduling (``schedule_style``) is not supported.
    """
    return {
        CAP_REQ_DISAGG_REQUEST_ID: kv_transfer_enabled,
        CAP_DISAGG_GENERATION_FIRST: False,
        CAP_DISAGG_ORIG_ELAPSED_S: kv_transfer_enabled,
    }


def is_disaggregated_request(request: generation_pb2.GenerateRequest) -> bool:
    """Whether the request is one leg of a disaggregated request.

    That is a request with a ``kv.session``, or one whose ``extra.request_type`` is
    context_only or generation_only.
    """
    if request.HasField("kv") and request.kv.HasField("session"):
        return True
    if not request.HasField("extra"):
        return False
    value = request.extra.fields.get(_REQUEST_TYPE_KEY)
    return value is not None and value.string_value in (
        REQUEST_TYPE_CONTEXT_ONLY,
        REQUEST_TYPE_GENERATION_ONLY,
    )


def accepted_bucket(params: Optional[DisaggregatedParams]) -> str:
    """The ``generate_accepted.<bucket>`` counter a request with these params counts in."""
    request_type = None if params is None else params.request_type
    return request_bucket(
        has_kv_session=request_type == REQUEST_TYPE_GENERATION_ONLY,
        request_type=request_type,
    )


def _check_disagg_keys(extra: Mapping[str, Any], declared: Mapping[str, bool]) -> None:
    """Reject a disaggregated-serving request key whose capability is not declared."""
    for key, capability in DISAGG_REQUEST_KEY_CAPABILITY.items():
        if key in extra and not declared.get(capability, False):
            raise UnsupportedFeatureError(f"extra key '{key}' is not supported by this engine")


def _reject_off_leg(extra: Mapping[str, Any], keys: Sequence[str], leg: str) -> None:
    for key in keys:
        if key in extra:
            raise ValueError(f"extra.{key} is not accepted on {leg}")


def _orig_elapsed_s(value: Any) -> float:
    # Struct numbers arrive as float; a bool is a separate Struct kind and must not pass.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"extra.{REQ_ORIG_ELAPSED_S} must be a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"extra.{REQ_ORIG_ELAPSED_S} must be a finite number >= 0, got {value}")
    return float(value)


_DISAGG_REQUEST_TYPES = {"context_only", "generation_only", "context_and_generation"}


# generation_only is deliberately not in the set above: the phase needs the
# context handoff (an address plus an id), which only a kv.session can carry.
# Naming it through `extra` alone yields params whose every id and address field
# is None, and the engine then waits out its receive timeout on the executor
# loop against a session no context worker is keyed under -- the engine-wide
# stall _validate_kv_session exists to prevent, reached through the path that
# does not call it.
_EXTRA_REQUEST_TYPES = _DISAGG_REQUEST_TYPES - {"generation_only"}


def _struct_to_dict(struct: struct_pb2.Struct) -> dict[str, Any]:
    # Prepared media metadata is decoded separately, not as KV rendezvous state.
    return {
        key: MessageToDict(value)
        for key, value in struct.fields.items()
        if key != "trtllm_decode_preparation"
    }


def _endpoint_to_str(endpoints: Sequence[Any]) -> Optional[str]:
    if not endpoints:
        return None
    endpoint = endpoints[0]
    if not endpoint.host:
        return None
    return f"{endpoint.host}:{endpoint.port}" if endpoint.port else endpoint.host


def _split_endpoint(value: str) -> Optional[tuple[str, int, str]]:
    """Split an engine endpoint into (host, port, scheme).

    The engine's ctx_info_endpoint is an opaque URL, so a bare ``partition(":")``
    would take the scheme for the host and drop the rest.
    """
    try:
        parts = urlsplit(value if "://" in value else f"//{value}")
        host = parts.hostname or ""
        port = parts.port or 0
    except ValueError:
        return None
    if not host:
        return None
    return host, port, parts.scheme


def kv_endpoint(endpoint: str, transfer_backend: str) -> Optional[kv_pb2.KvEndpoint]:
    """The KvEndpoint of a context info endpoint, as a PrefillReady carries it.

    None when the endpoint does not parse or has no host.
    """
    parts = _split_endpoint(endpoint)
    if parts is None:
        return None
    host, port, scheme = parts
    return kv_pb2.KvEndpoint(host=host, port=port, protocol=scheme or transfer_backend or "")


def _serialize_first_gen_log_probs(first_gen_log_probs: Sequence[Any]) -> list[Any]:
    """Serialize first-gen logprobs into a Struct-storable form.

    ``list[dict[int, Logprob]] | list[float]``: each position is either a float
    (simple format) or a list of ``[token_id, logprob, rank]`` triples (verbose).
    """
    positions: list[Any] = []
    for pos in first_gen_log_probs:
        if isinstance(pos, dict):
            positions.append(
                [
                    [
                        int(token_id),
                        clamp_logprob(lp.logprob),
                        (lp.rank if lp.rank is not None else None),
                    ]
                    for token_id, lp in pos.items()
                ]
            )
        else:
            positions.append(clamp_logprob(pos))
    return positions


def _deserialize_first_gen_log_probs(positions: Sequence[Any]) -> list[Any]:
    """Inverse of :func:`_serialize_first_gen_log_probs`, rebuilding Logprob objects."""
    result: list[Any] = []
    for pos in positions:
        if isinstance(pos, list):
            result.append(
                {
                    int(token_id): Logprob(
                        logprob=clamp_logprob(logprob), rank=None if rank is None else int(rank)
                    )
                    for token_id, logprob, rank in pos
                }
            )
        else:
            result.append(clamp_logprob(pos))
    return result


def _validate_kv_session(
    session: Any, attrs: Mapping[str, Any], gen_only_benchmark: bool = False
) -> None:
    """Reject a session the generation worker cannot resolve.

    The engine consumes the handoff on its executor loop with no timeout and no
    guard: a session naming an unreachable context worker blocks that loop
    indefinitely, so every later request on this engine stalls too. Validating
    what we can here turns an engine-wide hang into one INVALID_ARGUMENT.

    With ``gen_only_benchmark`` the session need not locate a context worker:
    the engine then fetches nothing.

    ``attrs`` is the caller's already-converted attributes Struct: converting it
    twice would walk the whole handoff -- including first_gen_log_probs -- twice
    per request on the decode server's event loop.
    """
    endpoint = attrs.get("ctx_info_endpoint") or _endpoint_to_str(session.endpoints)
    if endpoint:
        parts = _split_endpoint(str(endpoint))
        if parts is None or not parts[1]:
            raise ValueError(f"kv.session ctx_info_endpoint '{endpoint}' is not a usable address")
    elif not attrs.get("opaque_state") and not gen_only_benchmark:
        # One or the other locates the context worker: an endpoint on builds that
        # fetch its info over ZMQ, opaque_state on builds that carry it inline.
        raise ValueError("kv.session carries neither a ctx_info_endpoint nor an opaque_state")
    if not session.session_id and attrs.get("disagg_request_id") is None:
        raise ValueError("kv.session carries neither a session_id nor a disagg_request_id")


def disaggregated_params_from_request(
    request: generation_pb2.GenerateRequest,
    *,
    kv_transfer_enabled: bool = True,
    gen_only_benchmark: Optional[bool] = None,
) -> Optional[DisaggregatedParams]:
    """Map an OpenEngine request to TensorRT-LLM disaggregated-serving params.

    Returns None for a request that names no phase. A ``kv.session`` handle marks
    a generation_only request and is decoded back into the context handoff
    (``opaque_state`` / ``ctx_request_id`` / ``first_gen_tokens`` / endpoint /
    dp_rank). Otherwise the phase comes from ``extra["request_type"]``.

    A context_only request takes its ``disagg_request_id`` from
    ``extra["disagg_request_id"]`` when present, and a generation_only request
    its ``orig_elapsed_s`` from ``extra["orig_elapsed_s"]``.

    Args:
        request: The Generate request.
        kv_transfer_enabled: Whether the server has a KV cache transceiver. Without
            one, a disaggregated request (:func:`is_disaggregated_request`) is refused.
        gen_only_benchmark: Whether a kv.session may omit the context worker's
            location; None reads :data:`GEN_ONLY_BENCHMARK_ENV`.

    Raises:
        DisaggUnavailableError: A disaggregated request and no KV cache transceiver.
        UnsupportedFeatureError: A request key of an undeclared capability
            (:func:`disagg_capabilities`).
        ValueError: The request is malformed.
    """
    if not kv_transfer_enabled and is_disaggregated_request(request):
        raise DisaggUnavailableError(
            "this engine has no KV cache transceiver (cache_transceiver_config is not set), "
            "so it cannot serve either leg of a disaggregated request (context_only, "
            "generation_only, or a request with a kv.session)"
        )
    extra = _struct_to_dict(request.extra) if request.HasField("extra") else {}
    _check_disagg_keys(extra, disagg_capabilities(kv_transfer_enabled))
    has_session = request.HasField("kv") and request.kv.HasField("session")
    if has_session:
        _reject_off_leg(extra, (REQ_DISAGG_REQUEST_ID,), "a request with a kv.session")
        if gen_only_benchmark is None:
            gen_only_benchmark = os.environ.get(GEN_ONLY_BENCHMARK_ENV) == "1"
        session = request.kv.session
        attrs = _struct_to_dict(session.attributes_struct)
        _validate_kv_session(session, attrs, gen_only_benchmark)
        params = DisaggregatedParams(request_type="generation_only")
        if REQ_ORIG_ELAPSED_S in extra:
            params.orig_elapsed_s = _orig_elapsed_s(extra[REQ_ORIG_ELAPSED_S])
        if session.session_id:
            try:
                params.ctx_request_id = int(session.session_id)
            except ValueError as error:
                raise ValueError(
                    "kv.session.session_id must be an integer context request id"
                ) from error
        # Prefer the verbatim endpoint; the structured fields are a fallback for
        # a peer that did not carry it.
        endpoint = attrs.get("ctx_info_endpoint") or _endpoint_to_str(session.endpoints)
        if endpoint:
            params.ctx_info_endpoint = str(endpoint)
        opaque_state = attrs.get("opaque_state")
        if opaque_state:
            params.opaque_state = base64.b64decode(opaque_state, validate=True)
        first_gen_tokens = attrs.get("first_gen_tokens")
        if first_gen_tokens:
            params.first_gen_tokens = [int(token) for token in first_gen_tokens]
        draft_tokens = attrs.get("draft_tokens")
        if draft_tokens:
            params.draft_tokens = [int(token) for token in draft_tokens]
        disagg_request_id = attrs.get("disagg_request_id")
        if disagg_request_id is not None:
            params.disagg_request_id = int(disagg_request_id)
        # ctx_dp_rank: prefer the attribute (preserves an explicit 0 and, by its
        # absence, an unset None) and fall back to the native dp_rank field.
        if "ctx_dp_rank" in attrs:
            params.ctx_dp_rank = int(attrs["ctx_dp_rank"])
        elif session.dp_rank:
            params.ctx_dp_rank = session.dp_rank
        schedule_style = attrs.get("schedule_style")
        if schedule_style is not None:
            params.schedule_style = DisaggScheduleStyle(int(schedule_style))
        ctx_usage = attrs.get("ctx_usage")
        if ctx_usage is not None:
            # Struct has only a double number type, so token counts come back as
            # floats; restore the integer typing the engine and Usage expect.
            params.ctx_usage = {
                key: int(value) if isinstance(value, float) and value.is_integer() else value
                for key, value in ctx_usage.items()
            }
        first_gen_log_probs = attrs.get("first_gen_log_probs")
        if first_gen_log_probs is not None:
            params.first_gen_log_probs = _deserialize_first_gen_log_probs(first_gen_log_probs)
        return params

    request_type = extra.get(_REQUEST_TYPE_KEY)
    if request_type is None:
        _reject_off_leg(extra, (REQ_DISAGG_REQUEST_ID, REQ_ORIG_ELAPSED_S), "an aggregated request")
        return None
    if request_type == "generation_only":
        raise ValueError(
            "a generation_only request must carry its context handoff in "
            "kv.session; extra.request_type alone cannot address one"
        )
    if request_type not in _EXTRA_REQUEST_TYPES:
        raise ValueError(f"extra.{_REQUEST_TYPE_KEY} must be one of {sorted(_EXTRA_REQUEST_TYPES)}")
    params = DisaggregatedParams(request_type=request_type)
    if request_type != REQUEST_TYPE_CONTEXT_ONLY:
        _reject_off_leg(extra, (REQ_DISAGG_REQUEST_ID, REQ_ORIG_ELAPSED_S), "an aggregated request")
        return params
    _reject_off_leg(extra, (REQ_ORIG_ELAPSED_S,), "a context_only request")
    if REQ_DISAGG_REQUEST_ID in extra:
        disagg_request_id = parse_disagg_request_id(extra[REQ_DISAGG_REQUEST_ID])
        if disagg_request_id is None:
            raise ValueError(
                f"extra.{REQ_DISAGG_REQUEST_ID} must be a decimal string without sign or "
                f"leading zeros in [{DISAGG_REQUEST_ID_MIN}, {DISAGG_REQUEST_ID_LIMIT})"
            )
        params.disagg_request_id = disagg_request_id
    else:
        # Local context IDs can collide across independent prefill engines.
        params.disagg_request_id = DISAGG_REQUEST_ID_MIN + secrets.randbelow(
            DISAGG_REQUEST_ID_LIMIT - DISAGG_REQUEST_ID_MIN
        )
    # This server schedules every prefill context-first, and says so explicitly as
    # the HTTP orchestrator does: where the context side checks KV ownership, an
    # unset style skips the checks that CONTEXT_FIRST runs.
    params.schedule_style = DisaggScheduleStyle.CONTEXT_FIRST
    return params


def prefill_ready_response(
    request_id: str,
    disaggregated_params: Any,
    transfer_backend: str,
    usage: Optional[generation_pb2.Usage] = None,
    decode_preparation: Optional[str] = None,
) -> generation_pb2.GenerateResponse:
    """Build the PrefillReady event a context worker returns after prefill.

    Packs the TensorRT-LLM context handoff into a KvSessionRef: ``ctx_request_id``
    -> session_id, ``ctx_dp_rank`` -> dp_rank, ``ctx_info_endpoint`` -> endpoint,
    and the fields with no native slot (opaque_state as base64, first/draft tokens,
    disagg_request_id, schedule_style, ctx_usage) into the
    session's attributes Struct.

    ``first_gen_log_probs`` is carried (serialized) so logprobs-in-disagg returns
    the first token's logprob on the generation worker. ``first_gen_logits`` is
    NOT carried: OpenEngine has no "return generation logits" request field, so it
    is never populated for a gRPC request (like beam search, it is unreachable).
    """
    ctx_request_id = disaggregated_params.ctx_request_id
    session = kv_pb2.KvSessionRef(
        session_id="" if ctx_request_id is None else str(ctx_request_id),
        transfer_backend=transfer_backend or "",
        dp_rank=disaggregated_params.ctx_dp_rank or 0,
    )
    attributes: dict[str, Any] = {}
    if decode_preparation is not None:
        attributes["trtllm_decode_preparation"] = decode_preparation
    endpoint = disaggregated_params.ctx_info_endpoint
    if endpoint:
        # Carried verbatim as well as split: the generation worker connects a
        # ZMQ socket straight to this string, so any lossy round trip through
        # the structured fields would hand it an unusable address.
        attributes["ctx_info_endpoint"] = endpoint
        located = kv_endpoint(endpoint, transfer_backend)
        if located is not None:
            session.endpoints.add().CopyFrom(located)
    opaque_state = disaggregated_params.opaque_state
    if opaque_state is not None:
        attributes["opaque_state"] = base64.b64encode(opaque_state).decode("utf-8")
    first_gen_tokens = disaggregated_params.first_gen_tokens
    if first_gen_tokens:
        attributes["first_gen_tokens"] = list(first_gen_tokens)
    draft_tokens = disaggregated_params.draft_tokens
    if draft_tokens:
        attributes["draft_tokens"] = list(draft_tokens)
    disagg_request_id = disaggregated_params.disagg_request_id
    if disagg_request_id is not None:
        attributes["disagg_request_id"] = str(disagg_request_id)
    ctx_dp_rank = disaggregated_params.ctx_dp_rank
    if ctx_dp_rank is not None:
        attributes["ctx_dp_rank"] = ctx_dp_rank
    schedule_style = disaggregated_params.schedule_style
    if schedule_style is not None:
        attributes["schedule_style"] = int(schedule_style)
    ctx_usage = disaggregated_params.ctx_usage
    if ctx_usage is not None:
        attributes["ctx_usage"] = dict(ctx_usage)
    first_gen_log_probs = disaggregated_params.first_gen_log_probs
    if first_gen_log_probs is not None:
        attributes["first_gen_log_probs"] = _serialize_first_gen_log_probs(first_gen_log_probs)
    if attributes:
        struct = struct_pb2.Struct()
        struct.update(attributes)
        session.attributes_struct.CopyFrom(struct)
    return generation_pb2.GenerateResponse(
        request_id=request_id,
        prefill_ready=generation_pb2.PrefillReady(kv_session=session),
        usage=usage,
    )


__all__ = [
    "DisaggUnavailableError",
    "GEN_ONLY_BENCHMARK_ENV",
    "accepted_bucket",
    "disagg_capabilities",
    "disaggregated_params_from_request",
    "is_disaggregated_request",
    "kv_endpoint",
    "prefill_ready_response",
]

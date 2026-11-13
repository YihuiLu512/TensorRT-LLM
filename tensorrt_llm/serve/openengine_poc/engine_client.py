# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The interface trtllm-serve uses to run requests on an engine, and the types it exchanges.

``EngineClient`` is internal to trtllm-serve: engines do not implement it. The
OpenEngine implementation translates its requests and events to and from
OpenEngine v1 messages. Besides running requests and reading the engine's
state, it covers the LoRA and KV-event methods the OpenEngine path needs.

A request is accepted or rejected before ``generate`` yields its first event:
an error before acceptance is raised by the first ``__anext__``, an error after
acceptance is raised while iterating. Closing the iterator cancels the request.

Served disaggregated, a request is two Generate calls on two engines: a
prefill call (``request_type="context_only"``), which ends with a
:class:`PrefillReady` event carrying the handoff, and a decode call, which
carries that handoff in ``kv_session``. A prefill call is not cancelled once
its :class:`PrefillReady` has arrived, since the engine would free the KV
cache it just handed over: the caller reads such a stream to its end rather
than closing the iterator.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from tensorrt_llm.grpc.openengine_conventions import ModelCapabilities

__all__ = [
    "DisaggRequestType",
    "EngineClient",
    "EngineEvent",
    "EngineHealth",
    "EngineInfo",
    "EngineLoad",
    "EngineRequest",
    "EngineRole",
    "FinishReason",
    "GuidedMode",
    "GuidedSpec",
    "HealthState",
    "KvConnectorInfo",
    "KvEndpoint",
    "KvEventSource",
    "KvSessionSpec",
    "LoraAdapter",
    "MediaSpec",
    "Modality",
    "ModelCapabilities",
    "OutputFinished",
    "PrefillReady",
    "PromptInfo",
    "TokenDelta",
    "TokenLogprob",
    "Usage",
]

GuidedMode = Literal["json_schema", "regex", "ebnf_grammar", "structural_tag", "json_object"]
Modality = Literal["image", "video", "audio"]
FinishReason = Literal["stop", "length", "cancelled"]
HealthState = Literal["starting", "ready", "degraded", "not_ready", "unspecified"]
# ``ServerInfo.engine_role``; logged only, never used to decide a worker's role.
EngineRole = Literal["unspecified", "aggregated", "prefill", "decode"]
# The ``request_type`` a call sends in ``extra``; a decode call sends none.
DisaggRequestType = Literal["context_only", "context_and_generation"]


@dataclass(frozen=True)
class GuidedSpec:
    """One guided-decoding constraint.

    Attributes:
        mode: The constraint kind, named as the OpenEngine ``GuidedDecoding`` oneof field.
        value: The schema, regex, grammar or structural tag; None for ``json_object``.
    """

    mode: GuidedMode
    value: str | None = None


@dataclass(frozen=True)
class MediaSpec:
    """One media item of a prompt; exactly one of ``url``, ``data_uri`` and ``raw_bytes`` is set.

    Attributes:
        uuid: An id the engine may use to reuse its media cache, for example an mm hash.
    """

    modality: Modality
    url: str | None = None
    data_uri: str | None = None
    raw_bytes: bytes | None = None
    mime_type: str | None = None
    uuid: str | None = None

    def __post_init__(self) -> None:
        sources = [s for s in (self.url, self.data_uri, self.raw_bytes) if s is not None]
        if len(sources) != 1:
            raise ValueError("a MediaSpec needs exactly one of url, data_uri and raw_bytes")


@dataclass(frozen=True)
class KvEndpoint:
    """An OpenEngine ``KvEndpoint``."""

    host: str
    port: int
    protocol: str = ""


@dataclass(frozen=True)
class KvSessionSpec:
    """An OpenEngine ``KvSessionRef``: the handoff from the context engine to the generation engine.

    Attributes:
        session_id: The context request id as a decimal string; empty if the engine gave none.
        transfer_backend: An informational label; the transfer does not depend on it.
        dp_rank: The context engine's attention-DP rank.
        endpoints: The context engine's endpoint split into parts; informational,
            since ``attributes["ctx_info_endpoint"]`` holds it as the engine wrote it.
        attributes: ``attributes_struct`` as plain data: the keys named in
            ``HANDOFF_MAPPED_ATTRIBUTES`` of the conventions, and any other key
            passed through unread. Values are as ``attributes_struct`` holds
            them: every number is a float (a protobuf ``Struct`` has no
            integers), ``disagg_request_id`` is a decimal string and
            ``opaque_state`` base64 text.
    """

    session_id: str
    transfer_backend: str = ""
    dp_rank: int = 0
    endpoints: tuple[KvEndpoint, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EngineRequest:
    """One Generate call.

    An unset sampling field is left out of ``sampling`` rather than sent with a
    default value, so the engine applies its own default.

    Attributes:
        request_id: ``"trtllm-"`` followed by a uuid4 in hex; unique per call.
        sampling: OpenEngine ``SamplingParams`` fields by name (``temperature``,
            ``top_p``, ``top_k``, ``min_p``, ``seed`` and the penalties); not
            ``num_sequences``, which has its own field.
        stop_texts: Stop strings for the engine to match; empty unless the
            engine declares ``trtllm.stopping.stop_text``.
        output_logprobs_top_n: None for no logprobs; 0 for the sampled token's only.
        extra: ``GenerateRequest.extra`` entries; only keys the conventions define.
        metadata: gRPC metadata such as ``traceparent``, ``openengine-priority``
            and ``openengine-target-dp-rank``.
        request_type: The ``request_type`` the client sends in ``extra``:
            ``"context_only"`` for the prefill call of a disaggregated request,
            ``"context_and_generation"`` when the request asked for it, None
            otherwise. ``extra`` itself does not hold ``request_type``.
        kv_session: The handoff a decode call carries in ``kv.session``; set
            only on a decode call, whose ``request_type`` is None.
    """

    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int | None
    sampling: dict[str, float | int] = field(default_factory=dict)
    min_tokens: int | None = None
    stop_token_ids: list[int] = field(default_factory=list)
    stop_texts: list[str] = field(default_factory=list)
    ignore_eos: bool | None = None
    include_stop_in_output: bool | None = None
    num_sequences: int = 1
    output_logprobs_top_n: int | None = None
    prompt_logprobs_top_n: int | None = None
    guided: GuidedSpec | None = None
    cache_salt: str | None = None
    media: list[MediaSpec] = field(default_factory=list)
    lora_name: str | None = None
    model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    request_type: DisaggRequestType | None = None
    kv_session: KvSessionSpec | None = None


@dataclass(frozen=True)
class TokenLogprob:
    """The logprob of one token, with its rank and the top candidates when requested.

    Attributes:
        top: ``(token_id, logprob)`` of the candidates, highest logprob first.
    """

    token_id: int
    logprob: float
    rank: int | None = None
    top: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class Usage:
    """Token counts of a whole Generate call."""

    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int | None = None


@dataclass(frozen=True)
class PromptInfo:
    """Prompt logprobs; sent once, before any token, and only when requested."""

    token_logprobs: tuple[TokenLogprob, ...] = ()


@dataclass(frozen=True)
class TokenDelta:
    """New tokens of one output; may hold several tokens."""

    output_index: int
    token_ids: tuple[int, ...]
    logprobs: tuple[TokenLogprob, ...] | None = None


@dataclass(frozen=True)
class OutputFinished:
    """One output ended.

    At most one of ``eos_token_id``, ``stop_token_id`` and ``stop_text`` is set,
    and only for ``reason == "stop"``: it says what ended the output. ``usage``
    is set only on the last event of the call, a cancelled call included.
    """

    output_index: int
    reason: FinishReason
    eos_token_id: int | None = None
    stop_token_id: int | None = None
    stop_text: str | None = None
    usage: Usage | None = None


@dataclass(frozen=True)
class PrefillReady:
    """The prefill call of a disaggregated request handed over its KV cache; ends the call.

    Only a prefill call (``request_type="context_only"``) may end this way;
    it ends either with this event or with :class:`OutputFinished`, when the
    whole request finished on the context engine. The call's tokens before it
    are the context engine's first token, which the decode call delivers again.

    Attributes:
        output_index: Always 0: a disaggregated request has one output.
        kv_session: The handoff, for the decode call's ``kv_session``.
        usage: The context engine's token counts for the call.
    """

    output_index: int
    kv_session: KvSessionSpec
    usage: Usage | None = None


EngineEvent = PromptInfo | TokenDelta | OutputFinished | PrefillReady


@dataclass(frozen=True)
class EngineHealth:
    """The engine's answer to a health check.

    Attributes:
        checks: ``(name, state, message)`` of each sub-check the engine reported.
    """

    state: HealthState
    checks: tuple[tuple[str, HealthState, str], ...] = ()

    @property
    def serving(self) -> bool:
        """Whether the engine takes new requests."""
        return self.state in ("ready", "degraded")


@dataclass(frozen=True)
class EngineLoad:
    """A load snapshot of the engine.

    Attributes:
        attributes: ``LoadInfo.attributes``, including the counters of the conventions.
    """

    instance_id: str
    running_requests: int | None = None
    queued_requests: int | None = None
    used_kv_blocks: int | None = None
    total_kv_blocks: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KvConnectorInfo:
    """An engine's ``ServerInfo.kv_connector``; a field the engine leaves unset reads as its default.

    Attributes:
        enabled: KV transfer is enabled; required to serve disaggregated requests.
        transfer_backend: The KV transfer backend, such as ``"NIXL"``; informational.
        local_endpoints: Where a generation-first decode call finds this
            context engine; empty when the engine gives none.
        supported_protocols: As the engine reports it; the serving side does not read it.
        supports_remote_prefill: The engine takes prefill calls ending with
            ``PrefillReady`` and decode calls with ``kv.session``; required too.
        supports_decode_pull: As the engine reports it; the serving side does not read it.
        supports_abort_cleanup: ``Abort`` with a ``kv_session`` frees the
            handoff's KV cache; when False, it is not called.
        schema_version: As the engine reports it, or None; the serving side does not read it.
    """

    enabled: bool = False
    transfer_backend: str = ""
    local_endpoints: tuple[KvEndpoint, ...] = ()
    supported_protocols: tuple[str, ...] = ()
    supports_remote_prefill: bool = False
    supports_decode_pull: bool = False
    supports_abort_cleanup: bool = False
    schema_version: int | None = None


@dataclass(frozen=True)
class EngineInfo:
    """Who the engine is.

    Attributes:
        instance_id: Changes when the engine process restarts.
        extra: ``ServerInfo.extra``.
        engine_role: ``ServerInfo.engine_role``, for logs only.
        kv_connector: ``ServerInfo.kv_connector``; None when the engine reports none.
    """

    engine_name: str
    engine_version: str
    instance_id: str
    schema_revision: int
    minimum_client_revision: int = 0
    schema_release: str = ""
    max_running_requests: int | None = None
    max_loras: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    engine_role: EngineRole = "unspecified"
    kv_connector: KvConnectorInfo | None = None


@dataclass(frozen=True)
class KvEventSource:
    """Where the engine publishes KV-cache events."""

    transport: str
    host: str
    port: int
    protocol: str = ""
    topic: str = ""
    replay_endpoint: str = ""
    data_parallel_rank: int | None = None
    encoding: str = ""


@dataclass(frozen=True)
class LoraAdapter:
    """A LoRA adapter registered with the engine."""

    name: str
    int_id: int
    path: str


@runtime_checkable
class EngineClient(Protocol):
    """How trtllm-serve runs requests on an engine."""

    def generate(self, request: EngineRequest) -> AsyncIterator[EngineEvent]:
        """Run one Generate call and stream its events; closing the iterator cancels it."""
        ...

    async def abort(self, request_id: str) -> None:
        """Abort a request; aborting one that is unknown or already finished is not an error."""
        ...

    async def health(self, *, deep: bool = False) -> EngineHealth:
        """Check the engine's health; ``deep`` adds an inference probe."""
        ...

    async def load(self) -> EngineLoad | None:
        """The engine's load and counters; None if the engine does not report load."""
        ...

    def server_info(self) -> EngineInfo:
        """The engine's identity, read at start and again when its instance id changes."""
        ...

    def model_info(self) -> ModelCapabilities:
        """The engine's capabilities, read at start and again when its instance id changes."""
        ...

    async def load_lora(self, name: str, int_id: int, path: str) -> bool:
        """Register a LoRA adapter; returns whether it was already loaded."""
        ...

    async def list_loras(self) -> Sequence[LoraAdapter]:
        """The LoRA adapters the engine has registered."""
        ...

    async def kv_event_sources(self) -> Sequence[KvEventSource]:
        """Where the engine publishes KV-cache events; empty if it publishes none."""
        ...

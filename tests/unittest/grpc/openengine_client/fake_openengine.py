# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A deterministic OpenEngine v1 server that needs no GPU.

``FakeEngine`` answers every OpenEngine RPC on a local port (loopback by
default), with ``UNIMPLEMENTED`` for the ones it does not support. It
declares the capabilities of a :class:`FakeProfile`, admits or rejects each ``Generate``
call by those declarations with the status a real engine uses, records each
request with its metadata unless recording is off, and counts calls in the
``trtllm.counters.*`` counters, which ``GetLoad`` reports when the profile
follows the ``trtllm.*`` conventions.

An admitted request is answered in one of two ways:

* By the built-in generator, when the engine has no script for it. Output
  ``i`` replays the prompt's token ids cyclically, starting at prompt position
  ``i``, and applies ``max_tokens``, ``min_tokens``, ``ignore_eos``, stop
  token ids and stop strings. See :class:`FakeEngine` for the exact rules.
* By a script: a sequence of steps, each a ``GenerateResponse`` to send or a
  :class:`Delay`, :class:`Fail`, :class:`Disconnect` or :class:`Hang` step.

Built-in profiles (:data:`PROFILES`):

``trtllm-like``
    The TensorRT-LLM OpenEngine server following the ``trtllm.*`` conventions,
    without speculative decoding: it accepts every sampling value. It declares
    the guided decoding modes of an engine started with
    ``guided_decoding_backend: xgrammar``.
``glm-like``
    The glm-5.3-flash OpenEngine server following the conventions. It accepts
    the penalties only at the values that leave sampling unchanged.
``ds-like``
    The DeepSeek-V4.1-Flash OpenEngine server (``dsv4_serve``): token-id input,
    one sequence, no logprobs, the penalties only at their no-effect values.

Disaggregated serving: ``trtllm-like`` and ``ds-like`` take the two calls of a
disaggregated request once the profile declares a :class:`FakeKvConnector`
that is enabled and supports remote prefill (see :func:`disagg_profile`);
otherwise they refuse them with ``FAILED_PRECONDITION``. A prefill call
(``extra.request_type = "context_only"``) gets its first token, then (unless,
on ``ds-like``, that token stops it) a ``prefill_ready`` event whose
``kv_session`` has the shape of that engine's handoff; a decode call (with
``kv.session``) starts its output with the handoff's ``first_gen_tokens``.
:class:`DisaggFaults` makes the prefill call misbehave on purpose. ``glm-like``
refuses ``kv.session`` and ignores ``request_type``.

Run as a script to serve a profile from its own process::

    python fake_openengine.py --profile trtllm-like --port 0 --port-file PORT_FILE
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import math
import os
import secrets
import signal
import struct
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import grpc
from google.protobuf import json_format, struct_pb2
from grpc_health.v1 import health_pb2, health_pb2_grpc

from tensorrt_llm.grpc.openengine.affinity import (
    AFFINITY_AUTH_HEADER,
    AFFINITY_ID_HEADER,
    validate_parent_affinity,
)
from tensorrt_llm.grpc.openengine.bindings import (
    MINIMUM_CLIENT_REVISION,
    SCHEMA_RELEASE,
    SCHEMA_REVISION,
    generation_pb2,
    kv_pb2,
    lifecycle_pb2,
    lora_pb2,
    model_pb2,
    openengine_pb2_grpc,
    server_pb2,
)

# The OpenEngine conventions (extra and metadata keys, capability tables, limits),
# under the names this module uses.
from tensorrt_llm.grpc.openengine_conventions import (
    ACCEPT_ANY,
    ACCEPT_RANGE,
    ACCEPT_VALUES,
    ANY_VALUE,
    BOOL_CAPABILITY_KEYS,
    BUCKET_AGGREGATED,
    BUCKET_CONTEXT_ONLY,
    CAP_DISAGG_GENERATION_FIRST,
    CAP_DISAGG_ORIG_ELAPSED_S,
    CAP_FREQUENCY_PENALTY,
    CAP_MIN_P,
    CAP_PRESENCE_PENALTY,
    CAP_REPETITION_PENALTY,
    CAP_REQ_DISAGG_REQUEST_ID,
    CAP_SEED_SCOPE,
    CAP_STOP_TEXT,
    CAP_STOPPING_MIN_TOKENS,
    CONVENTIONS_VERSION,
    COUNTER_GENERATE_ACCEPTED,
    COUNTER_GENERATE_ACCEPTED_BUCKETS,
    COUNTER_GENERATE_RECEIVED,
    COUNTER_PREFILL_READY_SENT,
    DISAGG_REQUEST_ID_LIMIT,
    DISAGG_REQUEST_ID_MIN,
    DISAGG_REQUEST_KEY_CAPABILITY,
    FP_BOS_TOKEN_ID,
    FP_EOS_TOKEN_IDS,
    FP_FILES_SHA256,
    FP_VOCAB_SIZE,
    HANDOFF_MAPPED_ATTRIBUTES,
    HANDOFF_REGISTERED_PASSTHROUGH_ATTRIBUTES,
    LOGIT_BIAS_MAX,
    LOGIT_BIAS_MIN,
    NO_VALUE,
    PARAMS_KEY,
    PRIORITY_SCALE,
    REQ_BEST_OF,
    REQ_CONVERSATION_ID,
    REQ_DETOKENIZE,
    REQ_DISAGG_REQUEST_ID,
    REQ_KV_HINTS,
    REQ_LOGIT_BIAS,
    REQ_REQUEST_TYPE,
    REQ_SCHEDULE_STYLE,
    REQ_SCHEDULING,
    REQ_THINKING_TOKEN_BUDGET,
    REQUEST_KEY_CAPABILITY,
    REQUEST_TYPE_CONTEXT_AND_GENERATION,
    REQUEST_TYPE_CONTEXT_ONLY,
    REQUEST_TYPE_GENERATION_ONLY,
    SAMPLING_FIELD_CAPABILITY,
    SAMPLING_PARAM_NAMES,
    SCHEDULE_STYLE_CONTEXT_FIRST,
    SCHEDULE_STYLE_GENERATION_FIRST,
    VERSION_4_CAPABILITY_KEYS,
    VERSION_4_ENGINE_COUNTERS,
    AcceptedValues,
    parse_disagg_request_id,
    request_bucket,
)
from tensorrt_llm.grpc.openengine_conventions import CAP_REQ_BEST_OF as CAP_BEST_OF
from tensorrt_llm.grpc.openengine_conventions import CAP_REQ_CONVERSATION_ID as CAP_CONVERSATION_ID
from tensorrt_llm.grpc.openengine_conventions import CAP_REQ_KV_HINTS as CAP_KV_HINTS
from tensorrt_llm.grpc.openengine_conventions import CAP_REQ_LOGIT_BIAS as CAP_LOGIT_BIAS
from tensorrt_llm.grpc.openengine_conventions import CAP_REQ_SCHEDULING as CAP_SCHEDULING
from tensorrt_llm.grpc.openengine_conventions import (
    CAP_REQ_THINKING_TOKEN_BUDGET as CAP_THINKING_TOKEN_BUDGET,
)
from tensorrt_llm.grpc.openengine_conventions import (
    CONVENTIONS_VERSION_KEY as KEY_CONVENTIONS_VERSION,
)
from tensorrt_llm.grpc.openengine_conventions import (
    LOGIT_BIAS_MAX_ENTRIES as MAX_LOGIT_BIAS_ENTRIES,
)
from tensorrt_llm.grpc.openengine_conventions import META_PRIORITY as METADATA_PRIORITY
from tensorrt_llm.grpc.openengine_conventions import META_TARGET_DP_RANK as METADATA_TARGET_DP_RANK
from tensorrt_llm.grpc.openengine_conventions import (
    MODEL_SUPPORTED_MODALITIES_KEY as SUPPORTED_MODALITIES,
)
from tensorrt_llm.grpc.openengine_conventions import (
    SERVER_DP_RANK_TARGETING_KEY as DP_RANK_TARGETING,
)

__all__ = [
    "Admitted",
    "CAP_BEST_OF",
    "CAP_CONVERSATION_ID",
    "CAP_DISAGG_GENERATION_FIRST",
    "CAP_DISAGG_ORIG_ELAPSED_S",
    "CAP_FREQUENCY_PENALTY",
    "CAP_KV_HINTS",
    "CAP_LOGIT_BIAS",
    "CAP_MIN_P",
    "CAP_PRESENCE_PENALTY",
    "CAP_REPETITION_PENALTY",
    "CAP_REQ_DISAGG_REQUEST_ID",
    "CAP_SCHEDULING",
    "CAP_SEED_SCOPE",
    "CAP_STOP_TEXT",
    "CAP_STOPPING_MIN_TOKENS",
    "CAP_THINKING_TOKEN_BUDGET",
    "CONVENTIONS_VERSION",
    "COUNTER_GENERATE_ACCEPTED",
    "COUNTER_GENERATE_ACCEPTED_BUCKETS",
    "COUNTER_GENERATE_RECEIVED",
    "COUNTER_PREFILL_READY_SENT",
    "DP_RANK_TARGETING",
    "Delay",
    "DisaggFaults",
    "Disconnect",
    "FP_BOS_TOKEN_ID",
    "FP_EOS_TOKEN_IDS",
    "FP_FILES_SHA256",
    "FP_VOCAB_SIZE",
    "Fail",
    "FakeEngine",
    "FakeKvConnector",
    "FakeProfile",
    "HANDOFF_SHAPES",
    "Hang",
    "INSTANCE_ROLES",
    "KEY_CONVENTIONS_VERSION",
    "KV_HINTS_TEXT_MAX_BYTES",
    "MAX_LOGIT_BIAS_ENTRIES",
    "MAX_MESSAGE_BYTES",
    "MAX_STOP_CONDITIONS",
    "METADATA_PRIORITY",
    "METADATA_TARGET_DP_RANK",
    "PRIORITY_SCALE",
    "PROFILES",
    "REQ_BEST_OF",
    "REQ_CONVERSATION_ID",
    "REQ_DETOKENIZE",
    "REQ_KV_HINTS",
    "REQ_LOGIT_BIAS",
    "REQ_SCHEDULING",
    "REQ_THINKING_TOKEN_BUDGET",
    "ReceivedRequest",
    "Rejection",
    "SUBAGENT_AFFINITY",
    "SUPPORTED_MODALITIES",
    "ScriptSource",
    "Step",
    "admit",
    "default_detokenizer",
    "disagg_profile",
    "main",
    "resolve_profile",
    "running_in_thread",
]


# The capability that admits each request extra key and sampling field, and every
# boolean capability the fake reports.
_REQUEST_KEY_CAPABILITY = REQUEST_KEY_CAPABILITY
_SAMPLING_FIELD_CAPABILITY = SAMPLING_FIELD_CAPABILITY
_BOOLEAN_CAPABILITIES = BOOL_CAPABILITY_KEYS
_SAMPLING_PARAM_BY_KEY = {key: name for name, key in SAMPLING_FIELD_CAPABILITY.items()}
# The bounds the protocol puts on a sampling field, (low, high); None for no bound.
_VALID_RANGE: dict[str, tuple[float | None, float | None]] = {
    "temperature": (0, None),
    "top_p": (0, 1),
    "top_k": (0, None),
    "min_p": (0, 1),
}
_FLOAT_SAMPLING_FIELDS = frozenset(SAMPLING_PARAM_NAMES) - {"top_k", "seed"}

# -- Protocol limits ---------------------------------------------------------------

MAX_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_STOP_CONDITIONS = 64
# Longest trtllm.kv_hints JSON text: four times the 64 KiB the envelope may take.
KV_HINTS_TEXT_MAX_BYTES = 256 * 1024

_GUIDE_MODE = {
    "json_schema": model_pb2.GUIDED_DECODING_MODE_JSON_SCHEMA,
    "regex": model_pb2.GUIDED_DECODING_MODE_REGEX,
    "ebnf_grammar": model_pb2.GUIDED_DECODING_MODE_EBNF_GRAMMAR,
    "structural_tag": model_pb2.GUIDED_DECODING_MODE_STRUCTURAL_TAG,
    "choice": model_pb2.GUIDED_DECODING_MODE_CHOICE,
    "json_object": model_pb2.GUIDED_DECODING_MODE_JSON_OBJECT,
}
_XGRAMMAR_GUIDES = frozenset(
    {"json_schema", "regex", "ebnf_grammar", "structural_tag", "json_object"}
)
_MODALITY_NAME = {
    generation_pb2.MODALITY_IMAGE: "image",
    generation_pb2.MODALITY_VIDEO: "video",
    generation_pb2.MODALITY_AUDIO: "audio",
}


# -- Profiles --------------------------------------------------------------------

HANDOFF_SHAPES = ("trtllm", "ds")
INSTANCE_ROLES = ("context", "generation")
# ServerInfo.engine_role of a ds instance by its role.
_DS_ENGINE_ROLE = {
    "context": server_pb2.ENGINE_ROLE_PREFILL,
    "generation": server_pb2.ENGINE_ROLE_DECODE,
}
# ServerInfo.extra key of the TensorRT-LLM server: whether it checks the
# subagent affinity metadata.
SUBAGENT_AFFINITY = "trtllm_supports_subagent_affinity"


@dataclass(frozen=True)
class FakeKvConnector:
    """What an engine reports in ``ServerInfo.kv_connector``.

    Attributes:
        enabled: KV transfer is on; without it the engine refuses both calls
            of a disaggregated request with ``FAILED_PRECONDITION``.
        transfer_backend: Reported, and written into each handoff's ``transfer_backend``.
        supports_remote_prefill: Reported; the engine also needs it to take the two calls.
        supports_abort_cleanup: Reported; ``Abort`` by ``kv_session`` stays
            ``UNIMPLEMENTED`` either way, so keep it False to report truthfully.
        local_endpoints: Reported as ``(host, port, protocol)`` triples.
        ctx_info_endpoint: ``(host, port)`` the handoff names as the context
            engine's KV endpoint; None for the engine's own gRPC host and port.
            Nothing listens for KV there: a fake engine moves no KV cache.
    """

    enabled: bool = True
    transfer_backend: str = "DEFAULT"
    supports_remote_prefill: bool = True
    supports_abort_cleanup: bool = False
    local_endpoints: tuple[tuple[str, int, str], ...] = ()
    ctx_info_endpoint: tuple[str, int] | None = None

    def to_proto(self) -> kv_pb2.KvConnectorInfo:
        info = kv_pb2.KvConnectorInfo(
            enabled=self.enabled,
            transfer_backend=self.transfer_backend,
            supports_remote_prefill=self.supports_remote_prefill,
            supports_decode_pull=False,
            supports_abort_cleanup=self.supports_abort_cleanup,
            schema_version=SCHEMA_REVISION,
        )
        for host, port, protocol in self.local_endpoints:
            info.local_endpoints.add(host=host, port=port, protocol=protocol)
        return info

    @property
    def takes_disaggregated_calls(self) -> bool:
        return self.enabled and self.supports_remote_prefill


@dataclass(frozen=True)
class FakeProfile:
    """What a fake engine declares, and so what it admits.

    ``None`` leaves ``max_num_sequences``, ``max_loras`` and
    ``supports_token_ids_input`` unset in what the engine reports; for
    ``output_logprobs_max_top_n`` and ``prompt_logprobs_max_top_n`` it means
    that kind of logprobs is unsupported. A capability key left out of
    ``capabilities`` counts as unsupported, except a version 2 sampling key.

    Attributes:
        follows_conventions: Whether the engine follows the ``trtllm.*``
            conventions. When it does, it reports ``trtllm.conventions_version``,
            every capability key, ``trtllm.params`` and the counters, rejects
            undeclared request keys (``trtllm.*`` and ``conversation_id``),
            and validates the request keys it declares. When it does not, it
            reports none of these, accepts what ``capabilities`` marks true
            without declaring it, and ignores the other ``extra`` keys except
            ``detokenize``. Either way it admits sampling values by
            :meth:`accepted`.
        capabilities: Values of the capability keys, by key. The version 2
            sampling keys are reported as true when the parameter accepts
            every value the protocol allows; one set here overrides that
            parameter (see :meth:`accepted`).
        params: The values accepted for each sampling parameter, reported
            under ``trtllm.params``; a parameter missing from the mapping
            accepts none. By default every parameter accepts every value, and
            :meth:`replace` merges new entries into the current ones.
        float32_sampling: Whether the engine samples in 32-bit floats and,
            as glm does, checks that a sampling value is valid before checking
            ``params``: a float value must be finite as a 32-bit float, inside
            the protocol's bounds, and for ``top_p`` not 0 as a 32-bit float.
        max_loras: Reported as ``capacity.max_loras``; ``LoadLora`` does not
            refuse because of it.
        max_registered_loras: The most distinct ``lora_id`` values
            ``LoadLora`` accepts over the engine's life, unloaded ones
            included, as the TensorRT-LLM server's registry counts them
            against its fixed limit; ``None`` sets no limit. Unlike the
            server, the fake does not check that ``source_path`` exists.
        reject_kv_session: Reject ``kv.session`` with ``UNIMPLEMENTED``, as
            glm does. Only for a profile without ``handoff_shape``, which
            otherwise ignores ``kv.session`` and ``extra.request_type``.
        handoff_shape: Whose disaggregated serving the engine follows, or
            None for none. ``"trtllm"``: the TensorRT-LLM server, which takes
            both calls on any instance and carries ``first_gen_log_probs`` in
            the handoff when output logprobs are requested. ``"ds"``: the ds
            server, whose handoff never carries ``first_gen_log_probs``, whose
            prefill call ends without a handoff when its first token is an
            EOS or stop token, and whose decode call needs
            ``ctx_info_endpoint`` and ``first_gen_tokens`` and refuses handoff
            attributes the conventions do not name. A profile with a
            shape that follows the conventions reports the version 4
            capability keys; one with ``kv_connector`` also reports the
            version 4 counters.
        kv_connector: ``ServerInfo.kv_connector``; None reports none. The
            engine takes the two calls of a disaggregated request only when
            it is enabled with ``supports_remote_prefill``.
        instance_role: For a ds profile with a ``kv_connector``, ``"context"``
            or ``"generation"``: the one call of a disaggregated request it
            takes, as a ds instance started with that role does. The fake does
            not model a ds instance in both roles. None for the trtllm shape,
            which takes both calls. Every engine takes aggregated calls.
        subagent_affinity_key: The key that signs the subagent affinity
            metadata. With a shape, the engine checks that metadata with the
            TensorRT-LLM server's check, and reports
            ``trtllm_supports_subagent_affinity`` true when the key is set;
            the server also needs conversation-aware attention DP for that,
            which the fake does not model.
        eos_token_ids: Tokens that end an output unless ``ignore_eos`` is set;
            the one sampled is reported in ``stop_match.eos_token_id``.
        fingerprint: Tokenizer fingerprint reported in ``ModelInfo.extra``.
        kv_event_sources: What ``GetKvEventSources`` returns.
        model_info_extra: More ``ModelInfo.extra`` entries; they win over the
            ones derived from the other fields.
        server_info_extra: More ``ServerInfo.extra`` entries; same rule.
        server_info: ``ServerInfo`` field overrides, for example
            ``schema_revision``.
    """

    name: str
    engine_name: str
    engine_version: str = "fake"
    engine_role: int = server_pb2.ENGINE_ROLE_UNSPECIFIED
    model: str = "fake-model"
    max_context_length: int = 4096
    vocab_size: int = 32000
    supports_text_input: bool = True
    supports_token_ids_input: bool | None = True
    output_logprobs_max_top_n: int | None = 100
    prompt_logprobs_max_top_n: int | None = 100
    guided_modes: frozenset[str] = _XGRAMMAR_GUIDES
    max_num_sequences: int | None = 16
    supports_priority: bool = False
    supports_stop_in_output: bool = True
    supports_cache_salt: bool = True
    supports_lora: bool = False
    max_loras: int | None = None
    max_registered_loras: int | None = None
    modalities: frozenset[str] = frozenset()
    reasoning_parser: str = ""
    tool_call_parser: str = ""
    dp_rank_targeting: bool = True
    data_parallel_size: int = 1
    max_running_requests: int = 64
    follows_conventions: bool = True
    # Mappings and protobuf messages are unhashable, so they stay out of hash().
    capabilities: Mapping[str, bool | str] = field(default_factory=dict, hash=False)
    params: Mapping[str, AcceptedValues] = field(
        default_factory=lambda: dict.fromkeys(SAMPLING_PARAM_NAMES, ANY_VALUE), hash=False
    )
    float32_sampling: bool = False
    reject_kv_session: bool = False
    handoff_shape: str | None = None
    kv_connector: FakeKvConnector | None = None
    instance_role: str | None = None
    subagent_affinity_key: str | None = field(default=None, repr=False)
    eos_token_ids: tuple[int, ...] = (2,)
    bos_token_id: int | None = 1
    fingerprint: Mapping[str, Any] | None = field(default=None, hash=False)
    kv_event_sources: tuple[kv_pb2.KvEventSource, ...] = field(default=(), hash=False)
    model_info_extra: Mapping[str, Any] = field(default_factory=dict, hash=False)
    server_info_extra: Mapping[str, Any] = field(default_factory=dict, hash=False)
    server_info: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        # Profiles are shared module constants: keep their mappings read-only.
        for name in (
            "capabilities",
            "params",
            "fingerprint",
            "model_info_extra",
            "server_info_extra",
            "server_info",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, MappingProxyType(dict(value)))
        object.__setattr__(self, "kv_event_sources", tuple(self.kv_event_sources))
        if self.handoff_shape not in (None, *HANDOFF_SHAPES):
            raise ValueError(f"handoff_shape must be one of {HANDOFF_SHAPES} or None")
        if self.instance_role not in (None, *INSTANCE_ROLES):
            raise ValueError(f"instance_role must be one of {INSTANCE_ROLES} or None")
        if self.handoff_shape is None and (
            self.kv_connector is not None
            or self.instance_role is not None
            or self.subagent_affinity_key is not None
        ):
            raise ValueError(
                "kv_connector, instance_role and subagent_affinity_key need a handoff_shape"
            )
        if self.handoff_shape is not None and self.reject_kv_session:
            raise ValueError("reject_kv_session is only for a profile without a handoff_shape")
        if self.handoff_shape == "trtllm" and self.instance_role is not None:
            raise ValueError("a trtllm-shaped engine takes both calls; it has no instance_role")
        if self.handoff_shape == "ds" and self.kv_connector is not None:
            if self.instance_role is None:
                raise ValueError(
                    "a ds-shaped engine with a kv_connector needs an instance_role: "
                    "the fake does not model a ds instance in both roles"
                )
        if (
            self.instance_role is not None
            and self.engine_role != _DS_ENGINE_ROLE[self.instance_role]
        ):
            raise ValueError(
                f"a {self.instance_role} instance reports engine_role "
                f"{server_pb2.EngineRole.Name(_DS_ENGINE_ROLE[self.instance_role])}"
            )

    def replace(self, **changes: Any) -> FakeProfile:
        """A copy with ``changes`` applied; ``capabilities`` and ``params`` entries are merged."""
        for name in ("capabilities", "params"):
            if name in changes:
                changes[name] = {**getattr(self, name), **changes[name]}
        return dataclasses.replace(self, **changes)

    def capability(self, key: str) -> bool:
        name = _SAMPLING_PARAM_BY_KEY.get(key)
        if name is not None:
            return _accepts_every_valid_value(name, self.accepted(name))
        return self.capabilities.get(key) is True

    def accepted(self, name: str) -> AcceptedValues:
        """The values accepted for the sampling parameter ``name``.

        From ``params``, unless ``capabilities`` sets the parameter's version 2
        key: then every value when it is true and none otherwise, keeping the
        declared default.
        """
        declared = self.params.get(name, NO_VALUE)
        key = _SAMPLING_FIELD_CAPABILITY.get(name)
        if key is not None and key in self.capabilities:
            accepted = ANY_VALUE if self.capabilities[key] is True else NO_VALUE
            return dataclasses.replace(accepted, default=declared.default)
        return declared

    def tokenizer_fingerprint(self) -> dict[str, Any]:
        if self.fingerprint is not None:
            return dict(self.fingerprint)
        return {
            FP_VOCAB_SIZE: self.vocab_size,
            FP_EOS_TOKEN_IDS: list(self.eos_token_ids),
            FP_BOS_TOKEN_ID: self.bos_token_id,
            FP_FILES_SHA256: {},
        }

    def build_server_info(self, instance_id: str) -> server_pb2.ServerInfo:
        info = server_pb2.ServerInfo(
            engine_name=self.engine_name,
            engine_version=self.engine_version,
            engine_role=self.engine_role,
            instance_id=instance_id,
            supported_models=[self.model],
            schema_revision=SCHEMA_REVISION,
            minimum_client_revision=MINIMUM_CLIENT_REVISION,
            schema_release=SCHEMA_RELEASE,
        )
        info.parallelism.CopyFrom(
            server_pb2.ParallelismInfo(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                decode_context_parallel_size=1,
                data_parallel_size=self.data_parallel_size,
            )
        )
        info.capacity.max_running_requests = self.max_running_requests
        if self.max_loras is not None:
            info.capacity.max_loras = self.max_loras
        if self.kv_connector is not None:
            info.kv_connector.CopyFrom(self.kv_connector.to_proto())
        extra: dict[str, Any] = {DP_RANK_TARGETING: self.dp_rank_targeting}
        if self.follows_conventions:
            extra[KEY_CONVENTIONS_VERSION] = CONVENTIONS_VERSION
        if self.handoff_shape is not None:
            extra[SUBAGENT_AFFINITY] = self.subagent_affinity_key is not None
        extra.update(self.server_info_extra)
        info.extra.update(extra)
        for name, value in self.server_info.items():
            setattr(info, name, value)
        return info

    def build_model_info(self) -> model_pb2.ModelInfo:
        info = model_pb2.ModelInfo(
            model_id=self.model,
            served_model_name=self.model,
            max_context_length=self.max_context_length,
            supports_text_input=self.supports_text_input,
            supports_lora=self.supports_lora,
            supports_multimodal=bool(self.modalities),
            reasoning_parser=self.reasoning_parser,
            tool_call_parser=self.tool_call_parser,
        )
        if self.supports_token_ids_input is not None:
            info.supports_token_ids_input = self.supports_token_ids_input
        generation = info.generation
        for capability, max_top_n in (
            (generation.output_logprobs, self.output_logprobs_max_top_n),
            (generation.prompt_logprobs, self.prompt_logprobs_max_top_n),
        ):
            capability.supported = max_top_n is not None
            capability.max_top_n = max_top_n or 0
            if max_top_n is not None:
                capability.candidate_selection_modes.append(
                    model_pb2.CANDIDATE_TOKEN_SELECTION_MODE_TOP_N
                )
        generation.guided_decoding.supported = bool(self.guided_modes)
        generation.guided_decoding.modes.extend(
            sorted(_GUIDE_MODE[guide] for guide in self.guided_modes)
        )
        if self.max_num_sequences is not None:
            generation.max_num_sequences = self.max_num_sequences
        generation.supports_priority = self.supports_priority
        generation.supports_stop_in_output = self.supports_stop_in_output
        generation.supports_cache_salt = self.supports_cache_salt
        generation.supports_prefix_cache_bypass = False
        extra: dict[str, Any] = {SUPPORTED_MODALITIES: sorted(self.modalities)}
        if self.follows_conventions:
            extra.update({key: self.capability(key) for key in _BOOLEAN_CAPABILITIES})
            if self.handoff_shape is not None:
                extra.update({key: self.capability(key) for key in VERSION_4_CAPABILITY_KEYS})
            extra[PARAMS_KEY] = {
                name: self.accepted(name).to_extra() for name in SAMPLING_PARAM_NAMES
            }
            extra[CAP_SEED_SCOPE] = str(self.capabilities.get(CAP_SEED_SCOPE, "none"))
            extra.update(self.tokenizer_fingerprint())
        extra.update(self.model_info_extra)
        info.extra.update(extra)
        return info


def _accepts_every_valid_value(name: str, accepted: AcceptedValues) -> bool:
    """Whether ``accepted`` holds every value the protocol allows for ``name``."""
    if accepted.accept == ACCEPT_ANY:
        return True
    if accepted.accept != ACCEPT_RANGE:
        return False
    low, high = _VALID_RANGE.get(name, (None, None))
    covers_low = accepted.min is None or (
        low is not None
        and (accepted.min < low or (accepted.min == low and not accepted.min_exclusive))
    )
    covers_high = accepted.max is None or (
        high is not None
        and (accepted.max > high or (accepted.max == high and not accepted.max_exclusive))
    )
    return covers_low and covers_high


# What the TensorRT-LLM server reports without speculative decoding and with
# generation_config "trtllm": every value, and the value Generate uses when a
# field is unset.
_TRTLLM_PARAMS = {
    name: AcceptedValues(ACCEPT_ANY, default=default)
    for name, default in {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": None,
    }.items()
}

# What glm-5.3-flash reports: the penalties, which it does not apply, only at
# their no-effect values; temperature and top_p within what its float32
# sampling can carry out.
_GLM_PARAMS = {
    "temperature": AcceptedValues(ACCEPT_RANGE, min=0.0, max=3.4028234663852886e38, default=1.0),
    "top_p": AcceptedValues(ACCEPT_RANGE, min=2.0**-150, min_exclusive=True, max=1.0, default=1.0),
    "top_k": AcceptedValues(ACCEPT_RANGE, min=0, default=0),
    "min_p": AcceptedValues(ACCEPT_RANGE, min=0.0, max=1.0, default=0.0),
    "frequency_penalty": AcceptedValues(ACCEPT_VALUES, values=(0.0,), default=0.0),
    "presence_penalty": AcceptedValues(ACCEPT_VALUES, values=(0.0,), default=0.0),
    "repetition_penalty": AcceptedValues(ACCEPT_VALUES, values=(1.0,), default=1.0),
    "seed": ANY_VALUE,
}

_TRTLLM_CAPABILITIES: dict[str, bool | str] = {
    CAP_SEED_SCOPE: "batch_dependent",
    CAP_STOP_TEXT: True,
    CAP_CONVERSATION_ID: True,
    CAP_THINKING_TOKEN_BUDGET: True,
    CAP_LOGIT_BIAS: True,
    CAP_KV_HINTS: True,
    CAP_SCHEDULING: True,
    CAP_BEST_OF: False,
    # The version 4 keys, all true except generation-first, which the fake
    # does not serve.
    CAP_REQ_DISAGG_REQUEST_ID: True,
    CAP_DISAGG_GENERATION_FIRST: False,
    CAP_DISAGG_ORIG_ELAPSED_S: True,
    CAP_STOPPING_MIN_TOKENS: True,
}

PROFILES: dict[str, FakeProfile] = {
    "trtllm-like": FakeProfile(
        name="trtllm-like",
        engine_name="tensorrt_llm",
        capabilities=_TRTLLM_CAPABILITIES,
        params=_TRTLLM_PARAMS,
        handoff_shape="trtllm",
    ),
    "glm-like": FakeProfile(
        name="glm-like",
        engine_name="glm5_serve",
        engine_role=server_pb2.ENGINE_ROLE_AGGREGATED,
        supports_text_input=False,
        output_logprobs_max_top_n=20,
        prompt_logprobs_max_top_n=None,
        guided_modes=frozenset(),
        max_num_sequences=1,
        supports_cache_salt=False,
        dp_rank_targeting=False,
        reject_kv_session=True,
        params=_GLM_PARAMS,
        float32_sampling=True,
        capabilities={
            CAP_SEED_SCOPE: "batch_dependent",
            CAP_STOP_TEXT: True,
            CAP_CONVERSATION_ID: False,
            CAP_THINKING_TOKEN_BUDGET: False,
            CAP_LOGIT_BIAS: False,
            CAP_KV_HINTS: False,
            CAP_SCHEDULING: False,
            CAP_BEST_OF: False,
        },
    ),
    "ds-like": FakeProfile(
        name="ds-like",
        engine_name="dsv4_serve",
        engine_role=server_pb2.ENGINE_ROLE_AGGREGATED,
        supports_text_input=False,
        output_logprobs_max_top_n=None,
        prompt_logprobs_max_top_n=None,
        guided_modes=frozenset(),
        max_num_sequences=1,
        supports_cache_salt=False,
        dp_rank_targeting=False,
        params=_GLM_PARAMS,
        float32_sampling=True,
        handoff_shape="ds",
        capabilities={
            CAP_SEED_SCOPE: "batch_dependent",
            CAP_STOP_TEXT: True,
            CAP_CONVERSATION_ID: True,
            CAP_THINKING_TOKEN_BUDGET: False,
            CAP_LOGIT_BIAS: False,
            CAP_KV_HINTS: False,
            CAP_SCHEDULING: False,
            CAP_BEST_OF: False,
            CAP_REQ_DISAGG_REQUEST_ID: True,
            CAP_DISAGG_GENERATION_FIRST: False,
            CAP_DISAGG_ORIG_ELAPSED_S: False,
            CAP_STOPPING_MIN_TOKENS: False,
        },
    ),
}


def disagg_profile(
    profile: str | FakeProfile = "trtllm-like",
    role: str | None = None,
    *,
    kv_connector: FakeKvConnector | None = None,
    **changes: Any,
) -> FakeProfile:
    """``profile`` serving disaggregated requests, with ``changes`` applied.

    It declares ``kv_connector``: by default the one ``profile`` already
    declares, else an enabled one with the ``DEFAULT`` backend. A ``"ds"``
    profile needs ``role`` (by default the one ``profile`` already has),
    which sets ``instance_role`` and ``engine_role`` as a ds instance with
    that role reports them. The TensorRT-LLM server has no roles, so a
    ``"trtllm"`` profile refuses one. ``changes`` may not set what the
    arguments set.
    """
    base = resolve_profile(profile)
    if base.handoff_shape is None:
        raise ValueError(f"profile {base.name!r} does not serve disaggregated requests")
    set_here = {"kv_connector"}
    if base.handoff_shape == "ds":
        set_here |= {"instance_role", "engine_role"}
    clashing = sorted(set(changes) & set_here)
    if clashing:
        raise ValueError(f"disagg_profile sets {clashing} from its arguments")
    fields: dict[str, Any] = {
        "kv_connector": kv_connector or base.kv_connector or FakeKvConnector()
    }
    if base.handoff_shape == "ds":
        role = role or base.instance_role
        if role not in INSTANCE_ROLES:
            raise ValueError(f"a ds-shaped engine needs a role, one of {INSTANCE_ROLES}")
        fields.update(instance_role=role, engine_role=_DS_ENGINE_ROLE[role])
    elif role is not None:
        raise ValueError("a trtllm-shaped engine takes both calls; it has no role")
    return base.replace(**fields, **changes)


def resolve_profile(profile: str | FakeProfile) -> FakeProfile:
    if isinstance(profile, FakeProfile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}") from None


# -- Admission ---------------------------------------------------------------------


class Rejection(Exception):
    """A request the engine refuses before accepting it."""

    def __init__(self, code: grpc.StatusCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _unimplemented(message: str) -> Rejection:
    return Rejection(grpc.StatusCode.UNIMPLEMENTED, message)


def _invalid(message: str) -> Rejection:
    return Rejection(grpc.StatusCode.INVALID_ARGUMENT, message)


@dataclass
class Admitted:
    """The settings of an accepted request that drive the built-in generator."""

    prompt_token_ids: list[int]
    num_sequences: int
    max_tokens: int
    min_tokens: int
    ignore_eos: bool
    include_stop: bool
    stop_token_ids: frozenset[int]
    stop_texts: tuple[str, ...]
    output_top_n: int | None
    prompt_top_n: int | None
    priority: int | None = None
    target_dp_rank: int | None = None
    bucket: str = BUCKET_AGGREGATED
    disagg_request_id: int | None = None
    kv_session: kv_pb2.KvSessionRef | None = None
    first_gen_tokens: tuple[int, ...] = ()
    subagent_affinity_id: str | None = None


def _top_n(selection: generation_pb2.CandidateTokenSelection, name: str, limit: int) -> int:
    kind = selection.WhichOneof("selection")
    if kind is None:
        return 0
    if kind != "top_n":
        raise _unimplemented(f"{name} candidate selection '{kind}' is not supported; use top_n")
    if selection.top_n > limit:
        raise _invalid(f"{name} top_n {selection.top_n} exceeds the supported maximum {limit}")
    return selection.top_n


def _admit_metadata(
    metadata: Sequence[tuple[str, str]], profile: FakeProfile
) -> tuple[int | None, int | None]:
    seen: set[str] = set()
    priority = target_dp_rank = None
    for key, value in metadata:
        if not key.startswith("openengine-"):
            continue
        if key in seen:
            raise _invalid(f"gRPC metadata key '{key}' must not be repeated")
        seen.add(key)
        if key == METADATA_TARGET_DP_RANK and profile.dp_rank_targeting:
            try:
                target_dp_rank = int(value)
            except ValueError:
                raise _invalid(f"{key} must be an integer, got {value!r}") from None
            if not 0 <= target_dp_rank < profile.data_parallel_size:
                raise _invalid(
                    f"{key} must be in [0, {profile.data_parallel_size}), got {target_dp_rank}"
                )
        elif key == METADATA_PRIORITY and profile.supports_priority:
            try:
                priority = int(value)
            except ValueError:
                raise _invalid(f"{key} must be an integer, got {value!r}") from None
            if not 0 <= priority <= PRIORITY_SCALE:
                raise _invalid(f"{key} must be in [0, {PRIORITY_SCALE}], got {priority}")
        else:
            raise _unimplemented(f"gRPC metadata key '{key}' is not supported")
    return priority, target_dp_rank


def _admit_input(request: generation_pb2.GenerateRequest, profile: FakeProfile) -> list[int]:
    kind = request.WhichOneof("input")
    if kind == "token_ids":
        if not profile.supports_token_ids_input:
            raise _unimplemented("token-ID input is not supported; send a text prompt")
        ids = list(request.token_ids.ids)
    elif kind == "prompt":
        if not profile.supports_text_input:
            raise _unimplemented("text input is not supported; send token_ids")
        # No tokenizer: each UTF-8 byte of the prompt stands for one token.
        ids = list(request.prompt.encode("utf-8"))
    else:
        raise _invalid("exactly one of prompt or token_ids must be set")
    if not ids:
        raise _invalid("the prompt must not be empty")
    out_of_vocab = [token_id for token_id in ids if token_id >= profile.vocab_size]
    if out_of_vocab:
        raise _invalid(
            f"prompt token ids {out_of_vocab[:4]} are outside the vocabulary "
            f"of size {profile.vocab_size}"
        )
    return ids


def _float32(value: float) -> float:
    """``value`` as a 32-bit float; inf when outside its range."""
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return math.inf


def _check_bounds(name: str, value: float) -> None:
    low, high = _VALID_RANGE[name]
    if high is not None and not low <= value <= high:
        raise _invalid(f"require {low} <= {name} <= {high}, got {name}={value}")
    if low is not None and value < low:
        raise _invalid(f"require {name} >= {low}, got {name}={value}")


def _check_float32_sampling_value(name: str, value: float) -> None:
    if name in _FLOAT_SAMPLING_FIELDS and not math.isfinite(_float32(value)):
        raise _invalid(f"sampling.{name} must be a finite 32-bit float, got {value}")
    if name in _VALID_RANGE:
        _check_bounds(name, value)
    if name == "top_p" and _float32(value) == 0:
        raise _invalid(f"sampling.top_p must not be 0 as a 32-bit float, got {value}")


def _admit_sampling(request: generation_pb2.GenerateRequest, profile: FakeProfile) -> int:
    """Validate ``sampling`` and return the number of sequences.

    Each value in turn: one that is not finite is invalid; then one the
    profile does not accept is not supported; afterwards a value outside the
    protocol's bounds is invalid. A profile with ``float32_sampling`` checks
    each value's validity as a 32-bit float and its bounds before ``params``.
    For a request whose only fault is one sampling value, this gives the
    status of the TensorRT-LLM server, or of glm for ``glm-like``.
    """
    sampling = request.sampling
    for name in SAMPLING_PARAM_NAMES:
        if not sampling.HasField(name):
            continue
        value = getattr(sampling, name)
        if profile.float32_sampling:
            _check_float32_sampling_value(name, value)
        elif name in _FLOAT_SAMPLING_FIELDS and not math.isfinite(value):
            raise _invalid(f"sampling.{name} must be finite")
        accepted = profile.accepted(name)
        if not accepted.accepts(value):
            raise _unimplemented(
                f"sampling.{name}={value!r} is not supported by this engine: "
                f"{PARAMS_KEY}.{name} accepts {accepted.describe()}"
            )
    for name in _VALID_RANGE:
        if sampling.HasField(name):
            _check_bounds(name, getattr(sampling, name))
    num_sequences = sampling.num_sequences if sampling.HasField("num_sequences") else 1
    if num_sequences == 0:
        raise _invalid("sampling.num_sequences must be greater than zero")
    if profile.max_num_sequences is not None and num_sequences > profile.max_num_sequences:
        raise _invalid(
            f"sampling.num_sequences {num_sequences} exceeds the supported maximum "
            f"{profile.max_num_sequences}"
        )
    greedy = (
        (sampling.HasField("temperature") and sampling.temperature == 0)
        or (sampling.HasField("top_k") and sampling.top_k == 1)
        or (sampling.HasField("top_p") and sampling.top_p == 0)
        or (sampling.HasField("min_p") and sampling.min_p == 1)
    )
    if num_sequences > 1 and greedy:
        raise _invalid("greedy decoding does not allow multiple output sequences")
    return num_sequences


def _admit_extra(
    request: generation_pb2.GenerateRequest, profile: FakeProfile, num_sequences: int
) -> None:
    fields = request.extra.fields
    detokenize = fields.get(REQ_DETOKENIZE)
    if detokenize is not None and detokenize.WhichOneof("kind") != "bool_value":
        raise _invalid(f"extra.{REQ_DETOKENIZE} must be a boolean")
    for key, value in fields.items():
        capability = _REQUEST_KEY_CAPABILITY.get(key)
        if capability is None:
            if key.startswith("trtllm.") and profile.follows_conventions:
                raise _unimplemented(f"extra.{key} is not supported by this engine")
            continue
        if not profile.capability(capability):
            if profile.follows_conventions:
                raise _unimplemented(f"extra.{key} is not supported by this engine")
            continue
        _validate_request_key(key, value, profile, num_sequences)


def _is_int(value: struct_pb2.Value, minimum: int | None = None) -> bool:
    if value.WhichOneof("kind") != "number_value":
        return False
    number = value.number_value
    return float(number).is_integer() and (minimum is None or number >= minimum)


def _json_data(value: struct_pb2.Value, key: str) -> Any:
    """A ``Struct`` value as JSON data, with integral numbers as ints.

    A finite double of magnitude ``2**53`` or more may be a rounded integer, so
    it is refused rather than read as a value the client did not send.
    """
    kind = value.WhichOneof("kind")
    if kind == "struct_value":
        return {name: _json_data(item, key) for name, item in value.struct_value.fields.items()}
    if kind == "list_value":
        return [_json_data(item, key) for item in value.list_value.values]
    if kind == "number_value":
        number = value.number_value
        if math.isfinite(number) and abs(number) >= 2**53:
            raise _invalid(f"extra.{key} as an object holds a number of magnitude 2**53 or more")
        return int(number) if number.is_integer() else number
    if kind == "string_value":
        return value.string_value
    if kind == "bool_value":
        return value.bool_value
    return None


def _refuse_json_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _validate_kv_hints(key: str, value: struct_pb2.Value) -> None:
    """Admit ``trtllm.kv_hints``: the envelope's JSON text, or the envelope as an object.

    The text is at most ``KV_HINTS_TEXT_MAX_BYTES`` long, has no ``NaN`` or
    ``Infinity``, and holds a JSON object. Either form is then refused when
    the KV hints protocol's admission would drop the envelope.
    """
    # Imported here: only this key needs the KV hints protocol and its dependencies.
    from tensorrt_llm.kv_hints.protocol import admit_kv_hints

    kind = value.WhichOneof("kind")
    if kind == "string_value":
        text = value.string_value
        limit = KV_HINTS_TEXT_MAX_BYTES
        if len(text) > limit or len(text.encode("utf-8")) > limit:
            raise _invalid(f"extra.{key} is longer than {KV_HINTS_TEXT_MAX_BYTES} bytes")
        try:
            envelope = json.loads(text, parse_constant=_refuse_json_constant)
        except (ValueError, RecursionError):
            raise _invalid(f"extra.{key} is not valid JSON") from None
        if not isinstance(envelope, dict):
            raise _invalid(f"extra.{key} must be the JSON text of an object")
    elif kind == "struct_value":
        envelope = _json_data(value, key)
    else:
        raise _invalid(f"extra.{key} must be a JSON object as text, or an object")
    if admit_kv_hints(envelope) is None:
        raise _invalid(
            f"extra.{key} was not admitted: unsupported protocol_version, oversized or malformed"
        )


def _validate_request_key(
    key: str, value: struct_pb2.Value, profile: FakeProfile, num_sequences: int
) -> None:
    kind = value.WhichOneof("kind")
    if key == REQ_CONVERSATION_ID:
        if kind != "string_value" or not value.string_value.strip():
            raise _invalid(f"extra.{key} must be a string that is not blank")
        return
    if key == REQ_BEST_OF:
        if not _is_int(value) or value.number_value <= num_sequences:
            raise _invalid(f"extra.{key} must be an integer greater than num_sequences")
        return
    if key == REQ_KV_HINTS:
        _validate_kv_hints(key, value)
        return
    if kind != "struct_value":
        raise _invalid(f"extra.{key} must be an object")
    entries = value.struct_value.fields
    if key == REQ_THINKING_TOKEN_BUDGET:
        budget = entries.get("budget")
        if budget is None or not _is_int(budget, minimum=0):
            raise _invalid(f"extra.{key}.budget must be an integer >= 0")
        for name in ("start_token_ids", "end_token_ids"):
            ids = entries.get(name)
            if (
                ids is None
                or ids.WhichOneof("kind") != "list_value"
                or not ids.list_value.values
                or not all(_is_int(item, minimum=0) for item in ids.list_value.values)
            ):
                raise _invalid(f"extra.{key}.{name} must be a non-empty list of token ids")
    elif key == REQ_LOGIT_BIAS:
        if len(entries) > MAX_LOGIT_BIAS_ENTRIES:
            raise _invalid(f"extra.{key} has more than {MAX_LOGIT_BIAS_ENTRIES} entries")
        for token, bias in entries.items():
            if not (token.isascii() and token.isdigit()) or int(token) >= profile.vocab_size:
                raise _invalid(f"extra.{key} key {token!r} is not a token id in the vocabulary")
            if bias.WhichOneof("kind") != "number_value" or not (
                LOGIT_BIAS_MIN <= bias.number_value <= LOGIT_BIAS_MAX
            ):
                raise _invalid(
                    f"extra.{key}[{token}] must be a number in "
                    f"[{LOGIT_BIAS_MIN:g}, {LOGIT_BIAS_MAX:g}]"
                )
    elif key == REQ_SCHEDULING:
        unknown = set(entries) - {"attention_dp_relax", "agent_hierarchy"}
        if unknown:
            raise _invalid(f"extra.{key} has unknown fields {sorted(unknown)}")
        relax = entries.get("attention_dp_relax")
        if relax is not None and relax.WhichOneof("kind") != "bool_value":
            raise _invalid(f"extra.{key}.attention_dp_relax must be a boolean")
        hierarchy = entries.get("agent_hierarchy")
        if hierarchy is not None:
            pairs = (
                hierarchy.list_value.values
                if hierarchy.WhichOneof("kind") == "list_value"
                else None
            )
            if pairs is None or not all(
                pair.WhichOneof("kind") == "list_value"
                and len(pair.list_value.values) == 2
                and pair.list_value.values[0].WhichOneof("kind") == "string_value"
                and _is_int(pair.list_value.values[1])
                for pair in pairs
            ):
                raise _invalid(f"extra.{key}.agent_hierarchy must be a list of [string, int]")


def _admit_stopping(
    request: generation_pb2.GenerateRequest, profile: FakeProfile, prompt_length: int
) -> tuple[int, int, bool, bool, frozenset[int], tuple[str, ...]]:
    stopping = request.stopping
    room = profile.max_context_length - prompt_length
    if room <= 0:
        raise _invalid(
            f"prompt of {prompt_length} tokens does not fit the context window of "
            f"{profile.max_context_length} tokens"
        )
    max_tokens = stopping.max_tokens if stopping.HasField("max_tokens") else room
    if max_tokens == 0:
        raise _invalid("stopping.max_tokens must be greater than zero")
    if max_tokens > room:
        raise _invalid(
            f"prompt of {prompt_length} tokens plus max_tokens {max_tokens} exceeds the "
            f"context window of {profile.max_context_length} tokens"
        )
    if len(stopping.conditions) > MAX_STOP_CONDITIONS:
        raise _invalid(
            f"at most {MAX_STOP_CONDITIONS} stopping conditions are supported, "
            f"got {len(stopping.conditions)}"
        )
    stop_token_ids: set[int] = set()
    stop_texts: list[str] = []
    for condition in stopping.conditions:
        kind = condition.WhichOneof("condition")
        if kind == "stop_token_id":
            stop_token_ids.add(condition.stop_token_id)
        elif kind == "stop_text":
            if not profile.capability(CAP_STOP_TEXT):
                raise _unimplemented("stopping.conditions[].stop_text is not supported")
            if not condition.stop_text:
                raise _invalid("stopping.conditions[].stop_text must not be empty")
            stop_texts.append(condition.stop_text)
        else:
            raise _invalid("each stopping condition must set exactly one condition")
    include_stop = stopping.HasField("include_stop_in_output") and stopping.include_stop_in_output
    if include_stop and not profile.supports_stop_in_output:
        raise _unimplemented("stopping.include_stop_in_output is not supported")
    return (
        max_tokens,
        stopping.min_tokens,
        stopping.HasField("ignore_eos") and stopping.ignore_eos,
        include_stop,
        frozenset(stop_token_ids),
        tuple(stop_texts),
    )


def _admit_response(
    request: generation_pb2.GenerateRequest, profile: FakeProfile
) -> tuple[int | None, int | None]:
    response = request.response
    if response.HasField("prompt_logprob_start") and response.prompt_logprob_start != 0:
        raise _unimplemented("response.prompt_logprob_start is not supported")
    output_top_n = prompt_top_n = None
    if response.return_output_logprobs:
        if profile.output_logprobs_max_top_n is None:
            raise _unimplemented("output logprobs are not supported")
        output_top_n = _top_n(
            response.output_candidates, "output", profile.output_logprobs_max_top_n
        )
    if response.return_prompt_logprobs:
        if profile.prompt_logprobs_max_top_n is None:
            raise _unimplemented("prompt logprobs are not supported")
        prompt_top_n = _top_n(
            response.prompt_candidates, "prompt", profile.prompt_logprobs_max_top_n
        )
    return output_top_n, prompt_top_n


def _admit_options(
    request: generation_pb2.GenerateRequest, profile: FakeProfile, loras: Mapping[str, Any]
) -> None:
    if request.HasField("guided"):
        if request.guided.backend:
            raise _unimplemented("per-request guided decoding backend selection is not supported")
        guide = request.guided.WhichOneof("guide")
        if guide is not None and guide not in profile.guided_modes:
            raise _unimplemented(f"guided decoding mode '{guide}' is not supported")
    kv = request.kv
    if kv.HasField("bypass_prefix_cache") and kv.bypass_prefix_cache:
        raise _unimplemented("prefix cache bypass is not supported")
    if kv.HasField("cache_salt") and not profile.supports_cache_salt:
        raise _unimplemented("kv.cache_salt is not supported")
    if kv.HasField("session") and profile.reject_kv_session:
        raise _unimplemented("kv.session is not supported")
    for index, item in enumerate(request.media):
        modality = _MODALITY_NAME.get(item.modality)
        if modality not in profile.modalities:
            raise _unimplemented(
                f"media[{index}] modality {modality or 'unspecified'} is not supported"
            )
        if item.WhichOneof("source") is None:
            raise _invalid(f"media[{index}] has no source")
    if request.lora_name:
        if not profile.supports_lora:
            raise _unimplemented("LoRA selection is not supported")
        if request.lora_name not in loras:
            raise Rejection(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"LoRA adapter '{request.lora_name}' is not loaded",
            )


@dataclass
class _DisaggCall:
    """The fields of :class:`Admitted` that :func:`_admit_disagg` decides."""

    bucket: str
    disagg_request_id: int | None = None
    kv_session: kv_pb2.KvSessionRef | None = None
    first_gen_tokens: tuple[int, ...] = ()
    subagent_affinity_id: str | None = None


def _check_leg(key: str, leg: str, is_prefill: bool, is_decode: bool) -> None:
    if (leg == "prefill" and not is_prefill) or (leg == "decode" and not is_decode):
        raise _invalid(f"extra.{key} belongs only on the {leg} call of a disaggregated request")


def _admit_disagg_keys(
    fields: Mapping[str, struct_pb2.Value], profile: FakeProfile, is_prefill: bool, is_decode: bool
) -> int | None:
    """Check the disaggregated request keys; return ``disagg_request_id`` if sent."""
    disagg_request_id = None
    for key, capability in DISAGG_REQUEST_KEY_CAPABILITY.items():
        value = fields.get(key)
        if value is None:
            continue
        if not profile.capability(capability):
            raise _unimplemented(f"extra.{key} is not supported by this engine")
        kind = value.WhichOneof("kind")
        if key == REQ_DISAGG_REQUEST_ID:
            _check_leg(key, "prefill", is_prefill, is_decode)
            disagg_request_id = parse_disagg_request_id(
                value.string_value if kind == "string_value" else None
            )
            if disagg_request_id is None:
                raise _invalid(
                    f"extra.{key} must be a decimal string in "
                    f"[{DISAGG_REQUEST_ID_MIN}, {DISAGG_REQUEST_ID_LIMIT})"
                )
        elif key == REQ_SCHEDULE_STYLE:
            _check_leg(key, "prefill", is_prefill, is_decode)
            if not _is_int(value) or value.number_value not in (
                SCHEDULE_STYLE_CONTEXT_FIRST,
                SCHEDULE_STYLE_GENERATION_FIRST,
            ):
                raise _invalid(
                    f"extra.{key} must be {SCHEDULE_STYLE_CONTEXT_FIRST} "
                    f"or {SCHEDULE_STYLE_GENERATION_FIRST}"
                )
        else:
            _check_leg(key, "decode", is_prefill, is_decode)
            if (
                kind != "number_value"
                or not math.isfinite(value.number_value)
                or value.number_value < 0
            ):
                raise _invalid(f"extra.{key} must be a non-negative number")
    return disagg_request_id


def _admit_handoff(
    request: generation_pb2.GenerateRequest, profile: FakeProfile
) -> tuple[int, ...]:
    """Check the ``kv.session`` of ``request`` as the engine of ``profile`` does.

    Returns the handoff's first tokens. It reads the handoff with the
    TensorRT-LLM server's mapping, whose ``ValueError`` the server also
    answers with ``INVALID_ARGUMENT``. Unlike the server, it answers the
    mapping's other failures (a value of the wrong type or out of range)
    that way too, and
    refuses first tokens outside the vocabulary.
    """
    # Imported here: only a decode call needs the server's handoff mapping and
    # its dependencies.
    from tensorrt_llm.grpc.openengine.disagg import disaggregated_params_from_request

    try:
        params = disaggregated_params_from_request(request)
    except (ValueError, TypeError, AttributeError, OverflowError) as error:
        raise _invalid(str(error)) from None
    first_gen_tokens = tuple(params.first_gen_tokens or ())
    if profile.handoff_shape == "ds":
        if not params.ctx_info_endpoint:
            raise _invalid("kv.session has no ctx_info_endpoint")
        if not first_gen_tokens:
            raise _invalid("kv.session has no first_gen_tokens")
        known = set(HANDOFF_MAPPED_ATTRIBUTES) | set(HANDOFF_REGISTERED_PASSTHROUGH_ATTRIBUTES)
        unknown = sorted(set(request.kv.session.attributes_struct.fields) - known)
        if unknown:
            raise _invalid(f"kv.session has attributes the conventions do not name: {unknown}")
    if any(token >= profile.vocab_size or token < 0 for token in first_gen_tokens):
        raise _invalid("kv.session first_gen_tokens are outside the vocabulary")
    return first_gen_tokens


def _affinity_headers(metadata: Sequence[tuple[str, str]]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in metadata:
        if key in (AFFINITY_ID_HEADER, AFFINITY_AUTH_HEADER):
            if key in headers:
                raise _invalid(f"gRPC metadata key '{key}' must not be repeated")
            headers[key] = value
    return headers


def _admit_disagg(
    request: generation_pb2.GenerateRequest,
    metadata: Sequence[tuple[str, str]],
    profile: FakeProfile,
    num_sequences: int,
) -> _DisaggCall:
    """Admit the disaggregated-serving parts of a call to a profile with a ``handoff_shape``.

    Checks the request type and ``kv.session`` first, as the TensorRT-LLM
    server does, then the engine's KV connector, which only the fake checks
    here, then the rest.
    """
    fields = request.extra.fields
    is_decode = request.kv.HasField("session")
    # As on the server, kv.session alone makes a decode call: request_type is not read.
    request_type_value = None if is_decode else fields.get(REQ_REQUEST_TYPE)
    request_type = None
    if request_type_value is not None:
        if request_type_value.WhichOneof("kind") != "string_value" or (
            request_type_value.string_value
            not in (REQUEST_TYPE_CONTEXT_ONLY, REQUEST_TYPE_CONTEXT_AND_GENERATION)
        ):
            if request_type_value.string_value == REQUEST_TYPE_GENERATION_ONLY:
                raise _invalid(
                    "a generation_only request carries its handoff in kv.session; "
                    f"extra.{REQ_REQUEST_TYPE} alone cannot address one"
                )
            raise _invalid(
                f"extra.{REQ_REQUEST_TYPE} must be "
                f"'{REQUEST_TYPE_CONTEXT_ONLY}' or '{REQUEST_TYPE_CONTEXT_AND_GENERATION}'"
            )
        request_type = request_type_value.string_value
    is_prefill = not is_decode and request_type == REQUEST_TYPE_CONTEXT_ONLY
    call = _DisaggCall(bucket=request_bucket(has_kv_session=is_decode, request_type=request_type))
    if is_decode:
        call.first_gen_tokens = _admit_handoff(request, profile)
        call.kv_session = request.kv.session
    if is_prefill or is_decode:
        connector = profile.kv_connector
        if connector is None or not connector.takes_disaggregated_calls:
            raise Rejection(
                grpc.StatusCode.FAILED_PRECONDITION,
                "this engine has no KV transfer enabled, so it serves no disaggregated call",
            )
        if num_sequences > 1:
            raise _invalid("a disaggregated call has one output sequence")
        wanted = "context" if is_prefill else "generation"
        if profile.instance_role not in (None, wanted):
            raise _invalid(
                f"this {profile.instance_role} instance does not take {call.bucket} calls"
            )
    call.disagg_request_id = _admit_disagg_keys(fields, profile, is_prefill, is_decode)
    headers = _affinity_headers(metadata)
    conversation_id = fields.get(REQ_CONVERSATION_ID)
    try:
        call.subagent_affinity_id = validate_parent_affinity(
            profile.subagent_affinity_key,
            request,
            conversation_id.string_value if conversation_id is not None else None,
            REQUEST_TYPE_GENERATION_ONLY if is_decode else request_type,
            headers,
        )
    except ValueError as error:
        raise _invalid(str(error)) from None
    return call


def admit(
    request: generation_pb2.GenerateRequest,
    metadata: Sequence[tuple[str, str]],
    profile: FakeProfile,
    loras: Mapping[str, Any] | None = None,
) -> Admitted:
    """Check ``request`` against ``profile``; raise :class:`Rejection` if refused.

    Decides from the request, its metadata, the profile and the LoRA adapters
    in ``loras``. :class:`FakeEngine` checks the rest before calling it: an
    empty or already running ``request_id``, and whether the engine is ready.
    """
    if not request.model:
        raise _invalid("model must be non-empty")
    prompt_token_ids = _admit_input(request, profile)
    num_sequences = _admit_sampling(request, profile)
    max_tokens, min_tokens, ignore_eos, include_stop, stop_token_ids, stop_texts = _admit_stopping(
        request, profile, len(prompt_token_ids)
    )
    output_top_n, prompt_top_n = _admit_response(request, profile)
    _admit_options(request, profile, loras or {})
    _admit_extra(request, profile, num_sequences)
    disagg = (
        _DisaggCall(bucket=BUCKET_AGGREGATED)
        if profile.handoff_shape is None
        else _admit_disagg(request, metadata, profile, num_sequences)
    )
    # INVALID_ARGUMENT, not UNIMPLEMENTED, as ds answers: a known exception to CAP-GEN-3.
    if (
        profile.handoff_shape is not None
        and min_tokens >= 1
        and not profile.capability(CAP_STOPPING_MIN_TOKENS)
    ):
        raise _invalid("stopping.min_tokens is not supported")
    priority, target_dp_rank = _admit_metadata(metadata, profile)
    if target_dp_rank is not None and num_sequences > 1:
        raise _invalid(f"{METADATA_TARGET_DP_RANK} does not support multiple output sequences")
    return Admitted(
        prompt_token_ids=prompt_token_ids,
        num_sequences=num_sequences,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        ignore_eos=ignore_eos,
        include_stop=include_stop,
        stop_token_ids=stop_token_ids,
        stop_texts=stop_texts,
        output_top_n=output_top_n,
        prompt_top_n=prompt_top_n,
        priority=priority,
        target_dp_rank=target_dp_rank,
        bucket=disagg.bucket,
        disagg_request_id=disagg.disagg_request_id,
        kv_session=disagg.kv_session,
        first_gen_tokens=disagg.first_gen_tokens,
        subagent_affinity_id=disagg.subagent_affinity_id,
    )


# -- Script steps ------------------------------------------------------------------


@dataclass(frozen=True)
class Delay:
    """Script step: wait ``seconds`` before the next step."""

    seconds: float


@dataclass(frozen=True)
class Fail:
    """Script step: end the call with a non-OK status.

    As the first step it refuses the request before acceptance, so it does not
    count in ``generate_accepted``.
    """

    code: grpc.StatusCode
    details: str = "scripted failure"


@dataclass(frozen=True)
class Disconnect:
    """Script step: drop the call as a lost connection does (``UNAVAILABLE``)."""

    details: str = "fake engine connection lost"


class Hang:
    """Script step: send nothing more until the call is cancelled or aborted."""


Step = Any  # GenerateResponse | Delay | Fail | Disconnect | Hang
ScriptSource = Callable[[generation_pb2.GenerateRequest], "Iterable[Step] | None"]


def _checked_steps(steps: Iterable[Step]) -> list[Step]:
    steps = list(steps)
    for step in steps:
        if step is not Hang and not isinstance(
            step, (generation_pb2.GenerateResponse, Delay, Fail, Disconnect, Hang)
        ):
            raise TypeError(
                f"unsupported script step {step!r}: use a GenerateResponse or "
                "a Delay, Fail, Disconnect or Hang from this module"
            )
    return steps


@dataclass
class DisaggFaults:
    """Disaggregated-serving misbehaviour a :class:`FakeEngine` shows on purpose.

    Everything is off by default. Only calls answered by the built-in
    generator misbehave; a scripted call does what its script says.

    Attributes:
        finish_prefill_early: A prefill call ends as if its first token were
            EOS: ``finished`` with ``STOP`` and usage, and no ``prefill_ready``.
        extra_attributes: Added to the ``attributes_struct`` of every handoff,
            over the engine's own keys.
        close_before_terminal: Request buckets (``BUCKET_*``) whose calls end
            with status OK right after their first ``token`` event, without
            a terminal event.
        prefill_ready_on_any_call: Every call, not only a prefill call, gets
            its first token and then ``prefill_ready``.
    """

    finish_prefill_early: bool = False
    extra_attributes: Mapping[str, Any] = field(default_factory=dict)
    close_before_terminal: frozenset[str] = frozenset()
    prefill_ready_on_any_call: bool = False


def default_detokenizer(token_ids: Sequence[int]) -> str:
    """Text of ``token_ids`` when the engine has no tokenizer: ``<id>`` per token."""
    return "".join(f"<{token_id}>" for token_id in token_ids)


# -- Built-in generator --------------------------------------------------------------

_SAMPLED_LOGPROB = -0.5


def _token_info(
    token_id: int,
    *,
    logprob: float | None,
    top_n: int,
    vocab_size: int,
    detokenizer: Callable[[Sequence[int]], str],
) -> generation_pb2.TokenInfo:
    if logprob is None:
        return generation_pb2.TokenInfo(token_id=token_id)
    info = generation_pb2.TokenInfo(
        token_id=token_id, token=detokenizer([token_id]), logprob=logprob, rank=1
    )
    for rank in range(1, top_n + 1):
        candidate = (token_id + rank - 1) % vocab_size
        info.candidates.append(
            generation_pb2.LogProb(
                token_id=candidate,
                logprob=logprob - (rank - 1),
                token=detokenizer([candidate]),
                rank=rank,
            )
        )
    return info


@dataclass
class _Output:
    index: int
    cursor: int = 0
    sampled: int = 0
    tokens: list[int] = field(default_factory=list)
    sent: int = 0
    finished: generation_pb2.GenerationFinished | None = None


class _Generator:
    """Deterministic token streams for one admitted request."""

    def __init__(
        self,
        admitted: Admitted,
        profile: FakeProfile,
        detokenizer: Callable[[Sequence[int]], str],
    ) -> None:
        self.admitted = admitted
        self.profile = profile
        self.detokenizer = detokenizer
        self.outputs = [_Output(index) for index in range(admitted.num_sequences)]
        eos = () if admitted.ignore_eos else profile.eos_token_ids
        self._eos = frozenset(eos)
        self._suppressed = self._eos | admitted.stop_token_ids
        self._hold_back = bool(admitted.stop_texts) and not admitted.include_stop

    def _next_token(self, output: _Output) -> int:
        token_id = self._walk(output)
        # A decode call starts with the tokens its handoff carries. Each takes
        # the place of one step of the walk, which still moves the cursor as
        # it would have, so the rest of the output is the aggregated call's.
        forced = self.admitted.first_gen_tokens
        if output.index == 0 and output.sampled < len(forced):
            return forced[output.sampled]
        return token_id

    def _walk(self, output: _Output) -> int:
        prompt = self.admitted.prompt_token_ids
        suppressing = output.sampled < self.admitted.min_tokens
        for _ in range(len(prompt)):
            token_id = prompt[(output.index + output.cursor) % len(prompt)]
            output.cursor += 1
            if not (suppressing and token_id in self._suppressed):
                return token_id
        # Every prompt token is suppressed: sample the smallest id that is not,
        # or the next prompt token if the whole vocabulary is.
        return next(
            (i for i in range(self.profile.vocab_size) if i not in self._suppressed),
            prompt[(output.index + output.cursor) % len(prompt)],
        )

    def _finish(self, output: _Output, reason: int, **stop_match: Any) -> None:
        output.finished = generation_pb2.GenerationFinished(
            output_index=output.index, reason=reason
        )
        if stop_match:
            output.finished.stop_match.CopyFrom(generation_pb2.StopMatch(**stop_match))

    def _tokens_within(self, tokens: Sequence[int], length: int) -> int:
        """How many leading ``tokens`` have text no longer than ``length``."""
        count = len(tokens)
        while count and len(self.detokenizer(tokens[:count])) > length:
            count -= 1
        return count

    def _sample(self, output: _Output) -> None:
        admitted = self.admitted
        token_id = self._next_token(output)
        output.sampled += 1
        if token_id in self._eos:
            self._finish(output, generation_pb2.FINISH_REASON_STOP, eos_token_id=token_id)
            return
        if token_id in admitted.stop_token_ids:
            if admitted.include_stop:
                output.tokens.append(token_id)
            self._finish(output, generation_pb2.FINISH_REASON_STOP, stop_token_id=token_id)
            return
        before = self.detokenizer(output.tokens) if admitted.stop_texts else ""
        output.tokens.append(token_id)
        if admitted.stop_texts and output.sampled > admitted.min_tokens:
            text = self.detokenizer(output.tokens)
            for stop_text in admitted.stop_texts:
                # Only a match that ends in the new token's text counts.
                start = text.find(stop_text, max(0, len(before) - len(stop_text) + 1))
                if start != -1:
                    if not admitted.include_stop:
                        del output.tokens[self._tokens_within(output.tokens, start) :]
                    self._finish(output, generation_pb2.FINISH_REASON_STOP, stop_text=stop_text)
                    return
        if output.sampled >= admitted.max_tokens:
            self._finish(output, generation_pb2.FINISH_REASON_LENGTH)

    def _sendable(self, output: _Output) -> int:
        """How many tokens of ``output`` can be sent: all but a possible stop-string start."""
        if output.finished is not None or not self._hold_back:
            return len(output.tokens)
        text = self.detokenizer(output.tokens)
        longest = max(len(stop_text) for stop_text in self.admitted.stop_texts)
        for start in range(max(0, len(text) - longest + 1), len(text)):
            if any(stop_text.startswith(text[start:]) for stop_text in self.admitted.stop_texts):
                return max(output.sent, self._tokens_within(output.tokens, start))
        return len(output.tokens)

    def token_info(self, token_id: int) -> generation_pb2.TokenInfo:
        """``token_id`` as it is sent, with logprobs when they were requested."""
        top_n = self.admitted.output_top_n
        return _token_info(
            token_id,
            logprob=None if top_n is None else _SAMPLED_LOGPROB,
            top_n=top_n or 0,
            vocab_size=self.profile.vocab_size,
            detokenizer=self.detokenizer,
        )

    def _deliver(
        self, output: _Output, sendable: int | None = None
    ) -> list[generation_pb2.GenerateResponse]:
        if sendable is None:
            sendable = self._sendable(output)
        tokens = [self.token_info(token_id) for token_id in output.tokens[output.sent : sendable]]
        output.sent = max(output.sent, sendable)
        responses = []
        if tokens:
            responses.append(
                generation_pb2.GenerateResponse(
                    token=generation_pb2.TokenOutput(output_index=output.index, tokens=tokens)
                )
            )
        if output.finished is not None:
            responses.append(generation_pb2.GenerateResponse(finished=output.finished))
        return responses

    def prompt_event(self) -> generation_pb2.PromptOutput | None:
        top_n = self.admitted.prompt_top_n
        if top_n is None:
            return None
        prompt = self.admitted.prompt_token_ids
        return generation_pb2.PromptOutput(
            tokens=[
                _token_info(
                    token_id,
                    logprob=None if position == 0 else -1.0,
                    top_n=0 if position == 0 else top_n,
                    vocab_size=self.profile.vocab_size,
                    detokenizer=self.detokenizer,
                )
                for position, token_id in enumerate(prompt)
            ]
        )

    def step(self, tokens_per_step: int) -> list[generation_pb2.GenerateResponse]:
        """Advance every unfinished output by up to ``tokens_per_step`` tokens."""
        responses = []
        for output in self.outputs:
            if output.finished is not None:
                continue
            for _ in range(tokens_per_step):
                self._sample(output)
                if output.finished is not None:
                    break
            responses.extend(self._deliver(output))
        return responses

    def first_token(self, *, stops: bool) -> list[generation_pb2.GenerateResponse]:
        """Sample the first token of output 0, the one token of a prefill call, and send it.

        With ``stops``, an EOS or stop token ends the output with ``STOP`` as
        in :meth:`step`. Otherwise no stopping condition and not
        ``max_tokens`` end it: the token is sent, whatever it is.
        """
        output = self.outputs[0]
        token_id = self._next_token(output)
        output.sampled += 1
        if stops and token_id in self._eos:
            self._finish(output, generation_pb2.FINISH_REASON_STOP, eos_token_id=token_id)
        elif stops and token_id in self.admitted.stop_token_ids:
            if self.admitted.include_stop:
                output.tokens.append(token_id)
            self._finish(output, generation_pb2.FINISH_REASON_STOP, stop_token_id=token_id)
        else:
            output.tokens.append(token_id)
        return self._deliver(output, sendable=len(output.tokens))

    def finish_on_eos(self) -> list[generation_pb2.GenerateResponse]:
        """End output 0 with ``STOP`` as if it had sampled EOS first."""
        output = self.outputs[0]
        output.sampled += 1
        stop_match = (
            {"eos_token_id": self.profile.eos_token_ids[0]} if self.profile.eos_token_ids else {}
        )
        self._finish(output, generation_pb2.FINISH_REASON_STOP, **stop_match)
        return self._deliver(output)

    def cancel(self) -> list[generation_pb2.GenerateResponse]:
        """End every unfinished output with ``CANCELLED``, sending what it held back."""
        responses = []
        for output in self.outputs:
            if output.finished is None:
                self._finish(output, generation_pb2.FINISH_REASON_CANCELLED)
                responses.extend(self._deliver(output))
        return responses

    @property
    def done(self) -> bool:
        return all(output.finished is not None for output in self.outputs)

    def usage(self, cached_prompt_tokens: int) -> generation_pb2.Usage:
        prompt_tokens = len(self.admitted.prompt_token_ids)
        completion_tokens = sum(len(output.tokens) for output in self.outputs)
        return generation_pb2.Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
        )


# -- The engine ----------------------------------------------------------------------


@dataclass
class ReceivedRequest:
    """One ``Generate`` call as the engine saw it.

    Attributes:
        request: The request.
        metadata: Invocation metadata, in arrival order.
        admitted: Settings of the accepted request; None if it was refused.
        rejection: Why it was refused before acceptance; None if accepted.
    """

    request: generation_pb2.GenerateRequest
    metadata: tuple[tuple[str, str], ...]
    admitted: Admitted | None = None
    rejection: Rejection | None = None

    def metadata_value(self, key: str) -> str | None:
        """The value of metadata ``key``; None if absent."""
        return next((value for name, value in self.metadata if name == key), None)


class _Call:
    def __init__(self) -> None:
        self.aborted = asyncio.Event()
        # Set before the terminal event goes out: Abort no longer applies.
        self.finished = False

    async def pause(self, seconds: float) -> None:
        """Sleep up to ``seconds``; return early once the call is aborted."""
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.aborted.wait(), seconds)


def _cancelled_outputs(
    request_id: str, indices: Iterable[int], usage: generation_pb2.Usage
) -> list[generation_pb2.GenerateResponse]:
    responses = [
        generation_pb2.GenerateResponse(
            request_id=request_id,
            finished=generation_pb2.GenerationFinished(
                output_index=index, reason=generation_pb2.FINISH_REASON_CANCELLED
            ),
        )
        for index in indices
    ]
    if responses:
        responses[-1].usage.CopyFrom(usage)
    return responses


class _Inference(openengine_pb2_grpc.InferenceServicer):
    def __init__(self, engine: FakeEngine) -> None:
        self._engine = engine

    async def Generate(
        self, request: generation_pb2.GenerateRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[generation_pb2.GenerateResponse]:
        engine = self._engine
        metadata = tuple((item.key, item.value) for item in context.invocation_metadata())
        record = ReceivedRequest(request=request, metadata=metadata)
        if engine.record:
            record.request = generation_pb2.GenerateRequest()
            record.request.CopyFrom(request)
            engine.received.append(record)
        engine._counters[COUNTER_GENERATE_RECEIVED] += 1
        request_id = request.request_id
        try:
            if not request_id:
                raise _invalid("request_id must be non-empty")
            if request_id in engine._active:
                raise Rejection(
                    grpc.StatusCode.ALREADY_EXISTS, f"request_id '{request_id}' is already active"
                )
            if engine.health_state != lifecycle_pb2.HEALTH_STATE_READY:
                raise Rejection(grpc.StatusCode.UNAVAILABLE, "the engine is not ready")
            admitted = admit(request, metadata, engine.profile, engine.loras)
            steps = engine._script_for(request)
            if steps and isinstance(steps[0], Fail):
                raise Rejection(steps[0].code, steps[0].details)
        except Rejection as rejection:
            record.rejection = rejection
            await context.abort(rejection.code, rejection.message)
            return
        record.admitted = admitted
        engine._counters[COUNTER_GENERATE_ACCEPTED] += 1
        engine._counters[COUNTER_GENERATE_ACCEPTED_BUCKETS[admitted.bucket]] += 1
        call = engine._active[request_id] = _Call()
        try:
            if steps is not None:
                responses = self._scripted(request_id, admitted, steps, call, context)
            elif (
                admitted.bucket == BUCKET_CONTEXT_ONLY
                or engine.disagg_faults.prefill_ready_on_any_call
            ):
                responses = self._prefilled(request_id, admitted, call)
            else:
                responses = self._generated(request_id, admitted, call)
            async for response in responses:
                yield response
        except asyncio.CancelledError:
            if not engine._stopping:
                if engine.record:
                    engine.cancelled.append(request_id)
                engine.cancel_seen.set()
            raise
        finally:
            if engine._active.get(request_id) is call:
                del engine._active[request_id]

    async def _generated(
        self, request_id: str, admitted: Admitted, call: _Call
    ) -> AsyncIterator[generation_pb2.GenerateResponse]:
        engine = self._engine
        close_early = admitted.bucket in engine.disagg_faults.close_before_terminal
        generator = _Generator(admitted, engine.profile, engine.detokenizer)
        prompt = generator.prompt_event()
        if prompt is not None:
            yield generation_pb2.GenerateResponse(request_id=request_id, prompt=prompt)
        while not generator.done:
            if call.aborted.is_set():
                batch = generator.cancel()
            else:
                batch = generator.step(engine.tokens_per_step)
            if close_early and any(response.HasField("token") for response in batch):
                call.finished = True
                for response in batch:
                    if response.HasField("token"):
                        response.request_id = request_id
                        yield response
                return
            if generator.done:
                batch[-1].usage.CopyFrom(generator.usage(engine.cached_prompt_tokens))
                call.finished = True
            for response in batch:
                response.request_id = request_id
                yield response
            if not generator.done:
                await call.pause(engine.step_delay)

    async def _prefilled(
        self, request_id: str, admitted: Admitted, call: _Call
    ) -> AsyncIterator[generation_pb2.GenerateResponse]:
        """A prefill call: the first token, then ``prefill_ready`` with the handoff."""
        engine = self._engine
        faults = engine.disagg_faults
        generator = _Generator(admitted, engine.profile, engine.detokenizer)
        prompt = generator.prompt_event()
        if prompt is not None:
            yield generation_pb2.GenerateResponse(request_id=request_id, prompt=prompt)
        if faults.finish_prefill_early:
            batch = generator.finish_on_eos()
        else:
            # Only the ds shape lets the first token stop a prefill call (see FakeEngine).
            batch = generator.first_token(stops=engine.profile.handoff_shape == "ds")
        if not generator.done:
            for response in batch:
                response.request_id = request_id
                yield response
            if admitted.bucket in faults.close_before_terminal:
                call.finished = True
                return
            await call.pause(engine.step_delay)
            if not call.aborted.is_set():
                call.finished = True
                yield engine._prefill_ready(request_id, admitted, generator)
                return
            batch = generator.cancel()
        batch[-1].usage.CopyFrom(generator.usage(engine.cached_prompt_tokens))
        call.finished = True
        for response in batch:
            response.request_id = request_id
            yield response

    async def _scripted(
        self,
        request_id: str,
        admitted: Admitted,
        steps: Sequence[Step],
        call: _Call,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[generation_pb2.GenerateResponse]:
        finished: set[int] = set()
        completion_tokens = 0
        for step in steps:
            if call.aborted.is_set():
                break
            if isinstance(step, Delay):
                await call.pause(step.seconds)
            elif isinstance(step, Fail):
                await context.abort(step.code, step.details)
            elif isinstance(step, Disconnect):
                await context.abort(grpc.StatusCode.UNAVAILABLE, step.details)
            elif step is Hang or isinstance(step, Hang):
                await call.aborted.wait()
            else:
                response = generation_pb2.GenerateResponse()
                response.CopyFrom(step)
                if not response.request_id:
                    response.request_id = request_id
                event = response.WhichOneof("event")
                if event == "token":
                    completion_tokens += len(response.token.tokens)
                elif event == "finished":
                    finished.add(response.finished.output_index)
                elif event == "prefill_ready":
                    self._engine._counters[COUNTER_PREFILL_READY_SENT] += 1
                terminal = event in ("error", "prefill_ready") or (
                    event == "finished" and len(finished) >= admitted.num_sequences
                )
                call.finished = terminal
                yield response
                if terminal:
                    return
        if not call.aborted.is_set():
            await context.abort(
                grpc.StatusCode.INTERNAL, "the script ended before every output finished"
            )
        else:
            prompt_tokens = len(admitted.prompt_token_ids)
            usage = generation_pb2.Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cached_prompt_tokens=self._engine.cached_prompt_tokens,
            )
            unfinished = [i for i in range(admitted.num_sequences) if i not in finished]
            call.finished = True
            for response in _cancelled_outputs(request_id, unfinished, usage):
                yield response


class _Control(openengine_pb2_grpc.ControlServicer):
    def __init__(self, engine: FakeEngine) -> None:
        self._engine = engine

    def _record(self, rpc: str, request: Any) -> None:
        if self._engine.record:
            self._engine.control_calls.append((rpc, request))

    async def GetServerInfo(
        self, request: server_pb2.GetServerInfoRequest, context: grpc.aio.ServicerContext
    ) -> server_pb2.ServerInfo:
        self._record("GetServerInfo", request)
        return self._engine.server_info()

    async def GetModelInfo(
        self, request: model_pb2.GetModelInfoRequest, context: grpc.aio.ServicerContext
    ) -> model_pb2.ModelInfo:
        self._record("GetModelInfo", request)
        return self._engine.profile.build_model_info()

    async def GetLoad(
        self, request: server_pb2.GetLoadRequest, context: grpc.aio.ServicerContext
    ) -> server_pb2.LoadInfo:
        self._record("GetLoad", request)
        engine = self._engine
        load = server_pb2.LoadInfo(
            instance_id=engine.instance_id,
            timestamp_unix_nanos=time.time_ns(),
            running_requests=len(engine._active),
        )
        if engine.profile.follows_conventions:
            load.attributes.update(engine.counters)
        return load

    async def Health(
        self, request: lifecycle_pb2.HealthRequest, context: grpc.aio.ServicerContext
    ) -> lifecycle_pb2.HealthResponse:
        self._record("Health", request)
        engine = self._engine
        state = engine.health_state
        response = lifecycle_pb2.HealthResponse(
            state=state,
            checks=[
                lifecycle_pb2.HealthCheck(
                    name="grpc", state=lifecycle_pb2.HEALTH_STATE_READY, message="serving"
                ),
                lifecycle_pb2.HealthCheck(name="model", state=state, message=engine.profile.model),
            ],
        )
        connector = engine.profile.kv_connector
        if connector is not None and connector.enabled:
            response.checks.add(
                name="kv_connector",
                state=lifecycle_pb2.HEALTH_STATE_READY,
                message=connector.transfer_backend,
            )
        return response

    async def Abort(
        self, request: lifecycle_pb2.AbortRequest, context: grpc.aio.ServicerContext
    ) -> lifecycle_pb2.AbortResponse:
        self._record("Abort", request)
        engine = self._engine
        target = request.WhichOneof("target")
        if target == "request_id":
            aborted = engine._abort([request.request_id])
            message = request.request_id
        elif target == "all_requests":
            aborted = engine._abort(list(engine._active))
            message = f"aborted {aborted} request(s)"
        elif target == "kv_session":
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "aborting by kv_session is not supported; abort by request_id",
            )
        else:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "AbortRequest has no target set")
        return lifecycle_pb2.AbortResponse(
            status=(
                lifecycle_pb2.ABORT_STATUS_ABORTED
                if aborted
                else lifecycle_pb2.ABORT_STATUS_ALREADY_FINISHED
            ),
            message=message,
        )

    async def _require_lora(self, context: grpc.aio.ServicerContext) -> None:
        if not self._engine.profile.supports_lora:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, "LoRA is not supported")

    async def LoadLora(
        self, request: lora_pb2.LoadLoraRequest, context: grpc.aio.ServicerContext
    ) -> lora_pb2.LoadLoraResponse:
        self._record("LoadLora", request)
        await self._require_lora(context)
        adapter = request.adapter
        if not adapter.lora_name or not adapter.source_path or adapter.lora_id < 0:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "adapter.lora_name and source_path are required and lora_id must be non-negative",
            )
        loras = self._engine.loras
        name, lora_id, path = adapter.lora_name, adapter.lora_id, adapter.source_path
        loaded = loras.get(name)
        if loaded is not None:
            if (loaded.lora_id, loaded.source_path) != (lora_id, path):
                await context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    f"LoRA '{name}' is registered with id {loaded.lora_id} and path "
                    f"'{loaded.source_path}', not id {lora_id} and path '{path}'",
                )
            return lora_pb2.LoadLoraResponse(adapter=loaded, already_loaded=True)
        owner = next((other for other in loras.values() if other.lora_id == lora_id), None)
        if owner is not None:
            await context.abort(
                grpc.StatusCode.ALREADY_EXISTS,
                f"LoRA id {lora_id} is registered to '{owner.lora_name}'",
            )
        bound_path = self._engine.lora_paths.get(lora_id, path)
        if bound_path != path:
            await context.abort(
                grpc.StatusCode.ALREADY_EXISTS,
                f"LoRA id {lora_id} was used with path '{bound_path}' and the engine keeps "
                "its weights; register this adapter with a new id",
            )
        limit = self._engine.profile.max_registered_loras
        new_id = lora_id not in self._engine.lora_paths
        if new_id and limit is not None and len(self._engine.lora_paths) >= limit:
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"at most {limit} LoRA ids can be registered",
            )
        stored = lora_pb2.LoraAdapter()
        stored.CopyFrom(adapter)
        loras[name] = stored
        self._engine.lora_paths[lora_id] = path
        return lora_pb2.LoadLoraResponse(adapter=stored, already_loaded=False)

    async def UnloadLora(
        self, request: lora_pb2.UnloadLoraRequest, context: grpc.aio.ServicerContext
    ) -> lora_pb2.UnloadLoraResponse:
        self._record("UnloadLora", request)
        await self._require_lora(context)
        adapter = self._engine.loras.pop(request.lora_name, None)
        if adapter is None:
            await context.abort(
                grpc.StatusCode.NOT_FOUND, f"LoRA adapter '{request.lora_name}' is not loaded"
            )
        return lora_pb2.UnloadLoraResponse(adapter=adapter)

    async def ListLoras(
        self, request: lora_pb2.ListLorasRequest, context: grpc.aio.ServicerContext
    ) -> lora_pb2.ListLorasResponse:
        self._record("ListLoras", request)
        await self._require_lora(context)
        return lora_pb2.ListLorasResponse(adapters=list(self._engine.loras.values()))

    async def GetKvEventSources(
        self, request: kv_pb2.GetKvEventSourcesRequest, context: grpc.aio.ServicerContext
    ) -> kv_pb2.GetKvEventSourcesResponse:
        self._record("GetKvEventSources", request)
        sources = self._engine.profile.kv_event_sources
        # No sources means KV events are off: the answer is empty for any ranks.
        if not sources or not request.data_parallel_ranks:
            return kv_pb2.GetKvEventSourcesResponse(sources=sources)
        by_rank = {source.data_parallel_rank: source for source in sources}
        unknown = sorted(set(request.data_parallel_ranks) - by_rank.keys())
        if unknown:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"no KV event source for data-parallel rank(s) {unknown}",
            )
        return kv_pb2.GetKvEventSourcesResponse(
            sources=[by_rank[rank] for rank in sorted(set(request.data_parallel_ranks))]
        )

    async def SubscribeKvEvents(
        self, request: kv_pb2.SubscribeKvEventsRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[kv_pb2.SubscribeKvEventsResponse]:
        self._record("SubscribeKvEvents", request)
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "SubscribeKvEvents is not implemented")
        yield kv_pb2.SubscribeKvEventsResponse()  # pragma: no cover - makes this a generator


class _Health(health_pb2_grpc.HealthServicer):
    """Standard ``grpc.health.v1`` service: ``inference`` (and ``""``) and ``liveness``."""

    def __init__(self, engine: FakeEngine) -> None:
        self._engine = engine

    def _status(self, service: str) -> int | None:
        if service == "liveness":
            return health_pb2.HealthCheckResponse.SERVING
        if service in ("", "inference"):
            ready = self._engine.health_state == lifecycle_pb2.HEALTH_STATE_READY
            return (
                health_pb2.HealthCheckResponse.SERVING
                if ready
                else health_pb2.HealthCheckResponse.NOT_SERVING
            )
        return None

    async def Check(
        self, request: health_pb2.HealthCheckRequest, context: grpc.aio.ServicerContext
    ) -> health_pb2.HealthCheckResponse:
        status = self._status(request.service)
        if status is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "unknown service")
        return health_pb2.HealthCheckResponse(status=status)

    async def Watch(
        self, request: health_pb2.HealthCheckRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[health_pb2.HealthCheckResponse]:
        last = None
        while True:
            status = self._status(request.service)
            if status is None:
                status = health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
            if status != last:
                last = status
                yield health_pb2.HealthCheckResponse(status=status)
            await asyncio.sleep(0.05)


class FakeEngine:
    """A deterministic OpenEngine v1 server on a local port.

    Each ``Generate`` call is counted in ``generate_received``, recorded in
    :attr:`received` when ``record`` is on, and checked against the profile (:func:`admit`). An
    accepted call counts in ``generate_accepted`` and is answered by the next
    script queued with :meth:`push_script`, else by ``script``, else by the
    built-in generator:

    * Output ``i`` walks the prompt's token ids cyclically from position
      ``i``, taking the next one as each sampled token. While fewer than
      ``min_tokens`` tokens are sampled, it skips EOS and stop token ids
      (taking the smallest id that is neither if the whole prompt is skipped),
      and stop strings do not count.
    * Each step samples up to ``tokens_per_step`` tokens for every unfinished
      output in turn and sends each output's sendable tokens in one message;
      steps are ``step_delay`` seconds apart.
    * A token in ``profile.eos_token_ids`` (unless ``ignore_eos``) or in the
      request's stop token ids ends the output with ``STOP`` and a
      ``stop_match`` naming that token. A stop token goes to the output only
      with ``include_stop_in_output``; an EOS never does.
    * A stop string ends the output with ``STOP`` once the output's text
      (``detokenizer``) contains it, in a match that ends in the newest
      token. Without ``include_stop_in_output`` the tokens whose text overlaps
      the match are left out; to make that possible, a token whose text could
      begin a stop string is not sent until a later token rules that out or
      the output ends.
    * Otherwise the output ends with ``LENGTH`` after ``max_tokens`` tokens, or
      when the context window is full if ``max_tokens`` is unset.
    * Requested logprobs are ``-0.5`` for each returned token (rank 1), with
      ``top_n`` candidates of ids ``t, t+1, ...``. Requested prompt logprobs
      come in one ``prompt`` event first; the first prompt token has none.
    * ``TokenInfo.token`` is set only on tokens that carry a logprob, and
      ``TokenOutput.text`` is never set.
    * Usage is on the last response only. ``completion_tokens`` counts the
      returned tokens, so an EOS, stop token or stop string left out of the
      output is not counted; ``cached_prompt_tokens`` is the attribute of the same
      name.

    For a profile with a ``handoff_shape`` and an enabled ``kv_connector``:

    * A prefill call (``extra.request_type = "context_only"``) gets its prompt
      event if prompt logprobs were requested, then its first token, then
      after ``step_delay`` a ``prefill_ready`` event as its terminal event,
      carrying the handoff (:meth:`_handoff`) and usage counting the one
      token. With the ``"trtllm"`` shape nothing ends it earlier: the
      TensorRT-LLM server hands off a first token that is a stop token, and
      under ``max_tokens`` or ``ignore_eos``, and the fake does the same for
      EOS and stop strings. With the ``"ds"`` shape an EOS or stop token
      ends it as in an aggregated call, with ``finished`` and usage and no
      ``prefill_ready``. Each handoff sent is recorded in :attr:`handoffs`
      when ``record`` is on.
    * A decode call (with ``kv.session``) starts its output with the
      handoff's ``first_gen_tokens``, each counted as a sampled token, then
      continues as the generator would after them. Given a handoff from a
      fake engine, it returns the tokens an aggregated call with the same
      prompt and settings returns.
    * :attr:`disagg_faults` (:class:`DisaggFaults`) makes calls misbehave.

    A scripted ``prefill_ready`` event ends the call too. Each one sent, by
    script or generator, counts in ``prefill_ready_sent`` when it is handed
    to gRPC.

    ``Abort`` (or :meth:`abort`) ends the unfinished outputs of a running call
    with ``CANCELLED`` after sending the tokens they held back; a prefill call
    aborted before its ``prefill_ready`` ends that way too. ``Generate``
    is refused with ``UNAVAILABLE`` unless :attr:`health_state` is ``READY``.

    :attr:`cancel_seen` is an ``asyncio.Event`` set when a client cancels a
    running call (not by ``Abort`` or :meth:`stop`). It is never cleared, and
    belongs to the event loop that first waits on it: wait on it only from
    the engine's event loop. From another thread, read :attr:`cancelled` and
    :attr:`aborted`.

    Args:
        profile: A :class:`FakeProfile` or the name of one in :data:`PROFILES`.
        script: Steps for every accepted call, or a function from the request
            to its steps (None to use the built-in generator). A call whose
            steps run out before an error event or the last output's
            ``finished`` event, and without an abort, fails with ``INTERNAL``.
        detokenizer: Text of a token sequence, for stop strings and for the
            ``token`` strings that come with logprobs.
        health_state: The initial ``HealthState``.
        instance_id: ``ServerInfo.instance_id``; a new one per engine by default.
        record: Keep :attr:`received`, :attr:`cancelled`, :attr:`aborted` and
            :attr:`control_calls`. Turn it off for a long-running engine; the
            counters are kept either way.
    """

    def __init__(
        self,
        profile: str | FakeProfile = "trtllm-like",
        *,
        script: Sequence[Step] | ScriptSource | None = None,
        detokenizer: Callable[[Sequence[int]], str] = default_detokenizer,
        tokens_per_step: int = 1,
        step_delay: float = 0.0,
        cached_prompt_tokens: int = 0,
        health_state: int = lifecycle_pb2.HEALTH_STATE_READY,
        instance_id: str | None = None,
        record: bool = True,
    ) -> None:
        if tokens_per_step < 1:
            raise ValueError("tokens_per_step must be at least 1")
        self.profile = resolve_profile(profile)
        self.script = script if script is None or callable(script) else _checked_steps(script)
        self.detokenizer = detokenizer
        self.tokens_per_step = tokens_per_step
        self.step_delay = step_delay
        self.cached_prompt_tokens = cached_prompt_tokens
        self.health_state = health_state
        self.instance_id = instance_id or f"fake-{uuid.uuid4().hex[:12]}"
        self.record = record
        self.received: list[ReceivedRequest] = []
        self.cancelled: list[str] = []
        self.aborted: list[str] = []
        self.cancel_seen = asyncio.Event()
        self.control_calls: list[tuple[str, Any]] = []
        self.loras: dict[str, lora_pb2.LoraAdapter] = {}
        # The path each LoRA id was first loaded from. Never shrinks: like the
        # TensorRT-LLM server, the engine keeps serving an id's first weights.
        self.lora_paths: dict[int, str] = {}
        self.disagg_faults = DisaggFaults()
        self.handoffs: list[tuple[str, kv_pb2.KvSessionRef]] = []
        self._scripts: list[Sequence[Step]] = []
        self._active: dict[str, _Call] = {}
        self._counters = dict.fromkeys(
            (COUNTER_GENERATE_RECEIVED, COUNTER_GENERATE_ACCEPTED, *VERSION_4_ENGINE_COUNTERS), 0
        )
        self._server: grpc.aio.Server | None = None
        self._stopping = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self.host = "127.0.0.1"
        self.port = 0

    # -- Inspection ------------------------------------------------------------

    @property
    def requests(self) -> list[generation_pb2.GenerateRequest]:
        """Every recorded ``GenerateRequest``, accepted or not, in arrival order."""
        return [record.request for record in self.received]

    @property
    def accepted(self) -> list[ReceivedRequest]:
        return [record for record in self.received if record.admitted is not None]

    @property
    def counters(self) -> dict[str, int]:
        """The ``trtllm.counters.*`` values.

        ``GetLoad`` reports them when the profile follows the conventions;
        the version 4 ones only when the profile has a ``kv_connector``.
        """
        if self.profile.kv_connector is not None:
            return dict(self._counters)
        return {
            key: value
            for key, value in self._counters.items()
            if key not in VERSION_4_ENGINE_COUNTERS
        }

    @property
    def ctx_info_endpoint(self) -> tuple[str, int]:
        """``(host, port)`` this engine names in its handoffs as its KV endpoint.

        The profile's, else the address it serves on, with a wildcard host
        replaced by the loopback address of its family.
        """
        connector = self.profile.kv_connector
        if connector is not None and connector.ctx_info_endpoint is not None:
            return connector.ctx_info_endpoint
        host = self.host.removeprefix("[").removesuffix("]")
        host = {"": "127.0.0.1", "0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
        return host, self.port

    def server_info(self) -> server_pb2.ServerInfo:
        """What ``GetServerInfo`` returns.

        A ds context instance also reports its KV endpoint in
        ``kv_connector.local_endpoints`` unless the profile lists some.
        """
        info = self.profile.build_server_info(self.instance_id)
        connector = self.profile.kv_connector
        if (
            connector is not None
            and self.profile.handoff_shape == "ds"
            and self.profile.instance_role == "context"
            and not connector.local_endpoints
        ):
            host, port = self.ctx_info_endpoint
            info.kv_connector.local_endpoints.add(host=host, port=port, protocol="tcp")
        return info

    @property
    def running_request_ids(self) -> list[str]:
        return list(self._active)

    @property
    def target(self) -> str:
        return f"127.0.0.1:{self.port}"

    # -- Control ---------------------------------------------------------------

    def push_script(self, *steps: Step) -> None:
        """Answer the next accepted call with ``steps``."""
        self._scripts.append(_checked_steps(steps))

    def set_health(self, state: int) -> None:
        """Report ``state`` from ``Health`` and the standard health service."""
        self.health_state = state

    def abort(self, request_id: str) -> bool:
        """Abort a running call as ``Abort`` does; True if it was running.

        Safe to call from a thread other than the engine's.
        """
        if request_id not in self._active:
            return False
        if self._loop is not None and not self._on_engine_loop():
            return self.run_coroutine(self._abort_async([request_id])) > 0
        return self._abort([request_id]) > 0

    async def _abort_async(self, request_ids: Sequence[str]) -> int:
        return self._abort(request_ids)

    def _on_engine_loop(self) -> bool:
        try:
            return asyncio.get_running_loop() is self._loop
        except RuntimeError:
            return False

    def _abort(self, request_ids: Sequence[str]) -> int:
        aborted = 0
        for request_id in request_ids:
            call = self._active.get(request_id)
            if call is None or call.finished or call.aborted.is_set():
                continue
            call.aborted.set()
            if self.record:
                self.aborted.append(request_id)
            aborted += 1
        return aborted

    def _handoff(self, admitted: Admitted, generator: _Generator) -> kv_pb2.KvSessionRef:
        """The ``kv_session`` of a prefill call, in the shape of the profile's engine.

        ``session_id`` and the ``disagg_request_id`` attribute are the call's
        ``extra.disagg_request_id``, or a new id in the same range without it.
        Numbers in ``attributes_struct`` are floats, as on the wire.
        """
        profile = self.profile
        connector = profile.kv_connector
        disagg_request_id = admitted.disagg_request_id
        if disagg_request_id is None:
            disagg_request_id = DISAGG_REQUEST_ID_MIN + secrets.randbelow(
                DISAGG_REQUEST_ID_LIMIT - DISAGG_REQUEST_ID_MIN
            )
        host, port = self.ctx_info_endpoint
        url_host = f"[{host}]" if ":" in host else host
        session = kv_pb2.KvSessionRef(
            session_id=str(disagg_request_id),
            transfer_backend=connector.transfer_backend if connector is not None else "",
        )
        session.endpoints.add(host=host, port=port, protocol="tcp")
        attributes: dict[str, Any] = {
            "ctx_info_endpoint": f"tcp://{url_host}:{port}",
            "first_gen_tokens": list(generator.outputs[0].tokens),
            "disagg_request_id": str(disagg_request_id),
            "ctx_dp_rank": 0,
        }
        if profile.handoff_shape == "trtllm" and admitted.output_top_n is not None:
            # The server's verbose form: per position, [token_id, logprob, rank]
            # for each candidate.
            attributes["first_gen_log_probs"] = [
                [
                    [candidate.token_id, candidate.logprob, candidate.rank]
                    for candidate in (info.candidates or [info])
                ]
                for info in map(generator.token_info, generator.outputs[0].tokens)
            ]
        attributes.update(self.disagg_faults.extra_attributes)
        session.attributes_struct.update(attributes)
        return session

    def _prefill_ready(
        self, request_id: str, admitted: Admitted, generator: _Generator
    ) -> generation_pb2.GenerateResponse:
        session = self._handoff(admitted, generator)
        self._counters[COUNTER_PREFILL_READY_SENT] += 1
        if self.record:
            recorded = kv_pb2.KvSessionRef()
            recorded.CopyFrom(session)
            self.handoffs.append((request_id, recorded))
        return generation_pb2.GenerateResponse(
            request_id=request_id,
            prefill_ready=generation_pb2.PrefillReady(kv_session=session),
            usage=generator.usage(self.cached_prompt_tokens),
        )

    def _script_for(self, request: generation_pb2.GenerateRequest) -> list[Step] | None:
        if self._scripts:
            steps: Iterable[Step] | None = self._scripts.pop(0)
        elif self.script is None:
            return None
        else:
            steps = self.script(request) if callable(self.script) else self.script
        if steps is None:
            return None
        return _checked_steps(steps)

    # -- Lifecycle -------------------------------------------------------------

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        """Start serving on ``host:port``; return the bound port."""
        options = [
            ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
            ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ]
        self._loop = asyncio.get_running_loop()
        self.host = host
        self._stopping = False
        self._server = grpc.aio.server(options=options)
        openengine_pb2_grpc.add_InferenceServicer_to_server(_Inference(self), self._server)
        openengine_pb2_grpc.add_ControlServicer_to_server(_Control(self), self._server)
        health_pb2_grpc.add_HealthServicer_to_server(_Health(self), self._server)
        self.port = self._server.add_insecure_port(f"{host}:{port}")
        if not self.port:
            raise RuntimeError(f"could not bind {host}:{port}")
        await self._server.start()
        return self.port

    async def stop(self, grace: float | None = 0) -> None:
        """Stop serving; running calls end as a lost connection does.

        Calls ended this way are not recorded as cancelled by a client.
        """
        if self._server is not None:
            # Handlers see their cancellation after stop() returns, so the flag
            # stays set until the next start().
            self._stopping = True
            try:
                await self._server.stop(grace)
            finally:
                self._server = None
        self._loop = None

    async def __aenter__(self) -> FakeEngine:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.stop()

    def run_coroutine(self, coroutine: Any, timeout: float = 10.0) -> Any:
        """Run ``coroutine`` on the engine's event loop from another thread."""
        assert self._loop is not None, "the engine is not running"
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout)


@contextlib.contextmanager
def running_in_thread(
    engine: FakeEngine, host: str = "127.0.0.1", port: int = 0, timeout: float = 10.0
) -> Iterator[FakeEngine]:
    """Serve ``engine`` from a background thread, for synchronous tests."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="fake-openengine", daemon=True)
    thread.start()
    try:
        try:
            asyncio.run_coroutine_threadsafe(engine.start(host, port), loop).result(timeout)
            yield engine
        finally:
            asyncio.run_coroutine_threadsafe(engine.stop(), loop).result(timeout)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout)
        if not thread.is_alive():
            loop.close()


# -- Command line -------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", default="trtllm-like", choices=sorted(PROFILES))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--port-file", help="Write the bound port here once the engine serves requests."
    )
    parser.add_argument("--model", help="Model name the engine serves.")
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--eos-token-ids", type=int, nargs="+")
    parser.add_argument(
        "--profile-json",
        help="JSON object of FakeProfile field overrides, applied after the options above.",
    )
    parser.add_argument("--tokens-per-step", type=int, default=1)
    parser.add_argument("--step-delay", type=float, default=0.0)
    return parser.parse_args(argv)


def _profile_from_args(args: argparse.Namespace) -> FakeProfile:
    changes: dict[str, Any] = {}
    if args.model is not None:
        changes["model"] = args.model
    if args.max_context_length is not None:
        changes["max_context_length"] = args.max_context_length
    if args.vocab_size is not None:
        changes["vocab_size"] = args.vocab_size
    if args.eos_token_ids is not None:
        changes["eos_token_ids"] = tuple(args.eos_token_ids)
    if args.profile_json:
        overrides = json.loads(args.profile_json)
        for name in ("guided_modes", "modalities"):
            if name in overrides:
                overrides[name] = frozenset(overrides[name])
        if "eos_token_ids" in overrides:
            overrides["eos_token_ids"] = tuple(overrides["eos_token_ids"])
        if "params" in overrides:
            overrides["params"] = {
                name: AcceptedValues.from_extra(name, described)
                for name, described in overrides["params"].items()
            }
        connector = overrides.get("kv_connector")
        if connector is not None:
            if "local_endpoints" in connector:
                connector["local_endpoints"] = tuple(map(tuple, connector["local_endpoints"]))
            if connector.get("ctx_info_endpoint") is not None:
                connector["ctx_info_endpoint"] = tuple(connector["ctx_info_endpoint"])
            overrides["kv_connector"] = FakeKvConnector(**connector)
        if "kv_event_sources" in overrides:
            overrides["kv_event_sources"] = tuple(
                json_format.ParseDict(source, kv_pb2.KvEventSource())
                for source in overrides["kv_event_sources"]
            )
        changes.update(overrides)
    return PROFILES[args.profile].replace(**changes)


async def _serve(args: argparse.Namespace) -> None:
    engine = FakeEngine(
        _profile_from_args(args),
        tokens_per_step=args.tokens_per_step,
        step_delay=args.step_delay,
        record=False,
    )
    port = await engine.start(args.host, args.port)
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stopping.set)
    if args.port_file:
        partial = f"{args.port_file}.partial"
        with open(partial, "w") as handle:
            handle.write(f"{port}\n")
        os.replace(partial, args.port_file)
    print(f"fake OpenEngine '{engine.profile.name}' serving on {args.host}:{port}", flush=True)
    await stopping.wait()
    await engine.stop(grace=1.0)


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(_serve(_parse_args(argv)))


if __name__ == "__main__":
    main()

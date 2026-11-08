# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate an OpenEngine GenerateRequest into TensorRT-LLM request inputs."""

import asyncio
import importlib
import json
import math
import threading
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import grpc
from google.protobuf import struct_pb2

from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.grpc.openengine_conventions import (
    ACCEPT_ANY,
    ACCEPT_VALUES,
    ANY_VALUE,
    DEFAULT_FROM_GENERATION_CONFIG,
    NO_VALUE,
    PARAMS_KEY,
    SAMPLING_FIELD_CAPABILITY,
    SAMPLING_PARAM_NAMES,
    AcceptedValues,
    decode_priority,
)
from tensorrt_llm.grpc.openengine_conventions import LOGIT_BIAS_MAX as _LOGIT_BIAS_MAX
from tensorrt_llm.grpc.openengine_conventions import (
    LOGIT_BIAS_MAX_ENTRIES as _LOGIT_BIAS_MAX_ENTRIES,
)
from tensorrt_llm.grpc.openengine_conventions import LOGIT_BIAS_MIN as _LOGIT_BIAS_MIN
from tensorrt_llm.grpc.openengine_conventions import META_PRIORITY as _PRIORITY_METADATA_KEY
from tensorrt_llm.grpc.openengine_conventions import META_TARGET_DP_RANK as _TARGET_DP_RANK_KEY
from tensorrt_llm.grpc.openengine_conventions import REQ_KV_HINTS as _KV_HINTS_EXTRA_KEY
from tensorrt_llm.grpc.openengine_conventions import REQ_LOGIT_BIAS as _LOGIT_BIAS_EXTRA_KEY
from tensorrt_llm.grpc.openengine_conventions import REQ_SCHEDULING as _SCHEDULING_EXTRA_KEY
from tensorrt_llm.grpc.openengine_conventions import (
    REQ_THINKING_TOKEN_BUDGET as _THINKING_TOKEN_BUDGET_EXTRA_KEY,
)
from tensorrt_llm.inputs.media_io import (
    AudioMediaIO,
    ImageMediaIO,
    MediaFetchError,
    VideoMediaIO,
    _decoded_video_bytes,
)
from tensorrt_llm.inputs.multimodal_data import AudioData
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor
from tensorrt_llm.sampling_params import (
    _GENERATION_CONFIG_SAMPLING_FIELDS,
    GuidedDecodingParams,
    SamplingParams,
)

from .bindings import generation_pb2
from .capabilities import (
    MAX_NUM_SEQUENCES,
    REQUEST_KV_HINTS_KEY,
    REQUEST_LOGIT_BIAS_KEY,
    REQUEST_SCHEDULING_KEY,
    REQUEST_THINKING_TOKEN_BUDGET_KEY,
    MediaCapabilities,
    resolve_media_capabilities,
    supported_guides,
)
from .errors import MediaUnavailableError, UnsupportedFeatureError

if TYPE_CHECKING:
    from tensorrt_llm.kv_hints.protocol import KvHintsEnvelope

# trtllm.kv_hints JSON text is parsed only up to this multiple of the admitted
# envelope size limit.
_KV_HINTS_TEXT_BYTES_PER_ENVELOPE_BYTE = 4
# Cap on the number of stop conditions. Each one costs a KMP automaton per
# output index and a scan of every generated token, so an unbounded list lets a
# single request tax the event loop for the whole stream.
_MAX_STOP_CONDITIONS = 64
# Bound per-request decode work and retained decoded media.
_MAX_MEDIA_ITEMS = 64
_MAX_CONCURRENT_MEDIA_LOADS = 8
# Images are charged after decoding. Video decoders reserve retained bytes and
# counted peak buffers separately before allocating frames.
# Encoded inputs have separate gRPC and remote-loader caps.
_MAX_TOTAL_DECODED_MEDIA_BYTES = 512 * 1024 * 1024
_CONVERSATION_ID_EXTRA_KEY = "conversation_id"
_DETOKENIZE_EXTRA_KEY = "detokenize"
# Every `extra` request key of the conventions in this namespace is checked
# against the engine's capabilities.
_TRTLLM_EXTRA_PREFIX = "trtllm."
# The processor scans each sequence for its markers at every decode step.
_MAX_REASONING_MARKER_TOKENS = 32
# The AgentTree reorder policy has one level of nodes below its root; a deeper
# hierarchy fails the scheduler step for the whole batch. Node ids are C++ int.
_MAX_AGENT_HIERARCHY_DEPTH = 1
_MAX_AGENT_NODE_ID = 2**31 - 1
# Float sampling fields. Admission rejects non-finite values in all of them;
# ``SamplingParams`` validation does not catch NaN in every one.
_FLOAT_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
)


@dataclass(frozen=True)
class RequestCapabilities:
    """What ``Generate`` accepts beyond the portable request, fixed per engine.

    ``Generate`` accepts a request option only when its field here allows it;
    ``GetModelInfo`` and ``GetServerInfo`` report the request keys, priority,
    LoRA support and ``max_loras`` from this object, and ``max_num_sequences``
    from the same :func:`max_num_sequences_for`.
    """

    max_num_sequences: int
    """int: Largest ``sampling.num_sequences`` accepted."""
    supports_priority: bool
    """bool: Whether the scheduler orders waiting requests by ``openengine-priority``."""
    vocab_size: int | None
    """int | None: Width of the logits, which bounds token ids; None when unknown."""
    supports_logit_bias: bool
    """bool: Whether ``trtllm.logit_bias`` is accepted; requires ``vocab_size``."""
    supports_thinking_token_budget: bool
    """bool: Whether ``trtllm.thinking_token_budget`` is accepted."""
    supports_kv_hints: bool
    """bool: Whether ``trtllm.kv_hints`` is accepted."""
    supports_scheduling: bool
    """bool: Whether ``trtllm.scheduling`` is accepted; the executor reads both of its fields."""
    lora_ckpt_source: str | None
    """str | None: Checkpoint format of LoRA adapters; None when LoRA is not supported."""
    max_loras: int | None
    """int | None: Adapters the LoRA cache keeps on the GPU; None when unknown or unsupported."""
    sampling: Mapping[str, AcceptedValues] = field(
        default_factory=lambda: dict.fromkeys(SAMPLING_PARAM_NAMES, ANY_VALUE), hash=False
    )
    """Mapping[str, AcceptedValues]: The values accepted for each ``sampling`` field;
    a field missing here accepts none."""

    def __post_init__(self) -> None:
        if self.supports_logit_bias and self.vocab_size is None:
            raise ValueError("supports_logit_bias requires vocab_size")
        object.__setattr__(self, "sampling", MappingProxyType(dict(self.sampling)))

    @property
    def supports_lora(self) -> bool:
        return self.lora_ckpt_source is not None

    def extra_capabilities(self) -> dict[str, bool]:
        """The ``trtllm.request.*`` capability keys this object decides."""
        return {
            REQUEST_THINKING_TOKEN_BUDGET_KEY: self.supports_thinking_token_budget,
            REQUEST_LOGIT_BIAS_KEY: self.supports_logit_bias,
            REQUEST_KV_HINTS_KEY: self.supports_kv_hints,
            REQUEST_SCHEDULING_KEY: self.supports_scheduling,
        }

    def sampling_declaration(self) -> dict[str, Any]:
        """``trtllm.params`` and the version 2 sampling keys, both from :attr:`sampling`.

        A version 2 key is true exactly when its field accepts any value.
        """
        declaration: dict[str, Any] = {
            PARAMS_KEY: {name: accepted.to_extra() for name, accepted in self.sampling.items()}
        }
        for name, key in SAMPLING_FIELD_CAPABILITY.items():
            declaration[key] = self.sampling.get(name, NO_VALUE).accept == ACCEPT_ANY
        return declaration

    def check_sampling_value(self, name: str, value: float) -> None:
        """Reject a ``sampling`` value outside the declared set.

        Raises:
            UnsupportedFeatureError: ``value`` is not accepted for ``name``.
        """
        accepted = self.sampling.get(name, NO_VALUE)
        if not accepted.accepts(value):
            raise UnsupportedFeatureError(
                f"sampling.{name}={value!r} is not supported by this engine: "
                f"{PARAMS_KEY}.{name} accepts {accepted.describe()}"
            )

    def accepted_extra_keys(self) -> frozenset[str]:
        """The ``trtllm.*`` request keys ``Generate`` accepts."""
        supported = (
            (_THINKING_TOKEN_BUDGET_EXTRA_KEY, self.supports_thinking_token_budget),
            (_LOGIT_BIAS_EXTRA_KEY, self.supports_logit_bias),
            (_KV_HINTS_EXTRA_KEY, self.supports_kv_hints),
            (_SCHEDULING_EXTRA_KEY, self.supports_scheduling),
        )
        return frozenset(key for key, accepted in supported if accepted)


# Used when no engine is given: only the portable request is accepted.
PORTABLE_REQUEST_CAPABILITIES = RequestCapabilities(
    max_num_sequences=MAX_NUM_SEQUENCES,
    supports_priority=False,
    vocab_size=None,
    supports_logit_bias=False,
    supports_thinking_token_budget=False,
    supports_kv_hints=False,
    supports_scheduling=False,
    lora_ckpt_source=None,
    max_loras=None,
)


def _positive_int(value: Any) -> int | None:
    """``value`` as a positive int, else None; unset limits mean "unknown"."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def max_num_sequences_for(llm_args: Any) -> int:
    """The ``num_sequences`` limit.

    n > 1 is served by sampling, not beam search, so max_beam_width is not its
    limit. Without attention DP, the scheduler admits a request only when its n
    sequences fit in max_batch_size x pipeline-stage slots; a larger n would
    never be scheduled. With attention DP, the per-rank limit can be lower when
    max_num_tokens is small, which this value does not account for. A request
    with an attention-DP rank hint is limited to one sequence.
    """
    limit = MAX_NUM_SEQUENCES
    batch_size = _positive_int(getattr(llm_args, "max_batch_size", None))
    if batch_size is not None:
        stages = _positive_int(getattr(llm_args, "pipeline_parallel_size", None)) or 1
        limit = min(limit, batch_size * stages)
    return limit


def _priority_scheduling_enabled(llm_args: Any) -> bool:
    """Whether waiting requests are ordered by priority.

    The ``TLLM_GEN_ADMISSION`` ordering replaces the priority queue when it is
    enabled, and the executor reads it from the same environment.
    """
    from tensorrt_llm.llmapi.llm_args import WaitingQueuePolicy

    scheduler_config = getattr(llm_args, "scheduler_config", None)
    if getattr(scheduler_config, "waiting_queue_policy", None) != WaitingQueuePolicy.PRIORITY:
        return False
    from tensorrt_llm._torch.pyexecutor.scheduler.gen_admission_queue import GenAdmissionConfig

    return not GenAdmissionConfig.from_env().enabled


def _one_model_spec_config(llm_args: Any) -> Any:
    """The speculative config the executor runs when it is a one-model mode, else None."""
    spec_config = getattr(llm_args, "speculative_config", None)
    if spec_config is None:
        return None
    if spec_config.decoding_type == "AUTO":
        from tensorrt_llm._torch.speculative import suggest_spec_config

        # The executor replaces AUTO with this config when it starts.
        spec_config = suggest_spec_config(
            _positive_int(getattr(llm_args, "max_batch_size", None)) or 1
        )
    return spec_config if spec_config.spec_dec_mode.use_one_engine() else None


def uses_one_model_speculation(llm_args: Any) -> bool:
    """Whether a one-model speculative decoding mode runs.

    It rejects ``embedding_bias`` and narrows the sampling values ``Generate``
    accepts (:func:`accepted_sampling_values`).
    """
    return _one_model_spec_config(llm_args) is not None


# The parameters one-model speculation does not always apply. Under one-model
# speculation each accepts only its _SAMPLING_DEFAULTS value, which leaves sampling unchanged.
_NOT_ALWAYS_APPLIED_UNDER_SPECULATION = (
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
)
# Generate's value for a sampling field the request does not set: it fills in
# temperature, top_p and top_k (sampling_params_from_request); for the others the
# executor's own default applies. Each leaves sampling unchanged. When the LLM
# applies the model's generation config, the config's values take the place of
# these for the fields it sets (accepted_sampling_values). The seed's is not reported.
_SAMPLING_DEFAULTS: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": None,
}


def accepted_sampling_values(llm_args: Any) -> dict[str, AcceptedValues]:
    """The values ``Generate`` accepts for each ``sampling`` field.

    Every value, except under one-model speculation: its draft and acceptance
    paths apply min_p and the occurrence penalties only on some paths, chosen by
    settings such as ``advanced_sampling_mode``, ``enable_penalty``, the dynamic
    tree, relaxed acceptance and the checkpoint's MTP layer count, which the
    server process cannot all see. Only the values that leave sampling
    unchanged are then accepted for them, and for top_k and top_p when
    ``advanced_sampling_mode`` skips that filter.

    A field's default is ``generation_config`` when the LLM applies the model's
    generation config to it, else the value Generate uses.
    """
    from_generation_config = (
        getattr(llm_args, "backend", None) == "pytorch"
        and getattr(llm_args, "generation_config", None) == "auto"
    )
    spec_config = _one_model_spec_config(llm_args)
    no_effect_only: set[str] = set()
    if spec_config is not None:
        no_effect_only.update(_NOT_ALWAYS_APPLIED_UNDER_SPECULATION)
        mode = getattr(spec_config, "advanced_sampling_mode", None)
        if mode is not None and mode.skips_top_k:
            no_effect_only.add("top_k")
        if mode is not None and mode.skips_top_p:
            no_effect_only.add("top_p")
    accepted = {}
    for name in SAMPLING_PARAM_NAMES:
        default = _SAMPLING_DEFAULTS[name]
        if from_generation_config and name in _GENERATION_CONFIG_SAMPLING_FIELDS:
            default = DEFAULT_FROM_GENERATION_CONFIG
        if name in no_effect_only:
            accepted[name] = AcceptedValues(
                ACCEPT_VALUES, values=(_SAMPLING_DEFAULTS[name],), default=default
            )
        else:
            accepted[name] = AcceptedValues(ACCEPT_ANY, default=default)
    return accepted


def _agent_tree_scheduling_enabled(llm_args: Any) -> bool:
    """Whether the configuration asks for the agent-tree reorder policy.

    ``agent_hierarchy`` is read only by the ``reorder_policy_config`` policy,
    which builds its tree only from a non-empty ``agent_types`` list and is
    installed only by the C++ capacity scheduler built for a V1 KV cache
    manager, so the configuration must select V1 explicitly
    (``use_kv_cache_manager_v2="auto"`` resolves per model). A model that
    requires the V2 manager gets it anyway, which the configuration does not
    show.
    """
    reorder_policy_config = getattr(llm_args, "reorder_policy_config", None)
    if reorder_policy_config is None:
        return False
    if not getattr(getattr(reorder_policy_config, "policy_args", None), "agent_types", None):
        return False
    if getattr(getattr(llm_args, "scheduler_config", None), "use_python_scheduler", False):
        return False
    kv_cache_config = getattr(llm_args, "kv_cache_config", None)
    return getattr(kv_cache_config, "use_kv_cache_manager_v2", True) is False


def _vocab_size(llm: Any) -> int | None:
    """The logits width from the model config; None when the config does not give it.

    The tokenizer's ``vocab_size`` is not a substitute: it leaves out added
    tokens, and a bias vector narrower than the logits fails in the sampler.
    """
    config = getattr(llm, "_hf_model_config", None)
    for candidate in (config, getattr(config, "text_config", None)):
        vocab_size = _positive_int(getattr(candidate, "vocab_size", None))
        if vocab_size is not None:
            return vocab_size
    return None


def request_capabilities_for_model(llm: Any) -> RequestCapabilities:
    """Resolve what ``Generate`` accepts for the loaded engine."""
    args = getattr(llm, "args", None)
    lora_config = getattr(args, "lora_config", None)
    # The LoRA registry lives in one frontend process, while every frontend
    # feeds the same executor, whose LoRA cache is keyed by adapter id: two
    # frontends could bind one id to different adapters.
    lora_enabled = (
        lora_config is not None
        and (_positive_int(getattr(args, "num_serve_frontends", None)) or 1) == 1
    )
    vocab_size = _vocab_size(llm)
    return RequestCapabilities(
        max_num_sequences=max_num_sequences_for(args),
        supports_priority=_priority_scheduling_enabled(args),
        vocab_size=vocab_size,
        supports_logit_bias=vocab_size is not None and not uses_one_model_speculation(args),
        # Declared only without speculative decoding: the per-request logits
        # processor runs after the forward pass, one-model speculation samples
        # inside it, DSpark verification fails the whole batch on it, and the
        # other modes are unverified. Its token ids are checked against the
        # vocabulary, so that must be known.
        supports_thinking_token_budget=(
            vocab_size is not None and getattr(args, "speculative_config", None) is None
        ),
        supports_kv_hints=True,
        # One capability covers both fields, so it is declared only where the
        # executor reads both: attention DP reads attention_dp_relax and the
        # agent-tree reorder policy reads agent_hierarchy.
        supports_scheduling=(
            bool(getattr(args, "enable_attention_dp", False))
            and _agent_tree_scheduling_enabled(args)
        ),
        lora_ckpt_source=(
            str(getattr(lora_config, "lora_ckpt_source", "hf")) if lora_enabled else None
        ),
        max_loras=_positive_int(lora_config.max_loras) if lora_enabled else None,
        sampling=accepted_sampling_values(args),
    )


@dataclass(frozen=True)
class ExtraOptions:
    """Request options carried in ``GenerateRequest.extra`` outside ``SamplingParams``."""

    kv_hints: "KvHintsEnvelope | None" = None
    attention_dp_relax: bool | None = None
    agent_hierarchy: list[tuple[str, int]] | None = None


_MediaLoader = Callable[[generation_pb2.MediaItem, int], Awaitable[Any]]
_MediaValidator = Callable[[generation_pb2.MediaItem, int], None]


class _DecodedMediaReservation:
    def __init__(self, budget: "_DecodedMediaBudget", peak_bytes: int, retained_bytes: int) -> None:
        self._budget = budget
        self._peak_bytes = peak_bytes
        self._retained_bytes = retained_bytes
        self._active = True

    def check_cancelled(self) -> None:
        with self._budget._condition:
            if self._budget._cancelled:
                raise ValueError("decoded media loading was cancelled")

    def shrink(self, retained_bytes: int) -> None:
        if not self._active or not 0 <= retained_bytes <= self._retained_bytes:
            raise ValueError("invalid decoded-media reservation shrink")
        self._budget._shrink(self._peak_bytes, self._retained_bytes, retained_bytes)
        self._peak_bytes = self._retained_bytes = retained_bytes

    def commit(self, retained_bytes: int) -> None:
        if not self._active:
            raise RuntimeError("decoded-media reservation is no longer active")
        try:
            self._budget._settle(self._peak_bytes, self._retained_bytes, retained_bytes)
        finally:
            self._active = False

    def release(self) -> None:
        if self._active:
            self._budget._settle(self._peak_bytes, self._retained_bytes, 0)
            self._active = False


class _DecodedMediaBudget:
    """Thread-safe shared budget for all decoded media in one request."""

    def __init__(self, limit: int, peak_limit: int | None = None) -> None:
        self._limit = limit
        # Conversion buffers have a separate allowance from retained output.
        self._peak_limit = peak_limit if peak_limit is not None else 2 * limit
        self._retained = 0
        self._reserved_retained = 0
        self._used = 0
        self._condition = threading.Condition()
        self._cancelled = False

    def _limit_error(self) -> ValueError:
        return ValueError(f"decoded media exceeds the per-request limit of {self._limit} bytes")

    def charge(self, retained_bytes: int) -> None:
        if retained_bytes < 0:
            raise ValueError("decoded media size cannot be negative")
        with self._condition:
            if self._retained + self._reserved_retained + retained_bytes > self._limit:
                raise self._limit_error()
            self._retained += retained_bytes
            self._used += retained_bytes

    def reserve(self, peak_bytes: int, retained_bytes: int) -> _DecodedMediaReservation:
        if not 0 <= retained_bytes <= peak_bytes:
            raise ValueError("decoded media sizes must satisfy 0 <= retained <= peak")
        with self._condition:
            while True:
                if self._cancelled:
                    raise ValueError("decoded media loading was cancelled")
                if self._retained + self._reserved_retained + retained_bytes > self._limit:
                    raise self._limit_error()
                if self._retained + peak_bytes > self._peak_limit:
                    raise ValueError(
                        f"decoded media exceeds the peak buffer limit of {self._peak_limit} bytes"
                    )
                if self._used + peak_bytes <= self._peak_limit:
                    break
                self._condition.wait()
            self._used += peak_bytes
            self._reserved_retained += retained_bytes
        return _DecodedMediaReservation(self, peak_bytes, retained_bytes)

    def _shrink(self, peak_bytes: int, reserved_retained: int, retained_bytes: int) -> None:
        with self._condition:
            self._used -= peak_bytes - retained_bytes
            self._reserved_retained -= reserved_retained - retained_bytes
            self._condition.notify_all()

    def _settle(self, peak_bytes: int, reserved_retained: int, retained_bytes: int) -> None:
        with self._condition:
            self._used -= peak_bytes
            self._reserved_retained -= reserved_retained
            try:
                if retained_bytes < 0:
                    raise ValueError("decoded media size cannot be negative")
                if self._retained + self._reserved_retained + retained_bytes > self._limit:
                    raise self._limit_error()
                if self._used + retained_bytes > self._peak_limit:
                    raise self._limit_error()
                self._used += retained_bytes
                self._retained += retained_bytes
            finally:
                self._condition.notify_all()

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()


def _always_available() -> bool:
    return True


@dataclass(frozen=True)
class _MediaHandler:
    name: str
    validate: _MediaValidator
    loader_factory: Callable[[Any, _DecodedMediaBudget], _MediaLoader]
    decoded_bytes: Callable[[Any], int]
    runtime_available: Callable[[], bool] = _always_available
    reserves_decoded_bytes: bool = False


@dataclass(frozen=True)
class _PreparedInput:
    inputs: str | dict[str, list[int]]
    media: tuple[tuple[generation_pb2.MediaItem, _MediaHandler], ...]
    capabilities: MediaCapabilities | None


def conversation_affinity_enabled(llm: Any) -> bool:
    """Whether TensorRT-LLM owns attention-DP placement for conversations."""
    args = getattr(llm, "args", None)
    config = getattr(args, "attention_dp_config", None)
    return getattr(args, "enable_attention_dp", False) is True and (
        getattr(config, "kv_cache_routing_conversation_affinity", False) is True
    )


def token_ids_input_supported(llm: Any) -> bool:
    """Whether the loaded input processor consumes pre-tokenized prompts."""
    input_processor = getattr(llm, "input_processor", None)
    return getattr(input_processor, "supports_token_ids_input", True) is not False


def _top_n_candidates(selection: Any, name: str) -> int:
    kind = selection.WhichOneof("selection")
    if kind is None:
        return 0
    if kind == "top_n":
        return selection.top_n
    raise UnsupportedFeatureError(
        f"{name} candidate selection '{kind}' is not supported; use top_n"
    )


def _guided_decoding_from_request(
    request: generation_pb2.GenerateRequest,
    guided_backend: str | None = None,
) -> GuidedDecodingParams | None:
    if not request.HasField("guided"):
        return None

    guided = request.guided
    if guided.backend:
        raise UnsupportedFeatureError(
            "per-request guided decoding backend selection is not supported"
        )

    guide = guided.WhichOneof("guide")
    if guide is None:
        return None
    # Reject here rather than letting the grammar compiler fail it in-band, so
    # the client gets an actionable status.
    if guide not in supported_guides(guided_backend):
        raise UnsupportedFeatureError(
            f"guided decoding mode '{guide}' is not supported by the "
            f"'{guided_backend or 'configured'}' grammar backend"
        )
    if guide == "json_schema":
        return GuidedDecodingParams(json=guided.json_schema)
    if guide == "regex":
        return GuidedDecodingParams(regex=guided.regex)
    if guide == "ebnf_grammar":
        return GuidedDecodingParams(grammar=guided.ebnf_grammar)
    if guide == "structural_tag":
        return GuidedDecodingParams(structural_tag=guided.structural_tag)
    if guide == "json_object":
        return GuidedDecodingParams(json_object=True)
    raise UnsupportedFeatureError(f"guided decoding mode '{guide}' is not supported")


def _extra_value(request: generation_pb2.GenerateRequest, key: str) -> struct_pb2.Value | None:
    if not request.HasField("extra"):
        return None
    return request.extra.fields.get(key)


def _check_extra_keys(
    request: generation_pb2.GenerateRequest, capabilities: RequestCapabilities
) -> None:
    """Reject ``trtllm.*`` request keys this engine does not declare."""
    if not request.HasField("extra"):
        return
    accepted = capabilities.accepted_extra_keys()
    for key in request.extra.fields:
        if key.startswith(_TRTLLM_EXTRA_PREFIX) and key not in accepted:
            raise UnsupportedFeatureError(f"extra key '{key}' is not supported by this engine")


def _struct_fields(value: struct_pb2.Value, name: str, keys: Iterable[str]) -> dict:
    """The fields of an object value, which may only use ``keys``."""
    if value.WhichOneof("kind") != "struct_value":
        raise ValueError(f"{name} must be an object")
    fields = dict(value.struct_value.fields)
    unknown = sorted(set(fields) - set(keys))
    if unknown:
        raise ValueError(f"{name} has unknown fields {unknown}")
    return fields


def _integer(value: struct_pb2.Value, name: str, minimum: int = 0) -> int:
    """An integral number of at least ``minimum``; ``Struct`` numbers are doubles."""
    if value.WhichOneof("kind") != "number_value":
        raise ValueError(f"{name} must be a number")
    number = value.number_value
    if not math.isfinite(number) or not number.is_integer() or abs(number) > 2**53:
        raise ValueError(f"{name} must be an integer, got {number}")
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {int(number)}")
    return int(number)


def _token_id(value: struct_pb2.Value, name: str, vocab_size: int | None) -> int:
    token_id = _integer(value, name)
    if vocab_size is not None and token_id >= vocab_size:
        raise ValueError(f"{name} {token_id} is outside the vocabulary [0, {vocab_size})")
    return token_id


def _token_ids(value: struct_pb2.Value | None, name: str, vocab_size: int | None) -> list[int]:
    if value is None or value.WhichOneof("kind") != "list_value":
        raise ValueError(f"{name} must be a list of token ids")
    ids = [
        _token_id(item, f"{name}[{index}]", vocab_size)
        for index, item in enumerate(value.list_value.values)
    ]
    if not ids:
        raise ValueError(f"{name} must not be empty")
    if len(ids) > _MAX_REASONING_MARKER_TOKENS:
        raise ValueError(
            f"{name} has {len(ids)} ids; at most {_MAX_REASONING_MARKER_TOKENS} are supported"
        )
    return ids


def _thinking_budget_processor(value: struct_pb2.Value, vocab_size: int | None) -> Any:
    """Build the logits processor ``trtllm-serve`` would attach for a thinking budget."""
    from tensorrt_llm.llmapi.thinking_budget import ThinkingBudgetLogitsProcessor

    name = _THINKING_TOKEN_BUDGET_EXTRA_KEY
    fields = _struct_fields(value, name, ("budget", "start_token_ids", "end_token_ids"))
    if "budget" not in fields:
        raise ValueError(f"{name}.budget must be set")
    return ThinkingBudgetLogitsProcessor(
        thinking_token_budget=_integer(fields["budget"], f"{name}.budget"),
        reasoning_start_token_ids=_token_ids(
            fields.get("start_token_ids"), f"{name}.start_token_ids", vocab_size
        ),
        reasoning_end_token_ids=_token_ids(
            fields.get("end_token_ids"), f"{name}.end_token_ids", vocab_size
        ),
    )


def _embedding_bias(value: struct_pb2.Value, vocab_size: int) -> Any:
    """Materialize ``trtllm.logit_bias`` into an ``embedding_bias`` vector."""
    import torch

    name = _LOGIT_BIAS_EXTRA_KEY
    if value.WhichOneof("kind") != "struct_value":
        raise ValueError(f"{name} must be an object")
    entries = value.struct_value.fields
    if len(entries) > _LOGIT_BIAS_MAX_ENTRIES:
        raise ValueError(
            f"{name} has {len(entries)} entries; at most {_LOGIT_BIAS_MAX_ENTRIES} are supported"
        )
    if not entries:
        return None
    bias = torch.zeros(vocab_size, dtype=torch.float32)
    for key, entry in entries.items():
        # Canonical form only, so two keys never name the same token.
        if not (key.isascii() and key.isdecimal()) or str(int(key)) != key:
            raise ValueError(f"{name} key '{key}' must be a decimal token id without leading zeros")
        token_id = int(key)
        if token_id >= vocab_size:
            raise ValueError(
                f"{name} token id {token_id} is outside the vocabulary [0, {vocab_size})"
            )
        if entry.WhichOneof("kind") != "number_value":
            raise ValueError(f"{name}['{key}'] must be a number")
        number = entry.number_value
        if not math.isfinite(number) or not _LOGIT_BIAS_MIN <= number <= _LOGIT_BIAS_MAX:
            raise ValueError(
                f"{name}['{key}'] must be in [{_LOGIT_BIAS_MIN:g}, {_LOGIT_BIAS_MAX:g}], got {number}"
            )
        bias[token_id] = number
    return bias


def _plain(value: struct_pb2.Value) -> Any:
    """A ``Struct`` value as JSON data, with integral numbers as ints (EX-GEN-3).

    A finite double of magnitude 2**53 or more may be a rounded integer, so it
    is rejected rather than passed on with a value the client did not send.
    """
    kind = value.WhichOneof("kind")
    if kind == "struct_value":
        return {key: _plain(item) for key, item in value.struct_value.fields.items()}
    if kind == "list_value":
        return [_plain(item) for item in value.list_value.values]
    if kind == "number_value":
        number = value.number_value
        if math.isfinite(number) and abs(number) >= 2**53:
            raise ValueError(
                f"{_KV_HINTS_EXTRA_KEY} as an object holds a number of magnitude 2**53 or "
                "more, which a double cannot carry exactly; send the envelope's JSON text"
            )
        return int(number) if number.is_integer() else number
    if kind == "string_value":
        return value.string_value
    if kind == "bool_value":
        return value.bool_value
    return None


class _JsonConstantError(ValueError):
    pass


def _reject_json_constant(name: str) -> Any:
    raise _JsonConstantError(f"{_KV_HINTS_EXTRA_KEY} must not contain {name}")


def _kv_hints_from_json(text: str) -> Any:
    from tensorrt_llm.kv_hints.protocol import MAX_KV_HINTS_BYTES

    # Bounds the parse only; admission measures the compact encoding against
    # MAX_KV_HINTS_BYTES, so insignificant whitespace in the text is allowed.
    limit = _KV_HINTS_TEXT_BYTES_PER_ENVELOPE_BYTE * MAX_KV_HINTS_BYTES
    if len(text) > limit or len(text.encode("utf-8")) > limit:
        raise ValueError(f"{_KV_HINTS_EXTRA_KEY} is longer than {limit} bytes")
    try:
        hints = json.loads(text, parse_constant=_reject_json_constant)
    except RecursionError:
        raise ValueError(f"{_KV_HINTS_EXTRA_KEY} is nested too deeply") from None
    except _JsonConstantError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"{_KV_HINTS_EXTRA_KEY} is not valid JSON: {error}") from None
    except ValueError:
        # For example an integer with more digits than the interpreter converts.
        raise ValueError(f"{_KV_HINTS_EXTRA_KEY} cannot be parsed") from None
    if not isinstance(hints, dict):
        raise ValueError(f"{_KV_HINTS_EXTRA_KEY} must be the JSON text of an object")
    return hints


def _kv_hints(value: struct_pb2.Value) -> "KvHintsEnvelope":
    """Admit ``trtllm.kv_hints`` the way ``generate_async`` admits hints.

    The value is the envelope's JSON text, parsed like an HTTP request body, so
    every number keeps its JSON type and value (EX-REQ-5). An object is still
    accepted, but a ``Struct`` holds every number as a double: integral values
    become ints, so ``2.0`` arrives as ``2``, and a number of magnitude 2**53
    or more is rejected.

    Hints that admission would drop are rejected instead, so a client never
    believes a hint took effect when it did not.
    """
    from tensorrt_llm.kv_hints.protocol import admit_kv_hints

    kind = value.WhichOneof("kind")
    if kind == "string_value":
        hints = _kv_hints_from_json(value.string_value)
    elif kind == "struct_value":
        hints = _plain(value)
    else:
        raise ValueError(f"{_KV_HINTS_EXTRA_KEY} must be the envelope's JSON text or an object")
    admitted = admit_kv_hints(hints)
    if admitted is None:
        raise ValueError(
            f"{_KV_HINTS_EXTRA_KEY} was not admitted: unsupported protocol_version, "
            "oversized or malformed"
        )
    return admitted


def _scheduling(value: struct_pb2.Value) -> tuple[bool | None, list[tuple[str, int]] | None]:
    name = _SCHEDULING_EXTRA_KEY
    fields = _struct_fields(value, name, ("attention_dp_relax", "agent_hierarchy"))
    relax = None
    if "attention_dp_relax" in fields:
        if fields["attention_dp_relax"].WhichOneof("kind") != "bool_value":
            raise ValueError(f"{name}.attention_dp_relax must be a boolean")
        relax = fields["attention_dp_relax"].bool_value
    hierarchy = None
    if "agent_hierarchy" in fields:
        entries = fields["agent_hierarchy"]
        if entries.WhichOneof("kind") != "list_value":
            raise ValueError(f"{name}.agent_hierarchy must be a list")
        if len(entries.list_value.values) > _MAX_AGENT_HIERARCHY_DEPTH:
            raise ValueError(
                f"{name}.agent_hierarchy has {len(entries.list_value.values)} levels; "
                f"the agent-tree reorder policy supports {_MAX_AGENT_HIERARCHY_DEPTH}"
            )
        hierarchy = []
        for index, entry in enumerate(entries.list_value.values):
            item = f"{name}.agent_hierarchy[{index}]"
            pair = entry.list_value.values if entry.WhichOneof("kind") == "list_value" else ()
            if len(pair) != 2 or pair[0].WhichOneof("kind") != "string_value":
                raise ValueError(f"{item} must be a [string, integer] pair")
            node_id = _integer(pair[1], f"{item}[1]")
            if node_id > _MAX_AGENT_NODE_ID:
                raise ValueError(f"{item}[1] must be at most {_MAX_AGENT_NODE_ID}, got {node_id}")
            hierarchy.append((pair[0].string_value, node_id))
    return relax, hierarchy or None


def extra_options_from_request(
    request: generation_pb2.GenerateRequest,
    capabilities: RequestCapabilities = PORTABLE_REQUEST_CAPABILITIES,
) -> ExtraOptions:
    """Translate the ``extra`` keys that are not sampling parameters."""
    _check_extra_keys(request, capabilities)
    kv_hints = None
    if (value := _extra_value(request, _KV_HINTS_EXTRA_KEY)) is not None:
        kv_hints = _kv_hints(value)
    relax = hierarchy = None
    if (value := _extra_value(request, _SCHEDULING_EXTRA_KEY)) is not None:
        relax, hierarchy = _scheduling(value)
    return ExtraOptions(kv_hints=kv_hints, attention_dp_relax=relax, agent_hierarchy=hierarchy)


def sampling_params_from_request(
    request: generation_pb2.GenerateRequest,
    guided_backend: str | None = None,
    capabilities: RequestCapabilities = PORTABLE_REQUEST_CAPABILITIES,
) -> SamplingParams:
    """Translate OpenEngine generation options to TensorRT-LLM.

    ``capabilities`` decides which ``trtllm.*`` request keys and ``sampling``
    values are accepted, and the ``num_sequences`` limit.
    """
    _check_extra_keys(request, capabilities)
    kwargs: dict[str, Any] = {}
    num_sequences = 1
    provided_fields: set[str] = set()

    if request.HasField("sampling"):
        sampling = request.sampling
        for name in SAMPLING_PARAM_NAMES:
            if sampling.HasField(name):
                value = getattr(sampling, name)
                if name in _FLOAT_SAMPLING_FIELDS and not math.isfinite(value):
                    raise ValueError(f"sampling.{name} must be finite, got {value}")
                capabilities.check_sampling_value(name, value)
                kwargs[name] = value
                provided_fields.add(name)
        if sampling.HasField("num_sequences"):
            if sampling.num_sequences == 0:
                raise ValueError("sampling.num_sequences must be greater than zero")
            if sampling.num_sequences > capabilities.max_num_sequences:
                raise ValueError(
                    f"sampling.num_sequences must be at most {capabilities.max_num_sequences}, "
                    f"got {sampling.num_sequences}"
                )
            num_sequences = sampling.num_sequences

    if request.HasField("stopping"):
        stopping = request.stopping
        if stopping.HasField("max_tokens"):
            kwargs["max_tokens"] = stopping.max_tokens
        if stopping.HasField("min_tokens"):
            kwargs["min_tokens"] = stopping.min_tokens

        if len(stopping.conditions) > _MAX_STOP_CONDITIONS:
            raise ValueError(
                f"at most {_MAX_STOP_CONDITIONS} stopping conditions are supported, "
                f"got {len(stopping.conditions)}"
            )
        stop_texts = []
        stop_token_ids = []
        for condition in stopping.conditions:
            kind = condition.WhichOneof("condition")
            if kind == "stop_text":
                if not condition.stop_text:
                    raise ValueError("stopping condition stop_text must be non-empty")
                stop_texts.append(condition.stop_text)
            elif kind == "stop_token_id":
                stop_token_ids.append(condition.stop_token_id)
            else:
                raise ValueError("each stopping condition must set exactly one condition")
        if stop_texts:
            kwargs["stop"] = stop_texts
        if stop_token_ids:
            kwargs["stop_token_ids"] = stop_token_ids
        if stopping.HasField("ignore_eos"):
            kwargs["ignore_eos"] = stopping.ignore_eos
        if stopping.HasField("include_stop_in_output"):
            kwargs["include_stop_str_in_output"] = stopping.include_stop_in_output

    if request.HasField("response"):
        response = request.response
        if response.HasField("prompt_logprob_start") and response.prompt_logprob_start != 0:
            raise UnsupportedFeatureError("response.prompt_logprob_start is not supported")
        if response.HasField("return_prompt_logprobs") and response.return_prompt_logprobs:
            kwargs["prompt_logprobs"] = _top_n_candidates(response.prompt_candidates, "prompt")
        if response.HasField("return_output_logprobs") and response.return_output_logprobs:
            kwargs["logprobs"] = _top_n_candidates(response.output_candidates, "output")

    if (
        request.HasField("extra")
        and (detokenize := request.extra.fields.get(_DETOKENIZE_EXTRA_KEY)) is not None
    ):
        if detokenize.WhichOneof("kind") != "bool_value":
            raise ValueError(f"extra.{_DETOKENIZE_EXTRA_KEY} must be a boolean")
        kwargs["detokenize"] = detokenize.bool_value

    guided_decoding = _guided_decoding_from_request(request, guided_backend)
    if guided_decoding is not None:
        kwargs["guided_decoding"] = guided_decoding

    if (value := _extra_value(request, _THINKING_TOKEN_BUDGET_EXTRA_KEY)) is not None:
        kwargs["logits_processor"] = _thinking_budget_processor(value, capabilities.vocab_size)
    if (value := _extra_value(request, _LOGIT_BIAS_EXTRA_KEY)) is not None:
        kwargs["embedding_bias"] = _embedding_bias(value, capabilities.vocab_size)

    # Normalize unspecified sampling knobs to the same protocol defaults the
    # trtllm-serve OpenAI path (/v1/completions) applies, so a sampling request
    # produces identical results across the HTTP and OpenEngine transports. These
    # are no-ops for greedy decoding (argmax).
    for name in ("temperature", "top_p", "top_k"):
        kwargs.setdefault(name, _SAMPLING_DEFAULTS[name])
    # SamplingParams.max_tokens defaults to 32, but the protocol default for an
    # omitted stopping.max_tokens is "unbounded", as on /v1/completions (whose
    # max_tokens is Optional[int] = None). None is the engine's documented unset
    # sentinel: _deduce_max_tokens derives the budget from max_seq_len.
    kwargs.setdefault("max_tokens", None)

    # n/best_of must be constructor arguments: SamplingParams._validate() runs
    # only from __post_init__, so assigning them afterwards skips it. That guard
    # is what rejects greedy decoding with multiple returns -- without it a
    # temperature=0 request with num_sequences>1 is accepted here and returns
    # duplicate sequences as distinct samples, while the same request over
    # /v1/completions is a 400.
    if num_sequences > 1:
        kwargs["n"] = num_sequences
        kwargs["best_of"] = num_sequences

    sampling_params = SamplingParams(**kwargs)
    # Record which sampling fields the client actually supplied (mirrors the HTTP
    # `_record_sampling_params_request_fields`). Without this, the materialized
    # defaults above would look client-provided and suppress a model's
    # generation_config.json sampling defaults under generation_config="auto".
    sampling_params._set_request_provided_fields(provided_fields)
    return sampling_params


def conversation_params_from_request(
    request: generation_pb2.GenerateRequest,
) -> ConversationParams | None:
    """Translate the caller's stable conversation identity, when supplied."""
    if not request.HasField("extra"):
        return None
    conversation_id = request.extra.fields.get(_CONVERSATION_ID_EXTRA_KEY)
    if conversation_id is None:
        return None
    if conversation_id.WhichOneof("kind") != "string_value":
        raise ValueError(f"extra.{_CONVERSATION_ID_EXTRA_KEY} must be a string")
    return ConversationParams(conversation_id=conversation_id.string_value)


@dataclass(frozen=True)
class RequestMetadata:
    """What ``Generate`` reads from the gRPC invocation metadata."""

    trace_headers: Mapping[str, str] | None
    target_dp_rank: int | None
    affinity_headers: dict[str, str]
    priority: float | None


def _priority(value: str) -> float:
    """Decode ``openengine-priority`` (OE-META-3): an integer in [0, 1000] scaled to [0, 1]."""
    # int() would also take signs, underscores, spaces and non-ASCII digits.
    if not (value.isascii() and value.isdigit()):
        raise ValueError(f"{_PRIORITY_METADATA_KEY} must be a decimal integer, got '{value}'")
    return decode_priority(int(value))


def _metadata_from_context(
    context: grpc.aio.ServicerContext,
    capabilities: RequestCapabilities = PORTABLE_REQUEST_CAPABILITIES,
) -> RequestMetadata:
    """Extract tracing, priority and strict attention-DP routing metadata."""
    headers: dict[str, str] = {}
    affinity_headers: dict[str, str] = {}
    openengine_keys: set[str] = set()
    target_dp_rank = None
    priority = None
    for item in context.invocation_metadata():
        key = item.key
        value = item.value
        if key.startswith("openengine-"):
            if key in openengine_keys:
                raise ValueError(f"gRPC metadata key '{key}' must not be repeated")
            openengine_keys.add(key)
            if key == "openengine-routing-key":
                raise UnsupportedFeatureError("openengine-routing-key is not supported")
            elif key == _PRIORITY_METADATA_KEY:
                if not capabilities.supports_priority:
                    raise UnsupportedFeatureError(
                        f"gRPC metadata key '{key}' is not supported: the scheduler does not "
                        "order requests by priority"
                    )
                priority = _priority(value)
            elif key == _TARGET_DP_RANK_KEY:
                if not (value.isascii() and value.isdigit()):
                    raise ValueError(
                        f"{_TARGET_DP_RANK_KEY} must be a non-negative decimal integer, "
                        f"got '{value}'"
                    )
                target_dp_rank = int(value)
            else:
                raise UnsupportedFeatureError(f"gRPC metadata key '{key}' is not supported")
        elif key in ("traceparent", "tracestate"):
            headers[key] = value
        elif key in ("x-trtllm-subagent-affinity-id", "x-trtllm-subagent-affinity-auth"):
            if key in affinity_headers:
                raise ValueError(f"gRPC metadata key '{key}' must not be repeated")
            affinity_headers[key] = value
    return RequestMetadata(headers or None, target_dp_rank, affinity_headers, priority)


def _base_input_from_request(
    request: generation_pb2.GenerateRequest, llm: Any
) -> str | dict[str, list[int]]:
    input_kind = request.WhichOneof("input")
    if input_kind == "prompt":
        return request.prompt
    if input_kind == "token_ids":
        if not token_ids_input_supported(llm):
            raise UnsupportedFeatureError(
                "token-ID input is not supported by the loaded model; send a text prompt"
            )
        return {"prompt_token_ids": request.token_ids.ids[:]}
    raise ValueError("exactly one of prompt or token_ids must be set")


def _media_io_for_model(
    llm: Any,
    modality: str,
    media_io_type: type[Any],
    runtime_kwargs: dict[str, Any] | None = None,
) -> Any:
    input_processor = getattr(llm, "input_processor", None)
    defaults = None
    if isinstance(input_processor, BaseMultimodalInputProcessor):
        preferences = input_processor.get_preferred_media_io_kwargs() or {}
        defaults = preferences.get(modality)
    return media_io_type.create(defaults, runtime_kwargs)


def _validate_media_item(item: generation_pb2.MediaItem, index: int, modality: str) -> None:
    source = item.WhichOneof("source")
    if source is None:
        raise ValueError(f"media[{index}] must set exactly one source")
    if source == "url":
        try:
            parsed = urlparse(item.url)
            hostname = parsed.hostname
            _ = parsed.port  # Access validates the port syntax and range.
        except ValueError as error:
            raise ValueError(f"media[{index}].url is malformed: {error}") from error
        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError(f"media[{index}].url must use http or https")
        if not hostname:
            raise ValueError(f"media[{index}].url must include a hostname")
    elif source == "data_uri":
        parsed = urlparse(item.data_uri)
        if parsed.scheme.lower() != "data":
            raise ValueError(f"media[{index}].data_uri must use the data scheme")
        data_spec, separator, payload = parsed.path.partition(",")
        if not separator or not payload:
            raise ValueError(f"media[{index}].data_uri must include a comma and a payload")
        media_type, _, encoding = data_spec.partition(";")
        if encoding != "base64":
            raise ValueError(f"media[{index}].data_uri must use base64 encoding")
        if media_type and not media_type.lower().startswith(f"{modality}/"):
            raise ValueError(f"media[{index}].data_uri must declare an {modality} type")
    elif source == "raw_bytes":
        if item.mime_type and not item.mime_type.lower().startswith(f"{modality}/"):
            raise ValueError(
                f"media[{index}].mime_type must start with {modality}/ for raw {modality} bytes"
            )
    else:
        raise ValueError(f"media[{index}] has an unknown source '{source}'")


def _validate_image_item(item: generation_pb2.MediaItem, index: int) -> None:
    _validate_media_item(item, index, "image")


def _validate_audio_item(item: generation_pb2.MediaItem, index: int) -> None:
    _validate_media_item(item, index, "audio")


def _validate_video_item(item: generation_pb2.MediaItem, index: int) -> None:
    _validate_media_item(item, index, "video")


async def _load_media_item(
    item: generation_pb2.MediaItem, media_io: Any, index: int, modality: str
) -> Any:
    source = item.WhichOneof("source")
    if source == "raw_bytes":
        try:
            return await media_io.async_load_bytes(bytes(item.raw_bytes))
        except Exception as error:
            raise ValueError(f"media[{index}].raw_bytes could not be decoded: {error}") from error

    value = item.url if source == "url" else item.data_uri
    try:
        return await media_io.async_load(value)
    except MediaFetchError as error:
        raise MediaUnavailableError(
            f"media[{index}].{source} is temporarily unavailable: {error}"
        ) from error
    except Exception as error:
        raise ValueError(f"media[{index}].{source} could not be loaded: {error}") from error


def _image_loader_factory(llm: Any, _budget: _DecodedMediaBudget) -> _MediaLoader:
    image_io = _media_io_for_model(llm, "image", ImageMediaIO)

    async def load(item: generation_pb2.MediaItem, index: int) -> Any:
        return await _load_media_item(item, image_io, index, "image")

    return load


def _video_loader_factory(llm: Any, budget: _DecodedMediaBudget) -> _MediaLoader:
    video_io = _media_io_for_model(
        llm,
        "video",
        VideoMediaIO,
        {"decoded_memory_reserver": budget.reserve},
    )

    async def load(item: generation_pb2.MediaItem, index: int) -> Any:
        return await _load_media_item(item, video_io, index, "video")

    return load


def _audio_loader_factory(llm: Any, budget: _DecodedMediaBudget) -> _MediaLoader:
    audio_io = _media_io_for_model(
        llm, "audio", AudioMediaIO, {"decoded_memory_reserver": budget.reserve}
    )

    async def load(item: generation_pb2.MediaItem, index: int) -> Any:
        return await _load_media_item(item, audio_io, index, "audio")

    return load


def _decoded_image_bytes(image: Any) -> int:
    import numpy as np
    import torch
    from PIL import Image

    if isinstance(image, torch.Tensor):
        return image.numel() * image.element_size()
    if isinstance(image, np.ndarray):
        return image.nbytes
    if isinstance(image, Image.Image):
        return image.width * image.height * len(image.getbands())
    raise TypeError(f"unsupported decoded image type: {type(image).__name__}")


def _decoded_audio_data_bytes(audio: AudioData) -> int:
    import numpy as np
    import torch

    if isinstance(audio.samples, torch.Tensor):
        return audio.samples.numel() * audio.samples.element_size()
    if isinstance(audio.samples, np.ndarray):
        return audio.samples.nbytes
    raise TypeError(f"unsupported decoded audio type: {type(audio.samples).__name__}")


def _decoded_audio_bytes(audio: Any) -> int:
    if not isinstance(audio, tuple) or len(audio) != 2:
        raise TypeError(f"unsupported decoded audio type: {type(audio).__name__}")
    return _decoded_audio_data_bytes(AudioData(samples=audio[0], sample_rate=audio[1]))


def _video_runtime_available() -> bool:
    try:
        importlib.import_module("cv2")
    except (ImportError, OSError):
        return False
    return True


_MEDIA_HANDLERS: dict[int, _MediaHandler] = {
    generation_pb2.MODALITY_IMAGE: _MediaHandler(
        "image", _validate_image_item, _image_loader_factory, _decoded_image_bytes
    ),
    generation_pb2.MODALITY_VIDEO: _MediaHandler(
        "video",
        _validate_video_item,
        _video_loader_factory,
        _decoded_video_bytes,
        _video_runtime_available,
        reserves_decoded_bytes=True,
    ),
    generation_pb2.MODALITY_AUDIO: _MediaHandler(
        "audio",
        _validate_audio_item,
        _audio_loader_factory,
        _decoded_audio_bytes,
        reserves_decoded_bytes=True,
    ),
}


def media_capabilities_for_model(llm: Any) -> MediaCapabilities:
    return resolve_media_capabilities(
        llm,
        {handler.name for handler in _MEDIA_HANDLERS.values() if handler.runtime_available()},
    )


def _modality_name(modality: int) -> str:
    try:
        return generation_pb2.Modality.Name(modality).removeprefix("MODALITY_").lower()
    except ValueError:
        return f"unknown ({modality})"


def _prepare_input_from_request(
    request: generation_pb2.GenerateRequest,
    llm: Any,
    media_capabilities: MediaCapabilities | None = None,
) -> _PreparedInput:
    """Validate request media without starting any fetch or decode work."""
    inputs = _base_input_from_request(request, llm)
    if not request.media:
        return _PreparedInput(inputs, (), None)

    capabilities = media_capabilities or media_capabilities_for_model(llm)
    modalities = capabilities.modalities
    if not modalities:
        raise UnsupportedFeatureError("media input is not supported by the loaded model")

    if len(request.media) > _MAX_MEDIA_ITEMS:
        raise ValueError(
            f"at most {_MAX_MEDIA_ITEMS} media items are supported per request, "
            f"got {len(request.media)}"
        )

    resolved_handlers: list[_MediaHandler] = []
    for index, item in enumerate(request.media):
        if item.modality == generation_pb2.MODALITY_UNSPECIFIED:
            raise ValueError(f"media[{index}].modality must be set")
        handler = _MEDIA_HANDLERS.get(item.modality)
        if handler is None or handler.name not in modalities:
            modality = _modality_name(item.modality)
            supported_text = ", ".join(sorted(modalities)) or "none"
            raise UnsupportedFeatureError(
                f"multimodal modality '{modality}' is not supported by the loaded model; "
                f"supported modalities: {supported_text}"
            )
        resolved_handlers.append(handler)

    media = tuple(zip(request.media, resolved_handlers))
    for index, (item, handler) in enumerate(media):
        handler.validate(item, index)

    return _PreparedInput(inputs, media, capabilities)


def _count_token_sequences(
    prompt_ids: Sequence[int], sequences: Iterable[tuple[int, ...]]
) -> Counter[tuple[int, ...]]:
    """Count overlapping markers in one traversal, sharing common prefixes."""
    trie: dict = {}
    for sequence in sequences:
        node = trie
        for token_id in sequence:
            node = node.setdefault(token_id, {})
        node[None] = sequence
    counts: Counter[tuple[int, ...]] = Counter()
    for start, token_id in enumerate(prompt_ids):
        node = trie.get(token_id)
        index = start + 1
        while node is not None:
            sequence = node.get(None)
            if sequence is not None:
                counts[sequence] += 1
            if index == len(prompt_ids):
                break
            node = node.get(prompt_ids[index])
            index += 1
    return counts


def _validate_media_placeholders(
    prepared: _PreparedInput, llm: Any, *, add_special_tokens: bool = False
) -> list[int] | None:
    """Reject placeholder mismatches before probing the stateful MM processor."""
    if not prepared.media or prepared.capabilities is None:
        return
    processor = llm.input_processor
    if not isinstance(processor, BaseMultimodalInputProcessor):
        return
    if not processor.requires_input_placeholders(tokenized=not isinstance(prepared.inputs, str)):
        return
    tokenizer = llm.tokenizer
    prompt_ids = (
        tokenizer.encode(prepared.inputs, add_special_tokens=add_special_tokens)
        if isinstance(prepared.inputs, str)
        else prepared.inputs["prompt_token_ids"]
    )
    item_counts: Counter[str] = Counter()
    expected_sequences: Counter[tuple[int, ...]] = Counter()
    for _, handler in prepared.media:
        item_counts[handler.name] += 1
        placeholder = prepared.capabilities.placeholder(handler.name, item_counts[handler.name])
        sequence = tuple(tokenizer.encode(placeholder, add_special_tokens=False))
        if not sequence:
            raise ValueError(f"{handler.name} placeholder must encode to at least one token")
        expected_sequences[sequence] += 1

    actual_sequences = _count_token_sequences(prompt_ids, expected_sequences)
    for sequence, expected in expected_sequences.items():
        actual = actual_sequences[sequence]
        if actual != expected:
            raise ValueError(
                f"media placeholder count does not match media items: expected {expected}, "
                f"got {actual} for placeholder token IDs {list(sequence)}"
            )

    mm_ids = processor.get_mm_token_ids()
    if mm_ids is not None:
        special_ids = processor.get_mm_special_token_ids()
        content_ids = set(mm_ids.tolist()) - (
            set(special_ids.tolist()) if special_ids is not None else set()
        )
        expected_tokens: Counter[int] = Counter()
        for sequence, count in expected_sequences.items():
            for token_id in sequence:
                if token_id in content_ids:
                    expected_tokens[token_id] += count
        actual_tokens = Counter(token_id for token_id in prompt_ids if token_id in content_ids)
        if actual_tokens != expected_tokens:
            raise ValueError("multimodal placeholder token counts do not match media items")
    return prompt_ids


async def _load_prepared_input(prepared: _PreparedInput, llm: Any) -> str | dict[str, Any]:
    """Fetch and decode media from an already validated request plan."""
    if not prepared.media:
        return prepared.inputs

    capabilities = prepared.capabilities
    if capabilities is None:
        raise RuntimeError("media input plan is missing capabilities")

    budget = _DecodedMediaBudget(_MAX_TOTAL_DECODED_MEDIA_BYTES)
    loaders: dict[str, _MediaLoader] = {}
    for _, handler in prepared.media:
        if handler.name not in loaders:
            loaders[handler.name] = handler.loader_factory(llm, budget)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_MEDIA_LOADS)

    async def load_bounded(
        item: generation_pb2.MediaItem, index: int, handler: _MediaHandler
    ) -> Any:
        async with semaphore:
            decoded = await loaders[handler.name](item, index)
            if not handler.reserves_decoded_bytes:
                budget.charge(handler.decoded_bytes(decoded))
            return decoded

    tasks = [
        asyncio.create_task(load_bounded(item, index, handler))
        for index, (item, handler) in enumerate(prepared.media)
    ]
    try:
        decoded_media = await asyncio.gather(*tasks)
    except (Exception, asyncio.CancelledError):
        # gather leaves siblings running after the first failure.
        budget.cancel()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    grouped_media: dict[str, list[Any]] = {}
    grouped_uuids: dict[str, list[str | None]] = {}
    item_order: list[dict[str, Any]] = []
    for (item, handler), decoded in zip(prepared.media, decoded_media):
        modality_items = grouped_media.setdefault(handler.name, [])
        modality_index = len(modality_items)
        modality_items.append(decoded)
        grouped_uuids.setdefault(handler.name, []).append(item.uuid or None)
        item_order.append(
            {
                "modality": handler.name,
                "index": modality_index,
                "placeholder": capabilities.placeholder(handler.name, modality_index + 1),
            }
        )

    mapped: dict[str, Any] = (
        {"prompt": prepared.inputs} if isinstance(prepared.inputs, str) else dict(prepared.inputs)
    )
    mapped["multi_modal_data"] = grouped_media
    if any(item.uuid for item, _ in prepared.media):
        mapped["multi_modal_uuids"] = grouped_uuids
    mapped["mm_item_order"] = item_order
    return mapped


async def _input_from_request(
    request: generation_pb2.GenerateRequest,
    llm: Any,
    media_capabilities: MediaCapabilities | None = None,
) -> str | dict[str, Any]:
    """Build LLM API inputs without blocking the event loop on media."""
    prepared = _prepare_input_from_request(request, llm, media_capabilities)
    return await _load_prepared_input(prepared, llm)


__all__ = [
    "ExtraOptions",
    "PORTABLE_REQUEST_CAPABILITIES",
    "RequestCapabilities",
    "RequestMetadata",
    "conversation_params_from_request",
    "extra_options_from_request",
    "max_num_sequences_for",
    "request_capabilities_for_model",
    "sampling_params_from_request",
    "token_ids_input_supported",
    "uses_one_model_speculation",
    "accepted_sampling_values",
]

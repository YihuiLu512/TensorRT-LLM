# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM conventions on top of OpenEngine v1, shared by the serving client and the engines.

OpenEngine v1 has no fields for some things trtllm-serve needs: capabilities
beyond the protocol's own, request options that ``LLM`` takes as callables or
tensors, scheduling options, a tokenizer fingerprint, and the counters that
prove a request went through the engine. These conventions carry them in the
protocol's ``extra`` fields (``ServerInfo.extra``, ``ModelInfo.extra``,
``GenerateRequest.extra``) and in ``LoadInfo.attributes``. They bind only an
engine that reports :data:`CONVENTIONS_VERSION_KEY`.

The client library, the serving side and the engines all import this module, so
it must not import ``tensorrt_llm.serve``; protobuf is imported only by the
functions that read protobuf messages.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.protobuf import struct_pb2

    from tensorrt_llm.grpc.openengine.bindings import model_pb2, server_pb2

# --- Version (ServerInfo.extra) ----------------------------------------------

CONVENTIONS_VERSION = 2
CONVENTIONS_VERSION_KEY = "trtllm.conventions_version"

# --- Request keys (GenerateRequest.extra) -------------------------------------

REQ_DETOKENIZE = "detokenize"
REQ_CONVERSATION_ID = "conversation_id"
REQ_THINKING_TOKEN_BUDGET = "trtllm.thinking_token_budget"
REQ_LOGIT_BIAS = "trtllm.logit_bias"
REQ_KV_HINTS = "trtllm.kv_hints"
REQ_SCHEDULING = "trtllm.scheduling"
REQ_BEST_OF = "trtllm.best_of"

REQUEST_KEYS = (
    REQ_DETOKENIZE,
    REQ_CONVERSATION_ID,
    REQ_THINKING_TOKEN_BUDGET,
    REQ_LOGIT_BIAS,
    REQ_KV_HINTS,
    REQ_SCHEDULING,
    REQ_BEST_OF,
)

LOGIT_BIAS_MIN = -100.0
LOGIT_BIAS_MAX = 100.0
LOGIT_BIAS_MAX_ENTRIES = 300

# --- Capability keys (ModelInfo.extra) ----------------------------------------

CAP_MIN_P = "trtllm.sampling.min_p"
CAP_FREQUENCY_PENALTY = "trtllm.sampling.frequency_penalty"
CAP_PRESENCE_PENALTY = "trtllm.sampling.presence_penalty"
CAP_REPETITION_PENALTY = "trtllm.sampling.repetition_penalty"
CAP_SEED_SCOPE = "trtllm.sampling.seed_scope"
CAP_STOP_TEXT = "trtllm.stopping.stop_text"
CAP_REQ_CONVERSATION_ID = "trtllm.request.conversation_id"
CAP_REQ_THINKING_TOKEN_BUDGET = "trtllm.request.thinking_token_budget"
CAP_REQ_LOGIT_BIAS = "trtllm.request.logit_bias"
CAP_REQ_KV_HINTS = "trtllm.request.kv_hints"
CAP_REQ_SCHEDULING = "trtllm.request.scheduling"
CAP_REQ_BEST_OF = "trtllm.request.best_of"

SAMPLING_CAPABILITY_KEYS = (
    CAP_MIN_P,
    CAP_FREQUENCY_PENALTY,
    CAP_PRESENCE_PENALTY,
    CAP_REPETITION_PENALTY,
)
REQUEST_CAPABILITY_KEYS = (
    CAP_REQ_CONVERSATION_ID,
    CAP_REQ_THINKING_TOKEN_BUDGET,
    CAP_REQ_LOGIT_BIAS,
    CAP_REQ_KV_HINTS,
    CAP_REQ_SCHEDULING,
    CAP_REQ_BEST_OF,
)
# Every boolean capability key; a conforming engine reports all of them.
BOOL_CAPABILITY_KEYS = SAMPLING_CAPABILITY_KEYS + (CAP_STOP_TEXT,) + REQUEST_CAPABILITY_KEYS

# The capability a request key needs. ``detokenize`` needs none.
REQUEST_KEY_CAPABILITY = {
    REQ_CONVERSATION_ID: CAP_REQ_CONVERSATION_ID,
    REQ_THINKING_TOKEN_BUDGET: CAP_REQ_THINKING_TOKEN_BUDGET,
    REQ_LOGIT_BIAS: CAP_REQ_LOGIT_BIAS,
    REQ_KV_HINTS: CAP_REQ_KV_HINTS,
    REQ_SCHEDULING: CAP_REQ_SCHEDULING,
    REQ_BEST_OF: CAP_REQ_BEST_OF,
}
# The capability a ``GenerateRequest.sampling`` field needs.
SAMPLING_FIELD_CAPABILITY = {
    "min_p": CAP_MIN_P,
    "frequency_penalty": CAP_FREQUENCY_PENALTY,
    "presence_penalty": CAP_PRESENCE_PENALTY,
    "repetition_penalty": CAP_REPETITION_PENALTY,
}

SEED_SCOPE_NONE = "none"
SEED_SCOPE_BATCH_DEPENDENT = "batch_dependent"
SEED_SCOPE_PER_REQUEST = "per_request"
SEED_SCOPES = (SEED_SCOPE_NONE, SEED_SCOPE_BATCH_DEPENDENT, SEED_SCOPE_PER_REQUEST)

# Older ``extra`` keys of the TRT-LLM server; their names stay as they are.
SERVER_DP_RANK_TARGETING_KEY = "trtllm_supports_dp_rank_targeting"
MODEL_SUPPORTED_MODALITIES_KEY = "trtllm_supported_modalities"

# --- Tokenizer fingerprint (ModelInfo.extra) ----------------------------------

FP_VOCAB_SIZE = "trtllm.tokenizer.vocab_size"
FP_EOS_TOKEN_IDS = "trtllm.tokenizer.eos_token_ids"
FP_BOS_TOKEN_ID = "trtllm.tokenizer.bos_token_id"
FP_FILES_SHA256 = "trtllm.tokenizer.files_sha256"
# Model-directory files hashed into the fingerprint, when present.
FINGERPRINT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "chat_template.json",
)

# --- Counters (LoadInfo.attributes; serving side) ------------------------------

COUNTER_GENERATE_RECEIVED = "trtllm.counters.generate_received"
COUNTER_GENERATE_ACCEPTED = "trtllm.counters.generate_accepted"
# The serving side's count of Generate calls it sent, and where it writes it.
SERVE_COUNTER_GENERATE_SENT = "generate_sent"
SERVE_COUNTER_DIR_ENV = "TRTLLM_OPENENGINE_COUNTER_DIR"

# --- Request metadata ---------------------------------------------------------

META_TRACEPARENT = "traceparent"
META_TRACESTATE = "tracestate"
META_PRIORITY = "openengine-priority"
META_TARGET_DP_RANK = "openengine-target-dp-rank"
PRIORITY_SCALE = 1000

# The only engine for which a client may assume capabilities it does not
# report, and only while its ServerInfo.extra has no conventions version key.
BUILTIN_CAPABILITY_ENGINE = "tensorrt_llm"


def encode_priority(priority: float) -> int:
    """Map an OpenAI ``priority`` in [0, 1] to the ``openengine-priority`` value in [0, 1000]."""
    if isinstance(priority, bool) or not 0.0 <= priority <= 1.0:
        raise ValueError(f"priority must be in [0, 1], got {priority!r}")
    return round(priority * PRIORITY_SCALE)


def decode_priority(value: int) -> float:
    """Map an ``openengine-priority`` value in [0, 1000] back to a priority in [0, 1]."""
    if isinstance(value, bool) or not 0 <= value <= PRIORITY_SCALE:
        raise ValueError(f"{META_PRIORITY} must be in [0, {PRIORITY_SCALE}], got {value!r}")
    return value / PRIORITY_SCALE


def struct_to_dict(struct: struct_pb2.Struct | None) -> dict[str, Any]:
    """Return a ``google.protobuf.Struct`` as a dict; an unset Struct gives an empty dict."""
    if struct is None:
        return {}
    from google.protobuf import json_format

    return dict(json_format.MessageToDict(struct))


_DECIMAL_INT = re.compile(r"-?[0-9]+")


def as_int(value: Any) -> int | None:
    """Read an integer from a Struct value: a whole number, or a decimal string for large ones.

    Returns None for anything else, including booleans.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str) and _DECIMAL_INT.fullmatch(value):
        try:
            return int(value)
        except ValueError:  # More digits than int() converts.
            return None
    return None


def conventions_version(server_extra: Mapping[str, Any]) -> int | None:
    """The conventions version an engine reports in ``ServerInfo.extra``; None if it reports none."""
    return as_int(server_extra.get(CONVENTIONS_VERSION_KEY))


@dataclass(frozen=True)
class ExtraCapabilities:
    """The capability keys an engine declares in ``ModelInfo.extra``.

    Attributes:
        declared: The boolean capability keys declared ``true``.
        seed_scope: How far a seed makes sampling reproducible; one of :data:`SEED_SCOPES`.
        builtin: Whether these come from the built-in table rather than from the engine.
    """

    declared: frozenset[str] = frozenset()
    seed_scope: str = SEED_SCOPE_NONE
    builtin: bool = False

    def supports(self, key: str) -> bool:
        return key in self.declared


# What a TRT-LLM server whose ServerInfo.extra has no conventions version key supports.
BUILTIN_CAPABILITIES = ExtraCapabilities(
    declared=frozenset(SAMPLING_CAPABILITY_KEYS + (CAP_STOP_TEXT, CAP_REQ_CONVERSATION_ID)),
    builtin=True,
)


def parse_extra_capabilities(model_extra: Mapping[str, Any]) -> ExtraCapabilities:
    """Read the capability keys from ``ModelInfo.extra``.

    A key that is missing or not a boolean counts as not supported; a seed scope
    that is missing or unknown counts as ``"none"``.
    """
    declared = frozenset(key for key in BOOL_CAPABILITY_KEYS if model_extra.get(key) is True)
    seed_scope = model_extra.get(CAP_SEED_SCOPE)
    if seed_scope not in SEED_SCOPES:
        seed_scope = SEED_SCOPE_NONE
    return ExtraCapabilities(declared=declared, seed_scope=seed_scope)


def uses_builtin_capabilities(engine_name: str, server_extra: Mapping[str, Any]) -> bool:
    """Whether the client should use :data:`BUILTIN_CAPABILITIES` instead of reading the keys."""
    return engine_name == BUILTIN_CAPABILITY_ENGINE and CONVENTIONS_VERSION_KEY not in server_extra


def resolve_extra_capabilities(
    engine_name: str, server_extra: Mapping[str, Any], model_extra: Mapping[str, Any]
) -> ExtraCapabilities:
    """The capability keys to check requests against, from the engine or the built-in table."""
    if uses_builtin_capabilities(engine_name, server_extra):
        return BUILTIN_CAPABILITIES
    return parse_extra_capabilities(model_extra)


@dataclass(frozen=True)
class TokenizerFingerprint:
    """What identifies a tokenizer well enough to tell two copies apart.

    Attributes:
        vocab_size: Vocabulary size including added tokens.
        eos_token_ids: Every EOS id that ends generation, ascending.
        bos_token_id: The BOS id, or None.
        files_sha256: ``(file name, sha256 hex)`` pairs of the hashed files, sorted by name.
    """

    vocab_size: int
    eos_token_ids: tuple[int, ...]
    bos_token_id: int | None
    files_sha256: tuple[tuple[str, str], ...] = ()

    @property
    def files(self) -> dict[str, str]:
        return dict(self.files_sha256)

    def to_extra(self) -> dict[str, Any]:
        """The ``ModelInfo.extra`` entries an engine reports for this fingerprint."""
        return {
            FP_VOCAB_SIZE: self.vocab_size,
            FP_EOS_TOKEN_IDS: list(self.eos_token_ids),
            FP_BOS_TOKEN_ID: self.bos_token_id,
            FP_FILES_SHA256: self.files,
        }

    @classmethod
    def from_extra(cls, model_extra: Mapping[str, Any]) -> TokenizerFingerprint | None:
        """Read the fingerprint from ``ModelInfo.extra``; None if it is missing or malformed."""
        vocab_size = as_int(model_extra.get(FP_VOCAB_SIZE))
        eos = model_extra.get(FP_EOS_TOKEN_IDS)
        files = model_extra.get(FP_FILES_SHA256, {})
        if vocab_size is None or not isinstance(eos, list) or not isinstance(files, Mapping):
            return None
        eos_ids = [as_int(token_id) for token_id in eos]
        if any(token_id is None for token_id in eos_ids):
            return None
        bos_raw = model_extra.get(FP_BOS_TOKEN_ID)
        bos = None if bos_raw is None else as_int(bos_raw)
        if bos_raw is not None and bos is None:
            return None
        if not all(isinstance(name, str) and isinstance(h, str) for name, h in files.items()):
            return None
        return cls(
            vocab_size=vocab_size,
            eos_token_ids=tuple(sorted(set(eos_ids))),  # type: ignore[arg-type]
            bos_token_id=bos,
            files_sha256=tuple(sorted(files.items())),
        )


def _token_ids(value: Any) -> list[int]:
    """The integer token ids in an unset, scalar or list token-id setting."""
    values = value if isinstance(value, list) else [value]
    return [v for v in values if isinstance(v, int) and not isinstance(v, bool)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_tokenizer_fingerprint(
    tokenizer: Any, model_dir: str | os.PathLike[str] | None
) -> TokenizerFingerprint:
    """Compute the fingerprint the way both the serving side and the engines must.

    - ``vocab_size``: ``len()`` of the Hugging Face tokenizer, which counts added tokens.
    - ``eos_token_ids``: the tokenizer's ``eos_token_id`` together with the
      ``eos_token_id`` of ``generation_config.json`` in ``model_dir`` when that
      file holds a JSON object (each a number or a list), without duplicates, ascending.
      Ids that are not integers are left out.
    - ``bos_token_id``: the tokenizer's ``bos_token_id`` when it is an integer, else None.
    - ``files_sha256``: sha256 of each of :data:`FINGERPRINT_FILES` present in ``model_dir``.

    Args:
        tokenizer: A Hugging Face tokenizer, or a TRT-LLM wrapper whose
            ``tokenizer`` attribute holds one.
        model_dir: The model directory, or None to hash no files and read no
            generation config.

    Raises:
        TypeError: The tokenizer does not report a vocabulary size.
        NotImplementedError: The tokenizer derives from Hugging Face's
            ``PreTrainedTokenizerBase`` without implementing ``__len__``.
        OSError: A file in ``model_dir`` cannot be read.
        ValueError: ``generation_config.json`` is not valid JSON.
    """
    hf_tokenizer = getattr(tokenizer, "tokenizer", None)
    if hf_tokenizer is None:
        hf_tokenizer = tokenizer
    vocab_size = len(hf_tokenizer)
    eos_ids = set(_token_ids(getattr(hf_tokenizer, "eos_token_id", None)))
    files: dict[str, str] = {}
    if model_dir is not None:
        directory = Path(model_dir)
        generation_config = directory / "generation_config.json"
        if generation_config.is_file():
            config = json.loads(generation_config.read_text(encoding="utf-8"))
            if isinstance(config, dict):
                eos_ids.update(_token_ids(config.get("eos_token_id")))
        for name in FINGERPRINT_FILES:
            path = directory / name
            if path.is_file():
                files[name] = _sha256(path)
    bos = getattr(hf_tokenizer, "bos_token_id", None)
    return TokenizerFingerprint(
        vocab_size=vocab_size,
        eos_token_ids=tuple(sorted(eos_ids)),
        bos_token_id=bos if isinstance(bos, int) and not isinstance(bos, bool) else None,
        files_sha256=tuple(sorted(files.items())),
    )


@dataclass(frozen=True)
class FingerprintMismatch:
    """One item on which the serving side's tokenizer and the engine's differ."""

    item: str
    local: Any
    engine: Any

    def __str__(self) -> str:
        return f"{self.item}: serve={self.local!r} engine={self.engine!r}"


def compare_tokenizer_fingerprints(
    local: TokenizerFingerprint, engine: TokenizerFingerprint
) -> list[FingerprintMismatch]:
    """List where two fingerprints differ; an empty list means they match.

    Files are compared only where both sides hashed the same file name.
    """
    mismatches = [
        FingerprintMismatch(name, getattr(local, name), getattr(engine, name))
        for name in ("vocab_size", "eos_token_ids", "bos_token_id")
        if getattr(local, name) != getattr(engine, name)
    ]
    local_files, engine_files = local.files, engine.files
    for name in sorted(local_files.keys() & engine_files.keys()):
        if local_files[name] != engine_files[name]:
            mismatches.append(
                FingerprintMismatch(f"files_sha256[{name}]", local_files[name], engine_files[name])
            )
    return mismatches


def engine_counters(attributes: Mapping[str, Any]) -> tuple[int, int] | None:
    """``(generate_received, generate_accepted)`` from ``LoadInfo.attributes``; None if absent."""
    received = as_int(attributes.get(COUNTER_GENERATE_RECEIVED))
    accepted = as_int(attributes.get(COUNTER_GENERATE_ACCEPTED))
    if received is None or accepted is None:
        return None
    return received, accepted


@dataclass(frozen=True)
class ModelCapabilities:
    """What an engine declares it can do: ``ModelInfo`` and the capability keys merged.

    A field the engine leaves unset counts as not supported.

    Attributes:
        output_logprobs_max_top_n: Most candidates per output token, or None if
            output logprobs are not supported; 0 means only the sampled token.
        prompt_logprobs_max_top_n: The same for prompt tokens.
        guided_modes: Guided decoding modes, named as the ``GuidedDecoding``
            oneof fields (``"json_schema"``, ``"regex"``, ...).
        max_num_sequences: Most sequences one Generate call can return; 1 if unset.
        supported_modalities: The media modalities the engine accepts.
        supports_dp_rank_targeting: Whether ``openengine-target-dp-rank`` is honoured.
        extra: The declared capability keys.
        tokenizer_fingerprint: The engine's fingerprint, or None if it reports none.
    """

    served_model_name: str
    max_context_length: int | None = None
    max_output_tokens: int | None = None
    supports_token_ids_input: bool = False
    supports_text_input: bool = False
    output_logprobs_max_top_n: int | None = None
    prompt_logprobs_max_top_n: int | None = None
    guided_modes: frozenset[str] = frozenset()
    max_num_sequences: int = 1
    supports_priority: bool = False
    supports_stop_in_output: bool = False
    supports_cache_salt: bool = False
    supports_lora: bool = False
    supports_multimodal: bool = False
    supported_modalities: frozenset[str] = frozenset()
    supports_dp_rank_targeting: bool = False
    reasoning_parser: str = ""
    tool_call_parser: str = ""
    extra: ExtraCapabilities = ExtraCapabilities()
    tokenizer_fingerprint: TokenizerFingerprint | None = None

    def supports(self, key: str) -> bool:
        """Whether a capability key is declared."""
        return self.extra.supports(key)


_GUIDED_MODE_PREFIX = "GUIDED_DECODING_MODE_"


def model_capabilities_from_proto(
    server_info: server_pb2.ServerInfo, model_info: model_pb2.ModelInfo
) -> ModelCapabilities:
    """Merge an engine's ``ServerInfo`` and ``ModelInfo`` into :class:`ModelCapabilities`."""
    from tensorrt_llm.grpc.openengine.bindings import model_pb2

    server_extra = struct_to_dict(server_info.extra) if server_info.HasField("extra") else {}
    model_extra = struct_to_dict(model_info.extra) if model_info.HasField("extra") else {}

    def flag(message: Any, name: str) -> bool:
        return message.HasField(name) and bool(getattr(message, name))

    def optional_int(message: Any, name: str) -> int | None:
        return int(getattr(message, name)) if message.HasField(name) else None

    def logprob_top_n(caps: model_pb2.LogprobCapabilities) -> int | None:
        if not flag(caps, "supported"):
            return None
        return optional_int(caps, "max_top_n") or 0

    generation = model_info.generation
    guided = generation.guided_decoding
    guided_modes: frozenset[str] = frozenset()
    if flag(guided, "supported"):
        guided_modes = frozenset(
            model_pb2.GuidedDecodingMode.Name(mode).removeprefix(_GUIDED_MODE_PREFIX).lower()
            for mode in guided.modes
            if mode in model_pb2.GuidedDecodingMode.values()
            and mode != model_pb2.GUIDED_DECODING_MODE_UNSPECIFIED
        )
    modalities = model_extra.get(MODEL_SUPPORTED_MODALITIES_KEY)
    return ModelCapabilities(
        served_model_name=model_info.served_model_name,
        max_context_length=optional_int(model_info, "max_context_length"),
        max_output_tokens=optional_int(model_info, "max_output_tokens"),
        supports_token_ids_input=flag(model_info, "supports_token_ids_input"),
        supports_text_input=flag(model_info, "supports_text_input"),
        output_logprobs_max_top_n=logprob_top_n(generation.output_logprobs),
        prompt_logprobs_max_top_n=logprob_top_n(generation.prompt_logprobs),
        guided_modes=guided_modes,
        max_num_sequences=optional_int(generation, "max_num_sequences") or 1,
        supports_priority=flag(generation, "supports_priority"),
        supports_stop_in_output=flag(generation, "supports_stop_in_output"),
        supports_cache_salt=flag(generation, "supports_cache_salt"),
        supports_lora=flag(model_info, "supports_lora"),
        supports_multimodal=flag(model_info, "supports_multimodal"),
        supported_modalities=frozenset(m for m in modalities if isinstance(m, str))
        if isinstance(modalities, list)
        else frozenset(),
        supports_dp_rank_targeting=server_extra.get(SERVER_DP_RANK_TARGETING_KEY) is True,
        reasoning_parser=model_info.reasoning_parser,
        tool_call_parser=model_info.tool_call_parser,
        extra=resolve_extra_capabilities(server_info.engine_name, server_extra, model_extra),
        tokenizer_fingerprint=TokenizerFingerprint.from_extra(model_extra),
    )

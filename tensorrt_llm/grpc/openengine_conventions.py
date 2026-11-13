# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM conventions on top of OpenEngine v1, shared by the serving client and the engines.

OpenEngine v1 has no fields for some things trtllm-serve needs: capabilities
beyond the protocol's own, request options that ``LLM`` takes as callables or
tensors, scheduling options, a tokenizer fingerprint, and the counters that
prove a request went through the engine. These conventions carry them in the
protocol's ``extra`` fields (``ServerInfo.extra``, ``ModelInfo.extra``,
``GenerateRequest.extra``) and in ``LoadInfo.attributes``. An engine that
reports :data:`CONVENTIONS_VERSION_KEY` follows them; the serving side reads
the capability keys an engine declares whether or not it reports a version.

The serving side and the engines both import this module, so it must not import
``tensorrt_llm.serve``; protobuf is imported only by the functions that read
protobuf messages.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.protobuf import struct_pb2

    from tensorrt_llm.grpc.openengine.bindings import model_pb2, server_pb2

# --- Version (ServerInfo.extra) ----------------------------------------------

# Version 4 adds the disaggregated-serving keys (request keys, capability keys,
# counters and the handoff attribute names below) and the
# ``trtllm.stopping.min_tokens`` capability key; it changes no version 3 key.
CONVENTIONS_VERSION = 4
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
# The boolean capability keys of version 3; a conforming engine reports all of
# them, and VERSION_4_CAPABILITY_KEYS too when it reports version 4.
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

# --- Disaggregated serving (version 4) ------------------------------------------
#
# A user request served disaggregated is two Generate calls: a prefill call on
# the context engine, which ends with ``PrefillReady`` carrying the handoff,
# and a decode call on the generation engine, which carries that handoff in
# ``kv.session``. The request keys keep the names of the matching
# ``DisaggregatedParams`` fields, without the ``trtllm.`` prefix.

# Request keys (GenerateRequest.extra).
REQ_REQUEST_TYPE = "request_type"
REQ_DISAGG_REQUEST_ID = "disagg_request_id"
REQ_SCHEDULE_STYLE = "schedule_style"
REQ_ORIG_ELAPSED_S = "orig_elapsed_s"
DISAGG_REQUEST_KEYS = (
    REQ_REQUEST_TYPE,
    REQ_DISAGG_REQUEST_ID,
    REQ_SCHEDULE_STYLE,
    REQ_ORIG_ELAPSED_S,
)

# Values of ``request_type``. Only context_only and context_and_generation are
# sent in ``extra``; a decode call is marked by its ``kv.session`` instead.
REQUEST_TYPE_CONTEXT_ONLY = "context_only"
REQUEST_TYPE_GENERATION_ONLY = "generation_only"
REQUEST_TYPE_CONTEXT_AND_GENERATION = "context_and_generation"

# ``disagg_request_id`` is a decimal string of ASCII digits in
# [DISAGG_REQUEST_ID_MIN, DISAGG_REQUEST_ID_LIMIT): too large for a Struct number.
DISAGG_REQUEST_ID_MIN = 1 << 40
DISAGG_REQUEST_ID_LIMIT = 1 << 63

# ``schedule_style`` values, as ``DisaggScheduleStyle``; only generation-first is sent.
SCHEDULE_STYLE_CONTEXT_FIRST = 0
SCHEDULE_STYLE_GENERATION_FIRST = 1

# Capability keys (ModelInfo.extra). An engine that reports version 4 reports
# these as well as BOOL_CAPABILITY_KEYS; a missing key counts as false.
CAP_REQ_DISAGG_REQUEST_ID = "trtllm.request.disagg_request_id"
CAP_DISAGG_GENERATION_FIRST = "trtllm.disagg.generation_first"
CAP_DISAGG_ORIG_ELAPSED_S = "trtllm.disagg.orig_elapsed_s"
# ``stopping.min_tokens >= 1`` takes effect, holding back both EOS and stop
# tokens until the minimum is reached.
CAP_STOPPING_MIN_TOKENS = "trtllm.stopping.min_tokens"
VERSION_4_CAPABILITY_KEYS = (
    CAP_REQ_DISAGG_REQUEST_ID,
    CAP_DISAGG_GENERATION_FIRST,
    CAP_DISAGG_ORIG_ELAPSED_S,
    CAP_STOPPING_MIN_TOKENS,
)

# The capability a disaggregated request key needs. ``request_type`` needs none:
# a prefill call needs ``ServerInfo.kv_connector`` to be enabled with
# ``supports_remote_prefill``.
DISAGG_REQUEST_KEY_CAPABILITY: Mapping[str, str] = MappingProxyType(
    {
        REQ_DISAGG_REQUEST_ID: CAP_REQ_DISAGG_REQUEST_ID,
        REQ_SCHEDULE_STYLE: CAP_DISAGG_GENERATION_FIRST,
        REQ_ORIG_ELAPSED_S: CAP_DISAGG_ORIG_ELAPSED_S,
    }
)

# The parts of a GenerateRequest in which the prefill and the decode call of
# one user request may differ, and on which call each appears ("both": on both,
# with different values). Every other part of the two calls is the same.
LEG_SPECIFIC_FIELDS: Mapping[str, str] = MappingProxyType(
    {
        "request_id": "both",
        f"extra.{REQ_REQUEST_TYPE}": "prefill",
        f"extra.{REQ_DISAGG_REQUEST_ID}": "prefill",
        f"extra.{REQ_SCHEDULE_STYLE}": "prefill",
        "kv.session": "decode",
        f"extra.{REQ_ORIG_ELAPSED_S}": "decode",
    }
)

# KvSessionRef.attributes_struct keys. The mapped keys are named as the
# ``DisaggregatedParams`` fields they carry; every other key passes through the
# serving side untouched. A key an engine adds must be registered here first.
HANDOFF_MAPPED_ATTRIBUTES = (
    "ctx_info_endpoint",
    "opaque_state",
    "first_gen_tokens",
    "draft_tokens",
    "disagg_request_id",
    "ctx_dp_rank",
    "schedule_style",
    "ctx_usage",
    "first_gen_log_probs",
)
HANDOFF_DECODE_PREPARATION = "trtllm_decode_preparation"
HANDOFF_REGISTERED_PASSTHROUGH_ATTRIBUTES = (HANDOFF_DECODE_PREPARATION,)

# Buckets of the per-request-type counters: a call with ``kv.session`` is
# generation_only; one without it whose ``request_type`` is context_only is
# context_only; every other call is aggregated.
BUCKET_CONTEXT_ONLY = REQUEST_TYPE_CONTEXT_ONLY
BUCKET_GENERATION_ONLY = REQUEST_TYPE_GENERATION_ONLY
BUCKET_AGGREGATED = "aggregated"
REQUEST_BUCKETS = (BUCKET_CONTEXT_ONLY, BUCKET_GENERATION_ONLY, BUCKET_AGGREGATED)


def request_bucket(*, has_kv_session: bool, request_type: Any) -> str:
    """The counter bucket of one Generate call, for ``generate_accepted.*`` and ``generate_sent.*``.

    The engine and the serving side bucket a call alike. ``http_received.*``
    is not bucketed by this: it goes by the HTTP request's ``request_type``.
    """
    if has_kv_session:
        return BUCKET_GENERATION_ONLY
    if request_type == REQUEST_TYPE_CONTEXT_ONLY:
        return BUCKET_CONTEXT_ONLY
    return BUCKET_AGGREGATED


# The canonical decimal form, without leading zeros. DISAGG_REQUEST_ID_LIMIT - 1
# has 19 digits; the bound keeps int() off long strings.
_DISAGG_REQUEST_ID = re.compile(r"[1-9][0-9]{0,18}")


def parse_disagg_request_id(value: Any) -> int | None:
    """Read a ``disagg_request_id`` request key; None unless it is a valid decimal string.

    Valid means ASCII digits only, without sign, spaces or leading zeros, in
    [DISAGG_REQUEST_ID_MIN, DISAGG_REQUEST_ID_LIMIT).
    """
    if not isinstance(value, str) or not _DISAGG_REQUEST_ID.fullmatch(value):
        return None
    number = int(value)
    return number if DISAGG_REQUEST_ID_MIN <= number < DISAGG_REQUEST_ID_LIMIT else None


# --- Accepted values (ModelInfo.extra, version 3) -------------------------------

# The values an engine accepts for each request parameter, keyed by the names in
# SAMPLING_PARAM_NAMES. An engine that reports this key is read from it alone;
# a parameter missing from it accepts no value.
PARAMS_KEY = "trtllm.params"
# The ``GenerateRequest.sampling`` fields, by the names ``PARAMS_KEY`` uses.
SAMPLING_PARAM_NAMES = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "seed",
)
ACCEPT_ANY = "any"
ACCEPT_RANGE = "range"
ACCEPT_VALUES = "values"
ACCEPT_NONE = "none"
ACCEPT_FORMS = (ACCEPT_ANY, ACCEPT_RANGE, ACCEPT_VALUES, ACCEPT_NONE)
# A ``default`` that names the model's generation_config.json.
DEFAULT_FROM_GENERATION_CONFIG = "generation_config"

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
# The serving side's count of Generate calls it sent, where it publishes it, and
# the field of its file that names the socket it answers on while it runs.
SERVE_COUNTER_GENERATE_SENT = "generate_sent"
SERVE_COUNTER_DIR_ENV = "TRTLLM_OPENENGINE_COUNTER_DIR"
SERVE_COUNTER_ADDRESS = "address"

# Version 4 engine counters (LoadInfo.attributes): generate_accepted by request
# bucket, whose sum is generate_accepted, and the PrefillReady events sent.
# generate_received has no buckets: it counts before the request is read.
COUNTER_GENERATE_ACCEPTED_BUCKETS: Mapping[str, str] = MappingProxyType(
    {bucket: f"{COUNTER_GENERATE_ACCEPTED}.{bucket}" for bucket in REQUEST_BUCKETS}
)
COUNTER_PREFILL_READY_SENT = "trtllm.counters.prefill_ready_sent"
VERSION_4_ENGINE_COUNTERS = (
    *COUNTER_GENERATE_ACCEPTED_BUCKETS.values(),
    COUNTER_PREFILL_READY_SENT,
)

# Version 4 serving-side counters, published next to generate_sent:
# - generate_sent.<bucket>: generate_sent by the bucket of the call sent; they sum to generate_sent.
# - prefill_ready_received: prefill_ready events received.
# - http_received.<bucket>: generation HTTP requests received, counted before
#   any refusal; a request without disaggregated_params, or whose request_type
#   is context_and_generation, is aggregated.
# - http_rejected: generation HTTP requests refused before a Generate call was sent.
# - handoff_unregistered_attributes: handoff attributes passed through that are
#   neither mapped nor registered (HANDOFF_MAPPED_ATTRIBUTES,
#   HANDOFF_REGISTERED_PASSTHROUGH_ATTRIBUTES), one per key per handoff.
SERVE_COUNTER_GENERATE_SENT_BUCKETS: Mapping[str, str] = MappingProxyType(
    {bucket: f"{SERVE_COUNTER_GENERATE_SENT}.{bucket}" for bucket in REQUEST_BUCKETS}
)
SERVE_COUNTER_PREFILL_READY_RECEIVED = "prefill_ready_received"
SERVE_COUNTER_HTTP_RECEIVED_BUCKETS: Mapping[str, str] = MappingProxyType(
    {bucket: f"http_received.{bucket}" for bucket in REQUEST_BUCKETS}
)
SERVE_COUNTER_HTTP_REJECTED = "http_rejected"
SERVE_COUNTER_HANDOFF_UNREGISTERED_ATTRIBUTES = "handoff_unregistered_attributes"
# Every serving-side counter, generate_sent first.
SERVE_COUNTERS = (
    SERVE_COUNTER_GENERATE_SENT,
    *SERVE_COUNTER_GENERATE_SENT_BUCKETS.values(),
    SERVE_COUNTER_PREFILL_READY_RECEIVED,
    *SERVE_COUNTER_HTTP_RECEIVED_BUCKETS.values(),
    SERVE_COUNTER_HTTP_REJECTED,
    SERVE_COUNTER_HANDOFF_UNREGISTERED_ATTRIBUTES,
)

# Set on a serving frontend, the directory it writes the GenerateRequests it
# sends to; unset, nothing is written.
SERVE_REQUEST_DUMP_DIR_ENV = "TRTLLM_OPENENGINE_REQUEST_DUMP_DIR"

# --- Request metadata ---------------------------------------------------------

META_TRACEPARENT = "traceparent"
META_TRACESTATE = "tracestate"
META_PRIORITY = "openengine-priority"
META_TARGET_DP_RANK = "openengine-target-dp-rank"
PRIORITY_SCALE = 1000


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


class CapabilityFormatError(ValueError):
    """An engine's ``trtllm.params`` declaration is malformed."""


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    try:
        return _number(value) and math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


@dataclass(frozen=True)
class AcceptedValues:
    """The values an engine accepts for one request parameter.

    Values are compared as the request sends them (a protobuf double or
    integer), never at the precision the engine computes with.

    Attributes:
        accept: One of :data:`ACCEPT_FORMS`: every value, a range, listed values, or none.
        min: ``range``: the lower bound, or None for no lower bound.
        max: ``range``: the upper bound, or None for no upper bound.
        min_exclusive: ``range``: whether ``min`` itself is left out.
        max_exclusive: ``range``: whether ``max`` itself is left out.
        values: ``values``: the accepted values.
        default: The value the engine uses when the request does not set the
            parameter, :data:`DEFAULT_FROM_GENERATION_CONFIG`, or None when not
            reported. For information only: it never changes what is sent.
    """

    accept: str
    min: float | None = None
    max: float | None = None
    min_exclusive: bool = False
    max_exclusive: bool = False
    values: tuple[float, ...] = ()
    default: Any = None

    def __post_init__(self) -> None:
        if self.accept not in ACCEPT_FORMS:
            raise CapabilityFormatError(f"unknown accept form {self.accept!r}")
        if self.accept == ACCEPT_VALUES and not self.values:
            raise CapabilityFormatError("accept 'values' needs a non-empty values list")
        if self.accept == ACCEPT_RANGE and self.min is None and self.max is None:
            raise CapabilityFormatError("accept 'range' needs min or max")

    def accepts(self, value: float) -> bool:
        """Whether ``value`` is in the set; NaN and infinities are only in ``any``."""
        if self.accept == ACCEPT_ANY:
            return True
        if self.accept == ACCEPT_NONE or not _finite_number(value):
            return False
        if self.accept == ACCEPT_VALUES:
            return any(value == accepted for accepted in self.values)
        if self.min is not None and (
            value < self.min or (self.min_exclusive and value == self.min)
        ):
            return False
        return self.max is None or not (
            value > self.max or (self.max_exclusive and value == self.max)
        )

    def describe(self) -> str:
        """The set in words, for error messages."""
        if self.accept == ACCEPT_ANY:
            return "any value"
        if self.accept == ACCEPT_NONE:
            return "no value"
        if self.accept == ACCEPT_VALUES:
            return "only " + ", ".join(repr(value) for value in self.values)
        low = "(" if self.min_exclusive or self.min is None else "["
        high = ")" if self.max_exclusive or self.max is None else "]"
        low_text = "-inf" if self.min is None else repr(self.min)
        high_text = "inf" if self.max is None else repr(self.max)
        return f"values in {low}{low_text}, {high_text}{high}"

    def to_extra(self) -> dict[str, Any]:
        """The value description an engine reports under ``trtllm.params``."""
        described: dict[str, Any] = {"accept": self.accept}
        if self.accept == ACCEPT_RANGE:
            for name in ("min", "max"):
                if getattr(self, name) is not None:
                    described[name] = getattr(self, name)
            if self.min_exclusive:
                described["min_exclusive"] = True
            if self.max_exclusive:
                described["max_exclusive"] = True
        elif self.accept == ACCEPT_VALUES:
            described["values"] = list(self.values)
        described["default"] = self.default
        return described

    @classmethod
    def from_extra(cls, name: str, described: Any) -> AcceptedValues:
        """Read one value description of ``trtllm.params``.

        Raises:
            CapabilityFormatError: The description is malformed; the message
                names the parameter and quotes what was reported.
        """

        def malformed(why: str) -> CapabilityFormatError:
            return CapabilityFormatError(f"{PARAMS_KEY}.{name} {why}: {described!r}")

        if not isinstance(described, Mapping):
            raise malformed("is not an object")
        accept = described.get("accept")
        if accept not in ACCEPT_FORMS:
            raise malformed(f"has an unknown accept form {accept!r}")
        default = described.get("default")
        if default is not None and not (_number(default) or isinstance(default, str)):
            raise malformed("has a default that is not a number or a string")
        fields: dict[str, Any] = {"accept": accept, "default": default}
        if accept == ACCEPT_RANGE:
            for bound in ("min", "max"):
                value = described.get(bound)
                if value is not None and not _finite_number(value):
                    raise malformed(f"has a {bound} that is not a finite number")
                fields[bound] = value
            for flag in ("min_exclusive", "max_exclusive"):
                value = described.get(flag, False)
                if not isinstance(value, bool):
                    raise malformed(f"has a {flag} that is not a boolean")
                fields[flag] = value
            low, high = fields["min"], fields["max"]
            if low is None and high is None:
                raise malformed("is a range without min or max")
            if (
                low is not None
                and high is not None
                and (
                    low > high
                    or (low == high and (fields["min_exclusive"] or fields["max_exclusive"]))
                )
            ):
                raise malformed("is an empty range")
        elif accept == ACCEPT_VALUES:
            values = described.get("values")
            if (
                not isinstance(values, list)
                or not values
                or not all(_finite_number(value) for value in values)
            ):
                raise malformed("has no non-empty list of finite numbers in values")
            fields["values"] = tuple(values)
        return cls(**fields)


ANY_VALUE = AcceptedValues(ACCEPT_ANY)
NO_VALUE = AcceptedValues(ACCEPT_NONE)


def parse_params(model_extra: Mapping[str, Any]) -> dict[str, AcceptedValues] | None:
    """Read ``trtllm.params`` from ``ModelInfo.extra``; None when the engine does not report it.

    Parameters this module does not know are kept as reported.

    Raises:
        CapabilityFormatError: The declaration is malformed.
    """
    if PARAMS_KEY not in model_extra:
        return None
    declared = model_extra[PARAMS_KEY]
    if not isinstance(declared, Mapping):
        raise CapabilityFormatError(f"{PARAMS_KEY} is not an object: {declared!r}")
    return {
        name: AcceptedValues.from_extra(name, described) for name, described in declared.items()
    }


@dataclass(frozen=True)
class ExtraCapabilities:
    """The capability keys an engine declares in ``ModelInfo.extra``.

    Attributes:
        declared: The boolean capability keys declared ``true``.
        seed_scope: How far a seed makes sampling reproducible; one of :data:`SEED_SCOPES`.
        params: The ``trtllm.params`` declaration, or None when the engine does
            not report one.
    """

    declared: frozenset[str] = frozenset()
    seed_scope: str = SEED_SCOPE_NONE
    params: Mapping[str, AcceptedValues] | None = field(default=None, hash=False)

    def __post_init__(self) -> None:
        if self.params is not None:
            object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def supports(self, key: str) -> bool:
        return key in self.declared

    def accepted(self, name: str) -> AcceptedValues:
        """The values accepted for the sampling parameter ``name`` (a :data:`SAMPLING_PARAM_NAMES` entry).

        From ``trtllm.params`` when the engine reports it, where a missing
        parameter accepts nothing. Otherwise from the boolean keys of the
        version 2 conventions: a parameter with a key accepts any value when it
        is declared ``true`` and none otherwise; one without a key accepts any value.
        """
        if self.params is not None:
            return self.params.get(name, NO_VALUE)
        key = SAMPLING_FIELD_CAPABILITY.get(name)
        if key is None or self.supports(key):
            return ANY_VALUE
        return NO_VALUE


def parse_extra_capabilities(model_extra: Mapping[str, Any]) -> ExtraCapabilities:
    """Read the capability keys from ``ModelInfo.extra``.

    A boolean key that is missing or not a boolean counts as not supported; a
    seed scope that is missing or unknown counts as ``"none"``.

    Raises:
        CapabilityFormatError: ``trtllm.params`` is malformed.
    """
    declared = frozenset(
        key
        for key in BOOL_CAPABILITY_KEYS + VERSION_4_CAPABILITY_KEYS
        if model_extra.get(key) is True
    )
    seed_scope = model_extra.get(CAP_SEED_SCOPE)
    if seed_scope not in SEED_SCOPES:
        seed_scope = SEED_SCOPE_NONE
    return ExtraCapabilities(
        declared=declared, seed_scope=seed_scope, params=parse_params(model_extra)
    )


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

    def accepted(self, name: str) -> AcceptedValues:
        """The values accepted for a sampling parameter; see :meth:`ExtraCapabilities.accepted`."""
        return self.extra.accepted(name)


_GUIDED_MODE_PREFIX = "GUIDED_DECODING_MODE_"


def model_capabilities_from_proto(
    server_info: server_pb2.ServerInfo, model_info: model_pb2.ModelInfo
) -> ModelCapabilities:
    """Merge an engine's ``ServerInfo`` and ``ModelInfo`` into :class:`ModelCapabilities`.

    Raises:
        CapabilityFormatError: ``trtllm.params`` is malformed, or ``ModelInfo.extra``
            cannot be read.
    """
    from tensorrt_llm.grpc.openengine.bindings import model_pb2

    server_extra = struct_to_dict(server_info.extra) if server_info.HasField("extra") else {}
    try:
        model_extra = struct_to_dict(model_info.extra) if model_info.HasField("extra") else {}
    except ValueError as error:  # json_format cannot represent NaN or an infinity
        raise CapabilityFormatError(f"ModelInfo.extra cannot be read: {error}") from error

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
        extra=parse_extra_capabilities(model_extra),
        tokenizer_fingerprint=TokenizerFingerprint.from_extra(model_extra),
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map a request to the OpenEngine Generate calls that serve it.

:func:`map_request` implements the ``map_request`` slot of :mod:`.contracts`.
Every field of ``SamplingParams`` and ``GenerateOptions`` is classified in
:mod:`.field_classes`. A mapped field reaches the engine when the engine
declares what it needs; otherwise it is rejected, or, where the table names a
fallback (stop strings), left to the serving side. A field the serving side
handles, and an internal field (state derived from other fields, or an input of
the mapping itself), is not sent. A field classified as rejected, or missing
from the tables, is rejected when it is set. A rejection is a
``ValueError`` raised before anything is sent, which ``OpenAIServer`` returns
as HTTP 400.

``n`` beyond the sequences one call can return is split into several calls.

:func:`build_generate_request` and :func:`builtin_capabilities` at the end of
this module serve the PoC-S1 ``OpenEngineLLMAdapter`` only.
"""

from __future__ import annotations

import dataclasses
import json
import math
import threading
from collections.abc import Collection, Mapping
from typing import Any

from google.protobuf import struct_pb2

from tensorrt_llm.executor.utils import context_length_exceeded_message
from tensorrt_llm.grpc.openengine.bindings import generation_pb2, model_pb2, server_pb2
from tensorrt_llm.grpc.openengine_conventions import (
    CAP_REQ_BEST_OF,
    CAP_REQ_CONVERSATION_ID,
    CAP_REQ_KV_HINTS,
    CAP_REQ_LOGIT_BIAS,
    CAP_REQ_SCHEDULING,
    CAP_REQ_THINKING_TOKEN_BUDGET,
    CAP_STOP_TEXT,
    LOGIT_BIAS_MAX,
    LOGIT_BIAS_MAX_ENTRIES,
    LOGIT_BIAS_MIN,
    REQ_BEST_OF,
    REQ_CONVERSATION_ID,
    REQ_DETOKENIZE,
    REQ_KV_HINTS,
    REQ_LOGIT_BIAS,
    REQ_SCHEDULING,
    REQ_THINKING_TOKEN_BUDGET,
    SAMPLING_FIELD_CAPABILITY,
    ModelCapabilities,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import _GENERATION_CONFIG_SAMPLING_FIELDS, SamplingParams

from .contracts import GenerateOptions, new_request_id
from .engine_client import EngineRequest, GuidedSpec
from .field_classes import GENERATE_OPTIONS, SAMPLING_PARAMS, Kind

__all__ = [
    "build_generate_request",
    "builtin_capabilities",
    "log_sequence_capability",
    "map_request",
]

# Values that leave sampling unchanged. OpenAIServer fills some of them in when
# the request does not set the field; such a value is sent only when the
# request set it explicitly (SV-MAP-2). SamplingParams records explicit values
# only for _GENERATION_CONFIG_SAMPLING_FIELDS, so a neutral frequency_penalty or
# presence_penalty is never sent, even when the request set it.
_NEUTRAL: dict[str, Any] = {
    "top_k": 0,
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}
_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
)
_SAMPLING_DEFAULTS = {
    f.name: f.default if f.default is not dataclasses.MISSING else f.default_factory()
    for f in dataclasses.fields(SamplingParams)
    if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
}
# OpenEngine's seed is a uint64.
_SEED_LIMIT = 1 << 64
# A protobuf Struct holds numbers as doubles, exact for integers up to 2**53.
_STRUCT_EXACT_INT = 1 << 53
# Ranges of the GenerateRequest fields that take request integers.
_UINT32_LIMIT = 1 << 32
_INT32_MAX = (1 << 31) - 1
# OpenAIServer sends 0 for "not set".
_ZERO_IS_UNSET = frozenset({"prompt_ignore_length"})
# GuidedDecodingParams field -> OpenEngine guided mode.
_GUIDED_MODES = {
    "json": "json_schema",
    "regex": "regex",
    "grammar": "ebnf_grammar",
    "json_object": "json_object",
    "structural_tag": "structural_tag",
}


def _short(value: Any) -> str:
    """``repr`` of a scalar or a short list; the type name of anything else."""
    if isinstance(value, (bool, int, float, str, type(None))):
        return repr(value)
    if isinstance(value, (list, tuple)) and len(value) <= 8:
        if all(isinstance(item, (bool, int, float, str, type(None))) for item in value):
            return repr(value)
    return f"<{type(value).__name__}>"


def _not_supported(name: str, value: Any, engine_name: str, why: str = "") -> ValueError:
    reason = f" ({why})" if why else ""
    return ValueError(
        f"{name}={_short(value)} is not supported when serving through the OpenEngine engine "
        f"{engine_name!r}{reason}"
    )


def _check_struct_ints(name: str, value: Any, engine_name: str) -> None:
    # GenerateRequest.extra is a Struct: kv_hints, scheduling_params and the thinking budget
    # carry request integers into it.
    """Reject integers a ``Struct`` would round, rather than send a different value."""
    if isinstance(value, Mapping):
        for item in value.values():
            _check_struct_ints(name, item, engine_name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_struct_ints(name, item, engine_name)
    elif isinstance(value, int) and not isinstance(value, bool) and abs(value) > _STRUCT_EXACT_INT:
        raise _not_supported(
            name,
            value,
            engine_name,
            f"integers beyond 2**53 cannot be sent exactly; {value} is one",
        )


def _unset(name: str, value: Any) -> bool:
    if value is None or (isinstance(value, (list, tuple, dict)) and not value):
        return True
    if name in _ZERO_IS_UNSET and value == 0:
        return True
    if name not in _SAMPLING_DEFAULTS:
        return False
    try:
        return bool(value == _SAMPLING_DEFAULTS[name])
    except (TypeError, ValueError, RuntimeError):  # A tensor compares elementwise.
        return False


def _check_rejected(sampling_params: SamplingParams, engine_name: str) -> None:
    """Reject a set field that is classified ``REJECTED``, or not classified at all."""
    for f in dataclasses.fields(SamplingParams):
        field_class = SAMPLING_PARAMS.get(f.name)
        if field_class is not None and field_class.kind is not Kind.REJECTED:
            continue
        value = getattr(sampling_params, f.name)
        if not _unset(f.name, value):
            why = field_class.target if field_class is not None else "an unclassified field"
            raise _not_supported(f.name, value, engine_name, why)


def _explicit(sampling_params: SamplingParams, name: str) -> bool:
    provided = sampling_params._request_provided_fields
    return provided is not None and name in provided


def _requested(sampling_params: SamplingParams, name: str) -> Any:
    """The value to send for a sampling field, or None.

    A value the request set explicitly is sent (OE-GEN-2: the engine's own
    default would apply otherwise). A value ``OpenAIServer`` filled in although
    the request did not set it is not sent. For fields whose explicitness
    ``SamplingParams`` does not record, a value is sent when it is not neutral.
    """
    value = getattr(sampling_params, name)
    if value is None:
        return None
    if _explicit(sampling_params, name):
        return value
    provided = sampling_params._request_provided_fields
    if provided is not None and name in _GENERATION_CONFIG_SAMPLING_FIELDS:
        return None  # Filled in by the server.
    if _NEUTRAL.get(name, object()) == value:
        return None
    return value


_logged_lock = threading.Lock()
_logged: set[tuple[str, str, int]] = set()


def log_sequence_capability(engine_name: str, capabilities: ModelCapabilities) -> None:
    """Log once per engine how many sequences one Generate call returns (SV-MAP-6).

    Safe to call at every capability detection: the warning is logged again
    only when the engine, its model or the limit changes.
    """
    limit = max(1, capabilities.max_num_sequences)
    key = (engine_name, capabilities.served_model_name, limit)
    if key in _logged:
        return
    with _logged_lock:
        if key in _logged:
            return
        _logged.add(key)
    logger.warning(
        f"OpenEngine engine {engine_name!r} returns at most {limit} sequence(s) per Generate "
        f"call (max_num_sequences); requests with n > {limit} are split into several calls"
    )


def _sampling(
    sampling_params: SamplingParams, capabilities: ModelCapabilities, engine_name: str
) -> dict[str, float | int]:
    sampling: dict[str, float | int] = {}
    for name in _SAMPLING_FIELDS:
        value = _requested(sampling_params, name)
        if value is None:
            continue
        if isinstance(value, bool) or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
        capability = SAMPLING_FIELD_CAPABILITY.get(name)
        if capability is not None and not capabilities.supports(capability):
            raise _not_supported(name, value, engine_name)
        if name == "top_k" and not 0 <= value <= _INT32_MAX:
            raise ValueError(f"top_k must be in [0, {_INT32_MAX}], got {value}")
        sampling[name] = value
    if sampling_params.seed is not None:
        if not 0 <= sampling_params.seed < _SEED_LIMIT:
            raise ValueError(f"seed must be in [0, 2**64), got {sampling_params.seed}")
        sampling["seed"] = sampling_params.seed
    return sampling


def _check_prompt(prompt_token_ids: list[int], capabilities: ModelCapabilities) -> None:
    if not prompt_token_ids:
        raise ValueError("the prompt must have at least one token")
    fingerprint = capabilities.tokenizer_fingerprint
    low, high = min(prompt_token_ids), max(prompt_token_ids)
    vocab = fingerprint.vocab_size if fingerprint is not None else _UINT32_LIMIT
    if low < 0 or high >= vocab:
        bad = low if low < 0 else high
        raise ValueError(f"prompt token id {bad} is out of range: token ids must be [0, {vocab})")
    max_context = capabilities.max_context_length
    if max_context is not None and len(prompt_token_ids) >= max_context:
        # The LLM path's wording, so OpenAIServer reports code context_length_exceeded.
        raise ValueError(
            context_length_exceeded_message(
                max_context_length=max_context, num_prompt_tokens=len(prompt_token_ids)
            )
        )


def _max_tokens(
    sampling_params: SamplingParams,
    capabilities: ModelCapabilities,
    prompt_len: int,
    engine_name: str,
) -> int | None:
    limits = []
    if capabilities.max_context_length is not None:
        limits.append(capabilities.max_context_length - prompt_len)
    if capabilities.max_output_tokens is not None:
        limits.append(capabilities.max_output_tokens)
    limit = min(limits) if limits else None
    max_tokens = sampling_params.max_tokens
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError(f"max_tokens must be greater than 0, got {max_tokens}")
    if max_tokens is None:
        return limit
    if limit is not None and max_tokens > limit:
        # As LLM does: cap the request at what the engine can still generate.
        logger.warning(
            f"max_tokens={max_tokens} is more than the {limit} tokens engine {engine_name!r} "
            "can generate for this prompt; using it instead"
        )
        return limit
    if max_tokens >= _UINT32_LIMIT:
        raise ValueError(f"max_tokens must be less than 2**32, got {max_tokens}")
    return max_tokens


def _stop_strings(sampling_params: SamplingParams) -> list[str]:
    stop = sampling_params.stop
    if stop is None:
        return []
    stops = [stop] if isinstance(stop, str) else list(stop)
    if any(not isinstance(s, str) or not s for s in stops):
        raise ValueError("stop strings must be non-empty strings")
    return list(dict.fromkeys(stops))


def _stopping(
    sampling_params: SamplingParams, capabilities: ModelCapabilities, engine_name: str
) -> tuple[list[str], list[int], bool | None]:
    """``(stop_texts, stop_token_ids, include_stop_in_output)`` for the engine."""
    stops = _stop_strings(sampling_params)
    stop_token_ids = list(dict.fromkeys(sampling_params.stop_token_ids or ()))
    # Only the field's range: in-process does not check stop token ids against the vocabulary.
    for token_id in stop_token_ids:
        if not 0 <= token_id < _UINT32_LIMIT:
            raise ValueError(f"stop token id {token_id} is out of range: it must be in [0, 2**32)")
    include = bool(sampling_params.include_stop_str_in_output)
    engine_stops = capabilities.supports(CAP_STOP_TEXT) and (
        not include or capabilities.supports_stop_in_output
    )
    stop_texts = stops if engine_stops else []
    if stops and not engine_stops and not sampling_params.detokenize:
        # The fallback matches stop strings on the text the serving side decodes.
        raise ValueError(
            "stop strings are not supported when serving through the OpenEngine engine "
            f"{engine_name!r} without detokenizing (detokenize=False, or a harmony model)"
        )
    include_stop: bool | None = None
    if include and (stop_texts or stop_token_ids):
        if not capabilities.supports_stop_in_output:
            # Only stop token ids are left here: stop strings fell back above.
            raise _not_supported(
                "include_stop_str_in_output", True, engine_name, "with stop_token_ids"
            )
        include_stop = True
    return stop_texts, stop_token_ids, include_stop


def _logprobs_top_n(
    name: str, value: int | None, max_top_n: int | None, engine_name: str
) -> int | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    if max_top_n is None:
        raise _not_supported(name, value, engine_name)
    if value > max_top_n:
        raise ValueError(
            f"{name}={value} is more than the {max_top_n} candidates the OpenEngine engine "
            f"{engine_name!r} returns"
        )
    return value


def _guided(
    sampling_params: SamplingParams, capabilities: ModelCapabilities, engine_name: str
) -> GuidedSpec | None:
    params = sampling_params.guided_decoding
    if params is None:
        return None
    # An empty schema or pattern is still a choice; json_object=False is not.
    chosen = [
        (name, value)
        for name in _GUIDED_MODES
        if (value := getattr(params, name)) is not None and value is not False
    ]
    if not chosen:
        return None
    if len(chosen) > 1:
        raise ValueError(f"only one guided decoding mode can be used, got {len(chosen)}")
    name, value = chosen[0]
    mode = _GUIDED_MODES[name]
    if mode not in capabilities.guided_modes:
        raise _not_supported(f"guided_decoding.{name}", value, engine_name)
    if mode == "json_object":
        return GuidedSpec(mode)
    if mode == "json_schema" and not isinstance(value, str):
        schema = value.model_json_schema() if hasattr(value, "model_json_schema") else value
        value = json.dumps(schema)
    return GuidedSpec(mode, value)


def _thinking_budget(
    sampling_params: SamplingParams,
    capabilities: ModelCapabilities,
    options: GenerateOptions,
    engine_name: str,
) -> dict[str, Any] | None:
    """EX-REQ-3, with the start and end ids computed as ``add_thinking_budget_logits_processor`` does."""
    budget = sampling_params.thinking_token_budget
    if budget is None:
        return None
    if not capabilities.supports(CAP_REQ_THINKING_TOKEN_BUDGET):
        raise _not_supported("thinking_token_budget", budget, engine_name)
    # The same messages as the in-process path.
    if options.reasoning_parser is None:
        raise ValueError("thinking_token_budget requires a configured reasoning_parser")
    if options.tokenizer is None:
        raise ValueError("thinking_token_budget requires a tokenizer")
    from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
    from tensorrt_llm.llmapi.thinking_budget import _encode_token_ids, _get_reasoning_boundary

    parser = ReasoningParserFactory.create_reasoning_parser(options.reasoning_parser, None)
    start = _encode_token_ids(options.tokenizer, _get_reasoning_boundary(parser, "start"))
    end = _encode_token_ids(options.tokenizer, _get_reasoning_boundary(parser, "end"))
    if not start or not end:
        raise ValueError(
            f"reasoning parser {options.reasoning_parser!r} has an empty reasoning start or end "
            "after tokenization"
        )
    return {"budget": budget, "start_token_ids": list(start), "end_token_ids": list(end)}


def _logit_bias(
    logit_bias: Mapping[Any, Any], capabilities: ModelCapabilities, engine_name: str
) -> dict[str, float]:
    """EX-REQ-4: canonical decimal token ids, biases in range, at most the conventions' count."""
    if not capabilities.supports(CAP_REQ_LOGIT_BIAS):
        raise _not_supported("logit_bias", dict(logit_bias), engine_name)
    # OpenAIServer has checked the token ids against the model's vocabulary, as in-process.
    out: dict[str, float] = {}
    for key, bias in logit_bias.items():
        try:
            token_id = int(key)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid logit_bias key {key!r}: must be a valid integer token ID"
            ) from None
        if not 0 <= token_id < _UINT32_LIMIT:
            raise ValueError(f"Token ID {token_id} out of range [0, 2**32)")
        if isinstance(bias, bool) or not isinstance(bias, (int, float)):
            raise ValueError(f"logit_bias value for token ID {token_id} must be a number")
        bias = float(bias)
        if not math.isfinite(bias) or not LOGIT_BIAS_MIN <= bias <= LOGIT_BIAS_MAX:
            raise ValueError(
                f"logit_bias value for token ID {token_id} must be in "
                f"[{LOGIT_BIAS_MIN:g}, {LOGIT_BIAS_MAX:g}]"
            )
        out[str(token_id)] = bias
    # Counted once spellings such as "7" and "07" have become one id.
    if len(out) > LOGIT_BIAS_MAX_ENTRIES:
        raise ValueError(
            f"logit_bias has {len(out)} entries; at most {LOGIT_BIAS_MAX_ENTRIES} are supported "
            f"when serving through the OpenEngine engine {engine_name!r}"
        )
    return out


def _scheduling(scheduling_params: Any, engine_name: str) -> dict[str, Any] | None:
    """EX-REQ-6: the parts of ``scheduling_params`` other than ``attention_dp_rank``."""
    if scheduling_params is None:
        return None
    affinity = getattr(scheduling_params, "subagent_affinity_id", None)
    if affinity is not None:
        raise _not_supported("scheduling_params.subagent_affinity_id", affinity, engine_name)
    payload: dict[str, Any] = {}
    if getattr(scheduling_params, "attention_dp_relax", True) is False:
        payload["attention_dp_relax"] = False
    hierarchy = getattr(scheduling_params, "agent_hierarchy", None)
    if hierarchy:
        payload["agent_hierarchy"] = [[str(kind), int(node)] for kind, node in hierarchy]
    return payload or None


def _extra(
    sampling_params: SamplingParams,
    capabilities: ModelCapabilities,
    options: GenerateOptions,
    engine_name: str,
) -> dict[str, Any]:
    """``GenerateRequest.extra``: only keys the engine declared (EX-GEN-1)."""
    extra: dict[str, Any] = {REQ_DETOKENIZE: False}

    def put(key: str, capability: str, name: str, value: Any) -> None:
        if not capabilities.supports(capability):
            raise _not_supported(name, value, engine_name)
        _check_struct_ints(name, value, engine_name)
        extra[key] = value

    budget = _thinking_budget(sampling_params, capabilities, options, engine_name)
    if budget is not None:
        _check_struct_ints("thinking_token_budget", budget, engine_name)
        extra[REQ_THINKING_TOKEN_BUDGET] = budget
    if options.logit_bias:
        extra[REQ_LOGIT_BIAS] = _logit_bias(options.logit_bias, capabilities, engine_name)
    if options.kv_hints is not None:
        hints = options.kv_hints
        if hasattr(hints, "model_dump"):
            hints = hints.model_dump(mode="json")
        put(REQ_KV_HINTS, CAP_REQ_KV_HINTS, "kv_hints", hints)
    if options.conversation_params is not None and capabilities.supports(CAP_REQ_CONVERSATION_ID):
        # A routing hint, also taken from gateway headers; in-process ignores it
        # too where conversation routing is off.
        conversation_id = options.conversation_params.conversation_id
        put(REQ_CONVERSATION_ID, CAP_REQ_CONVERSATION_ID, "conversation_id", conversation_id)
    scheduling = _scheduling(options.scheduling_params, engine_name)
    if scheduling is not None:
        put(REQ_SCHEDULING, CAP_REQ_SCHEDULING, "scheduling_params", scheduling)
    return extra


def _check_options(
    options: GenerateOptions, capabilities: ModelCapabilities, engine_name: str
) -> None:
    for f in dataclasses.fields(GenerateOptions):
        field_class = GENERATE_OPTIONS.get(f.name)
        if field_class is not None and field_class.kind is not Kind.REJECTED:
            continue
        value = getattr(options, f.name)
        if value is not None and value != f.default:
            why = field_class.target if field_class is not None else "an unclassified option"
            raise _not_supported(f.name, value, engine_name, why)
    if options.lora_name is not None and not capabilities.supports_lora:
        raise _not_supported("lora_request", options.lora_name, engine_name)
    if options.lora_request is not None and options.lora_name is None:
        raise ValueError(
            "lora_request is not supported when serving through the OpenEngine engine "
            f"{engine_name!r}: the adapter is not registered with the engine"
        )
    if options.cache_salt is not None and not capabilities.supports_cache_salt:
        raise _not_supported("cache_salt", options.cache_salt, engine_name)


def map_request(
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    capabilities: ModelCapabilities,
    options: GenerateOptions,
    *,
    engine_name: str,
) -> list[EngineRequest]:
    """Map one request to the Generate calls that serve it, in output order.

    ``n`` up to the engine's ``max_num_sequences`` (M) is one call; above that
    the request is split into ceil(n / M) calls of M sequences, the last one
    taking the rest, and only the first call asks for prompt logprobs. TRT-LLM
    seeds sequence i of a request with its seed (0 when unset) plus i, so each
    call of a split request is sent that seed plus the number of sequences
    before it: the n sequences get the seeds one in-process request gives them.

    ``priority``, ``trace_headers`` and ``scheduling_params.attention_dp_rank``
    become metadata, which the caller adds (OE-META).

    Raises:
        ValueError: The request asks for something the engine cannot do, or
            the prompt does not fit; reported as HTTP 400.
    """
    log_sequence_capability(engine_name, capabilities)
    _check_rejected(sampling_params, engine_name)
    _check_options(options, capabilities, engine_name)
    _check_prompt(prompt_token_ids, capabilities)

    sampling = _sampling(sampling_params, capabilities, engine_name)
    max_tokens = _max_tokens(sampling_params, capabilities, len(prompt_token_ids), engine_name)
    stop_texts, stop_token_ids, include_stop = _stopping(sampling_params, capabilities, engine_name)
    output_top_n = _logprobs_top_n(
        "logprobs", sampling_params.logprobs, capabilities.output_logprobs_max_top_n, engine_name
    )
    prompt_top_n = _logprobs_top_n(
        "prompt_logprobs",
        sampling_params.prompt_logprobs,
        capabilities.prompt_logprobs_max_top_n,
        engine_name,
    )
    guided = _guided(sampling_params, capabilities, engine_name)
    extra = _extra(sampling_params, capabilities, options, engine_name)

    n = sampling_params.n
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not 0 <= (sampling_params.min_tokens or 0) < _UINT32_LIMIT:
        raise ValueError(f"min_tokens must be in [0, 2**32), got {sampling_params.min_tokens}")
    limit = max(1, capabilities.max_num_sequences)
    sizes = [limit] * (n // limit) + ([n % limit] if n % limit else [])
    best_of = sampling_params.best_of
    if best_of is not None and best_of != n:
        if len(sizes) > 1:
            raise _not_supported("best_of", best_of, engine_name, f"n={n} is split into calls")
        if not capabilities.supports(CAP_REQ_BEST_OF):
            raise _not_supported("best_of", best_of, engine_name)
        if best_of > limit:
            raise _not_supported(
                "best_of", best_of, engine_name, f"one Generate call returns at most {limit}"
            )
        extra[REQ_BEST_OF] = best_of
    if len(sizes) > 1:
        logger.debug(f"n={n} is served by {len(sizes)} Generate calls of at most {limit}")

    if len(sizes) > 1:
        sampling.setdefault("seed", 0)
    requests = []
    offset = 0
    for k, size in enumerate(sizes):
        call_sampling = dict(sampling)
        if offset:
            call_sampling["seed"] = (int(sampling["seed"]) + offset) % _SEED_LIMIT
        offset += size
        requests.append(
            EngineRequest(
                request_id=new_request_id(),
                prompt_token_ids=list(prompt_token_ids),
                max_tokens=max_tokens,
                sampling=call_sampling,
                min_tokens=sampling_params.min_tokens or None,
                stop_token_ids=list(stop_token_ids),
                stop_texts=list(stop_texts),
                ignore_eos=True if sampling_params.ignore_eos else None,
                include_stop_in_output=include_stop,
                num_sequences=size,
                output_logprobs_top_n=output_top_n,
                prompt_logprobs_top_n=prompt_top_n if k == 0 else None,
                guided=guided,
                cache_salt=options.cache_salt,
                media=list(options.media),
                lora_name=options.lora_name,
                model=capabilities.served_model_name,
                extra=dict(extra),
            )
        )
    return requests


# --- PoC-S1 adapter ------------------------------------------------------------

# ModelInfo.extra keys that declare sampling support (TRT-LLM OpenEngine
# conventions). A missing key means "not supported".
CAPABILITY_KEY_PREFIX = "trtllm.sampling."
_CAPABILITY_GATED = ("min_p", "frequency_penalty", "presence_penalty", "repetition_penalty")
# TRT-LLM's own OpenEngine server supports every gated field but reports no
# capability keys; without a conventions version it counts as declaring them.
# No other engine gets a built-in table.
CONVENTIONS_VERSION_KEY = "trtllm.conventions_version"
_BUILTIN_CAPABILITIES = {"tensorrt_llm": frozenset(_CAPABILITY_GATED)}

# Values that leave sampling unchanged. OpenAIServer always fills these fields,
# so a neutral value is treated as "not set" and not sent.
_S1_NEUTRAL = {
    "top_k": 0,
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}

# SamplingParams fields this adapter cannot carry to the engine; each must be
# unset or equal to the listed value.
_S1_UNSUPPORTED_DEFAULTS: dict[str, Any] = {
    "use_beam_search": False,
    "logprobs": None,
    "prompt_logprobs": None,
    "guided_decoding": None,
    # Checked before logits_processor: OpenAIServer turns a thinking budget
    # into a logits processor, and the budget is what the request asked for.
    "thinking_token_budget": None,
    "logits_processor": None,
    "embedding_bias": None,
    "bad": None,
    "bad_token_ids": None,
    "additional_model_outputs": None,
    "return_context_logits": False,
    "return_generation_logits": False,
    "return_routed_experts": False,
    "truncate_prompt_tokens": None,
    "length_penalty": None,
    "early_stopping": None,
    "no_repeat_ngram_size": None,
    "top_p_min": None,
    "top_p_decay": None,
    "top_p_reset_ids": None,
    "beam_search_diversity_rate": None,
    "beam_width_array": None,
}


def _s1_unsupported(field: str, value: Any, engine: str) -> ValueError:
    return ValueError(
        f"{field}={value!r} is not supported when serving through the OpenEngine engine {engine!r}"
    )


def _s1_requested(sampling_params: SamplingParams, field: str) -> Any:
    """Return the value to send for ``field``.

    None means "do not send": the value is unset, neutral (even when the request
    set it explicitly), or a value ``OpenAIServer`` filled in although the request
    did not set it.
    """
    value = getattr(sampling_params, field)
    if value is None or _S1_NEUTRAL.get(field, object()) == value:
        return None
    provided = sampling_params._request_provided_fields
    if (
        provided is not None
        and field in _GENERATION_CONFIG_SAMPLING_FIELDS
        and field not in provided
    ):
        return None
    return value


def builtin_capabilities(server_info: server_pb2.ServerInfo) -> frozenset[str]:
    """Gated fields to treat as declared for an engine that reports no capability keys."""
    if server_info.HasField("extra") and CONVENTIONS_VERSION_KEY in server_info.extra.fields:
        return frozenset()
    return _BUILTIN_CAPABILITIES.get(server_info.engine_name, frozenset())


def _declared(model_info: model_pb2.ModelInfo, field: str) -> bool:
    key = CAPABILITY_KEY_PREFIX + field
    if not model_info.HasField("extra") or key not in model_info.extra.fields:
        return False
    value = model_info.extra.fields[key]
    return value.WhichOneof("kind") == "bool_value" and value.bool_value


def default_max_tokens(model_info: model_pb2.ModelInfo, prompt_len: int) -> int | None:
    """Most tokens the engine can generate after ``prompt_len`` prompt tokens; None if it sets no limit."""
    limits = []
    if model_info.HasField("max_context_length"):
        limits.append(model_info.max_context_length - prompt_len)
    if model_info.HasField("max_output_tokens"):
        limits.append(model_info.max_output_tokens)
    return min(limits) if limits else None


def build_generate_request(
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    model_info: model_pb2.ModelInfo,
    *,
    engine_name: str,
    builtin: Collection[str] = (),
) -> generation_pb2.GenerateRequest:
    """Build the ``GenerateRequest`` for one prompt.

    Args:
        prompt_token_ids: The tokenized prompt.
        sampling_params: The parameters ``OpenAIServer`` built for the request.
        model_info: The engine's ``ModelInfo``.
        engine_name: The engine's name, used in error messages.
        builtin: Gated fields to treat as declared; see ``builtin_capabilities``.

    Raises:
        ValueError: A parameter is not supported, or the prompt does not fit.
    """
    for field, default in _S1_UNSUPPORTED_DEFAULTS.items():
        value = getattr(sampling_params, field)
        if default is None:
            unset = value is None or (isinstance(value, (list, tuple)) and not value)
        else:
            unset = value == default
        if not unset:
            if field in ("logits_processor", "embedding_bias", "guided_decoding"):
                # OpenAI requests set embedding_bias through logit_bias.
                name = "logit_bias (embedding_bias)" if field == "embedding_bias" else field
                raise ValueError(
                    f"{name} is not supported when serving through the OpenEngine engine "
                    f"{engine_name!r}"
                )
            raise _s1_unsupported(field, value, engine_name)
    if sampling_params.prompt_ignore_length:
        raise _s1_unsupported(
            "prompt_ignore_length", sampling_params.prompt_ignore_length, engine_name
        )

    n = sampling_params.n
    if sampling_params.best_of is not None and sampling_params.best_of != n:
        raise _s1_unsupported("best_of", sampling_params.best_of, engine_name)
    generation = model_info.generation
    max_n = generation.max_num_sequences if generation.HasField("max_num_sequences") else 1
    if n > max_n:
        raise ValueError(f"n={n} exceeds the {max_n} sequence(s) engine {engine_name!r} supports")

    if (
        model_info.HasField("max_context_length")
        and len(prompt_token_ids) >= model_info.max_context_length
    ):
        # The LLM path's wording, so OpenAIServer reports code context_length_exceeded.
        raise ValueError(
            context_length_exceeded_message(
                max_context_length=model_info.max_context_length,
                num_prompt_tokens=len(prompt_token_ids),
            )
        )

    sampling = generation_pb2.SamplingParams()
    for field in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
    ):
        value = _s1_requested(sampling_params, field)
        if value is None:
            continue
        if field in _CAPABILITY_GATED and field not in builtin and not _declared(model_info, field):
            raise _s1_unsupported(field, value, engine_name)
        setattr(sampling, field, value)
    if sampling_params.seed is not None:
        if sampling_params.seed < 0:
            raise ValueError(f"seed must be non-negative, got {sampling_params.seed}")
        sampling.seed = sampling_params.seed
    if n > 1:
        sampling.num_sequences = n

    stopping = generation_pb2.StoppingOptions()
    max_tokens = sampling_params.max_tokens
    limit = default_max_tokens(model_info, len(prompt_token_ids))
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError(f"max_tokens must be greater than 0, got {max_tokens}")
    if max_tokens is None:
        max_tokens = limit
    elif limit is not None and max_tokens > limit:
        # As LLM does: cap the request at what the engine can still generate.
        logger.warning(
            f"max_tokens={max_tokens} is more than the {limit} tokens engine {engine_name!r} "
            "can generate for this prompt; using it instead"
        )
        max_tokens = limit
    if max_tokens is not None:
        stopping.max_tokens = max_tokens
    if sampling_params.min_tokens:
        stopping.min_tokens = sampling_params.min_tokens
    if sampling_params.ignore_eos:
        stopping.ignore_eos = True
    for token_id in dict.fromkeys(sampling_params.stop_token_ids or ()):
        stopping.conditions.add(stop_token_id=token_id)

    extra = struct_pb2.Struct()
    extra.update({"detokenize": False})
    return generation_pb2.GenerateRequest(
        model=model_info.served_model_name,
        token_ids=generation_pb2.TokenIds(ids=prompt_token_ids),
        sampling=sampling,
        stopping=stopping,
        extra=extra,
    )

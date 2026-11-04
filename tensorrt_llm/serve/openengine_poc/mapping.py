# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map a ``SamplingParams`` built by ``OpenAIServer`` to an OpenEngine ``GenerateRequest``.

The prompt reaches the engine as token ids: stop strings, detokenization and
the special-token options stay on the serving side. A field that differs from
its default in ``_UNSUPPORTED_DEFAULTS``, or a capability-gated one (``min_p``
and the penalties) about to be sent to an engine that has not declared it
(see ``builtin_capabilities``), is rejected with ``ValueError`` before
anything is sent, which ``OpenAIServer`` returns as HTTP 400.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from google.protobuf import struct_pb2

from tensorrt_llm.executor.utils import context_length_exceeded_message
from tensorrt_llm.grpc.openengine.bindings import generation_pb2, model_pb2, server_pb2
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import _GENERATION_CONFIG_SAMPLING_FIELDS, SamplingParams

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
_NEUTRAL = {
    "top_k": 0,
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}

# SamplingParams fields this adapter cannot carry to the engine; each must be
# unset or equal to the listed value.
_UNSUPPORTED_DEFAULTS: dict[str, Any] = {
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


def _unsupported(field: str, value: Any, engine: str) -> ValueError:
    return ValueError(
        f"{field}={value!r} is not supported when serving through the OpenEngine engine {engine!r}"
    )


def _requested(sampling_params: SamplingParams, field: str) -> Any:
    """Return the value to send for ``field``.

    None means "do not send": the value is unset, neutral (even when the request
    set it explicitly), or a value ``OpenAIServer`` filled in although the request
    did not set it.
    """
    value = getattr(sampling_params, field)
    if value is None or _NEUTRAL.get(field, object()) == value:
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
    for field, default in _UNSUPPORTED_DEFAULTS.items():
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
            raise _unsupported(field, value, engine_name)
    if sampling_params.prompt_ignore_length:
        raise _unsupported(
            "prompt_ignore_length", sampling_params.prompt_ignore_length, engine_name
        )

    n = sampling_params.n
    if sampling_params.best_of is not None and sampling_params.best_of != n:
        raise _unsupported("best_of", sampling_params.best_of, engine_name)
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
        value = _requested(sampling_params, field)
        if value is None:
            continue
        if field in _CAPABILITY_GATED and field not in builtin and not _declared(model_info, field):
            raise _unsupported(field, value, engine_name)
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

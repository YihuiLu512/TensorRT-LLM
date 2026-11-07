# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where every request field goes when serving through an OpenEngine engine.

Each field of ``SamplingParams``, of :class:`~.contracts.GenerateOptions` and of
the OpenAI request models is in exactly one of these tables, and has one of
these kinds:

- ``MAPPED``: it reaches the engine, at ``target``. When the engine has not
  declared ``capability``, a set value is rejected, unless ``fallback`` names
  what happens to it instead.
- ``SERVE``: the serving side handles it; ``verified_by`` names the feature
  row (``row F-xx``) whose tests verify that handling, or says why none is
  needed (``none needed: ...``).
- ``REJECTED``: a set value is rejected with HTTP 400.
- ``INTERNAL``: not a request option (state derived from other fields, or an
  input of the mapping itself).

:mod:`.mapping` rejects a ``SamplingParams`` or ``GenerateOptions`` field it
finds set that is ``REJECTED`` here, or that is in neither table. A
request-model field names the ``SamplingParams`` field or the
``GenerateOptions`` field it becomes, or is handled on the serving side. The
no-silent-drop test checks that the tables cover every field, so a field added
later fails that test until it is classified here.

This module imports nothing from TensorRT-LLM, so tests and tools can read it
without loading the package.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Kind(str, enum.Enum):
    MAPPED = "mapped"
    SERVE = "serve"
    REJECTED = "rejected"
    INTERNAL = "internal"


@dataclass(frozen=True)
class FieldClass:
    """How one field is handled.

    Attributes:
        target: ``MAPPED``: where it lands, as a ``GenerateRequest`` path, an
            ``extra`` key or a metadata key. Otherwise what handles it or why.
        capability: ``MAPPED``: what the engine must declare; empty when every
            engine supports it.
        fallback: ``MAPPED``: what happens to a set value when the capability
            is missing; empty means the field is then rejected.
        verified_by: ``SERVE``: the feature row whose tests verify the handling.
    """

    kind: Kind
    target: str
    capability: str = ""
    fallback: str = ""
    verified_by: str = ""


def _mapped(target: str, capability: str = "", fallback: str = "") -> FieldClass:
    return FieldClass(Kind.MAPPED, target, capability, fallback)


def _serve(target: str, verified_by: str) -> FieldClass:
    return FieldClass(Kind.SERVE, target, verified_by=verified_by)


def _rejected(why: str) -> FieldClass:
    return FieldClass(Kind.REJECTED, why)


def _internal(why: str) -> FieldClass:
    return FieldClass(Kind.INTERNAL, why)


_NO_CARRIER = "OpenEngine v1 has no field for it"
_BEAM = "OpenEngine has no beam search"
_DETOK = "the response assembler (SV-DTK)"
_DETOK_TEST = "row F-24"
_FACADE_META = "the facade adds it after map_request and checks the capability (OE-META)"

# Every field of SamplingParams, including the private ones.
SAMPLING_PARAMS: dict[str, FieldClass] = {
    "end_id": _rejected(f"replaces the EOS id; {_NO_CARRIER}"),
    "pad_id": _rejected(_NO_CARRIER),
    "max_tokens": _mapped("stopping.max_tokens"),
    "bad": _rejected(_NO_CARRIER),
    "bad_token_ids": _rejected(_NO_CARRIER),
    "_bad_word_ids": _internal("derived from bad and bad_token_ids"),
    "stop": _mapped(
        "stopping.conditions[].stop_text",
        "trtllm.stopping.stop_text",
        fallback="the serving-side detokenizer matches the stop strings (SV-RSP-3)",
    ),
    "stop_token_ids": _mapped("stopping.conditions[].stop_token_id"),
    "include_stop_str_in_output": _mapped(
        "stopping.include_stop_in_output",
        "supports_stop_in_output",
        fallback="stop strings are matched on the serving side, which keeps them",
    ),
    "_stop_word_ids": _internal("derived from stop and stop_token_ids"),
    "embedding_bias": _rejected("a bias tensor; logit_bias is sent instead"),
    "logits_processor": _rejected("a callable; thinking_token_budget is sent instead"),
    "apply_batched_logits_processor": _rejected(_NO_CARRIER),
    "n": _mapped("sampling.num_sequences, or ceil(n / max_num_sequences) calls (SV-MAP-6)"),
    "best_of": _mapped("extra[trtllm.best_of]", "trtllm.request.best_of"),
    "use_beam_search": _rejected(_BEAM),
    "logprobs_mode": _rejected(f"only raw logprobs; {_NO_CARRIER}"),
    "top_k": _mapped("sampling.top_k"),
    "top_p": _mapped("sampling.top_p"),
    "top_p_min": _rejected(_NO_CARRIER),
    "top_p_reset_ids": _rejected(_NO_CARRIER),
    "top_p_decay": _rejected(_NO_CARRIER),
    "seed": _mapped("sampling.seed"),
    "temperature": _mapped("sampling.temperature"),
    "min_tokens": _mapped("stopping.min_tokens"),
    "beam_search_diversity_rate": _rejected(_BEAM),
    "repetition_penalty": _mapped(
        "sampling.repetition_penalty", "trtllm.sampling.repetition_penalty"
    ),
    "presence_penalty": _mapped("sampling.presence_penalty", "trtllm.sampling.presence_penalty"),
    "frequency_penalty": _mapped("sampling.frequency_penalty", "trtllm.sampling.frequency_penalty"),
    "prompt_ignore_length": _rejected(_NO_CARRIER),
    "length_penalty": _rejected(_BEAM),
    "early_stopping": _rejected(_BEAM),
    "no_repeat_ngram_size": _rejected(_NO_CARRIER),
    "min_p": _mapped("sampling.min_p", "trtllm.sampling.min_p"),
    "beam_width_array": _rejected(_BEAM),
    "logprobs": _mapped("response.output_candidates.top_n", "output logprobs"),
    "prompt_logprobs": _mapped("response.prompt_candidates.top_n", "prompt logprobs"),
    "logprobs_simple_format": _rejected("the response assembler builds no simple format"),
    "prompt_logprobs_simple_format": _rejected("the response assembler builds no simple format"),
    "return_context_logits": _rejected(f"a tensor; {_NO_CARRIER}"),
    "return_generation_logits": _rejected(f"a tensor; {_NO_CARRIER}"),
    "exclude_input_from_output": _rejected(f"False echoes the prompt ids; {_NO_CARRIER}"),
    "return_encoder_output": _rejected(f"a tensor; {_NO_CARRIER}"),
    "return_perf_metrics": _serve("the frontend's metrics recorder (SV-MET)", "row F-42: T11, T12"),
    "return_routed_experts": _rejected(f"a tensor; {_NO_CARRIER}"),
    "additional_model_outputs": _rejected(f"tensors; {_NO_CARRIER}"),
    "_decoder_output_token_prefix": _rejected("encoder-decoder models only"),
    "guided_decoding": _mapped("guided", "the guided decoding mode"),
    "thinking_token_budget": _mapped(
        "extra[trtllm.thinking_token_budget]", "trtllm.request.thinking_token_budget"
    ),
    "ignore_eos": _mapped("stopping.ignore_eos"),
    "detokenize": _serve(_DETOK, _DETOK_TEST),
    "add_special_tokens": _serve("the facade's input processor, as in-process", _DETOK_TEST),
    "truncate_prompt_tokens": _serve(
        "the facade's input processor truncates while tokenizing, as in-process",
        "row F-09: the truncate_prompt_tokens case of T12",
    ),
    "skip_special_tokens": _serve(_DETOK, _DETOK_TEST),
    "spaces_between_special_tokens": _serve(_DETOK, _DETOK_TEST),
    "_stream_interval": _internal("the facade streams each engine event as it arrives"),
    "_request_provided_fields": _internal("tells explicit values from server-filled ones"),
}

# Every field of GenerateOptions.
GENERATE_OPTIONS: dict[str, FieldClass] = {
    "streaming": _serve("every Generate call streams; the handler decides", "row F-01+F-02"),
    "lora_request": _mapped("lora_name, once the LoRA registry resolves it", "LoRA"),
    "lora_name": _mapped("lora_name", "LoRA"),
    "disaggregated_params": _rejected("this frontend does not map disaggregated requests"),
    "cache_salt": _mapped("kv.cache_salt", "supports_cache_salt"),
    "kv_hints": _mapped("extra[trtllm.kv_hints]", "trtllm.request.kv_hints"),
    "conversation_params": _mapped(
        "extra[conversation_id]",
        "trtllm.request.conversation_id",
        fallback="not sent: a routing and cache-retention hint that does not change the output",
    ),
    "scheduling_params": _mapped(
        "extra[trtllm.scheduling]; attention_dp_rank as metadata openengine-target-dp-rank, "
        + _FACADE_META,
        "trtllm.request.scheduling",
    ),
    "priority": _mapped("metadata openengine-priority; " + _FACADE_META, "supports_priority"),
    "trace_headers": _mapped("metadata traceparent, tracestate; " + _FACADE_META),
    "media": _mapped("media, from map_media", "the media modality"),
    "logit_bias": _mapped("extra[trtllm.logit_bias]", "trtllm.request.logit_bias"),
    "reasoning_parser": _internal("input of the thinking-budget mapping (SV-MAP-5)"),
    "tokenizer": _internal("input of the thinking-budget mapping (SV-MAP-5)"),
}


def _sp(name: str) -> str:
    return f"sampling_params.{name}"


def _opt(name: str) -> str:
    return f"options.{name}"


_RENDER = "rendered and tokenized by the serving side"
_IGNORED = "not used to generate, as in-process"
_TOOLS = "the chat template and the tool parser"

# Request-model fields that become SamplingParams or GenerateOptions fields of
# the same name.
_SAME_NAME_SP = (
    "best_of",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stop",
    "temperature",
    "top_p",
    "n",
    "prompt_ignore_length",
    "use_beam_search",
    "top_k",
    "top_p_min",
    "min_p",
    "repetition_penalty",
    "length_penalty",
    "early_stopping",
    "stop_token_ids",
    "include_stop_str_in_output",
    "ignore_eos",
    "min_tokens",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "truncate_prompt_tokens",
    "thinking_token_budget",
    "add_special_tokens",
)
_SAME_NAME_OPT = (
    "kv_hints",
    "lora_request",
    "disaggregated_params",
    "priority",
    "conversation_params",
    "logit_bias",
)

# Where each field of a request model goes: "sampling_params.<field>",
# "options.<field>", or a FieldClass of kind SERVE.
Route = str | FieldClass

_COMMON: dict[str, Route] = {
    **{name: _sp(name) for name in _SAME_NAME_SP},
    **{name: _opt(name) for name in _SAME_NAME_OPT},
    "model": _serve("checked against the served model name", "row F-01+F-02"),
    "stream": _opt("streaming"),
    "stream_options": _serve("the handler's usage chunks", "row F-17+F-18+F-30"),
    "user": _serve(_IGNORED, "none needed: in-process ignores it too"),
    "response_format": _sp("guided_decoding"),
    "echo": _serve("the handler echoes the prompt", _DETOK_TEST),
}

REQUEST_MODELS: dict[str, dict[str, Route]] = {
    "CompletionRequest": {
        **_COMMON,
        "prompt": _serve(_RENDER, "row F-01+F-02; row F-08 for token ids"),
        "logprobs": _sp("logprobs"),
        "max_tokens": _sp("max_tokens"),
        "suffix": _serve(_IGNORED, "none needed: in-process ignores it too"),
        "return_context_logits": _sp("return_context_logits"),
        "detokenize": _sp("detokenize"),
    },
    "ChatCompletionRequest": {
        **_COMMON,
        "messages": _serve(_RENDER, "row F-01+F-02"),
        "prompt_token_ids": _serve("token-id input", "row F-08: T8, T27"),
        "prompt_token_ids_b64": _serve("token-id input", "row F-08: T8, T27"),
        "logprobs": _sp("logprobs"),
        "top_logprobs": _sp("logprobs"),
        "max_completion_tokens": _sp("max_tokens"),
        "service_tier": _serve(_IGNORED, "none needed: in-process ignores it too"),
        "store": _serve(_IGNORED, "none needed: in-process ignores it too"),
        "tools": _serve(_TOOLS, "row F-27+F-28: T8"),
        "tool_choice": _serve(_TOOLS + "; a forced choice adds guided decoding", "row F-27+F-28"),
        "parallel_tool_calls": _serve(_TOOLS, "row F-27+F-28"),
        "reasoning_effort": _serve("the chat template", "row F-07"),
        "thinking": _serve("the chat template", "row F-27+F-28: T7"),
        "client_metadata": _serve(_IGNORED, "none needed: in-process ignores it too"),
        "add_generation_prompt": _serve("the chat template", "row F-07"),
        "documents": _serve("the chat template", "row F-07"),
        "chat_template": _serve("the chat template", "row F-07"),
        "chat_template_kwargs": _serve("the chat template", "row F-07"),
        "injected_chat_template_kwargs": _serve("the chat template", "row F-07"),
        "media_io_kwargs": _serve("media loading", "row F-10: T14, T15"),
        "mm_processor_kwargs": _serve("multimodal processing", "row F-10: T14, T15"),
        "agent_hierarchy": _opt("scheduling_params"),
        "cache_salt": _opt("cache_salt"),
    },
    "ResponsesRequest": {
        **{
            name: _COMMON[name]
            for name in (
                "kv_hints",
                "model",
                "stream",
                "temperature",
                "top_p",
                "user",
                "thinking_token_budget",
                "disaggregated_params",
                "conversation_params",
            )
        },
        "background": _serve("the Responses store", "row F-03+F-04: T29"),
        "include": _serve("the Responses output builder", "row F-03+F-04: T29"),
        "input": _serve(_RENDER, "row F-03+F-04: T29"),
        "instructions": _serve(_RENDER, "row F-03+F-04: T29"),
        "max_output_tokens": _sp("max_tokens"),
        "max_tool_calls": _serve(_IGNORED, "none needed: in-process ignores it too"),
        "metadata": _serve("echoed in the response", "row F-03+F-04: T29"),
        "parallel_tool_calls": _serve(_TOOLS, "row F-03+F-04: T29"),
        "previous_response_id": _serve("the Responses store", "row F-03+F-04: T29"),
        "prompt": _serve(_IGNORED, "none needed: in-process ignores it too"),
        "reasoning": _serve("the chat template", "row F-03+F-04: T29"),
        "service_tier": _serve(_IGNORED, "none needed: in-process ignores it too"),
        "store": _serve("the Responses store", "row F-03+F-04: T29"),
        "text": _sp("guided_decoding"),
        "tool_choice": _serve(_TOOLS, "row F-03+F-04: T29"),
        "tools": _serve(_TOOLS, "row F-03+F-04: T29"),
        "top_logprobs": _sp("logprobs"),
        "truncation": _serve(_IGNORED, "none needed: in-process ignores it too"),
        "request_id": _serve("names the response", "row F-03+F-04: T29"),
        "prompt_token_ids": _serve("token-id input", "row F-08"),
        "prompt_token_ids_b64": _serve("token-id input", "row F-08"),
    },
}

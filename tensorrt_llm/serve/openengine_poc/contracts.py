# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pluggable parts of serving through an OpenEngine engine, with their default implementations.

Each part is a *slot*: a function with a fixed signature, looked up in a fixed
module. When that module exists and defines the function, the slot calls it;
otherwise the slot calls the default implementation in this file. So a part is
replaced by adding its module, without touching this file or the code that
calls the slot.

| Slot | Implementation looked up in |
|---|---|
| :func:`map_request` | ``openengine_poc.mapping:map_request`` |
| :func:`create_response_assembler` | ``openengine_poc.response:create_response_assembler`` |
| :func:`create_postproc_dispatch` | ``openengine_poc.postproc_pool:create_postproc_dispatch`` |
| :func:`create_postproc_pool` | ``openengine_poc.postproc_pool:create_postproc_pool`` |
| :func:`launch_frontends` | ``openengine_poc.multi_frontend:launch_frontends`` |
| :func:`create_metrics_recorder` | ``openengine_poc.metrics:create_metrics_recorder`` |
| :func:`map_media` | ``openengine_poc.media:map_media`` |
| :func:`create_lora_registry` | ``openengine_poc.lora:create_lora_registry`` |

The default implementation is used only when the module is missing (and not
when one of its own imports is) or does not define the function; any other
import error propagates. A slot logs where its implementation came from the
first time it resolves it. A slot module is imported while the slot is looked
up, not while this module is, so it may call other slots at import time; a slot
called while its own module is still being imported, before that module defines
the slot's function, uses the default for that call and is looked up again on
the next one.

The module also holds the messages between a frontend and its postprocessing
processes, the plain-data check for what crosses into them, the interface of
those processes, and the frontend's count of Generate calls.

This module is imported by ``OpenAIServer``, so at module level it imports only
the standard library, the logger, :mod:`tensorrt_llm.grpc.openengine_conventions`
and :mod:`.engine_client`, none of which loads protobuf or gRPC.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
import inspect
import json
import os
import sys
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, cast

from tensorrt_llm.grpc.openengine_conventions import (
    REQ_DETOKENIZE,
    SAMPLING_FIELD_CAPABILITY,
    SERVE_COUNTER_DIR_ENV,
    SERVE_COUNTER_GENERATE_SENT,
)
from tensorrt_llm.logger import logger

from .engine_client import (
    EngineClient,
    EngineEvent,
    EngineLoad,
    EngineRequest,
    MediaSpec,
    ModelCapabilities,
    OutputFinished,
    TokenDelta,
    Usage,
)

if TYPE_CHECKING:
    from tensorrt_llm.executor.result import CompletionOutput
    from tensorrt_llm.sampling_params import SamplingParams

# Bump on any change to a signature, type or message in this module or in
# engine_client.py, in a commit of its own.
POC2_CONTRACT_VERSION = 3

_PACKAGE = "tensorrt_llm.serve.openengine_poc"


class OpenEngineUnsupportedError(NotImplementedError):
    """A feature the OpenEngine path does not provide; endpoints report it as 501."""


# --- Slots --------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """Where a slot's implementation is looked up, and the default it falls back to."""

    name: str
    module: str
    attribute: str
    owner: str
    default: Callable[..., Any]


_SLOTS: dict[str, Slot] = {}
_resolved: dict[str, tuple[Callable[..., Any], str]] = {}
_RESOLVE_LOCK = threading.Lock()


def _slot(name: str, module: str, owner: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def register(default: Callable[..., Any]) -> Callable[..., Any]:
        _SLOTS[name] = Slot(name, f"{_PACKAGE}.{module}", name, owner, default)
        return default

    return register


def resolve_slot(name: str) -> tuple[Callable[..., Any], str]:
    """Return a slot's implementation and where it came from (``"module:function"`` or ``"default"``)."""
    with _RESOLVE_LOCK:
        if name in _resolved:
            return _resolved[name]
    slot = _SLOTS[name]
    impl, source = slot.default, "default"
    # Imported without holding the lock, so a slot module may resolve other
    # slots while it is imported; the import system serializes the import itself.
    try:
        module = importlib.import_module(slot.module)
    except ModuleNotFoundError as error:
        if error.name != slot.module:
            raise
    else:
        found = getattr(module, slot.attribute, None)
        # A module that only re-exports this module's public function of the
        # same name has not implemented the slot (calling it would recurse).
        if found is not None and found is not globals().get(slot.name):
            impl, source = found, f"{slot.module}:{slot.attribute}"
        elif getattr(getattr(module, "__spec__", None), "_initializing", False):
            # The slot module is still being imported (it called its own slot
            # at import time): use the default now, and look again next time.
            return impl, source
    with _RESOLVE_LOCK:
        if name in _resolved:
            return _resolved[name]
        logger.info(f"OpenEngine slot {name}: using {source}")
        _resolved[name] = (impl, source)
        return impl, source


def reset_slot_cache() -> None:
    """Forget resolved slots, so the next call looks them up again (for tests)."""
    with _RESOLVE_LOCK:
        _resolved.clear()


def slots() -> Mapping[str, Slot]:
    """Every slot by name."""
    return dict(_SLOTS)


# --- Request mapping ----------------------------------------------------------


@dataclass
class GenerateOptions:
    """The ``generate_async`` arguments besides the prompt and the sampling parameters.

    Attributes:
        media: The prompt's media, from :func:`map_media`.
        lora_name: The registered adapter name, from :class:`LoraRegistry`.
    """

    streaming: bool = False
    lora_request: Any = None
    lora_name: str | None = None
    disaggregated_params: Any = None
    cache_salt: str | None = None
    kv_hints: Any = None
    conversation_params: Any = None
    scheduling_params: Any = None
    priority: float | None = None
    trace_headers: Mapping[str, str] | None = None
    media: Sequence[MediaSpec] = ()


def map_request(
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    capabilities: ModelCapabilities,
    options: GenerateOptions,
    *,
    engine_name: str,
) -> list[EngineRequest]:
    """Map one request to the Generate calls that serve it, in output order.

    Raises:
        ValueError: The request asks for something the engine cannot do, or
            the prompt does not fit; reported as HTTP 400.
    """
    impl, _ = resolve_slot("map_request")
    return cast(
        list[EngineRequest],
        impl(prompt_token_ids, sampling_params, capabilities, options, engine_name=engine_name),
    )


# Values that leave sampling unchanged; OpenAIServer always fills these fields,
# so a neutral value counts as not set.
_NEUTRAL: dict[str, Any] = {
    "top_k": 0,
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}
# SamplingParams fields the default mapping cannot carry; each must be unset
# (None or empty) or equal to the value given.
_UNSUPPORTED_DEFAULTS: dict[str, Any] = {
    "use_beam_search": False,
    "logprobs": None,
    "prompt_logprobs": None,
    "guided_decoding": None,
    # Before logits_processor: OpenAIServer turns a thinking budget into a
    # logits processor, and the budget is what the request asked for.
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
_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
)


def _not_supported(field_name: str, value: Any, engine_name: str) -> ValueError:
    return ValueError(
        f"{field_name}={value!r} is not supported when serving through the OpenEngine engine "
        f"{engine_name!r}"
    )


def _requested(sampling_params: SamplingParams, name: str) -> Any:
    """The value to send for a sampling field; None when it is unset, neutral or a server-filled default."""
    from tensorrt_llm.sampling_params import _GENERATION_CONFIG_SAMPLING_FIELDS

    value = getattr(sampling_params, name)
    if value is None or _NEUTRAL.get(name, object()) == value:
        return None
    provided = sampling_params._request_provided_fields
    if provided is not None and name in _GENERATION_CONFIG_SAMPLING_FIELDS and name not in provided:
        return None
    return value


def new_request_id() -> str:
    """A fresh Generate ``request_id``."""
    return f"trtllm-{uuid.uuid4().hex}"


@_slot("map_request", "mapping", owner="WT-11")
def default_map_request(
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    capabilities: ModelCapabilities,
    options: GenerateOptions,
    *,
    engine_name: str,
) -> list[EngineRequest]:
    """One call, serving-side stop strings, no request options.

    ``priority``, ``scheduling_params``, ``conversation_params`` and
    ``trace_headers`` are accepted and not forwarded; LoRA, disaggregated
    serving, ``cache_salt``, ``kv_hints`` and media are rejected. ``streaming``
    does not change the call, which always streams.
    """
    from tensorrt_llm.executor.utils import context_length_exceeded_message

    for name, value in (
        ("lora_request", options.lora_request),
        ("lora_name", options.lora_name),
        ("disaggregated_params", options.disaggregated_params),
        ("cache_salt", options.cache_salt),
        ("kv_hints", options.kv_hints),
    ):
        if value is not None:
            raise ValueError(f"{name} is not supported when serving through an OpenEngine engine")
    if options.media:
        raise ValueError(
            "multimodal inputs are not supported when serving through an OpenEngine engine"
        )
    for name, default in _UNSUPPORTED_DEFAULTS.items():
        value = getattr(sampling_params, name)
        if default is None:
            unset = value is None or (isinstance(value, (list, tuple)) and not value)
        else:
            unset = value == default
        if unset:
            continue
        if name in ("logits_processor", "embedding_bias", "guided_decoding"):
            # OpenAI requests set embedding_bias through logit_bias.
            label = "logit_bias (embedding_bias)" if name == "embedding_bias" else name
            raise ValueError(
                f"{label} is not supported when serving through the OpenEngine engine "
                f"{engine_name!r}"
            )
        raise _not_supported(name, value, engine_name)
    if sampling_params.prompt_ignore_length:
        raise _not_supported(
            "prompt_ignore_length", sampling_params.prompt_ignore_length, engine_name
        )

    n = sampling_params.n
    if sampling_params.best_of is not None and sampling_params.best_of != n:
        raise _not_supported("best_of", sampling_params.best_of, engine_name)
    if n > capabilities.max_num_sequences:
        raise ValueError(
            f"n={n} exceeds the {capabilities.max_num_sequences} sequence(s) engine "
            f"{engine_name!r} supports"
        )
    max_context = capabilities.max_context_length
    if max_context is not None and len(prompt_token_ids) >= max_context:
        # The LLM path's wording, so OpenAIServer reports code context_length_exceeded.
        raise ValueError(
            context_length_exceeded_message(
                max_context_length=max_context, num_prompt_tokens=len(prompt_token_ids)
            )
        )

    sampling: dict[str, float | int] = {}
    for name in _SAMPLING_FIELDS:
        value = _requested(sampling_params, name)
        if value is None:
            continue
        capability = SAMPLING_FIELD_CAPABILITY.get(name)
        if capability is not None and not capabilities.supports(capability):
            raise _not_supported(name, value, engine_name)
        sampling[name] = value
    if sampling_params.seed is not None:
        if sampling_params.seed < 0:
            raise ValueError(f"seed must be non-negative, got {sampling_params.seed}")
        sampling["seed"] = sampling_params.seed

    limits = []
    if max_context is not None:
        limits.append(max_context - len(prompt_token_ids))
    if capabilities.max_output_tokens is not None:
        limits.append(capabilities.max_output_tokens)
    limit = min(limits) if limits else None
    max_tokens = sampling_params.max_tokens
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

    if not sampling_params.detokenize and sampling_params.stop:
        # Stop strings are matched on the text the serving side decodes.
        raise ValueError(
            "stop strings are not supported when serving through an OpenEngine engine "
            "without detokenizing (detokenize=False, or a harmony model)"
        )
    return [
        EngineRequest(
            request_id=new_request_id(),
            prompt_token_ids=list(prompt_token_ids),
            max_tokens=max_tokens,
            sampling=sampling,
            min_tokens=sampling_params.min_tokens or None,
            stop_token_ids=list(dict.fromkeys(sampling_params.stop_token_ids or ())),
            ignore_eos=True if sampling_params.ignore_eos else None,
            num_sequences=n,
            model=capabilities.served_model_name,
            extra={REQ_DETOKENIZE: False},
        )
    ]


# --- Response assembly --------------------------------------------------------


class ResponseAssembler(Protocol):
    """Builds the outputs of one Generate call from its events.

    ``outputs`` hold the ``CompletionOutput`` objects a ``RequestOutput`` hands
    to the handlers. Their ``*_diff`` views cover the change made by the last
    :meth:`apply` that returned True, or by :meth:`finish_cancelled`, so the
    caller hands the outputs on after each such call.
    """

    outputs: list[CompletionOutput]
    cached_tokens: int

    @property
    def finished(self) -> bool:
        """Whether every output has finished."""
        ...

    def apply(self, event: EngineEvent) -> bool:
        """Apply one event; return whether any output changed."""
        ...

    def finish_cancelled(self) -> None:
        """End the outputs that have not finished, after the call was cancelled by the serving side."""
        ...


def create_response_assembler(
    *,
    sampling_params: SamplingParams,
    tokenizer: Any,
    num_outputs: int,
    engine_stop_texts: bool = False,
) -> ResponseAssembler:
    """Create the assembler for one Generate call returning ``num_outputs`` outputs.

    Args:
        engine_stop_texts: The call's ``EngineRequest.stop_texts`` holds the
            request's stop strings, so the engine matches them and the
            assembler waits for the engine to end each output; otherwise the
            serving side matches them.
    """
    impl, _ = resolve_slot("create_response_assembler")
    return cast(
        ResponseAssembler,
        impl(
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            num_outputs=num_outputs,
            engine_stop_texts=engine_stop_texts,
        ),
    )


class _DefaultResponseAssembler:
    """Serving-side detokenization and stop strings."""

    def __init__(self, detokenizers: list[Any]) -> None:
        from tensorrt_llm.executor.result import CompletionOutput

        self.outputs = [CompletionOutput(index=i) for i in range(len(detokenizers))]
        self.cached_tokens = 0
        self._detokenizers = detokenizers

    @property
    def finished(self) -> bool:
        return all(output.finish_reason is not None for output in self.outputs)

    def _start_step(self) -> None:
        for output in self.outputs:
            output._last_text_len = len(output.text)
            output._last_token_ids_len = len(output.token_ids)
            output._last_logprobs_len = len(output.logprobs)

    def apply(self, event: EngineEvent) -> bool:
        if isinstance(event, TokenDelta):
            return self._apply_tokens(event)
        if isinstance(event, OutputFinished):
            if event.usage is not None and event.usage.cached_prompt_tokens is not None:
                self.cached_tokens = event.usage.cached_prompt_tokens
            return self._apply_finished(event)
        return False

    def _apply_tokens(self, event: TokenDelta) -> bool:
        output = self.outputs[event.output_index]
        if output.finish_reason is not None:
            return False
        token_ids = list(event.token_ids)
        self._start_step()
        detokenizer = self._detokenizers[event.output_index]
        if detokenizer is None:
            output.token_ids.extend(token_ids)
            return True
        delta = detokenizer.step(token_ids)
        output.text += delta.text
        if delta.stop is None:
            output.token_ids.extend(token_ids)
            return True
        # Count the output up to the token that completed the stop string.
        keep = delta.stop.token_index + 1 - len(output.token_ids)
        output.token_ids.extend(token_ids[: max(keep, 0)])
        output.finish_reason = "stop"
        output.stop_reason = delta.stop.stop_str
        return True

    def _apply_finished(self, event: OutputFinished) -> bool:
        output = self.outputs[event.output_index]
        if output.finish_reason is not None:
            return False
        self._start_step()
        detokenizer = self._detokenizers[event.output_index]
        delta = detokenizer.finish() if detokenizer is not None else None
        if delta is not None:
            output.text += delta.text
        if delta is not None and delta.stop is not None:
            output.finish_reason = "stop"
            output.stop_reason = delta.stop.stop_str
            return True
        output.finish_reason = event.reason
        if event.stop_token_id is not None:
            output.stop_reason = event.stop_token_id
        elif event.stop_text is not None:
            output.stop_reason = event.stop_text
        return True

    def finish_cancelled(self) -> None:
        # As with LLM, an unfinished output ends with "cancelled", unless the
        # text its detokenizer still holds completes a stop string.
        self._start_step()
        for output, detokenizer in zip(self.outputs, self._detokenizers):
            if output.finish_reason is not None:
                continue
            delta = detokenizer.finish() if detokenizer is not None else None
            if delta is not None:
                output.text += delta.text
            if delta is not None and delta.stop is not None:
                output.finish_reason = "stop"
                output.stop_reason = delta.stop.stop_str
            else:
                output.finish_reason = "cancelled"


@_slot("create_response_assembler", "response", owner="WT-12")
def default_create_response_assembler(
    *,
    sampling_params: SamplingParams,
    tokenizer: Any,
    num_outputs: int,
    engine_stop_texts: bool = False,
) -> ResponseAssembler:
    """One serving-side detokenizer per output, or none when ``detokenize`` is off.

    It matches the stop strings on the serving side whatever ``engine_stop_texts`` says.
    """
    from tensorrt_llm.serve.detokenize import IncrementalDetokenizer

    detokenizers = [
        None
        if not sampling_params.detokenize
        else IncrementalDetokenizer(
            tokenizer,
            stop=sampling_params.stop,
            include_stop_str_in_output=sampling_params.include_stop_str_in_output,
            skip_special_tokens=sampling_params.skip_special_tokens,
            spaces_between_special_tokens=sampling_params.spaces_between_special_tokens,
        )
        for _ in range(num_outputs)
    ]
    return _DefaultResponseAssembler(detokenizers)


# --- Postprocessing -----------------------------------------------------------

_UNSET: Any = object()


class PostprocDispatch(Protocol):
    """Where ``OpenAIServer`` runs a request's postprocessor.

    With ``enabled``, postprocessing runs outside the handler: the handler
    passes its ``PostprocParams`` to ``generate_async`` and reads each result
    from ``output.outputs[0]._postprocess_result``. Otherwise the handler runs
    the postprocessor itself.

    ``OpenAIServer`` creates one dispatch per generator and uses it for every
    request. The ``/v1/completions`` handler passes its ``PostprocParams`` to
    ``generate_async`` whether or not ``enabled`` is set, without calling
    :meth:`generate_params`.
    """

    @property
    def enabled(self) -> bool: ...

    def generate_params(self, postproc_params: Any) -> Any:
        """The ``_postproc_params`` to pass to ``generate_async``."""
        ...

    def prepare(self, postproc_args: Any, promise: Any, *, tokenizer: Any = _UNSET) -> None:
        """Fill in what a handler-run postprocessor needs once the request has started.

        Args:
            tokenizer: Set ``postproc_args.tokenizer`` to this; left alone if not given.
        """
        ...

    def result(self, output: Any, postproc_params: Any) -> Any:
        """The postprocessed result of one ``RequestOutput`` step."""
        ...


@dataclass(frozen=True)
class DefaultPostprocDispatch:
    """Postprocess outside the handler when ``enabled``, else in the handler.

    Outside the handler means the in-process ``LLM``'s postprocessing workers,
    or the OpenEngine facade's postprocessing pool.
    """

    enabled: bool

    def generate_params(self, postproc_params: Any) -> Any:
        return postproc_params if self.enabled else None

    def prepare(self, postproc_args: Any, promise: Any, *, tokenizer: Any = _UNSET) -> None:
        if self.enabled:
            return
        if tokenizer is not _UNSET:
            postproc_args.tokenizer = tokenizer
        postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)

    def result(self, output: Any, postproc_params: Any) -> Any:
        if self.enabled:
            return output.outputs[0]._postprocess_result
        return postproc_params.post_processor(output, postproc_params.postproc_args)


def create_postproc_dispatch(generator: Any, *, is_visual_gen: bool) -> PostprocDispatch:
    """The postprocessing dispatch for ``OpenAIServer`` serving ``generator``.

    An implementation must return :class:`DefaultPostprocDispatch` semantics for
    every generator other than the OpenEngine facade, in-process ``LLM`` included.
    The facade is the generator whose class sets ``postproc_host = True``
    (:class:`PostprocHost`); unless ``is_visual_gen``, its dispatch is enabled
    when its ``postproc_pool`` is not None. Every ``OpenAIServer`` looks this
    slot up, so its module must import without the gRPC stack, and imports
    what the pool runs on (ZeroMQ, the tokenizer) only inside functions.
    """
    impl, _ = resolve_slot("create_postproc_dispatch")
    return cast(PostprocDispatch, impl(generator, is_visual_gen=is_visual_gen))


@_slot("create_postproc_dispatch", "postproc_pool", owner="WT-13")
def default_create_postproc_dispatch(generator: Any, *, is_visual_gen: bool) -> PostprocDispatch:
    if is_visual_gen:
        return DefaultPostprocDispatch(enabled=False)
    # Read from the class, so that a mock or a delegating object is not taken for the facade.
    if getattr(type(generator), "postproc_host", False) is True:
        return DefaultPostprocDispatch(enabled=generator.postproc_pool is not None)
    return DefaultPostprocDispatch(enabled=generator.args.num_postprocess_workers > 0)


# Messages between a frontend process and its postprocessing processes.


@dataclass(frozen=True)
class DetokOptions:
    """How a postprocessing process detokenizes and matches stop strings for one request."""

    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    stop: tuple[str, ...] | None = None
    include_stop_str_in_output: bool = False
    min_tokens: int = 0
    # The engine matches the stop strings (the ``engine_stop_texts`` given to
    # :meth:`PostprocPool.open`).
    engine_stop: bool = False


@dataclass(frozen=True)
class PPOpen:
    """Start postprocessing a request.

    Attributes:
        post_processor: Dotted import path of the postprocessor, see :func:`post_processor_path`.
        postproc_args: ``PostprocArgs`` holding plain data only, see :func:`check_plain_data`.
        n: The number of outputs, the ``num_outputs`` given to :meth:`PostprocPool.open`.
        detok: Derived by the pool from ``sampling_params``, so the two agree,
            and from the ``engine_stop_texts`` given to :meth:`PostprocPool.open`.
        num_prompt_tokens: The prompt's length in tokens.
        sampling_params: A copy of the request's ``SamplingParams`` with
            ``logits_processor`` and ``embedding_bias`` unset, holding plain data only.
    """

    request_id: str
    streaming: bool
    n: int
    post_processor: str
    postproc_args: Any
    detok: DetokOptions
    num_prompt_tokens: int
    sampling_params: Any


@dataclass(frozen=True)
class PPEvent:
    """One engine event of a request, in arrival order."""

    request_id: str
    event: EngineEvent


@dataclass(frozen=True)
class PPClose:
    """The request ended; ``cancelled`` when the serving side stopped it."""

    request_id: str
    reason: Literal["done", "cancelled"]


@dataclass(frozen=True)
class PPResult:
    """Postprocessed output of a request.

    Attributes:
        chunks: SSE strings for a streaming request, else the response object.
        stop_hit: Indices of the outputs a stop string ended; the frontend may
            cancel a call once every output it serves is listed.
        usage_fix: Corrected token counts after a serving-side stop.
    """

    request_id: str
    chunks: Any
    is_final: bool
    stop_hit: tuple[int, ...] | None = None
    usage_fix: Usage | None = None


PPMessage = PPOpen | PPEvent | PPClose | PPResult

# Postprocessors are imported by path in the postprocessing processes, and
# only from modules under this prefix.
POST_PROCESSOR_MODULE_PREFIX = "tensorrt_llm.serve."


class PostprocDataError(TypeError):
    """Something that is not plain data was about to cross into a postprocessing process."""


def post_processor_path(post_processor: Callable[..., Any]) -> str:
    """The dotted import path of a module-level postprocessor.

    Raises:
        PostprocDataError: The function is not importable by that path, or lives
            outside :data:`POST_PROCESSOR_MODULE_PREFIX`.
    """
    if not inspect.isfunction(post_processor):
        raise PostprocDataError(f"{post_processor!r} is not a function")
    module = getattr(post_processor, "__module__", None)
    qualname = getattr(post_processor, "__qualname__", None)
    if not module or not qualname or "<" in qualname or "." in qualname:
        raise PostprocDataError(f"{post_processor!r} is not a module-level function")
    path = f"{module}.{qualname}"
    if resolve_post_processor(path) is not post_processor:
        raise PostprocDataError(f"{path} does not import back to {post_processor!r}")
    return path


def resolve_post_processor(path: str) -> Callable[..., Any]:
    """Import a postprocessor by the path :func:`post_processor_path` gave.

    Raises:
        PostprocDataError: The path is outside :data:`POST_PROCESSOR_MODULE_PREFIX`,
            importing its module raises ImportError, or it does not name a
            function defined in that module.
    """
    if not path.startswith(POST_PROCESSOR_MODULE_PREFIX):
        raise PostprocDataError(
            f"postprocessors must come from {POST_PROCESSOR_MODULE_PREFIX}*, got {path!r}"
        )
    module_name, _, name = path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise PostprocDataError(f"cannot import {module_name} for {path!r}: {error}") from error
    function = getattr(module, name, None)
    if not inspect.isfunction(function):
        raise PostprocDataError(f"{path!r} is not a function")
    if getattr(function, "__module__", None) != module_name:
        raise PostprocDataError(f"{path!r} is not defined in {module_name}")
    return cast(Callable[..., Any], function)


_PLAIN_SCALARS = (type(None), bool, int, float, str, bytes)
# None before Python 3.11, where object has no __getstate__.
_OBJECT_GETSTATE = getattr(object, "__getstate__", None)


def _importable(cls: type) -> bool:
    """Whether pickle can find ``cls`` again by its module and name."""
    owner: Any = sys.modules.get(cls.__module__)
    for part in cls.__qualname__.split("."):
        owner = getattr(owner, part, None)
    return owner is cls


def _is_plain_named_tuple(item: Any) -> bool:
    """A named tuple whose class pickles by name and adds no per-instance state."""
    cls = type(item)
    return (
        isinstance(item, tuple)
        and hasattr(cls, "_fields")
        and _importable(cls)
        and not hasattr(item, "__dict__")
    )


def check_plain_data(value: Any, *, path: str = "postproc_args") -> None:
    """Check that ``value`` holds plain data only, so it can cross into another process.

    Plain data is None, bool, int, float, str, bytes, lists, tuples, sets and
    dicts of plain data (not subclasses of them, which may carry state of their
    own), and instances of classes pickle can import by name: enum members
    whose values are plain data, named tuples without per-instance state, and dataclass or pydantic model
    instances whose attributes are all plain data, declared or not. Callables,
    tensors, parsers, tokenizers and any other object are not.

    Raises:
        PostprocDataError: Names the first offending value by its path.
    """
    seen: set[int] = set()

    def check(item: Any, where: str) -> None:
        if type(item) in _PLAIN_SCALARS:
            return
        is_record = isinstance(item, enum.Enum) or (
            dataclasses.is_dataclass(item) and not isinstance(item, type)
        )
        model_fields = getattr(type(item), "model_fields", None)
        is_record = is_record or (isinstance(model_fields, dict) and hasattr(item, "model_dump"))
        # Pickle stores these by class name, so the class must be importable by it.
        if is_record and not _importable(type(item)):
            raise PostprocDataError(
                f"{where} is a {type(item).__qualname__}, which pickle cannot import by name"
            )
        if isinstance(item, enum.Enum):
            # Pickle stores a member by its class and value.
            check(item.value, f"{where}.value")
            return
        if id(item) in seen:
            return
        seen.add(id(item))
        if type(item) in (list, tuple, set, frozenset) or _is_plain_named_tuple(item):
            for i, element in enumerate(item):
                check(element, f"{where}[{i}]")
            return
        if type(item) is dict:
            for key, element in item.items():
                check(key, f"{where}.keys()")
                check(element, f"{where}[{key!r}]")
            return
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            # Everything pickled with the instance: the fields, and any other
            # attribute in its __dict__.
            names = [f.name for f in dataclasses.fields(item)]
            names += [name for name in getattr(item, "__dict__", {}) if name not in names]
            for name in names:
                if hasattr(item, name):
                    check(getattr(item, name), f"{where}.{name}")
                elif getattr(type(item), "__getstate__", None) is not _OBJECT_GETSTATE:
                    # An init=False field may not be set yet. The default
                    # pickling skips it, but a class with a __getstate__ of its
                    # own (a frozen dataclass with __slots__, or a subclass of
                    # one) may read it and fail.
                    raise PostprocDataError(f"{where}.{name} is not set")
            return
        if isinstance(model_fields, dict) and hasattr(item, "model_dump"):
            # Everything pickled with a pydantic model: its __dict__, the
            # extra fields and the private attributes.
            attributes = dict(getattr(item, "__dict__", {}))
            attributes.update(getattr(item, "__pydantic_extra__", None) or {})
            attributes.update(getattr(item, "__pydantic_private__", None) or {})
            for name, element in attributes.items():
                check(element, f"{where}.{name}")
            return
        raise PostprocDataError(f"{where} is a {type(item).__qualname__}, which is not plain data")

    check(value, path)


# The postprocessing processes of a frontend.

# Events of one request (over all its Generate calls) that are not postprocessed yet, at most.
POSTPROC_PENDING_EVENT_LIMIT = 4096


class PostprocError(RuntimeError):
    """Postprocessing a request failed.

    When postprocessing the request raises in its process (in the
    postprocessor, the hook, the detokenizer or the response assembler, for
    example), the session raises this class, not the original exception. The
    message says what failed, without the request's content.

    The frontend cancels the request's calls still running, and the facade
    ends the request with ``http_status`` the way it ends a request on its own
    errors, including a response that has already started.
    """

    http_status: ClassVar[int] = 500


class PostprocWorkerError(PostprocError):
    """The process holding the request exited, or the pool was closed, before the final result."""


class PostprocOverflowError(PostprocError):
    """The request already had :data:`POSTPROC_PENDING_EVENT_LIMIT` events not postprocessed yet."""

    http_status: ClassVar[int] = 503


class PostprocSession(Protocol):
    """One request postprocessed in one process of the pool.

    The frontend sends the events of every Generate call serving the request,
    in arrival order, with ``output_index`` counted over the request's outputs,
    then closes the session. Iterating the session yields the results in order;
    the last one has ``is_final`` set, and the iteration ends after it.

    The final result comes once every output has finished (by the engine, by
    a stop string, or by the ``post_processor_hook`` terminating the request),
    or after :meth:`close`, whichever is first. When the final result comes
    before the frontend closed the session, the frontend cancels the calls
    still running, if any, then closes the session. The frontend may also
    cancel a call once every output it serves is in some result's ``stop_hit``.

    The session is over after its final result, or once :meth:`send` or the
    iteration raised a :class:`PostprocError`. Then the pool has forgotten the
    request, and :meth:`send` and :meth:`close` do nothing; the frontend still
    calls :meth:`close`, as it does for every session. Nobody has to read the
    results of a session closed with ``"cancelled"``.
    """

    def send(self, event: EngineEvent) -> None:
        """Queue one event without blocking; does nothing once the session is over.

        Raises:
            PostprocOverflowError: The request already has
                :data:`POSTPROC_PENDING_EVENT_LIMIT` events not postprocessed
                yet. The event is not queued, the session is over, and its
                iteration raises the same error.
        """
        ...

    def close(self, reason: Literal["done", "cancelled"]) -> None:
        """No more events will come.

        ``"cancelled"``: the serving side stopped the request (a client that
        went away, an abort, an error), and the outputs not finished end as
        :meth:`ResponseAssembler.finish_cancelled` ends them.
        """
        ...

    def __aiter__(self) -> AsyncIterator[PPResult]:
        """The results in order.

        Raises:
            PostprocError: Postprocessing failed; see its subclasses.
        """
        ...


class PostprocPool(Protocol):
    """The postprocessing processes of one frontend."""

    @property
    def size(self) -> int:
        """The number of processes."""
        ...

    def open(
        self,
        request_id: str,
        postproc_params: Any,
        sampling_params: SamplingParams,
        *,
        streaming: bool,
        num_outputs: int,
        num_prompt_tokens: int,
        engine_stop_texts: bool = False,
    ) -> PostprocSession:
        """Start postprocessing a request in the process ``request_id`` routes to.

        Args:
            request_id: The ``id`` of the output object the facade yields for
                this request (postprocessors read it as ``rsp.id``); unique
                over the life of the pool.
            postproc_params: The ``PostprocParams`` the handler passed to ``generate_async``.
            sampling_params: The request's sampling parameters, as given to
                ``generate_async``; detokenization and stop strings follow them.
                The pool sends the copy described in :class:`PPOpen`.
            num_prompt_tokens: The prompt's length in tokens, for the usage the
                postprocessor reports.
            engine_stop_texts: As for :func:`create_response_assembler`; the
                pool sends it as ``DetokOptions.engine_stop``.

        Raises:
            PostprocDataError: ``postproc_params``, or a field of
                ``sampling_params`` other than those :class:`PPOpen` unsets,
                cannot cross into a process (see :func:`post_processor_path`
                and :func:`check_plain_data`); a program error, and the facade
                ends the request with HTTP 500.
            PostprocWorkerError: The pool is closed.
        """
        ...

    async def aclose(self) -> None:
        """Stop the processes; sessions without a final result fail with :class:`PostprocWorkerError`."""
        ...


class PostprocHost(Protocol):
    """What the OpenEngine facade exposes about postprocessing.

    The facade's class sets ``postproc_host = True``. Its ``postproc_pool`` is
    the pool from :func:`create_postproc_pool`, or None; it is set before the
    facade is handed to ``OpenAIServer``, which decides where postprocessing
    runs from it once, and is never replaced, closing the pool included.
    ``OpenAIServer`` recognizes the facade by its class, so it must be handed
    the facade itself, not an object wrapping it.

    With a pool, ``generate_async`` postprocesses in it every request given
    ``_postproc_params`` and sets ``outputs[0]._postprocess_result`` of each
    output it yields to the ``chunks`` of the next result; without one, it
    ignores ``_postproc_params``.
    """

    postproc_host: ClassVar[bool]
    postproc_pool: PostprocPool | None


def create_postproc_pool(args: Any, *, num_workers: int, tokenizer_dir: str) -> PostprocPool | None:
    """The postprocessing pool of a frontend serving with ``args`` (its ``TorchLlmArgs``).

    The facade calls this once, when it is constructed (see :class:`PostprocHost`).

    Args:
        args: The arguments as the user configured them, before the facade
            overrides any; the pool reads ``post_processor_hook`` from them.
        num_workers: This frontend's share of ``num_postprocess_workers``, the
            number of processes to start; 0 for no pool.
        tokenizer_dir: The local directory the frontend loads its tokenizer from.

    Returns:
        None when the frontend postprocesses in its own process.
    """
    impl, _ = resolve_slot("create_postproc_pool")
    return cast(
        "PostprocPool | None", impl(args, num_workers=num_workers, tokenizer_dir=tokenizer_dir)
    )


@_slot("create_postproc_pool", "postproc_pool", owner="WT-13")
def default_create_postproc_pool(
    args: Any, *, num_workers: int, tokenizer_dir: str
) -> PostprocPool | None:
    """No pool: the handlers postprocess in the frontend process."""
    return None


# --- Frontends ----------------------------------------------------------------


@dataclass
class FrontendLaunchConfig:
    """Everything needed to start the serving frontends for one engine.

    Attributes:
        llm_args: The ``trtllm-serve`` LLM arguments, including ``num_serve_frontends``.
        ready_timeout: Seconds to wait for the engine; None for the frontend's own default.
    """

    host: str
    port: int
    llm_args: dict[str, Any]
    endpoint: str
    tool_parser: str | None = None
    middleware: Sequence[str] = ()
    chat_template: str | None = None
    served_model_name: str | None = None
    allow_request_chat_template: bool = False
    multimodal_server_config: Any = None
    num_input_processor_workers: int = 8
    num_media_load_workers: int = 8
    ready_timeout: float | None = None
    report_addr: str | None = None


def launch_frontends(config: FrontendLaunchConfig) -> None:
    """Start the frontends and serve until they stop."""
    impl, _ = resolve_slot("launch_frontends")
    impl(config)


@_slot("launch_frontends", "multi_frontend", owner="WT-14")
def default_launch_frontends(config: FrontendLaunchConfig) -> None:
    """One frontend, in this process."""
    import click

    from . import launch

    if config.llm_args.get("num_serve_frontends", 1) not in (None, 1):
        raise click.UsageError("--openengine-endpoint supports a single serving frontend")
    if config.report_addr is not None:
        raise click.UsageError("--report_addr is not supported with --openengine-endpoint")
    kwargs: dict[str, Any] = {}
    if config.ready_timeout is not None:
        kwargs["ready_timeout"] = config.ready_timeout
    launch.launch_openengine_server(
        config.host,
        config.port,
        config.llm_args,
        config.endpoint,
        tool_parser=config.tool_parser,
        middleware=config.middleware,
        chat_template=config.chat_template,
        served_model_name=config.served_model_name,
        allow_request_chat_template=config.allow_request_chat_template,
        multimodal_server_config=config.multimodal_server_config,
        num_input_processor_workers=config.num_input_processor_workers,
        num_media_load_workers=config.num_media_load_workers,
        **kwargs,
    )


# --- Metrics ------------------------------------------------------------------


class RequestMetrics(Protocol):
    """Timing of one request, measured from its engine events on the serving side."""

    def on_event(self, event: EngineEvent, now: float) -> None: ...

    def metrics_dict(self) -> dict[str, Any]:
        """The ``RequestOutput.metrics_dict`` entries measured so far."""
        ...


class MetricsRecorder(Protocol):
    """Request metrics, iteration stats and KV-cache events of a frontend."""

    def start_request(self, request_id: str, arrival_time: float) -> RequestMetrics: ...

    def iteration_stats(self, load: EngineLoad | None) -> dict[str, Any] | None:
        """One ``/metrics`` entry derived from an engine load snapshot; None for none."""
        ...

    def kv_cache_events(self, client: EngineClient, timeout: float | None) -> Awaitable[Any]:
        """The KV-cache events received so far, in ``/kv_cache_events`` format.

        Raises:
            OpenEngineUnsupportedError: The engine publishes no KV-cache events.
        """
        ...


def create_metrics_recorder(args: Any) -> MetricsRecorder:
    """The metrics recorder for a frontend serving with ``args`` (its ``TorchLlmArgs``)."""
    impl, _ = resolve_slot("create_metrics_recorder")
    return cast(MetricsRecorder, impl(args))


class _NoRequestMetrics:
    def on_event(self, event: EngineEvent, now: float) -> None:
        pass

    def metrics_dict(self) -> dict[str, Any]:
        return {}


class _NoMetrics:
    """Records nothing; the OpenEngine path serves with perf metrics off."""

    def start_request(self, request_id: str, arrival_time: float) -> RequestMetrics:
        return _NoRequestMetrics()

    def iteration_stats(self, load: EngineLoad | None) -> dict[str, Any] | None:
        return None

    async def kv_cache_events(self, client: EngineClient, timeout: float | None) -> Any:
        raise OpenEngineUnsupportedError(
            "/kv_cache_events is not supported when serving through an OpenEngine engine"
        )


@_slot("create_metrics_recorder", "metrics", owner="WT-15")
def default_create_metrics_recorder(args: Any) -> MetricsRecorder:
    return _NoMetrics()


# --- Media and LoRA -----------------------------------------------------------

_MULTIMODAL_KEYS = ("multi_modal_data", "multi_modal_embeddings", "mm_processor_kwargs")


def map_media(
    inputs: Mapping[str, Any], capabilities: ModelCapabilities, *, engine_name: str
) -> list[MediaSpec]:
    """The media of a prompt, in placeholder order.

    Raises:
        ValueError: The engine does not take this media; reported as HTTP 400.
    """
    impl, _ = resolve_slot("map_media")
    return cast(list[MediaSpec], impl(inputs, capabilities, engine_name=engine_name))


@_slot("map_media", "media", owner="WT-16")
def default_map_media(
    inputs: Mapping[str, Any], capabilities: ModelCapabilities, *, engine_name: str
) -> list[MediaSpec]:
    """Reject every multimodal input."""
    if any(inputs.get(key) is not None for key in _MULTIMODAL_KEYS):
        raise ValueError(
            "multimodal inputs are not supported when serving through an OpenEngine engine"
        )
    return []


class LoraRegistry(Protocol):
    """Registers LoRA adapters with the engine the first time a request uses them."""

    async def resolve(self, lora_request: Any) -> str | None:
        """The adapter name to send for ``lora_request``; None when there is none.

        Raises:
            ValueError: The adapter cannot be used; reported as HTTP 400.
        """
        ...


def create_lora_registry(client: EngineClient, capabilities: ModelCapabilities) -> LoraRegistry:
    """The LoRA registry of a frontend connected to ``client``."""
    impl, _ = resolve_slot("create_lora_registry")
    return cast(LoraRegistry, impl(client, capabilities))


class _NoLora:
    async def resolve(self, lora_request: Any) -> str | None:
        if lora_request is not None:
            raise ValueError(
                "lora_request is not supported when serving through an OpenEngine engine"
            )
        return None


@_slot("create_lora_registry", "lora", owner="WT-16")
def default_create_lora_registry(
    client: EngineClient, capabilities: ModelCapabilities
) -> LoraRegistry:
    """Reject every LoRA request."""
    return _NoLora()


# --- Generate-call counter ----------------------------------------------------


class GenerateCounter:
    """Counts the Generate calls a frontend sent, for proving requests went through the engine.

    The count starts at 0 and only grows; every call counts, including each
    call of a split request. Call :meth:`increment` once a call has been sent.
    When ``counter_dir`` is set, every change writes
    ``{"pid": ..., "generate_sent": ...}`` to ``<counter_dir>/<pid>.json``,
    replacing the file atomically; the directory is created if missing. A
    process keeps one counter, and each run uses a directory of its own:
    :func:`read_generate_sent` sums every file in it.

    Args:
        counter_dir: Where to write the count; defaults to ``$TRTLLM_OPENENGINE_COUNTER_DIR``.
    """

    def __init__(self, counter_dir: str | os.PathLike[str] | None = _UNSET) -> None:
        if counter_dir is _UNSET:
            counter_dir = os.environ.get(SERVE_COUNTER_DIR_ENV) or None
        self._dir = Path(counter_dir) if counter_dir is not None else None
        self._value = 0
        self._lock = threading.Lock()

    @property
    def value(self) -> int:
        return self._value

    def increment(self, count: int = 1) -> int:
        """Add ``count`` calls and return the new total.

        Raises:
            ValueError: ``count`` is negative.
            OSError: The file could not be written. The calls are counted
                anyway, and the next write that succeeds brings the file up to date.
        """
        if count < 0:
            raise ValueError(f"the count only grows, got {count}")
        with self._lock:
            self._value += count
            if self._dir is not None:
                self._write(self._value)
            return self._value

    def _write(self, value: int) -> None:
        assert self._dir is not None
        pid = os.getpid()
        # A fresh temporary file (created exclusively), then an atomic rename.
        try:
            fd, tmp = tempfile.mkstemp(prefix=f".{pid}.", suffix=".tmp", dir=self._dir)
        except FileNotFoundError:
            self._dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=f".{pid}.", suffix=".tmp", dir=self._dir)
        try:
            with os.fdopen(fd, "w") as file:
                file.write(json.dumps({"pid": pid, SERVE_COUNTER_GENERATE_SENT: value}))
            os.replace(tmp, self._dir / f"{pid}.json")
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def read_generate_sent(counter_dir: str | os.PathLike[str]) -> int:
    """The total of ``generate_sent`` over every frontend's file in ``counter_dir``."""
    total = 0
    for path in sorted(Path(counter_dir).glob("*.json")):
        total += int(json.loads(path.read_text())[SERVE_COUNTER_GENERATE_SENT])
    return total

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An in-process ``EngineClient`` for unit tests of the serving side.

``FakeEngineClient`` answers every ``EngineClient`` method without a server:
it records the requests and aborts it receives, keeps the conventions'
counters, and plays back a script of events per request. A script is a
function from the request to a sequence of steps:

- an ``EngineEvent`` is yielded;
- ``Reject(error)`` as the first step rejects the request before acceptance,
  so the first ``__anext__`` raises ``error`` and it is not counted as accepted;
- ``Fail(error)`` raises ``error`` after acceptance;
- ``Delay(seconds)`` sleeps.

Without a script, every output gets :data:`DEFAULT_TOKENS` and finishes with
reason ``length``.

Use it in tests that exercise one serving module. Cross-process tests use the
gRPC fake engine instead.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tensorrt_llm.grpc.openengine_conventions import (
    BOOL_CAPABILITY_KEYS,
    CAP_MIN_P,
    CAP_REQ_BEST_OF,
    CAP_STOP_TEXT,
    CONVENTIONS_VERSION,
    CONVENTIONS_VERSION_KEY,
    COUNTER_GENERATE_ACCEPTED,
    COUNTER_GENERATE_RECEIVED,
    SEED_SCOPE_BATCH_DEPENDENT,
    ExtraCapabilities,
)
from tensorrt_llm.serve.openengine_poc.engine_client import (
    EngineEvent,
    EngineHealth,
    EngineInfo,
    EngineLoad,
    EngineRequest,
    HealthState,
    KvEventSource,
    LoraAdapter,
    ModelCapabilities,
    OutputFinished,
    TokenDelta,
    Usage,
)

DEFAULT_TOKENS = (11, 12, 13)


@dataclass(frozen=True)
class Reject:
    """Reject the request before acceptance with ``error``."""

    error: BaseException


@dataclass(frozen=True)
class Fail:
    """Fail the request after acceptance with ``error``."""

    error: BaseException


@dataclass(frozen=True)
class Delay:
    seconds: float


Step = EngineEvent | Reject | Fail | Delay
Script = Callable[[EngineRequest], Sequence[Step]]


def default_script(request: EngineRequest) -> list[Step]:
    """Every output gets :data:`DEFAULT_TOKENS` (capped at ``max_tokens``) and ends with ``length``."""
    tokens = DEFAULT_TOKENS[: request.max_tokens] if request.max_tokens else DEFAULT_TOKENS
    steps: list[Step] = []
    for index in range(request.num_sequences):
        steps.append(TokenDelta(output_index=index, token_ids=tuple(tokens)))
    for index in range(request.num_sequences):
        last = index == request.num_sequences - 1
        usage = (
            Usage(
                prompt_tokens=len(request.prompt_token_ids),
                completion_tokens=len(tokens) * request.num_sequences,
                cached_prompt_tokens=0,
            )
            if last
            else None
        )
        steps.append(OutputFinished(output_index=index, reason="length", usage=usage))
    return steps


def trtllm_like() -> tuple[EngineInfo, ModelCapabilities]:
    """What the TRT-LLM OpenEngine server declares once it follows the conventions."""
    info = EngineInfo(
        engine_name="tensorrt_llm",
        engine_version="fake",
        instance_id="fake-trtllm-1",
        schema_revision=1,
        minimum_client_revision=1,
        extra={CONVENTIONS_VERSION_KEY: CONVENTIONS_VERSION},
    )
    caps = ModelCapabilities(
        served_model_name="fake-model",
        max_context_length=4096,
        supports_token_ids_input=True,
        supports_text_input=True,
        output_logprobs_max_top_n=100,
        prompt_logprobs_max_top_n=100,
        guided_modes=frozenset(
            {"json_schema", "regex", "ebnf_grammar", "structural_tag", "json_object"}
        ),
        max_num_sequences=16,
        supports_stop_in_output=True,
        supports_cache_salt=True,
        extra=ExtraCapabilities(
            declared=frozenset(BOOL_CAPABILITY_KEYS) - {CAP_REQ_BEST_OF},
            seed_scope=SEED_SCOPE_BATCH_DEPENDENT,
        ),
    )
    return info, caps


def glm_like() -> tuple[EngineInfo, ModelCapabilities]:
    """What the glm-5.3-flash OpenEngine server declares once it follows the conventions."""
    info = EngineInfo(
        engine_name="glm5",
        engine_version="fake",
        instance_id="fake-glm-1",
        schema_revision=1,
        minimum_client_revision=1,
        extra={CONVENTIONS_VERSION_KEY: CONVENTIONS_VERSION},
    )
    caps = ModelCapabilities(
        served_model_name="fake-glm",
        max_context_length=4096,
        supports_token_ids_input=True,
        output_logprobs_max_top_n=20,
        max_num_sequences=1,
        supports_stop_in_output=True,
        extra=ExtraCapabilities(
            declared=frozenset({CAP_MIN_P, CAP_STOP_TEXT}),
            seed_scope=SEED_SCOPE_BATCH_DEPENDENT,
        ),
    )
    return info, caps


@dataclass
class FakeEngineClient:
    """An ``EngineClient`` that plays back scripted events in-process.

    Attributes:
        requests: Every request received, rejected ones included, in order.
        aborted: The request ids passed to :meth:`abort`.
        cancelled: The ids of accepted requests that ended before every output finished,
            because their iterator was closed or a ``Fail`` step raised.
        active: The ids of requests accepted and not yet finished.
    """

    info: EngineInfo = field(default_factory=lambda: trtllm_like()[0])
    capabilities: ModelCapabilities = field(default_factory=lambda: trtllm_like()[1])
    script: Script = default_script
    health_state: HealthState = "ready"
    report_load: bool = True
    event_sources: list[KvEventSource] = field(default_factory=list)
    requests: list[EngineRequest] = field(default_factory=list)
    aborted: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    active: set[str] = field(default_factory=set)
    loras: dict[str, LoraAdapter] = field(default_factory=dict)
    generate_received: int = 0
    generate_accepted: int = 0
    _abort_events: dict[str, asyncio.Event] = field(default_factory=dict, repr=False)

    @classmethod
    def like(cls, preset: str, **kwargs: Any) -> FakeEngineClient:
        """A client declaring the ``"trtllm"`` or ``"glm"`` capability preset."""
        info, caps = {"trtllm": trtllm_like, "glm": glm_like}[preset]()
        return cls(info=info, capabilities=caps, **kwargs)

    def with_capabilities(self, **changes: Any) -> FakeEngineClient:
        """Replace some :class:`ModelCapabilities` fields, returning ``self``."""
        self.capabilities = dataclasses.replace(self.capabilities, **changes)
        return self

    def generate(self, request: EngineRequest) -> AsyncIterator[EngineEvent]:
        # Counted at the call, as an engine counts a Generate call it receives.
        self.generate_received += 1
        self.requests.append(request)
        steps = list(self.script(request))
        # Created now, so an abort that arrives before the first read still counts.
        self._abort_events[request.request_id] = asyncio.Event()
        return self._run(request, steps)

    async def _run(self, request: EngineRequest, steps: list[Step]) -> AsyncIterator[EngineEvent]:
        request_id = request.request_id
        aborted = self._abort_events[request_id]
        if steps and isinstance(steps[0], Reject):
            self._abort_events.pop(request_id, None)
            raise steps[0].error
        if any(isinstance(step, Reject) for step in steps):
            self._abort_events.pop(request_id, None)
            raise ValueError("Reject is only allowed as the first step of a script")
        self.generate_accepted += 1
        self.active.add(request_id)
        unfinished = set(range(request.num_sequences))
        completed = False
        try:
            for step in steps:
                if aborted.is_set():
                    for index in sorted(unfinished):
                        yield OutputFinished(output_index=index, reason="cancelled")
                    completed = True
                    return
                if isinstance(step, Delay):
                    await asyncio.sleep(step.seconds)
                elif isinstance(step, Fail):
                    raise step.error
                else:
                    if isinstance(step, OutputFinished):
                        unfinished.discard(step.output_index)
                    yield step
            completed = True
        finally:
            self.active.discard(request_id)
            self._abort_events.pop(request_id, None)
            if not completed and unfinished:
                self.cancelled.append(request_id)

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)
        event = self._abort_events.get(request_id)
        if event is not None:
            event.set()

    async def health(self, *, deep: bool = False) -> EngineHealth:
        return EngineHealth(state=self.health_state)

    async def load(self) -> EngineLoad | None:
        if not self.report_load:
            return None
        return EngineLoad(
            instance_id=self.info.instance_id,
            running_requests=len(self.active),
            queued_requests=0,
            attributes={
                COUNTER_GENERATE_RECEIVED: self.generate_received,
                COUNTER_GENERATE_ACCEPTED: self.generate_accepted,
            },
        )

    def server_info(self) -> EngineInfo:
        return self.info

    def model_info(self) -> ModelCapabilities:
        return self.capabilities

    async def load_lora(self, name: str, int_id: int, path: str) -> bool:
        if not self.capabilities.supports_lora:
            raise NotImplementedError(f"engine {self.info.engine_name!r} does not support LoRA")
        existing = self.loras.get(name)
        if existing is not None:
            if (existing.int_id, existing.path) != (int_id, path):
                raise ValueError(f"LoRA {name!r} is already registered differently")
            return True
        self.loras[name] = LoraAdapter(name=name, int_id=int_id, path=path)
        return False

    async def list_loras(self) -> list[LoraAdapter]:
        return list(self.loras.values())

    async def kv_event_sources(self) -> list[KvEventSource]:
        return list(self.event_sources)


# A word-level tokenizer small enough to reason about token by token.
TINY_VOCAB = {"<unk>": 0, "</s>": 1, "He": 2, "llo": 3, " world": 4, "!": 5, " again": 6}
TINY_EOS = 1


def tiny_hf_tokenizer() -> Any:
    """A Hugging Face fast tokenizer over :data:`TINY_VOCAB` with ``</s>`` as EOS."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteFallback, Fuse
    from tokenizers.decoders import Sequence as DecoderSequence
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel(TINY_VOCAB, unk_token="<unk>"))
    backend.decoder = DecoderSequence([ByteFallback(), Fuse()])
    backend.add_special_tokens(["</s>"])
    hf = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>")
    hf.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    return hf


def tiny_tokenizer() -> Any:
    """:func:`tiny_hf_tokenizer` wrapped as the serving side uses it."""
    from tensorrt_llm.tokenizer import TransformersTokenizer

    return TransformersTokenizer(tiny_hf_tokenizer())

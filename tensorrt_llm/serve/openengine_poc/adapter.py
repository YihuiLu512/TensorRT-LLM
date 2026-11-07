# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Earlier names of the OpenEngine ``LLM`` facade (see :mod:`.facade`), and the PoC-S1 request output.

``OpenEngineLLMAdapter`` is the facade. ``OpenEngineRequestOutput`` here is the
request output of the PoC-S1 adapter, which read one Generate stream through
per-output detokenizers; the facade does not use it. It stays as the reference
the default response assembler is checked against. The facade's request
output is :class:`.facade.OpenEngineRequestOutput`, which the package exports.
"""

from __future__ import annotations

import asyncio
from typing import Any

import grpc

from tensorrt_llm.executor.result import CompletionOutput
from tensorrt_llm.grpc.openengine.bindings import error_pb2, generation_pb2
from tensorrt_llm.grpc.openengine_client import (
    GenerateStream,
    OpenEngineClientError,
    OpenEngineEngineError,
    OpenEngineRpcError,
)
from tensorrt_llm.serve.detokenize import IncrementalDetokenizer

from .facade import LOG_PROMPT_IDS_ENV, OpenEngineLLM

OpenEngineLLMAdapter = OpenEngineLLM

__all__ = ["LOG_PROMPT_IDS_ENV", "OpenEngineLLMAdapter", "OpenEngineRequestOutput"]

_CLIENT_ERROR_STATUSES = (grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.UNIMPLEMENTED)
_CLIENT_ERROR_CODES = (
    error_pb2.ERROR_CODE_INVALID_ARGUMENT,
    error_pb2.ERROR_CODE_UNSUPPORTED_FEATURE,
)
# Strong references to the background closes started by OpenEngineRequestOutput,
# so they are not garbage-collected while running.
_BACKGROUND_CLOSES: set[asyncio.Task[None]] = set()
_FINISH_REASONS = {
    generation_pb2.FINISH_REASON_STOP: "stop",
    generation_pb2.FINISH_REASON_LENGTH: "length",
    generation_pb2.FINISH_REASON_CANCELLED: "cancelled",
}


def _request_error(error: OpenEngineClientError) -> Exception:
    """Turn a client error into ``ValueError`` when the request was at fault, else ``RuntimeError``."""
    if isinstance(error, OpenEngineRpcError) and error.code in _CLIENT_ERROR_STATUSES:
        return ValueError(str(error))
    if isinstance(error, OpenEngineEngineError) and error.code in _CLIENT_ERROR_CODES:
        return ValueError(str(error))
    return RuntimeError(str(error))


class OpenEngineRequestOutput:
    """The ``RequestOutput`` of one request, read from an OpenEngine Generate stream.

    Iterating it yields itself after every change to an output, the way a
    streaming ``RequestOutput`` does; ``aresult()`` and ``await`` read it to the
    end. The ``CompletionOutput`` objects in :attr:`outputs` keep their
    ``*_diff`` views relative to the previous yield.
    """

    def __init__(
        self,
        stream: GenerateStream,
        prompt_token_ids: list[int],
        detokenizers: list[IncrementalDetokenizer | None],
    ) -> None:
        self._stream = stream
        self._closing = False
        self._aborted = False
        self.request_id = stream.request_id
        # The streaming postprocessors build the response id from ``id``.
        self.id = stream.request_id
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [CompletionOutput(index=i) for i in range(len(detokenizers))]
        self._detokenizers = detokenizers
        self.finished = False
        self.error: str | None = None
        self.cached_tokens = 0
        self.context_logits = None
        self.metrics_dict: dict[str, Any] = {}
        self.candidate_metrics = None
        self.disaggregated_params = None

    @property
    def _done(self) -> bool:
        return self.finished

    def abort(self) -> None:
        """Cancel the request; the engine treats a cancelled Generate call as an abort.

        As with ``LLM``, the outputs that have not finished end with finish
        reason ``cancelled``, unless the text their detokenizer still holds
        completes a stop string.
        """
        if not self.finished:
            self._aborted = True
        self._stream.cancel()

    def __aiter__(self) -> OpenEngineRequestOutput:
        return self

    async def __anext__(self) -> OpenEngineRequestOutput:
        if self.finished:
            if not self._closing:
                # Read the call to its end in the background, so the response
                # is not held back until the engine closes the call.
                self._closing = True
                task = asyncio.get_running_loop().create_task(self._stream.aclose())
                _BACKGROUND_CLOSES.add(task)
                task.add_done_callback(_BACKGROUND_CLOSES.discard)
            raise StopAsyncIteration
        while True:
            try:
                response = await self._stream.__anext__()
            except StopAsyncIteration:
                # This path sends no prefill-only requests, so a call ends before
                # every output finished only when it was cancelled.
                if self._aborted:
                    self._start_step()
                    for output, detokenizer in zip(self.outputs, self._detokenizers):
                        if output.finish_reason is not None:
                            continue
                        # Release the text the detokenizer still holds; it can
                        # still complete a stop string.
                        delta = detokenizer.finish() if detokenizer is not None else None
                        if delta is not None:
                            output.text += delta.text
                        if delta is not None and delta.stop is not None:
                            output.finish_reason = "stop"
                            output.stop_reason = delta.stop.stop_str
                        else:
                            output.finish_reason = "cancelled"
                    self.finished = True
                    return self
                self._fail("the engine call was cancelled before the request finished")
                raise RuntimeError(self.error) from None
            except OpenEngineClientError as error:
                self._fail(str(error))
                raise _request_error(error) from error
            applied = False
            try:
                changed = self._apply(response)
                applied = True
            finally:
                if not applied:
                    # Whatever went wrong, the engine must not keep generating.
                    self._stream.cancel()
                    self._fail("failed to apply an engine response")
            if changed:
                return self

    async def aresult(self) -> OpenEngineRequestOutput:
        async for _ in self:
            pass
        return self

    def __await__(self):
        return self.aresult().__await__()

    def _fail(self, message: str) -> None:
        self.error = message
        self.finished = True

    def _start_step(self) -> None:
        for output in self.outputs:
            output._last_text_len = len(output.text)
            output._last_token_ids_len = len(output.token_ids)
            output._last_logprobs_len = len(output.logprobs)

    def _apply(self, response: generation_pb2.GenerateResponse) -> bool:
        """Apply one response; return whether any output changed."""
        event = response.WhichOneof("event")
        if response.HasField("usage") and response.usage.HasField("cached_prompt_tokens"):
            self.cached_tokens = response.usage.cached_prompt_tokens
        if event == "token":
            changed = self._apply_tokens(response.token)
        elif event == "finished":
            changed = self._apply_finished(response.finished)
        else:
            return False
        if changed and all(output.finish_reason is not None for output in self.outputs):
            self.finished = True
            # Tokens generated past a stop string are not needed: cancelling
            # the call aborts the request on the engine.
            self._stream.cancel()
        return changed

    def _apply_tokens(self, event: generation_pb2.TokenOutput) -> bool:
        index = event.output_index
        output = self.outputs[index]
        if output.finish_reason is not None:
            return False
        token_ids = [token.token_id for token in event.tokens]
        self._start_step()
        detokenizer = self._detokenizers[index]
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

    def _apply_finished(self, event: generation_pb2.GenerationFinished) -> bool:
        index = event.output_index
        output = self.outputs[index]
        if output.finish_reason is not None:
            return False
        self._start_step()
        detokenizer = self._detokenizers[index]
        delta = detokenizer.finish() if detokenizer is not None else None
        if delta is not None:
            output.text += delta.text
        if delta is not None and delta.stop is not None:
            output.finish_reason = "stop"
            output.stop_reason = delta.stop.stop_str
            return True
        # The client rejects a finished event without a known reason.
        output.finish_reason = _FINISH_REASONS[event.reason]
        match = event.stop_match.WhichOneof("match")
        if match == "stop_token_id":
            output.stop_reason = event.stop_match.stop_token_id
        elif match == "stop_text":
            output.stop_reason = event.stop_match.stop_text
        return True

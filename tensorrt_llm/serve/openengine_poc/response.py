# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Builds the ``CompletionOutput`` objects of a request from its OpenEngine events.

This module implements the ``create_response_assembler`` slot of
:mod:`.contracts`. It builds the text, ``token_ids``, ``logprobs``,
``prompt_logprobs``, ``finish_reason`` and ``stop_reason`` of each output, and
the request's ``cached_tokens``, as ``LLM`` does for the same token ids, except
for the differences listed at the end.

Who matches stop strings decides how the output ends:

- the engine (``engine_stop_texts``: the request's stop strings were sent as
  ``EngineRequest.stop_texts``): its tokens are kept as they come, and when the
  output finishes, its text is cut as ``LLM`` cuts it, at the first stop string
  in request order that it holds (``stop_reason`` is then that string, and
  ``finish_reason`` is ``stop``). The usage the engine sends with its last
  event is complete;
- the serving side (the fallback): each output's
  :class:`~tensorrt_llm.serve.detokenize.IncrementalDetokenizer` matches the
  stop strings and ends the output at the first match; the output's tokens
  and logprobs are cut at the token that completed it (without the stop
  string's own tokens when they matched and are not to be kept, as ``LLM``
  drops them), so the completion token count the handlers derive from
  ``token_ids`` counts no token generated past it (for text the decoder held
  back, the token that released it). Tokens that may start a stop string's
  token ids are held back, text, ids and logprobs alike, until the match is
  decided, so nothing streamed is taken back. With ``ignore_eos``, as
  ``LLM`` gives its engine no stop words then, the stop strings are looked
  for in the text only.

Differences from ``LLM``:

- with the engine matching: a stop string found only in the text (its token
  ids are not the ones generated) ends ``LLM``'s streamed output at once,
  while here the output waits for the engine to end it, as ``LLM``'s
  non-streamed output does; its text is the same. This difference arises
  with an engine that matches stop strings on their token ids only, as
  ``LLM``'s engine does. A cancelled output has
  its text cut at a stop string it holds, as the text streamed so far was;
  ``LLM`` does not cut a cancelled output;
- with the serving side matching: several stop strings, a match found only
  in the text, and stop token ids that start in the prompt; see
  :class:`~tensorrt_llm.serve.detokenize.IncrementalDetokenizer`. ``LLM``'s
  non-streamed output keeps the tokens generated past a match found only in
  the text. A cancelled output whose detokenizer still holds text that
  completes a stop string ends at that stop string; ``LLM`` ends a cancelled
  output with ``cancelled``. The engine does not know the stop strings, so,
  unlike ``LLM``'s sampler, it does not ban a single-token stop string's
  token before ``min_tokens``; when generated, that token stays in the
  output. When the serving side ends an output at a stop string, the facade
  cancels the engine's calls, so the usage the engine would send at their
  end, and with it ``cached_tokens``, may not arrive (``cached_tokens`` is
  then 0);
- ``prompt_logprobs`` lacks ``LLM``'s entry for the first generated token;
- the text, where the detokenizer's decoding differs from ``decode``; see
  :class:`~tensorrt_llm.serve.detokenize.IncrementalDetokenizer`.

The module is imported in postprocessing processes as well, so it does not
import the gRPC stack or the facade.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .engine_client import EngineEvent, OutputFinished, PromptInfo, TokenDelta, TokenLogprob

if TYPE_CHECKING:
    from tensorrt_llm.executor.result import CompletionOutput
    from tensorrt_llm.sampling_params import SamplingParams
    from tensorrt_llm.serve.detokenize import IncrementalDetokenizer

__all__ = ["OpenEngineResponseAssembler", "create_response_assembler"]


def create_response_assembler(
    *,
    sampling_params: SamplingParams,
    tokenizer: Any,
    num_outputs: int,
    engine_stop_texts: bool = False,
) -> OpenEngineResponseAssembler:
    """The assembler of one request returning ``num_outputs`` outputs.

    Args:
        engine_stop_texts: The request's stop strings were sent to the engine,
            which matches them; otherwise the serving side does.

    Raises:
        TypeError: The outputs need the tokenizer and it is not a
            ``TransformersTokenizer``, or a stop string is not a ``str``.
        ValueError: A stop string is empty; reported as HTTP 400.
    """
    return OpenEngineResponseAssembler(
        sampling_params=sampling_params,
        tokenizer=tokenizer,
        num_outputs=num_outputs,
        engine_stop_texts=engine_stop_texts,
    )


class OpenEngineResponseAssembler:
    """Builds the outputs of one request from the events of its Generate calls.

    Events carry ``output_index`` over the request's outputs. ``cached_tokens``
    is the engine's count of prompt tokens served from its cache, from the
    first usage it reports. The ``*_diff`` views of :attr:`outputs` cover the
    change made by the last :meth:`apply` that returned True, or by
    :meth:`finish_cancelled`.

    See :func:`create_response_assembler` for the arguments.
    """

    def __init__(
        self,
        *,
        sampling_params: SamplingParams,
        tokenizer: Any,
        num_outputs: int,
        engine_stop_texts: bool = False,
    ) -> None:
        from tensorrt_llm.executor.result import CompletionOutput
        from tensorrt_llm.serve.detokenize import normalize_stop
        from tensorrt_llm.tokenizer import TransformersTokenizer

        self.outputs: list[CompletionOutput] = [
            CompletionOutput(index=i) for i in range(num_outputs)
        ]
        # Per output, the token ids and logprobs held back (see the module docstring).
        self._held: list[tuple[list[int], list[Any]]] = [([], []) for _ in range(num_outputs)]
        self.cached_tokens = 0
        self._usage_seen = False
        self._engine_stop = engine_stop_texts
        self._include_stop = bool(sampling_params.include_stop_str_in_output)
        stop = normalize_stop(sampling_params.stop)
        detokenize = bool(sampling_params.detokenize)
        # With ignore_eos, LLM's engine gets no stop words, so the serving side matches no token ids.
        self._match_token_ids = bool(
            stop and not engine_stop_texts and not sampling_params.ignore_eos
        )
        # Without text, only a serving-side match on token ids needs a detokenizer.
        self._needs_detokenizer = bool(detokenize or self._match_token_ids)
        if self._needs_detokenizer and not isinstance(tokenizer, TransformersTokenizer):
            raise TypeError(
                f"tokenizer must be a TransformersTokenizer, got {type(tokenizer).__name__}"
            )
        self._tokenizer = tokenizer
        self._stop = stop
        self._detokenizer_options: dict[str, Any] = dict(
            stop=stop,
            include_stop_str_in_output=self._include_stop,
            skip_special_tokens=sampling_params.skip_special_tokens,
            spaces_between_special_tokens=sampling_params.spaces_between_special_tokens,
            min_tokens=sampling_params.min_tokens,
            engine_stop=engine_stop_texts,
            detokenize=detokenize,
        )
        # Created at an output's first tokens: outputs postprocessed elsewhere
        # (in a postprocessing process) never need theirs.
        self._detokenizers: list[IncrementalDetokenizer | None] = [None] * num_outputs
        self._logprobs = sampling_params.logprobs
        self._logprobs_simple = bool(getattr(sampling_params, "logprobs_simple_format", False))
        self._prompt_logprobs = sampling_params.prompt_logprobs
        self._prompt_logprobs_simple = bool(
            getattr(sampling_params, "prompt_logprobs_simple_format", False)
        )

    @property
    def finished(self) -> bool:
        """Whether every output has finished."""
        return all(output.finish_reason is not None for output in self.outputs)

    def apply(self, event: EngineEvent) -> bool:
        """Apply one event; return whether any output changed.

        An unfinished output that receives tokens counts as changed even when
        all their text and ids are held back, as ``LLM`` gives a response for
        every step. A ``PromptInfo`` event sets ``prompt_logprobs`` and returns
        False: it carries nothing to send by itself.
        """
        if isinstance(event, TokenDelta):
            return self._apply_tokens(event)
        if isinstance(event, OutputFinished):
            if event.usage is not None and not self._usage_seen:
                self._usage_seen = True
                self.cached_tokens = event.usage.cached_prompt_tokens or 0
            return self._apply_finished(event)
        if isinstance(event, PromptInfo):
            self._apply_prompt(event)
        return False

    def finish_cancelled(self) -> None:
        """End the outputs that have not finished, after the serving side cancelled the request.

        They end with ``cancelled``, as with ``LLM``, except when the serving
        side matches the stop strings and the text a detokenizer still holds
        completes one: the output then ends at that stop string. With the
        engine matching, the text is cut at the stop string it holds, as the
        text streamed so far was, and the output ends with ``cancelled``.
        """
        self._start_step()
        for index, (output, detokenizer) in enumerate(zip(self.outputs, self._detokenizers)):
            if output.finish_reason is not None:
                continue
            # None also when the output got no tokens: there is no text to finish.
            if detokenizer is not None:
                delta = detokenizer.finish()
                output.text += delta.text
                if delta.stop is not None and not self._engine_stop:
                    self._end_at_stop(index, delta.stop)
                    continue
            self._release_tokens(index, 0)
            output.finish_reason = "cancelled"

    # -- events --------------------------------------------------------------

    def _start_step(self) -> None:
        for output in self.outputs:
            output._last_text_len = len(output.text)
            output._last_token_ids_len = len(output.token_ids)
            output._last_logprobs_len = len(output.logprobs)

    def _apply_tokens(self, event: TokenDelta) -> bool:
        output = self.outputs[event.output_index]
        if output.finish_reason is not None:
            return False
        self._start_step()
        held_ids, held_logprobs = self._held[event.output_index]
        held_ids.extend(event.token_ids)
        if self._logprobs is not None and event.logprobs is not None:
            held_logprobs.extend(self._output_logprob(info) for info in event.logprobs)
        detokenizer = self._detokenizer(event.output_index)
        hold = 0
        if detokenizer is not None:
            delta = detokenizer.step(event.token_ids)
            output.text += delta.text
            if delta.stop is not None:
                self._end_at_stop(event.output_index, delta.stop)
                return True
            hold = detokenizer.held_tokens
        self._release_tokens(event.output_index, hold)
        return True

    def _apply_finished(self, event: OutputFinished) -> bool:
        output = self.outputs[event.output_index]
        if output.finish_reason is not None:
            return False
        self._start_step()
        # None also when the output got no tokens: there is no text to finish.
        detokenizer = self._detokenizers[event.output_index]
        if detokenizer is not None:
            delta = detokenizer.finish()
            output.text += delta.text
            # With the engine matching, as with LLM, a cancelled output gets no
            # stop_reason from its text (the fallback differs: see the module docstring).
            if delta.stop is not None and not (self._engine_stop and event.reason == "cancelled"):
                self._end_at_stop(event.output_index, delta.stop)
                return True
        self._release_tokens(event.output_index, 0)
        output.finish_reason = event.reason
        if event.stop_token_id is not None:
            output.stop_reason = event.stop_token_id
        elif event.stop_text is not None:
            output.stop_reason = event.stop_text
        return True

    def _detokenizer(self, index: int) -> IncrementalDetokenizer | None:
        """The detokenizer of output ``index``, created on first use; None when not needed."""
        if not self._needs_detokenizer:
            return None
        detokenizer = self._detokenizers[index]
        if detokenizer is None:
            from tensorrt_llm.serve.detokenize import IncrementalDetokenizer, encode_stop

            if not self._engine_stop and "encoded_stop" not in self._detokenizer_options:
                # Encoded once for all outputs of the request.
                self._detokenizer_options["encoded_stop"] = (
                    encode_stop(self._tokenizer, self._stop) if self._match_token_ids else []
                )
            detokenizer = IncrementalDetokenizer(self._tokenizer, **self._detokenizer_options)
            self._detokenizers[index] = detokenizer
        return detokenizer

    def _release_tokens(self, index: int, hold: int) -> None:
        """Move the held token ids and logprobs of output ``index`` into it, all but the last ``hold``."""
        output = self.outputs[index]
        for held, released in zip(self._held[index], (output.token_ids, output.logprobs)):
            count = len(held) - hold
            if count > 0:
                released.extend(held[:count])
                del held[:count]

    def _end_at_stop(self, index: int, stop: Any) -> None:
        """End output ``index`` at a stop string its detokenizer found.

        The output keeps the tokens up to the one that completed the match,
        without the stop string's own tokens when they matched and are not to
        be kept; its logprobs are cut at the same place. A match the
        detokenizer reports at :meth:`~IncrementalDetokenizer.finish` with
        ``engine_stop`` ends at the last token, so the engine's tokens stay.
        """
        self._release_tokens(index, 0)
        output = self.outputs[index]
        keep = stop.token_index + 1
        if not self._include_stop:
            keep -= stop.num_stop_tokens
        del output.token_ids[keep:]
        del output.logprobs[keep:]
        output.finish_reason = "stop"
        output.stop_reason = stop.stop_str

    def _apply_prompt(self, event: PromptInfo) -> None:
        """Prompt logprobs, as ``LLM`` gives them: one entry per prompt token after the first.

        ``LLM`` adds one more entry, for the first generated token; the
        OpenEngine prompt event does not carry it.
        """
        if self._prompt_logprobs is None:
            return
        entries = [self._prompt_logprob(info) for info in event.token_logprobs[1:]]
        for output in self.outputs:
            output.prompt_logprobs = list(entries)

    # -- logprobs ------------------------------------------------------------

    def _output_logprob(self, info: TokenLogprob) -> Any:
        """One generated token's logprobs: the sampled token first, then the top candidates."""
        from tensorrt_llm.executor.result import Logprob

        if self._logprobs_simple:
            return info.logprob
        entry = {info.token_id: Logprob(logprob=info.logprob, rank=info.rank)}
        for rank, (token_id, logprob) in enumerate(_top(info, self._logprobs), start=1):
            entry[token_id] = Logprob(logprob=logprob, rank=rank)
        return entry

    def _prompt_logprob(self, info: TokenLogprob) -> Any:
        """One prompt token's logprobs: the top candidates, then the token if not among them."""
        from tensorrt_llm.executor.result import Logprob

        if self._prompt_logprobs_simple:
            return info.logprob
        entry = {
            token_id: Logprob(logprob=logprob, rank=rank)
            for rank, (token_id, logprob) in enumerate(_top(info, self._prompt_logprobs), start=1)
        }
        if info.token_id not in entry:
            entry[info.token_id] = Logprob(logprob=info.logprob, rank=info.rank)
        return entry


def _top(info: TokenLogprob, top_n: int | None) -> Sequence[tuple[int, float]]:
    return info.top[: top_n or 0]

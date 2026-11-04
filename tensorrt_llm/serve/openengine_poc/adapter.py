# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A generator for ``OpenAIServer`` that runs requests on an external OpenEngine engine.

``OpenEngineLLMAdapter`` stands in for ``LLM``. Besides its own lifecycle
(``from_llm_args``, ``start``, ``aclose``) and its engine ``client``, it
provides only the members that ``OpenAIServer`` uses to build itself, report
health, serve chat and completion requests, and shut down. The engine sees
token ids only: tokenization, detokenization and stop-string matching happen on
the serving side.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import grpc

from tensorrt_llm.executor.result import CompletionOutput
from tensorrt_llm.grpc.openengine.bindings import error_pb2, generation_pb2, lifecycle_pb2
from tensorrt_llm.grpc.openengine_client import (
    GenerateStream,
    OpenEngineClient,
    OpenEngineClientError,
    OpenEngineEngineError,
    OpenEngineRpcError,
)
from tensorrt_llm.inputs.data import prompt_inputs
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.detokenize import IncrementalDetokenizer
from tensorrt_llm.tokenizer import TransformersTokenizer, tokenizer_factory

from .mapping import build_generate_request, builtin_capabilities

HEALTH_POLL_SECONDS = 5.0
LOG_PROMPT_IDS_ENV = "TRTLLM_OPENENGINE_POC_LOG_PROMPT_IDS"
_HEALTHY_STATES = (lifecycle_pb2.HEALTH_STATE_READY, lifecycle_pb2.HEALTH_STATE_DEGRADED)
# Serving options the OpenEngine path cannot honour; from_llm_args forces these values.
_FORCED_ARGS = {
    "num_postprocess_workers": 0,
    "return_perf_metrics": False,
    "perf_metrics_output_dir": None,
    "enable_iter_perf_stats": False,
    "enable_energy_metrics": False,
    "enable_resource_governor": False,
    "max_beam_width": 1,
}
_MULTIMODAL_KEYS = ("multi_modal_data", "multi_modal_embeddings", "mm_processor_kwargs")
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


class OpenEngineLLMAdapter:
    """Stands in for ``LLM`` in ``OpenAIServer`` and forwards generation to an OpenEngine engine.

    Build it outside the event loop, then ``await start()`` on the loop that
    serves requests before the server accepts them.

    Args:
        args: The ``TorchLlmArgs`` of the deployment; ``OpenAIServer`` reads its
            serving options from it.
        tokenizer: The tokenizer matching the engine's model.
        input_processor: Tokenizes text prompts, as for ``LLM``.
        hf_model_dir: Directory holding the model's config and tokenizer files.
        endpoint: The engine's ``host:port``.
    """

    def __init__(
        self,
        *,
        args: Any,
        tokenizer: Any,
        input_processor: Any,
        hf_model_dir: str,
        endpoint: str,
    ) -> None:
        if not isinstance(tokenizer, TransformersTokenizer):
            # The serving-side detokenizer only works on Hugging Face tokenizers.
            raise TypeError(
                "serving through an OpenEngine engine needs a Hugging Face tokenizer, got "
                f"{type(tokenizer).__name__}"
            )
        self.args = args
        self.tokenizer = tokenizer
        self.input_processor = input_processor
        self._hf_model_dir = hf_model_dir
        # OpenAIServer.register_routes reads the executor's resource governor.
        self._executor = SimpleNamespace(resource_governor_queue=None)
        self._endpoint = endpoint
        self._client: OpenEngineClient | None = None
        self._healthy = False
        self._health_task: asyncio.Task | None = None
        self._closing = False

    @classmethod
    def from_llm_args(cls, llm_args: dict[str, Any], endpoint: str) -> OpenEngineLLMAdapter:
        """Build the adapter from the ``trtllm-serve`` LLM arguments without loading weights."""
        from tensorrt_llm.inputs.registry import create_input_processor
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
        from tensorrt_llm.llmapi.llm_utils import ModelLoader

        llm_args = dict(llm_args)
        llm_args.pop("build_config", None)
        for name, value in _FORCED_ARGS.items():
            if llm_args.get(name) not in (None, value):
                logger.warning(
                    f"{name}={llm_args[name]!r} is not supported with an OpenEngine engine; "
                    f"using {value!r}"
                )
            llm_args[name] = value
        model_dir = Path(str(llm_args.get("model")))
        if not model_dir.is_dir():
            raise ValueError(
                f"--openengine-endpoint needs a local model directory for the tokenizer and "
                f"config, got {llm_args.get('model')!r}"
            )
        args = TorchLlmArgs(**llm_args)
        if args.skip_tokenizer_init:
            raise ValueError("--openengine-endpoint needs the tokenizer; do not skip its loading")
        tokenizer = args.tokenizer
        if tokenizer is None:
            tokenizer = ModelLoader.load_hf_tokenizer(
                str(model_dir),
                trust_remote_code=args.trust_remote_code,
                use_fast=args.tokenizer_mode != "slow",
            )
        if tokenizer is None:
            raise ValueError(f"no tokenizer could be loaded from {model_dir}")
        input_processor = create_input_processor(
            str(model_dir),
            tokenizer,
            getattr(args, "checkpoint_format", None),
            trust_remote_code=args.trust_remote_code,
            enable_tokenization_cache=args.enable_tokenization_cache,
        )
        # Some tokenizers raise on len(), so compare with None instead of testing truth.
        if getattr(input_processor, "tokenizer", None) is not None:
            # A multimodal input processor can hold a bare Hugging Face tokenizer.
            tokenizer = tokenizer_factory(input_processor.tokenizer)
        return cls(
            args=args,
            tokenizer=tokenizer,
            input_processor=input_processor,
            hf_model_dir=str(model_dir),
            endpoint=endpoint,
        )

    @property
    def client(self) -> OpenEngineClient:
        if self._client is None:
            raise RuntimeError("the OpenEngine engine is not connected; call start() first")
        return self._client

    async def start(self, *, ready_timeout: float = 600.0) -> None:
        """Connect to the engine, wait until it is ready and start watching its health."""
        logger.info(
            f"Waiting up to {ready_timeout:.0f} s for the OpenEngine engine at {self._endpoint}"
        )
        self._client = await OpenEngineClient.connect(self._endpoint, ready_timeout=ready_timeout)
        info = self._client.server_info
        model = self._client.model_info
        # A capability the engine does not declare counts as unsupported.
        if not (model.HasField("supports_token_ids_input") and model.supports_token_ids_input):
            await self._client.close()
            raise ValueError(
                f"OpenEngine engine {info.engine_name} at {self._endpoint} does not declare "
                "token id input"
            )
        logger.info(
            f"Connected to OpenEngine engine {info.engine_name} {info.engine_version} at "
            f"{self._endpoint}: model {model.served_model_name!r}, max_context_length="
            f"{model.max_context_length if model.HasField('max_context_length') else 'unset'}"
        )
        self._healthy = True
        self._health_task = asyncio.create_task(self._watch_health())
        self._health_task.add_done_callback(self._health_watch_ended)

    def _health_watch_ended(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                f"Stopped watching the health of the OpenEngine engine at {self._endpoint}; "
                f"reporting it unhealthy: {task.exception()!r}"
            )

    async def _watch_health(self) -> None:
        while not self._closing:
            await asyncio.sleep(HEALTH_POLL_SECONDS)
            try:
                state = (await self.client.health()).state
                healthy = state in _HEALTHY_STATES
            except OpenEngineClientError as error:
                logger.warning(f"OpenEngine health check of {self._endpoint} failed: {error}")
                healthy = False
            if healthy != self._healthy:
                logger.warning(
                    f"OpenEngine engine at {self._endpoint} is now "
                    f"{'healthy' if healthy else 'unhealthy'}"
                )
            self._healthy = healthy

    def _check_health(self) -> bool:
        # The watcher only stops on close; if it died, the engine state is unknown.
        watching = self._health_task is not None and not self._health_task.done()
        return self._client is not None and self._healthy and watching and not self._closing

    async def aclose(self) -> None:
        """Stop watching health, cancel open streams and close the channel. Safe to call again."""
        self._closing = True
        task, self._health_task = self._health_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._client is not None:
            await self._client.close()

    def shutdown(self) -> None:
        """Called by ``OpenAIServer`` on exit; the launcher finishes the close with :meth:`aclose`."""
        self._closing = True

    def _prompt_token_ids(self, inputs: Any, sampling_params: SamplingParams) -> list[int]:
        prompt = prompt_inputs(inputs)
        if any(prompt.get(key) is not None for key in _MULTIMODAL_KEYS):
            raise ValueError(
                "multimodal inputs are not supported when serving through an OpenEngine engine"
            )
        if prompt.get("prompt_token_ids") is not None:
            return list(prompt["prompt_token_ids"])
        prompt_token_ids, extra = self.input_processor(prompt, sampling_params)
        if extra and any(extra.get(key) for key in ("multimodal_data", "multimodal_input")):
            raise ValueError(
                "multimodal inputs are not supported when serving through an OpenEngine engine"
            )
        return list(prompt_token_ids)

    def preprocess(
        self,
        inputs: Any,
        sampling_params: SamplingParams | None = None,
        disaggregated_params: Any = None,
    ) -> dict[str, list[int]]:
        """Tokenize a prompt; ``OpenAIServer`` calls this in its input-processing thread pool.

        Returns:
            A token prompt to pass to :meth:`generate_async`.
        """
        if disaggregated_params is not None:
            raise ValueError(
                "disaggregated_params is not supported when serving through an OpenEngine engine"
            )
        return {
            "prompt_token_ids": self._prompt_token_ids(inputs, sampling_params or SamplingParams())
        }

    def generate_async(
        self,
        inputs: Any,
        sampling_params: SamplingParams | None = None,
        *,
        streaming: bool = False,
        lora_request: Any = None,
        disaggregated_params: Any = None,
        cache_salt: Any = None,
        kv_hints: Any = None,
        _postproc_params: Any = None,
        **_ignored: Any,
    ) -> OpenEngineRequestOutput:
        """Start one request on the engine.

        ``priority``, ``scheduling_params``, ``conversation_params`` and
        ``trace_headers`` are accepted and not forwarded. ``streaming`` does not
        change anything: the returned object can be iterated or awaited.
        ``_postproc_params`` is ignored: postprocess workers are disabled, so
        ``OpenAIServer`` runs the postprocessor itself.

        Raises:
            ValueError: The request uses something this path does not support.
        """
        del streaming, _postproc_params
        for name, value in (
            ("lora_request", lora_request),
            ("disaggregated_params", disaggregated_params),
            ("cache_salt", cache_salt),
            ("kv_hints", kv_hints),
        ):
            if value is not None:
                raise ValueError(
                    f"{name} is not supported when serving through an OpenEngine engine"
                )
        sampling_params = sampling_params or SamplingParams()
        client = self.client
        prompt_token_ids = self._prompt_token_ids(inputs, sampling_params)
        request = build_generate_request(
            prompt_token_ids,
            sampling_params,
            client.model_info,
            engine_name=client.server_info.engine_name,
            builtin=builtin_capabilities(client.server_info),
        )
        if not sampling_params.detokenize and sampling_params.stop:
            # Stop strings are matched on the text this adapter decodes.
            raise ValueError(
                "stop strings are not supported when serving through an OpenEngine engine "
                "without detokenizing (detokenize=False, or a harmony model)"
            )
        detokenizers = [
            None
            if not sampling_params.detokenize
            else IncrementalDetokenizer(
                self.tokenizer,
                stop=sampling_params.stop,
                include_stop_str_in_output=sampling_params.include_stop_str_in_output,
                skip_special_tokens=sampling_params.skip_special_tokens,
                spaces_between_special_tokens=sampling_params.spaces_between_special_tokens,
            )
            for _ in range(sampling_params.n)
        ]
        stream = client.generate(request)
        if os.environ.get(LOG_PROMPT_IDS_ENV) == "1":
            # Opt-in, for comparing prompt rendering with another server; the ids
            # encode the prompt, so this is off by default.
            logger.info(
                f"OpenEngine request {stream.request_id} prompt_token_ids={prompt_token_ids}"
            )
        return OpenEngineRequestOutput(stream, prompt_token_ids, detokenizers)

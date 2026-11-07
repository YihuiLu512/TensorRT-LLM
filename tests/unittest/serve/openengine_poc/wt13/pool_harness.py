# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A minimal OpenEngine facade over ``FakeEngineClient`` for the postprocessing-pool tests.

It follows the facade's postprocessing contract (``contracts.PostprocHost``):
with a pool it hands a request's events to the pool and yields one step per
result; without one it assembles the outputs itself and the handler
postprocesses. Pool processes import this module by name for its hooks,
so it must not import the test modules.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import itertools
import os
from types import SimpleNamespace
from typing import Any

from tensorrt_llm.executor.postprocessor_hook import emit, terminate
from tensorrt_llm.executor.result import CompletionOutput
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openengine_poc import contracts

# Where RecordingHook appends "<pid> <request id>" lines.
HOOK_LOG_ENV = "WT13_HOOK_LOG"


class RecordingHook:
    """Upper-cases the text and records which process ran it for which request.

    In a streaming request, a chunk with the word "AGAIN" (after upper-casing)
    terminates the request.
    """

    def __call__(self, chunk: Any) -> Any:
        log = os.environ.get(HOOK_LOG_ENV)
        if log:
            with open(log, "a") as file:
                file.write(f"{os.getpid()} {chunk.request_id}\n")
        text = chunk.text_diff.upper()
        if chunk.streaming and "AGAIN" in text:
            return terminate("hook")
        return emit(text)


class UnbuildableHook:
    """A hook whose constructor fails."""

    def __init__(self) -> None:
        raise RuntimeError("this hook cannot be built")

    def __call__(self, chunk: Any) -> Any:
        return emit(chunk.text_diff)


def serve_args(num_postprocess_workers: int, post_processor_hook: str | None = None) -> Any:
    """The ``args`` members ``OpenAIServer`` reads, with the given postprocessing options."""
    return SimpleNamespace(
        backend="pytorch",
        reasoning_parser=None,
        gather_generation_logits=False,
        guided_decoding_backend=None,
        num_postprocess_workers=num_postprocess_workers,
        cache_transceiver_config=None,
        trust_remote_code=False,
        checkpoint_format=None,
        return_perf_metrics=False,
        perf_metrics_output_dir=None,
        speculative_config=None,
        post_processor_hook=post_processor_hook,
        sleep_config=None,
        num_serve_frontends=1,
        enable_iter_perf_stats=False,
        max_beam_width=1,
        custom_tokenizer=None,
    )


class FacadeOutput:
    """The ``RequestOutput`` of one request, as the facade yields it."""

    def __init__(
        self,
        facade: PoolFacade,
        request: Any,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        streaming: bool,
        postproc_params: Any,
    ) -> None:
        self.id = self.request_id = request.request_id
        self.prompt_token_ids = prompt_token_ids
        self.cached_tokens = 0
        self.context_logits = None
        self.metrics_dict: dict[str, Any] = {}
        self.candidate_metrics = None
        self.disaggregated_params = None
        self.error: str | None = None
        self._done = False
        self._aborted = False
        self._facade = facade
        self._request = request
        self._steps = self._run(sampling_params, streaming, postproc_params)
        n = request.num_sequences
        pool = facade.postproc_pool
        self._session = None
        if pool is not None and postproc_params is not None:
            self._session = pool.open(
                self.id,
                postproc_params,
                sampling_params,
                streaming=streaming,
                num_outputs=n,
                num_prompt_tokens=len(prompt_token_ids),
            )
            self.outputs = [CompletionOutput(index=i) for i in range(n)]
        else:
            self._assembler = contracts.create_response_assembler(
                sampling_params=sampling_params, tokenizer=facade.tokenizer, num_outputs=n
            )
            self.outputs = self._assembler.outputs
        self._cancel = asyncio.Event()

    @property
    def finished(self) -> bool:
        return self._done

    def abort(self) -> None:
        self._aborted = True
        self._cancel.set()

    async def _events(self):
        async for event in self._facade.client.generate(self._request):
            if self._cancel.is_set():
                return
            yield event

    async def _run(self, sampling_params, streaming, postproc_params):
        if self._session is None:
            async for event in self._events():
                if self._assembler.apply(event):
                    self.cached_tokens = self._assembler.cached_tokens
                    self._done = self._assembler.finished
                    yield
                    if self._done:
                        return
            if not self._assembler.finished:
                if not self._cancel.is_set():
                    # As the facade does: an engine stream that ends early is an error.
                    raise RuntimeError("the engine ended the request before its outputs")
                self._assembler.finish_cancelled()
                self._done = True
                yield
            return
        session = self._session

        async def forward():
            try:
                async for event in self._events():
                    session.send(event)
            finally:
                session.close("cancelled" if self._cancel.is_set() else "done")

        task = asyncio.get_running_loop().create_task(forward())
        try:
            async for result in session:
                self.outputs[0]._postprocess_result = result.chunks
                self._done = result.is_final
                if result.is_final:
                    self._cancel.set()
                yield
        finally:
            self._cancel.set()
            await task

    def __aiter__(self) -> FacadeOutput:
        return self

    async def __anext__(self) -> FacadeOutput:
        await self._steps.__anext__()
        return self

    async def aresult(self) -> FacadeOutput:
        async for _ in self:
            pass
        return self

    def __await__(self):
        return self.aresult().__await__()


class PoolFacade:
    """The facade members ``OpenAIServer`` uses, over a ``FakeEngineClient``."""

    postproc_host = True

    def __init__(
        self, client: Any, tokenizer: Any, tokenizer_dir: str, args: Any, input_processor: Any
    ) -> None:
        self.client = client
        self.tokenizer = tokenizer
        self.args = args
        self.input_processor = input_processor
        self._hf_model_dir = tokenizer_dir
        self._executor = SimpleNamespace(resource_governor_queue=None)
        self._ids = itertools.count()
        self.postproc_pool = contracts.create_postproc_pool(
            args, num_workers=args.num_postprocess_workers, tokenizer_dir=tokenizer_dir
        )

    def _check_health(self) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    async def aclose(self) -> None:
        if self.postproc_pool is not None:
            await self.postproc_pool.aclose()

    def preprocess(
        self, inputs: Any, sampling_params: Any = None, disaggregated_params: Any = None
    ) -> dict[str, list[int]]:
        prompt = inputs if isinstance(inputs, dict) else {"prompt": inputs}
        if prompt.get("prompt_token_ids") is not None:
            return {"prompt_token_ids": list(prompt["prompt_token_ids"])}
        ids, _ = self.input_processor(prompt, sampling_params or SamplingParams())
        return {"prompt_token_ids": list(ids)}

    def generate_async(
        self,
        inputs: Any,
        sampling_params: SamplingParams | None = None,
        *,
        streaming: bool = False,
        _postproc_params: Any = None,
        **_ignored: Any,
    ) -> FacadeOutput:
        sampling_params = sampling_params or SamplingParams()
        prompt_token_ids = self.preprocess(inputs, sampling_params)["prompt_token_ids"]
        # OpenAIServer asks for logprobs=0 on every Responses request; the
        # default mapping cannot carry logprobs, and the fake engine sends none.
        mapped_params = sampling_params
        if sampling_params.logprobs == 0:
            mapped_params = copy.copy(sampling_params)
            mapped_params.logprobs = None
        (request,) = contracts.map_request(
            prompt_token_ids,
            mapped_params,
            self.client.model_info(),
            contracts.GenerateOptions(streaming=streaming),
            engine_name="fake",
        )
        # Deterministic ids, so that runs with and without a pool compare equal;
        # the first six go to both processes of a pool of two.
        request = dataclasses.replace(request, request_id=f"trtllm-{next(self._ids)}")
        return FacadeOutput(
            self, request, prompt_token_ids, sampling_params, streaming, _postproc_params
        )

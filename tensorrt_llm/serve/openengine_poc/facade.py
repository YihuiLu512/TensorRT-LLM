# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``LLM`` facade ``OpenAIServer`` serves from when the engine runs elsewhere.

:class:`OpenEngineLLM` provides the members of ``LLM`` that ``OpenAIServer``
uses and runs each request on an engine through :class:`EngineClient`: the
request is mapped to Generate calls (the ``map_request`` slot), the calls'
events are assembled into ``CompletionOutput`` objects (the
``create_response_assembler`` slot), and :class:`OpenEngineRequestOutput`
hands them to the handlers the way a streaming ``RequestOutput`` does.

Startup (:meth:`OpenEngineLLM.start`) connects, waits for the engine, checks
its schema revision and conventions version, reads its capabilities, checks
the tokenizer fingerprint and the serving prerequisites. Afterwards the facade
polls the engine's health; when the engine stays unreachable or not ready
longer than ``engine_lost_timeout``, it records a fatal error and stops the
frontend. :func:`adapt_server` adds the ``engine`` block to ``/server_info``
and answers the endpoints this path does not support with 501.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import signal
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path
from typing import Any

import grpc

from tensorrt_llm.grpc.openengine.bindings import error_pb2
from tensorrt_llm.grpc.openengine_client import (
    OpenEngineClientError,
    OpenEngineEngineError,
    OpenEngineIncompatibleServerError,
    OpenEngineProtocolError,
    OpenEngineRpcError,
)
from tensorrt_llm.grpc.openengine_conventions import (
    CONVENTIONS_VERSION,
    CONVENTIONS_VERSION_KEY,
    META_PRIORITY,
    META_TARGET_DP_RANK,
    META_TRACEPARENT,
    META_TRACESTATE,
    SERVE_COUNTER_DIR_ENV,
    compare_tokenizer_fingerprints,
    compute_tokenizer_fingerprint,
    conventions_version,
    encode_priority,
)
from tensorrt_llm.inputs.data import prompt_inputs
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams

from . import contracts
from .engine_client import EngineInfo, EngineLoad, EngineRequest, ModelCapabilities
from .engine_client_impl import EngineEventStream, OpenEngineEngineClient

__all__ = [
    "DEFAULT_CONTROL_TIMEOUT_S",
    "DEFAULT_ENGINE_LOST_TIMEOUT_S",
    "DEFAULT_READY_TIMEOUT_S",
    "HEALTH_POLL_SECONDS",
    "OpenEngineLLM",
    "OpenEngineRequestOutput",
    "OpenEngineServeError",
    "OpenEngineStartupError",
    "adapt_server",
]

DEFAULT_READY_TIMEOUT_S = 1800.0
DEFAULT_CONTROL_TIMEOUT_S = 5.0
DEFAULT_ENGINE_LOST_TIMEOUT_S = 30.0
HEALTH_POLL_SECONDS = 5.0
LOG_PROMPT_IDS_ENV = "TRTLLM_OPENENGINE_POC_LOG_PROMPT_IDS"

# Endpoints that need the engine in this process; answered with 501.
UNSUPPORTED_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("POST", "/start_profile"),
    ("POST", "/stop_profile"),
    ("POST", "/release_memory"),
    ("POST", "/resume_memory"),
    ("GET", "/memory_status"),
    ("POST", "/update_weights"),
    ("GET", "/energy_metrics"),
    ("GET", "/v1/data_transceiver_state"),
    ("POST", "/_resource_governor/truncate"),
    ("POST", "/_control/kv_cache/truncate"),
    ("POST", "/_control/kv_cache/truncate_tokens"),
)

_BAD_REQUEST_STATUSES = (grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.UNIMPLEMENTED)
_UNAVAILABLE_STATUSES = (grpc.StatusCode.RESOURCE_EXHAUSTED, grpc.StatusCode.UNAVAILABLE)
_BAD_REQUEST_CODES = (
    error_pb2.ERROR_CODE_INVALID_ARGUMENT,
    error_pb2.ERROR_CODE_UNSUPPORTED_FEATURE,
)
# gRPC's own status details name the peer, e.g. "ipv4:10.0.0.1:50051".
_GRPC_ADDRESS = re.compile(r"\b(?:ipv4|ipv6|unix|unix-abstract|dns|vsock):\S+")
_ERROR_TYPES = {
    HTTPStatus.SERVICE_UNAVAILABLE: "ServiceUnavailableError",
    HTTPStatus.GATEWAY_TIMEOUT: "GatewayTimeoutError",
    HTTPStatus.INTERNAL_SERVER_ERROR: "InternalServerError",
}
# Options the facade cannot honour at all; the frontend refuses to start with them.
_ENGINE_ONLY_OPTIONS = {
    "enable_energy_metrics": "energy metrics are measured where the engine runs",
    "enable_resource_governor": "the resource governor runs inside the engine's executor",
}


class OpenEngineStartupError(RuntimeError):
    """The frontend cannot serve from this engine; the message names the step and the engine."""


class OpenEngineServeError(RuntimeError):
    """A request failed for a reason that is not the client's; ``status_code`` is the HTTP status."""

    def __init__(self, message: str, status_code: HTTPStatus) -> None:
        super().__init__(message)
        self.status_code = status_code


def _without_addresses(text: str, target: str) -> str:
    if target:
        text = text.replace(target, "<engine>")
    return _GRPC_ADDRESS.sub("<engine>", text)


def request_error(error: OpenEngineClientError, engine_name: str) -> Exception:
    """The exception a handler reports for a failed engine call.

    ``ValueError`` (HTTP 400) when the request was at fault, else
    :class:`OpenEngineServeError` with the HTTP status. The message names the
    engine and repeats what it said, with the engine's address and any gRPC
    peer address replaced by ``<engine>``; the error as raised goes to the log.
    """
    logger.warning(f"OpenEngine engine {engine_name!r} failed a request: {error}")
    if isinstance(error, OpenEngineRpcError):
        details = _without_addresses(error.details, error.target)
        message = (
            f"OpenEngine engine {engine_name!r} failed the request ({error.code.name}): {details}"
        )
        if error.code in _BAD_REQUEST_STATUSES:
            return ValueError(message)
        if error.code in _UNAVAILABLE_STATUSES:
            return OpenEngineServeError(message, HTTPStatus.SERVICE_UNAVAILABLE)
        if error.code == grpc.StatusCode.DEADLINE_EXCEEDED:
            return OpenEngineServeError(message, HTTPStatus.GATEWAY_TIMEOUT)
        return OpenEngineServeError(message, HTTPStatus.INTERNAL_SERVER_ERROR)
    if isinstance(error, OpenEngineEngineError):
        name = (
            error_pb2.ErrorCode.Name(error.code)
            if error.code in error_pb2.ErrorCode.values()
            else error.code
        )
        engine_message = _without_addresses(error.engine_message, error.target)
        message = f"OpenEngine engine {engine_name!r} failed the request ({name}): {engine_message}"
        if error.code in _BAD_REQUEST_CODES:
            return ValueError(message)
        if error.code == error_pb2.ERROR_CODE_OVERLOADED:
            return OpenEngineServeError(message, HTTPStatus.SERVICE_UNAVAILABLE)
        return OpenEngineServeError(message, HTTPStatus.INTERNAL_SERVER_ERROR)
    if isinstance(error, OpenEngineProtocolError):
        return OpenEngineServeError(
            f"OpenEngine engine {engine_name!r} broke the OpenEngine protocol",
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    return OpenEngineServeError(
        f"the call to OpenEngine engine {engine_name!r} failed", HTTPStatus.INTERNAL_SERVER_ERROR
    )


class _FacadeExecutor:
    """The part of ``LLM._executor`` that ``OpenAIServer`` reads.

    ``_fatal_error`` is set when the engine is lost, so ``/health`` answers 503
    as it does for an in-process executor that failed. The facade then stops
    the frontend itself (``on_engine_lost``) and sets ``doing_shutdown``, which
    keeps ``OpenAIServer`` from sending a second SIGINT.
    """

    resource_governor_queue = None

    def __init__(self) -> None:
        self._fatal_error: BaseException | None = None
        self.doing_shutdown = False

    def _set_fatal_error(self, error: BaseException) -> None:
        if self._fatal_error is None:
            self._fatal_error = error


def _stop_frontend(facade: OpenEngineLLM) -> None:
    """Stop this frontend the way ``OpenAIServer`` does on a fatal executor error.

    Only one SIGINT is sent: ``OpenAIServer`` sends none while ``doing_shutdown``
    is set, and a second one would end uvicorn's graceful shutdown.
    """
    if not facade._executor.doing_shutdown:
        facade._executor.doing_shutdown = True
        signal.raise_signal(signal.SIGINT)


class OpenEngineLLM:
    """Stands in for ``LLM`` in ``OpenAIServer`` and runs requests on an OpenEngine engine.

    Build it outside the event loop (:meth:`from_llm_args`), then
    ``await start()`` on the loop that serves requests before the server
    accepts them, and ``await aclose()`` after it stops.

    Args:
        args: The ``TorchLlmArgs`` of the frontend; ``OpenAIServer`` reads its
            serving options from it.
        tokenizer: The Hugging Face tokenizer matching the engine's model.
        input_processor: Tokenizes text prompts, as for ``LLM``.
        hf_model_dir: The local model directory holding the config and tokenizer files.
        endpoint: The engine's ``host:port``.
        ready_timeout: Seconds to wait at startup for the engine to be ready.
        control_timeout: Seconds allowed for each Control RPC.
        allow_newer_schema: Accept an engine with a newer schema revision.
        engine_lost_timeout: Seconds the engine may stay unreachable or not
            ready before the frontend stops.
        health_poll_interval: Seconds between health checks.
        on_engine_lost: Called once the engine is declared lost; defaults to
            stopping the process with SIGINT.
    """

    def __init__(
        self,
        *,
        args: Any,
        tokenizer: Any,
        input_processor: Any,
        hf_model_dir: str,
        endpoint: str,
        ready_timeout: float = DEFAULT_READY_TIMEOUT_S,
        control_timeout: float = DEFAULT_CONTROL_TIMEOUT_S,
        allow_newer_schema: bool = False,
        engine_lost_timeout: float = DEFAULT_ENGINE_LOST_TIMEOUT_S,
        health_poll_interval: float = HEALTH_POLL_SECONDS,
        on_engine_lost: Callable[[OpenEngineLLM], None] = _stop_frontend,
    ) -> None:
        from tensorrt_llm.tokenizer import TransformersTokenizer

        if not isinstance(tokenizer, TransformersTokenizer):
            # The serving-side detokenizer and the fingerprint need a Hugging Face tokenizer.
            raise TypeError(
                "serving through an OpenEngine engine needs a Hugging Face tokenizer, got "
                f"{type(tokenizer).__name__}"
            )
        self.args = args
        self.tokenizer = tokenizer
        self.input_processor = input_processor
        self._hf_model_dir = hf_model_dir
        self._executor = _FacadeExecutor()
        self.disaggregated_params = None
        self._endpoint = endpoint
        self._ready_timeout = ready_timeout
        self._control_timeout = control_timeout
        self._allow_newer_schema = allow_newer_schema
        self._engine_lost_timeout = engine_lost_timeout
        self._health_poll_interval = health_poll_interval
        self._on_engine_lost = on_engine_lost
        self._client: OpenEngineEngineClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._healthy = False
        self._health_task: asyncio.Task[None] | None = None
        self._closing = False
        self._generate_sent = 0
        self._counter_dir = os.environ.get(SERVE_COUNTER_DIR_ENV) or None
        self._counter = contracts.GenerateCounter(self._counter_dir)
        self._counter_writer: ThreadPoolExecutor | None = None
        self._metrics = contracts.create_metrics_recorder(args)
        self._lora: contracts.LoraRegistry | None = None
        self._local_fingerprint: Any = None

    # -- Construction --------------------------------------------------------

    @classmethod
    def from_llm_args(cls, llm_args: dict[str, Any], endpoint: str, **kwargs: Any) -> OpenEngineLLM:
        """Build the facade from the ``trtllm-serve`` LLM arguments without loading weights.

        Raises:
            ValueError: The configuration asks for something this frontend cannot do.
        """
        from tensorrt_llm.inputs.registry import create_input_processor
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
        from tensorrt_llm.llmapi.llm_utils import ModelLoader
        from tensorrt_llm.tokenizer import tokenizer_factory

        llm_args = dict(llm_args)
        llm_args.pop("build_config", None)
        model_dir = Path(str(llm_args.get("model")))
        if not model_dir.is_dir():
            raise ValueError(
                "--openengine-endpoint needs a local model directory for the tokenizer and "
                f"config, got {llm_args.get('model')!r}"
            )
        args = TorchLlmArgs(**llm_args)
        check_frontend_options(args)
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
            **kwargs,
        )

    # -- Lifecycle -----------------------------------------------------------

    @property
    def client(self) -> OpenEngineEngineClient:
        if self._client is None:
            raise RuntimeError("the OpenEngine engine is not connected; call start() first")
        return self._client

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def generate_sent(self) -> int:
        """Generate calls this frontend has sent."""
        return self._generate_sent

    def _startup_error(self, step: str, problem: str) -> OpenEngineStartupError:
        return OpenEngineStartupError(
            f"OpenEngine startup failed at step '{step}' for the engine at {self._endpoint}: {problem}"
        )

    async def start(self) -> None:
        """Connect to and check the engine, then start watching its health.

        The checks: the schema revision, the conventions version, token-id
        input and the tokenizer fingerprint.

        Raises:
            OpenEngineStartupError: A step failed; the message names the step.
        """
        self._loop = asyncio.get_running_loop()
        logger.info(
            f"Waiting up to {self._ready_timeout:.0f} s for the OpenEngine engine at {self._endpoint}"
        )
        try:
            self._client = await OpenEngineEngineClient.connect(
                self._endpoint,
                ready_timeout=self._ready_timeout,
                control_timeout=self._control_timeout,
                allow_newer_schema=self._allow_newer_schema,
            )
        except OpenEngineIncompatibleServerError as error:
            raise self._startup_error("schema revision", str(error)) from error
        except OpenEngineClientError as error:
            raise self._startup_error("connect and wait until ready", str(error)) from error
        try:
            self._check_engine(self._client.server_info(), self._client.model_info())
        except BaseException:
            await self._client.close()
            raise
        info = self._client.server_info()
        caps = self._client.model_info()
        logger.info(
            f"Connected to OpenEngine engine {info.engine_name} {info.engine_version} "
            f"(instance {info.instance_id}) at {self._endpoint}: model "
            f"{caps.served_model_name!r}, max_context_length={caps.max_context_length}"
        )
        self._lora = contracts.create_lora_registry(self._client, caps)
        # The count file exists from the start, so a run that sends nothing still proves it.
        self._record_sent(0)
        self._healthy = True
        self._health_task = asyncio.create_task(self._watch_health())
        self._health_task.add_done_callback(self._health_watch_ended)

    def _check_engine(self, info: EngineInfo, caps: ModelCapabilities) -> None:
        """Check the conventions version, token-id input and the tokenizer fingerprint."""
        version = conventions_version(info.extra)
        if version is None:
            logger.warning(
                f"OpenEngine engine {info.engine_name!r} at {self._endpoint} reports no "
                f"{CONVENTIONS_VERSION_KEY}; using "
                f"{'the built-in capability table' if caps.extra.builtin else 'no extra capabilities'}"
            )
        elif version != CONVENTIONS_VERSION:
            raise self._startup_error(
                "conventions version",
                f"{CONVENTIONS_VERSION_KEY} differs: serve={CONVENTIONS_VERSION} engine={version}",
            )
        if not caps.supports_token_ids_input:
            raise self._startup_error(
                "capabilities",
                f"engine {info.engine_name!r} does not declare supports_token_ids_input",
            )
        engine_fingerprint = caps.tokenizer_fingerprint
        if engine_fingerprint is None:
            logger.warning(
                f"OpenEngine engine {info.engine_name!r} at {self._endpoint} reports no tokenizer "
                "fingerprint; the tokenizers are not compared"
            )
            return
        if self._local_fingerprint is None:
            # Hashing the tokenizer files takes a moment; the tokenizer does not change.
            self._local_fingerprint = compute_tokenizer_fingerprint(
                self.tokenizer, self._hf_model_dir
            )
        mismatches = compare_tokenizer_fingerprints(self._local_fingerprint, engine_fingerprint)
        if mismatches:
            raise self._startup_error(
                "tokenizer fingerprint", "; ".join(str(mismatch) for mismatch in mismatches)
            )

    def _health_watch_ended(self, task: asyncio.Task[None]) -> None:
        if task.cancelled() or task.exception() is None or self._closing:
            return
        # Without the watcher the engine's state is unknown: stop rather than linger.
        self._engine_lost(
            f"watching the health of the OpenEngine engine at {self._endpoint} failed: "
            f"{task.exception()!r}"
        )

    async def _watch_health(self) -> None:
        loop = asyncio.get_running_loop()
        lost_since: float | None = None
        while not self._closing:
            await asyncio.sleep(self._health_poll_interval)
            problem = None
            try:
                health = await self.client.health()
                if not health.serving:
                    problem = f"state {health.state}"
            except OpenEngineClientError as error:
                problem = str(error)
            if problem is None:
                try:
                    if not await self._follow_restart():
                        return
                except OpenEngineClientError as error:
                    problem = str(error)
            healthy = problem is None
            if healthy != self._healthy:
                logger.warning(
                    f"OpenEngine engine at {self._endpoint} is now "
                    f"{'healthy' if healthy else 'unhealthy: ' + str(problem)}"
                )
            self._healthy = healthy
            if healthy:
                lost_since = None
                continue
            now = loop.time()
            lost_since = now if lost_since is None else lost_since
            if now - lost_since >= self._engine_lost_timeout:
                self._engine_lost(
                    f"the OpenEngine engine at {self._endpoint} has been unavailable for "
                    f"{now - lost_since:.0f} s: {problem}"
                )
                return

    async def _follow_restart(self) -> bool:
        """Re-read the engine's identity and capabilities if it restarted.

        Returns:
            False if the restarted engine cannot be served from; it is then
            declared lost.
        """
        client = self.client
        before_info, before_caps = client.server_info(), client.model_info()
        current = await client.current_instance_id()
        if current == before_info.instance_id:
            return True
        await client.refresh()
        info, caps = client.server_info(), client.model_info()
        logger.warning(
            f"OpenEngine engine at {self._endpoint} restarted: instance {before_info.instance_id} "
            f"-> {info.instance_id}"
        )
        try:
            self._check_engine(info, caps)
        except OpenEngineStartupError as error:
            self._engine_lost(f"the restarted OpenEngine engine is not usable: {error}")
            return False
        if caps != before_caps:
            logger.error(
                f"The capabilities of the OpenEngine engine at {self._endpoint} changed when it "
                "restarted; requests are checked against the new ones"
            )
        # A restarted engine has none of the adapters registered before.
        self._lora = contracts.create_lora_registry(client, caps)
        return True

    def _engine_lost(self, reason: str) -> None:
        logger.error(f"{reason}; stopping this frontend")
        self._healthy = False
        self._executor._set_fatal_error(RuntimeError(reason))
        self._on_engine_lost(self)

    def _check_health(self) -> bool:
        # The watcher only stops on close or when the engine is lost.
        watching = self._health_task is not None and not self._health_task.done()
        return (
            self._client is not None
            and self._healthy
            and watching
            and not self._closing
            and self._executor._fatal_error is None
        )

    def check_health(self) -> bool:
        return self._check_health()

    def shutdown(self) -> None:
        """Called by ``OpenAIServer`` on exit; the launcher finishes the close with :meth:`aclose`."""
        self._closing = True
        self._executor.doing_shutdown = True

    async def aclose(self) -> None:
        """Stop watching health, cancel open calls and close the connection. Safe to call again."""
        self.shutdown()
        task, self._health_task = self._health_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._client is not None:
            await self._client.close()
        if self._counter_writer is not None:
            self._counter_writer.shutdown(wait=True)

    # -- Counting ------------------------------------------------------------

    def _record_sent(self, count: int) -> None:
        """Count ``count`` more Generate calls; the count file is written off the event loop."""
        self._generate_sent += count
        if self._counter_dir is None or self._closing:
            # Calls that end while the frontend closes are written here, after the writer stopped.
            try:
                self._counter.increment(count)
            except OSError as error:
                logger.warning(f"Could not write the OpenEngine generate_sent count: {error}")
            return
        if self._counter_writer is None:
            # One thread keeps the writes in order, so the last one holds the total.
            self._counter_writer = ThreadPoolExecutor(1, thread_name_prefix="openengine-counter")
        future = self._counter_writer.submit(self._counter.increment, count)
        future.add_done_callback(_log_counter_failure)

    def engine_block(self) -> dict[str, Any]:
        """The ``engine`` block of ``/server_info``."""
        info = self.client.server_info()
        return {
            "kind": "openengine",
            "endpoint": self._endpoint,
            "engine_name": info.engine_name,
            "engine_version": info.engine_version,
            "instance_id": info.instance_id,
            "generate_sent": self._generate_sent,
        }

    # -- Requests ------------------------------------------------------------

    def _prompt_token_ids(self, inputs: Any, sampling_params: SamplingParams) -> list[int]:
        prompt = prompt_inputs(inputs)
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
        prompt = prompt_inputs(inputs)
        tokenized: dict[str, Any] = {
            "prompt_token_ids": self._prompt_token_ids(inputs, sampling_params or SamplingParams())
        }
        # Media stay with the prompt, so generate_async maps or refuses them.
        for key in contracts._MULTIMODAL_KEYS:
            if prompt.get(key) is not None:
                tokenized[key] = prompt[key]
        return tokenized

    def _metadata(
        self, options: contracts.GenerateOptions, caps: ModelCapabilities
    ) -> dict[str, str]:
        """The gRPC metadata of a request's calls: trace headers, priority, DP rank."""
        from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY

        engine_name = self.client.server_info().engine_name
        metadata: dict[str, str] = {}
        for key in (META_TRACEPARENT, META_TRACESTATE):
            value = (options.trace_headers or {}).get(key)
            if value:
                metadata[key] = value
        priority = options.priority
        if priority is not None and priority != DEFAULT_REQUEST_PRIORITY:
            if not caps.supports_priority:
                raise ValueError(
                    f"priority={priority!r} is not supported when serving through the OpenEngine "
                    f"engine {engine_name!r}"
                )
            metadata[META_PRIORITY] = str(encode_priority(priority))
        dp_rank = getattr(options.scheduling_params, "attention_dp_rank", None)
        if dp_rank is not None:
            if not caps.supports_dp_rank_targeting:
                raise ValueError(
                    f"scheduling_params.attention_dp_rank={dp_rank!r} is not supported when "
                    f"serving through the OpenEngine engine {engine_name!r}"
                )
            metadata[META_TARGET_DP_RANK] = str(dp_rank)
        return metadata

    def _map(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        options: contracts.GenerateOptions,
    ) -> list[EngineRequest]:
        client = self.client
        caps = client.model_info()
        engine_name = client.server_info().engine_name
        requests = contracts.map_request(
            prompt_token_ids, sampling_params, caps, options, engine_name=engine_name
        )
        metadata = self._metadata(options, caps)
        for request in requests:
            # What the mapping set wins over the facade's defaults.
            request.metadata = {**metadata, **request.metadata}
        return requests

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
        conversation_params: Any = None,
        scheduling_params: Any = None,
        priority: float | None = None,
        trace_headers: Any = None,
        _postproc_params: Any = None,
        **_ignored: Any,
    ) -> OpenEngineRequestOutput:
        """Start one request on the engine.

        ``streaming`` does not change the calls, which always stream; the
        returned object can be iterated or awaited.

        Raises:
            ValueError: The request asks for something the engine cannot do; reported as 400.
        """
        del _postproc_params
        sampling_params = sampling_params or SamplingParams()
        client = self.client
        caps = client.model_info()
        engine_name = client.server_info().engine_name
        prompt = prompt_inputs(inputs)
        media = contracts.map_media(prompt, caps, engine_name=engine_name)
        prompt_token_ids = self._prompt_token_ids(inputs, sampling_params)
        options = contracts.GenerateOptions(
            streaming=streaming,
            lora_request=lora_request,
            disaggregated_params=disaggregated_params,
            cache_salt=cache_salt,
            kv_hints=kv_hints,
            conversation_params=conversation_params,
            scheduling_params=scheduling_params,
            priority=priority,
            trace_headers=trace_headers,
            media=media,
        )
        if lora_request is None:
            requests: list[EngineRequest] | None = self._map(
                prompt_token_ids, sampling_params, options
            )
            prepare = None
        else:
            # Registering the adapter is a Control call, so the mapping waits for the first read.
            requests = None

            async def prepare() -> list[EngineRequest]:
                assert self._lora is not None
                options.lora_name = await self._lora.resolve(lora_request)
                return self._map(prompt_token_ids, sampling_params, options)

        if os.environ.get(LOG_PROMPT_IDS_ENV) == "1":
            # Opt-in, for comparing prompt rendering with another server; the ids
            # encode the prompt, so this is off by default.
            logger.info(f"OpenEngine request prompt_token_ids={prompt_token_ids}")
        assembler = contracts.create_response_assembler(
            sampling_params=sampling_params, tokenizer=self.tokenizer, num_outputs=sampling_params.n
        )
        return OpenEngineRequestOutput(
            self,
            prompt_token_ids,
            assembler,
            requests=requests,
            prepare=prepare,
            engine_name=engine_name,
        )

    # -- Stats and events ----------------------------------------------------

    async def _iteration_stats(self) -> list[dict[str, Any]]:
        load: EngineLoad | None = await self.client.load()
        stat = self._metrics.iteration_stats(load)
        return [] if stat is None else [stat]

    def get_stats(self, timeout: float) -> list[dict[str, Any]]:
        """Iteration stats derived from the engine's load; called from a worker thread."""
        if self._loop is None or self._closing:
            return []
        future = asyncio.run_coroutine_threadsafe(self._iteration_stats(), self._loop)
        try:
            return future.result(max(timeout, 0) + self._control_timeout)
        except TimeoutError:
            future.cancel()
            raise

    async def get_stats_async(self, timeout: float | None = None):
        for stat in await self._iteration_stats():
            yield stat

    async def get_kv_cache_events_async(self, timeout: float | None = None):
        for event in await self._metrics.kv_cache_events(self.client, timeout):
            yield event

    def get_data_transceiver_state(self) -> bytes:
        raise contracts.OpenEngineUnsupportedError(
            "/v1/data_transceiver_state is not supported when serving through an OpenEngine engine"
        )


def _log_counter_failure(future: Any) -> None:
    error = future.exception()
    if error is not None:
        logger.warning(f"Could not write the OpenEngine generate_sent count: {error}")


_END = object()
# Strong references to the tasks reading Generate calls, so they are not
# garbage-collected while running; a task drops out when its call ends.
_RUNNING_PUMPS: set[asyncio.Task[None]] = set()


class OpenEngineRequestOutput:
    """The ``RequestOutput`` of one request, read from its Generate calls.

    Iterating it yields itself after every change to an output, as a streaming
    ``RequestOutput`` does; ``aresult()`` and ``await`` read it to the end. The
    ``CompletionOutput`` objects in :attr:`outputs` keep their ``*_diff`` views
    relative to the previous yield. When the request is served by several calls,
    output ``i`` of call ``k`` is request output ``offset_k + i``.
    """

    def __init__(
        self,
        facade: OpenEngineLLM,
        prompt_token_ids: list[int],
        assembler: contracts.ResponseAssembler,
        *,
        requests: list[EngineRequest] | None,
        prepare: Callable[[], Any] | None,
        engine_name: str,
    ) -> None:
        self._facade = facade
        self._assembler = assembler
        self._requests = requests
        self._prepare = prepare
        self._engine_name = engine_name
        self._streams: list[EngineEventStream] = []
        self._queue: asyncio.Queue[tuple[int, Any]] | None = None
        self._open_calls = 0
        self._aborted = False
        self.request_id = requests[0].request_id if requests else contracts.new_request_id()
        # The streaming postprocessors build the response id from ``id``.
        self.id = self.request_id
        self.prompt_token_ids = prompt_token_ids
        self.finished = False
        self.error: str | None = None
        self.context_logits = None
        self.candidate_metrics = None
        self.disaggregated_params = None
        self._request_metrics = facade._metrics.start_request(self.request_id, time.time())

    @property
    def outputs(self) -> list[Any]:
        return self._assembler.outputs

    @property
    def cached_tokens(self) -> int:
        return self._assembler.cached_tokens

    @property
    def metrics_dict(self) -> dict[str, Any]:
        return self._request_metrics.metrics_dict()

    @property
    def _done(self) -> bool:
        return self.finished

    def abort(self) -> None:
        """Cancel the request; the engine treats a cancelled Generate call as an abort.

        As with ``LLM``, the outputs that have not finished end with finish
        reason ``cancelled``, unless the text the assembler still holds
        completes a stop string.
        """
        if not self.finished:
            self._aborted = True
        for stream in self._streams:
            stream.cancel()

    def __aiter__(self) -> OpenEngineRequestOutput:
        return self

    async def aresult(self) -> OpenEngineRequestOutput:
        async for _ in self:
            pass
        return self

    def __await__(self):
        return self.aresult().__await__()

    async def _start(self) -> None:
        if self._requests is None:
            assert self._prepare is not None
            self._requests = await self._prepare()
            # As on the other path, the request is known by its first call's id.
            self._requests[0] = dataclasses.replace(self._requests[0], request_id=self.request_id)
        self._queue = asyncio.Queue()
        client = self._facade.client
        offset = 0
        for request in self._requests:
            stream = client.generate(request)
            self._streams.append(stream)
            pump = asyncio.create_task(self._pump(stream, offset))
            _RUNNING_PUMPS.add(pump)
            pump.add_done_callback(_RUNNING_PUMPS.discard)
            offset += request.num_sequences
        self._open_calls = len(self._streams)
        if self._aborted:
            for stream in self._streams:
                stream.cancel()

    async def _pump(self, stream: EngineEventStream, offset: int) -> None:
        """Read one call to its end into the queue.

        The call counts as sent once its first read returns. The pump keeps
        reading after the request finished, so a call the engine ends itself
        is closed by the engine; ``stream.cancel()`` only takes effect before
        the engine's last event.
        """
        queue = self._queue
        assert queue is not None
        counted = False

        def count() -> None:
            nonlocal counted
            if not counted and stream.sent:
                counted = True
                self._facade._record_sent(1)

        try:
            async for event in stream:
                count()
                queue.put_nowait((offset, event))
        except asyncio.CancelledError:
            stream.cancel()
            raise
        except Exception as error:  # noqa: BLE001 - handed to the reader
            queue.put_nowait((offset, error))
            return
        finally:
            count()
            await stream.aclose()
        queue.put_nowait((offset, _END))

    def _cancel_calls(self) -> None:
        for stream in self._streams:
            stream.cancel()

    def _fail(self, error: Exception) -> Exception:
        self.error = str(error)
        self.finished = True
        self._cancel_calls()
        return error

    async def __anext__(self) -> OpenEngineRequestOutput:
        if self.finished:
            raise StopAsyncIteration
        if self._queue is None:
            try:
                await self._start()
            except OpenEngineClientError as error:
                raise self._fail(request_error(error, self._engine_name)) from error
            except ValueError as error:
                raise self._fail(error) from error
        assert self._queue is not None
        while True:
            offset, item = await self._queue.get()
            if item is _END:
                self._open_calls -= 1
                if self._open_calls > 0:
                    continue
                if self._aborted:
                    self._assembler.finish_cancelled()
                    self.finished = True
                    return self
                raise self._fail(
                    OpenEngineServeError(
                        f"the OpenEngine engine {self._engine_name!r} ended a call before the "
                        "request finished",
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                )
            if isinstance(item, BaseException):
                if isinstance(item, OpenEngineClientError):
                    raise self._fail(request_error(item, self._engine_name)) from item
                raise self._fail(
                    OpenEngineServeError(str(item), HTTPStatus.INTERNAL_SERVER_ERROR)
                ) from item
            event = item
            if offset and hasattr(event, "output_index"):
                event = _shifted(event, offset)
            self._request_metrics.on_event(event, time.time())
            try:
                changed = self._assembler.apply(event)
            except Exception as error:
                # Whatever went wrong, the engine must not keep generating.
                raise self._fail(
                    OpenEngineServeError(
                        f"failed to apply an OpenEngine response: {error}",
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                ) from error
            if not changed:
                continue
            if self._assembler.finished:
                self.finished = True
                # Tokens past a serving-side stop string are not needed: cancelling
                # the calls aborts them on the engine; calls it ended are not affected.
                self._cancel_calls()
            return self


def _shifted(event: Any, offset: int) -> Any:
    """``event`` with its ``output_index`` moved by ``offset``."""
    import dataclasses

    return dataclasses.replace(event, output_index=event.output_index + offset)


def check_frontend_options(args: Any) -> None:
    """Refuse options this frontend cannot honour, instead of ignoring them.

    ``post_processor_hook`` and the metrics options are accepted once the slot
    of the component that implements them (the postprocessing dispatch, the
    metrics recorder) has an implementation, and refused with the reason
    until then. ``num_postprocess_workers`` is refused until the
    postprocessing pool is wired into the facade.

    Raises:
        ValueError: An option cannot be honoured.
    """
    problems = []
    for name, reason in _ENGINE_ONLY_OPTIONS.items():
        if getattr(args, name, None):
            problems.append(f"{name} is not supported: {reason}")
    if (getattr(args, "max_beam_width", 1) or 1) > 1:
        problems.append("max_beam_width > 1 is not supported: OpenEngine has no beam search")
    postproc_dispatch = contracts.resolve_slot("create_postproc_dispatch")[1] != "default"
    metrics = contracts.resolve_slot("create_metrics_recorder")[1] != "default"
    needs = [
        (
            "num_postprocess_workers",
            (getattr(args, "num_postprocess_workers", 0) or 0) > 0,
            False,
            "the OpenEngine postprocessing pool",
        ),
        (
            "post_processor_hook",
            getattr(args, "post_processor_hook", None) is not None,
            postproc_dispatch,
            "the OpenEngine postprocessing dispatch",
        ),
        (
            "return_perf_metrics",
            bool(getattr(args, "return_perf_metrics", False)),
            metrics,
            "the OpenEngine metrics recorder",
        ),
        (
            "perf_metrics_output_dir",
            getattr(args, "perf_metrics_output_dir", None) is not None,
            metrics,
            "the OpenEngine metrics recorder",
        ),
        (
            "enable_iter_perf_stats",
            bool(getattr(args, "enable_iter_perf_stats", False)),
            metrics,
            "the OpenEngine metrics recorder",
        ),
        (
            "otlp_traces_endpoint",
            getattr(args, "otlp_traces_endpoint", None) is not None,
            metrics,
            "the OpenEngine metrics recorder",
        ),
    ]
    for name, used, available, component in needs:
        if used and not available:
            problems.append(f"{name} needs {component}, which this frontend does not have yet")
    if problems:
        raise ValueError(
            "--openengine-endpoint cannot serve this configuration: " + "; ".join(problems)
        )


def adapt_server(server: Any, facade: OpenEngineLLM) -> None:
    """Adapt an ``OpenAIServer`` built on ``facade`` to the OpenEngine path.

    - ``/server_info`` gains the ``engine`` block;
    - the endpoints in :data:`UNSUPPORTED_ENDPOINTS` answer 501, as does a
      request whose handler lets ``OpenEngineUnsupportedError`` propagate;
    - a request that fails with :class:`OpenEngineServeError` is reported with
      that error's HTTP status rather than 400.
    """
    import sys

    from fastapi.responses import JSONResponse
    from fastapi.routing import APIRoute

    app = server.app

    def not_supported(path: str) -> JSONResponse:
        return server._create_not_supported_error(
            f"{path} is not supported when serving through an OpenEngine engine"
        )

    def replace_route(path: str, method: str, endpoint: Callable[..., Any]) -> None:
        routes = app.router.routes
        for index, route in enumerate(routes):
            if isinstance(route, APIRoute) and route.path == path and method in route.methods:
                del routes[index]
                break
        app.add_api_route(path, endpoint, methods=[method], include_in_schema=False)

    def unsupported_endpoint(path: str) -> Callable[[], Any]:
        async def unsupported() -> JSONResponse:
            return not_supported(path)

        return unsupported

    for method, path in UNSUPPORTED_ENDPOINTS:
        replace_route(path, method, unsupported_endpoint(path))

    original_server_info = server.get_server_info

    async def server_info() -> JSONResponse:
        response = await original_server_info()
        content = json.loads(response.body)
        content["engine"] = facade.engine_block()
        return JSONResponse(content=content)

    replace_route("/server_info", "GET", server_info)

    async def unsupported_error(request: Any, error: Exception) -> JSONResponse:
        return not_supported(request.url.path)

    app.add_exception_handler(contracts.OpenEngineUnsupportedError, unsupported_error)

    create_error_response = server.create_error_response

    def create_status_error_response(
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> Any:
        # Handlers report every failure through this method from inside their
        # ``except`` block; an engine-side failure keeps its own status.
        error = sys.exc_info()[1]
        if isinstance(error, OpenEngineServeError) and status_code == HTTPStatus.BAD_REQUEST:
            status_code = error.status_code
            err_type = _ERROR_TYPES.get(status_code, "InternalServerError")
        return create_error_response(message, err_type, status_code)

    server.create_error_response = create_status_error_response

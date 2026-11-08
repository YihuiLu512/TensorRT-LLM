# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``LLM`` facade ``OpenAIServer`` serves from when the engine runs elsewhere.

:class:`OpenEngineLLM` provides the members of ``LLM`` that ``OpenAIServer``
uses and runs each request on an engine through :class:`EngineClient`: the
request is mapped to Generate calls (the ``map_request`` slot), the calls'
events are assembled into ``CompletionOutput`` objects (the
``create_response_assembler`` slot), and :class:`OpenEngineRequestOutput`
hands them to the handlers the way a streaming ``RequestOutput`` does. With a
postprocessing pool (the ``create_postproc_pool`` slot), a request the handler
passes ``_postproc_params`` is postprocessed in the pool instead: the facade
sends the calls' events to the request's session and hands its results to the
handler in ``outputs[0]._postprocess_result``.

Startup (:meth:`OpenEngineLLM.start`) connects, waits for the engine, checks
its schema revision and conventions version, reads its capabilities, checks
the tokenizer fingerprint and the serving prerequisites. Afterwards the facade
polls the engine's health. It records a fatal error and stops the frontend
when the engine stays unreachable or not ready longer than
``engine_lost_timeout``, when a restarted engine fails the startup checks, and
when the health watcher itself fails. :func:`adapt_server` adds the ``engine``
block to ``/server_info`` and answers the endpoints this path does not support
with 501.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import json
import os
import re
import signal
import time
from collections.abc import Callable, Mapping
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
from .engine_client import EngineInfo, EngineLoad, EngineRequest, ModelCapabilities, OutputFinished
from .engine_client_impl import EngineEventStream, OpenEngineEngineClient

__all__ = [
    "DEFAULT_CONTROL_TIMEOUT_S",
    "DEFAULT_ENGINE_LOST_TIMEOUT_S",
    "DEFAULT_READY_TIMEOUT_S",
    "HEALTH_POLL_SECONDS",
    "MAX_BUFFERED_EVENTS",
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
# Where preprocess leaves a prompt's mapped media for generate_async.
_MAPPED_MEDIA_KEY = "openengine_media"


class _MappedMedia(list):
    """Media that ``preprocess`` mapped; ``generate_async`` sends only these without mapping."""


# Events a request may hold unread (results, when it is postprocessed in the pool); past
# this its calls are cancelled and it is answered 503.
MAX_BUFFERED_EVENTS = 4096

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


# What compute_tokenizer_fingerprint raises for a tokenizer or model directory it cannot read.
_FINGERPRINT_ERRORS = (NotImplementedError, OSError, TypeError, ValueError)


def _tokenizer_dir(tokenizer: Any, model_dir: Path) -> str:
    """The local directory the tokenizer files are in: ``tokenizer`` if it names one.

    A Hugging Face Hub name, or no ``tokenizer``, gives ``model_dir``.
    """
    if isinstance(tokenizer, (str, Path)) and Path(tokenizer).is_dir():
        return str(tokenizer)
    return str(model_dir)


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


# A request-data keyword the caller did not pass; None is a value OpenAIServer can pass.
_NOT_GIVEN: Any = object()


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
        hf_model_dir: The local model directory holding the config.
        tokenizer_dir: The local directory holding the tokenizer files; None for
            ``hf_model_dir``. The postprocessing pool loads its tokenizer from
            ``args.postprocess_tokenizer_dir`` if that is set, else from this
            one; it needs a local directory, so a pool is refused when
            ``postprocess_tokenizer_dir`` is something else, such as the Hugging
            Face Hub name ``trtllm-serve`` sets it to for a Hub ``--tokenizer``.
            The fingerprint is computed over ``hf_model_dir``, as the engine
            computes it over its model directory.
        endpoint: The engine's ``host:port``.
        ready_timeout: Seconds to wait at startup for the engine to be ready.
        control_timeout: Seconds allowed for each Control RPC.
        allow_newer_schema: Accept an engine with a newer schema revision.
        engine_lost_timeout: Seconds the engine may stay unreachable or not
            ready before the frontend stops.
        health_poll_interval: Seconds between health checks.
        on_engine_lost: Called once the engine is declared lost; defaults to
            stopping the process with SIGINT.
        postproc_workers: This frontend's share of ``num_postprocess_workers``,
            the size of its postprocessing pool; None for the whole of it.

    Attributes:
        postproc_pool: The postprocessing pool, or None; set once, here, and
            kept after :meth:`aclose`. ``OpenAIServer`` recognizes the facade
            by its class (``postproc_host``) and reads this when it is built.
    """

    postproc_host = True
    # A thinking budget and logit_bias reach map_request as request data.
    takes_request_data = True

    def __init__(
        self,
        *,
        args: Any,
        tokenizer: Any,
        input_processor: Any,
        hf_model_dir: str,
        endpoint: str,
        tokenizer_dir: str | None = None,
        ready_timeout: float = DEFAULT_READY_TIMEOUT_S,
        control_timeout: float = DEFAULT_CONTROL_TIMEOUT_S,
        allow_newer_schema: bool = False,
        engine_lost_timeout: float = DEFAULT_ENGINE_LOST_TIMEOUT_S,
        health_poll_interval: float = HEALTH_POLL_SECONDS,
        on_engine_lost: Callable[[OpenEngineLLM], None] = _stop_frontend,
        postproc_workers: int | None = None,
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
        self._tokenizer_dir = tokenizer_dir or hf_model_dir
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
        self._counter_writer: concurrent.futures.ThreadPoolExecutor | None = None
        self._metrics = contracts.create_metrics_recorder(args)
        # The default recorder derives nothing from the engine's load, so none is fetched.
        self._metrics_use_load = contracts.resolve_slot("create_metrics_recorder")[1] != "default"
        self._lora: contracts.LoraRegistry | None = None
        self._local_fingerprint: Any = None
        self._lost_close: asyncio.Task[None] | None = None
        # The default registry refuses every adapter; such requests fail before they start.
        self._lora_supported = contracts.resolve_slot("create_lora_registry")[1] != "default"
        # The default map_media slot refuses every media item, so it needs no references.
        self._media_supported = contracts.resolve_slot("map_media")[1] != "default"
        if postproc_workers is None:
            postproc_workers = getattr(args, "num_postprocess_workers", 0) or 0
        postproc_tokenizer_dir = getattr(args, "postprocess_tokenizer_dir", None)
        if (
            postproc_workers > 0
            and postproc_tokenizer_dir
            and not Path(postproc_tokenizer_dir).is_dir()
        ):
            # The pool would load another tokenizer than the frontend's.
            raise ValueError(
                "num_postprocess_workers > 0 with an OpenEngine engine needs the tokenizer in a "
                f"local directory, got {postproc_tokenizer_dir!r}"
            )
        self.postproc_pool: contracts.PostprocPool | None = contracts.create_postproc_pool(
            args,
            num_workers=postproc_workers,
            tokenizer_dir=postproc_tokenizer_dir or self._tokenizer_dir,
        )

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
            tokenizer_dir=_tokenizer_dir(llm_args.get("tokenizer"), model_dir),
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
        reported = info.extra.get(CONVENTIONS_VERSION_KEY)
        if version is None and reported is not None:
            raise self._startup_error(
                "conventions version",
                f"{CONVENTIONS_VERSION_KEY} is not an integer: serve={CONVENTIONS_VERSION} "
                f"engine={reported!r}",
            )
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
            # Hashed once, at startup or (off the event loop) at a restart; see _follow_restart.
            try:
                self._local_fingerprint = compute_tokenizer_fingerprint(
                    self.tokenizer, self._hf_model_dir
                )
            except _FINGERPRINT_ERRORS as error:
                raise self._startup_error(
                    "tokenizer fingerprint", f"cannot fingerprint the local tokenizer: {error!r}"
                ) from error
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
            False if the restarted engine cannot be served from (its schema
            revision or a startup check fails); it is then declared lost.
        """
        client = self.client
        before_info, before_caps = client.server_info(), client.model_info()
        current = await client.current_instance_id()
        if current == before_info.instance_id:
            return True
        try:
            await client.refresh()
        except OpenEngineIncompatibleServerError as error:
            self._engine_lost(f"the restarted OpenEngine engine is not usable: {error}")
            return False
        info, caps = client.server_info(), client.model_info()
        logger.warning(
            f"OpenEngine engine at {self._endpoint} restarted: instance {before_info.instance_id} "
            f"-> {info.instance_id}"
        )
        if self._local_fingerprint is None and caps.tokenizer_fingerprint is not None:
            # Hashing the tokenizer files takes a moment; the frontend is serving meanwhile.
            try:
                self._local_fingerprint = await asyncio.to_thread(
                    compute_tokenizer_fingerprint, self.tokenizer, self._hf_model_dir
                )
            except _FINGERPRINT_ERRORS as error:
                self._engine_lost(
                    f"the restarted OpenEngine engine cannot be checked: cannot fingerprint the "
                    f"local tokenizer: {error!r}"
                )
                return False
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
        if self._client is not None and self._loop is not None:
            # Calls to a lost engine may never end; the server only stops once its requests do.
            self._lost_close = self._loop.create_task(self._client.close())
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
        """Stop watching health, cancel open calls, close the connection and the pool.

        Safe to call again.
        """
        self.shutdown()
        task, self._health_task = self._health_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        try:
            if self._lost_close is not None:
                # The connection is closed again below whatever happened to this close.
                (lost_close,) = await asyncio.gather(self._lost_close, return_exceptions=True)
                if isinstance(lost_close, Exception):
                    logger.warning(
                        f"Closing the connection to the lost OpenEngine engine failed: {lost_close!r}"
                    )
            if self._client is not None:
                await self._client.close()
        finally:
            # The pool and the writer thread are closed whatever happened to the connection.
            try:
                if self.postproc_pool is not None:
                    await self.postproc_pool.aclose()
            finally:
                if self._counter_writer is not None:
                    # The last writes may go to a slow file system; the loop keeps running.
                    await asyncio.to_thread(self._counter_writer.shutdown, True)

    # -- Counting ------------------------------------------------------------

    def _record_sent(self, count: int) -> None:
        """Count ``count`` more Generate calls.

        With a counter directory, a writer thread updates the count file;
        while the frontend closes, the file is written on the calling thread.
        """
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
            self._counter_writer = concurrent.futures.ThreadPoolExecutor(
                1, thread_name_prefix="openengine-counter"
            )
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

    def media_references(
        self, messages: Any, media_io_kwargs: Any = None
    ) -> dict[str, list[str]] | None:
        """The media references of chat messages for the ``map_media`` slot; None to load the media.

        ``OpenAIServer`` attaches these to the prompt in place of the media it
        would download and decode: the engine fetches and decodes them itself
        (OE-MED). None when the slot is the default, which refuses all media.

        Raises:
            ValueError: With the slot implemented, a message holds media that
                cannot be sent (image embeddings, ``input_audio`` without
                data, a url that is not a string), or holds media while
                ``media_io_kwargs`` (how to decode them, which the engine does)
                is set; reported as 400.
        """
        if not self._media_supported:
            return None
        from . import media

        references = media.chat_media_references(messages)
        if references and media_io_kwargs:
            raise ValueError(
                "media_io_kwargs is not supported when serving through an OpenEngine engine"
            )
        return references

    def _media(self, inputs: Any) -> list[contracts.MediaSpec]:
        """The prompt's media as the ``map_media`` slot maps them; it refuses what it cannot map."""
        prompt = prompt_inputs(inputs)
        if _MAPPED_MEDIA_KEY in prompt:
            mapped = prompt[_MAPPED_MEDIA_KEY]
            if not isinstance(mapped, _MappedMedia):
                raise ValueError(f"{_MAPPED_MEDIA_KEY!r} is set by preprocess only")
            return list(mapped)
        client = self.client
        return contracts.map_media(
            prompt, client.model_info(), engine_name=client.server_info().engine_name
        )

    @staticmethod
    def _check_truncation(prompt: Mapping[str, Any], sampling_params: SamplingParams) -> None:
        """Refuse ``truncate_prompt_tokens`` for a prompt with media, before the media are read."""
        if sampling_params.truncate_prompt_tokens is None:
            return
        data = prompt.get("multi_modal_data")
        if prompt.get(_MAPPED_MEDIA_KEY) or (isinstance(data, Mapping) and data):
            # The prompt's placeholders are unexpanded (OE-MED-2); truncating it
            # can drop a placeholder whose media item is still sent.
            raise ValueError(
                "truncate_prompt_tokens is not supported for a prompt with media when "
                "serving through an OpenEngine engine"
            )

    def _prompt_token_ids(
        self, inputs: Any, sampling_params: SamplingParams, media: list[contracts.MediaSpec]
    ) -> list[int]:
        prompt = prompt_inputs(inputs)
        if prompt.get("prompt_token_ids") is not None:
            return list(prompt["prompt_token_ids"])
        if media:
            # The engine preprocesses the media and expands their placeholders,
            # so the prompt is only tokenized here, placeholders unexpanded (OE-MED-2).
            from tensorrt_llm.inputs.tokenization import tokenize_prompt

            return tokenize_prompt(
                self.tokenizer,
                prompt["prompt"],
                add_special_tokens=sampling_params.add_special_tokens,
            )
        prompt_token_ids, extra = self.input_processor(prompt, sampling_params)
        has_media = extra and any(extra.get(key) for key in ("multimodal_data", "multimodal_input"))
        if has_media and not media:
            # The input processor found media that the map_media slot did not map.
            raise ValueError(
                "multimodal inputs are not supported when serving through an OpenEngine engine"
            )
        return list(prompt_token_ids)

    def preprocess(
        self,
        inputs: Any,
        sampling_params: SamplingParams | None = None,
        disaggregated_params: Any = None,
    ) -> dict[str, Any]:
        """Tokenize a prompt; ``OpenAIServer`` calls this in its input-processing thread pool.

        Returns:
            A token prompt to pass to :meth:`generate_async`; with media, it
            also holds them as the ``map_media`` slot mapped them.
        """
        if disaggregated_params is not None:
            raise ValueError(
                "disaggregated_params is not supported when serving through an OpenEngine engine"
            )
        prompt = prompt_inputs(inputs)
        sampling_params = sampling_params or SamplingParams()
        self._check_truncation(prompt, sampling_params)
        # The map_media slot decides on the media before the input processor
        # spends time on them; the default slot refuses all media.
        media = self._media(prompt)
        tokenized: dict[str, Any] = {
            "prompt_token_ids": self._prompt_token_ids(inputs, sampling_params, media)
        }
        if media:
            # generate_async sends these as mapped here, without mapping (or reading) them again.
            tokenized[_MAPPED_MEDIA_KEY] = _MappedMedia(media)
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
            # The W3C trace headers are printable ASCII; others are dropped, not sent.
            if isinstance(value, str) and value and value.isascii() and value.isprintable():
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
        logit_bias: Any = None,
        reasoning_parser: str | None = _NOT_GIVEN,
        tokenizer: Any = _NOT_GIVEN,
        _postproc_params: Any = None,
        **_ignored: Any,
    ) -> OpenEngineRequestOutput:
        """Start one request on the engine.

        ``streaming`` does not change the calls, which always stream; the
        returned object can be iterated or awaited.

        With a postprocessing pool, a request given ``_postproc_params`` is
        postprocessed in it; without one, ``_postproc_params`` is ignored.

        ``logit_bias``, ``reasoning_parser`` and ``tokenizer`` are the request
        data ``OpenAIServer`` passes to a generator that takes it
        (:func:`.contracts.takes_request_data`); they go to ``map_request``.
        A caller that does not pass the reasoning parser or the tokenizer gets
        the facade's own, as ``LLM`` uses its own.

        Raises:
            ValueError: The request asks for something the engine cannot do; reported as 400.
            OpenEngineServeError: The pool cannot take the request.
        """
        sampling_params = sampling_params or SamplingParams()
        if lora_request is not None:
            if not self._lora_supported:
                raise ValueError(
                    "lora_request is not supported when serving through an OpenEngine engine"
                )
            # What the registry can refuse from the adapters registered, or being
            # registered, so far is refused here, before a streamed response starts.
            check = getattr(self._lora, "check", None)
            if check is not None:
                check(lora_request)
        engine_name = self.client.server_info().engine_name
        self._check_truncation(prompt_inputs(inputs), sampling_params)
        media = self._media(inputs)
        prompt_token_ids = self._prompt_token_ids(inputs, sampling_params, media)
        if reasoning_parser is _NOT_GIVEN:
            reasoning_parser = getattr(self.args, "reasoning_parser", None)
        if tokenizer is _NOT_GIVEN:
            tokenizer = self.tokenizer
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
            logit_bias=logit_bias,
            reasoning_parser=reasoning_parser,
            tokenizer=tokenizer,
        )
        if lora_request is None:
            requests: list[EngineRequest] | None = self._map(
                prompt_token_ids, sampling_params, options
            )
            prepare = None
        else:
            # Registering the adapter is a Control call, so the requests wait for the
            # first read. They are mapped now, under the name the registry returns for
            # a registered adapter, so a refusal comes before the adapter is registered.
            options.lora_name = getattr(lora_request, "lora_name", None)
            mapped = self._map(prompt_token_ids, sampling_params, options)
            requests = None

            async def prepare() -> list[EngineRequest]:
                assert self._lora is not None
                name = await self._lora.resolve(lora_request)
                if name == options.lora_name:
                    return mapped
                options.lora_name = name
                return self._map(prompt_token_ids, sampling_params, options)

        if os.environ.get(LOG_PROMPT_IDS_ENV) == "1":
            # Opt-in, for comparing prompt rendering with another server; the ids
            # encode the prompt, so this is off by default.
            logger.info(f"OpenEngine request prompt_token_ids={prompt_token_ids}")

        def create_assembler(engine_stop_texts: bool) -> contracts.ResponseAssembler:
            return contracts.create_response_assembler(
                sampling_params=sampling_params,
                tokenizer=self.tokenizer,
                num_outputs=sampling_params.n,
                engine_stop_texts=engine_stop_texts,
            )

        postproc = None
        if self.postproc_pool is not None and _postproc_params is not None:
            postproc = _Postproc(self.postproc_pool, _postproc_params, sampling_params, streaming)
        return OpenEngineRequestOutput(
            self,
            prompt_token_ids,
            create_assembler,
            requests=requests,
            prepare=prepare,
            postproc=postproc,
            engine_name=engine_name,
        )

    # -- Stats and events ----------------------------------------------------

    async def _iteration_stats(self) -> list[dict[str, Any]]:
        load: EngineLoad | None = None
        if self._metrics_use_load:
            try:
                load = await self.client.load()
            except OpenEngineClientError as error:
                # As in-process, /metrics answers with no entry rather than an error.
                logger.warning(f"Could not read the load of the OpenEngine engine: {error}")
                return []
        stat = self._metrics.iteration_stats(load)
        return [] if stat is None else [stat]

    def get_stats(self, timeout: float) -> list[dict[str, Any]]:
        """Iteration stats derived from the engine's load; called from a worker thread."""
        if self._loop is None or self._closing:
            return []
        future = asyncio.run_coroutine_threadsafe(self._iteration_stats(), self._loop)
        try:
            return future.result(max(timeout, 0) + self._control_timeout)
        except (TimeoutError, concurrent.futures.TimeoutError):
            # Before Python 3.11 the future raises its own TimeoutError class.
            # As in-process, no stats are reported rather than an error.
            future.cancel()
            logger.warning("Reading the OpenEngine engine's load for the stats timed out")
            return []

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
# Strong references to the tasks reading Generate calls and feeding the
# postprocessing pool, so they are not garbage-collected while running; a task
# drops out when it ends.
_RUNNING_PUMPS: set[asyncio.Task[None]] = set()


def _keep(task: asyncio.Task[None]) -> None:
    _RUNNING_PUMPS.add(task)
    task.add_done_callback(_RUNNING_PUMPS.discard)


@dataclasses.dataclass(frozen=True)
class _Postproc:
    """What :meth:`PostprocPool.open` is given for one request, besides what its calls decide."""

    pool: contracts.PostprocPool
    params: Any
    sampling_params: SamplingParams
    streaming: bool


def _engine_stop_texts(requests: list[EngineRequest]) -> bool:
    """Whether the stop strings went to the engine, which then matches them."""
    return any(request.stop_texts for request in requests)


def _postproc_failure(error: Exception) -> OpenEngineServeError:
    """The error a request ends with when postprocessing it failed."""
    status = getattr(error, "http_status", HTTPStatus.INTERNAL_SERVER_ERROR)
    return OpenEngineServeError(f"postprocessing the request failed: {error}", HTTPStatus(status))


class OpenEngineRequestOutput:
    """The ``RequestOutput`` of one request, read from its Generate calls.

    Iterating it yields itself after every change to an output, as a streaming
    ``RequestOutput`` does; ``aresult()`` and ``await`` read it to the end. The
    ``CompletionOutput`` objects in :attr:`outputs` keep their ``*_diff`` views
    relative to the previous yield. When the request is served by several calls,
    output ``i`` of call ``k`` is request output ``offset_k + i``.

    The assembler, and the pool session when there is one, are created once
    the calls are known (at once, or for a request with a LoRA adapter, at the
    first read), since they are told whether the engine matches the stop strings.

    When the request is postprocessed in the pool, the events go to its
    session instead of the assembler, and each yield carries the next result's
    ``chunks`` in ``outputs[0]._postprocess_result``; the last one is final.
    A final result cancels the calls still running as soon as it arrives, and
    at most :data:`MAX_BUFFERED_EVENTS` results wait for the reader.
    """

    def __init__(
        self,
        facade: OpenEngineLLM,
        prompt_token_ids: list[int],
        create_assembler: Callable[[bool], contracts.ResponseAssembler],
        *,
        requests: list[EngineRequest] | None,
        prepare: Callable[[], Any] | None,
        postproc: _Postproc | None,
        engine_name: str,
    ) -> None:
        self._facade = facade
        self._create_assembler = create_assembler
        self._assembler: contracts.ResponseAssembler | None = None
        self._postproc = postproc
        self._requests = requests
        self._prepare = prepare
        self._engine_name = engine_name
        self._streams: list[EngineEventStream] = []
        self._queue: asyncio.Queue[tuple[int, Any]] | None = None
        self._open_calls = 0
        self._aborted = False
        # Whether this side has cancelled the calls; a cancellation it did not ask for is an error.
        self._cancel_sent = False
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
        self._session: contracts.PostprocSession | None = None
        self._session_closed = False
        self._results: asyncio.Queue[Any] | None = None
        if requests is not None:
            self._calls_known(requests)

    def _calls_known(self, requests: list[EngineRequest]) -> None:
        """Create the assembler and open the pool session for these calls."""
        engine_stop_texts = _engine_stop_texts(requests)
        self._assembler = self._create_assembler(engine_stop_texts)
        if self._postproc is not None:
            self._open_session(self._postproc, engine_stop_texts)

    def _open_session(self, postproc: _Postproc, engine_stop_texts: bool) -> None:
        """Postprocess this request in the pool."""
        try:
            self._session = postproc.pool.open(
                self.id,
                postproc.params,
                postproc.sampling_params,
                streaming=postproc.streaming,
                num_outputs=postproc.sampling_params.n,
                num_prompt_tokens=len(self.prompt_token_ids),
                engine_stop_texts=engine_stop_texts,
            )
        except contracts.PostprocError as error:
            raise self._stopping_error(_postproc_failure(error)) from error
        except contracts.PostprocDataError as error:
            raise OpenEngineServeError(
                f"the request cannot be postprocessed: {error}", HTTPStatus.INTERNAL_SERVER_ERROR
            ) from error

    def _close_session(self, reason: str) -> None:
        if self._session is not None and not self._session_closed:
            self._session_closed = True
            self._session.close(reason)

    @property
    def outputs(self) -> list[Any]:
        # Empty for a request with a LoRA adapter until its calls start, at its first read.
        return [] if self._assembler is None else self._assembler.outputs

    @property
    def cached_tokens(self) -> int:
        return 0 if self._assembler is None else self._assembler.cached_tokens

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
        self._cancel_calls()

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
            if not self._requests:
                raise _no_calls(self._engine_name)
            # As on the other path, the request is known by its first call's id.
            self._requests[0] = dataclasses.replace(self._requests[0], request_id=self.request_id)
            self._calls_known(self._requests)
        if not self._requests:
            raise _no_calls(self._engine_name)
        self._queue = asyncio.Queue()
        client = self._facade.client
        offset = 0
        for request in self._requests:
            stream = client.generate(request)
            self._streams.append(stream)
            _keep(asyncio.create_task(self._pump(stream, offset)))
            offset += request.num_sequences
        self._open_calls = len(self._streams)
        if self._session is not None:
            self._results = asyncio.Queue()
            _keep(asyncio.create_task(self._feed()))
            _keep(asyncio.create_task(self._drain()))
        if self._aborted:
            for stream in self._streams:
                stream.cancel()

    async def _pump(self, stream: EngineEventStream, offset: int) -> None:
        """Read one call to its end into the queue.

        The call is counted as sent when its first read returns an event, or
        when reading it ends, provided the call was started (``stream.sent``),
        whether or not the engine accepted it. The pump keeps reading after the
        request finished, so a call the engine ends itself is closed by the
        engine; ``stream.cancel()`` only takes effect before the engine's last
        event.
        When the request holds :data:`MAX_BUFFERED_EVENTS` unread events, the
        pump cancels its call and hands the reader a 503 error.
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
                if queue.qsize() >= MAX_BUFFERED_EVENTS:
                    stream.cancel()
                    queue.put_nowait((offset, self._overflow_error()))
                    return
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
        self._cancel_sent = True
        for stream in self._streams:
            stream.cancel()
        self._close_session("cancelled")

    def _call_failure(self, error: BaseException) -> Exception:
        """The error a request ends with when one of its calls failed with ``error``."""
        if isinstance(error, OpenEngineClientError):
            return request_error(error, self._engine_name)
        if isinstance(error, OpenEngineServeError):
            return error
        return OpenEngineServeError(str(error), HTTPStatus.INTERNAL_SERVER_ERROR)

    def _engine_cancelled(self, event: Any) -> OpenEngineServeError | None:
        """The error for an output the engine cancelled although this side did not ask it to."""
        if (
            isinstance(event, OutputFinished)
            and event.reason == "cancelled"
            and not self._cancel_sent
        ):
            return OpenEngineServeError(
                f"the OpenEngine engine {self._engine_name!r} cancelled the request",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
        return None

    async def _feed(self) -> None:
        """Send the calls' events to the postprocessing session, in arrival order."""
        queue, results, session = self._queue, self._results, self._session
        assert queue is not None and results is not None and session is not None
        assert self._requests is not None
        num_outputs = sum(request.num_sequences for request in self._requests)
        finished_outputs: set[int] = set()
        try:
            while True:
                offset, item = await queue.get()
                if item is _END or (isinstance(item, BaseException) and self._aborted):
                    self._open_calls -= 1
                    if self._open_calls > 0:
                        continue
                    if self._aborted:
                        self._close_session("cancelled")
                    elif len(finished_outputs) == num_outputs:
                        self._close_session("done")
                    else:
                        # The calls ended before every output did: as without the pool,
                        # 503 when the frontend cut them off, else the engine's fault (500).
                        self._close_session("cancelled")
                        results.put_nowait(self._stopping_error(_ended_early(self._engine_name)))
                    return
                if isinstance(item, BaseException):
                    results.put_nowait(self._call_failure(item))
                    return
                cancelled = self._engine_cancelled(item)
                if cancelled is not None:
                    results.put_nowait(cancelled)
                    return
                event = _shifted(item, offset) if offset and hasattr(item, "output_index") else item
                if isinstance(event, OutputFinished):
                    finished_outputs.add(event.output_index)
                self._request_metrics.on_event(event, time.time())
                if not self._session_closed:
                    session.send(event)
        except contracts.PostprocError as error:
            results.put_nowait(_postproc_failure(error))
        except Exception as error:
            results.put_nowait(
                OpenEngineServeError(
                    f"failed to postprocess an OpenEngine response: {error}",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            )

    async def _drain(self) -> None:
        """Queue the session's results for the reader, then ``_END``."""
        results, session = self._results, self._session
        assert results is not None and session is not None
        try:
            async for result in session:
                if results.qsize() >= MAX_BUFFERED_EVENTS:
                    # As for unread events: the calls stop at once, and the reader fails.
                    self._cancel_calls()
                    results.put_nowait(self._overflow_error())
                    return
                results.put_nowait(result)
                if result.is_final:
                    # The calls still running are not needed any more.
                    self._cancel_calls()
        except contracts.PostprocError as error:
            results.put_nowait(_postproc_failure(error))
            return
        except Exception as error:
            results.put_nowait(
                OpenEngineServeError(
                    f"failed to postprocess an OpenEngine response: {error}",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            )
            return
        results.put_nowait(_END)

    async def _next_postprocessed(self) -> OpenEngineRequestOutput:
        results = self._results
        assert results is not None
        item = await results.get()
        if self._aborted and (item is _END or isinstance(item, BaseException)):
            # A session closed as cancelled need not give a final result, and a failure
            # after the abort ends the request as cancelled, as without the pool.
            self.outputs[0]._postprocess_result = []
            self.finished = True
            return self
        if item is _END:
            raise self._fail(
                OpenEngineServeError(
                    "postprocessing ended before the request finished",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            )
        if isinstance(item, BaseException):
            raise self._fail(item)
        self.outputs[0]._postprocess_result = item.chunks
        if item.is_final:
            self.finished = True
            self._cancel_calls()
        return self

    def _stopping_error(self, error: Exception) -> Exception:
        """``error``, or a 503 when the frontend is stopping and that caused it."""
        if isinstance(error, ValueError):
            return error
        if self._facade._executor._fatal_error is not None:
            # The calls were cancelled because the engine was declared lost.
            return OpenEngineServeError(
                f"the OpenEngine engine {self._engine_name!r} is lost; this frontend is stopping",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
        if self._facade._closing:
            # Closing the frontend cancels its calls; that is not the engine's failure.
            return OpenEngineServeError(
                "this frontend is shutting down", HTTPStatus.SERVICE_UNAVAILABLE
            )
        return error

    def _overflow_error(self) -> OpenEngineServeError:
        return OpenEngineServeError(
            f"the response of the OpenEngine engine {self._engine_name!r} was not read fast "
            f"enough ({MAX_BUFFERED_EVENTS} waiting)",
            HTTPStatus.SERVICE_UNAVAILABLE,
        )

    def _fail(self, error: Exception) -> Exception:
        error = self._stopping_error(error)
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
            except (ValueError, OpenEngineServeError) as error:
                raise self._fail(error) from error
            except Exception as error:
                raise self._fail(
                    OpenEngineServeError(
                        f"failed to start the request on the OpenEngine engine "
                        f"{self._engine_name!r}: {error}",
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                ) from error
            except BaseException:
                # Cancelled while starting: any call already sent must not keep running,
                # and reading again must not send the request a second time.
                self.finished = True
                self._cancel_calls()
                raise
        assert self._queue is not None and self._assembler is not None
        if self._session is not None:
            return await self._next_postprocessed()
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
                raise self._fail(_ended_early(self._engine_name))
            if isinstance(item, BaseException) and self._aborted:
                # The request was aborted; a call failing after that ends it as cancelled.
                self._open_calls -= 1
                if self._open_calls > 0:
                    continue
                self._assembler.finish_cancelled()
                self.finished = True
                return self
            if isinstance(item, BaseException):
                raise self._fail(self._call_failure(item)) from item
            event = item
            cancelled = self._engine_cancelled(event)
            if cancelled is not None:
                # The engine gave up on the request on its own: not a result to return.
                raise self._fail(cancelled)
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


def _ended_early(engine_name: str) -> OpenEngineServeError:
    return OpenEngineServeError(
        f"the OpenEngine engine {engine_name!r} ended a call before the request finished",
        HTTPStatus.INTERNAL_SERVER_ERROR,
    )


def _no_calls(engine_name: str) -> OpenEngineServeError:
    return OpenEngineServeError(
        f"the request mapped to no Generate call on the OpenEngine engine {engine_name!r}",
        HTTPStatus.INTERNAL_SERVER_ERROR,
    )


def _shifted(event: Any, offset: int) -> Any:
    """``event`` with its ``output_index`` moved by ``offset``."""
    return dataclasses.replace(event, output_index=event.output_index + offset)


def check_frontend_options(args: Any) -> None:
    """Refuse options this frontend cannot honour, instead of ignoring them.

    Energy metrics, the resource governor and beam search are always refused.
    An option that needs another component is accepted only when that
    component's slot has an implementation other than the default, and refused
    with the reason otherwise: ``post_processor_hook`` needs
    ``create_postproc_dispatch``, ``num_postprocess_workers`` greater than 0
    needs ``create_postproc_pool``, and the metrics and tracing options need
    ``create_metrics_recorder``.

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
    postproc_pool = contracts.resolve_slot("create_postproc_pool")[1] != "default"
    metrics = contracts.resolve_slot("create_metrics_recorder")[1] != "default"
    needs = [
        (
            "num_postprocess_workers",
            (getattr(args, "num_postprocess_workers", 0) or 0) > 0,
            postproc_pool,
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
            problems.append(f"{name} needs {component}, which this frontend does not have")
    if problems:
        raise ValueError(
            "--openengine-endpoint cannot serve this configuration: " + "; ".join(problems)
        )


def adapt_server(server: Any, facade: OpenEngineLLM) -> None:
    """Adapt an ``OpenAIServer`` built on ``facade`` to the OpenEngine path.

    - ``/server_info`` gains the ``engine`` block;
    - the endpoints in :data:`UNSUPPORTED_ENDPOINTS` answer 501, as does a
      request whose handler lets ``OpenEngineUnsupportedError`` propagate;
    - when a handler reports an :class:`OpenEngineServeError` as a 400 error
      response, the response carries that error's HTTP status instead. A
      streaming request's Generate calls start at its first read, after the
      response has started, so its status stays 200 and the handler decides
      how the error reaches the client (``/v1/completions`` sends an error
      event; ``/v1/chat/completions`` ends the stream).
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
        include_in_schema = False
        # Every route for the path and method goes, so none is matched before the new one.
        for route in list(routes):
            if isinstance(route, APIRoute) and route.path == path and method in route.methods:
                include_in_schema = include_in_schema or route.include_in_schema
                routes.remove(route)
        # The handlers build their own responses: FastAPI is not to derive a
        # response model from their return annotations.
        app.add_api_route(
            path,
            endpoint,
            methods=[method],
            include_in_schema=include_in_schema,
            response_model=None,
        )

    def unsupported_endpoint(path: str) -> Callable[[], Any]:
        async def unsupported() -> JSONResponse:
            return not_supported(path)

        return unsupported

    for method, path in UNSUPPORTED_ENDPOINTS:
        replace_route(path, method, unsupported_endpoint(path))

    original_server_info = server.get_server_info

    async def server_info() -> JSONResponse:
        response = await original_server_info()
        if response.status_code != HTTPStatus.OK:
            return response
        content = json.loads(response.body)
        content["engine"] = facade.engine_block()
        return JSONResponse(content=content, status_code=response.status_code)

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

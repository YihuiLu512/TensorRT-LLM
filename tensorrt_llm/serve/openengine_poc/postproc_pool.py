# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The postprocessing processes of an OpenEngine frontend.

A frontend that serves through the OpenEngine facade postprocesses in a pool
of its own processes: each process detokenizes a request's engine events,
matches stop strings, runs the ``post_processor_hook`` and the endpoint's
postprocessor, and returns what the handler sends. Without a pool the
handlers run the same functions in the frontend process, so the responses
are the same either way, except that only the frontend process can store a
Responses response (``OpenAIServer`` turns storing off with a pool).

This module implements the slots ``create_postproc_dispatch`` and
``create_postproc_pool`` of :mod:`.contracts`. ``OpenAIServer`` looks the
dispatch slot up, so at module level this module imports only the standard
library, the logger and the OpenEngine modules :mod:`.contracts` already
imports; the rest is imported when used.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import multiprocessing
import queue
import time
import traceback
import weakref
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from tensorrt_llm.logger import logger

from . import contracts
from .contracts import (
    POSTPROC_PENDING_EVENT_LIMIT,
    DefaultPostprocDispatch,
    PostprocError,
    PostprocOverflowError,
    PostprocWorkerError,
    PPClose,
    PPEvent,
    PPOpen,
    PPResult,
)
from .engine_client import OutputFinished

__all__ = [
    "FacadePostprocDispatch",
    "PostprocPool",
    "PostprocResult",
    "create_postproc_dispatch",
    "create_postproc_pool",
    "plain_sampling_params",
    "postprocess_step",
    "run_hook",
]

# SamplingParams fields that cannot cross into a process and that assembling
# the outputs does not read; a copy that crosses has them unset.
_SAMPLING_FIELDS_LEFT_BEHIND = ("logits_processor", "embedding_bias")
# How often an idle pool process checks that the frontend process is still running.
_PARENT_CHECK_SECONDS = 5.0


# --- Shared by the pool processes and the frontend process ----------------------


class PostprocResult:
    """What a postprocessor reads as ``rsp`` for one request.

    It holds the request's ``CompletionOutput`` objects and the request-level
    fields the postprocessors read. ``_done`` is set on the step that finished
    the last output, or by the ``post_processor_hook``'s ``terminate``.
    """

    def __init__(
        self,
        request_id: Any,
        outputs: list[Any],
        *,
        cached_tokens: int = 0,
        done: bool = False,
        abort: Callable[[], None] | None = None,
    ) -> None:
        self.id = request_id
        self.request_id = request_id
        self.outputs = outputs
        self.cached_tokens = cached_tokens
        self.context_logits = None
        self._done = done
        self._aborted = False
        self._abort = abort

    def abort(self) -> None:
        self._aborted = True
        if self._abort is not None:
            self._abort()


def postprocess_step(
    result: PostprocResult,
    post_processor: Callable[[Any, Any], Any],
    postproc_args: Any,
    *,
    hook: Any,
    streaming: bool,
) -> Any:
    """Run the hook (when there is one) and then the postprocessor on one step of a request.

    The pool processes and the frontend run this same function, so a request
    is postprocessed the same with and without a pool.
    """
    if hook is not None:
        run_hook(result, hook, streaming=streaming)
    return post_processor(result, postproc_args)


def run_hook(result: PostprocResult, hook: Any, *, streaming: bool) -> None:
    """Run the ``post_processor_hook`` on one step of a request, in place."""
    from tensorrt_llm.executor.postprocessor_hook import apply_post_processor_hook

    if not streaming:
        # A non-streaming request has one step, so the hook sees its whole
        # output as the diff, as with the in-process ``LLM``.
        for output in result.outputs:
            output._last_text_len = 0
            output._last_token_ids_len = 0
            output._last_logprobs_len = 0
    apply_post_processor_hook(hook, result, streaming=streaming)


def plain_sampling_params(sampling_params: Any) -> Any:
    """A copy of ``sampling_params`` that can cross into a pool process.

    The fields in :data:`_SAMPLING_FIELDS_LEFT_BEHIND` are unset; any other
    field that is not plain data is a program error.

    Raises:
        PostprocDataError: A field other than those is not plain data.
    """
    plain = copy.copy(sampling_params)
    for name in _SAMPLING_FIELDS_LEFT_BEHIND:
        setattr(plain, name, None)
    contracts.check_plain_data(plain, path="sampling_params")
    return plain


def _plain_postproc_args(
    postproc_args: Any, sampling_params: Any = None, plain_params: Any = None
) -> Any:
    """A copy of ``postproc_args`` without the live objects a pool process builds itself.

    ``plain_params`` is :func:`plain_sampling_params` of ``sampling_params``,
    reused when the arguments hold that same ``sampling_params``.
    """
    from tensorrt_llm.serve.postprocess_handlers import ResponsesAPIPostprocArgs

    plain = copy.copy(postproc_args)
    plain.tokenizer = None
    if isinstance(plain, ResponsesAPIPostprocArgs):
        if plain_params is not None and plain.sampling_params is sampling_params:
            plain.sampling_params = plain_params
        else:
            plain.sampling_params = plain_sampling_params(plain.sampling_params)
        processor = plain.streaming_processor
        if processor is not None:
            # Rebuilt in the pool process; the opening events were sent with
            # this creation time.
            plain.streaming_created_time = processor.response_creation_time
            plain.streaming_processor = None
    contracts.check_plain_data(plain)
    return plain


def _rebuild_live_args(postproc_args: Any, tokenizer: Any, num_prompt_tokens: int) -> None:
    """Fill in, in a pool process, what :func:`_plain_postproc_args` left out."""
    from tensorrt_llm.serve.postprocess_handlers import ResponsesAPIPostprocArgs

    postproc_args.tokenizer = tokenizer
    postproc_args.num_prompt_tokens = num_prompt_tokens
    if (
        isinstance(postproc_args, ResponsesAPIPostprocArgs)
        and postproc_args.streaming_created_time is not None
    ):
        from tensorrt_llm.serve.responses_utils import ResponsesStreamingProcessor

        postproc_args.streaming_processor = ResponsesStreamingProcessor(
            request=postproc_args.request,
            sampling_params=postproc_args.sampling_params,
            model_name=postproc_args.model,
            create_time=postproc_args.streaming_created_time,
            use_harmony=postproc_args.use_harmony,
            reasoning_parser=postproc_args.reasoning_parser,
            tool_parser=postproc_args.tool_parser,
        )


def _streaming_of(post_processor: Callable[..., Any]) -> bool:
    """Whether a handler postprocessor formats a streaming response.

    Raises:
        ValueError: ``post_processor`` is not one of the handlers' postprocessors.
    """
    from tensorrt_llm.serve import postprocess_handlers as handlers

    streaming = (
        handlers.chat_stream_post_processor,
        handlers.completion_stream_post_processor,
        handlers.chat_harmony_streaming_post_processor,
        handlers.responses_api_streaming_post_processor,
    )
    non_streaming = (
        handlers.chat_response_post_processor,
        handlers.completion_response_post_processor,
        handlers.chat_harmony_post_processor,
        handlers.responses_api_post_processor,
    )
    if post_processor in streaming:
        return True
    if post_processor in non_streaming:
        return False
    raise ValueError(
        f"unknown postprocessor {getattr(post_processor, '__name__', post_processor)!r}"
    )


@dataclass(frozen=True)
class _TokenizerSpec:
    """How a pool process loads the tokenizer.

    It loads it from the directory the frontend gives the pool, with the
    frontend's tokenizer options.
    """

    tokenizer_dir: str
    trust_remote_code: bool = False
    use_fast: bool = True
    custom_tokenizer: str | None = None

    @classmethod
    def from_args(cls, args: Any, tokenizer_dir: str) -> _TokenizerSpec:
        return cls(
            tokenizer_dir=tokenizer_dir,
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
            use_fast=getattr(args, "tokenizer_mode", "auto") != "slow",
            custom_tokenizer=getattr(args, "custom_tokenizer", None),
        )

    def load(self) -> Any:
        """The tokenizer.

        Raises:
            RuntimeError: No tokenizer was found in the tokenizer directory; loading
                may also raise what the tokenizer loaders raise.
        """
        from tensorrt_llm.tokenizer.tokenizer import load_custom_tokenizer, load_hf_tokenizer

        if self.custom_tokenizer:
            return load_custom_tokenizer(
                self.custom_tokenizer,
                self.tokenizer_dir,
                trust_remote_code=self.trust_remote_code,
                use_fast=self.use_fast,
            )
        tokenizer = load_hf_tokenizer(
            self.tokenizer_dir, trust_remote_code=self.trust_remote_code, use_fast=self.use_fast
        )
        if tokenizer is None:
            raise RuntimeError(f"cannot load a tokenizer from {self.tokenizer_dir}")
        return tokenizer


# --- In a pool process ------------------------------------------------------------


@dataclass(frozen=True)
class _Reply:
    """A pool process's answer about one request: a result, or an error, and the events it used."""

    request_id: str
    events: int
    result: PPResult | None = None
    error: str | None = None


class _Record:
    """One request in a pool process: its outputs, built from its engine events, and its postprocessor."""

    def __init__(self, message: PPOpen, tokenizer: Any, hook: Any) -> None:
        self.request_id = message.request_id
        self.streaming = message.streaming
        self.post_processor = contracts.resolve_post_processor(message.post_processor)
        self.args = message.postproc_args
        _rebuild_live_args(self.args, tokenizer, message.num_prompt_tokens)
        self.hook = hook
        self.assembler = contracts.create_response_assembler(
            sampling_params=message.sampling_params,
            tokenizer=tokenizer,
            num_outputs=message.n,
            engine_stop_texts=message.detok.engine_stop,
        )
        # A postprocessor that calls ``rsp.abort()`` (Harmony on a repeated
        # message) ends the request as a cancel does; see ``_step``.
        self._abort_requested = False
        self.result = PostprocResult(
            message.request_id, self.assembler.outputs, abort=self._request_abort
        )
        self.final = False
        # Outputs a stop string ended on this side, not reported yet.
        self._stop_hit: list[int] = []
        # The steps it hands on are those the facade hands on without a pool.
        max_tokens = message.max_tokens
        if max_tokens is None:
            max_tokens = (message.sampling_params.max_tokens,) * message.n
        self._steps = contracts.StepJoiner(self.assembler, max_tokens)

    def _request_abort(self) -> None:
        self._abort_requested = True

    def on_event(self, event: Any) -> PPResult | None:
        before = [output.finish_reason for output in self.result.outputs]
        changed = self._steps.apply(event)
        self.result.cached_tokens = self.assembler.cached_tokens
        # An output the engine ended with this event ended there, even on a
        # stop string; any other output that ends with a stop string ended here.
        ended_by_engine = isinstance(event, OutputFinished)
        for output, previous in zip(self.result.outputs, before):
            if (
                previous is None
                and output.finish_reason == "stop"
                and isinstance(output.stop_reason, str)
                and not (ended_by_engine and output.index == event.output_index)
            ):
                self._stop_hit.append(output.index)
        return self._step(changed, closing=False)

    def on_close(self, reason: Literal["done", "cancelled"]) -> PPResult | None:
        changed = False
        if not self.assembler.finished:
            if reason == "done":
                # The engine ended the calls without finishing every output; the
                # facade fails such a request without a pool too.
                raise UnfinishedOutputsError("the engine ended the request before its outputs")
            self.result._aborted = True
            self.assembler.finish_cancelled()
            changed = True
        if self._steps.release():
            changed = True
        return self._step(changed, closing=True)

    def _step(self, changed: bool, *, closing: bool) -> PPResult | None:
        if self.final:
            return None
        done = closing or self.assembler.finished
        if not done and self.streaming and not changed:
            # Nothing to send; stop strings matched while a step is held go with that step.
            return None
        stop_hit = tuple(self._stop_hit) or None
        self._stop_hit = []
        if not done and not self.streaming:
            # A non-streaming response is formatted once, at the end; until
            # then a result only reports stop strings, so calls can stop early.
            return PPResult(self.request_id, None, False, stop_hit=stop_hit) if stop_hit else None
        self.result._done = done
        chunks = self._postprocess()
        if self._abort_requested and not self.result._done:
            # As after a cancel, the outputs end as cancelled and the
            # postprocessor runs once more on the ended request, as the
            # handler runs it without a pool; both steps go in this result.
            self.assembler.finish_cancelled()
            self.result._done = True
            chunks = [*chunks, *self._postprocess()]
        # The hook's terminate ends the request too.
        self.final = self.result._done
        return PPResult(self.request_id, chunks, self.final, stop_hit=stop_hit)

    def _postprocess(self) -> Any:
        return postprocess_step(
            self.result, self.post_processor, self.args, hook=self.hook, streaming=self.streaming
        )


class UnfinishedOutputsError(RuntimeError):
    """A request was closed as done while some of its outputs had not finished."""


def _handle(records: dict[str, _Record], message: Any, tokenizer: Any, hook: Any) -> _Reply | None:
    """Apply one message from the frontend; return what to send back, if anything."""
    request_id = message.request_id
    try:
        if isinstance(message, PPOpen):
            records[request_id] = _Record(message, tokenizer, hook)
            return None
        record = records.get(request_id)
        if record is None:
            # The request already ended or failed here.
            return None
        if isinstance(message, PPEvent):
            result, events = record.on_event(message.event), 1
        elif isinstance(message, PPClose):
            result, events = record.on_close(message.reason), 0
        else:
            raise TypeError(f"unexpected message {type(message).__name__}")
        if record.final:
            del records[request_id]
        if result is None and events == 0:
            return None
        return _Reply(request_id, events, result)
    except Exception as error:  # noqa: BLE001 - one request fails, the process goes on
        # The reply leaves out the exception's message, which may quote the
        # request's content, and so does this log; a hook's own exception is
        # logged with its message by apply_post_processor_hook.
        frames = "".join(traceback.format_tb(error.__traceback__))
        logger.error(
            f"postprocessing request {request_id} failed with {type(error).__name__}:\n{frames}"
        )
        records.pop(request_id, None)
        return _Reply(request_id, 0, error=f"{type(error).__name__} in postprocessing")


def _coalesce(replies: list[_Reply]) -> list[_Reply]:
    """The replies of one batch, with the bare acknowledgements of a request summed into one."""
    kept: list[_Reply] = []
    acknowledged: dict[str, int] = {}
    for reply in replies:
        if reply.result is None and reply.error is None:
            acknowledged[reply.request_id] = acknowledged.get(reply.request_id, 0) + reply.events
        else:
            kept.append(reply)
    kept.extend(_Reply(request_id, events) for request_id, events in acknowledged.items())
    return kept


def _process_main(
    index: int,
    in_address: tuple[str, bytes],
    out_address: tuple[str, bytes],
    tokenizer_spec: _TokenizerSpec,
    hook_path: str | None,
) -> None:
    """The loop of pool process ``index``: messages in, replies out, until a None arrives."""
    import signal

    import zmq

    from tensorrt_llm.executor.ipc import ZeroMqQueue
    from tensorrt_llm.executor.postprocessor_hook import load_post_processor_hook

    # A Ctrl-C in the terminal reaches this process too; the frontend stops it,
    # and it stops by itself when the frontend is gone.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    tokenizer = tokenizer_spec.load()
    hook = load_post_processor_hook(hook_path) if hook_path else None
    inbox = ZeroMqQueue(
        in_address, socket_type=zmq.PULL, is_server=False, name=f"postproc_in_{index}"
    )
    outbox = ZeroMqQueue(
        out_address, socket_type=zmq.PUSH, is_server=False, name=f"postproc_out_{index}"
    )
    records: dict[str, _Record] = {}
    parent = multiprocessing.parent_process()
    while True:
        try:
            batch = inbox.get(timeout=_PARENT_CHECK_SECONDS)
        except queue.Empty:
            # A frontend that was killed sends no stop message.
            if parent is not None and not parent.is_alive():
                break
            continue
        except Exception as error:  # noqa: BLE001 - drop the message, keep serving
            # For example a message that fails the HMAC check, or cannot be unpickled.
            logger.error(
                f"postprocessing process {index} dropped a message: {type(error).__name__}"
            )
            if parent is not None and not parent.is_alive():
                break
            continue
        if batch is None:
            break
        replies = _coalesce(
            [r for r in (_handle(records, m, tokenizer, hook) for m in batch) if r is not None]
        )
        if replies:
            _send_replies(outbox, replies, records)
    inbox.close()
    outbox.close()


def _send_replies(outbox: Any, replies: list[_Reply], records: dict[str, _Record]) -> None:
    """Send a batch of replies; a reply that cannot be sent fails its request instead."""
    try:
        outbox.put(replies)
        return
    except Exception:  # noqa: BLE001 - find the reply that cannot be sent
        pass
    for reply in replies:
        try:
            outbox.put([reply])
        except Exception as error:  # noqa: BLE001 - fail this request only
            logger.error(
                f"could not send the result of postprocessing request {reply.request_id}: "
                f"{type(error).__name__}"
            )
            records.pop(reply.request_id, None)
            failure = _Reply(
                reply.request_id, reply.events, error=f"{type(error).__name__} in postprocessing"
            )
            try:
                outbox.put([failure])
            except Exception as again:  # noqa: BLE001 - the other replies still go
                logger.error(
                    f"could not report the failure of postprocessing request "
                    f"{reply.request_id}: {type(again).__name__}"
                )


# --- In the frontend process ------------------------------------------------------


def _route(request_id: str, size: int) -> int:
    """The process a request goes to: a stable hash of its id."""
    digest = hashlib.blake2b(request_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % size


class _Session:
    """One request's postprocessing, seen from the frontend (``contracts.PostprocSession``)."""

    def __init__(self, pool: PostprocPool, request_id: str, worker: _Worker) -> None:
        self._pool = pool
        self.request_id = request_id
        self.worker = worker
        self.pending = 0
        self.closed = False
        self.ended = False
        self._results: asyncio.Queue[PPResult | BaseException] = asyncio.Queue()

    def send(self, event: Any) -> None:
        if self.closed or self.ended:
            return
        if self.pending >= POSTPROC_PENDING_EVENT_LIMIT:
            error = PostprocOverflowError(
                f"postprocessing fell {self.pending} events behind on one request"
            )
            # The session is over: the process drops the request, and the
            # iteration raises the same error.
            self.worker.post(PPClose(self.request_id, "cancelled"))
            self.deliver(error)
            raise error
        self.pending += 1
        self.worker.post(PPEvent(self.request_id, event))

    def close(self, reason: Literal["done", "cancelled"]) -> None:
        if self.closed:
            return
        self.closed = True
        if not self.ended:
            self.worker.post(PPClose(self.request_id, reason))

    def deliver(self, item: PPResult | BaseException) -> None:
        if self.ended:
            return
        if isinstance(item, BaseException) or item.is_final:
            self.ended = True
            self._pool._forget(self)
        self._results.put_nowait(item)

    async def __aiter__(self) -> AsyncIterator[PPResult]:
        while True:
            item = await self._results.get()
            if isinstance(item, BaseException):
                raise item
            yield item
            if item.is_final:
                return


class _Worker:
    """One pool process, the socket the frontend sends to it on, and its requests.

    The process starts with :meth:`launch`; until then the messages for its
    requests wait in this process, as a PUSH socket without a connected
    process does not take them.
    """

    def __init__(self, pool: PostprocPool, index: int) -> None:
        import zmq

        from tensorrt_llm.executor.ipc import ZeroMqQueue

        self.index = index
        self.inbox = ZeroMqQueue(
            socket_type=zmq.PUSH, is_server=True, is_async=True, name=f"postproc_in_{index}"
        )
        self.sessions: dict[str, _Session] = {}
        self._outgoing: list[Any] = []
        self._wakeup = asyncio.Event()
        self.process = pool._context.Process(
            target=_process_main,
            args=(
                index,
                self.inbox.address,
                pool._results.address,
                pool._tokenizer,
                pool._hook_path,
            ),
            name=f"trtllm-postproc-{index}",
            daemon=True,
        )
        # When the process started; None until it does.
        self.started: float | None = None
        # Set once the process exited or could not start; no request goes to it then.
        self.retired = False
        self._sender: asyncio.Task | None = None
        self._stopping: asyncio.Task | None = None

    def launch(self) -> None:
        self.process.start()
        self.started = time.monotonic()

    @property
    def exited(self) -> bool:
        return self.started is not None and not self.process.is_alive()

    def post(self, message: Any) -> None:
        self._outgoing.append(message)
        self._wakeup.set()

    def start_sending(self) -> None:
        if self._sender is None:
            self._sender = asyncio.get_running_loop().create_task(self._send_loop())

    async def _send_loop(self) -> None:
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            batch, self._outgoing = self._outgoing, []
            if not batch:
                continue
            try:
                await self.inbox.put_async(batch)
            except Exception:  # noqa: BLE001 - find the message that cannot be sent
                # Resent one by one, so that only the request whose message
                # cannot be sent fails.
                for message in batch:
                    await self._send_one(message)

    async def _send_one(self, message: Any) -> None:
        try:
            await self.inbox.put_async([message])
        except Exception as error:  # noqa: BLE001 - fail this request, keep sending
            logger.error(
                f"could not send a message of postprocessing request {message.request_id}: "
                f"{type(error).__name__}"
            )
            session = self.sessions.get(message.request_id)
            if session is not None:
                # The process may already hold the request; let it drop it.
                self.post(PPClose(message.request_id, "cancelled"))
                session.deliver(PostprocWorkerError("a postprocessing message was lost"))

    async def stop(self, timeout: float) -> None:
        """Stop the process, killing it after ``timeout`` seconds, and close the socket.

        Later calls wait for the first one.
        """
        if self._stopping is None:
            self._stopping = asyncio.get_running_loop().create_task(self._stop(timeout))
        await asyncio.shield(self._stopping)

    async def _stop(self, timeout: float) -> None:
        import zmq

        if self._sender is not None:
            self._sender.cancel()
        deadline = time.monotonic() + timeout
        if self.started is not None and self.process.is_alive() and timeout > 0:
            # A PUSH socket without a connected process blocks, so the stop
            # message is only sent to a live process, and not waited on forever.
            try:
                await asyncio.wait_for(self.inbox.put_async(None), timeout)
            except (asyncio.TimeoutError, zmq.ZMQError) as error:
                logger.warning(
                    f"could not stop postprocessing process {self.process.pid}: "
                    f"{type(error).__name__}"
                )
            while self.process.is_alive() and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
        if self.started is not None:
            if self.process.is_alive():
                self.process.kill()
            while self.process.is_alive():
                await asyncio.sleep(0.05)
            self.process.join()
        self.close_socket()

    def close_socket(self) -> None:
        """Close the socket at once, dropping the messages still queued in it."""
        import zmq

        if self.inbox.socket is not None:
            self.inbox.socket.setsockopt(zmq.LINGER, 0)
        self.inbox.close()


# Seconds before replacing a process that exited soon after it started, at
# most; the wait doubles with each such exit and each failed start, so a
# process that cannot start is not restarted in a tight loop.
_RESTART_DELAY_LIMIT = 30.0
# A process that ran at least this long before it exited is replaced at once.
_HEALTHY_SECONDS = 60.0


def _longer_delay(delay: float) -> float:
    return min(max(2 * delay, 0.5), _RESTART_DELAY_LIMIT)


class PostprocPool:
    """The postprocessing processes of one frontend (``contracts.PostprocPool``).

    Built outside the event loop; the tasks that talk to the processes start
    with the first request.
    """

    def __init__(self, num_workers: int, tokenizer: _TokenizerSpec, hook_path: str | None) -> None:
        import zmq

        from tensorrt_llm.executor.ipc import ZeroMqQueue

        if num_workers < 1:
            raise ValueError(f"a postprocessing pool needs at least one process, got {num_workers}")
        if hook_path:
            from tensorrt_llm.executor.postprocessor_hook import load_post_processor_hook

            # Fail at startup, as the in-process LLM does, on a hook that cannot
            # be built; each pool process builds its own.
            load_post_processor_hook(hook_path)
        self._tokenizer = tokenizer
        self._hook_path = hook_path
        self._context = multiprocessing.get_context("spawn")
        self._results = ZeroMqQueue(
            socket_type=zmq.PULL, is_server=True, is_async=True, name="postproc_results"
        )
        self._workers: list[_Worker] = []
        try:
            for i in range(num_workers):
                self._workers.append(_Worker(self, i))
                self._workers[-1].launch()
        except BaseException:
            for worker in self._workers:
                if worker.started is not None:
                    worker.process.kill()
                    worker.process.join()
                worker.close_socket()
            self._results.close()
            raise
        self._restart_delays = [0.0] * num_workers
        self._replacing: dict[int, asyncio.Task] = {}
        # The stopping of retired workers.
        self._stopping: set[asyncio.Task] = set()
        self._sessions: dict[str, _Session] = {}
        self._tasks: list[asyncio.Task] = []
        self._closed = False

    @property
    def size(self) -> int:
        return len(self._workers)

    @property
    def pids(self) -> list[int | None]:
        """The process ids, in routing order; None for a process that has not started yet."""
        return [worker.process.pid for worker in self._workers]

    def _start(self) -> None:
        if self._tasks:
            return
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(self._read_loop()), loop.create_task(self._watch_loop())]
        for task in self._tasks:
            task.add_done_callback(self._task_ended)
        for worker in self._workers:
            worker.start_sending()

    def open(
        self,
        request_id: str,
        postproc_params: Any,
        sampling_params: Any,
        *,
        streaming: bool,
        num_outputs: int,
        num_prompt_tokens: int,
        engine_stop_texts: bool = False,
        max_tokens: Sequence[int | None] | None = None,
    ) -> _Session:
        if self._closed:
            raise PostprocWorkerError("the postprocessing pool is closed")
        if request_id in self._sessions:
            # The facade gives every request its own id; reusing an open one is a program error.
            raise PostprocError(f"request id {request_id!r} is already being postprocessed")
        plain_params = plain_sampling_params(sampling_params)
        detok = contracts.DetokOptions(
            skip_special_tokens=plain_params.skip_special_tokens,
            spaces_between_special_tokens=plain_params.spaces_between_special_tokens,
            stop=tuple(
                [plain_params.stop]
                if isinstance(plain_params.stop, str)
                else plain_params.stop or ()
            )
            or None,
            include_stop_str_in_output=plain_params.include_stop_str_in_output,
            min_tokens=plain_params.min_tokens or 0,
            engine_stop=engine_stop_texts,
        )
        message = PPOpen(
            request_id=request_id,
            streaming=streaming,
            n=num_outputs,
            post_processor=contracts.post_processor_path(postproc_params.post_processor),
            postproc_args=_plain_postproc_args(
                postproc_params.postproc_args, sampling_params, plain_params
            ),
            detok=detok,
            num_prompt_tokens=num_prompt_tokens,
            sampling_params=plain_params,
            max_tokens=None if max_tokens is None else tuple(max_tokens),
        )
        self._start()
        index = _route(request_id, self.size)
        worker = self._workers[index]
        if worker.retired:
            # Only while a worker to replace it cannot be built.
            raise PostprocWorkerError(f"postprocessing process {index} is not running")
        session = _Session(self, request_id, worker)
        self._sessions[request_id] = session
        worker.sessions[request_id] = session
        worker.post(message)
        return session

    @staticmethod
    def _task_ended(task: asyncio.Task) -> None:
        # The reading and watching tasks end only when the pool closes.
        if not task.cancelled() and task.exception() is not None:
            logger.error(f"a postprocessing pool task stopped: {type(task.exception()).__name__}")

    def _forget(self, session: _Session) -> None:
        self._sessions.pop(session.request_id, None)
        session.worker.sessions.pop(session.request_id, None)

    async def _read_loop(self) -> None:
        while True:
            try:
                replies = await self._results.get_async()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - drop the message, keep reading
                # For example a message that fails the HMAC check.
                frames = "".join(traceback.format_tb(error.__traceback__))
                logger.error(f"dropped a postprocessing reply: {type(error).__name__}\n{frames}")
                continue
            for reply in replies:
                session = self._sessions.get(reply.request_id)
                if session is None:
                    continue
                session.pending -= reply.events
                if reply.error is not None:
                    session.deliver(PostprocError(reply.error))
                elif reply.result is not None:
                    session.deliver(reply.result)

    def _retire(self, worker: _Worker, why: str) -> None:
        """Fail the requests of ``worker`` and stop it in the background."""
        worker.retired = True
        for session in list(worker.sessions.values()):
            session.deliver(PostprocWorkerError(why))
        task = asyncio.get_running_loop().create_task(worker.stop(timeout=0))
        self._stopping.add(task)
        task.add_done_callback(self._stopped)

    def _stopped(self, task: asyncio.Task) -> None:
        self._stopping.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                f"could not stop a postprocessing process: {type(task.exception()).__name__}"
            )

    async def _watch_loop(self) -> None:
        """Replace each process that exited, failing the requests it held; one task per process."""
        while True:
            await asyncio.sleep(0.2)
            for i, worker in enumerate(self._workers):
                if i in self._replacing or not worker.exited:
                    continue
                lived = time.monotonic() - worker.started
                logger.error(
                    f"postprocessing process {worker.process.pid} exited with "
                    f"{worker.process.exitcode} after {lived:.1f} s; failing its "
                    f"{len(worker.sessions)} request(s) and starting another"
                )
                self._restart_delays[i] = (
                    0.0 if lived >= _HEALTHY_SECONDS else _longer_delay(self._restart_delays[i])
                )
                self._retire(worker, "the postprocessing process exited")
                try:
                    # In place before any request can be routed to the index.
                    self._build(i)
                except Exception as error:  # noqa: BLE001 - _replace tries again
                    logger.error(
                        f"could not build postprocessing process {i}: {type(error).__name__}"
                    )
                task = asyncio.get_running_loop().create_task(self._replace(i))
                self._replacing[i] = task
                task.add_done_callback(lambda _, i=i: self._replacing.pop(i, None))

    def _build(self, i: int) -> _Worker:
        """Put a new worker, without its process yet, in place of the retired one at index ``i``."""
        worker = _Worker(self, i)
        self._workers[i] = worker
        worker.start_sending()
        return worker

    async def _replace(self, i: int) -> None:
        """Start the process of the new worker at index ``i`` after the restart delay.

        The requests routed to the worker wait for its process meanwhile. A
        worker that cannot be built or started is tried again, later each time.
        """
        while not self._closed:
            worker = self._workers[i]
            try:
                if worker.retired:
                    worker = self._build(i)
                await asyncio.sleep(self._restart_delays[i])
                if self._closed:
                    return
                worker.launch()
                return
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - try again, waiting longer each time
                frames = "".join(traceback.format_tb(error.__traceback__))
                logger.error(
                    f"could not start postprocessing process {i}: {type(error).__name__}\n{frames}"
                )
                self._restart_delays[i] = _longer_delay(self._restart_delays[i])
                if self._workers[i] is worker and not worker.retired:
                    # Its process did not start; the next try builds another worker.
                    self._retire(worker, "the postprocessing process could not start")
                else:
                    await asyncio.sleep(self._restart_delays[i])

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        replacing = list(self._replacing.values())
        for task in [*self._tasks, *replacing]:
            task.cancel()
        # Once the replacements stopped, no worker is put in place any more.
        await asyncio.gather(*replacing, return_exceptions=True)
        for session in list(self._sessions.values()):
            session.deliver(PostprocWorkerError("the postprocessing pool is closed"))
        # A retired worker is already stopping, in ``_stopping``.
        stops = [worker.stop(timeout=5) for worker in self._workers if not worker.retired]
        try:
            outcomes = await asyncio.gather(*stops, *self._stopping, return_exceptions=True)
        finally:
            self._results.close()
        # The failures of the retired workers were logged by ``_stopped``.
        for outcome in outcomes[: len(stops)]:
            if isinstance(outcome, Exception):
                logger.error(f"could not stop a postprocessing process: {type(outcome).__name__}")


@dataclass(frozen=True)
class FacadePostprocDispatch(DefaultPostprocDispatch):
    """The dispatch for the OpenEngine facade.

    With a pool it reads the results the pool produced, as the default does.
    Without one the handler postprocesses, and the ``post_processor_hook``
    runs first on every step, through :func:`postprocess_step` as in a pool
    process.
    """

    hook: Any = None
    _terminated: Any = dataclasses.field(default_factory=weakref.WeakSet, compare=False)

    def result(self, output: Any, postproc_params: Any) -> Any:
        if self.enabled or self.hook is None:
            return super().result(output, postproc_params)
        streaming = _streaming_of(postproc_params.post_processor)
        if output in self._terminated:
            # The hook ended this request; what the engine still sent is dropped.
            return [] if streaming else None
        view = self._view(output)
        chunks = postprocess_step(
            view,
            postproc_params.post_processor,
            postproc_params.postproc_args,
            hook=self.hook,
            streaming=streaming,
        )
        if view._done and not output._done:
            self._terminated.add(output)
        return chunks

    def apply_hook(self, output: Any) -> None:
        """Run the hook on the finished ``output`` of a non-streaming request formatted elsewhere."""
        if self.enabled or self.hook is None:
            return
        run_hook(self._view(output), self.hook, streaming=False)

    @staticmethod
    def _view(output: Any) -> PostprocResult:
        view = PostprocResult(
            output.id,
            output.outputs,
            cached_tokens=output.cached_tokens,
            done=output._done,
            abort=output.abort,
        )
        view._aborted = bool(getattr(output, "_aborted", False))
        return view


def create_postproc_dispatch(generator: Any, *, is_visual_gen: bool) -> contracts.PostprocDispatch:
    """The dispatch for ``OpenAIServer`` serving ``generator`` (the ``create_postproc_dispatch`` slot).

    The facade without a pool runs the hook its ``args`` name, the same
    ``post_processor_hook`` a pool would get from the facade.
    """
    if is_visual_gen or getattr(type(generator), "postproc_host", False) is not True:
        return contracts.default_create_postproc_dispatch(generator, is_visual_gen=is_visual_gen)
    pool = generator.postproc_pool
    hook = None
    hook_path = getattr(generator.args, "post_processor_hook", None)
    if pool is None and hook_path:
        from tensorrt_llm.executor.postprocessor_hook import load_post_processor_hook

        hook = load_post_processor_hook(hook_path)
    return FacadePostprocDispatch(enabled=pool is not None, hook=hook)


def create_postproc_pool(args: Any, *, num_workers: int, tokenizer_dir: str) -> PostprocPool | None:
    """The pool of a frontend (the ``create_postproc_pool`` slot); None for ``num_workers`` 0.

    The processes load the tokenizer from ``tokenizer_dir`` with the
    tokenizer options of ``args``, and the hook ``args`` names.
    """
    if num_workers == 0:
        return None
    return PostprocPool(
        num_workers,
        _TokenizerSpec.from_args(args, tokenizer_dir),
        getattr(args, "post_processor_hook", None),
    )

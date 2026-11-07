# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The postprocessing pool of an OpenEngine frontend, through OpenAIServer and directly."""

import asyncio
import json
import os
import re
import signal
import time
from collections import defaultdict
from types import SimpleNamespace

import httpx
import pytest

from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.executor.postprocessor_hook import terminate
from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.openengine_poc import contracts, postproc_pool
from tensorrt_llm.serve.openengine_poc.engine_client import OutputFinished, TokenDelta, Usage
from tensorrt_llm.serve.postprocess_handlers import (
    CompletionPostprocArgs,
    completion_response_post_processor,
    completion_stream_post_processor,
)

__extra_import_path__ = ["..", "."]
from fakes import TINY_EOS, FakeEngineClient, tiny_hf_tokenizer, tiny_tokenizer  # noqa: E402
from pool_harness import HOOK_LOG_ENV, PoolFacade, serve_args  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

MODEL = "tiny"
PROMPT_IDS = [2, 3]
# "He", "llo", " world", "!", " again", one token per event.
REPLY = (2, 3, 4, 5, 6)
HOOK = "pool_harness.RecordingHook"
WAIT_SECONDS = 300


def _script(request):
    steps = [TokenDelta(output_index=0, token_ids=(token,)) for token in REPLY]
    steps.append(
        OutputFinished(
            output_index=0,
            reason="stop",
            eos_token_id=TINY_EOS,
            usage=Usage(
                prompt_tokens=len(request.prompt_token_ids),
                completion_tokens=len(REPLY),
                cached_prompt_tokens=1,
            ),
        )
    )
    return steps


def _input_processor(prompt, sampling_params):
    return list(PROMPT_IDS), None


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_tokenizer")
    tiny_hf_tokenizer().save_pretrained(path)
    return str(path)


@pytest.fixture(autouse=True)
def _hook_importable(monkeypatch):
    # The hook is loaded by name, here and in the pool processes, which start
    # with this process's sys.path.
    monkeypatch.syspath_prepend(os.path.dirname(__file__))


@pytest.fixture(autouse=True)
def _no_store(monkeypatch):
    # With storage on, the handler-run Responses path stores the response and
    # the pool path cannot; compare them with storage off.
    monkeypatch.setenv("TRTLLM_RESPONSES_API_DISABLE_STORE", "1")


def _serve(tokenizer_dir, workers, body, hook=None):
    """Run ``await body(facade, http)`` against an OpenAIServer over the pool facade."""

    async def main():
        facade = PoolFacade(
            FakeEngineClient.like("trtllm", script=_script),
            tiny_tokenizer(),
            tokenizer_dir,
            serve_args(workers, hook),
            _input_processor,
        )
        previous_media_executor = BaseMediaIO._executor
        server = OpenAIServer(
            generator=facade,
            model=MODEL,
            tool_parser=None,
            server_role=None,
            metadata_server_cfg=None,
        )
        BaseMediaIO.set_executor(previous_media_executor)
        server._media_load_executor.shutdown(wait=False)
        try:
            transport = httpx.ASGITransport(app=server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                return await asyncio.wait_for(body(facade, http), WAIT_SECONDS)
        finally:
            await facade.aclose()
            server._input_proc_executor.shutdown(wait=False)
            server._metrics_stats_executor.shutdown(wait=False)

    return asyncio.run(main())


REQUESTS = {
    "chat": (
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8},
    ),
    "completions": ("/v1/completions", {"prompt": "Hi", "max_tokens": 8}),
    "responses": ("/v1/responses", {"input": "Hi", "max_output_tokens": 8}),
}
# Fields that differ between any two runs: generated ids and creation times.
_VOLATILE = {"id", "item_id", "response_id", "created", "created_at", "system_fingerprint"}
_ID_VALUE = re.compile(r"^(resp|msg|rs|chatcmpl|cmpl|fc|call)[-_][0-9A-Za-z_-]+$")


def _normalize(value):
    if isinstance(value, dict):
        return {k: ("<v>" if k in _VOLATILE else _normalize(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, str) and _ID_VALUE.match(value):
        return "<id>"
    return value


def _body(text: str, streaming: bool):
    if not streaming:
        return _normalize(json.loads(text))
    events = []
    for line in text.splitlines():
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            events.append(_normalize(json.loads(line[len("data: ") :])))
        elif line.startswith("event: ") or line.strip() == "data: [DONE]":
            events.append(line.strip())
    return events


def _check_one_responses_stream(text: str) -> None:
    """The events the handler sends and those a pool process sends belong to one response."""
    events = [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ") and line.strip() != "data: [DONE]"
    ]
    responses = [event["response"] for event in events if "response" in event]
    assert len(responses) >= 3, events  # created, in_progress, completed
    assert len({response["id"] for response in responses}) == 1, responses
    assert len({response["created_at"] for response in responses}) == 1, responses
    assert [event["sequence_number"] for event in events] == list(range(len(events)))


async def _all_requests(facade, http):
    bodies = {}
    for name, (path, payload) in REQUESTS.items():
        for streaming in (False, True):
            response = await http.post(path, json={"model": MODEL, **payload, "stream": streaming})
            assert response.status_code == 200, (name, streaming, response.text)
            if name == "responses" and streaming:
                _check_one_responses_stream(response.text)
            bodies[name, streaming] = _body(response.text, streaming)
    return bodies


def test_outputs_are_the_same_with_and_without_a_pool(tokenizer_dir):
    without = _serve(tokenizer_dir, 0, _all_requests)
    with_pool = _serve(tokenizer_dir, 2, _all_requests)
    assert set(without) == set(with_pool) == {(n, s) for n in REQUESTS for s in (False, True)}
    for key in without:
        assert with_pool[key] == without[key], key
    # The text did reach the clients.
    assert without["chat", False]["choices"][0]["message"]["content"] == "Hello world! again"


def test_the_hook_runs_in_the_pool_processes(tokenizer_dir, tmp_path, monkeypatch):
    log = tmp_path / "hook.log"
    monkeypatch.setenv(HOOK_LOG_ENV, str(log))

    async def body(facade, http):
        bodies = await _all_requests(facade, http)
        return bodies, facade.postproc_pool.pids

    with_pool, pids = _serve(tokenizer_dir, 2, body, hook=HOOK)
    lines = [line.split() for line in log.read_text().splitlines()]
    assert lines, "the hook never ran"
    by_request = defaultdict(set)
    for pid, request_id in lines:
        by_request[request_id].add(int(pid))
    # The work ran in the pool's processes, not in this one, every request
    # stayed in one process, and the requests went to both processes.
    assert all(len(seen) == 1 for seen in by_request.values()), by_request
    used = set().union(*by_request.values())
    assert used <= set(pids) and os.getpid() not in used
    assert len(used) == 2, by_request

    log.unlink()
    without = _serve(tokenizer_dir, 0, _all_requests, hook=HOOK)
    assert {int(line.split()[0]) for line in log.read_text().splitlines()} == {os.getpid()}
    for key in without:
        assert with_pool[key] == without[key], key
    # The hook upper-cases the text; in a streaming request its terminate on
    # "AGAIN" ends the request with the text before it.
    content = without["chat", False]["choices"][0]["message"]["content"]
    assert content == "HELLO WORLD! AGAIN", content
    streamed = "".join(
        event["choices"][0]["delta"].get("content") or ""
        for event in without["chat", True]
        if isinstance(event, dict) and event["choices"]
    )
    assert streamed == "HELLO WORLD!", streamed


def test_routing_is_stable():
    ids = [f"req-{i}" for i in range(64)]
    assert [postproc_pool._route(i, 4) for i in ids] == [postproc_pool._route(i, 4) for i in ids]
    assert len({postproc_pool._route(i, 4) for i in ids}) == 4


def test_live_objects_do_not_cross():
    args = CompletionPostprocArgs(model=MODEL)
    args.ctx_usage = lambda: None
    with pytest.raises(contracts.PostprocDataError, match="ctx_usage"):
        postproc_pool._plain_postproc_args(args)
    with pytest.raises(contracts.PostprocDataError):
        contracts.post_processor_path(lambda rsp, args: None)
    # A tokenizer the handler set is left behind and set again in the process.
    args = CompletionPostprocArgs(model=MODEL)
    args.tokenizer = tiny_tokenizer()
    assert postproc_pool._plain_postproc_args(args).tokenizer is None
    params = SamplingParams(max_tokens=4, logits_processor=[lambda *a: None])
    assert postproc_pool.plain_sampling_params(params).logits_processor is None


def test_a_hook_that_cannot_be_built_fails_at_startup(tokenizer_dir):
    with pytest.raises(RuntimeError, match="cannot be built"):
        postproc_pool.PostprocPool(
            1, postproc_pool._TokenizerSpec(tokenizer_dir), "pool_harness.UnbuildableHook"
        )


def test_open_rejects_what_cannot_cross(tokenizer_dir):
    async def main():
        pool = postproc_pool.PostprocPool(1, postproc_pool._TokenizerSpec(tokenizer_dir), None)
        try:
            args = CompletionPostprocArgs(model=MODEL)
            args.prompt = object()
            with pytest.raises(contracts.PostprocDataError):
                pool.open(
                    "r",
                    PostprocParams(
                        post_processor=completion_stream_post_processor, postproc_args=args
                    ),
                    SamplingParams(max_tokens=4),
                    streaming=True,
                    num_outputs=1,
                    num_prompt_tokens=2,
                )
        finally:
            await pool.aclose()

    asyncio.run(main())


def test_a_crashed_process_fails_its_requests_and_is_replaced(tokenizer_dir):
    async def main():
        pool = postproc_pool.PostprocPool(1, postproc_pool._TokenizerSpec(tokenizer_dir), None)
        try:
            params = PostprocParams(
                post_processor=completion_stream_post_processor,
                postproc_args=CompletionPostprocArgs(model=MODEL),
            )
            session = pool.open(
                "r",
                params,
                SamplingParams(max_tokens=8),
                streaming=True,
                num_outputs=1,
                num_prompt_tokens=2,
            )
            session.send(TokenDelta(output_index=0, token_ids=(2,)))
            first = await asyncio.wait_for(session.__aiter__().__anext__(), WAIT_SECONDS)
            assert not first.is_final
            old = pool.pids[0]
            os.kill(old, 9)
            with pytest.raises(contracts.PostprocWorkerError):
                async for _ in session:
                    pass
            await _wait_for(lambda: pool.pids[0] not in (old, None))
            # The replacement serves new requests.
            session = pool.open(
                "r2",
                params,
                SamplingParams(max_tokens=8),
                streaming=True,
                num_outputs=1,
                num_prompt_tokens=2,
            )
            session.send(TokenDelta(output_index=0, token_ids=(2,)))
            session.send(
                OutputFinished(
                    output_index=0,
                    reason="length",
                    usage=Usage(prompt_tokens=2, completion_tokens=1),
                )
            )
            results = [r async for r in session]
            assert results[-1].is_final
        finally:
            await pool.aclose()

    asyncio.run(asyncio.wait_for(main(), WAIT_SECONDS))


def test_a_session_past_the_event_limit_overflows(monkeypatch):
    monkeypatch.setattr(postproc_pool, "POSTPROC_PENDING_EVENT_LIMIT", 2)
    posted, forgotten = [], []
    worker = SimpleNamespace(post=posted.append)
    pool = SimpleNamespace(_forget=forgotten.append)

    async def main():
        session = postproc_pool._Session(pool=pool, request_id="r", worker=worker)
        session.send(TokenDelta(output_index=0, token_ids=(2,)))
        session.send(TokenDelta(output_index=0, token_ids=(3,)))
        with pytest.raises(contracts.PostprocOverflowError) as raised:
            session.send(TokenDelta(output_index=0, token_ids=(4,)))
        # The session is over: the process is told to drop the request, the
        # pool forgets it, the iteration raises the same error, and later
        # sends and closes do nothing.
        assert [type(m).__name__ for m in posted] == ["PPEvent", "PPEvent", "PPClose"]
        assert forgotten == [session]
        with pytest.raises(contracts.PostprocOverflowError) as iterated:
            async for _ in session:
                pass
        assert iterated.value is raised.value
        session.send(TokenDelta(output_index=0, token_ids=(5,)))
        session.close("done")
        assert len(posted) == 3

    asyncio.run(main())


def test_dispatch_slot_keeps_the_default_for_other_generators():
    class _LLM:
        args = type("A", (), {"num_postprocess_workers": 2, "post_processor_hook": None})()

    dispatch = postproc_pool.create_postproc_dispatch(_LLM(), is_visual_gen=False)
    assert type(dispatch) is contracts.DefaultPostprocDispatch and dispatch.enabled
    assert postproc_pool.create_postproc_pool(_LLM.args, num_workers=0, tokenizer_dir="/m") is None


# --- One request in a pool process, run in this process --------------------------


def _record(sampling_params, *, streaming=True, n=1, hook=None, engine_stop=False):
    post_processor = (
        completion_stream_post_processor if streaming else completion_response_post_processor
    )
    message = contracts.PPOpen(
        request_id="r",
        streaming=streaming,
        n=n,
        post_processor=contracts.post_processor_path(post_processor),
        postproc_args=CompletionPostprocArgs(model=MODEL, num_choices=n),
        detok=contracts.DetokOptions(engine_stop=engine_stop),
        num_prompt_tokens=len(PROMPT_IDS),
        sampling_params=postproc_pool.plain_sampling_params(sampling_params),
    )
    return postproc_pool._Record(message, tiny_tokenizer(), hook)


def test_engine_stop_texts_go_from_open_to_the_assembler(monkeypatch):
    posted = []
    pool = object.__new__(postproc_pool.PostprocPool)
    worker = SimpleNamespace(retired=False, sessions={}, post=posted.append)
    pool._closed, pool._sessions, pool._workers, pool._tasks = False, {}, [worker], ["started"]
    for request_id, engine_stop_texts in (("a", True), ("b", False)):
        pool.open(
            request_id,
            _stream_params(),
            SamplingParams(max_tokens=8),
            streaming=True,
            num_outputs=1,
            num_prompt_tokens=2,
            engine_stop_texts=engine_stop_texts,
        )
    assert [message.detok.engine_stop for message in posted] == [True, False]

    seen = []
    create = contracts.create_response_assembler

    def capture(**kwargs):
        seen.append(kwargs["engine_stop_texts"])
        return create(**kwargs)

    monkeypatch.setattr(contracts, "create_response_assembler", capture)
    _record(SamplingParams(max_tokens=8), engine_stop=True)
    _record(SamplingParams(max_tokens=8))
    assert seen == [True, False]


def _choices(result):
    return [json.loads(chunk[len("data: ") :])["choices"] for chunk in result.chunks]


def test_a_stop_string_is_reported_in_stop_hit():
    record = _record(SamplingParams(max_tokens=8, stop=[" world"]))
    results = [record.on_event(TokenDelta(output_index=0, token_ids=(t,))) for t in (2, 3, 4)]
    assert [r.stop_hit for r in results] == [None, None, (0,)]
    assert results[-1].is_final


def test_a_non_streaming_request_reports_stop_hit_before_its_end():
    # Two outputs need sampling; greedy decoding allows one.
    params = SamplingParams(max_tokens=8, stop=[" world"], n=2, temperature=1.0)
    record = _record(params, streaming=False, n=2)
    for token in (2, 3):
        assert record.on_event(TokenDelta(output_index=0, token_ids=(token,))) is None
    result = record.on_event(TokenDelta(output_index=0, token_ids=(4,)))
    assert result.stop_hit == (0,) and not result.is_final and result.chunks is None


def test_engine_stops_and_hook_terminations_are_not_stop_hits():
    record = _record(SamplingParams(max_tokens=8, stop=[" world"]))
    record.on_event(TokenDelta(output_index=0, token_ids=(2,)))
    result = record.on_event(OutputFinished(output_index=0, reason="stop", stop_text=" world"))
    assert result.is_final and result.stop_hit is None

    class Terminate:
        def __call__(self, chunk):
            return terminate("policy")

    record = _record(SamplingParams(max_tokens=8), hook=Terminate())
    result = record.on_event(TokenDelta(output_index=0, token_ids=(2,)))
    assert result.is_final and result.stop_hit is None


def test_an_output_the_engine_ended_is_not_a_stop_hit(monkeypatch):
    # With the engine's stop strings, the assembler may end on a stop string
    # its text holds an output the engine ended for another reason.
    create = contracts.create_response_assembler

    def ending_on_a_stop_string(**kwargs):
        assembler = create(**kwargs)
        apply = assembler.apply

        def apply_and_cut(event):
            changed = apply(event)
            if isinstance(event, OutputFinished):
                output = assembler.outputs[event.output_index]
                output.finish_reason, output.stop_reason = "stop", " world"
            return changed

        assembler.apply = apply_and_cut
        return assembler

    monkeypatch.setattr(contracts, "create_response_assembler", ending_on_a_stop_string)
    record = _record(SamplingParams(max_tokens=3), engine_stop=True)
    for token in (2, 3):
        record.on_event(TokenDelta(output_index=0, token_ids=(token,)))
    result = record.on_event(OutputFinished(output_index=0, reason="length"))
    assert result.is_final and result.stop_hit is None


def test_closing_before_the_engine_finished():
    # Cancelled: the outputs end as cancelled, and the text already sent is not sent again.
    record = _record(SamplingParams(max_tokens=8))
    first = record.on_event(TokenDelta(output_index=0, token_ids=(2,)))
    assert _choices(first)[0][0]["text"] == "He"
    last = record.on_close("cancelled")
    assert last.is_final and record.result._aborted
    (choice,) = _choices(last)[0]
    assert choice["text"] == "" and choice["finish_reason"] == "cancelled"
    # Done: the engine ended the request early, which fails it, as the facade does without a pool.
    record = _record(SamplingParams(max_tokens=8))
    record.on_event(TokenDelta(output_index=0, token_ids=(2,)))
    with pytest.raises(postproc_pool.UnfinishedOutputsError):
        record.on_close("done")


def test_a_postprocessor_abort_ends_the_request_as_a_cancel_does():
    calls = []

    def abort_on_first_step(rsp, args):
        calls.append((rsp._done, [output.finish_reason for output in rsp.outputs]))
        if len(calls) == 1:
            rsp.abort()
        return [f"step {len(calls)}"]

    record = _record(SamplingParams(max_tokens=8))
    record.post_processor = abort_on_first_step
    result = record.on_event(TokenDelta(output_index=0, token_ids=(2,)))
    # The postprocessor runs once more on the request ended as cancelled, as
    # the handler runs it without a pool, and both steps go in the final result.
    assert calls == [(False, [None]), (True, ["cancelled"])]
    assert result.is_final and record.result._aborted
    assert result.chunks == ["step 1", "step 2"]


def _stream_params():
    return PostprocParams(
        post_processor=completion_stream_post_processor,
        postproc_args=CompletionPostprocArgs(model=MODEL),
    )


def _open(pool, request_id):
    return pool.open(
        request_id,
        _stream_params(),
        SamplingParams(max_tokens=8),
        streaming=True,
        num_outputs=1,
        num_prompt_tokens=2,
    )


async def _finish(session):
    session.send(TokenDelta(output_index=0, token_ids=(2,)))
    session.send(
        OutputFinished(
            output_index=0, reason="length", usage=Usage(prompt_tokens=2, completion_tokens=1)
        )
    )
    return [r async for r in session]


async def _wait_for(condition):
    for _ in range(300):
        if condition():
            return
        await asyncio.sleep(0.1)
    assert condition()


def test_requests_wait_for_the_replacement_of_their_process(tokenizer_dir, monkeypatch):
    # Keep the replacement waiting, so that a request arrives before it starts.
    monkeypatch.setattr(postproc_pool, "_RESTART_DELAY_LIMIT", 3.0)
    monkeypatch.setattr(postproc_pool, "_HEALTHY_SECONDS", 1e9)

    async def main():
        pool = postproc_pool.PostprocPool(2, postproc_pool._TokenizerSpec(tokenizer_dir), None)
        pool._restart_delays[0] = 3.0
        try:
            _open(pool, "warm").close("cancelled")
            old = pool._workers[0]
            os.kill(old.process.pid, 9)
            await _wait_for(lambda: pool._workers[0] is not old)
            replacement = pool._workers[0]
            assert replacement.started is None
            request_id = next(f"r{i}" for i in range(1000) if postproc_pool._route(f"r{i}", 2) == 0)
            session = _open(pool, request_id)
            # It stays with the process its id routes to.
            assert session.worker is replacement
            results = await _finish(session)
            assert results[-1].is_final and replacement.started is not None
        finally:
            await pool.aclose()

    asyncio.run(asyncio.wait_for(main(), WAIT_SECONDS))


def test_a_process_that_cannot_start_fails_its_requests_and_is_tried_later(
    tokenizer_dir, monkeypatch
):
    monkeypatch.setattr(postproc_pool, "_HEALTHY_SECONDS", 1e9)

    async def main():
        pool = postproc_pool.PostprocPool(1, postproc_pool._TokenizerSpec(tokenizer_dir), None)
        launch = postproc_pool._Worker.launch
        tries = []

        def failing_launch(worker):
            tries.append(time.monotonic())
            raise OSError("cannot start")

        try:
            _open(pool, "warm").close("cancelled")
            old = pool._workers[0]
            monkeypatch.setattr(postproc_pool._Worker, "launch", failing_launch)
            os.kill(old.process.pid, 9)
            await _wait_for(lambda: pool._workers[0] is not old)
            session = _open(pool, "waiting")
            with pytest.raises(contracts.PostprocWorkerError):
                async for _ in session:
                    pass
            await _wait_for(lambda: len(tries) >= 3)
            # The wait grows with each failed start: 0.5 s, then 1 s, then 2 s.
            assert tries[2] - tries[1] > tries[1] - tries[0] > 0.4
            # Once a process starts again, it serves requests.
            monkeypatch.setattr(postproc_pool._Worker, "launch", launch)
            await _wait_for(lambda: pool._workers[0].started is not None)
            assert (await _finish(_open(pool, "served")))[-1].is_final
        finally:
            await pool.aclose()

    asyncio.run(asyncio.wait_for(main(), WAIT_SECONDS))


def test_an_interrupt_leaves_the_processes_to_the_frontend(tokenizer_dir):
    # A Ctrl-C in the terminal reaches the pool processes too.
    async def main():
        pool = postproc_pool.PostprocPool(1, postproc_pool._TokenizerSpec(tokenizer_dir), None)
        try:
            assert (await _finish(_open(pool, "before")))[-1].is_final
            pid = pool.pids[0]
            os.kill(pid, signal.SIGINT)
            assert (await _finish(_open(pool, "after")))[-1].is_final
            assert pool.pids[0] == pid
        finally:
            await pool.aclose()

    asyncio.run(asyncio.wait_for(main(), WAIT_SECONDS))

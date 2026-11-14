# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ds-engine OpenEngine server as context and generation engine, straight over gRPC.

Two ``dsv4-serve --grpc --grpc-protocol openengine`` instances, one started with
``--server_role context`` and one with ``--server_role generation``, take the two legs of a
disaggregated request from TensorRT-LLM's OpenEngine client, without the serve front end:
a ``context_only`` request ends with ``prefill_ready``, and its ``kv_session`` sent to the
generation instance decodes from the handoff's first token on (NT-042). Aborting the
generation leg cancels it and leaves both instances serving (NT-043).

The instances are started outside this module (the ds-engine repository's
``dev/openengine_disagg.sh`` starts them on one node). Set
``TRTLLM_TEST_DS_OE_CTX_TARGET`` and ``TRTLLM_TEST_DS_OE_GEN_TARGET`` to their ``host:port``
addresses, and optionally ``TRTLLM_TEST_DS_OE_MODEL`` to the checkpoint directory, whose
``tokenizer.json`` turns the prompts into token ids. Without the two targets the module is
skipped. The counter checks, and NT-043, which fills the generation instance's running slots,
expect no other client to use the two instances while the tests run.
"""

import asyncio
import json
import os
import time
from pathlib import Path

import grpc
import pytest
from google.protobuf import json_format

pytestmark = [pytest.mark.threadleak(enabled=False)]

CTX_ENV = "TRTLLM_TEST_DS_OE_CTX_TARGET"
GEN_ENV = "TRTLLM_TEST_DS_OE_GEN_TARGET"
MODEL_ENV = "TRTLLM_TEST_DS_OE_MODEL"
CTX_TARGET = os.environ.get(CTX_ENV)
GEN_TARGET = os.environ.get(GEN_ENV)

if not (CTX_TARGET and GEN_TARGET):
    pytest.skip(f"{CTX_ENV} and {GEN_ENV} are not set", allow_module_level=True)

from tensorrt_llm.grpc.openengine.bindings import (  # noqa: E402
    generation_pb2,
    kv_pb2,
    lifecycle_pb2,
    server_pb2,
)
from tensorrt_llm.grpc.openengine_client import (  # noqa: E402
    AbortOutcome,
    OpenEngineClient,
    OpenEngineEngineError,
    OpenEngineRpcError,
)

# An unfinished sentence: a context request whose first token is EOS ends without a handoff (unless
# it sets ignore_eos).
PROMPT = "The sky looks blue on a clear day because"
# A fixed prompt for a checkpoint without tokenizer.json: ids inside the vocabulary; unlike PROMPT, nothing
# keeps its first token from being EOS.
FALLBACK_PROMPT_IDS = [0] + list(range(1000, 1064))
FIRST_DISAGG_REQUEST_ID = (1 << 40) + 4200
ABORT_TIMEOUT_S = 30.0


def _prompt_ids() -> list[int]:
    model = os.environ.get(MODEL_ENV)
    tokenizer_file = Path(model) / "tokenizer.json" if model else None
    if tokenizer_file is None or not tokenizer_file.is_file():
        return list(FALLBACK_PROMPT_IDS)
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_file)).encode(PROMPT).ids


def _request(
    request_id: str, prompt_ids: list[int], max_tokens: int, **extra
) -> generation_pb2.GenerateRequest:
    request = generation_pb2.GenerateRequest(
        request_id=request_id,
        model="ds",
        token_ids=generation_pb2.TokenIds(ids=prompt_ids),
        sampling=generation_pb2.SamplingParams(temperature=0.0),
        stopping=generation_pb2.StoppingOptions(max_tokens=max_tokens),
    )
    if extra:
        request.extra.update(extra)
    return request


def _context_request(request_id, prompt_ids, max_tokens, disagg_request_id):
    return _request(
        request_id,
        prompt_ids,
        max_tokens,
        request_type="context_only",
        disagg_request_id=str(disagg_request_id),
    )


def _generation_request(request_id, prompt_ids, max_tokens, session):
    request = _request(request_id, prompt_ids, max_tokens)
    request.kv.session.CopyFrom(session)
    return request


async def _collect(client: OpenEngineClient, request) -> list[generation_pb2.GenerateResponse]:
    async with client.generate(request, timeout=600) as stream:
        return [response async for response in stream]


def _tokens(responses) -> list[int]:
    return [
        token.token_id
        for response in responses
        if response.WhichOneof("event") == "token"
        for token in response.token.tokens
    ]


def _counters(load: server_pb2.LoadInfo | None) -> dict[str, float]:
    assert load is not None, "the instance does not implement GetLoad"
    return {
        key: value for key, value in load.attributes.items() if key.startswith("trtllm.counters.")
    }


async def _connect() -> tuple[OpenEngineClient, OpenEngineClient]:
    ctx = await OpenEngineClient.connect(CTX_TARGET, ready_timeout=60)
    try:
        gen = await OpenEngineClient.connect(GEN_TARGET, ready_timeout=60)
    except BaseException:
        await ctx.close()
        raise
    return ctx, gen


async def _prefill(
    ctx: OpenEngineClient, prompt_ids, request_id, disagg_request_id, max_tokens, ignore_eos=False
):
    request = _context_request(request_id, prompt_ids, max_tokens, disagg_request_id)
    request.stopping.ignore_eos = ignore_eos
    responses = await _collect(ctx, request)
    events = [response.WhichOneof("event") for response in responses]
    assert events[-1] == "prefill_ready" and "finished" not in events, events
    return responses


def _session_attributes(session: kv_pb2.KvSessionRef) -> dict:
    return json.loads(json_format.MessageToJson(session.attributes_struct))


async def _handoff(ctx, gen, prompt_ids, name, disagg_request_id, max_tokens=64):
    """One request through both legs; returns (context responses, generation responses)."""
    ctx_responses = await _prefill(ctx, prompt_ids, f"{name}-ctx", disagg_request_id, max_tokens)
    session = ctx_responses[-1].prefill_ready.kv_session
    gen_responses = await _collect(
        gen, _generation_request(f"{name}-gen", prompt_ids, max_tokens, session)
    )
    return ctx_responses, gen_responses


async def _direct_grpc_handoff() -> None:
    ctx, gen = await _connect()
    try:
        assert ctx.server_info.engine_role == server_pb2.ENGINE_ROLE_PREFILL
        assert gen.server_info.engine_role == server_pb2.ENGINE_ROLE_DECODE
        for client in (ctx, gen):
            assert client.server_info.kv_connector.enabled
            assert client.server_info.extra["trtllm.conventions_version"] == 4
            assert client.model_info.extra["trtllm.request.disagg_request_id"] is True
        assert ctx.server_info.kv_connector.local_endpoints
        prompt_ids = _prompt_ids()
        ctx_before = _counters(await ctx.get_load())
        gen_before = _counters(await gen.get_load())

        ctx_responses, gen_responses = await _handoff(
            ctx, gen, prompt_ids, "nt042", FIRST_DISAGG_REQUEST_ID
        )
        ready = ctx_responses[-1]
        session = ready.prefill_ready.kv_session
        attributes = _session_attributes(session)
        assert session.session_id == str(FIRST_DISAGG_REQUEST_ID)
        assert attributes["disagg_request_id"] == str(FIRST_DISAGG_REQUEST_ID)
        first_tokens = [int(token) for token in attributes["first_gen_tokens"]]
        assert first_tokens == _tokens(ctx_responses)[:1]
        # The context leg generates one token whatever its max_tokens (EN-DS-13).
        assert ready.usage.prompt_tokens == len(prompt_ids)
        assert ready.usage.completion_tokens == 1

        tokens = _tokens(gen_responses)
        assert tokens[:1] == first_tokens  # the generation stream starts with the handoff's token
        finished = gen_responses[-1].finished
        assert finished.reason in (
            generation_pb2.FINISH_REASON_LENGTH,
            generation_pb2.FINISH_REASON_STOP,
        )
        assert gen_responses[-1].usage.completion_tokens >= len(tokens) > 1
        # The generation engine reports no prefix-cache hits of its own (DS-I10).
        assert gen_responses[-1].usage.HasField("cached_prompt_tokens")
        assert gen_responses[-1].usage.cached_prompt_tokens == 0

        ctx_after = _counters(await ctx.get_load())
        gen_after = _counters(await gen.get_load())
        context_key = "trtllm.counters.generate_accepted.context_only"
        generation_key = "trtllm.counters.generate_accepted.generation_only"
        assert ctx_after[context_key] == ctx_before[context_key] + 1
        assert (
            ctx_after["trtllm.counters.prefill_ready_sent"]
            == ctx_before["trtllm.counters.prefill_ready_sent"] + 1
        )
        assert gen_after[generation_key] == gen_before[generation_key] + 1

        # Concurrent handoffs, each under its own id.
        results = await asyncio.gather(
            *(
                _handoff(ctx, gen, prompt_ids, f"nt042-{i}", FIRST_DISAGG_REQUEST_ID + 1 + i)
                for i in range(8)
            )
        )
        for ctx_responses, gen_responses in results:
            first = _session_attributes(ctx_responses[-1].prefill_ready.kv_session)
            assert _tokens(gen_responses)[:1] == [int(t) for t in first["first_gen_tokens"]]
            assert gen_responses[-1].WhichOneof("event") == "finished"

        # A context request whose first token is a stop token (greedy: the same first token)
        # finishes on the context instance without a handoff. Greedy is not bitwise repeatable
        # (prefix-cache hits, other batches): a request that hands off another first token has its
        # KV decoded (freed), and the next attempt stops on that token.
        stop_token = first_tokens[0]
        for attempt in range(3):
            request = _context_request(
                f"nt042-stop{attempt}", prompt_ids, 64, FIRST_DISAGG_REQUEST_ID + 20 + attempt
            )
            request.stopping.conditions.add(stop_token_id=stop_token)
            responses = await _collect(ctx, request)
            events = [response.WhichOneof("event") for response in responses]
            if events[-1] != "prefill_ready":
                break
            handed = responses[-1].prefill_ready.kv_session
            stop_token = int(_session_attributes(handed)["first_gen_tokens"][0])
            await _collect(
                gen, _generation_request(f"nt042-stop{attempt}-gen", prompt_ids, 64, handed)
            )
        assert events[-1] == "finished" and "prefill_ready" not in events, events
        assert responses[-1].finished.reason == generation_pb2.FINISH_REASON_STOP

        # A handoff no context instance parked fails on the generation instance after acceptance.
        unknown = kv_pb2.KvSessionRef()
        unknown.CopyFrom(session)
        unknown.session_id = str(FIRST_DISAGG_REQUEST_ID + 30)
        unknown.attributes_struct.update({"disagg_request_id": unknown.session_id})
        with pytest.raises(OpenEngineEngineError):
            await _collect(gen, _generation_request("nt042-unknown", prompt_ids, 64, unknown))
    finally:
        await ctx.close()
        await gen.close()


async def _direct_grpc_abort() -> None:
    ctx, gen = await _connect()
    try:
        prompt_ids = _prompt_ids()
        assert await gen.abort("nt043-unknown") in (
            AbortOutcome.ALREADY_FINISHED,
            AbortOutcome.NOT_FOUND,
        )
        disagg_request_id = FIRST_DISAGG_REQUEST_ID + 100
        # The handoffs this test aborts ignore EOS on both legs: a context leg whose first token is
        # EOS would end without a handoff.
        ctx_responses = await _prefill(
            ctx, prompt_ids, "nt043-ctx", disagg_request_id, 4096, ignore_eos=True
        )
        session = ctx_responses[-1].prefill_ready.kv_session
        # The context request ended with its handoff; a parked KV is not released by session.
        assert await ctx.abort("nt043-ctx") in (
            AbortOutcome.ALREADY_FINISHED,
            AbortOutcome.NOT_FOUND,
        )
        with pytest.raises(OpenEngineRpcError) as error:
            await ctx.abort_kv_session(session)
        assert error.value.code == grpc.StatusCode.UNIMPLEMENTED

        # Aborted mid-stream (ignore_eos: the request cannot end before 4096 tokens on its own).
        # The abort deadline (ABORT_TIMEOUT_S) leaves room for the server to wait for the engine's
        # confirmation before it answers.
        outcome = None
        responses = []
        request = _generation_request("nt043-gen", prompt_ids, 4096, session)
        request.stopping.ignore_eos = True
        async with gen.generate(request, timeout=600) as stream:
            async for response in stream:
                responses.append(response)
                if outcome is None and response.WhichOneof("event") == "token":
                    outcome = await gen.abort("nt043-gen", timeout=ABORT_TIMEOUT_S)
                    if outcome != AbortOutcome.ABORTED:  # (it would decode on to max_tokens)
                        pytest.fail(f"Abort of a decoding request answered {outcome}")
        assert outcome == AbortOutcome.ABORTED
        assert responses[-1].finished.reason == generation_pb2.FINISH_REASON_CANCELLED
        assert len(_tokens(responses)) < 4096

        # Aborted before its KV pull. The generation instance starts a request's KV pull when it
        # admits the request, and the pull itself is far too short to aim an Abort at. So its running
        # slots are first taken by aggregated requests that cannot end on their own (ignore_eos): the
        # generation leg then waits unadmitted, before its pull and its first token. Abort is sent
        # until the instance has registered it (before that the answer is ALREADY_FINISHED).
        slots = gen.server_info.capacity.max_running_requests
        assert slots > 0, gen.server_info.capacity
        fillers = []
        try:
            for i in range(slots):
                filler = _request(f"nt043-fill{i}", prompt_ids, 4096)
                filler.stopping.ignore_eos = True
                fillers.append(gen.generate(filler, timeout=600))
            # Each filler runs once it has streamed its first event (a filler left waiting means the
            # instance runs fewer requests than it declares).
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(stream.__anext__() for stream in fillers)), timeout=120
                )
            except asyncio.TimeoutError:
                pytest.fail(
                    f"not all {slots} fillers ran within 120 s: the instance runs fewer requests "
                    "than it declares, or another client holds its slots"
                )
            ctx_responses = await _prefill(
                ctx, prompt_ids, "nt043-wait-ctx", disagg_request_id + 10, 4096, ignore_eos=True
            )
            request = _generation_request(
                "nt043-wait-gen", prompt_ids, 4096, ctx_responses[-1].prefill_ready.kv_session
            )
            request.stopping.ignore_eos = True
            async with gen.generate(request, timeout=600) as stream:
                first = asyncio.ensure_future(stream.__anext__())
                try:
                    deadline = time.monotonic() + 30
                    outcome = None
                    while not first.done() and time.monotonic() < deadline:
                        outcome = await gen.abort("nt043-wait-gen", timeout=ABORT_TIMEOUT_S)
                        if outcome == AbortOutcome.ABORTED:
                            break
                        await asyncio.sleep(0.01)
                    try:
                        head = await asyncio.wait_for(first, timeout=ABORT_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        pytest.fail(f"the generation leg neither streamed nor ended ({outcome})")
                    if head.WhichOneof("event") == "token":  # (it would decode on to max_tokens)
                        pytest.fail(
                            f"the generation leg was admitted although {slots} requests held its "
                            f"slots ({outcome})"
                        )
                    responses = [head] + [response async for response in stream]
                finally:
                    if not first.done():
                        first.cancel()
            for i in range(slots):
                filler_outcome = await gen.abort(f"nt043-fill{i}", timeout=ABORT_TIMEOUT_S)
                if filler_outcome != AbortOutcome.ABORTED:  # (it would decode on to max_tokens)
                    pytest.fail(f"Abort of filler {i} answered {filler_outcome}")
            filler_ends = []
            for stream in fillers:
                filler_ends.append([response async for response in stream])
        finally:
            for stream in fillers:
                await stream.aclose()  # (cancels a filler still running)
        assert outcome == AbortOutcome.ABORTED, (
            outcome,
            [r.WhichOneof("event") for r in responses],
        )
        assert responses[-1].finished.reason == generation_pb2.FINISH_REASON_CANCELLED
        assert _tokens(responses) == [], (
            f"the generation leg was admitted although {slots} requests held its slots"
        )
        for ends in filler_ends:
            assert ends[-1].finished.reason == generation_pb2.FINISH_REASON_CANCELLED, ends[-1]

        # Both instances keep serving.
        for client in (ctx, gen):
            health = await client.health()
            assert health.state == lifecycle_pb2.HEALTH_STATE_READY
        _, gen_responses = await _handoff(
            ctx, gen, prompt_ids, "nt043-after", disagg_request_id + 1
        )
        assert gen_responses[-1].WhichOneof("event") == "finished"
    finally:
        await ctx.close()
        await gen.close()


def test_direct_grpc_handoff():
    """NT-042: context_only -> prefill_ready -> kv.session -> decoding, straight over gRPC."""
    asyncio.run(_direct_grpc_handoff())


def test_direct_grpc_abort():
    """NT-043: aborting the generation leg over gRPC, mid-stream or before its KV pull (while it
    waits to be admitted), cancels it; both instances keep serving."""
    asyncio.run(_direct_grpc_abort())

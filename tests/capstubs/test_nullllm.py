# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NullLlm / NullHandle in one event loop with a recording source: one queued step per
__anext__ (never merged), cumulative token_ids, finish / abort / truncation, token sources,
refusals, the per-RPC context hook, no handle retention."""

import asyncio
import types

import pytest

from capstubs.nullllm import CURRENT_RPC, NullArgs, NullLlm, NullLlmError


class Src:
    """Plans every stream as `steps` (tokens per item); records open / took / close."""

    def __init__(self, steps=(1, 1, 1)):
        self.steps = tuple(steps)
        self.opened, self.taken, self.closed = [], 0, []

    def open(self, h, n):
        h.n = n
        h.toks = self.steps
        self.opened.append(h)

    def took(self, h):
        self.taken += 1

    def close(self, h, outcome):
        if not h.closed:
            h.closed = True
            h.outcome = outcome
            self.closed.append(outcome)


def sp(n):
    return types.SimpleNamespace(max_tokens=n, n=1)


def llm(src, **kw):
    return NullLlm(src, NullArgs(4096, 64, 8192), **kw)


def run(coro):
    return asyncio.run(coro)


def test_one_step_per_anext_even_when_several_are_queued():
    async def body():
        src = Src((2, 2, 1))
        h = llm(src, token_ids=(7, 8, 9)).generate_async({"prompt_token_ids": [1, 2]}, sp(5),
                                                         streaming=True)
        for j in range(3):  # all three queued before the servicer reads
            h.push(j)
        seen = []
        async for r in h:
            seen.append((list(r.outputs[0].token_ids), r.outputs[0].finish_reason, r.finished))
        return h, src, seen
    h, src, seen = run(body())
    # three results, one per step, cumulative ids cycled from token_ids; finish on the last
    assert seen == [([7, 8], None, False), ([7, 8, 9, 7], None, False),
                    ([7, 8, 9, 7, 8], "length", True)]
    assert src.taken == 3 and src.closed == ["completed"] and h.prompt_token_ids == [1, 2]
    assert h.error is None and h.produced == 5 and h.taken == 3


def test_waits_for_the_source_then_resumes():
    async def body():
        src = Src((1, 1))
        h = llm(src).generate_async("text", sp(2), streaming=True)
        got = []

        async def reader():
            async for r in h:
                got.append(len(r.outputs[0].token_ids))
        t = asyncio.ensure_future(reader())
        await asyncio.sleep(0.02)
        assert got == []  # nothing queued: the reader waits
        h.push(0)
        await asyncio.sleep(0.02)
        assert got == [1]
        h.push(1)
        await asyncio.wait_for(t, 1)
        return got
    assert run(body()) == [1, 2]


def test_abort_closes_once_and_ends_with_cancelled():
    async def body():
        src = Src((1, 1, 1))
        h = llm(src).generate_async("x", sp(3), streaming=True)
        h.push(0)
        h.push(1)
        r = await h.__anext__()
        assert r.outputs[0].token_ids == [11]
        h.abort()
        h.abort()
        r = await h.__anext__()  # queued step 1 is not delivered after the abort
        assert r.finished and r.outputs[0].finish_reason == "cancelled"
        assert r.outputs[0].token_ids == [11]
        with pytest.raises(StopAsyncIteration):
            await h.__anext__()
        return src
    assert run(body()).closed == ["cancelled"]


def test_truncation_becomes_an_engine_error():
    async def body():
        src = Src((1, 1))
        h = llm(src).generate_async("x", sp(2), streaming=True)
        h.push(0)
        h.truncate()
        r1 = await h.__anext__()  # the handle is the result object (as GenerationResult)
        assert r1.error is None and not r1.finished and r1.outputs[0].token_ids == [11]
        r2 = await h.__anext__()
        assert r2.error and "beat schedule ended" in r2.error
        with pytest.raises(StopAsyncIteration):
            await h.__anext__()
        return src
    assert run(body()).closed == ["truncated"]


def test_prompt_token_source_and_tokenizer():
    class Tok:
        def encode(self, text):
            return [ord(c) for c in text]

    async def body():
        src = Src((3,))
        m = llm(src, token_source="prompt", tokenizer=Tok())
        h = m.generate_async("ab", sp(3), streaming=True)
        h.push(0)
        r = await h.__anext__()
        assert r.outputs[0].token_ids == [97, 98, 97] and r.prompt_token_ids == [97, 98]
        with pytest.raises(NullLlmError, match="non-empty prompt"):
            llm(src, token_source="prompt").generate_async("ab", sp(3), streaming=True)
        h2 = llm(src).generate_async("ab", sp(3), streaming=True)  # no tokenizer: no prompt ids
        assert h2.prompt_token_ids == []
    run(body())


@pytest.mark.parametrize("call,match", [
    (lambda m: m.generate_async("x", sp(3), streaming=False), "streaming"),
    (lambda m: m.generate_async("x", sp(0), streaming=True), "max_tokens"),
    (lambda m: m.generate_async("x", sp(None), streaming=True), "max_tokens"),
    (lambda m: m.generate_async("x", sp(3), streaming=True, disaggregated_params=object()),
     "aggregated"),
    (lambda m: m.generate_async(3, sp(3), streaming=True), "text prompt"),
    (lambda m: m.generate_async("x", types.SimpleNamespace(max_tokens=2, n=2), streaming=True),
     "one output"),
    (lambda m: m.preprocess({}), "media"),
])
def test_refusals_are_value_errors(call, match):
    async def body():
        src = Src()
        with pytest.raises(ValueError, match=match):
            call(llm(src))
        assert src.opened == []
    run(body())


def test_context_hook_and_no_retention():
    async def body():
        src = Src((1,))
        m = llm(src)
        rec = types.SimpleNamespace(handle=None)
        tok = CURRENT_RPC.set(rec)
        h = m.generate_async("x", sp(1), streaming=True)
        CURRENT_RPC.reset(tok)
        assert rec.handle is h
        h2 = m.generate_async("x", sp(1), streaming=True)  # no RPC in context: nothing set
        assert rec.handle is h
        # NullLlm keeps no request list (FakeLlm does): only the source saw the handles
        assert not any(isinstance(v, list) and h2 in v for v in vars(m).values())
        assert m._check_health() is True
        m.shutdown()
        assert m._check_health() is False and m._executor.is_shutdown()
        assert m.get_stats() == []
    run(body())


def test_args_fields_the_servicer_reads():
    a = NullArgs(4096, 64, 8192)
    for name in ("max_seq_len", "max_num_tokens", "max_batch_size", "kv_cache_config",
                 "guided_decoding_backend", "reasoning_parser", "return_perf_metrics",
                 "enable_iter_perf_stats", "tensor_parallel_size", "pipeline_parallel_size",
                 "context_parallel_size"):
        assert hasattr(a, name), name
    assert a.kv_cache_config.tokens_per_block == 32 and a.kv_cache_config.kv_events_config is None
    m = llm(Src())
    for name in ("input_processor", "tokenizer", "preprocess", "generate_async", "_check_health",
                 "get_stats", "shutdown", "_executor", "args"):
        assert hasattr(m, name), name
    assert m.input_processor.supports_token_ids_input is True
    assert a.to_dict()["kv_cache_config"]["tokens_per_block"] == 32

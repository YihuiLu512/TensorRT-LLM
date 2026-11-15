# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on case-03, the ds-engine: sampling, stopping and decoding options.

Every request goes to the disaggregated server (form K: the ds proxy), one at a time: the
ds-engine's output depends on the batch a request runs in, so the tests that compare two
outputs send them one after the other. The forms, fixtures and environment are those of
``_test_openengine_ds_disagg_basic.py``.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from . import _test_openengine_ds_disagg_basic as basic

MODEL_NAME = basic.MODEL_NAME
# The module's fixtures.
model_path = basic.model_path
form = basic.form
groups = basic.groups
served = basic.served

pytestmark = [pytest.mark.threadleak(enabled=False)]

# A plain-text prompt whose continuation runs well past a few dozen tokens.
STORY_PROMPT = "Write a short story about a lighthouse keeper who finds a message in a bottle.\n"
MAX_SEQ_LEN = int(basic.ENGINE_ARGS[basic.ENGINE_ARGS.index("--max_seq_len") + 1])
MAX_DRAFT_LEN = basic.GENERATION_CONFIG["speculative_config"]["max_draft_len"]


@pytest.fixture(scope="module")
def tokenizer(model_path):
    from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer

    return DeepseekV41Tokenizer.from_pretrained(model_path)


@dataclass
class Completion:
    """One ``/v1/completions`` answer.

    Streamed only: ``chunks`` counts the chunks that carried a choice, ``done`` says whether the
    stream ended with ``[DONE]``, and ``chunk_tokens`` (with continuous usage) holds how many
    output tokens each chunk with a choice added.
    """

    text: str
    finish_reason: str | None
    stop_reason: int | str | None
    usage: object
    token_ids: list[int] | None
    chunk_tokens: list[int] = field(default_factory=list)
    chunks: int = 0
    done: bool = False


def complete(served, stream: bool = False, *, continuous: bool = False, **request) -> Completion:
    """One completions request to the disaggregated server, greedy unless asked otherwise.

    Parameters the OpenAI client does not know go in the body, as in :func:`basic.generate`.
    """
    c = served.client()
    request.setdefault("model", MODEL_NAME)
    request.setdefault("temperature", 0.0)
    request.setdefault("prompt", STORY_PROMPT)
    call = c.completions.create
    known = inspect.signature(call).parameters
    extra = {key: request.pop(key) for key in list(request) if key not in known}
    if extra:
        request["extra_body"] = {**request.get("extra_body", {}), **extra}
    if not stream:
        out = call(**request)
        choice = out.choices[0]
        return Completion(
            choice.text,
            choice.finish_reason,
            getattr(choice, "stop_reason", None),
            out.usage,
            getattr(choice, "token_ids", None),
        )
    # Streamed over plain HTTP, so that whether the stream ended with [DONE] is seen.
    import requests

    body = {**request.pop("extra_body", {}), **request, "stream": True}
    body["stream_options"] = {"include_usage": True, "continuous_usage_stats": continuous}
    out = Completion("", None, None, None, None)
    counted = 0
    headers = {"Authorization": f"Bearer {basic.DUMMY_API_KEY}"}
    url = f"{served.url}/v1/completions"
    with requests.post(url, json=body, headers=headers, stream=True, timeout=600) as response:
        assert response.status_code == 200, (response.status_code, response.text[:2000])
        for line in response.iter_lines():
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                out.done = True
                continue
            assert not out.done, "a chunk after [DONE]"
            chunk = json.loads(data)
            assert "error" not in chunk, chunk
            if chunk.get("usage"):
                out.usage = SimpleNamespace(**chunk["usage"])
            if not chunk.get("choices"):
                continue
            choice = chunk["choices"][0]
            assert out.finish_reason is None, "a chunk after the output finished"
            out.chunks += 1
            out.text += choice.get("text") or ""
            if choice.get("token_ids") is not None:
                out.token_ids = (out.token_ids or []) + list(choice["token_ids"])
            if continuous and chunk.get("usage"):
                out.chunk_tokens.append(chunk["usage"]["completion_tokens"] - counted)
                counted = chunk["usage"]["completion_tokens"]
            out.finish_reason = choice.get("finish_reason")
            out.stop_reason = choice.get("stop_reason")
    return out


def greedy_ids(served, **request) -> list[int]:
    """The greedy output's token ids (``detokenize=False``)."""
    out = complete(served, detokenize=False, **request)
    assert out.text == "" and out.token_ids, out
    assert len(out.token_ids) == out.usage.completion_tokens, out
    return out.token_ids


def rendered_chat(tokenizer, content: str) -> str:
    """A user turn rendered by the DeepSeek-V4.1 renderer without thinking, BOS included."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, thinking=False
    )


CONTEXT_ENDED_STREAM_SHAPE = "context_ended_stream_shape"


def check_context_ended(
    served, stream: bool, out: Completion, record_property, stop_reason=None
) -> None:
    """``out`` is a request the context leg ended with ``stop`` at its first token.

    The response carries no output text, at most the one token that ended it, and
    ``stop_reason`` unless it is ``None``; a stream ends with ``[DONE]``. Streamed through the
    chain, the response may also be ``[DONE]`` alone, with no choice and no usage: the
    disaggregated server does not forward the context worker's finished response (a known
    difference to form K, under review). Which of the two a stream was is recorded in the
    property ``context_ended_stream_shape`` (``done_only`` or ``finish_chunk``).
    """
    assert out.text == "", out
    if stream:
        assert out.done, out
        shape = "finish_chunk" if out.chunks else "done_only"
        record_property(CONTEXT_ENDED_STREAM_SHAPE, shape)
        if shape == "done_only" and served.chain:
            assert out.usage is None, out
            return
    assert out.finish_reason == "stop", out
    assert out.usage.completion_tokens <= 1, out
    if stop_reason is not None:
        assert out.stop_reason == stop_reason, out


def test_sampling_params(served):
    """NT-110 (F-11, DS-S05): ``top_k=1`` samples the greedy output; ``temperature`` varies it.

    ``top_k=1`` at ``temperature=1`` gives the ``temperature=0`` output (deterministic). Four
    samples at ``temperature=1`` are not all the same: a probabilistic check, which fails with
    sampling working only if four independent 24-token continuations of a story prompt
    coincide, with probability at most ``max_s p(s)**3`` over continuations ``s``. ``top_p``
    and ``min_p`` requests succeed; the distributions they sample from are not checked.
    """
    greedy = complete(served, max_tokens=24)
    top_k_1 = complete(served, max_tokens=24, temperature=1.0, top_k=1)
    assert top_k_1.text == greedy.text, (top_k_1.text, greedy.text)
    samples = {complete(served, max_tokens=24, temperature=1.0).text for _ in range(4)}
    assert len(samples) > 1, samples
    for request in ({"temperature": 0.8, "top_p": 0.5}, {"temperature": 1.0, "min_p": 0.1}):
        out = complete(served, max_tokens=24, **request)
        assert out.text and out.finish_reason in ("stop", "length"), (request, out)
        assert out.usage.completion_tokens > 0, (request, out)


@pytest.mark.parametrize("api", ["completions", "chat"])
def test_seed(served, api):
    """NT-111 (F-12, DS-S06): the same request with the same seed, sent alone twice, is the same.

    The ds-engine's samples depend on the batch a request runs in, so the requests are sent one
    after the other (UV-62: with other requests in the batch they differ). So that a request
    that is not sampled at all does not pass, four other seeds must not all give the first
    seed's output (probabilistic, as the sample check of NT-110).
    """
    request = {"max_tokens": 32, "temperature": 1.0}
    if api == "completions":
        request["prompt"] = "Write a short poem about the sea.\n"
    sampled = [basic.generate(served, api, False, seed=1234, **request)[0] for _ in range(2)]
    assert sampled[0], sampled
    assert sampled[0] == sampled[1], sampled
    others = {basic.generate(served, api, False, seed=seed, **request)[0] for seed in range(4)}
    assert others != {sampled[0]}, (sampled[0], others)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_and_length(served, stream):
    """NT-112 (F-14, DS-S04, DS-S11, DS-I06): ``max_tokens`` ends the output with ``length``.

    The output has exactly ``max_tokens`` tokens; ``max_tokens=1`` (the context leg's first
    token, then a generation leg with nothing left to generate) is the first token of the longer
    output. A ``max_tokens`` above what ``max_seq_len`` leaves after the prompt is accepted; the
    answer ends on its own long before that, so the clamp itself is not reached.
    """
    longer = complete(served, stream, max_tokens=12)
    assert longer.finish_reason == "length", longer
    assert longer.usage.completion_tokens == 12, longer.usage
    one = complete(served, stream, max_tokens=1)
    assert one.finish_reason == "length" and one.usage.completion_tokens == 1, one
    assert one.text and longer.text.startswith(one.text), (one.text, longer.text)

    # The chat answer ends on its own, long before the clamp.
    _, _, usage = basic.generate(served, "chat", stream, max_tokens=8)
    room = MAX_SEQ_LEN - usage.prompt_tokens
    text, finish, usage = basic.generate(served, "chat", stream, max_tokens=room + 1000)
    assert text and finish in ("stop", "length"), (text, finish)
    assert usage.completion_tokens <= room, (usage, room)


def _stop_token(token_ids: list[int]) -> int:
    """The index of the first output token, from the second on, that does not occur before it."""
    for index in range(1, len(token_ids)):
        if token_ids[index] not in token_ids[:index]:
            return index
    raise AssertionError(f"no token of {token_ids} is new after the first")


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_stop_token_ids_and_ignore_eos(served, tokenizer, stream, record_property):
    """NT-113 (F-16, F-18, DS-S07, DS-S08, DS-S11): stop token ids, ``ignore_eos``, EOS first.

    * A token of the greedy output, from the second token on, in ``stop_token_ids`` ends the
      output there with ``stop``, and ``stop_reason`` is that token id.
    * With ``ignore_eos`` an answer that ends at EOS goes on to ``max_tokens``.
    * A prompt whose greedy first token is EOS ends with ``stop`` and no output
      (:func:`check_context_ended`). Its ``stop_reason`` is recorded (property
      ``eos_first_stop_reason``), not judged: the two forms' sessions are compared on it.
    """
    token_ids = greedy_ids(served, max_tokens=32)
    index = _stop_token(token_ids)
    greedy = complete(served, max_tokens=32)
    out = complete(served, stream, max_tokens=32, stop_token_ids=[token_ids[index]])
    assert out.finish_reason == "stop" and out.stop_reason == token_ids[index], (out, index)
    assert out.usage.completion_tokens in (index, index + 1), (out.usage, index)
    assert greedy.text.startswith(out.text) and len(out.text) < len(greedy.text), (out, greedy)

    # ignore_eos: the answer without it ends at EOS after n tokens.
    _, finish, usage = basic.generate(served, "chat", stream, max_tokens=256)
    n = usage.completion_tokens
    assert finish == "stop" and n < 256, (finish, usage)
    _, finish, usage = basic.generate(served, "chat", stream, max_tokens=n + 8, ignore_eos=True)
    assert finish == "length" and usage.completion_tokens == n + 8, (finish, usage, n)

    # EOS as the first token: the prompt is a turn followed by its own complete answer.
    prompt = rendered_chat(tokenizer, basic.CHAT_MESSAGES[0]["content"])
    answer = complete(served, prompt=prompt, max_tokens=64, add_special_tokens=False)
    assert answer.finish_reason == "stop", answer
    request = {"prompt": prompt + answer.text, "max_tokens": 64, "add_special_tokens": False}
    first = complete(served, **request)
    assert first.usage.completion_tokens <= 1, ("the greedy first token is not EOS", first)
    out = complete(served, stream, **request)
    check_context_ended(served, stream, out, record_property)
    record_property("eos_first_stop_reason", repr(out.stop_reason))


def _stop_string(text: str) -> tuple[str, int]:
    """A word of ``text`` with its leading space, after the first few tokens, not seen before.

    Returns the stop string and where it starts.
    """
    position = 16
    while True:
        start = text.find(" ", position)
        assert start > 0, f"no stop string in {text!r}"
        end = start + 1
        while end < len(text) and text[end].isalpha():
            end += 1
        word = text[start:end]
        if len(word) >= 4 and text.find(word) == start:
            return word, start
        position = start + 1


@pytest.mark.parametrize("include", [False, True], ids=["exclude_stop", "include_stop"])
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_stop_strings(served, stream, include):
    """NT-114 (F-17): the output ends at the first stop string that occurs.

    The stop string is a word of the greedy output past its first 16 characters, normally past
    the first token, which the context leg makes. Listed after one that never occurs, it ends the
    output just the same.
    ``include_stop_str_in_output`` keeps the stop string at the end of the text.
    """
    greedy = complete(served, max_tokens=48)
    stop, start = _stop_string(greedy.text)
    expected = greedy.text[: start + (len(stop) if include else 0)]
    for stops in ([stop], ["zqxjv", stop]):
        out = complete(
            served, stream, max_tokens=48, stop=stops, include_stop_str_in_output=include
        )
        assert (out.text, out.finish_reason, out.stop_reason) == (expected, "stop", stop), (
            stops,
            out,
        )


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_first_token_stop_skips_gen(served, stream, record_property):
    """NT-115 (DG-16, DS-S10): a request that stops at its first token never reaches generation.

    The greedy first token is made a stop token id: the context leg ends the request with
    ``stop`` (:func:`check_context_ended`). On the chain the context engine sends no PrefillReady
    for it and the generation engine gets no call; in form K only the response is checked.
    """
    from tensorrt_llm.grpc.openengine_conventions import (
        COUNTER_GENERATE_RECEIVED,
        COUNTER_PREFILL_READY_SENT,
    )

    first = greedy_ids(served, max_tokens=4)[0]

    def counters():
        if not served.chain:
            return None
        return (
            basic.engine_counter(served.engines["context"], COUNTER_PREFILL_READY_SENT),
            basic.engine_counter(served.engines["generation"], COUNTER_GENERATE_RECEIVED),
        )

    before = counters()
    out = complete(served, stream, max_tokens=16, stop_token_ids=[first])
    check_context_ended(served, stream, out, record_property, first)
    assert counters() == before, (before, counters())


def test_decode_options(served, tokenizer):
    """NT-116 (F-24): the generation worker front end's detokenization options.

    With ``detokenize=False`` the output is token ids and no text; with ``skip_special_tokens``
    on or off the text is the tokenizer's decode of those ids, streamed or not. The output runs
    past EOS (``ignore_eos``) and holds special tokens: the two decodes differ.
    """
    basic.only_chain(served, "the options under test are the worker front end's")
    prompt = rendered_chat(tokenizer, "Say hello in one word.")
    request = {"prompt": prompt, "max_tokens": 48, "ignore_eos": True, "add_special_tokens": False}
    token_ids = greedy_ids(served, **request)
    decoded = {
        skip: tokenizer.decode(token_ids, skip_special_tokens=skip) for skip in (True, False)
    }
    # Otherwise the two settings could not be told apart.
    assert decoded[True] != decoded[False], f"the output holds no special token: {token_ids}"
    for skip, expected in decoded.items():
        for stream in (False, True):
            out = complete(served, stream, skip_special_tokens=skip, **request)
            assert (out.text, out.finish_reason) == (expected, "length"), (skip, stream, out)
            assert out.usage.completion_tokens == len(token_ids), (skip, stream, out.usage)


def test_token_id_prompt_and_truncate(served, tokenizer):
    """NT-117 (F-08, F-09): a prompt of token ids, and ``truncate_prompt_tokens``.

    A token-id prompt is used as given: ``prompt_tokens`` is the number of ids, and the greedy
    output is that of the same text prompt. With ``truncate_prompt_tokens`` the prompt has that
    many tokens.
    """
    basic.only_chain(served, "the prompt handling under test is the worker front end's")
    ids = tokenizer.encode(STORY_PROMPT)
    by_ids = complete(served, prompt=ids, max_tokens=16)
    assert by_ids.usage.prompt_tokens == len(ids), (by_ids.usage, len(ids))
    by_text = complete(served, max_tokens=16)
    assert by_text.usage.prompt_tokens == len(ids), (by_text.usage, len(ids))
    assert by_ids.text and by_ids.text == by_text.text, (by_ids.text, by_text.text)

    out = complete(served, max_tokens=16, truncate_prompt_tokens=8)
    assert out.usage.prompt_tokens == 8, out.usage
    assert out.text and out.finish_reason in ("stop", "length"), out


# The tests below run on engine groups of their own, after the default group's tests. A test
# file's group changes only in a fixture, during setup.

STREAM_INTERVAL = 8
STREAM_INTERVAL_GROUP = basic.Group(generation={"stream_interval": STREAM_INTERVAL})


@pytest.fixture
def interval_served(groups) -> basic.Served:
    """The module's default group with the generation engine's ``stream_interval`` set."""
    return groups.get(STREAM_INTERVAL_GROUP)


def test_stream_interval(interval_served):
    """NT-118 (DS-S17): the generation engine streams its tokens ``stream_interval`` at a time.

    The engine sends a chunk once ``stream_interval`` tokens are pending: past the first chunks
    (the first token) and before the last, each chunk holds at least ``stream_interval`` tokens,
    and at most ``stream_interval + max_draft_len`` (one DSpark step accepts up to
    ``max_draft_len + 1`` tokens).
    """
    out = complete(interval_served, True, continuous=True, max_tokens=96, ignore_eos=True)
    assert out.finish_reason == "length" and out.usage.completion_tokens == 96, out
    counts = [n for n in out.chunk_tokens if n > 0]
    assert sum(counts) == 96, out.chunk_tokens
    middle = counts[2:-1]
    assert len(middle) >= 3, counts
    assert all(STREAM_INTERVAL <= n <= STREAM_INTERVAL + MAX_DRAFT_LEN for n in middle), counts

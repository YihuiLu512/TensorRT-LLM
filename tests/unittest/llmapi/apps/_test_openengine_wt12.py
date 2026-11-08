# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stop strings, stop token ids, usage and decoding options over ``trtllm-serve``.

Every case first asks for the greedy output's token ids (``detokenize=False``)
and derives what ``LLM`` returns from them: a stop string the engine matches
on its token ids ends the output before its tokens (or after them with
``include_stop_str_in_output``), the text is the decoded tokens that are
kept (``LLM``'s streamed text also keeps the text of the stop string's
tokens before the last one), and ``completion_tokens`` counts the kept
tokens. The stop strings are taken from
the output itself, so they span several tokens wherever the model happened
to put them. The requests with stop strings or stop token ids leave out
``ignore_eos``, with which ``LLM`` gives its engine no stop words.
"""

import os
from dataclasses import dataclass

import openai
import pytest
from transformers import AutoTokenizer

from tensorrt_llm.tokenizer import TransformersTokenizer

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")

_PROMPT = "Write a short story about a lighthouse keeper who finds a message in a bottle."
_MAX_TOKENS = 48
# A chat-formatted prompt whose greedy output ends its turn and goes on past it,
# so the output holds special tokens.
_CHAT_PROMPT = (
    "<|im_start|>user\nSay hello in one word.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


@pytest.fixture(scope="module")
def model_name():
    return "Qwen3/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server(model_name: str):
    args = ["--backend", "pytorch", "--kv_cache_free_gpu_memory_fraction", "0.2"]
    with RemoteOpenAIServer(get_model_path(model_name), args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def tokenizer(model_name: str) -> TransformersTokenizer:
    return TransformersTokenizer(AutoTokenizer.from_pretrained(get_model_path(model_name)))


def _complete(client, model_name, prompt=_PROMPT, *, stream=False, **extra_body):
    kwargs = dict(
        model=model_name,
        prompt=prompt,
        max_tokens=_MAX_TOKENS,
        temperature=0.0,
        extra_body=extra_body,
    )
    if not stream:
        completion = client.completions.create(**kwargs)
        choice = completion.choices[0]
        return choice.text, choice.finish_reason, choice.stop_reason, completion.usage
    text, finish_reason, stop_reason, usage = "", None, None, None
    for chunk in client.completions.create(
        stream=True, stream_options={"include_usage": True}, **kwargs
    ):
        if chunk.usage is not None:
            usage = chunk.usage
        for choice in chunk.choices:
            assert finish_reason is None, "no text after the output finished"
            text += choice.text
            finish_reason, stop_reason = choice.finish_reason, choice.stop_reason
    return text, finish_reason, stop_reason, usage


@pytest.fixture(scope="module")
def output_ids(client, model_name):
    completion = client.completions.create(
        model=model_name,
        prompt=_PROMPT,
        max_tokens=_MAX_TOKENS,
        temperature=0.0,
        extra_body={"detokenize": False},
    )
    token_ids = completion.choices[0].token_ids
    assert token_ids
    return token_ids


@dataclass(frozen=True)
class _Stop:
    text: str
    start: int
    num_tokens: int


def _stops(tokenizer, token_ids) -> list[_Stop]:
    """Stop strings made of 2 or 3 consecutive output tokens, in output order.

    Each one encodes to exactly those tokens, so the engine matches it on them
    as ``LLM``'s does, and neither its tokens nor its text occur earlier.
    """
    text = tokenizer.decode(token_ids)
    stops, start = [], 3
    while start < len(token_ids) - 3:
        for num_tokens in (2, 3):
            word = token_ids[start : start + num_tokens]
            stop = tokenizer.decode(word)
            before = tokenizer.decode(token_ids[:start])
            if (
                stop.strip()
                and "�" not in stop
                and tokenizer.encode(stop, add_special_tokens=False) == word
                and tokenizer.decode(token_ids[: start + num_tokens]) == before + stop
                and text.find(stop) == len(before)
                and all(token_ids[i : i + num_tokens] != word for i in range(start))
            ):
                stops.append(_Stop(stop, start, num_tokens))
                start += num_tokens
                break
        else:
            start += 1
    assert len(stops) >= 2, f"no two stop strings in {text!r}"
    return stops


def _expected(tokenizer, token_ids, stop: _Stop, include: bool):
    kept = stop.start + (stop.num_tokens if include else 0)
    return tokenizer.decode(token_ids[:kept]), kept


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("include", [False, True], ids=["exclude_stop", "include_stop"])
def test_stop_string_variants(client, model_name, tokenizer, output_ids, stream, include):
    """A stop string spanning tokens, several stop strings, and a stop token id.

    The output ends where ``LLM``'s does. When streaming over the chain, the
    text held back while a stop string might be starting is never sent;
    ``LLM`` streams the text of a stop string's tokens before the last one and
    keeps it. ``completion_tokens`` counts the tokens kept, and a repeated
    request reuses the prompt's cache.
    """
    first, second = _stops(tokenizer, output_ids)[:2]

    for stops in ([first.text], [second.text, first.text]):
        text, finish_reason, stop_reason, usage = _complete(
            client, model_name, stream=stream, stop=stops, include_stop_str_in_output=include
        )
        expected_text, kept = _expected(tokenizer, output_ids, first, include)
        if stream and not include and not CHAIN:
            expected_text = tokenizer.decode(output_ids[: first.start + first.num_tokens - 1])
        assert (text, finish_reason, stop_reason) == (expected_text, "stop", first.text), stops
        assert usage.completion_tokens == kept, stops
        assert usage.total_tokens == usage.prompt_tokens + kept

    # The stop string ends the output when the engine reaches it, so the
    # prompt's blocks are reusable by the next request.
    _, _, _, usage = _complete(client, model_name, stream=stream, stop=[second.text])
    assert usage.completion_tokens == second.start
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens > 0

    # The output stops at the first occurrence of the stop token.
    stop_token = output_ids[first.start]
    stop_index = output_ids.index(stop_token)
    text, finish_reason, stop_reason, usage = _complete(
        client,
        model_name,
        stream=stream,
        stop_token_ids=[stop_token],
        include_stop_str_in_output=include,
    )
    kept = stop_index + (1 if include else 0)
    assert (text, finish_reason, stop_reason) == (
        tokenizer.decode(output_ids[:kept]),
        "stop",
        stop_token,
    )
    assert usage.completion_tokens == kept


@pytest.mark.parametrize("skip_special_tokens", [True, False], ids=["skip", "keep"])
@pytest.mark.parametrize(
    "spaces_between_special_tokens", [True, False], ids=["spaces", "no_spaces"]
)
def test_detokenize_options(
    client: openai.OpenAI,
    model_name,
    tokenizer,
    skip_special_tokens,
    spaces_between_special_tokens,
):
    """The text is ``LLM``'s decode of the output's token ids, streamed or not."""
    completion = client.completions.create(
        model=model_name,
        prompt=_CHAT_PROMPT,
        max_tokens=_MAX_TOKENS,
        temperature=0.0,
        extra_body={"ignore_eos": True, "detokenize": False},
    )
    token_ids = completion.choices[0].token_ids
    assert completion.choices[0].text == ""
    assert set(token_ids) & set(tokenizer.tokenizer.all_special_ids), (
        "the output holds special tokens"
    )
    expected = tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        spaces_between_special_tokens=spaces_between_special_tokens,
    )

    for stream in (False, True):
        text, finish_reason, _, usage = _complete(
            client,
            model_name,
            _CHAT_PROMPT,
            stream=stream,
            ignore_eos=True,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        assert (text, finish_reason) == (expected, "length"), stream
        assert usage.completion_tokens == len(token_ids)

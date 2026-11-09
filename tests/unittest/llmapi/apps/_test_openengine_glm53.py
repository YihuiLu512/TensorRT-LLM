# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash served through both OpenEngine cases, for a same-model comparison.

case-02 (``trtllm``) serves GLM-5.3-Flash from TRT-LLM's own OpenEngine engine; case-01
(``glm``) from the glm-5.3-flash fork's engine. The serve tests elsewhere use small models the
glm engine cannot serve, so this module runs one set of tests on both.

The chain is chosen per pytest session by ``TRTLLM_TEST_OPENENGINE_CHAIN`` (the harness in
``tests/integration/defs/openengine_chain`` proves each test went through it). Every test is
parametrized by ``case``; a session runs the case its chain serves and skips the other. With
the variable unset the server is the single in-process ``trtllm-serve``: the ``trtllm`` case
then runs as the same-model baseline, with case-02's expectations except where a test says the
in-process server differs.

On the chain a test also needs serve-side support for what it exercises, which the OpenEngine
front end gains feature by feature; a test whose feature it does not support yet fails there.

Needs 4 Blackwell GPUs and ``${LLM_MODELS_ROOT}/GLM-5.3-Flash`` (FP8). Functional servers
disable KV block reuse (glm's outputs change with prefix reuse); only the cached-token tests
enable it.
"""

import contextlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import openai
import pytest
import requests
import yaml

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = [pytest.mark.threadleak(enabled=False)]

MODEL_DIR = "GLM-5.3-Flash"
MODEL_NAME = "zai-org/GLM-5.3-Flash"
CASES = ("trtllm", "glm")
CHAIN_ENV = "TRTLLM_TEST_OPENENGINE_CHAIN"
# glm declares one sequence per request and the TRT-LLM engine 16.
MAX_NUM_SEQUENCES = {"trtllm": 16, "glm": 1}
# glm returns at most 20 top logprobs (its sampler's kMaxLogprobs); the TRT-LLM engine 100.
MAX_TOP_LOGPROBS = {"trtllm": 100, "glm": 20}

BASE_ARGS = [
    "--tp_size",
    "4",
    "--ep_size",
    "4",
    "--max_batch_size",
    "32",
    "--max_num_tokens",
    "8192",
    "--max_seq_len",
    "8192",
    "--reasoning_parser",
    "deepseek-r1",
    "--tool_parser",
    "glm47",
]
BASE_CONFIG = {
    "kv_cache_config": {"free_gpu_memory_fraction": 0.5, "enable_block_reuse": False},
    "cuda_graph_config": {"max_batch_size": 32, "enable_padding": True},
    "enable_chunked_prefill": True,
    "disable_mm_encoder": True,
    "guided_decoding_backend": "xgrammar",
}


def session_chain() -> str:
    chain = os.environ.get(CHAIN_ENV) or "none"
    if chain not in ("none", *CASES):  # a typo would otherwise skip every test
        raise ValueError(f"{CHAIN_ENV}={chain!r}; expected unset, 'trtllm' or 'glm'")
    return chain


@pytest.fixture(scope="module", params=CASES)
def case(request) -> str:
    chain = session_chain()
    served = "trtllm" if chain == "none" else chain
    if request.param != served:
        pytest.skip(f"this session serves case {served!r} ({CHAIN_ENV}={chain})")
    return request.param


@pytest.fixture(scope="module")
def chained() -> bool:
    """False for the in-process baseline, where no OpenEngine engine is involved."""
    return session_chain() != "none"


@pytest.fixture(scope="module")
def model_path() -> str:
    path = get_model_path(MODEL_DIR)
    if not os.path.isdir(path):
        pytest.skip(f"{path} is not available")
    return path


@pytest.fixture(scope="module")
def tokenizer(model_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


def merged_config(config=None):
    """BASE_CONFIG with ``config`` merged in one level deep."""
    merged = dict(BASE_CONFIG)
    for key, value in (config or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def merged_args(args=()):
    """BASE_ARGS with ``args``, flag-value pairs that replace or extend them. A flag given on the
    command line wins over the same field in the config file."""
    flags = dict(zip(BASE_ARGS[::2], BASE_ARGS[1::2]))
    flags.update(zip(list(args)[::2], list(args)[1::2]))
    return [item for pair in flags.items() for item in pair]


def start_server(model_path, *, args=(), config=None, env=None):
    """``trtllm-serve`` (or the chain the session selects) with ``merged_args(args)`` and
    ``merged_config(config)``."""
    return RemoteOpenAIServer(
        model_path, merged_args(args), extra_config=merged_config(config), env=env
    )


class Servers:
    """At most one GLM-5.3-Flash server at a time: two do not fit on 4 GPUs. Asking for another
    configuration stops the running server, so tests that share one are kept together."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.key = None
        self.srv = None

    def get(self, key, **kwargs):
        if self.key != key:
            self.close()
            self.srv = start_server(self.model_path, **kwargs)
            self.key = key
        return self.srv

    def close(self):
        srv, self.srv, self.key = self.srv, None, None
        if srv is not None:
            srv.terminate()


@pytest.fixture(scope="module")
def servers(case, model_path):
    s = Servers(model_path)
    yield s
    s.close()


# Block reuse with periodic recurrent-state snapshots, which GLM's hybrid cache needs for a
# prefix hit (``accuracy/test_glm53_flash.py::test_block_reuse``).
REUSE_CONFIG = {
    "kv_cache_config": {
        "enable_block_reuse": True,
        "mamba_state_config": {"periodic_snapshot_interval": 256},
    }
}


@pytest.fixture
def server(servers):
    return servers.get("base")


@pytest.fixture
def reuse_server(servers):
    return servers.get("reuse", config=REUSE_CONFIG)


def client(srv) -> openai.OpenAI:
    """One client per server, so its connections are reused."""
    if getattr(srv, "_module_client", None) is None:
        srv._module_client = srv.get_client()
    return srv._module_client


def expect_400(call, *reasons):
    """The call fails with 400, and the message names one of ``reasons`` when any are given."""
    with pytest.raises(openai.BadRequestError) as info:
        call()
    message = str(info.value)
    assert not reasons or any(r in message for r in reasons), message
    return message


# A prompt whose greedy continuation is long, so a reuse hit and a stop string both have room.
LONG_PROMPT = "The capital of France is Paris. The capital of Germany is Berlin. " * 40
COUNT_PROMPT = "1\n2\n3\n4\n"  # the greedy continuation counts on: 5\n6\n7\n...


def complete(srv, prompt=COUNT_PROMPT, **kwargs):
    kwargs.setdefault("max_tokens", 24)
    kwargs.setdefault("temperature", 0.0)
    return client(srv).completions.create(model=MODEL_NAME, prompt=prompt, **kwargs)


def stop_reason(choice):
    return getattr(choice, "stop_reason", None)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_and_completions_basic(server, case, stream):
    c = client(server)
    if stream:
        chunks = list(
            c.completions.create(
                model=MODEL_NAME,
                prompt=COUNT_PROMPT,
                max_tokens=8,
                temperature=0.0,
                stream=True,
                stream_options={"include_usage": True},
            )
        )
        text = "".join(ch.choices[0].text for ch in chunks if ch.choices)
        assert chunks[-1].usage.completion_tokens == 8
        assert [ch.choices[0].finish_reason for ch in chunks if ch.choices][-1] == "length"
    else:
        out = complete(server, max_tokens=8)
        text = out.choices[0].text
        assert out.choices[0].finish_reason == "length"
        assert out.usage.completion_tokens == 8
        assert out.usage.total_tokens == out.usage.prompt_tokens + 8
    assert text.startswith("5\n6"), repr(text)

    messages = [{"role": "user", "content": "Name the capital of France in one word."}]
    if stream:
        chunks = list(
            c.chat.completions.create(
                model=MODEL_NAME, messages=messages, max_tokens=256, temperature=0.0, stream=True
            )
        )
        deltas = [ch.choices[0].delta for ch in chunks if ch.choices]
        said = "".join(
            (d.content or "") + (getattr(d, "reasoning_content", None) or "") for d in deltas
        )
    else:
        out = c.chat.completions.create(
            model=MODEL_NAME, messages=messages, max_tokens=256, temperature=0.0
        )
        m = out.choices[0].message
        said = (m.content or "") + (getattr(m, "reasoning_content", None) or "")
        assert out.usage.completion_tokens > 0
    assert "Paris" in said, said


def repeated_ngrams(text: str, n: int = 3) -> int:
    words = text.split()
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(grams) - len(set(grams))


def test_penalties(server, case):
    prompt = "Repeat after me forever: the cat sat on the mat, the cat sat on the mat,"
    if case == "glm":
        for name in ("repetition_penalty", "presence_penalty", "frequency_penalty"):
            assert name in expect_400(
                lambda: complete(server, prompt=prompt, extra_body={name: 1.2})
            )
        return
    plain = complete(server, prompt=prompt, max_tokens=64).choices[0].text
    penalized = (
        complete(server, prompt=prompt, max_tokens=64, extra_body={"repetition_penalty": 1.5})
        .choices[0]
        .text
    )
    assert penalized != plain
    assert repeated_ngrams(penalized) < repeated_ngrams(plain), (plain, penalized)
    for name, value in (("presence_penalty", 1.0), ("frequency_penalty", 1.0)):
        assert complete(server, prompt=prompt, extra_body={name: value}).choices[0].text


def engine_calls(srv):
    """Generate calls the engine has received; None for the in-process server."""
    handle = getattr(srv, "_chain_handle", None)
    if handle is None:
        return None
    from openengine_chain.proof import read_engine_counter  # on PYTHONPATH when chained

    calls = read_engine_counter(handle)
    if calls is None:
        pytest.fail("the engine reports no Generate counter in GetLoad")
    return calls


def test_n_best_of(server, case, chained):
    def run(n, **kwargs):
        before = engine_calls(server)
        out = complete(
            server,
            prompt="Write a short poem about the sea.\n",
            max_tokens=16,
            temperature=1.0,
            n=n,
            **kwargs,
        )
        assert sorted(ch.index for ch in out.choices) == list(range(n))
        after = engine_calls(server)
        return None if before is None else after - before

    log = getattr(server._chain_handle, "frontend_log", None) if chained else None
    log_start = os.path.getsize(log) if log and os.path.exists(log) else 0
    calls = run(3)
    if not chained:
        run(2, extra_body={"best_of": 4})  # in-process: best_of > n works
        return
    limit = MAX_NUM_SEQUENCES[case]
    # n up to the engine's max_num_sequences is one native call; above it, serve splits the request.
    assert calls == -(-3 // limit)
    if case == "trtllm":
        assert run(limit + 1) == 2
    assert log
    # what the front end logged for these requests; the log is written asynchronously
    deadline = time.time() + 30
    while True:
        with open(log, errors="replace") as f:
            f.seek(log_start)
            if "max_num_sequences" in f.read():
                break
        assert time.time() < deadline, "the front end did not log max_num_sequences"
        time.sleep(1)
    # best_of > n needs trtllm.request.best_of, which neither engine declares.
    expect_400(lambda: complete(server, n=2, extra_body={"best_of": 4}), "best_of")


@pytest.mark.parametrize("top", [5, 20, 21])
def test_logprobs(server, case, chained, top):
    def call():
        return client(server).chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=8,
            temperature=0.0,
            logprobs=True,
            top_logprobs=top,
        )

    if chained and top > MAX_TOP_LOGPROBS[case]:
        assert "20" in expect_400(call, "logprobs")  # rejected, not truncated
        return
    content = call().choices[0].logprobs.content
    assert content and all(len(entry.top_logprobs) == top for entry in content)
    assert all(entry.logprob <= 0.0 for entry in content)


def test_guided_json_schema(server, case, chained):
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
        "required": ["city", "population"],
        "additionalProperties": False,
    }

    def call():
        return complete(
            server,
            prompt="Describe Paris as JSON.\n",
            max_tokens=128,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "city", "schema": schema},
                }
            },
        )

    if chained and case == "glm":
        expect_400(call, "guided_decoding", "response_format")
        return
    out = call().choices[0]
    data = json.loads(out.text)
    assert set(data) == {"city", "population"} and isinstance(data["population"], int)


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def test_reasoning_and_tools(server, case):
    def no_think_tags(message):
        return "<think>" not in (message.content or "") and "</think>" not in (
            message.content or ""
        )

    c = client(server)
    # The reasoning parser splits a reasoning out of the content; reasoning_effort is a
    # chat-template kwarg of GLM-5.3-Flash (low / high / max).
    message = (
        c.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=512,
            temperature=0.0,
            messages=[{"role": "user", "content": THINK_PROMPT}],
            extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}},
        )
        .choices[0]
        .message
    )
    assert getattr(message, "reasoning_content", None), "the reasoning parser split nothing out"
    assert no_think_tags(message)

    # The tool parser returns the call. The model may call a tool without reasoning first, so the
    # reasoning is not asserted here.
    out = c.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=4096,
        temperature=0.0,
        tools=[WEATHER_TOOL],
        messages=[
            {"role": "user", "content": "What is the weather in Paris right now? Use the tool."}
        ],
    )
    message = out.choices[0].message
    assert no_think_tags(message)
    assert out.choices[0].finish_reason == "tool_calls" and message.tool_calls
    call = message.tool_calls[0].function
    assert call.name == "get_weather" and "paris" in json.loads(call.arguments)["city"].lower()

    # clear_thinking (a chat-template kwarg) drops the earlier turn's reasoning from the prompt.
    history = [
        {"role": "user", "content": "Pick a number."},
        {
            "role": "assistant",
            "content": "7",
            "reasoning_content": "Let me think about numbers. " * 20,
        },
        {"role": "user", "content": "Double it."},
    ]
    tokens = {}
    for clear in (False, True):
        out = c.chat.completions.create(
            model=MODEL_NAME,
            messages=history,
            max_tokens=1,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"clear_thinking": clear}},
        )
        tokens[clear] = out.usage.prompt_tokens
    assert tokens[True] < tokens[False], tokens


THINK_PROMPT = "Think carefully, step by step, about how many primes are below 100, then answer."
# Reasons for well over 1024 tokens without a budget: room for the Anthropic API's minimum one.
LONG_THINK_PROMPT = (
    "List every prime number below 500, checking each candidate one by one, then give their count."
)


def test_thinking_token_budget(server, case, chained, tokenizer):
    budget = 64
    slack = (
        len(tokenizer.encode("</think>", add_special_tokens=False)) + 2
    )  # end marker; re-tokenization
    c = client(server)
    messages = [{"role": "user", "content": THINK_PROMPT}]

    def chat(**kw):
        return c.chat.completions.create(
            model=MODEL_NAME, messages=messages, max_tokens=512, temperature=0.0, **kw
        )

    if chained and case == "glm":
        expect_400(lambda: chat(extra_body={"thinking_token_budget": budget}), "thinking")
        return

    def n_tokens(text):
        return len(tokenizer.encode(text or "", add_special_tokens=False))

    unbounded = chat().choices[0].message.reasoning_content
    assert n_tokens(unbounded) > budget + slack  # otherwise the budget proves nothing

    # chat (a bounded reasoning is still a reasoning: an empty one would pass any bound)
    assert (
        0
        < n_tokens(
            chat(extra_body={"thinking_token_budget": budget}).choices[0].message.reasoning_content
        )
        <= budget + slack
    )
    # completions: the raw text holds the reasoning up to </think>
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = (
        complete(
            server, prompt=prompt, max_tokens=512, extra_body={"thinking_token_budget": budget}
        )
        .choices[0]
        .text
    )
    assert "</think>" in text and 0 < n_tokens(text.split("</think>")[0]) <= budget + slack
    # Responses
    r = requests.post(
        server.url_for("v1/responses"),
        timeout=600,
        json={
            "model": MODEL_NAME,
            "input": THINK_PROMPT,
            "max_output_tokens": 512,
            "temperature": 0.0,
            "thinking_token_budget": budget,
        },
    )
    assert r.status_code == 200, r.text
    resp = r.json()
    reasoning = "".join(
        part.get("text", "")
        for item in resp["output"]
        if item["type"] == "reasoning"
        for part in item.get("content") or item.get("summary") or []
    )
    assert 0 < n_tokens(reasoning) <= budget + slack, resp
    # Anthropic Messages: thinking.budget_tokens (at least 1024 by that API), on a prompt whose
    # reasoning runs past that without a budget
    long_messages = [{"role": "user", "content": LONG_THINK_PROMPT}]
    unbounded = (
        c.chat.completions.create(
            model=MODEL_NAME, messages=long_messages, max_tokens=4096, temperature=0.0
        )
        .choices[0]
        .message.reasoning_content
    )
    assert n_tokens(unbounded) > 1024 + slack  # otherwise the budget proves nothing
    r = requests.post(
        server.url_for("v1/messages"),
        timeout=600,
        json={
            "model": MODEL_NAME,
            "max_tokens": 1536,
            "temperature": 0.0,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": long_messages,
        },
    )
    assert r.status_code == 200, r.text
    msg = r.json()
    thinking = "".join(
        block.get("thinking", "") for block in msg["content"] if block["type"] == "thinking"
    )
    assert 0 < n_tokens(thinking) <= 1024 + slack, msg


def test_seed(server, case):
    # glm's seed scope is batch_dependent, so the same request is replayed serially.
    sampled = [
        complete(
            server,
            prompt="Write a short poem about the sea.\n",
            max_tokens=32,
            temperature=1.0,
            seed=1234,
        )
        .choices[0]
        .text
        for _ in range(2)
    ]
    assert sampled[0] == sampled[1]


def test_sampling_and_stop(servers, case, tokenizer):
    server = servers.get("base")
    # min_p is declared by both engines.
    assert complete(server, temperature=1.0, extra_body={"min_p": 0.5}).choices[0].text

    # stop strings: the request ends at the match.
    out = complete(server, stop=["7"]).choices[0]
    assert out.text == "5\n6\n" and out.finish_reason == "stop" and stop_reason(out) == "7"
    out = complete(server, stop=["7"], extra_body={"include_stop_str_in_output": True}).choices[0]
    assert out.text == "5\n6\n7" and stop_reason(out) == "7"
    # a stop string spanning tokens, listed after one that never occurs
    out = complete(server, stop=["zzz", "6\n7"]).choices[0]
    assert out.text == "5\n" and stop_reason(out) == "6\n7"

    # stop_token_ids: the newline token
    newline = tokenizer.encode("\n", add_special_tokens=False)
    assert len(newline) == 1
    out = complete(server, extra_body={"stop_token_ids": newline}).choices[0]
    assert out.text == "5" and out.finish_reason == "stop" and stop_reason(out) == newline[0]

    # min_tokens: no stop string ends the request before min_tokens output tokens. The
    # engines get there differently (TRT-LLM bans a one-token stop string's token until then; glm
    # generates it and does not count the match), so only the common outcome is asserted. The
    # stop string is kept so completion_tokens counts every generated token.
    out = complete(
        server, stop=["\n"], extra_body={"min_tokens": 6, "include_stop_str_in_output": True}
    )
    assert out.choices[0].finish_reason == "stop" and stop_reason(out.choices[0]) == "\n"
    assert out.choices[0].text.endswith("\n") and out.usage.completion_tokens >= 6
    # ... and EOS waits for min_tokens too, and ignore_eos generates past it: a control request
    # shows where this prompt's EOS comes without either.
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with the single word yes."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    control = complete(server, prompt=prompt, max_tokens=2048)  # room for the default reasoning
    n = control.usage.completion_tokens
    assert control.choices[0].finish_reason == "stop" and n < 2048, "the control never reached EOS"
    out = complete(server, prompt=prompt, max_tokens=n + 64, extra_body={"min_tokens": n + 8})
    assert out.usage.completion_tokens >= n + 8, n
    out = complete(server, prompt=prompt, max_tokens=n + 8, extra_body={"ignore_eos": True})
    assert out.usage.completion_tokens == n + 8 and out.choices[0].finish_reason == "length"

    # usage after a stop-string match: complete, with the prefix-cache hit. glm's greedy
    # continuation of LONG_PROMPT can leave the pattern (a numbered list) and never reach
    # " Berlin", so glm's prompt stops mid-sentence and the continuation finishes the sentence.
    reuse_server = servers.get("reuse", config=REUSE_CONFIG)
    stop_prompt = LONG_PROMPT + "The capital of France is" if case == "glm" else LONG_PROMPT
    complete(reuse_server, prompt=stop_prompt, max_tokens=4)
    for stream in (False, True):
        if stream:
            chunks = list(
                client(reuse_server).completions.create(
                    model=MODEL_NAME,
                    prompt=stop_prompt,
                    max_tokens=64,
                    temperature=0.0,
                    stop=[" Berlin"],
                    stream=True,
                    stream_options={"include_usage": True},
                )
            )
            usage = chunks[-1].usage
            text = "".join(ch.choices[0].text for ch in chunks if ch.choices)
        else:
            out = complete(reuse_server, prompt=stop_prompt, max_tokens=64, stop=[" Berlin"])
            usage, text = out.usage, out.choices[0].text
            assert (
                out.choices[0].finish_reason == "stop" and stop_reason(out.choices[0]) == " Berlin"
            )
        assert "Berlin" not in text
        assert usage.prompt_tokens > 256 and 0 < usage.completion_tokens < 64
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        assert usage.prompt_tokens_details.cached_tokens > 0, stream


@pytest.mark.parametrize("api", ["completions", "chat"])
def test_usage_and_cached_tokens(reuse_server, case, api):
    c = client(reuse_server)

    def call(stream):
        kwargs = dict(model=MODEL_NAME, max_tokens=8, temperature=0.0)
        if stream:
            kwargs.update(stream=True, stream_options={"include_usage": True})
        if api == "completions":
            out = c.completions.create(prompt=LONG_PROMPT, **kwargs)
        else:
            out = c.chat.completions.create(
                messages=[{"role": "user", "content": LONG_PROMPT}], **kwargs
            )
        return list(out)[-1].usage if stream else out.usage

    first = call(False)
    for stream in (False, True):
        usage = call(stream)
        assert usage.prompt_tokens == first.prompt_tokens and usage.completion_tokens == 8
        assert usage.total_tokens == usage.prompt_tokens + 8
        assert 0 < usage.prompt_tokens_details.cached_tokens < usage.prompt_tokens, (stream, usage)


def test_postproc_and_frontends(servers, case):
    srv = servers.get(
        "frontends", args=["--num_postprocess_workers", "4", "--num_serve_frontends", "8"]
    )
    expected = complete(srv, max_tokens=8).choices[0].text

    def one(i):
        if i % 2:
            chunks = client(srv).completions.create(
                model=MODEL_NAME, prompt=COUNT_PROMPT, max_tokens=8, temperature=0.0, stream=True
            )
            return "".join(ch.choices[0].text for ch in chunks if ch.choices)
        return complete(srv, max_tokens=8).choices[0].text

    with ThreadPoolExecutor(16) as pool:
        texts = list(pool.map(one, range(32)))
    assert texts == [expected] * 32


def test_video_url(servers, case, chained):
    """The request body of ``accuracy/test_glm53_flash.py::test_video_url``."""
    from ..test_llm import llm_models_root

    video = os.path.join(llm_models_root(), "multimodals/test_data/OAI-sora-tokyo-walk.mp4")
    if not os.path.isfile(video):
        pytest.skip(f"video fixture is unavailable: {video}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video}},
                {"type": "text", "text": "Describe what happens in this video briefly."},
            ],
        }
    ]
    kwargs = dict(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_completion_tokens=256,
        extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
    )
    pytest.importorskip("cv2", reason="video decoding requires OpenCV")
    pytest.importorskip("transformers.models.glm5_next.processing_glm5_next")
    args = [
        "--max_batch_size",
        "4",
        "--media_io_kwargs",
        json.dumps({"video": {"num_frames": 4, "fps": 2}}),
    ]
    env = dict(os.environ, TRTLLM_MEDIA_ALLOWED_LOCAL_PATH=os.path.dirname(video))
    config = {"disable_mm_encoder": False, "cuda_graph_config": {"max_batch_size": 4}}
    srv = servers.get("video", args=args, config=config, env=env)
    if chained and case == "glm":
        expect_400(lambda: client(srv).chat.completions.create(**kwargs), "video", "multimodal")
        return
    out = client(srv).chat.completions.create(**kwargs)
    message = out.choices[0].message
    assert ((message.content or "") + (getattr(message, "reasoning_content", None) or "")).strip()
    assert out.usage.completion_tokens > 0


MTP3 = {"decoding_type": "MTP", "max_draft_len": 3}
# Attention DP runs with the per-rank token budget of accuracy/test_glm53_flash.py's attention-DP
# tests, on the command line (BASE_ARGS sets --max_num_tokens, which wins over the config file).
ADP_ARGS = ["--max_num_tokens", "4096"]
# name -> (args, config overrides, spec_dec_algo, kv_cache_quant_algo of the reference row)
GSM8K_RUNS = {
    "tp4ep4": ([], {}, None, None),
    "mtp3": ([], {"speculative_config": MTP3}, "MTP", None),
    # the glm engine has no FP8 KV cache: its attention-DP run keeps the default KV dtype
    "adp_mtp3": (ADP_ARGS, {"speculative_config": MTP3, "enable_attention_dp": True}, "MTP", None),
    "adp_mtp3_fp8kv": (
        ADP_ARGS,
        {
            "speculative_config": MTP3,
            "enable_attention_dp": True,
            "kv_cache_config": {"dtype": "fp8"},
        },
        "MTP",
        "FP8",
    ),
}
# GSM8K sends no guided-decoding requests, and with MTP the xgrammar backend can hang the engine,
# so the GSM8K servers run without a guided-decoding backend.
GSM8K_CONFIG = {"guided_decoding_backend": None}
GSM8K_ALPHA, GSM8K_SIGMA, GSM8K_SAMPLES = 0.05, 50.0, 1319  # accuracy_core.GSM8K
GSM8K_MAX_OUTPUT = 256
MCNEMAR_ALPHA = 0.05


class EndpointLLM:
    """The parts of ``LLM`` the GSM8K evaluator uses, posting each prompt to /v1/completions:
    only max_tokens, temperature 0 and stop are sent."""

    def __init__(self, base_url, tokenizer, workers=32):
        self.base_url, self.tokenizer = base_url, tokenizer
        self.pool = ThreadPoolExecutor(workers)
        self.session = requests.Session()
        self.session.mount("http://", requests.adapters.HTTPAdapter(pool_maxsize=workers))
        self.errors = []
        self.lock = threading.Lock()

    def _send(self, prompt, sampling_params):
        body = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": sampling_params.max_tokens or GSM8K_MAX_OUTPUT,
            "temperature": 0.0,
        }
        stop = sampling_params.stop
        if stop:
            body["stop"] = [stop] if isinstance(stop, str) else list(stop)
        r = self.session.post(f"{self.base_url}/v1/completions", json=body, timeout=900)
        if r.status_code != 200:
            with self.lock:
                self.errors.append(r.text[:500])
            return SimpleNamespace(outputs=[SimpleNamespace(text="")])
        return SimpleNamespace(outputs=[SimpleNamespace(text=r.json()["choices"][0]["text"])])

    def generate_async(self, prompt, sampling_params=None, streaming=False):
        return self.pool.submit(self._send, prompt, sampling_params)


def gsm8k_threshold(spec_dec_algo, kv_cache_quant_algo):
    """The accuracy suite's pass threshold (accuracy_core.compute_threshold)."""
    from scipy.stats import norm

    path = os.path.join(
        os.path.dirname(__file__),
        *[os.pardir] * 3,
        "integration/defs/accuracy/references/gsm8k.yaml",
    )
    with open(path) as f:
        rows = yaml.safe_load(f)[MODEL_NAME]
    row = next(
        r
        for r in rows
        if r.get("quant_algo") == "FP8_BLOCK_SCALES"
        and r.get("spec_dec_algo") == spec_dec_algo
        and r.get("kv_cache_quant_algo") == kv_cache_quant_algo
    )
    return row["accuracy"] + norm.ppf(GSM8K_ALPHA) * (2 * GSM8K_SIGMA**2 / GSM8K_SAMPLES) ** 0.5


def run_gsm8k(base_url, model_path, out_dir):
    """Accuracy and per-question correctness (strict-match) of one endpoint."""
    from tensorrt_llm.evaluate import GSM8K
    from tensorrt_llm.llmapi.tokenizer import load_hf_tokenizer
    from tensorrt_llm.sampling_params import SamplingParams

    from ..test_llm import llm_models_root

    llm = EndpointLLM(base_url, load_hf_tokenizer(model_path))
    evaluator = GSM8K(
        dataset_path=os.path.join(llm_models_root(), "datasets/openai/gsm8k"),
        random_seed=0,
        num_samples=GSM8K_SAMPLES,
        log_samples=True,
        output_path=out_dir,
    )
    try:
        score = evaluator.evaluate(
            llm,
            SamplingParams(max_tokens=GSM8K_MAX_OUTPUT, truncate_prompt_tokens=4096),
            scores_filter=None,
        )
    finally:
        # after a failure, do not wait for the questions still queued
        llm.pool.shutdown(wait=False, cancel_futures=True)
        llm.session.close()
    assert not llm.errors, llm.errors[:3]
    with open(os.path.join(out_dir, "samples_gsm8k.json")) as f:
        samples = json.load(f)["samples"]["gsm8k"]
    correct = {
        s["doc_id"]: float(s["exact_match"]) >= 0.5
        for s in samples
        if s.get("filter") == "strict-match"
    }
    return score, correct


def mcnemar_exact(b, c):
    """Exact McNemar p-value: the two-sided binomial test of the discordant pairs."""
    from scipy.stats import binomtest

    return binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0


@pytest.mark.timeout(10800)
@pytest.mark.parametrize("run", list(GSM8K_RUNS))
def test_gsm8k_over_chain(servers, case, chained, model_path, run, tmp_path, monkeypatch):
    # EndpointLLM's outputs carry text only, not the perf metrics the spec-decoding stats read
    monkeypatch.delenv("TLLM_EVAL_SPEC_STATS", raising=False)
    args, overrides, spec_dec_algo, kv_algo = GSM8K_RUNS[run]
    overrides = {**GSM8K_CONFIG, **overrides}
    if case == "glm" and kv_algo is not None:
        pytest.skip("the glm engine has no FP8 KV cache")
    if case == "trtllm" and run == "adp_mtp3":
        pytest.skip("adp_mtp3_fp8kv covers attention DP on this engine")
    srv = servers.get(f"gsm8k-{run}", args=args, config=overrides)
    score, correct = run_gsm8k(srv.url_root, model_path, str(tmp_path / "served"))
    threshold = gsm8k_threshold(spec_dec_algo, kv_algo)
    assert score >= threshold, f"GSM8K {score:.2f} < threshold {threshold:.2f}"
    if not (chained and case == "glm"):
        return
    # case-01: pair the questions with glm's own HTTP server on the same configuration.
    servers.close()
    native, port = _start_glm_http(model_path, args, overrides, tmp_path)
    try:
        _, native_correct = run_gsm8k(
            f"http://127.0.0.1:{port}", model_path, str(tmp_path / "native")
        )
    finally:
        _stop_process_group(native)
    b = sum(correct[d] and not native_correct[d] for d in correct)
    c = sum(native_correct[d] and not correct[d] for d in correct)
    assert mcnemar_exact(b, c) >= MCNEMAR_ALPHA, f"chain-only right {b}, native-only right {c}"


def _start_glm_http(model_path, args, overrides, tmp_path):
    """glm's native HTTP server (``$TRTLLM_TEST_GLM_SERVE_CMD``) with the engine's arguments."""
    import shlex
    import subprocess

    from openengine_chain.launch import free_port  # on PYTHONPATH when chained

    config_path = tmp_path / "glm_http.yaml"
    config_path.write_text(yaml.safe_dump(merged_config(overrides)))
    port = free_port()
    cmd = (
        shlex.split(os.environ["TRTLLM_TEST_GLM_SERVE_CMD"])
        + [model_path, "--host", "127.0.0.1", "--port", str(port), "--config", str(config_path)]
        + merged_args(args)
    )
    log = tmp_path / "glm_http.log"
    with open(log, "w") as out:
        # its own process group: stopping it also stops the engine's MPI workers
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + RemoteOpenAIServer.MAX_SERVER_START_WAIT_S
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"glm HTTP server exited; see {log}")
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200:
                    return proc, port
            except requests.RequestException:
                pass
            time.sleep(5)
        raise TimeoutError(f"glm HTTP server not healthy; see {log}")
    except BaseException:  # including a pytest timeout or Ctrl-C while waiting
        _stop_process_group(proc)
        raise


def _stop_process_group(proc):
    """SIGTERM the server's process group, SIGKILL whatever is left after a grace period."""
    import signal
    import subprocess

    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(60)
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):  # never hide the caller's exception
        proc.wait(30)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on case-03, the ds-engine: what the worker front end does.

Chat rendering with the DeepSeek-V4.1 tokenizer, and the reasoning and tool-call parsers, run in
the worker front end; the ds engine only sees token ids. The forms, fixtures and environment are
those of ``_test_openengine_ds_disagg_basic.py``.
"""

from __future__ import annotations

import json

import pytest

from . import _test_openengine_ds_disagg_basic as basic

MODEL_NAME = basic.MODEL_NAME
# The module's fixtures.
model_path = basic.model_path
ds_chain = basic.ds_chain

pytestmark = [pytest.mark.threadleak(enabled=False)]

MESSAGES = [{"role": "user", "content": "What is 17 times 23? Answer with the number."}]
THINK_PROMPT = "Think step by step about how many primes are below 50, then answer."
# Room for the default reasoning effort's chain of thought before the answer.
THINK_MAX_TOKENS = 8192
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


def _no_think_tags(text):
    return "<think>" not in (text or "") and "</think>" not in (text or "")


def test_chat_template_and_kwargs(ds_chain, model_path):
    """NT-124: the prompt is the DeepSeek-V4.1 renderer's, and its thinking switch reaches it."""
    from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer

    tokenizer = DeepseekV41Tokenizer.from_pretrained(model_path)
    c = ds_chain.client()
    prompt_tokens = {}
    for thinking in (False, True):
        out = c.chat.completions.create(
            model=MODEL_NAME,
            messages=MESSAGES,
            max_tokens=1,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"thinking": thinking}},
        )
        expected = tokenizer.apply_chat_template(MESSAGES, tokenize=True, thinking=thinking)
        assert out.usage.prompt_tokens == len(expected), (thinking, out.usage)
        prompt_tokens[thinking] = out.usage.prompt_tokens
    # Thinking mode adds the reasoning-effort preamble and the opening <think>.
    assert prompt_tokens[True] > prompt_tokens[False], prompt_tokens


def _chat(client, stream: bool, **request):
    """One chat request: reasoning, content, tool calls as ``(name, arguments)``, finish reason."""
    if not stream:
        choice = client.chat.completions.create(**request).choices[0]
        message = choice.message
        calls = [(c.function.name, c.function.arguments) for c in message.tool_calls or []]
        reasoning = getattr(message, "reasoning_content", None) or ""
        return reasoning, message.content or "", calls, choice.finish_reason
    reasoning, content, finish_reason = "", "", None
    calls: dict[int, list[str]] = {}
    for chunk in client.chat.completions.create(stream=True, **request):
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        reasoning += getattr(delta, "reasoning_content", None) or ""
        content += delta.content or ""
        for call in delta.tool_calls or []:
            name_and_arguments = calls.setdefault(call.index, ["", ""])
            if call.function is not None:
                name_and_arguments[0] += call.function.name or ""
                name_and_arguments[1] += call.function.arguments or ""
        finish_reason = choice.finish_reason or finish_reason
    return reasoning, content, [tuple(calls[i]) for i in sorted(calls)], finish_reason


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_reasoning_and_tools(ds_chain, stream):
    """NT-125: the reasoning parser splits the reasoning out; the tool parser returns the call."""
    c = ds_chain.client()
    reasoning, content, _, finish_reason = _chat(
        c,
        stream,
        model=MODEL_NAME,
        messages=[{"role": "user", "content": THINK_PROMPT}],
        max_tokens=THINK_MAX_TOKENS,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"thinking": True}},
    )
    assert finish_reason == "stop", (finish_reason, reasoning[-2000:], content)
    assert reasoning, "the reasoning parser split nothing out"
    assert "15" in content, content
    assert _no_think_tags(reasoning) and _no_think_tags(content)

    # Without thinking the answer carries no reasoning.
    reasoning, content, _, _ = _chat(
        c,
        stream,
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=64,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"thinking": False}},
    )
    assert not reasoning and "391" in content, (reasoning, content)

    _, content, calls, finish_reason = _chat(
        c,
        stream,
        model=MODEL_NAME,
        max_tokens=1024,
        temperature=0.0,
        tools=[WEATHER_TOOL],
        messages=[
            {"role": "user", "content": "What is the weather in Paris right now? Use the tool."}
        ],
    )
    assert finish_reason == "tool_calls" and calls, (finish_reason, content, calls)
    name, arguments = calls[0]
    assert name == "get_weather" and "paris" in json.loads(arguments)["city"].lower()
    assert _no_think_tags(content)

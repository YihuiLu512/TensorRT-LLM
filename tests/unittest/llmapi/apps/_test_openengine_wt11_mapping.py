# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Request fields that reach the engine, end to end.

Every test runs against the in-process server and, with
``TRTLLM_TEST_OPENENGINE_CHAIN`` set, against ``trtllm-serve`` -> OpenEngine ->
engine. Tests that count engine calls or expect a field the chain rejects run
on the chain only.
"""

import os
from collections import Counter
from typing import Iterator, Optional

import openai
import pytest
import requests
from transformers import AutoTokenizer

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)

# Sequences one Generate call returns on the chain: the engine caps it at
# max_batch_size x pipeline stages, so this server's limit is MAX_BATCH_SIZE.
MAX_BATCH_SIZE = 4
REASONING_END = "</think>"
GREEDY_PROMPT = "The capital of France is"
REPEAT_PROMPT = "Write the word 'apple' again and again: apple apple apple"
THINK_PROMPT = "List the prime numbers below 200, checking each number in turn."
THINKING_BUDGET = 16

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
chain_only = pytest.mark.skipif(
    not CHAIN, reason="checks the OpenEngine chain; set TRTLLM_TEST_OPENENGINE_CHAIN"
)


@pytest.fixture(scope="module")
def model_name() -> str:
    return "Qwen3/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server(model_name: str) -> Iterator[RemoteOpenAIServer]:
    args = [
        "--backend",
        "pytorch",
        "--kv_cache_free_gpu_memory_fraction",
        "0.2",
        "--max_batch_size",
        str(MAX_BATCH_SIZE),
        "--reasoning_parser",
        "qwen3",
    ]
    with RemoteOpenAIServer(get_model_path(model_name), args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer) -> openai.OpenAI:
    return server.get_client()


@pytest.fixture(scope="module")
def tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(get_model_path(model_name))


def _engine_calls(server: RemoteOpenAIServer) -> int:
    """Generate calls the front end has sent so far (chain only)."""
    response = requests.get(server.url_for("server_info"), timeout=30)
    response.raise_for_status()
    return response.json()["engine"]["generate_sent"]


def _frontend_log(server: RemoteOpenAIServer) -> str:
    path = server._chain_handle.frontend_log
    assert path, "the chain front end has no log file"
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _complete(client: openai.OpenAI, model: str, prompt: str, **kwargs) -> str:
    kwargs.setdefault("max_tokens", 32)
    response = client.completions.create(model=model, prompt=prompt, **kwargs)
    return response.choices[0].text


def _repeated_trigrams(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    counts = Counter(zip(ids, ids[1:], ids[2:]))
    return sum(c - 1 for c in counts.values() if c > 1)


def _token_count(tokenizer, text: Optional[str]) -> int:
    return len(tokenizer.encode((text or "").strip(), add_special_tokens=False))


def _budget_limit(tokenizer) -> int:
    return THINKING_BUDGET + len(tokenizer.encode(REASONING_END, add_special_tokens=False))


# ------------------------------------------------------------------- F-11, F-12, F-13


def test_sampling_knobs_reach_engine(client: openai.OpenAI, model_name: str):
    greedy = _complete(client, model_name, GREEDY_PROMPT, temperature=0.0)
    # Sampling at temperature 1 that keeps only the most likely token is greedy;
    # an engine that dropped the knob would sample.
    assert (
        _complete(client, model_name, GREEDY_PROMPT, temperature=1.0, extra_body={"top_k": 1})
        == greedy
    )
    assert (
        _complete(client, model_name, GREEDY_PROMPT, temperature=1.0, extra_body={"min_p": 1.0})
        == greedy
    )
    seeded = [
        _complete(client, model_name, GREEDY_PROMPT, temperature=1.0, seed=1234) for _ in range(2)
    ]
    assert seeded[0] == seeded[1]


def test_penalties(client: openai.OpenAI, model_name: str, tokenizer):
    plain = _complete(client, model_name, REPEAT_PROMPT, temperature=0.0, max_tokens=64)
    penalized = _complete(
        client,
        model_name,
        REPEAT_PROMPT,
        temperature=0.0,
        max_tokens=64,
        extra_body={"repetition_penalty": 2.0},
    )
    assert penalized != plain
    assert _repeated_trigrams(tokenizer, penalized) < _repeated_trigrams(tokenizer, plain)


# ---------------------------------------------------------------------------- F-19


@pytest.mark.parametrize(
    "n",
    [
        pytest.param(3, id="native"),
        # In-process, more sequences than --max_batch_size may never be scheduled.
        pytest.param(MAX_BATCH_SIZE + 1, id="split", marks=chain_only),
    ],
)
def test_n_choices(client: openai.OpenAI, model_name: str, n: int):
    response = client.completions.create(
        model=model_name, prompt=GREEDY_PROMPT, max_tokens=8, temperature=1.0, n=n, seed=7
    )
    assert sorted(c.index for c in response.choices) == list(range(n))
    chat = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Name a colour."}],
        max_completion_tokens=8,
        temperature=1.0,
        n=n,
    )
    assert sorted(c.index for c in chat.choices) == list(range(n))


@chain_only
@pytest.mark.parametrize("n, calls", [(3, 1), (MAX_BATCH_SIZE + 1, 2)], ids=["native", "split"])
def test_n_engine_call_count(
    server: RemoteOpenAIServer, client: openai.OpenAI, model_name: str, n: int, calls: int
):
    before = _engine_calls(server)
    response = client.completions.create(
        model=model_name, prompt=GREEDY_PROMPT, max_tokens=8, temperature=1.0, n=n
    )
    assert len(response.choices) == n
    assert _engine_calls(server) - before == calls
    assert f"requests with n > {MAX_BATCH_SIZE} are split" in _frontend_log(server)


# ------------------------------------------------------------------------ F-25+F-26


@chain_only
def test_unsupported_fields_rejected(client: openai.OpenAI, model_name: str):
    with pytest.raises(openai.BadRequestError, match="return_context_logits"):
        client.completions.create(
            model=model_name,
            prompt=GREEDY_PROMPT,
            max_tokens=4,
            extra_body={"return_context_logits": True},
        )


# ---------------------------------------------------------------------------- F-29


def test_thinking_token_budget_chat(client: openai.OpenAI, model_name: str, tokenizer):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": THINK_PROMPT}],
        max_completion_tokens=128,
        temperature=0.0,
        extra_body={"thinking_token_budget": THINKING_BUDGET},
    )
    reasoning = response.choices[0].message.reasoning_content
    assert reasoning, "no reasoning content"
    assert _token_count(tokenizer, reasoning) <= _budget_limit(tokenizer)


def test_thinking_token_budget_completions(client: openai.OpenAI, model_name: str, tokenizer):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": THINK_PROMPT}], tokenize=False, add_generation_prompt=True
    )
    text = _complete(
        client,
        model_name,
        prompt,
        temperature=0.0,
        max_tokens=128,
        extra_body={"thinking_token_budget": THINKING_BUDGET},
    )
    # The budget closes the reasoning well before max_tokens.
    assert REASONING_END in text, text
    reasoning = text.split(REASONING_END, 1)[0].replace("<think>", "")
    assert _token_count(tokenizer, reasoning) <= _budget_limit(tokenizer)


def test_thinking_token_budget_responses(client: openai.OpenAI, model_name: str, tokenizer):
    response = client.responses.create(
        model=model_name,
        input=THINK_PROMPT,
        max_output_tokens=128,
        temperature=0.0,
        extra_body={"thinking_token_budget": THINKING_BUDGET},
    )
    reasoning = [o for o in response.output if o.type == "reasoning"]
    assert reasoning, "no reasoning item"
    text = "".join(c.text for c in reasoning[0].content)
    assert _token_count(tokenizer, text) <= _budget_limit(tokenizer)


def test_thinking_token_budget_anthropic(server: RemoteOpenAIServer, model_name: str, tokenizer):
    # Anthropic's smallest budget is 1024 tokens.
    budget = 1024
    response = requests.post(
        f"{server.url_root}/v1/messages",
        json={
            "model": model_name,
            "max_tokens": budget + 128,
            "thinking": {"type": "enabled", "budget_tokens": budget},
            "messages": [{"role": "user", "content": THINK_PROMPT}],
        },
        headers={"content-type": "application/json", "anthropic-version": "2023-06-01"},
        timeout=300,
    )
    assert response.status_code == 200, response.text
    thinking = [b["thinking"] for b in response.json()["content"] if b["type"] == "thinking"]
    assert thinking, response.json()
    limit = budget + len(tokenizer.encode(REASONING_END, add_special_tokens=False))
    assert _token_count(tokenizer, thinking[0]) <= limit


# ---------------------------------------------------------- Responses top_logprobs


@pytest.mark.parametrize("top_logprobs", [None, 2], ids=["absent", "given"])
def test_responses_top_logprobs(client: openai.OpenAI, model_name: str, top_logprobs):
    """A Responses request is served with and without ``top_logprobs`` and reports it (0 if absent).

    A non-regression check: this server can return logprobs, so it serves the request either way.
    What the engine is asked for is checked on CPU, including against an engine that declares no
    output logprobs.
    """
    extra_body = {} if top_logprobs is None else {"top_logprobs": top_logprobs}
    response = client.responses.create(
        model=model_name,
        input=GREEDY_PROMPT,
        max_output_tokens=16,
        temperature=0.0,
        extra_body=extra_body,
    )
    assert response.output, response
    assert response.top_logprobs == (top_logprobs or 0)

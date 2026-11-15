# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A thinking token budget on a disaggregated request (F-29).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU, both with
``--reasoning_parser qwen3``), in-process or on the OpenEngine chain. The
context leg samples the first token, ``<think>``, and the generation leg the
rest of the reasoning, so the budget must reach the generation leg for the
reasoning to stop in time.

- NT-067 (F-29): through the orchestrator, a request with a thinking budget on
  each of the four endpoints that take one (chat, completions, responses and
  Anthropic messages) gets reasoning of at most the budget plus the tokens of
  the reasoning end marker.
"""

import pytest
import requests
from transformers import AutoTokenizer

from ._test_openengine_disagg_common import REQUEST_TIMEOUT_S, disagg_deployment, model_path

pytestmark = pytest.mark.threadleak(enabled=False)

REASONING_START = "<think>"
REASONING_END = "</think>"
THINK_PROMPT = "List the prime numbers below 200, checking each number in turn."
THINKING_BUDGET = 16
# Anthropic's smallest thinking budget.
ANTHROPIC_BUDGET = 1024
MAX_TOKENS = 128


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    with disagg_deployment(
        tmp_path_factory.mktemp("disagg_reasoning"), worker_args=["--reasoning_parser", "qwen3"]
    ) as deployment:
        yield deployment


@pytest.fixture(scope="module")
def tokenizer(deployment):
    return AutoTokenizer.from_pretrained(model_path(deployment.model))


def _token_count(tokenizer, text: str | None) -> int:
    return len(tokenizer.encode((text or "").strip(), add_special_tokens=False))


def _limit(tokenizer, budget: int) -> int:
    return budget + len(tokenizer.encode(REASONING_END, add_special_tokens=False))


def _reasoning(deployment, tokenizer, endpoint: str) -> str:
    """The reasoning of one request with a thinking budget, sent to ``endpoint`` of the orchestrator."""
    client = deployment.client()
    messages = [{"role": "user", "content": THINK_PROMPT}]
    budget = {"thinking_token_budget": THINKING_BUDGET}
    if endpoint == "chat":
        response = client.chat.completions.create(
            model=deployment.model,
            messages=messages,
            max_completion_tokens=MAX_TOKENS,
            temperature=0.0,
            extra_body=budget,
        )
        return response.choices[0].message.reasoning_content
    if endpoint == "completions":
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = (
            client.completions.create(
                model=deployment.model,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temperature=0.0,
                extra_body=budget,
            )
            .choices[0]
            .text
        )
        # The budget closes the reasoning well before max_tokens.
        assert REASONING_END in text, text
        return text.split(REASONING_END, 1)[0].replace(REASONING_START, "")
    if endpoint == "responses":
        response = client.responses.create(
            model=deployment.model,
            input=THINK_PROMPT,
            max_output_tokens=MAX_TOKENS,
            temperature=0.0,
            extra_body=budget,
        )
        items = [item for item in response.output if item.type == "reasoning"]
        assert items, response
        return "".join(part.text for part in items[0].content)
    response = requests.post(
        f"{deployment.url}/v1/messages",
        json={
            "model": deployment.model,
            "max_tokens": ANTHROPIC_BUDGET + MAX_TOKENS,
            "thinking": {"type": "enabled", "budget_tokens": ANTHROPIC_BUDGET},
            "messages": messages,
        },
        headers={"anthropic-version": "2023-06-01"},
        timeout=REQUEST_TIMEOUT_S,
    )
    assert response.status_code == 200, response.text
    blocks = [block for block in response.json()["content"] if block["type"] == "thinking"]
    assert blocks, response.json()
    return blocks[0]["thinking"]


@pytest.mark.parametrize("endpoint", ["chat", "completions", "responses", "anthropic"])
def test_thinking_token_budget(deployment, tokenizer, endpoint):
    reasoning = _reasoning(deployment, tokenizer, endpoint)
    assert reasoning, f"no reasoning from {endpoint}"
    budget = ANTHROPIC_BUDGET if endpoint == "anthropic" else THINKING_BUDGET
    assert _token_count(tokenizer, reasoning) <= _limit(tokenizer, budget), reasoning

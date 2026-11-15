# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input handling of a disaggregated deployment, through the orchestrator (F-07, F-09).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU), in-process or on the
OpenEngine chain.

- NT-056 (F-07): ``chat_template_kwargs`` reach the chat template the
  context worker renders, so the prompt token count is the one the template
  gives with them.
- NT-057 (F-09): ``truncate_prompt_tokens`` keeps the prompt's last tokens,
  and ``add_special_tokens`` is applied as the tokenizer applies it.
- NT-214 (F-07): a request's own ``chat_template`` is refused by the
  orchestrator unless it is allowed there, before any worker sees it.
"""

import pytest
import requests
from transformers import AutoTokenizer

from ._test_openengine_disagg_common import REQUEST_TIMEOUT_S, disagg_deployment, model_path

pytestmark = pytest.mark.threadleak(enabled=False)

QUESTION = [{"role": "user", "content": "What is 1 + 1?"}]
LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 8


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    with disagg_deployment(tmp_path_factory.mktemp("disagg_inputs")) as deployment:
        yield deployment


@pytest.fixture(scope="module")
def tokenizer(deployment):
    return AutoTokenizer.from_pretrained(model_path(deployment.model))


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_chat_template_kwargs(deployment, tokenizer, enable_thinking):
    text = tokenizer.apply_chat_template(
        QUESTION, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
    )
    expected = len(tokenizer.encode(text, add_special_tokens=False))
    response = deployment.client().chat.completions.create(
        model=deployment.model,
        messages=QUESTION,
        max_completion_tokens=4,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    assert response.usage.prompt_tokens == expected


def test_truncate_and_add_special_tokens(deployment, tokenizer):
    client = deployment.client()
    keep = 8
    truncated = client.completions.create(
        model=deployment.model,
        prompt=LONG_PROMPT,
        max_tokens=1,
        extra_body={"truncate_prompt_tokens": keep},
    )
    assert truncated.usage.prompt_tokens == keep

    for add_special_tokens in (True, False):
        expected = len(tokenizer.encode(LONG_PROMPT, add_special_tokens=add_special_tokens))
        response = client.completions.create(
            model=deployment.model,
            prompt=LONG_PROMPT,
            max_tokens=1,
            extra_body={"add_special_tokens": add_special_tokens},
        )
        assert response.usage.prompt_tokens == expected, add_special_tokens


def test_request_chat_template_gate(deployment):
    body = {
        "model": deployment.model,
        "messages": QUESTION,
        "max_tokens": 4,
        "chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}",
    }
    response = requests.post(
        f"{deployment.url}/v1/chat/completions", json=body, timeout=REQUEST_TIMEOUT_S
    )
    assert response.status_code == 400, response.text
    assert "chat_template" in response.text, response.text

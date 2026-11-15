# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input handling of a disaggregated deployment, through the orchestrator (F-07, F-09).

The deployments are ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU), in-process or on the
OpenEngine chain. On the chain the tests also read the two legs'
GenerateRequests (``_test_openengine_disagg_common.legs``).

- NT-214 (F-07): a request's own ``chat_template`` is refused by the
  orchestrator, before any worker sees it, unless the orchestrator and the
  workers allow it. Allowed, the template takes effect: the prompt is the one
  it renders, and on the chain the context leg's token ids are that prompt's
  and the generation leg gets the same ids.
- NT-056 (F-07): ``chat_template_kwargs`` reach the chat template the
  context worker renders, so the prompt token count is the one the template
  gives with them; on the chain the context leg's token ids are those of that
  prompt and the generation leg gets the same ids.
- NT-057 (F-09): ``truncate_prompt_tokens=k`` gives a prompt of ``k`` tokens,
  and with ``add_special_tokens`` true or false the prompt has as many tokens
  as the tokenizer gives with it. Qwen3's tokenizer adds no special token, so
  the two counts are the same.
"""

from pathlib import Path

import pytest
import requests
from transformers import AutoTokenizer

from ._test_openengine_disagg_common import (
    CHAIN,
    DUMP_ENV,
    MODEL_NAME,
    REQUEST_TIMEOUT_S,
    disagg_deployment,
    dump_names,
    legs,
    model_path,
)

pytestmark = pytest.mark.threadleak(enabled=False)

QUESTION = [{"role": "user", "content": "What is 1 + 1?"}]
LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 8
# Renders the messages' contents only, unlike the model's own template.
CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(model_path(MODEL_NAME))


@pytest.fixture(scope="module")
def dump_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("disagg_inputs_dumps")


@pytest.fixture(scope="module")
def deployment(tmp_path_factory, dump_dir):
    with disagg_deployment(
        tmp_path_factory.mktemp("disagg_inputs"), env={DUMP_ENV: str(dump_dir)}
    ) as deployment:
        yield deployment


def _token_ids(leg: dict) -> list[int]:
    return [int(token_id) for token_id in leg["token_ids"]["ids"]]


def _chat(deployment, dump_dir: Path, expected_ids: list[int], **fields) -> None:
    """A chat request whose prompt must be ``expected_ids``, on both legs on the chain."""
    before = dump_names(dump_dir)
    body = {"model": deployment.model, "messages": QUESTION, "max_tokens": 4, **fields}
    response = requests.post(
        f"{deployment.url}/v1/chat/completions", json=body, timeout=REQUEST_TIMEOUT_S
    )
    assert response.status_code == 200, response.text
    assert response.json()["usage"]["prompt_tokens"] == len(expected_ids), response.text
    if CHAIN:
        prefill, decode = legs(dump_dir, before)
        assert _token_ids(prefill) == expected_ids, prefill
        assert _token_ids(decode) == expected_ids, decode


# The allowed case runs first: its deployment stops before the module's starts.
@pytest.mark.parametrize("gate", ["open", "closed"])
def test_request_chat_template_gate(request, tmp_path, tokenizer, gate):
    if gate == "closed":
        deployment = request.getfixturevalue("deployment")
        body = {
            "model": deployment.model,
            "messages": QUESTION,
            "max_tokens": 4,
            "chat_template": CHAT_TEMPLATE,
        }
        response = requests.post(
            f"{deployment.url}/v1/chat/completions", json=body, timeout=REQUEST_TIMEOUT_S
        )
        assert response.status_code == 400, response.text
        assert "chat_template" in response.text, response.text
        return
    expected = tokenizer.encode(QUESTION[0]["content"], add_special_tokens=False)
    model_render = tokenizer.apply_chat_template(QUESTION, add_generation_prompt=True)
    assert len(model_render) != len(expected)
    dump_dir = tmp_path / "dumps"
    dump_dir.mkdir()
    with disagg_deployment(
        tmp_path,
        worker_config={"allow_request_chat_template": True},
        orchestrator_config={"allow_request_chat_template": True},
        env={DUMP_ENV: str(dump_dir)},
    ) as deployment:
        _chat(deployment, dump_dir, expected, chat_template=CHAT_TEMPLATE)


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_chat_template_kwargs(deployment, dump_dir, tokenizer, enable_thinking):
    text = tokenizer.apply_chat_template(
        QUESTION, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
    )
    expected = tokenizer.encode(text, add_special_tokens=False)
    _chat(
        deployment,
        dump_dir,
        expected,
        temperature=0.0,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )


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

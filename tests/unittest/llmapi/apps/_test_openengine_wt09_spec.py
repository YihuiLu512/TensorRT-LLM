# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Log probabilities and penalties under one-model speculative decoding, end to end.

One-model speculative decoding returns no log probabilities and, without
``enable_penalty``, applies no occurrence penalty: the in-process server
refuses such requests with 400 at admission, and through the OpenEngine
chain the front end refuses them with 400 from the engine's declaration,
before any Generate call. The ``plain`` server, without speculative decoding,
serves them.

Every test runs against the in-process server and, with
``TRTLLM_TEST_OPENENGINE_CHAIN`` set, against ``trtllm-serve`` -> OpenEngine ->
engine; engine calls are counted on the chain only. Needs one GPU.
"""

import os
from typing import Iterator

import openai
import pytest
import requests
import yaml

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
PROMPT = "1\n2\n3\n4\n"
LLAMA = "llama-3.1-model/Llama-3.1-8B-Instruct"
EAGLE3 = "EAGLE3-LLaMA3.1-Instruct-8B"
# name -> (model, extra LLM API options); the draft model's path is filled in by the fixture.
SERVERS = {
    "plain": (LLAMA, {}),
    "eagle3": (LLAMA, {"speculative_config": {"decoding_type": "Eagle3", "max_draft_len": 3}}),
}
# Every served completion runs to max_tokens, so its length is known.
RUN_TO_MAX = {"ignore_eos": True}
NO_EFFECT = {"repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
EFFECT = {"repetition_penalty": 1.2, "presence_penalty": 0.5, "frequency_penalty": 0.5}


@pytest.fixture(scope="module", params=sorted(SERVERS))
def name(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def server(name: str, tmp_path_factory) -> Iterator[RemoteOpenAIServer]:
    model, options = SERVERS[name]
    options = yaml.safe_load(yaml.safe_dump(options))
    if "speculative_config" in options:
        options["speculative_config"]["speculative_model"] = get_model_path(EAGLE3)
    options.update({"max_batch_size": 8, "kv_cache_config": {"free_gpu_memory_fraction": 0.5}})
    path = tmp_path_factory.mktemp(f"wt09-spec-{name}") / "options.yaml"
    path.write_text(yaml.safe_dump(options))
    args = ["--max_seq_len", "4096", "--extra_llm_api_options", str(path)]
    with RemoteOpenAIServer(get_model_path(model), args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer) -> openai.OpenAI:
    return server.get_client()


@pytest.fixture(scope="module")
def model_name(client: openai.OpenAI) -> str:
    return client.models.list().data[0].id


def _engine_calls(server: RemoteOpenAIServer) -> int | None:
    """Generate calls the front end has sent so far; None for the in-process server."""
    if not CHAIN:
        return None
    response = requests.get(server.url_for("server_info"), timeout=30)
    response.raise_for_status()
    return response.json()["engine"]["generate_sent"]


def _refused(server, call, *words) -> None:
    """The call fails with 400 naming ``words``, and through the chain sends no Generate call."""
    before = _engine_calls(server)
    with pytest.raises(openai.BadRequestError) as error:
        call()
    message = str(error.value)
    assert all(word in message for word in words), message
    assert _engine_calls(server) == before


def _served(server, call):
    before = _engine_calls(server)
    result = call()
    if before is not None:
        assert _engine_calls(server) == before + 1
    return result


def test_output_logprobs(server, client, model_name, name):
    def completion():
        return client.completions.create(
            model=model_name,
            prompt=PROMPT,
            max_tokens=8,
            temperature=0.0,
            logprobs=2,
            extra_body=RUN_TO_MAX,
        )

    def chat():
        return client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_tokens=8,
            temperature=0.0,
            logprobs=True,
            top_logprobs=2,
            extra_body=RUN_TO_MAX,
        )

    if name != "plain":
        _refused(server, completion, "logprobs")
        _refused(server, chat, "logprobs")
        return
    logprobs = _served(server, completion).choices[0].logprobs
    assert len(logprobs.tokens) == 8
    assert all(value <= 0.0 for value in logprobs.token_logprobs)
    assert all(len(top) == 2 for top in logprobs.top_logprobs)
    content = _served(server, chat).choices[0].logprobs.content
    assert len(content) == 8
    assert all(len(entry.top_logprobs) == 2 for entry in content)


@pytest.mark.parametrize("penalty", sorted(EFFECT))
def test_penalties(server, client, model_name, name, penalty):
    def complete(value):
        return client.completions.create(
            model=model_name,
            prompt=PROMPT,
            max_tokens=8,
            temperature=0.0,
            extra_body={penalty: value, **RUN_TO_MAX},
        )

    assert _served(server, lambda: complete(NO_EFFECT[penalty])).choices[0].text
    if name != "plain":
        _refused(server, lambda: complete(EFFECT[penalty]), penalty)
        return
    assert _served(server, lambda: complete(EFFECT[penalty])).choices[0].text

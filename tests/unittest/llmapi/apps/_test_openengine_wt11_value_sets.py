# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampling values an engine accepts, end to end (spec 06, CAP-*).

A request's explicit value is sent as given, also when it leaves sampling
unchanged, and is checked against the values the engine declares in
``trtllm.params``. The ``speculative`` server runs one-model speculative
decoding, under which the TRT-LLM engine accepts only the values of ``min_p``
and the penalties that leave sampling unchanged.

Every test runs against the in-process server and, with
``TRTLLM_TEST_OPENENGINE_CHAIN`` set, against ``trtllm-serve`` -> OpenEngine ->
engine. Tests that read the engine's declaration or count engine calls run on
the chain only.
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

MODEL = "Qwen3/Qwen3-0.6B"
PROMPT = "Write the word 'apple' again and again: apple apple apple"
# The explicit values that leave sampling unchanged.
NO_EFFECT = {
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
chain_only = pytest.mark.skipif(
    not CHAIN, reason="checks the OpenEngine chain; set TRTLLM_TEST_OPENENGINE_CHAIN"
)
ARMS = {
    "plain": {"max_batch_size": 8},
    # AUTO becomes one-model speculation, whose kernel takes at most 1024 slots.
    "speculative": {"max_batch_size": 8, "speculative_config": {"decoding_type": "AUTO"}},
}


@pytest.fixture(scope="module", params=sorted(ARMS))
def arm(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def server(arm: str, tmp_path_factory) -> Iterator[RemoteOpenAIServer]:
    options = tmp_path_factory.mktemp(f"value-sets-{arm}") / "options.yaml"
    options.write_text(yaml.safe_dump(ARMS[arm]))
    args = [
        "--backend",
        "pytorch",
        "--kv_cache_free_gpu_memory_fraction",
        "0.2",
        "--extra_llm_api_options",
        str(options),
    ]
    with RemoteOpenAIServer(get_model_path(MODEL), args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer) -> openai.OpenAI:
    return server.get_client()


def _engine_calls(server: RemoteOpenAIServer) -> int:
    """Generate calls the front end has sent so far (chain only)."""
    response = requests.get(server.url_for("server_info"), timeout=30)
    response.raise_for_status()
    return response.json()["engine"]["generate_sent"]


def _complete(client: openai.OpenAI, **extra_body) -> str:
    response = client.completions.create(
        model=MODEL, prompt=PROMPT, max_tokens=16, temperature=0.0, extra_body=extra_body
    )
    return response.choices[0].text


@pytest.mark.parametrize("name", sorted(NO_EFFECT))
def test_an_explicit_no_effect_value_is_served(client: openai.OpenAI, name: str):
    """The perf client's explicit ``repetition_penalty: 1.0`` and its kind are served on both arms."""
    assert _complete(client, **{name: NO_EFFECT[name]}) == _complete(client)


@chain_only
@pytest.mark.parametrize("name, value", [("repetition_penalty", 1.2), ("min_p", 0.1)])
def test_a_value_the_engine_does_not_accept_is_refused_before_the_engine(
    server: RemoteOpenAIServer, client: openai.OpenAI, arm: str, name: str, value: float
):
    before = _engine_calls(server)
    if arm == "plain":
        assert _complete(client, **{name: value}) is not None
        assert _engine_calls(server) == before + 1
        return
    with pytest.raises(openai.BadRequestError) as error:
        _complete(client, **{name: value})
    message = str(error.value)
    assert f"{name}={value}" in message
    assert f"it accepts only {NO_EFFECT[name]}" in message
    assert _engine_calls(server) == before

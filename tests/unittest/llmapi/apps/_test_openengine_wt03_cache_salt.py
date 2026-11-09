# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``cache_salt`` keeps requests from reusing each other's KV cache, end to end.

``_test_openai_cache_salt.py`` reads the KV-cache hit rate of the engine's
iteration stats, which an OpenEngine engine does not report. This test reads
the cached prompt tokens of each response instead, so it runs on the in-process
server and on the OpenEngine chain alike.
"""

import openai
import pytest

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

# Only full KV-cache blocks are reused: a long system prompt keeps the reused
# prefix from depending on the chat template's length.
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Keep responses brief. " * 12},
    {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
]


@pytest.fixture(scope="module")
def model_name() -> str:
    return "Qwen3/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server(model_name: str):
    args = [
        "--backend",
        "pytorch",
        "--kv_cache_free_gpu_memory_fraction",
        "0.2",
        "--max_seq_len",
        "2048",
    ]
    with RemoteOpenAIServer(
        get_model_path(model_name),
        args,
        extra_config={"kv_cache_config": {"enable_block_reuse": True}},
    ) as remote_server:
        yield remote_server


def _cached_tokens(client: openai.OpenAI, model_name: str, salt: str | None = None) -> int:
    response = client.chat.completions.create(
        model=model_name,
        messages=MESSAGES,
        max_completion_tokens=4,
        temperature=0.0,
        extra_body={} if salt is None else {"cache_salt": salt},
    )
    assert response.usage is not None and response.usage.prompt_tokens_details is not None
    return response.usage.prompt_tokens_details.cached_tokens


def test_cache_salt_prevents_reuse_by_cached_tokens(server: RemoteOpenAIServer, model_name: str):
    client = server.get_client()
    _cached_tokens(client, model_name)
    assert _cached_tokens(client, model_name) > 0, "the same request without a salt reuses"
    assert _cached_tokens(client, model_name, "user_123") == 0, "a new salt does not reuse"
    assert _cached_tokens(client, model_name, "user_123") > 0, "the same salt reuses"
    assert _cached_tokens(client, model_name, "user_456") == 0, "another salt does not reuse"

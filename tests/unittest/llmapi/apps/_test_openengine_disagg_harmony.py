# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gpt-oss's Harmony chat on a disaggregated deployment, context first (DG-67).

An orchestrator with the default router, and a context and a generation
worker on a GPU each, serve gpt-oss-20b. The context worker renders the chat
in Harmony and hands the prompt token ids on; the generation worker decodes
from them. The test runs on the in-process workers (the baseline form) and on
the OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN``,
``tests/integration/defs/openengine_chain``) alike.

- NT-098 (DG-67): a chat request, unstreamed and streamed, returns 200 with
  content or reasoning content, and both report the same prompt token count.

Needs two GPUs of Hopper or later and ``${LLM_MODELS_ROOT}/gpt_oss/gpt-oss-20b``
with ``${LLM_MODELS_ROOT}/datasets/tiktoken_vocab``.
"""

import json

import pytest
import requests

from ._test_openengine_disagg_per_role import (
    OWN_GPU_KV_CACHE,
    REQUEST_TIMEOUT_S,
    TWO_GPUS,
    deployment,
    model_dir,
    needs_gpus,
    tiktoken_env,
)

pytestmark = [pytest.mark.threadleak(enabled=False)]

MESSAGES = [{"role": "user", "content": "What is 2 + 3? Answer with the number only."}]
MAX_TOKENS = 256


def test_harmony_chat_context_first(tmp_path):
    """NT-098 (DG-67): unstreamed and streamed Harmony chats both answer, on the same prompt."""
    needs_gpus(2)
    with deployment(
        tmp_path,
        model_dir("gpt_oss/gpt-oss-20b"),
        config={"kv_cache_config": OWN_GPU_KV_CACHE},
        roles=TWO_GPUS,
        env=tiktoken_env(),
    ) as served:
        body = {
            "model": served.model,
            "messages": MESSAGES,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        }
        response = requests.post(
            served.url("v1", "chat", "completions"), json=body, timeout=REQUEST_TIMEOUT_S
        )
        assert response.status_code == 200, response.text
        answer = response.json()
        message = answer["choices"][0]["message"]
        assert message.get("content") or message.get("reasoning_content"), answer
        prompt_tokens = answer["usage"]["prompt_tokens"]
        assert prompt_tokens > 0, answer

        streamed = {**body, "stream": True, "stream_options": {"include_usage": True}}
        text, usage = [], None
        with requests.post(
            served.url("v1", "chat", "completions"),
            json=streamed,
            timeout=REQUEST_TIMEOUT_S,
            stream=True,
        ) as stream:
            assert stream.status_code == 200, stream.text
            for line in stream.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[len("data: ") :])
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    text.append(delta.get("content") or delta.get("reasoning_content") or "")
        assert "".join(text), "the streamed chat returned no content or reasoning content"
        assert usage is not None and usage["prompt_tokens"] == prompt_tokens, usage

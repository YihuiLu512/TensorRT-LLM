# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine features combined with a disaggregated deployment.

Each test starts an orchestrator and a context and a generation worker with
the engine settings it names, and sends its requests through the
orchestrator. The tests run on the in-process workers (the baseline form) and
on the OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN``,
``tests/integration/defs/openengine_chain``) alike.

- NT-210 (EG-34): gpt-oss-20b, whose sliding-window layers attend to the last
  128 tokens, answers a chat whose prompt is longer than the window: the
  final answer, not the reasoning, holds the number asked for.
- NT-211 (EG-36): the context worker has a KV-cache connector
  (``kv_connector_config``): the example connector of
  ``examples/llm-api/llm_kv_cache_connector.py``, which saves each computed
  block to a directory and loads it back for a prompt it has seen. A request
  answers and the connector saved its blocks; the same request answers the
  same again, and its blocks are loaded, not computed and saved again.
- NT-212 (EG-37): both workers run their executors with Ray
  (``orchestrator_type: ray``); a request answers correctly.
- NT-213 (EG-38): attention-DP workers on two GPUs each route by KV cache
  and count the requests whose KV cache is still in transfer
  (``kv_cache_routing_account_for_in_transfer``); the context worker is the
  one with such requests, those it is still sending. Requests sent at once
  all answer correctly.

Needs GPUs of Hopper or later (NT-211: one, NT-213: four, the others: two)
and the models under ``LLM_MODELS_ROOT`` that each test names.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from ._test_openengine_disagg_per_role import (
    OWN_GPU_KV_CACHE,
    REQUEST_TIMEOUT_S,
    TWO_GPUS,
    Role,
    attention_dp_roles,
    complete,
    deployment,
    model_dir,
    needs_gpus,
    tiktoken_env,
)

pytestmark = [pytest.mark.threadleak(enabled=False)]

QWEN = os.path.join("Qwen3", "Qwen3-0.6B")
EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "llm-api"
# gpt-oss-20b's sliding-window attention layers see this many tokens.
SLIDING_WINDOW = 128


def test_sliding_window_attention_two_legs(tmp_path):
    """NT-210 (EG-34): a prompt longer than the sliding window is answered across the handoff."""
    needs_gpus(2)
    filler = " ".join(f"Item {i} is a plain line of filler text." for i in range(60))
    question = "Ignore the items. What is 2 + 3? Answer with the number only."
    with deployment(
        tmp_path,
        model_dir("gpt_oss/gpt-oss-20b"),
        config={"kv_cache_config": OWN_GPU_KV_CACHE},
        roles=TWO_GPUS,
        env=tiktoken_env(),
    ) as served:
        response = requests.post(
            served.url("v1", "chat", "completions"),
            json={
                "model": served.model,
                "messages": [{"role": "user", "content": f"{filler}\n\n{question}"}],
                "max_tokens": 1024,
                "temperature": 0.0,
                "reasoning_effort": "low",
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        assert response.status_code == 200, response.text
        answer = response.json()
        assert answer["usage"]["prompt_tokens"] > SLIDING_WINDOW, answer["usage"]
        # The final answer, not the reasoning, which may mention the items' numbers.
        assert "5" in (answer["choices"][0]["message"].get("content") or ""), answer


def test_kv_cache_connector_with_disagg_prefill(tmp_path):
    """NT-211 (EG-36): a context worker with a KV-cache connector serves the context leg."""
    needs_gpus(1)
    saved = tmp_path / "connector_cache"
    connector = {
        "connector_module": "llm_kv_cache_connector",
        "connector_scheduler_class": "PersistentKvCacheConnectorLeader",
        "connector_worker_class": "PersistentKvCacheConnectorWorker",
    }
    python_path = os.pathsep.join(filter(None, (str(EXAMPLES), os.environ.get("PYTHONPATH"))))
    context = Role(
        config={"kv_connector_config": connector, "enable_chunked_prefill": False},
        env={"PYTHONPATH": python_path, "CONNECTOR_CACHE_FOLDER": str(saved)},
    )
    # Eight whole blocks of the default 32 tokens and a part block, which the connector does not save.
    prompt = [1000 + i for i in range(8 * 32 + 5)]
    with deployment(tmp_path, model_dir(QWEN), roles={"context": context}) as served:
        first = complete(served, prompt)["choices"][0]["text"]
        assert first
        blocks = {path: path.stat().st_mtime_ns for path in saved.glob("*.pt")}
        assert len(blocks) == 8, sorted(blocks)
        again = complete(served, prompt)["choices"][0]["text"]
        assert again == first
        # The second request loaded the blocks: the connector saves only the blocks it computes.
        assert {path: path.stat().st_mtime_ns for path in saved.glob("*.pt")} == blocks


def test_ray_orchestrator_engines_two_legs(tmp_path):
    """NT-212 (EG-37): workers whose executors run with Ray answer correctly."""
    needs_gpus(2)
    pytest.importorskip("ray", reason="orchestrator_type ray needs Ray installed")
    with deployment(
        tmp_path,
        model_dir(QWEN),
        config={"orchestrator_type": "ray", "kv_cache_config": OWN_GPU_KV_CACHE},
        roles=TWO_GPUS,
    ) as served:
        text = complete(served, "The capital of France is")["choices"][0]["text"]
        assert "Paris" in text, text


def test_kv_routing_accounts_for_in_transfer(tmp_path):
    """NT-213 (EG-38): KV-aware attention-DP routing that counts in-transfer requests."""
    needs_gpus(4)
    routing = {
        "enable_kv_cache_aware_routing": True,
        "kv_cache_routing_account_for_in_transfer": True,
    }
    with deployment(
        tmp_path,
        model_dir(QWEN),
        # The KV-aware attention-DP router is built only with block reuse.
        config={"kv_cache_config": {**OWN_GPU_KV_CACHE, "enable_block_reuse": True}},
        roles=attention_dp_roles(routing, routing),
    ) as served:
        with ThreadPoolExecutor(max_workers=8) as pool:
            texts = list(
                pool.map(
                    lambda _: complete(served, "The capital of France is")["choices"][0]["text"],
                    range(8),
                )
            )
        assert all("Paris" in text for text in texts), texts

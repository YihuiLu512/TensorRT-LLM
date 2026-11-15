# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV-cache transceiver of a disaggregated deployment: timeouts, cancels and C++ backends.

Each test starts an orchestrator and a context and a generation worker
serving Qwen3-0.6B. The tests run on the in-process workers (the baseline
form) and on the OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN``,
``tests/integration/defs/openengine_chain``) alike; on the chain they also
read the engines' ``Control.GetLoad``.

- NT-198 (EG-16b): the test sends a context leg to the context worker itself
  and no generation leg. The context worker's KV cache holds one such prompt,
  not two, so the request that follows through the orchestrator is only
  scheduled once ``kv_transfer_timeout_ms`` has freed the unclaimed blocks:
  the test checks it waited at least that long, less a slack. The
  expectations table says the test leaves one handoff unclaimed
  (``leg_expect: direct_ctx_leg``).
- NT-199 (EG-16c): a streamed request with a long prompt is left as soon as
  the orchestrator counts its context leg completed, about when the
  generation leg starts on the KV cache handed over; whether the transfer is
  still under way then is not controlled. The request asks for many tokens
  (``CANCEL_DECODE_TOKENS``); the test waits for the orchestrator to start
  the generation leg and end every leg, and checks that every generation leg
  ended as an error, as a leg cut short does. The workers then serve the next
  request. The test records the orchestrator's generation-leg counts over
  the request (``gen_leg_counts``), so that the runs of both forms can be set
  side by side.
- NT-202 (EG-22b, EG-22c, EG-22d, EP-04): both workers use the C++
  transceiver runtime with KV cache manager V1 and the backend of the
  parameter; a request answers correctly. The number of KV transfer rows the
  workers wrote under ``TRTLLM_KVCACHE_TIME_OUTPUT_PATH`` is recorded
  (``kv_transfer_rows``).

Needs one GPU (NT-199 and NT-202: two) of Hopper or later and
``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import glob
import json
import os
import time

import pytest
import requests

from tensorrt_llm.grpc.openengine_conventions import DISAGG_REQUEST_ID_MIN

from ._test_openengine_disagg_per_role import (
    CHAIN,
    OWN_GPU_KV_CACHE,
    REQUEST_TIMEOUT_S,
    TWO_GPUS,
    Deployment,
    Role,
    complete,
    deployment,
    load,
    model_dir,
    needs_gpus,
    wait_for,
)

pytestmark = [pytest.mark.threadleak(enabled=False)]

MODEL = os.path.join("Qwen3", "Qwen3-0.6B")
# Prompts as token ids, so that their KV-cache size is exact.
LONG_PROMPT_TOKENS = 640
TRANSFER_TIMEOUT_MS = 10000
# How much sooner than the transfer timeout the follow-up request may be served.
TIMEOUT_SLACK_S = 2
CANCEL_PROMPT_TOKENS = 16000
CANCEL_DECODE_TOKENS = 8192
SETTLE_S = 120


def _prompt(length: int, offset: int = 0) -> list[int]:
    return [1000 + (offset + i) % 5000 for i in range(length)]


def test_kv_transfer_timeout_frees_the_context_side(tmp_path):
    """NT-198 (EG-16b): an unclaimed handoff's blocks are freed by the transfer timeout."""
    needs_gpus(1)
    # 32 blocks of 32 tokens: one 640-token prompt (21 blocks with its first token) fits, two do not.
    kv_cache = {
        "max_tokens": 1024,
        "tokens_per_block": 32,
        "free_gpu_memory_fraction": 0.2,
        "enable_block_reuse": False,
    }
    context = {
        "cache_transceiver_config": {
            "backend": "DEFAULT",
            "kv_transfer_timeout_ms": TRANSFER_TIMEOUT_MS,
        }
    }
    with deployment(
        tmp_path,
        model_dir(MODEL),
        config={"kv_cache_config": kv_cache, "enable_chunked_prefill": False},
        roles={"context": Role(config=context)},
    ) as served:
        orphan = {
            "model": served.model,
            "prompt": _prompt(LONG_PROMPT_TOKENS),
            "max_tokens": 4,
            "disaggregated_params": {
                "request_type": "context_only",
                "disagg_request_id": DISAGG_REQUEST_ID_MIN + 1,
            },
        }
        response = requests.post(
            served.ctx.url_for("v1", "completions"), json=orphan, timeout=REQUEST_TIMEOUT_S
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0].get("disaggregated_params"), response.text
        left = time.monotonic()
        # Needs the blocks the unclaimed context leg holds until its transfer times out.
        answer = complete(served, _prompt(LONG_PROMPT_TOKENS, offset=7))
        waited = time.monotonic() - left
        assert answer["choices"][0]["text"], answer
        assert TRANSFER_TIMEOUT_MS / 1000 - TIMEOUT_SLACK_S <= waited < SETTLE_S, waited


def _orchestrator_counts(served: Deployment) -> dict[str, int]:
    """The orchestrator's ``<name>_total`` counters, from one scrape, by ``<name>``."""
    text = requests.get(served.url("prometheus", "metrics"), timeout=10).text
    counts: dict[str, int] = {}
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name.endswith("_total") and not line.startswith("#"):
            name = name.removesuffix("_total")
            counts[name] = counts.get(name, 0) + int(float(line.split()[-1]))
    return counts


def _legs_ended(served: Deployment, before: dict[str, int]) -> bool:
    """Whether the orchestrator has started a generation leg since ``before`` and ended every leg."""
    now = _orchestrator_counts(served)
    if now.get("gen_total_requests", 0) == before.get("gen_total_requests", 0):
        return False
    return all(
        now.get(f"{role}_completed_requests", 0) - before.get(f"{role}_completed_requests", 0)
        == now.get(f"{role}_total_requests", 0) - before.get(f"{role}_total_requests", 0)
        for role in ("ctx", "gen")
    )


def _engines_idle(served: Deployment) -> bool:
    return all(int(load(worker)["running_requests"]) == 0 for worker in served.workers.values())


def test_cancel_during_transfer_is_the_same_on_both_forms(tmp_path, record_property):
    """NT-199 (EG-16c): a client leaving as its KV cache moves ends the request; both serve on."""
    needs_gpus(2)
    with deployment(
        tmp_path,
        model_dir(MODEL),
        # The prompt is prefilled in one piece.
        config={"kv_cache_config": OWN_GPU_KV_CACHE, "max_num_tokens": CANCEL_PROMPT_TOKENS + 64},
        roles=TWO_GPUS,
    ) as served:
        names = ("gen_total_requests", "gen_completed_requests", "gen_error_requests")
        before = _orchestrator_counts(served)
        completed = before.get("ctx_completed_requests", 0)
        body = {
            "model": served.model,
            "prompt": _prompt(CANCEL_PROMPT_TOKENS),
            "max_tokens": CANCEL_DECODE_TOKENS,
            "ignore_eos": True,
            "stream": True,
        }
        with requests.post(
            served.url("v1", "completions"),
            json=body,
            headers={"Connection": "close"},
            timeout=REQUEST_TIMEOUT_S,
            stream=True,
        ) as response:
            assert response.status_code == 200, response.text
            wait_for(
                lambda: _orchestrator_counts(served).get("ctx_completed_requests", 0) > completed,
                "the context leg to complete",
                REQUEST_TIMEOUT_S,
                interval=0.05,
            )
        # Leaving the block closed the connection unread.
        wait_for(lambda: _legs_ended(served, before), "the orchestrator to end the legs", SETTLE_S)
        after = _orchestrator_counts(served)
        counts = {name: after.get(name, 0) - before.get(name, 0) for name in names}
        record_property("gen_leg_counts", json.dumps(counts, sort_keys=True))
        # The orchestrator counts every leg it ends as completed, and a leg cut short also as an error.
        assert counts["gen_error_requests"] == counts["gen_total_requests"], counts
        if CHAIN:
            wait_for(lambda: _engines_idle(served), "the engines to end the legs", SETTLE_S)
        assert complete(served, "The capital of France is")["choices"][0]["text"]


@pytest.mark.parametrize("backend", ["UCX", "MPI", "MOONCAKE"])
def test_cpp_runtime_backends(tmp_path, record_property, backend):
    """NT-202 (EG-22b to EG-22d, EP-04): the C++ transceiver runtime with each C++ backend."""
    needs_gpus(2)
    kv_times = tmp_path / "kv_times"
    config = {
        "cache_transceiver_config": {"backend": backend, "transceiver_runtime": "CPP"},
        "kv_cache_config": {
            "free_gpu_memory_fraction": 0.5,
            "enable_block_reuse": False,
            "use_kv_cache_manager_v2": False,
        },
    }
    with deployment(
        tmp_path,
        model_dir(MODEL),
        config=config,
        roles=TWO_GPUS,
        env={"TRTLLM_KVCACHE_TIME_OUTPUT_PATH": str(kv_times)},
    ) as served:
        text = complete(served, "The capital of France is", max_tokens=16)["choices"][0]["text"]
        assert "Paris" in text, text
    rows = 0
    for path in glob.glob(os.path.join(kv_times, "**", "*.csv"), recursive=True):
        with open(path, encoding="utf-8") as handle:
            rows += max(0, sum(1 for line in handle if line.strip()) - 1)
    record_property("kv_transfer_rows", rows)

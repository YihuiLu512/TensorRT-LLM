# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding on a disaggregated deployment, the context and generation workers on a GPU each.

Each test starts an orchestrator and a context and a generation worker with
the speculative configuration it names, and sends its requests through the
orchestrator. The tests run on the in-process workers (the baseline form) and
on the OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN``,
``tests/integration/defs/openengine_chain``) alike; a test of the same
behavior on both forms records its output text as the ``completion_text``
property, so that the two runs can be set side by side.

- NT-188 (EG-01e): DFlash on both workers.
- NT-189 (EG-01a): MTP on the generation worker only.
- NT-207 (EG-31): Eagle3 on both workers, one request at a time each; ten
  requests sent at once all complete, so each request frees its batch slot.
- NT-209 (EG-01g): Eagle3 with a dynamic draft tree on both workers.

Needs two GPUs of Hopper or later (NT-209: Blackwell, and 140000 MiB each) and
the models under ``LLM_MODELS_ROOT`` that each test names.
"""

import dataclasses
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from ._test_openengine_disagg_per_role import (
    OWN_GPU_KV_CACHE,
    TWO_GPUS,
    Deployment,
    Role,
    complete,
    deployment,
    model_dir,
    needs_gpus,
)

pytestmark = [pytest.mark.threadleak(enabled=False)]

PROMPT = "The capital of France is"
ANSWER = "Paris"


def _with_config(context: dict, generation: dict) -> dict[str, Role]:
    return {
        "context": dataclasses.replace(TWO_GPUS["context"], config=context),
        "generation": dataclasses.replace(TWO_GPUS["generation"], config=generation),
    }


def _complete(served: Deployment, prompt: str = PROMPT, max_tokens: int = 16, **extra) -> str:
    return complete(served, prompt, max_tokens, **extra)["choices"][0]["text"]


def _check_answer(served: Deployment, record_property) -> None:
    text = _complete(served)
    record_property("completion_text", text)
    assert ANSWER in text, text


def test_dflash_output_is_the_same_on_both_forms(tmp_path, record_property):
    """NT-188 (EG-01e): DFlash on both workers answers correctly; the text is recorded per form."""
    needs_gpus(2)
    spec = {
        "decoding_type": "DFlash",
        "max_draft_len": 4,
        "speculative_model": model_dir("Qwen3-8B-DFlash-b16"),
    }
    config = {"speculative_config": spec, "kv_cache_config": OWN_GPU_KV_CACHE, "max_batch_size": 8}
    with deployment(tmp_path, model_dir("Qwen3/Qwen3-8B"), config=config, roles=TWO_GPUS) as served:
        _check_answer(served, record_property)


def test_mtp_on_the_generation_side_only(tmp_path, record_property):
    """NT-189 (EG-01a): with MTP on the generation worker only, the answer is right."""
    needs_gpus(2)
    mtp = {"speculative_config": {"decoding_type": "MTP", "max_draft_len": 2}}
    with deployment(
        tmp_path,
        model_dir("DeepSeek-V3-Lite/bf16"),
        config={"kv_cache_config": OWN_GPU_KV_CACHE},
        roles=_with_config({}, mtp),
    ) as served:
        _check_answer(served, record_property)
        # A second request: the generation worker's draft state does not carry over.
        text = _complete(served)
        assert ANSWER in text, text


def test_eagle3_batch_slot_is_released_at_batch_size_one(tmp_path):
    """NT-207 (EG-31): one request at a time per worker; ten requests sent at once all complete."""
    needs_gpus(2)
    spec = {
        "decoding_type": "Eagle3",
        "max_draft_len": 3,
        "speculative_model": model_dir("Qwen3/qwen3_8b_eagle3"),
    }
    config = {
        "speculative_config": spec,
        "max_batch_size": 1,
        "kv_cache_config": {
            "max_tokens": 256,
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": 0.4,
        },
    }
    with deployment(tmp_path, model_dir("Qwen3/Qwen3-8B"), config=config, roles=TWO_GPUS) as served:

        def one(_):
            return _complete(
                served, "What is the capital of Germany?", max_tokens=25, ignore_eos=True
            )

        with ThreadPoolExecutor(max_workers=10) as pool:
            texts = list(pool.map(one, range(10)))
        assert len(texts) == 10 and all(texts), texts
        # Every slot is free again: one more request still completes.
        assert _complete(served)


def test_dynamic_tree_draft_tokens_cross_handoff(tmp_path, record_property):
    """NT-209 (EG-01g): Eagle3 with a dynamic draft tree on both workers answers correctly."""
    needs_gpus(2)
    if any(torch.cuda.get_device_capability(i) < (10, 0) for i in (0, 1)):
        pytest.skip("the dynamic draft tree needs Blackwell")
    if any(torch.cuda.get_device_properties(i).total_memory < 140_000 * 2**20 for i in (0, 1)):
        pytest.skip("Qwen3-30B-A3B with its draft model needs 140000 MiB on each GPU")
    spec = {
        "decoding_type": "Eagle3",
        "max_draft_len": 4,
        "speculative_model": model_dir("Qwen3/Qwen3-30B-eagle3"),
        "use_dynamic_tree": True,
        "dynamic_tree_max_topK": 4,
        "max_total_draft_tokens": 16,
    }
    config = {
        "speculative_config": spec,
        "max_batch_size": 32,
        "max_num_tokens": 4096,
        "kv_cache_config": OWN_GPU_KV_CACHE,
    }
    with deployment(
        tmp_path, model_dir("Qwen3/Qwen3-30B-A3B"), config=config, roles=TWO_GPUS
    ) as served:
        _check_answer(served, record_property)

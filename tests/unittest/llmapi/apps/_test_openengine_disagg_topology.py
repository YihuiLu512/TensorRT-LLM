# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-DP generation workers of a disaggregated deployment admit by KV load (EG-33).

An orchestrator and a context and a generation worker, each on two GPUs with
attention DP, serve Qwen3-0.6B. The test runs on the in-process workers (the
baseline form) and on the OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN``,
``tests/integration/defs/openengine_chain``) alike.

- NT-208 (EG-33): with ``attention_dp_config.enable_kv_balanced_admission``
  on the generation worker, requests sent at once all answer correctly. Which
  rank admits which request is not visible from outside the engine.

Needs four GPUs of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from ._test_openengine_disagg_per_role import (
    attention_dp_roles,
    complete,
    deployment,
    model_dir,
    needs_gpus,
)

pytestmark = [pytest.mark.threadleak(enabled=False)]

MODEL = os.path.join("Qwen3", "Qwen3-0.6B")
REQUESTS = 8


def test_kv_balanced_admission_on_the_generation_side(tmp_path):
    """NT-208 (EG-33): KV-balanced admission on the generation worker; every request answers."""
    needs_gpus(4)
    with deployment(
        tmp_path,
        model_dir(MODEL),
        config={"kv_cache_config": {"free_gpu_memory_fraction": 0.5, "enable_block_reuse": False}},
        roles=attention_dp_roles({"enable_kv_balanced_admission": True}),
    ) as served:
        with ThreadPoolExecutor(max_workers=REQUESTS) as pool:
            texts = list(
                pool.map(
                    lambda _: complete(served, "The capital of France is")["choices"][0]["text"],
                    range(REQUESTS),
                )
            )
        assert all("Paris" in text for text in texts), texts

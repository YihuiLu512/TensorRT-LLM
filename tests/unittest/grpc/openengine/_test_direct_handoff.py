# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two real TensorRT-LLM engines complete a context-first handoff over OpenEngine gRPC.

No ``trtllm-serve`` front end and no orchestrator: a ``context_only`` Generate on
one engine, and the ``KvSessionRef`` of its ``PrefillReady`` sent as ``kv.session``
of a Generate to the other (``openengine_chain.direct_replay``). Both engines run
Qwen3-0.6B on GPU 0, each with its own KV cache transceiver.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import os
import time

import pytest
import torch

__extra_import_path__ = ["../..", "~/tests/integration/defs"]
from openengine_chain import direct_replay as dr
from utils.llm_data import llm_models_root

pytestmark = [pytest.mark.threadleak(enabled=False)]

MODEL_SUBDIR = os.path.join("Qwen3", "Qwen3-0.6B")
MODEL_NAME = "Qwen3/Qwen3-0.6B"
PROMPT = "The capital of France is Paris. The capital of Germany is"
MAX_TOKENS = 16
ENGINE_READY_TIMEOUT_S = 1200
REQUEST_TIMEOUT_S = 120
KV_ROWS_WAIT_S = 30
# One GPU holds both engines, so each takes a small share of the free memory.
KV_CACHE_CONFIG = {"free_gpu_memory_fraction": 0.2}
TRANSCEIVER_CONFIG = {"backend": "DEFAULT"}
CTX_CONFIG = {
    "disable_overlap_scheduler": True,
    "kv_cache_config": KV_CACHE_CONFIG,
    "cache_transceiver_config": TRANSCEIVER_CONFIG,
}
GEN_CONFIG = {"kv_cache_config": KV_CACHE_CONFIG, "cache_transceiver_config": TRANSCEIVER_CONFIG}
# Both engines are on this host: loopback is the only network device UCX needs.
UCX_ENV = {"UCX_TLS": "^ib,gdr_copy", "UCX_NET_DEVICES": "lo"}


def _model_dir() -> str:
    root = llm_models_root()
    path = None if root is None else os.path.join(str(root), MODEL_SUBDIR)
    if path is None or not os.path.isdir(path):
        pytest.skip(f"{MODEL_SUBDIR} is not under LLM_MODELS_ROOT")
    return path


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (9, 0):
        pytest.skip("needs a GPU of Hopper or later")
    model = _model_dir()
    workdir = str(tmp_path_factory.mktemp("direct_handoff"))
    started = []
    kv_dirs = {}
    try:
        for name, config in (("ctx", CTX_CONFIG), ("gen", GEN_CONFIG)):
            kv_dirs[name] = os.path.join(workdir, f"kv-{name}")
            env = dict(os.environ, **UCX_ENV, TRTLLM_KVCACHE_TIME_OUTPUT_PATH=kv_dirs[name])
            env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            started.append(dr.start_engine(name, model, workdir, config=config, env=env))
        for engine in started:
            engine.wait_serving(ENGINE_READY_TIMEOUT_S)
        yield started[0], started[1], kv_dirs
    finally:
        for engine in started:
            engine.stop()


def _wait_kv_rows(directory: str, key: str) -> list[dict]:
    """The CSV rows of one kind, waiting briefly for the engine to write them."""
    deadline = time.monotonic() + KV_ROWS_WAIT_S
    while True:
        rows = dr.kv_transfer_rows(directory)[key]
        if rows or time.monotonic() > deadline:
            return rows
        time.sleep(1)


def test_two_engines_complete_a_handoff(engines):
    ctx_engine, gen_engine, kv_dirs = engines
    # The handoff goes first, so the generation engine has no cached prefix of the
    # prompt and must receive its KV.
    handoff = dr.run_handoff(
        ctx_engine.endpoint,
        gen_engine.endpoint,
        MODEL_NAME,
        PROMPT,
        max_tokens=MAX_TOKENS,
        timeout=REQUEST_TIMEOUT_S,
    )
    result = handoff.to_json()
    assert handoff.ctx.status == "OK", result
    assert handoff.ctx.terminal() == "prefill_ready", result
    assert handoff.gen is not None and handoff.gen.status == "OK", result
    assert handoff.gen.terminal() == "finished", result

    first = handoff.first_gen_tokens()
    assert len(first) == 1, result
    assert handoff.ctx.token_ids() == first, result
    gen_tokens = handoff.gen.token_ids()
    assert gen_tokens[:1] == first, result

    reference = dr.generate(
        gen_engine.endpoint,
        dr.build_request(MODEL_NAME, PROMPT, max_tokens=MAX_TOKENS),
        timeout=REQUEST_TIMEOUT_S,
    )
    assert reference.status == "OK", reference.to_json()
    assert gen_tokens == reference.token_ids(), (result, reference.to_json())

    ctx_usage, gen_usage = handoff.ctx.usage(), handoff.gen.usage()
    reference_usage = reference.usage()
    assert ctx_usage and gen_usage and reference_usage, result
    assert ctx_usage["prompt_tokens"] == reference_usage["prompt_tokens"], result
    assert gen_usage["prompt_tokens"] == reference_usage["prompt_tokens"], result
    assert gen_usage["completion_tokens"] == len(gen_tokens), result

    sent = _wait_kv_rows(kv_dirs["ctx"], "send")
    received = _wait_kv_rows(kv_dirs["gen"], "recv")
    assert any(int(row["transfer_size_bytes"]) > 0 for row in sent), sent
    assert any(int(row["transfer_size_bytes"]) > 0 for row in received), received

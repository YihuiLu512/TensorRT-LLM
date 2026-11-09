# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenEngine server's log-probability declaration under speculative decoding, on CPU.

One-model speculative decoding rejects requests for output or prompt log
probabilities at admission, so the server declares neither; the front
end then refuses such requests with 400 before they reach the engine.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("grpc")

from tensorrt_llm.grpc.openengine.bindings import model_pb2, server_pb2  # noqa: E402
from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer  # noqa: E402
from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer  # noqa: E402
from tensorrt_llm.grpc.openengine_conventions import model_capabilities_from_proto  # noqa: E402
from tensorrt_llm.llmapi.llm_args import (  # noqa: E402
    AutoDecodingConfig,
    Eagle3DecodingConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    SADecodingConfig,
)
from tensorrt_llm.sampling_params import MAX_TOP_LOGPROBS  # noqa: E402

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"


def _llm(speculative_config):
    """An engine stand-in with what GetModelInfo reads."""
    args = SimpleNamespace(
        max_seq_len=4096,
        max_batch_size=16,
        max_num_tokens=8192,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        enable_lora=False,
        guided_decoding_backend=None,
        max_beam_width=1,
        reasoning_parser=None,
        kv_cache_config=SimpleNamespace(tokens_per_block=32),
        speculative_config=speculative_config,
        _enable_routing_load=False,
    )
    executor = SimpleNamespace(
        is_shutdown=lambda: False,
        check_health=lambda: True,
        get_kv_cache_capacity=lambda: {},
        get_openengine_discovery=lambda: {},
        get_kv_cache_load=lambda: {},
    )
    return SimpleNamespace(
        args=args,
        llm_id="instance-1",
        tokenizer=object(),
        input_processor=SimpleNamespace(supports_token_ids_input=True),
        _executor=executor,
        _check_health=lambda: True,
    )


def _model_info(speculative_config):
    llm = _llm(speculative_config)
    control = OpenEngineControlServicer(llm, MODEL, OpenEngineInferenceServicer(llm, MODEL))
    return asyncio.run(control.GetModelInfo(model_pb2.GetModelInfoRequest(), SimpleNamespace()))


@pytest.mark.parametrize(
    ("make_config", "supported"),
    [
        (lambda: None, True),
        # NGram drafts outside the model and keeps TorchSampler.
        (lambda: NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2), True),
        (lambda: MTPDecodingConfig(max_draft_len=3), False),
        (lambda: Eagle3DecodingConfig(max_draft_len=3, speculative_model="/models/draft"), False),
        (lambda: SADecodingConfig(max_draft_len=3), False),
        # The executor replaces AUTO with a one-model SA config.
        (lambda: AutoDecodingConfig(), False),
    ],
    ids=["none", "ngram", "mtp", "eagle3", "sa", "auto"],
)
def test_logprobs_are_declared_only_without_one_model_speculation(make_config, supported):
    # Built in the test so that a config that fails validation fails only its own case.
    info = _model_info(make_config())

    for caps in (info.generation.output_logprobs, info.generation.prompt_logprobs):
        assert caps.supported is supported
        if supported:
            assert caps.max_top_n == MAX_TOP_LOGPROBS
            assert list(caps.candidate_selection_modes) == [
                model_pb2.CANDIDATE_TOKEN_SELECTION_MODE_TOP_N
            ]
        else:
            assert not caps.max_top_n and not caps.candidate_selection_modes

    # What the front end reads: None refuses every logprobs request with 400.
    capabilities = model_capabilities_from_proto(server_pb2.ServerInfo(), info)
    expected = MAX_TOP_LOGPROBS if supported else None
    assert capabilities.output_logprobs_max_top_n == expected
    assert capabilities.prompt_logprobs_max_top_n == expected

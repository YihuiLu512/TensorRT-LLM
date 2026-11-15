# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chain expectations for existing multi-GPU disaggregated tests.

On the trtllm chain the engine keeps its iteration stats, and a worker front
end's ``/metrics`` has no speculative decoding stats. The ``_SPEC_STATS``
entries skip accuracy tests that read those stats from the generation worker's
``/metrics`` after their accuracy evaluation (DV-06). The ``_DEEPSEEK_V41``
entries skip DeepSeek-V4.1 GSM8K tests, which read ``kv_cache_size`` from the
generation worker's perf record (DV-05) and the acceptance length
(``avg_decoded_tokens_per_iter``) of each GSM8K response (DV-06); a worker front
end reports neither.

``test_disaggregated_deepseek_v3_lite_fp8_attention_dp_gen_only`` runs the
generation-only benchmark mode with no context worker, so only the generation
side runs.
"""

from openengine_chain.conventions import CHAIN_TRTLLM
from openengine_chain.expectations import LEG_EXPECT, SINGLE_SIDE, SKIP, Expectation

_SERVING = "test_disaggregated_serving.py::"
_DISAGG = "test_disaggregated.py::"

_SPEC_STATS = (
    "reads the speculative decoding stats from the generation worker's /metrics, which a worker "
    "front end does not have: the engine keeps its iteration stats"
)
_DEEPSEEK_V41 = (
    "reads kv_cache_size from the generation worker's perf record and avg_decoded_tokens_per_iter "
    "from each GSM8K response, neither of which a worker front end reports"
)

EXPECTATIONS = [
    Expectation(
        "WT-57-fix2",
        _SERVING + "TestDeepSeekV3Lite::test_symmetric",
        SKIP,
        _SPEC_STATS,
        chains=(CHAIN_TRTLLM,),
    ),
    Expectation(
        "WT-57-fix2",
        _SERVING + "TestDeepSeekV3Lite::test_auto_dtype[mtp_nextn=2-*",
        SKIP,
        _SPEC_STATS,
        chains=(CHAIN_TRTLLM,),
    ),
    Expectation(
        "WT-57-fix2",
        _SERVING + "TestDeepSeekV4FlashDSpark::test_gsm8k_1p1d_dep4",
        SKIP,
        _SPEC_STATS,
        chains=(CHAIN_TRTLLM,),
    ),
    Expectation(
        "WT-57-fix2",
        _SERVING + "TestMiniMaxM3::test_nvfp4_eagle3_disagg_head_mismatched_bounce",
        SKIP,
        _SPEC_STATS,
        chains=(CHAIN_TRTLLM,),
    ),
    Expectation(
        "WT-57-fix2",
        _DISAGG + "test_disaggregated_deepseek_v41_gsm8k_ctxtp4_gentp4[*",
        SKIP,
        _DEEPSEEK_V41,
        chains=(CHAIN_TRTLLM,),
    ),
    Expectation(
        "WT-57-fix2",
        _DISAGG + "test_disaggregated_deepseek_v41_gsm8k_ctxdp4_gendp4[*",
        SKIP,
        _DEEPSEEK_V41,
        chains=(CHAIN_TRTLLM,),
    ),
    Expectation(
        "WT-57-fix2",
        _DISAGG + "test_disaggregated_deepseek_v3_lite_fp8_attention_dp_gen_only[*",
        LEG_EXPECT,
        "the generation-only benchmark mode with no context worker sends every request to the "
        "generation workers only",
        chains=(CHAIN_TRTLLM,),
        leg=SINGLE_SIDE,
    ),
]

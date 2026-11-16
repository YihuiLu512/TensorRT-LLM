# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chain expectations for the generation-only benchmark tests of ``test_disaggregated.py``.

With ``TRTLLM_DISAGG_BENCHMARK_GEN_ONLY`` set, the disaggregated server sends
every request to the generation workers without a context leg, so only the
generation side runs. ``test_disaggregated_benchmark_gen_only`` starts no
context worker; ``test_disaggregated_benchmark_gen_only_insufficient_kv`` and
``test_disaggregated_genbs1`` start one that takes no call.
"""

from openengine_chain.conventions import CHAIN_TRTLLM
from openengine_chain.expectations import LEG_EXPECT, SINGLE_SIDE, Expectation

_DISAGG = "test_disaggregated.py::"
_REASON = (
    "the generation-only benchmark mode sends every request to the generation workers "
    "without a context leg"
)

EXPECTATIONS = [
    Expectation(
        "WT-38-fix2",
        _DISAGG + name,
        LEG_EXPECT,
        _REASON + note,
        chains=(CHAIN_TRTLLM,),
        leg=SINGLE_SIDE,
    )
    for name, note in (
        ("test_disaggregated_benchmark_gen_only", "; its configuration has no context worker"),
        (
            "test_disaggregated_benchmark_gen_only_insufficient_kv",
            "; the context worker its configuration starts takes no call",
        ),
        (
            "test_disaggregated_genbs1",
            "; the context worker its configuration starts takes no call",
        ),
    )
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-51-fix2's expectation for NT-080: a request that ends on the context leg sends no generation leg."""

from openengine_chain.conventions import CHAIN_TRTLLM
from openengine_chain.expectations import GEN_ZERO_OK, LEG_EXPECT, Expectation

EXPECTATIONS = [
    Expectation(
        "WT-51-fix2",
        "_test_openengine_disagg_handoff.py::test_ctx_finished_on_eos_skips_the_gen_leg",
        LEG_EXPECT,
        "the first token ends each request on the context leg, so the generation role gets no call",
        chains=(CHAIN_TRTLLM,),
        leg=GEN_ZERO_OK,
    ),
]

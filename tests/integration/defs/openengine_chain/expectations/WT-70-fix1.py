# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gen_only_no_context perf sanity tests under the case-04 arms in the trtllm chain.

Such a test (``aggr-gen_only_no_context-<config>``) starts one generation
worker behind a disaggregated server that has no context worker, all in the
test's own pytest; the disaggregated server makes up each request's
generation-only leg. Only the generation side runs, so the test is
``leg_expect: single_side``. The ``ctx_only`` tests need no entry: their one
server has no worker role, and the proof of an aggregated server applies.
"""

from openengine_chain.conventions import CHAIN_TRTLLM
from openengine_chain.expectations import LEG_EXPECT, SINGLE_SIDE, Expectation

EXPECTATIONS = [
    Expectation(
        "WT-70-fix1",
        "test_perf_sanity.py::test_e2e[aggr-gen_only_no_context-*]",
        LEG_EXPECT,
        "only the generation side runs: the disaggregated server has no context worker",
        chains=(CHAIN_TRTLLM,),
        leg=SINGLE_SIDE,
    ),
]

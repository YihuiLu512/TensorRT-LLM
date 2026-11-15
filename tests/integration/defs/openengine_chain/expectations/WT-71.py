# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The disaggregated perf sanity tests under the case-03 arms in the ds chain.

As under the case-04 arms in the trtllm chain (``WT-70.py``), each worker of such a test
(``E3-chain``, ``E3-chain-lean``) runs in a pytest of its own and proves its own leg; the
benchmark's pytest judges what needs every worker's readings and the disaggregated
server's (``perf/openengine_arms.DisaggRunProbe``).
"""

from openengine_chain.conventions import CHAIN_DS
from openengine_chain.expectations import CROSS_NODE, LEG_EXPECT, Expectation

EXPECTATIONS = [
    Expectation(
        "WT-71",
        "test_perf_sanity.py::test_e2e[disagg-*]",
        LEG_EXPECT,
        "every worker of the test runs in a pytest of its own; the benchmark's pytest "
        "judges what needs both legs' readings",
        chains=(CHAIN_DS,),
        leg=CROSS_NODE,
    ),
]

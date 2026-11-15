# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The disaggregated perf sanity tests under a disaggregated comparison arm.

Each role of such a test (every context and generation worker, the
disaggregated server and the benchmark) runs the same test id in a pytest of
its own, on one node or several. The pytest of each worker in the chain
proves that worker's leg (an in-process worker has no leg to prove); the
benchmark's pytest judges what needs every worker's readings and the
disaggregated server's (``perf/openengine_arms.DisaggRunProbe``), so each
test is ``leg_expect: cross_node``. The disaggregated server's and the
benchmark's pytests start no chain server and prove nothing themselves.
"""

from openengine_chain.conventions import CHAIN_TRTLLM
from openengine_chain.expectations import CROSS_NODE, LEG_EXPECT, Expectation

EXPECTATIONS = [
    Expectation(
        "WT-70",
        "test_perf_sanity.py::test_e2e[disagg-*]",
        LEG_EXPECT,
        "every worker of the test runs in a pytest of its own; the benchmark's pytest "
        "judges what needs both legs' readings",
        chains=(CHAIN_TRTLLM,),
        leg=CROSS_NODE,
    ),
]

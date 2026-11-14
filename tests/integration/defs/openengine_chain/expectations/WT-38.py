# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The two-node disaggregated tests, whose context and generation workers run on two nodes.

Each node runs the same test in a pytest of its own and starts its own worker:
node 0 the context worker (with the disaggregated server and the client),
node 1 the generation worker. Each node proves its own worker; the runner
judges what needs both nodes' readings (``leg_expect: cross_node``).
"""

from openengine_chain.expectations import CROSS_NODE, LEG_EXPECT, Expectation

_REASON = "the context and generation workers run on two nodes, one in each node's pytest"

EXPECTATIONS = [
    # XD-301 to XD-309, through test_e2e.py::test_openai_disagg_multi_nodes_completion.
    Expectation(
        "WT-38",
        "_test_disagg_serving_multi_nodes.py::test_completion*",
        LEG_EXPECT,
        _REASON,
        leg=CROSS_NODE,
    ),
    # XD-310 and XD-311.
    Expectation(
        "WT-38",
        "_test_disagg_serving_multi_nodes_service_discovery.py::test_completion*",
        LEG_EXPECT,
        _REASON,
        leg=CROSS_NODE,
    ),
    # XD-314.
    Expectation(
        "WT-38",
        "test_disaggregated_multinode.py::TestDeepSeekV4ProDSparkMultinode::test_gsm8k_1p1d_dep4*",
        LEG_EXPECT,
        _REASON,
        leg=CROSS_NODE,
    ),
]

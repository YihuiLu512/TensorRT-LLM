# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-05's tests whose only request may be refused before it reaches the engine."""

from openengine_chain.expectations import NO_ENGINE_TRAFFIC, Expectation

EXPECTATIONS = [
    Expectation(
        "WT-05",
        "_test_openengine_wt05_scheduling.py::test_agent_hierarchy_reaches_the_engine_or_is_refused",
        NO_ENGINE_TRAFFIC,
        "an engine without trtllm.request.scheduling makes the frontend refuse it with 400",
    ),
    Expectation(
        "WT-05",
        "_test_openengine_wt05_scheduling.py::test_option_reaches_the_engine_or_is_refused*",
        NO_ENGINE_TRAFFIC,
        "an option the engine does not declare is refused with 400 before any Generate call",
    ),
]

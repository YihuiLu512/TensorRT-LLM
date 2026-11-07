# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-10's tests that send no Generate call by design."""

from openengine_chain.expectations import NO_ENGINE_TRAFFIC, Expectation

EXPECTATIONS = [
    Expectation(
        "WT-10",
        "_test_openengine_wt10_frontend.py::test_unsupported_endpoints*",
        NO_ENGINE_TRAFFIC,
        "every endpoint it calls answers 501 in the front end",
    ),
]

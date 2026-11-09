# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-09's serve tests that send no Generate call on the chain."""

from openengine_chain.expectations import NO_ENGINE_TRAFFIC, Expectation

EXPECTATIONS = [
    Expectation(
        "WT-09",
        "_test_openengine_wt09_spec.py::test_output_logprobs[eagle3]",
        NO_ENGINE_TRAFFIC,
        "it sends only logprobs requests, which the front end rejects under one-model"
        " speculative decoding",
    ),
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-03-fix1's table: an existing test that reads what only the engine measures."""

from openengine_chain.expectations import SKIP, Expectation

EXPECTATIONS = [
    Expectation(
        "WT-03-fix1",
        "_test_openai_cache_salt.py::test_cache_salt_prevents_reuse_chat*",
        SKIP,
        "reads the KV-cache hit rate of the engine's iteration stats, which the engine's "
        "load does not report; the chain's version is _test_openengine_wt03_cache_salt.py::"
        "test_cache_salt_prevents_reuse_by_cached_tokens",
    ),
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-03-fix1's table: tests that cancel Generate calls, and one that reads engine-only stats."""

from openengine_chain.expectations import CANCELS_CALLS, SKIP, Expectation

_DISCONNECTS = (
    "it disconnects some of its requests while they run, so the front end cancels their "
    "Generate calls; one cancelled before it reaches the engine is counted by the front end only"
)

EXPECTATIONS = [
    Expectation(
        "WT-03-fix1",
        "_test_openai_misc.py::test_request_cancellation[*",
        CANCELS_CALLS,
        "its client times out its requests, so the front end cancels their Generate calls; "
        "one cancelled before it reaches the engine is counted by the front end only",
    ),
    Expectation(
        "WT-03-fix1",
        "_test_openai_kv_cache_contamination.py::test_no_kv_cache_contamination_under_cancel*",
        CANCELS_CALLS,
        _DISCONNECTS,
    ),
    Expectation(
        "WT-03-fix1",
        "_test_openai_kv_cache_contamination.py::test_no_kv_cache_contamination_number_set_probe*",
        CANCELS_CALLS,
        _DISCONNECTS,
    ),
    Expectation(
        "WT-03-fix1",
        "_test_openai_cache_salt.py::test_cache_salt_prevents_reuse_chat*",
        SKIP,
        "reads the KV-cache hit rate of the engine's iteration stats, which the engine's "
        "load does not report; the chain's version is _test_openengine_wt03_cache_salt.py::"
        "test_cache_salt_prevents_reuse_by_cached_tokens",
    ),
]

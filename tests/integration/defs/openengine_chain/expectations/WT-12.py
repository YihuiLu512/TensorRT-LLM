# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of response assembly that do not apply to the chain.

The ``*_cached_tokens_stream`` tests also ask for ``continuous_usage_stats``
and expect every chunk's usage to carry the cached prompt tokens. An OpenEngine
engine sends usage only with the last response of a stream (OE-STR-4), so the
chunks before it cannot report them.
"""

from openengine_chain.expectations import SKIP, Expectation

_PER_CHUNK_CACHED_TOKENS = (
    "every streamed chunk must report cached_tokens; OpenEngine sends usage "
    "only with the last response (OE-STR-4)"
)

EXPECTATIONS = [
    Expectation("WT-12", match, SKIP, _PER_CHUNK_CACHED_TOKENS)
    for match in (
        "_test_openai_completions.py::test_completion_cached_tokens_stream*",
        "_test_openai_chat.py::test_chat_cached_tokens_stream*",
    )
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-11's table: serve tests whose server asks for beam search.

The chain has no beam search, so a front end started with ``--max_beam_width``
above 1 refuses to start, and every test that uses such a server fails at startup.
"""

from openengine_chain.expectations import STARTUP_REJECTED, Expectation

_REASON = "the server is started with --max_beam_width > 1; OpenEngine has no beam search"
_LOG = "max_beam_width > 1 is not supported"

_BEAM_SERVERS = (
    "_test_openai_completions.py::test_batch_completions_beam_search*",
    "_test_openai_chat.py::test_multiple_responses_with_beam_search*",
    "_test_openai_reasoning.py::*with_beam_search*",
    "_test_openai_misc.py::*with_beam_search*",
)

EXPECTATIONS = [
    Expectation("WT-11", match, STARTUP_REJECTED, _REASON, log_contains=_LOG)
    for match in _BEAM_SERVERS
]

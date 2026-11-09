# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-11's table: beam-search servers, and requests the front end refuses.

The chain has no beam search: the harness passes ``--max_beam_width`` to the
front end as well as to the engine, so a front end started with a beam width
above 1 refuses to start, and every test that uses such a server fails at startup.
"""

from openengine_chain.expectations import NO_ENGINE_TRAFFIC, STARTUP_REJECTED, Expectation

_BEAM_REASON = "the server is started with --max_beam_width > 1; OpenEngine has no beam search"
_BEAM_LOG = "max_beam_width > 1 is not supported"

_BEAM_SERVERS = (
    "_test_openai_completions.py::test_batch_completions_beam_search*",
    "_test_openai_chat.py::test_multiple_responses_with_beam_search*",
    "_test_openai_reasoning.py::*with_beam_search*",
    "_test_openai_misc.py::*with_beam_search*",
)

EXPECTATIONS = [
    Expectation("WT-11", match, STARTUP_REJECTED, _BEAM_REASON, log_contains=_BEAM_LOG)
    for match in _BEAM_SERVERS
] + [
    Expectation(
        "WT-11",
        "_test_openengine_wt11_mapping.py::test_unsupported_fields_rejected*",
        NO_ENGINE_TRAFFIC,
        "it sends only a request the front end rejects",
    ),
    Expectation(
        "WT-11",
        "_test_openengine_wt11_value_sets.py::"
        "test_a_value_the_engine_does_not_accept_is_refused_before_the_engine[speculative-*]",
        NO_ENGINE_TRAFFIC,
        "the speculative engine does not accept the value, so the front end refuses the request",
    ),
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-11's table: serve tests that send requests the request mapping refuses.

The harness passes ``--max_beam_width`` to the engine only, so a server that
asks for beam search starts with an engine fixed at that beam width.
OpenEngine has no beam search: a request with ``use_beam_search`` gets HTTP 400
from the front end and never reaches the engine. A request without beam search
sent to such a server reaches the engine, which refuses a beam width other than
its own (``test_health_generate`` on the ``with_beam_search`` server of
``_test_openai_misc.py``); no entry here excuses that failure.
"""

from openengine_chain.expectations import HTTP_STATUS, NO_ENGINE_TRAFFIC, Expectation

_BEAM_REASON = "use_beam_search=True: OpenEngine has no beam search"

_BEAM_REQUESTS = (
    "_test_openai_completions.py::test_batch_completions_beam_search*",
    "_test_openai_chat.py::test_multiple_responses_with_beam_search*",
    "_test_openai_reasoning.py::test_reasoning_parser[*with_beam_search-True]",
    "_test_openai_misc.py::test_request_cancellation[*with_beam_search*-True]",
)

EXPECTATIONS = [
    Expectation("WT-11", match, HTTP_STATUS, _BEAM_REASON, status=400) for match in _BEAM_REQUESTS
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

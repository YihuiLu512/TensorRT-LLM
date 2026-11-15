# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-55's generation-first tests on the chain.

NT-091 in ``_test_openengine_disagg_gen_first.py`` sends its requests through
the disaggregated server, so both legs carry traffic and no ``leg_expect``
entry applies; it cancels one of its requests, so the counts are judged as for
a test that cancels calls.
"""

from openengine_chain.conventions import CHAIN_TRTLLM
from openengine_chain.expectations import CANCELS_CALLS, Expectation

EXPECTATIONS = [
    Expectation(
        "WT-55",
        "_test_openengine_disagg_gen_first.py::test_gen_first_cancel_closes_both_legs",
        CANCELS_CALLS,
        "its client leaves one of its two requests mid-stream, so the generation front end "
        "cancels that request's Generate call",
        chains=(CHAIN_TRTLLM,),
    ),
]

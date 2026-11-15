# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The chain proof of the multi-GPU disaggregated tests that send a leg themselves or leave early.

The other tests of these files send every request through the
disaggregated server, so both legs carry traffic and need no entry.
"""

from openengine_chain.conventions import CHAIN_TRTLLM
from openengine_chain.expectations import CANCELS_CALLS, DIRECT_CTX_LEG, LEG_EXPECT, Expectation

_TRANSCEIVER = "_test_openengine_disagg_transceiver.py::"

EXPECTATIONS = [
    Expectation(
        "WT-57-fix1",
        _TRANSCEIVER + "test_kv_transfer_timeout_frees_the_context_side",
        LEG_EXPECT,
        "it sends one context leg to the context worker itself and no generation leg, then "
        "one request through the disaggregated server",
        chains=(CHAIN_TRTLLM,),
        leg=DIRECT_CTX_LEG,
        orphans=1,
    ),
    Expectation(
        "WT-57-fix1",
        _TRANSCEIVER + "test_cancel_during_transfer_is_the_same_on_both_forms",
        CANCELS_CALLS,
        "its client leaves once the context leg is done, and the worker frontend cancels the "
        "generation leg's call",
        chains=(CHAIN_TRTLLM,),
    ),
]

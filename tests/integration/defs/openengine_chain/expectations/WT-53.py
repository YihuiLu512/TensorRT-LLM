# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of KV-cache-aware routing, conditional disaggregation and the worker readings.

``test_workers.py`` sends the legs to the worker frontends itself: the
KV-cache-aware router tests send both legs (``direct_two_leg``), the eviction
test sends aggregated requests to the generation worker only
(``direct_worker``), and the conditional test sends the legs itself or, when
most of the prompt is the previous round's, the whole request to the
generation worker (``ctx_zero_ok``). With ``conditional_disagg_config`` the
disaggregated server may prefill a request on the generation worker
(``ctx_zero_ok``). The tests through the disaggregated server that send
both legs, such as ``test_disaggregated_cache_aware_balance*``, need no
entry.

``test_workers_kv_cache_events`` hashes the request with the default
algorithm (``v1_block_key``) and looks for the hashes among the events'
blocks; on the trtllm chain each event names the algorithm the engine declares in
its ``GetServerInfo`` (``trtllm_kv_event_hash_algo``), which is another one,
so the test cannot match a block. A router reads the algorithm from the events:
``_test_openengine_disagg_kv_aware.py::test_kv_cache_events_direct`` checks
the chain's events that way.

NT-055 reads each worker frontend's ``/server_info``, and NT-186 also its
engine's ``GetServerInfo``; neither is a generation request.
"""

from openengine_chain.conventions import CHAIN_TRTLLM
from openengine_chain.expectations import (
    CTX_ZERO_OK,
    DIRECT_TWO_LEG,
    DIRECT_WORKER,
    LEG_EXPECT,
    NO_ENGINE_TRAFFIC,
    SKIP,
    Expectation,
)

_CONDITIONAL = "the disaggregated server may prefill a request on the generation worker"
_BOTH_LEGS = "the test sends both legs to the worker frontends itself"

EXPECTATIONS = [
    Expectation(
        "WT-53",
        "test_workers.py::test_workers_kv_cache_events",
        SKIP,
        "hashes the request with the default algorithm (v1_block_key) while the engine's "
        "events name another one; "
        "_test_openengine_disagg_kv_aware.py::test_kv_cache_events_direct reads them as a "
        "router does",
        chains=(CHAIN_TRTLLM,),
    ),
    Expectation(
        "WT-53",
        "test_workers.py::test_workers_kv_cache_events",
        LEG_EXPECT,
        _BOTH_LEGS,
        leg=DIRECT_TWO_LEG,
    ),
    Expectation(
        "WT-53",
        "test_workers.py::test_workers_kv_cache_aware_router",
        LEG_EXPECT,
        _BOTH_LEGS,
        leg=DIRECT_TWO_LEG,
    ),
    # Hopper only.
    Expectation(
        "WT-53",
        "test_workers.py::test_workers_kv_cache_aware_router_deepseek_v3_lite_bf16*",
        LEG_EXPECT,
        _BOTH_LEGS,
        leg=DIRECT_TWO_LEG,
    ),
    Expectation(
        "WT-53",
        "test_workers.py::test_workers_kv_cache_aware_router_eviction",
        LEG_EXPECT,
        "the test sends aggregated requests to the generation worker frontend itself",
        leg=DIRECT_WORKER,
    ),
    Expectation(
        "WT-53",
        "test_disaggregated.py::test_disaggregated_conditional",
        LEG_EXPECT,
        _CONDITIONAL,
        leg=CTX_ZERO_OK,
    ),
    Expectation(
        "WT-53",
        "test_workers.py::test_workers_conditional_disaggregation*",
        LEG_EXPECT,
        "the test sends a request whose prompt is mostly the previous round's to the "
        "generation worker frontend alone",
        leg=CTX_ZERO_OK,
    ),
    # Hopper only.
    Expectation(
        "WT-53",
        "test_disaggregated.py::test_disaggregated_deepseek_v3_lite_bf16_conditional*",
        LEG_EXPECT,
        _CONDITIONAL,
        leg=CTX_ZERO_OK,
    ),
    Expectation(
        "WT-53",
        "_test_openengine_disagg_kv_aware.py::test_conditional_disagg_local_prefill_skips_the_ctx_leg",
        LEG_EXPECT,
        _CONDITIONAL,
        leg=CTX_ZERO_OK,
    ),
    Expectation(
        "WT-53",
        "_test_openengine_disagg_server_info.py::*",
        NO_ENGINE_TRAFFIC,
        "reads /server_info, and GetServerInfo, only",
    ),
]

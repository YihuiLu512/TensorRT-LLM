# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-54's perf metrics tests: what only the engine measures, and tests that send no Generate call.

On the chain a worker frontend measures a request from the engine's events.
The KV-cache transfer times and size and the per-step metrics are measured
inside the engine, so they are left out of the frontends' records and
headers, not reported as 0. The TensorRT-LLM OpenEngine server writes the KV-cache
transfer time and size, with its other durations of each leg, to a file of
its own, and ``scripts/poc3/wt54_join_perf.py`` joins them with the
frontends' records offline (DV-05); the per-step metrics are not recorded.
NT-203 in ``_test_openengine_disagg_perf_metrics.py`` makes, on the chain,
the two existing tests' checks that the chain can meet; each reason below
says what it leaves out.

No test of that file needs a ``leg_expect`` entry: NT-203 sends its requests
through the disaggregated server, so both legs carry traffic, and NT-096 asks
the worker frontends for ``/steady_clock_offset`` only, which is not a
generation request and is not counted in ``http_received``:
``no_engine_traffic`` lifts only the need for traffic, and a ``leg_expect``
such as ``direct_worker`` would also drop CH-4 and CH-6.
"""

from openengine_chain.conventions import CHAIN_TRTLLM
from openengine_chain.expectations import NO_ENGINE_TRAFFIC, SKIP, Expectation

_CHAIN_VERSION = (
    "the chain's version is _test_openengine_disagg_perf_metrics.py::"
    "test_engine_side_transfer_stats_join_by_disagg_request_id"
)

EXPECTATIONS = [
    Expectation(
        "WT-54",
        "test_disaggregated.py::test_disaggregated_perf_metrics",
        SKIP,
        "asserts the generation leg's kv_cache_transfer_start and kv_cache_transfer_end in "
        "the generation worker's perf record, which only the engine measures; "
        + _CHAIN_VERSION
        + " (with the default transceiver: this test's V2 KV-cache manager with the Python "
        "transceiver is not run on the chain)",
        chains=(CHAIN_TRTLLM,),
    ),
    Expectation(
        "WT-54",
        "test_disagg_serving_perf_metrics.py::test_return_perf_metrics_and_jsonl_dump",
        SKIP,
        "asserts the per-step and context-chunk metrics headers, which only the engine "
        "measures; " + _CHAIN_VERSION,
        chains=(CHAIN_TRTLLM,),
    ),
    Expectation(
        "WT-54",
        "_test_openengine_disagg_perf_metrics.py::test_clock_offset_reaches_each_worker_frontend",
        NO_ENGINE_TRAFFIC,
        "asks each worker frontend for /steady_clock_offset only",
        chains=(CHAIN_TRTLLM,),
    ),
    Expectation(
        "WT-54",
        "_test_openengine_disagg_perf_metrics.py::test_engine_leg_records_of_a_direct_handoff",
        NO_ENGINE_TRAFFIC,
        "starts its own two engines and calls them over OpenEngine gRPC, without a frontend",
        chains=(CHAIN_TRTLLM,),
    ),
]

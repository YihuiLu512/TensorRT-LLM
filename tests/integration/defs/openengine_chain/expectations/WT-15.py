# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-15's metrics and tracing tests that assert what only the engine measures.

On the chain the frontend measures a request from the engine's events; the
time in queue, the first scheduled time, iteration counts, per-request
KV-cache reuse and the iteration stats beyond the engine's load are measured
inside the engine and left out rather than reported as 0. The chain starts
the engine without load metrics, so its load reports no KV-cache blocks.
``_test_openengine_wt15_metrics.py`` checks the same features with the
chain's expectations.
"""

from openengine_chain.expectations import SKIP, Expectation

EXPECTATIONS = [
    Expectation(
        "WT-15",
        "test_tracing.py::test_tracing*",
        SKIP,
        "asserts gen_ai.latency.time_in_queue > 0, which only the engine measures; "
        "the chain's version is _test_openengine_wt15_metrics.py::test_tracing_over_chain",
    ),
    Expectation(
        "WT-15",
        "_test_openai_perf_metrics.py::test_return_perf_metrics_and_jsonl_dump*",
        SKIP,
        "asserts the per-step metrics headers and the first and last iteration, the first "
        "scheduled time and the KV-cache block counts of the perf record, which only the "
        "engine measures; the chain's version is "
        "_test_openengine_wt15_metrics.py::test_perf_metrics_over_chain",
    ),
    Expectation(
        "WT-15",
        "_test_openai_perf_metrics.py::test_responses_return_perf_metrics_and_jsonl_dump*",
        SKIP,
        "asserts the per-step metrics headers and the first and last iteration and the "
        "first scheduled time of the perf record, which only the engine measures; the "
        "chain's version is _test_openengine_wt15_metrics.py::test_perf_metrics_over_chain",
    ),
    Expectation(
        "WT-15",
        "_test_openai_prometheus.py::test_metrics_endpoint*",
        SKIP,
        "asserts the request queue time and a KV-cache hit rate above 0, which only the "
        "engine measures; the chain's version is _test_openengine_wt15_metrics.py::"
        "test_prometheus_over_chain",
    ),
    Expectation(
        "WT-15",
        "_test_openai_prometheus.py::test_new_metrics_exposed*",
        SKIP,
        "waits for the prefill batch histograms of the engine's iteration stats, which "
        "the engine's load does not report",
    ),
    Expectation(
        "WT-15",
        "_test_openai_prometheus.py::test_kv_cache_metrics_available_before_first_request*",
        SKIP,
        "waits for trtllm_kv_cache_utilization, which comes from the KV-cache blocks of "
        "the engine's load; the chain starts the engine without "
        "--openengine-enable-load-metrics, so its load reports none",
    ),
]

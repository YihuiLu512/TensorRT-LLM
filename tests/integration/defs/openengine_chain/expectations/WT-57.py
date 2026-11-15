# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-57's new disaggregated tests whose chain proof is not the default.

Each of WT-57's tests in the ``_test_openengine_disagg_*`` files starts or shares
a deployment: an orchestrator and workers with ``--server_role``. By default the
chain proof expects every request to go through the orchestrator to both
legs. A test listed here sends no Generate call, sends to the worker
frontends itself, kills a worker, may end on the context leg, expects a worker
frontend to refuse to start, or cannot run on the chain. The migrated tests
of WT-57 need no entry of their own: the ones that send no generation request
are listed by WT-03 already.
"""

from openengine_chain.conventions import CHAIN_TRTLLM
from openengine_chain.expectations import (
    DIRECT_TWO_LEG,
    DIRECT_WORKER,
    GEN_ZERO_OK,
    KILLED,
    LEG_EXPECT,
    NO_ENGINE_TRAFFIC,
    SKIP,
    STARTUP_REJECTED,
    Expectation,
)

_ENDPOINTS = "_test_openengine_disagg_endpoints.py::"
_UNSUPPORTED_OPS = "_test_openengine_disagg_unsupported_ops.py::"
_CHAIN = (CHAIN_TRTLLM,)

EXPECTATIONS = [
    *(
        Expectation("WT-57", match, NO_ENGINE_TRAFFIC, reason, chains=_CHAIN)
        for match, reason in (
            *(
                (
                    f"{_ENDPOINTS}test_orchestrator_limits_same_as_baseline[{case}]",
                    "the orchestrator refuses the request itself; no worker is asked",
                )
                for case in (
                    "multi_prompt",
                    "previous_response_id",
                    "responses_retrieve_delete",
                    "messages_batches",
                )
            ),
            (
                "_test_openengine_disagg_inputs.py::test_request_chat_template_gate[closed]",
                "the orchestrator refuses a request's own chat_template before any worker",
            ),
            (
                "_test_openengine_disagg_errors.py::"
                "test_worker_frontend_4xx_passes_through_the_orchestrator",
                "the context worker frontend refuses the request before calling its engine",
            ),
            (
                "_test_openengine_disagg_errors.py::"
                "test_orchestrator_rejects_an_invalid_body_before_any_worker",
                "the orchestrator refuses the request body itself; no worker is asked",
            ),
            (
                "_test_openengine_disagg_baseline_limits.py::"
                "test_tool_choice_required_is_rejected_like_the_baseline",
                "the context worker frontend refuses tool_choice='required' before calling "
                "its engine",
            ),
        )
    ),
    *(
        expectation
        for match, endpoint in (
            (f"{_ENDPOINTS}test_version_direct", "/version"),
            (f"{_ENDPOINTS}test_health_direct", "/health"),
            (f"{_UNSUPPORTED_OPS}test_profile_endpoints_refused_direct", "the profiling endpoints"),
        )
        for expectation in (
            Expectation(
                "WT-57",
                match,
                NO_ENGINE_TRAFFIC,
                f"asks each worker frontend for {endpoint} only",
                chains=_CHAIN,
            ),
            Expectation(
                "WT-57",
                match,
                LEG_EXPECT,
                f"sends {endpoint} to each worker frontend itself, not through the orchestrator",
                chains=_CHAIN,
                leg=DIRECT_WORKER,
            ),
        )
    ),
    Expectation(
        "WT-57",
        f"{_ENDPOINTS}test_health_generate_direct",
        LEG_EXPECT,
        "sends /health_generate to each worker frontend itself, which makes one aggregated "
        "call of its own",
        chains=_CHAIN,
        leg=DIRECT_WORKER,
        direct_requests=1,
    ),
    *(
        Expectation(
            "WT-57",
            f"_test_openengine_disagg_metrics.py::{test}",
            LEG_EXPECT,
            f"sends one aggregated completion straight to the generation worker frontend, "
            f"then reads its {endpoint}",
            chains=_CHAIN,
            leg=DIRECT_WORKER,
            direct_requests=1,
        )
        for test, endpoint in (
            ("test_metrics_snapshot_direct", "/metrics"),
            ("test_prometheus_direct", "/prometheus/metrics"),
        )
    ),
    Expectation(
        "WT-57",
        "_test_openengine_disagg_internal_auth.py::"
        "test_gen_worker_frontend_rejects_a_forged_generation_only_request",
        LEG_EXPECT,
        "sends a context leg and three generation legs (two refused by the internal auth "
        "check) to the worker frontends itself",
        chains=_CHAIN,
        leg=DIRECT_TWO_LEG,
    ),
    Expectation(
        "WT-57",
        "_test_openengine_disagg_retry.py::test_request_survives_a_lost_ctx_worker_with_a_new_id",
        LEG_EXPECT,
        "kills the first context worker (its frontend on the chain) and starts it again on its "
        "port; the orchestrator retries the context leg",
        chains=_CHAIN,
        leg=KILLED,
        killed=("ctx",),
    ),
    *(
        Expectation("WT-57", match, LEG_EXPECT, reason, chains=_CHAIN, leg=GEN_ZERO_OK)
        for match, reason in (
            (
                "_test_openengine_disagg_stopping.py::test_first_token_eos_ends_on_the_context_leg",
                "the first token is the end of sequence, so the request ends on the context leg",
            ),
            (
                "_test_openengine_disagg_sampling.py::test_stop_word_on_first_token[*]",
                "the first token is the stop word, so the request ends on the context leg",
            ),
        )
    ),
    Expectation(
        "WT-57",
        f"{_UNSUPPORTED_OPS}test_ops_options_startup_rejected[resource_governor]",
        STARTUP_REJECTED,
        "a worker frontend has no engine in its process to run the resource governor in",
        chains=_CHAIN,
        log_contains="enable_resource_governor is not supported",
    ),
    Expectation(
        "WT-57",
        f"{_UNSUPPORTED_OPS}test_ops_options_startup_rejected[sleep_mode]",
        SKIP,
        "the chain harness's option table has no side for --enable_sleep_mode, so the "
        "worker frontend cannot be started with it",
        chains=_CHAIN,
    ),
]

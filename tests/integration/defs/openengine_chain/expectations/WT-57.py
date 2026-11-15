# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-57's new disaggregated tests that do not send both legs through the orchestrator.

Each test of the ``_test_openengine_disagg_*`` files starts its own
deployment: an orchestrator and workers with ``--server_role``. A test listed
here sends no generation request through the orchestrator, or sends it to one
worker itself, so the chain proof cannot find both legs. The migrated tests
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
    STARTUP_REJECTED,
    Expectation,
)

_ENDPOINTS = "_test_openengine_disagg_endpoints.py::"
_INPUTS = "_test_openengine_disagg_inputs.py::"
_UNSUPPORTED_OPS = "_test_openengine_disagg_unsupported_ops.py::"

EXPECTATIONS = [
    *(
        Expectation(
            "WT-57",
            f"{_ENDPOINTS}test_orchestrator_limits_same_as_baseline[{case}]",
            NO_ENGINE_TRAFFIC,
            "the orchestrator refuses the request itself; no worker is asked",
            chains=(CHAIN_TRTLLM,),
        )
        for case in (
            "multi_prompt",
            "previous_response_id",
            "responses_retrieve_delete",
            "messages_batches",
        )
    ),
    *(
        Expectation(
            "WT-57",
            f"{_ENDPOINTS}{test}",
            LEG_EXPECT,
            "sends one aggregated completion straight to the generation worker frontend, "
            "then asks that worker for an endpoint the orchestrator does not serve for it",
            chains=(CHAIN_TRTLLM,),
            leg=DIRECT_WORKER,
        )
        for test in ("test_version_direct", "test_health_direct")
    ),
    Expectation(
        "WT-57",
        f"{_ENDPOINTS}test_health_generate_direct",
        LEG_EXPECT,
        "sends one aggregated completion straight to the generation worker frontend; its "
        "/health_generate makes one more aggregated call of its own",
        chains=(CHAIN_TRTLLM,),
        leg=DIRECT_WORKER,
    ),
    *(
        Expectation(
            "WT-57",
            f"_test_openengine_disagg_metrics.py::{test}",
            LEG_EXPECT,
            f"sends one aggregated completion straight to the generation worker frontend, "
            f"then reads its {endpoint}",
            chains=(CHAIN_TRTLLM,),
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
        chains=(CHAIN_TRTLLM,),
        leg=DIRECT_TWO_LEG,
    ),
    Expectation(
        "WT-57",
        "_test_openengine_disagg_retry.py::test_request_survives_a_lost_ctx_worker_with_a_new_id",
        LEG_EXPECT,
        "kills the first context worker (its frontend on the chain) and starts it again on its "
        "port; the orchestrator retries the context leg",
        chains=(CHAIN_TRTLLM,),
        leg=KILLED,
        killed=("ctx",),
    ),
    Expectation(
        "WT-57",
        f"{_INPUTS}test_request_chat_template_gate",
        NO_ENGINE_TRAFFIC,
        "the orchestrator refuses a request's own chat_template before any worker",
        chains=(CHAIN_TRTLLM,),
    ),
    *(
        Expectation("WT-57", match, NO_ENGINE_TRAFFIC, reason, chains=(CHAIN_TRTLLM,))
        for match, reason in (
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
            (
                f"{_UNSUPPORTED_OPS}test_profile_endpoints_refused_direct",
                "asks each worker frontend for the profiling endpoints only",
            ),
        )
    ),
    *(
        Expectation(
            "WT-57",
            f"{_UNSUPPORTED_OPS}test_ops_options_startup_rejected[{option}]",
            STARTUP_REJECTED,
            f"a worker frontend has no engine in its process to run {option} on",
            chains=(CHAIN_TRTLLM,),
            log_contains=refusal,
        )
        for option, refusal in (
            ("resource_governor", "enable_resource_governor is not supported"),
            ("sleep_mode", "--openengine-endpoint cannot be combined with --enable_sleep_mode"),
        )
    ),
    Expectation(
        "WT-57",
        "_test_openengine_disagg_stopping.py::test_first_token_eos_ends_on_the_context_leg",
        LEG_EXPECT,
        "the first token is the end of sequence, so the request ends on the context leg",
        chains=(CHAIN_TRTLLM,),
        leg=GEN_ZERO_OK,
    ),
]

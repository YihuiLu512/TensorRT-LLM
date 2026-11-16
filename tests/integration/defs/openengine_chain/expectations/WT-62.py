# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the DeepSeek-V4.1 disaggregated tests are judged on the ds chain."""

from openengine_chain.conventions import CHAIN_DS
from openengine_chain.expectations import (
    CANCELS_CALLS,
    DIRECT_CTX_LEG,
    DIRECT_TWO_LEG,
    DIRECT_WORKER,
    GEN_ZERO_OK,
    KILLED,
    LEG_EXPECT,
    NO_ENGINE_TRAFFIC,
    STARTUP_REJECTED,
    Expectation,
)

_BASIC = "_test_openengine_ds_disagg_basic.py::"
_SAMPLING = "_test_openengine_ds_disagg_sampling.py::"
_GSM8K = "_test_openengine_ds_disagg_gsm8k.py::"
_REJECTS = "_test_openengine_ds_disagg_rejects.py::"
_HANDOFF = "_test_openengine_ds_disagg_handoff.py::"
_CLUSTER = "_test_openengine_ds_disagg_cluster.py::"
_FRONTEND = "_test_openengine_ds_disagg_frontend.py::"
_CANCEL = "_test_openengine_ds_disagg_cancel.py::"


def _entry(match: str, kind: str, reason: str, **fields) -> Expectation:
    return Expectation("WT-62", match, kind, reason, chains=(CHAIN_DS,), **fields)


EXPECTATIONS = [
    _entry(
        _BASIC + "test_messages_count_tokens",
        NO_ENGINE_TRAFFIC,
        "the front end counts the rendered prompt's tokens without a Generate call",
    ),
    _entry(
        _BASIC + "test_worker_frontend_endpoints_direct*",
        NO_ENGINE_TRAFFIC,
        "it reads each worker frontend's own endpoints only, with no Generate call",
    ),
    _entry(
        _BASIC + "test_worker_frontend_endpoints_direct*",
        LEG_EXPECT,
        "it sends to the worker frontends themselves",
        leg=DIRECT_WORKER,
    ),
    _entry(
        _BASIC + "test_health_generate_and_aggregated_direct*",
        LEG_EXPECT,
        "it sends an aggregated request and /health_generate to one worker frontend itself",
        leg=DIRECT_WORKER,
    ),
    _entry(
        _SAMPLING + "test_first_token_stop_skips_gen*",
        LEG_EXPECT,
        "a request whose first token stops it ends on the context leg, with no generation leg",
        leg=GEN_ZERO_OK,
    ),
    _entry(
        _GSM8K + "test_two_endpoint_parity",
        LEG_EXPECT,
        "it also sends each prompt as an aggregated request to the context worker frontend",
        leg=DIRECT_WORKER,
    ),
    _entry(
        _REJECTS + "test_engine_unsupported_params_rejected*",
        NO_ENGINE_TRAFFIC,
        "every request it sends is refused by a worker frontend",
    ),
    _entry(
        _REJECTS + "test_engine_unsupported_params_rejected*",
        LEG_EXPECT,
        "it also sends each refused request as a context leg to the context worker frontend",
        leg=DIRECT_CTX_LEG,
        orphans=0,
    ),
    _entry(
        _REJECTS + "test_n_best_of_rejected*",
        NO_ENGINE_TRAFFIC,
        "every request with n > 1 or best_of > 1 is refused",
    ),
    _entry(
        _REJECTS + "test_n_best_of_rejected[direct_best_of_gt_n]",
        LEG_EXPECT,
        "it sends the aggregated request to the context worker frontend itself",
        leg=DIRECT_WORKER,
    ),
    _entry(
        _REJECTS + "test_profile_endpoints_rejected_direct",
        NO_ENGINE_TRAFFIC,
        "it asks the worker frontends' profiling endpoints only",
    ),
    _entry(
        _REJECTS + "test_profile_endpoints_rejected_direct",
        LEG_EXPECT,
        "it sends to the worker frontends themselves",
        leg=DIRECT_WORKER,
    ),
    _entry(
        _REJECTS + "test_tool_choice_required_rejected",
        NO_ENGINE_TRAFFIC,
        "the request is refused by a worker frontend",
    ),
    _entry(
        _REJECTS + "test_startup_rejects*",
        NO_ENGINE_TRAFFIC,
        "it starts workers that must refuse to start, or checks the front end's command line",
    ),
    _entry(
        _REJECTS + "test_startup_rejects[beam]",
        STARTUP_REJECTED,
        "the context worker is started with max_beam_width > 1; OpenEngine has no beam search",
        log_contains="max_beam_width > 1 is not supported: OpenEngine has no beam search",
    ),
    _entry(
        _REJECTS + "test_startup_rejects[resource_governor]",
        STARTUP_REJECTED,
        "the context worker is started with the resource governor, which runs in the executor",
        log_contains=(
            "enable_resource_governor is not supported: the resource governor runs inside the "
            "engine's executor"
        ),
    ),
    _entry(
        _REJECTS + "test_startup_rejects[otlp]",
        LEG_EXPECT,
        "it starts one context worker and sends it an aggregated request itself",
        leg=DIRECT_WORKER,
    ),
    _entry(
        _HANDOFF + "test_direct_handoff_engine_errors*",
        LEG_EXPECT,
        "it sends both legs to the two worker frontends itself",
        leg=DIRECT_TWO_LEG,
    ),
    _entry(
        _HANDOFF + "test_direct_handoff_silent_behaviors*",
        LEG_EXPECT,
        "it sends both legs to the two worker frontends itself",
        leg=DIRECT_TWO_LEG,
    ),
    _entry(
        _HANDOFF + "test_direct_handoff_auth_and_multimodal_attributes",
        LEG_EXPECT,
        "it sends both legs to the two worker frontends itself",
        leg=DIRECT_TWO_LEG,
    ),
    _entry(
        _HANDOFF + "test_invalid_prompt_rejected*",
        LEG_EXPECT,
        "it sends the refused context legs to the context worker frontend itself",
        leg=DIRECT_CTX_LEG,
        orphans=0,
    ),
    _entry(
        _HANDOFF + "test_role_startup_check",
        NO_ENGINE_TRAFFIC,
        "it starts an engine without a role and worker frontends that must refuse it, itself",
    ),
    _entry(
        _CLUSTER + "test_ctx_leg_retry_after_ctx_frontend_killed",
        LEG_EXPECT,
        "it kills one of the context worker's two front ends while requests are in flight",
        leg=KILLED,
        killed=("ctx",),
    ),
    _entry(
        _FRONTEND + "test_dspark_draft_handoff",
        LEG_EXPECT,
        "it sends the context and generation legs to the two worker frontends itself",
        leg=DIRECT_TWO_LEG,
    ),
    _entry(
        _CANCEL + "test_abort_stages[*",
        CANCELS_CALLS,
        "it leaves requests while they wait, prefill or decode",
    ),
    _entry(
        _CANCEL + "test_abort_stages[queued]",
        LEG_EXPECT,
        "it sends context legs to the context worker frontend itself and leaves them before a "
        "handoff",
        leg=DIRECT_CTX_LEG,
        orphans=0,
    ),
    _entry(
        _CANCEL + "test_abort_stages[prefill]",
        LEG_EXPECT,
        "it sends context legs to the context worker frontend itself and leaves them before a "
        "handoff",
        leg=DIRECT_CTX_LEG,
        orphans=0,
    ),
    _entry(
        _CANCEL + "test_parked_kv_timeout_and_orphans",
        CANCELS_CALLS,
        "it aborts parked requests through the engine's control service",
    ),
    _entry(
        _CANCEL + "test_parked_kv_timeout_and_orphans",
        LEG_EXPECT,
        "it parks two handoffs it never claims and pulls one of them after its timeout",
        leg=DIRECT_CTX_LEG,
        orphans=1,
    ),
    _entry(
        _CANCEL + "test_engine_loss[ctx]",
        LEG_EXPECT,
        "it kills the context worker's ds engine",
        leg=KILLED,
        killed=("ctx",),
    ),
    _entry(
        _CANCEL + "test_engine_loss[gen]",
        LEG_EXPECT,
        "it kills the generation worker's ds engine",
        leg=KILLED,
        killed=("gen",),
    ),
    _entry(
        _CANCEL + "test_engine_loss_pull_from_lost_context",
        LEG_EXPECT,
        "it sends both legs to the worker frontends itself and kills the context ds engine "
        "between them",
        leg=DIRECT_TWO_LEG,
    ),
]

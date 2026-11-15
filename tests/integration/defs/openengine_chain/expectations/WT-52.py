# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-52's table: refusals of the disaggregated chain, and tests that send legs to a worker.

A request the worker frontend refuses sends no Generate call; a worker the
frontend refuses to start serves nothing, and its log says why. A test that
sends context legs to the context worker itself, with no generation leg,
leaves their handoffs unclaimed (``direct_ctx_leg`` with the number it
leaves); one that sends both legs itself is ``direct_two_leg``. A test that
kills one role's engine is ``killed``.
"""

from openengine_chain.conventions import CHAIN_TRTLLM
from openengine_chain.expectations import (
    DIRECT_CTX_LEG,
    DIRECT_TWO_LEG,
    KILLED,
    LEG_EXPECT,
    NO_ENGINE_TRAFFIC,
    STARTUP_REJECTED,
    Expectation,
)

_REJECTS = "_test_openengine_disagg_rejects.py::"
_CANCEL = "_test_openengine_disagg_cancel.py::"
_LOSS = "_test_openengine_disagg_engine_loss.py::"
_BEAM_LOG = "max_beam_width > 1 is not supported"
_TRTLLM = (CHAIN_TRTLLM,)


def _refused(test: str, reason: str) -> Expectation:
    return Expectation(
        "WT-52",
        _REJECTS + test,
        NO_ENGINE_TRAFFIC,
        f"it sends only requests the worker frontend refuses: {reason}",
        chains=_TRTLLM,
    )


def _ctx_legs(match: str, orphans: int, reason: str) -> Expectation:
    return Expectation(
        "WT-52",
        match,
        LEG_EXPECT,
        reason,
        chains=_TRTLLM,
        leg=DIRECT_CTX_LEG,
        orphans=orphans,
    )


def _killed(test: str, role: str, killed: str) -> Expectation:
    return Expectation(
        "WT-52",
        _LOSS + test,
        LEG_EXPECT,
        f"it kills the {role} worker's engine",
        chains=_TRTLLM,
        leg=KILLED,
        killed=(killed,),
    )


EXPECTATIONS = [
    _refused("test_multimodal_rejected", "a chat request with an image"),
    _refused("test_n_gt_1_rejected", "n > 1 and best_of > 1"),
    _refused("test_logits_fields_rejected", "return_context_logits"),
    _refused("test_n_above_one_is_rejected_through_the_orchestrator", "a chat request with n > 1"),
    _ctx_legs(
        _REJECTS + "test_two_leg_rejections_return_400_naming_the_leg",
        1,
        "its one accepted context leg's handoff is sent with generation legs the frontend refuses",
    ),
    _ctx_legs(
        _REJECTS + "test_multimodal_request_is_rejected",
        1,
        "its one context leg's handoff is sent with generation legs the frontend refuses",
    ),
    _ctx_legs(
        _REJECTS + "test_generation_logits_request_is_rejected",
        1,
        "its one context leg's handoff is sent with a generation leg the frontend refuses",
    ),
    Expectation(
        "WT-52",
        _REJECTS + "test_beam_width_startup_rejected",
        STARTUP_REJECTED,
        "the context worker is started with max_beam_width > 1; OpenEngine has no beam search",
        log_contains=_BEAM_LOG,
        chains=_TRTLLM,
    ),
    Expectation(
        "WT-52",
        _REJECTS + "test_remote_tail_config_refused_at_startup",
        STARTUP_REJECTED,
        "the generation worker is started with bounded_replay_on_generation; a handoff does "
        "not carry remote_tail_start",
        log_contains="bounded_replay_on_generation is not supported",
        chains=_TRTLLM,
    ),
    # XD-219 (F-25, EG-06): its workers are started with max_beam_width 2.
    Expectation(
        "WT-52",
        "test_disaggregated_serving.py::TestGPTOSS20B::test_beam_search*",
        STARTUP_REJECTED,
        "the workers are started with max_beam_width > 1; OpenEngine has no beam search",
        log_contains=_BEAM_LOG,
        chains=_TRTLLM,
    ),
    _ctx_legs(
        _CANCEL + "test_disconnect_during_prefill_aborts_the_ctx_leg",
        0,
        "it sends a context leg to the context worker itself and leaves while it prefills",
    ),
    Expectation(
        "WT-52",
        _CANCEL + "test_abort_reaches_the_engine_at_each_stage",
        LEG_EXPECT,
        "it sends context and generation legs to the workers itself; the long context leg's "
        "handoff is not claimed",
        chains=_TRTLLM,
        leg=DIRECT_TWO_LEG,
    ),
    _ctx_legs(
        "_test_openengine_disagg_orphans.py::test_an_unclaimed_handoff_is_reported_as_an_orphan",
        2,
        "it sends two context legs to the context worker itself and no generation leg",
    ),
    Expectation(
        "WT-52",
        "_test_openengine_disagg_arbitrary_kv.py::test_data_transceiver_state_endpoint_unsupported",
        NO_ENGINE_TRAFFIC,
        "it asks each worker frontend for /v1/data_transceiver_state only",
        chains=_TRTLLM,
    ),
    Expectation(
        "WT-52",
        "_test_openengine_disagg_startup.py::"
        "test_role_frontend_refuses_an_engine_without_kv_connector",
        NO_ENGINE_TRAFFIC,
        "it starts an engine without KV transfer, and worker frontends that refuse it, itself",
        chains=_TRTLLM,
    ),
    _killed("test_ctx_engine_exit_stops_the_frontend_and_drops_the_worker", "context", "ctx"),
    _killed("test_gen_engine_exit_stops_the_frontend_and_drops_the_worker", "generation", "gen"),
]

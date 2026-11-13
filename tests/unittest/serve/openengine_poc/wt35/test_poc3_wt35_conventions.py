# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Version 4 of the OpenEngine conventions: the disaggregated-serving keys and the min_tokens gate."""

from __future__ import annotations

import pytest

from tensorrt_llm.grpc import openengine_conventions as conv

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def test_version_4():
    assert conv.CONVENTIONS_VERSION == 4
    assert conv.CONVENTIONS_VERSION_KEY == "trtllm.conventions_version"


def test_serving_side_and_engine_read_one_version():
    """The OpenEngine server and the serving side import the version from this module."""
    from tensorrt_llm.grpc.openengine import capabilities, control
    from tensorrt_llm.serve.openengine_poc import facade

    assert capabilities.CONVENTIONS_VERSION is conv.CONVENTIONS_VERSION
    assert control.CONVENTIONS_VERSION is conv.CONVENTIONS_VERSION
    assert facade.CONVENTIONS_VERSION is conv.CONVENTIONS_VERSION


def test_request_keys():
    assert conv.DISAGG_REQUEST_KEYS == (
        "request_type",
        "disagg_request_id",
        "schedule_style",
        "orig_elapsed_s",
    )
    # They keep the DisaggregatedParams field names and stay apart from the version 3 keys.
    assert not set(conv.DISAGG_REQUEST_KEYS) & set(conv.REQUEST_KEYS)
    assert conv.DISAGG_REQUEST_KEY_CAPABILITY == {
        "disagg_request_id": "trtllm.request.disagg_request_id",
        "schedule_style": "trtllm.disagg.generation_first",
        "orig_elapsed_s": "trtllm.disagg.orig_elapsed_s",
    }
    assert (conv.SCHEDULE_STYLE_CONTEXT_FIRST, conv.SCHEDULE_STYLE_GENERATION_FIRST) == (0, 1)


def test_schedule_style_values_match_the_llm_enum():
    from tensorrt_llm.disaggregated_params import DisaggScheduleStyle

    assert conv.SCHEDULE_STYLE_CONTEXT_FIRST == DisaggScheduleStyle.CONTEXT_FIRST
    assert conv.SCHEDULE_STYLE_GENERATION_FIRST == DisaggScheduleStyle.GENERATION_FIRST


def test_capability_keys():
    assert conv.VERSION_4_CAPABILITY_KEYS == (
        "trtllm.request.disagg_request_id",
        "trtllm.disagg.generation_first",
        "trtllm.disagg.orig_elapsed_s",
        "trtllm.stopping.min_tokens",
    )
    assert not set(conv.VERSION_4_CAPABILITY_KEYS) & set(conv.BOOL_CAPABILITY_KEYS)
    assert set(conv.DISAGG_REQUEST_KEY_CAPABILITY.values()) < set(conv.VERSION_4_CAPABILITY_KEYS)


@pytest.mark.parametrize("key", conv.VERSION_4_CAPABILITY_KEYS)
@pytest.mark.parametrize(
    "value, expected", [(True, True), (False, False), (1, False), ("true", False), (None, False)]
)
def test_version_4_capability_keys_are_read_as_booleans(key, value, expected):
    extra = {} if value is None else {key: value}
    caps = conv.parse_extra_capabilities(extra)
    assert caps.supports(key) is expected


def test_version_4_keys_leave_the_version_3_reading_unchanged():
    caps = conv.parse_extra_capabilities({conv.CAP_STOPPING_MIN_TOKENS: True, conv.CAP_MIN_P: True})
    assert caps.declared == {conv.CAP_STOPPING_MIN_TOKENS, conv.CAP_MIN_P}
    assert caps.accepted("min_p") == conv.ANY_VALUE
    assert caps.accepted("repetition_penalty") == conv.NO_VALUE


@pytest.mark.parametrize(
    "has_kv_session, request_type, bucket",
    [
        (True, None, "generation_only"),
        (True, "context_only", "generation_only"),
        (False, "context_only", "context_only"),
        (False, "context_and_generation", "aggregated"),
        (False, "generation_only", "aggregated"),
        (False, None, "aggregated"),
        (False, 3, "aggregated"),
    ],
)
def test_request_bucket(has_kv_session, request_type, bucket):
    assert conv.request_bucket(has_kv_session=has_kv_session, request_type=request_type) == bucket
    assert bucket in conv.REQUEST_BUCKETS


@pytest.mark.parametrize(
    "value, expected",
    [
        (str(1 << 40), 1 << 40),
        (str((1 << 63) - 1), (1 << 63) - 1),
        (str((1 << 40) - 1), None),
        (str(1 << 63), None),
        ("0", None),
        ("+1099511627776", None),
        (" 1099511627776", None),
        ("1099511627776\n", None),
        ("１" * 13, None),  # fullwidth digits are digits to int() but not ASCII
        ("0001099511627776", None),  # one id has one text
        ("0" * 20, None),
        ("1" * 4301, None),  # longer than int() converts by default
        ("", None),
        (1 << 40, None),
        (float(1 << 40), None),
        (True, None),
        (None, None),
    ],
)
def test_parse_disagg_request_id(value, expected):
    assert conv.parse_disagg_request_id(value) == expected


def test_disagg_request_id_range_matches_the_orchestrator():
    from tensorrt_llm.llmapi.disagg_utils import MIN_GLOBAL_ID

    assert conv.DISAGG_REQUEST_ID_MIN == MIN_GLOBAL_ID
    assert conv.DISAGG_REQUEST_ID_LIMIT == 1 << 63


def test_leg_specific_fields():
    assert dict(conv.LEG_SPECIFIC_FIELDS) == {
        "request_id": "both",
        "extra.request_type": "prefill",
        "extra.disagg_request_id": "prefill",
        "extra.schedule_style": "prefill",
        "kv.session": "decode",
        "extra.orig_elapsed_s": "decode",
    }
    with pytest.raises(TypeError):
        conv.LEG_SPECIFIC_FIELDS["sampling"] = "both"  # type: ignore[index]


@pytest.mark.parametrize(
    "name",
    [
        "LEG_SPECIFIC_FIELDS",
        "DISAGG_REQUEST_KEY_CAPABILITY",
        "COUNTER_GENERATE_ACCEPTED_BUCKETS",
        "SERVE_COUNTER_GENERATE_SENT_BUCKETS",
        "SERVE_COUNTER_HTTP_RECEIVED_BUCKETS",
    ],
)
def test_version_4_tables_are_read_only(name):
    with pytest.raises(TypeError):
        getattr(conv, name)["x"] = "y"


def test_counter_keys():
    assert conv.COUNTER_GENERATE_ACCEPTED_BUCKETS == {
        "context_only": "trtllm.counters.generate_accepted.context_only",
        "generation_only": "trtllm.counters.generate_accepted.generation_only",
        "aggregated": "trtllm.counters.generate_accepted.aggregated",
    }
    assert conv.COUNTER_PREFILL_READY_SENT == "trtllm.counters.prefill_ready_sent"
    assert conv.SERVE_COUNTERS == (
        "generate_sent",
        "generate_sent.context_only",
        "generate_sent.generation_only",
        "generate_sent.aggregated",
        "prefill_ready_received",
        "http_received.context_only",
        "http_received.generation_only",
        "http_received.aggregated",
        "http_rejected",
        "handoff_unregistered_attributes",
    )
    assert conv.SERVE_REQUEST_DUMP_DIR_ENV == "TRTLLM_OPENENGINE_REQUEST_DUMP_DIR"


def test_handoff_attribute_names():
    assert len(set(conv.HANDOFF_MAPPED_ATTRIBUTES)) == len(conv.HANDOFF_MAPPED_ATTRIBUTES) == 9
    assert conv.HANDOFF_REGISTERED_PASSTHROUGH_ATTRIBUTES == ("trtllm_decode_preparation",)
    assert not set(conv.HANDOFF_MAPPED_ATTRIBUTES) & set(
        conv.HANDOFF_REGISTERED_PASSTHROUGH_ATTRIBUTES
    )


def test_handoff_attribute_names_cover_what_the_trtllm_engine_packs():
    """Every attribute the TRT-LLM OpenEngine server writes into PrefillReady is named here."""
    from google.protobuf import json_format

    from tensorrt_llm.disaggregated_params import DisaggregatedParams
    from tensorrt_llm.grpc.openengine.disagg import prefill_ready_response

    params = DisaggregatedParams(
        request_type="context_only",
        first_gen_tokens=[5],
        ctx_request_id=3,
        opaque_state=b"state",
        draft_tokens=[6, 7],
        disagg_request_id=1 << 41,
        ctx_dp_rank=1,
        ctx_info_endpoint="tcp://127.0.0.1:5000",
        schedule_style=0,
        ctx_usage={"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        first_gen_log_probs=[-0.5],
    )
    response = prefill_ready_response("r", params, "NIXL", decode_preparation='{"version": 1}')
    attributes = json_format.MessageToDict(response.prefill_ready.kv_session.attributes_struct)
    named = set(conv.HANDOFF_MAPPED_ATTRIBUTES) | set(
        conv.HANDOFF_REGISTERED_PASSTHROUGH_ATTRIBUTES
    )
    # Every field is set, so every attribute the server knows is packed.
    assert set(attributes) == named
    # Each mapped key is a DisaggregatedParams field of the same name.
    assert set(conv.HANDOFF_MAPPED_ATTRIBUTES) <= set(DisaggregatedParams.__dataclass_fields__)

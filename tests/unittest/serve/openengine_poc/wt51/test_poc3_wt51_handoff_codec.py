# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The handoff codec (SV-DSG-3): ``KvSessionSpec`` to HTTP ``DisaggregatedParams`` and back.

NT-018: a handoff decoded on the context worker, carried over HTTP and encoded
on the generation worker gives the engine the session it sent, for the
TensorRT-LLM shape and the ds shape. NT-019: attributes in
``openengine_handoff_attributes`` that name a mapped attribute are refused.
"""

from __future__ import annotations

import base64
import json

import pytest
from google.protobuf import struct_pb2

from tensorrt_llm.grpc.openengine_conventions import struct_to_dict
from tensorrt_llm.serve.openai_protocol import (
    DisaggregatedParams,
    to_disaggregated_params,
    to_llm_disaggregated_params,
)
from tensorrt_llm.serve.openengine_poc import handoff
from tensorrt_llm.serve.openengine_poc.engine_client import KvEndpoint, KvSessionSpec

pytestmark = pytest.mark.cpu_only

DISAGG_REQUEST_ID = str((1 << 41) + 5)
OPAQUE_STATE = base64.b64encode(b"\x00engine state\xff").decode("ascii")
CTX_USAGE = {
    "prompt_tokens": 7,
    "total_tokens": 8,
    "completion_tokens": 1,
    "prompt_tokens_details": None,
}


def _wire(attributes: dict) -> dict:
    """``attributes`` as a client reads them from ``attributes_struct``: numbers are floats."""
    struct = struct_pb2.Struct()
    struct.update(attributes)
    return struct_to_dict(struct)


def _trtllm_session(**extra) -> KvSessionSpec:
    """A handoff shaped as the TensorRT-LLM OpenEngine server packs it."""
    attributes = {
        "ctx_info_endpoint": "tcp://10.1.2.3:4567",
        "opaque_state": OPAQUE_STATE,
        "first_gen_tokens": [11],
        "draft_tokens": [12, 13],
        "disagg_request_id": DISAGG_REQUEST_ID,
        "ctx_dp_rank": 1,
        "schedule_style": 0,
        # As the HTTP UsageInfo dumps it, which is what a generation worker sends on.
        "ctx_usage": CTX_USAGE,
        "first_gen_log_probs": [[[11, -0.25, 1], [3, -1.5, 2]]],
        # Not registered, so passed through as the registered ones are.
        "vendor_state": {"kind": "none"},
        **extra,
    }
    return KvSessionSpec(
        session_id=DISAGG_REQUEST_ID,
        transfer_backend="NIXL",
        dp_rank=1,
        endpoints=(KvEndpoint(host="10.1.2.3", port=4567, protocol="tcp"),),
        attributes=_wire(attributes),
    )


def _ds_session() -> KvSessionSpec:
    """A handoff shaped as the ds OpenEngine server packs it."""
    attributes = {
        "ctx_info_endpoint": "tcp://10.1.2.4:7000",
        "first_gen_tokens": [9],
        "disagg_request_id": DISAGG_REQUEST_ID,
        "ctx_dp_rank": 0,
    }
    return KvSessionSpec(
        session_id=DISAGG_REQUEST_ID,
        transfer_backend="ds-xfer",
        endpoints=(KvEndpoint(host="10.1.2.4", port=7000, protocol="tcp"),),
        attributes=_wire(attributes),
    )


def _over_http(session: KvSessionSpec) -> tuple[DisaggregatedParams, str | None]:
    """The context worker's response field, read back as the generation worker's request gets it."""
    params, attributes = handoff.decode_kv_session(session)
    http = to_disaggregated_params(params)
    http.openengine_handoff_attributes = attributes
    http.request_type = "generation_only"
    wire = json.loads(json.dumps(http.model_dump(mode="json")))
    received = DisaggregatedParams.model_validate(wire)
    return received, received.openengine_handoff_attributes


@pytest.mark.parametrize("session", [_trtllm_session(), _ds_session()], ids=["trtllm", "ds"])
def test_handoff_fields_round_trip(session):
    received, attributes = _over_http(session)
    encoded = handoff.encode_kv_session(
        to_llm_disaggregated_params(received), attributes, session.transfer_backend
    )
    assert encoded.session_id == session.session_id
    assert encoded.dp_rank == session.dp_rank
    assert encoded.endpoints == session.endpoints
    # As the engine reads it back from attributes_struct.
    assert _wire(dict(encoded.attributes)) == dict(session.attributes)


def test_mapped_attributes_are_restored_field_by_field():
    params, attributes = handoff.decode_kv_session(_trtllm_session())
    assert params.request_type == "context_only"
    assert params.ctx_request_id == int(DISAGG_REQUEST_ID)
    assert params.disagg_request_id == int(DISAGG_REQUEST_ID)
    assert params.ctx_info_endpoint == "tcp://10.1.2.3:4567"
    assert params.opaque_state == base64.b64decode(OPAQUE_STATE)
    assert params.first_gen_tokens == [11] and type(params.first_gen_tokens[0]) is int
    assert params.draft_tokens == [12, 13]
    assert params.ctx_dp_rank == 1 and int(params.schedule_style) == 0
    assert params.ctx_usage == CTX_USAGE
    [position] = params.first_gen_log_probs
    assert position[11].logprob == -0.25 and position[11].rank == 1
    # Only the attribute without a field crosses in the new field.
    assert json.loads(attributes) == {"vendor_state": {"kind": "none"}}


def test_encoded_opaque_state_is_the_engines_base64_text():
    params, _ = handoff.decode_kv_session(_trtllm_session())
    assert to_disaggregated_params(params).encoded_opaque_state == OPAQUE_STATE


def test_a_handoff_without_pass_through_attributes_has_no_new_field():
    params, attributes = handoff.decode_kv_session(_ds_session())
    assert attributes is None
    http = to_disaggregated_params(handoff.with_handoff_attributes(params, attributes))
    assert "openengine_handoff_attributes" not in http.model_dump()


def test_unmapped_session_attribute_rides_the_new_field():
    session = _trtllm_session(trtllm_decode_preparation={"kind": "mm"}, vendor_level=2)
    assert handoff.unregistered_attributes(session) == ["vendor_level", "vendor_state"]
    _, attributes = handoff.decode_kv_session(session)
    assert json.loads(attributes) == {
        "trtllm_decode_preparation": {"kind": "mm"},
        "vendor_level": 2.0,
        "vendor_state": {"kind": "none"},
    }


def test_a_generation_leg_with_a_decode_preparation_is_refused():
    # Multimodal requests are not served disaggregated (SV-DSG-3 rule 13).
    session = _trtllm_session(trtllm_decode_preparation={"kind": "mm"})
    params, attributes = handoff.decode_kv_session(session)
    with pytest.raises(ValueError, match="trtllm_decode_preparation"):
        handoff.encode_kv_session(params, attributes, "")


@pytest.mark.parametrize("name", ["first_gen_tokens", "ctx_info_endpoint", "disagg_request_id"])
def test_new_field_naming_a_mapped_attribute_is_refused(name):
    params, _ = handoff.decode_kv_session(_ds_session())
    with pytest.raises(ValueError, match=name):
        handoff.encode_kv_session(params, json.dumps({name: [1]}), "")


def test_the_params_keep_the_new_field_on_the_llm_side():
    params, attributes = handoff.decode_kv_session(_trtllm_session())
    carried = handoff.with_handoff_attributes(params, attributes)
    assert handoff.handoff_attributes_of(carried) == attributes
    assert handoff.handoff_attributes_of(params) is None
    assert carried.first_gen_tokens == params.first_gen_tokens
    # Carrying other attributes replaces those it carried.
    again = handoff.with_handoff_attributes(carried, None)
    assert handoff.handoff_attributes_of(again) is None
    assert again.first_gen_tokens == params.first_gen_tokens


@pytest.mark.parametrize(
    "attributes",
    [
        {"session_id": "x"},
        {"opaque_state": "not base64!"},
        {"first_gen_tokens": [1.5]},
        {"disagg_request_id": "007"},
    ],
)
def test_a_malformed_handoff_is_refused(attributes):
    session = _ds_session()
    if "session_id" in attributes:
        session = KvSessionSpec(session_id=attributes["session_id"])
    else:
        session = KvSessionSpec(
            session_id=session.session_id, attributes={**session.attributes, **attributes}
        )
    with pytest.raises(ValueError):
        handoff.decode_kv_session(session)

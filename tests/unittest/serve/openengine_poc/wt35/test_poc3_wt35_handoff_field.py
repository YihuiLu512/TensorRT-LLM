# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CS-20: the HTTP-only DisaggregatedParams field openengine_handoff_attributes and its signature."""

from __future__ import annotations

import json

import pytest

from tensorrt_llm.serve import disagg_auth
from tensorrt_llm.serve.disagg_auth import (
    INTERNAL_DISAGG_AUTH_HEADER,
    build_internal_disagg_auth_headers,
    request_requires_internal_disagg_auth,
    validate_internal_disagg_request,
)
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    DisaggregatedParams,
    to_disaggregated_params,
    to_llm_disaggregated_params,
)

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

FIELD = "openengine_handoff_attributes"
ATTRIBUTES = '{"trtllm_decode_preparation":"{}","x":1}'


def _params(**fields) -> DisaggregatedParams:
    return DisaggregatedParams(
        request_type="generation_only", ctx_request_id=1, disagg_request_id=2, **fields
    )


def _request(**fields) -> CompletionRequest:
    return CompletionRequest(
        model="m", prompt="hello", stream=False, disaggregated_params=_params(**fields)
    )


def test_the_field_is_optional_and_http_only():
    assert DisaggregatedParams.model_fields[FIELD].default is None
    llm_params = to_llm_disaggregated_params(_params(**{FIELD: ATTRIBUTES}))
    assert not hasattr(llm_params, FIELD)
    assert getattr(to_disaggregated_params(llm_params), FIELD) is None


@pytest.mark.parametrize(
    "dump",
    [
        lambda p: p.model_dump(),
        lambda p: p.model_dump(mode="json"),
        lambda p: p.model_dump(mode="json", exclude_unset=True, by_alias=True),
        lambda p: json.loads(p.model_dump_json()),
    ],
)
def test_an_absent_field_is_not_serialized(dump):
    data = dump(_params())
    assert FIELD not in data
    # Set explicitly to None, it is still left out.
    assert FIELD not in dump(_params(**{FIELD: None}))


def test_absent_the_params_serialize_as_before():
    fields = set(DisaggregatedParams.model_fields) - {FIELD}
    assert set(_params().model_dump()) == fields
    request = _request()
    nested = request.model_dump(mode="json", exclude_unset=True, by_alias=True)
    assert nested["disaggregated_params"] == {
        "request_type": "generation_only",
        "ctx_request_id": 1,
        "disagg_request_id": 2,
    }


def test_a_set_field_round_trips_through_json():
    params = _params(**{FIELD: ATTRIBUTES})
    for data in (
        params.model_dump(),
        params.model_dump(mode="json", exclude_unset=True, by_alias=True),
        json.loads(params.model_dump_json()),
    ):
        assert data[FIELD] == ATTRIBUTES
    again = DisaggregatedParams.model_validate_json(params.model_dump_json())
    assert getattr(again, FIELD) == ATTRIBUTES
    # The orchestrator reuses the context worker's params object for the
    # generation request and serializes it with exclude_unset.
    request = _request()
    request.disaggregated_params = again
    nested = request.model_dump(mode="json", exclude_unset=True, by_alias=True)
    assert nested["disaggregated_params"][FIELD] == ATTRIBUTES


def test_the_field_is_signed():
    assert FIELD in disagg_auth.get_internal_disagg_auth_fields()
    assert request_requires_internal_disagg_auth(_request(**{FIELD: ATTRIBUTES}))
    assert not request_requires_internal_disagg_auth(_request())


@pytest.mark.parametrize(
    "fields",
    [
        {"encoded_opaque_state": "b3BhcXVl"},
        {"ctx_info_endpoint": "tcp://10.0.0.1:5000"},
        {"encoded_opaque_state": "b3BhcXVl", "ctx_info_endpoint": "tcp://10.0.0.1:5000"},
    ],
)
def test_absent_the_signed_payload_is_byte_for_byte_the_base_one(fields):
    """The payload the base code signed: every protected field, None included, sorted, compact."""
    expected = {"ctx_info_endpoint": None, "encoded_opaque_state": None, **fields}
    base = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert disagg_auth._auth_payload(_request(**fields)) == base


def test_a_set_field_is_signed_whole():
    request = _request(ctx_info_endpoint="tcp://10.0.0.1:5000", **{FIELD: ATTRIBUTES})
    payload = json.loads(disagg_auth._auth_payload(request))
    assert payload[FIELD] == ATTRIBUTES
    headers = build_internal_disagg_auth_headers("secret", request)
    assert INTERNAL_DISAGG_AUTH_HEADER in headers
    validate_internal_disagg_request("secret", request, headers)


@pytest.mark.parametrize("tampered", ['{"x":2}', None, ATTRIBUTES + " "])
def test_changing_the_field_breaks_the_signature(tampered):
    request = _request(ctx_info_endpoint="tcp://10.0.0.1:5000", **{FIELD: ATTRIBUTES})
    headers = build_internal_disagg_auth_headers("secret", request)
    forged = _request(ctx_info_endpoint="tcp://10.0.0.1:5000", **{FIELD: tampered})
    with pytest.raises(ValueError, match="Invalid internal disaggregated request authentication"):
        validate_internal_disagg_request("secret", forged, headers)


def test_adding_the_field_to_a_signed_request_breaks_the_signature():
    request = _request(ctx_info_endpoint="tcp://10.0.0.1:5000")
    headers = build_internal_disagg_auth_headers("secret", request)
    forged = _request(ctx_info_endpoint="tcp://10.0.0.1:5000", **{FIELD: ATTRIBUTES})
    with pytest.raises(ValueError):
        validate_internal_disagg_request("secret", forged, headers)


def test_the_field_alone_needs_a_signature():
    request = _request(**{FIELD: ATTRIBUTES})
    with pytest.raises(ValueError):
        validate_internal_disagg_request("secret", request, {})

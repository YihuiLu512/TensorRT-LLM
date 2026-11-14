# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mapping a disaggregated request to its Generate call (SV-DSG-2), without an engine."""

from __future__ import annotations

import dataclasses

import pytest

from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openengine_poc import contracts, mapping

__extra_import_path__ = [".."]
from fakes import trtllm_like  # noqa: E402

pytestmark = pytest.mark.cpu_only

DISAGG_REQUEST_ID = conv.DISAGG_REQUEST_ID_MIN + 7


def _caps(*, declares_id: bool = True):
    caps = trtllm_like()[1]
    declared = set(caps.extra.declared)
    if declares_id:
        declared.add(conv.CAP_REQ_DISAGG_REQUEST_ID)
    else:
        declared.discard(conv.CAP_REQ_DISAGG_REQUEST_ID)
    return dataclasses.replace(caps, extra=dataclasses.replace(caps.extra, declared=declared))


def _map(params: DisaggregatedParams, caps=None):
    (request,) = mapping.map_request(
        [1, 2, 3],
        SamplingParams(max_tokens=4),
        caps or _caps(),
        contracts.GenerateOptions(disaggregated_params=params),
        engine_name="eng",
    )
    return request


def test_a_context_leg_sends_its_type_and_the_callers_id():
    request = _map(
        DisaggregatedParams(request_type="context_only", disagg_request_id=DISAGG_REQUEST_ID)
    )
    assert request.request_type == "context_only"
    assert request.extra[conv.REQ_DISAGG_REQUEST_ID] == str(DISAGG_REQUEST_ID)
    assert request.kv_session is None


def test_context_and_generation_is_sent_as_its_type_alone():
    request = _map(
        DisaggregatedParams(
            request_type="context_and_generation", disagg_request_id=DISAGG_REQUEST_ID
        )
    )
    assert request.request_type == "context_and_generation"
    assert conv.REQ_DISAGG_REQUEST_ID not in request.extra


def test_a_generation_leg_sends_neither_its_type_nor_an_id():
    # The facade adds its kv_session; generation-first travels in the handoff.
    request = _map(
        DisaggregatedParams(
            request_type="generation_only",
            disagg_request_id=DISAGG_REQUEST_ID,
            schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
        )
    )
    assert request.request_type is None
    assert conv.REQ_DISAGG_REQUEST_ID not in request.extra


@pytest.mark.parametrize(
    "params, caps, match",
    [
        (
            DisaggregatedParams(
                request_type="context_only",
                disagg_request_id=DISAGG_REQUEST_ID,
                schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
            ),
            None,
            "generation-first",
        ),
        (
            DisaggregatedParams(request_type="context_only", disagg_request_id=DISAGG_REQUEST_ID),
            _caps(declares_id=False),
            "disagg_request_id",
        ),
        (
            DisaggregatedParams(request_type="context_only", disagg_request_id=5),
            None,
            "disagg_request_id",
        ),
        (DisaggregatedParams(request_type="prefill_please"), None, "request_type"),
    ],
    ids=["generation-first", "undeclared-id", "id-out-of-range", "unknown-type"],
)
def test_a_context_leg_the_engine_cannot_serve_is_refused(params, caps, match):
    with pytest.raises(ValueError, match=match):
        _map(params, caps)

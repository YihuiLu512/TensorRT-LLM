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


def _caps(*, declares_id: bool = True, generation_first: bool = False):
    caps = trtllm_like()[1]
    declared = set(caps.extra.declared)
    for key, wanted in (
        (conv.CAP_REQ_DISAGG_REQUEST_ID, declares_id),
        (conv.CAP_DISAGG_GENERATION_FIRST, generation_first),
    ):
        if wanted:
            declared.add(key)
        else:
            declared.discard(key)
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


@pytest.mark.parametrize(
    "style", [None, DisaggScheduleStyle.GENERATION_FIRST], ids=["unset", "generation-first"]
)
def test_a_generation_leg_sends_neither_its_type_nor_an_id(style):
    # The facade adds its kv_session; a generation-first style travels in the handoff.
    request = _map(
        DisaggregatedParams(
            request_type="generation_only",
            disagg_request_id=DISAGG_REQUEST_ID,
            schedule_style=style,
        ),
        _caps(generation_first=True),
    )
    assert request.request_type is None
    assert conv.REQ_DISAGG_REQUEST_ID not in request.extra


def test_a_generation_first_generation_leg_to_an_engine_without_it_is_refused():
    params = DisaggregatedParams(
        request_type="generation_only",
        disagg_request_id=DISAGG_REQUEST_ID,
        schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
    )
    with pytest.raises(ValueError, match=r"generation-first.*trtllm\.disagg\.generation_first"):
        _map(params)


def test_context_and_generation_needs_no_generation_first_capability():
    # It is served as one aggregated call, which no schedule style applies to.
    request = _map(
        DisaggregatedParams(
            request_type="context_and_generation",
            schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
        )
    )
    assert request.request_type == "context_and_generation"
    assert conv.REQ_SCHEDULE_STYLE not in request.extra


def test_a_context_first_generation_leg_needs_no_generation_first_capability():
    request = _map(
        DisaggregatedParams(
            request_type="generation_only",
            disagg_request_id=DISAGG_REQUEST_ID,
            schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        )
    )
    assert request.request_type is None


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
            r"generation-first.*trtllm\.disagg\.generation_first",
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
        # DisaggregatedParams refuses an unknown request_type itself.
        (DisaggregatedParams(ctx_request_id=1), None, "request_type"),
    ],
    ids=["undeclared-generation-first", "undeclared-id", "id-out-of-range", "no-type"],
)
def test_a_context_leg_the_engine_cannot_serve_is_refused(params, caps, match):
    with pytest.raises(ValueError, match=match):
        _map(params, caps)

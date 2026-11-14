# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mapping the second-batch keys of a disaggregated request, without an engine.

NT-171: ``orig_elapsed_s`` goes in the ``extra`` of the generation leg only
(EX-DSG-3, CS-15). ``subagent_affinity_id`` is accepted on the two legs of a
disaggregated request, whose metadata the facade decides (SV-DSG-14, CS-09).
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import pytest

from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openengine_poc import contracts, mapping

__extra_import_path__ = [".."]
from fakes import trtllm_like  # noqa: E402

pytestmark = pytest.mark.cpu_only

DISAGG_REQUEST_ID = conv.DISAGG_REQUEST_ID_MIN + 53


def _caps(*declared: str, undeclared: tuple[str, ...] = ()):
    caps = trtllm_like()[1]
    keys = (set(caps.extra.declared) | {conv.CAP_REQ_DISAGG_REQUEST_ID, *declared}) - set(
        undeclared
    )
    return dataclasses.replace(caps, extra=dataclasses.replace(caps.extra, declared=keys))


def _map(params: DisaggregatedParams | None, caps=None, scheduling_params=None):
    (request,) = mapping.map_request(
        [1, 2, 3],
        SamplingParams(max_tokens=4),
        caps or _caps(conv.CAP_DISAGG_ORIG_ELAPSED_S),
        contracts.GenerateOptions(disaggregated_params=params, scheduling_params=scheduling_params),
        engine_name="eng",
    )
    return request


def _context(**fields) -> DisaggregatedParams:
    return DisaggregatedParams(
        request_type="context_only", disagg_request_id=DISAGG_REQUEST_ID, **fields
    )


def _generation(**fields) -> DisaggregatedParams:
    return DisaggregatedParams(
        request_type="generation_only",
        disagg_request_id=DISAGG_REQUEST_ID,
        ctx_request_id=7,
        opaque_state=b"state",
        **fields,
    )


def _scheduling(affinity: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        attention_dp_rank=None,
        attention_dp_relax=True,
        agent_hierarchy=None,
        subagent_affinity_id=affinity,
    )


def test_orig_elapsed_s_is_sent_on_the_generation_leg_only():
    """NT-171: the generation leg carries it in ``extra``; the context leg never does."""
    generation = _map(_generation(orig_elapsed_s=0.25))
    assert generation.extra[conv.REQ_ORIG_ELAPSED_S] == 0.25
    assert isinstance(generation.extra[conv.REQ_ORIG_ELAPSED_S], float)
    # The orchestrator stamps only the generation request; a context request
    # that carries one anyway does not send it either.
    context = _map(_context(orig_elapsed_s=0.25))
    assert conv.REQ_ORIG_ELAPSED_S not in context.extra
    # LEG_SPECIFIC_FIELDS names the key for the decode call alone.
    assert conv.LEG_SPECIFIC_FIELDS[f"extra.{conv.REQ_ORIG_ELAPSED_S}"] == "decode"


def test_a_zero_orig_elapsed_s_is_sent():
    request = _map(_generation(orig_elapsed_s=0.0))
    assert request.extra[conv.REQ_ORIG_ELAPSED_S] == 0.0


def test_a_generation_leg_without_orig_elapsed_s_sends_none():
    assert conv.REQ_ORIG_ELAPSED_S not in _map(_generation()).extra


def test_an_engine_that_does_not_declare_orig_elapsed_s_is_not_sent_it():
    """DQ-11: ds-engine has no gen admission order; the leg is served without the key."""
    caps = _caps(undeclared=(conv.CAP_DISAGG_ORIG_ELAPSED_S,))
    request = _map(_generation(orig_elapsed_s=1.5), caps)
    assert conv.REQ_ORIG_ELAPSED_S not in request.extra
    assert request.request_type is None


@pytest.mark.parametrize("value", [-0.5, math.inf, math.nan], ids=["negative", "inf", "nan"])
def test_an_orig_elapsed_s_the_engine_would_refuse_is_rejected(value):
    with pytest.raises(ValueError, match="orig_elapsed_s must be a finite number >= 0"):
        _map(_generation(orig_elapsed_s=value))


def test_an_orig_elapsed_s_that_is_not_a_number_is_rejected():
    params = _generation()
    params.orig_elapsed_s = True
    with pytest.raises(ValueError, match="orig_elapsed_s must be a number"):
        _map(params)


@pytest.mark.parametrize("leg", [_context, _generation], ids=["context", "generation"])
def test_subagent_affinity_is_accepted_on_both_legs(leg):
    """SV-DSG-14: the mapping leaves the metadata to the facade; nothing goes in ``extra``."""
    request = _map(leg(), scheduling_params=_scheduling("parent"))
    assert conv.REQ_SCHEDULING not in request.extra
    assert not request.metadata


@pytest.mark.parametrize(
    "params",
    [None, DisaggregatedParams(request_type="context_and_generation")],
    ids=["aggregated", "context_and_generation"],
)
def test_subagent_affinity_on_any_other_request_is_rejected(params):
    """The engine checks the metadata on context requests only."""
    with pytest.raises(ValueError, match=r"subagent_affinity_id.*two legs of a disaggregated"):
        _map(params, scheduling_params=_scheduling("parent"))


def test_an_orig_elapsed_s_the_engine_would_refuse_is_rejected_whatever_the_engine():
    """Checked before the capability, so ds-engine and TensorRT-LLM refuse the same values."""
    caps = _caps(undeclared=(conv.CAP_DISAGG_ORIG_ELAPSED_S,))
    with pytest.raises(ValueError, match="orig_elapsed_s must be a finite number >= 0"):
        _map(_generation(orig_elapsed_s=-1.0), caps)

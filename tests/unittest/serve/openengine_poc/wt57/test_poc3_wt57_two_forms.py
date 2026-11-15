# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The facade refuses a logits processor whatever the leg of the request (EG-17), on CPU.

NT-170: a logits processor is a Python callable, which no OpenEngine request
can carry. The facade's request mapping refuses it with the same error for an
aggregated request and for each leg of a disaggregated one, before any
Generate call. EG-17 has no end-to-end test: neither form's HTTP entry point
can take a callable.
"""

from __future__ import annotations

import dataclasses

import pytest

from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.sampling_params import LogitsProcessor, SamplingParams
from tensorrt_llm.serve.openengine_poc import contracts, mapping

__extra_import_path__ = [".."]
from fakes import trtllm_like  # noqa: E402

pytestmark = pytest.mark.cpu_only

DISAGG_REQUEST_ID = conv.DISAGG_REQUEST_ID_MIN + 57
LEGS = {
    "aggregated": None,
    "context_only": DisaggregatedParams(
        request_type="context_only", disagg_request_id=DISAGG_REQUEST_ID
    ),
    "generation_only": DisaggregatedParams(
        request_type="generation_only", disagg_request_id=DISAGG_REQUEST_ID
    ),
}


def _caps():
    caps = trtllm_like()[1]
    declared = set(caps.extra.declared) | {conv.CAP_REQ_DISAGG_REQUEST_ID}
    return dataclasses.replace(caps, extra=dataclasses.replace(caps.extra, declared=declared))


def _map(params: SamplingParams, leg: DisaggregatedParams | None):
    return mapping.map_request(
        [1, 2, 3],
        params,
        _caps(),
        contracts.GenerateOptions(disaggregated_params=leg),
        engine_name="eng",
    )


class _KeepLogits(LogitsProcessor):
    def __call__(self, req_id, logits, token_ids, stream_ptr, client_id):
        pass


@pytest.mark.parametrize("leg", list(LEGS))
def test_logits_processor_is_refused_on_a_two_phase_request(leg):
    # Without the processor each form maps: the refusal below is the processor's alone.
    assert len(_map(SamplingParams(max_tokens=4), LEGS[leg])) == 1
    params = SamplingParams(max_tokens=4, logits_processor=_KeepLogits())
    with pytest.raises(ValueError, match=r"^logits_processor=\S+ is not supported") as refused:
        _map(params, LEGS[leg])
    assert "OpenEngine engine 'eng'" in str(refused.value)

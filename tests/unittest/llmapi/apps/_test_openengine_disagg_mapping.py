# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A disaggregated request's sampling options reach the context and the generation engine alike (EG-12).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU), in-process or on the
OpenEngine chain. On the chain the test reads the two legs' GenerateRequests
(``_test_openengine_disagg_common.legs``).

- NT-195 (EG-12): a completion with ``temperature``, ``top_p``, ``top_k``,
  ``seed`` and the three penalties is served; on the chain both legs'
  GenerateRequests carry these sampling values, and the same sampling block.
"""

from pathlib import Path

import pytest
import requests

from ._test_openengine_disagg_common import (
    CHAIN,
    DUMP_ENV,
    REQUEST_TIMEOUT_S,
    disagg_deployment,
    dump_names,
    legs,
)

pytestmark = pytest.mark.threadleak(enabled=False)

PROMPT = "The capital of France is"
# Values other than the engines' defaults, as the request sends them.
KNOBS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 20,
    "seed": 1234,
    "repetition_penalty": 1.1,
    "frequency_penalty": 0.2,
    "presence_penalty": 0.3,
}
# The GenerateRequest's integer fields; protobuf JSON writes the uint64 seed as a string.
INTEGER_KNOBS = ("top_k", "seed")


@pytest.fixture(scope="module")
def dump_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("disagg_mapping_dumps")


@pytest.fixture(scope="module")
def deployment(tmp_path_factory, dump_dir):
    with disagg_deployment(
        tmp_path_factory.mktemp("disagg_mapping"),
        env={DUMP_ENV: str(dump_dir)},
    ) as deployment:
        yield deployment


def _received(leg: dict) -> dict:
    """The leg's sampling values that the request set, as numbers."""
    sampling = leg.get("sampling", {})
    return {
        name: (int(value) if name in INTEGER_KNOBS else float(value))
        for name, value in sampling.items()
        if name in KNOBS
    }


def test_sampling_knobs_reach_both_engines(deployment, dump_dir):
    before = dump_names(dump_dir)
    body = {"model": deployment.model, "prompt": PROMPT, "max_tokens": 8, **KNOBS}
    response = requests.post(
        f"{deployment.url}/v1/completions", json=body, timeout=REQUEST_TIMEOUT_S
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["text"], response.text
    if not CHAIN:
        return
    prefill, decode = legs(dump_dir, before)
    assert prefill.get("sampling") == decode.get("sampling"), (prefill, decode)
    received = _received(prefill)
    for name, value in KNOBS.items():
        assert name in received, (name, received)
        assert received[name] == pytest.approx(value), (name, received)

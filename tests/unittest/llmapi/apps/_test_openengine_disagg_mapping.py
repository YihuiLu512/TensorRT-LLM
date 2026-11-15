# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A disaggregated request's sampling options reach the context and the generation engine alike (EG-12).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU), in-process or on the
OpenEngine chain. On the chain the worker frontends write every
GenerateRequest they send to ``TRTLLM_OPENENGINE_REQUEST_DUMP_DIR``; the two
dumps of one request are its context leg (``extra.request_type`` is
``context_only``) and its generation leg (with ``kv.session``), paired by
``disagg_request_id``.

- NT-195 (EG-12): a completion with ``temperature``, ``top_p``, ``top_k``,
  ``seed`` and the three penalties is served; on the chain both legs'
  GenerateRequests carry these sampling values, and the same sampling block.
"""

import json
from pathlib import Path

import pytest
import requests

from tensorrt_llm.grpc import openengine_conventions as conv

from ._test_openengine_disagg_common import CHAIN, REQUEST_TIMEOUT_S, disagg_deployment

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
        env={conv.SERVE_REQUEST_DUMP_DIR_ENV: str(dump_dir)},
    ) as deployment:
        yield deployment


def _legs(dump_dir: Path, before: set[str]) -> tuple[dict, dict]:
    """The context and generation GenerateRequests of the one request sent since ``before``."""
    dumps = [
        json.loads(path.read_text())
        for path in sorted(dump_dir.glob("*.json"))
        if path.name not in before
    ]
    ctx = [
        d
        for d in dumps
        if d.get("extra", {}).get(conv.REQ_REQUEST_TYPE) == conv.REQUEST_TYPE_CONTEXT_ONLY
    ]
    gen = [d for d in dumps if "session" in d.get("kv", {})]
    assert len(ctx) == 1 and len(gen) == 1 and len(dumps) == 2, dumps
    [prefill], [decode] = ctx, gen
    ctx_id = conv.as_int(prefill["extra"][conv.REQ_DISAGG_REQUEST_ID])
    gen_id = conv.as_int(decode["kv"]["session"]["attributes_struct"]["disagg_request_id"])
    assert ctx_id is not None and ctx_id == gen_id, (prefill, decode)
    return prefill, decode


def _received(leg: dict) -> dict:
    """The leg's sampling values that the request set, as numbers."""
    sampling = leg.get("sampling", {})
    return {
        name: (int(value) if name in INTEGER_KNOBS else float(value))
        for name, value in sampling.items()
        if name in KNOBS
    }


def test_sampling_knobs_reach_both_engines(deployment, dump_dir):
    before = {path.name for path in dump_dir.glob("*.json")}
    body = {"model": deployment.model, "prompt": PROMPT, "max_tokens": 8, **KNOBS}
    response = requests.post(
        f"{deployment.url}/v1/completions", json=body, timeout=REQUEST_TIMEOUT_S
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["text"], response.text
    if not CHAIN:
        return
    prefill, decode = _legs(dump_dir, before)
    assert prefill.get("sampling") == decode.get("sampling"), (prefill, decode)
    received = _received(prefill)
    for name, value in KNOBS.items():
        assert name in received, (name, received)
        assert received[name] == pytest.approx(value), (name, received)

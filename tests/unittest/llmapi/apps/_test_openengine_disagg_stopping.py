# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a disaggregated request stops: on the context leg, or after both legs (EG-11).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU), in-process or on the
OpenEngine chain. A ``logit_bias`` on the end-of-sequence token makes it the
token every step picks unless something forbids it: the first token is EOS,
and with ``min_tokens`` the first token EOS may end the output at is the one
after ``min_tokens``. Whether the orchestrator asked the generation worker is
read from its own per-role counters on ``/prometheus/metrics``.

- NT-193 (EG-11): when the first token is EOS the request ends on the context
  leg: the client gets the context worker's whole answer, finished with
  ``stop`` and no text, and the orchestrator sends no generation leg.
- NT-194 (EG-11): ``min_tokens`` counts the tokens of both legs together: the
  output has at least ``min_tokens`` tokens and ends with ``stop`` on the token
  after them (one more if EOS is counted), not ``min_tokens`` after the
  generation leg starts.
"""

import re

import pytest
import requests
from transformers import AutoTokenizer

from ._test_openengine_disagg_common import REQUEST_TIMEOUT_S, disagg_deployment, model_path

pytestmark = pytest.mark.threadleak(enabled=False)

PROMPT = "The capital of France is"
EOS_BIAS = 100.0
MIN_TOKENS = 6
# The orchestrator's count of requests it sent to each role's workers.
_SENT = re.compile(r"^(ctx|gen)_total_requests_total(?:\{[^}]*\})?\s+(\S+)\s*$", re.MULTILINE)


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    with disagg_deployment(tmp_path_factory.mktemp("disagg_stopping")) as deployment:
        yield deployment


@pytest.fixture(scope="module")
def eos_bias(deployment) -> dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_path(deployment.model))
    assert tokenizer.eos_token_id is not None
    return {str(tokenizer.eos_token_id): EOS_BIAS}


def _sent(deployment) -> dict[str, int]:
    """Requests the orchestrator has sent to the context (``ctx``) and generation (``gen``) workers."""
    response = requests.get(f"{deployment.url}/prometheus/metrics", timeout=REQUEST_TIMEOUT_S)
    assert response.status_code == 200, response.text
    counts = {"ctx": 0, "gen": 0}
    for role, value in _SENT.findall(response.text):
        counts[role] += int(float(value))
    return counts


def _complete(deployment, **extra_body):
    return deployment.client().completions.create(
        model=deployment.model,
        prompt=PROMPT,
        max_tokens=MIN_TOKENS + 10,
        temperature=0.0,
        extra_body=extra_body,
    )


def test_first_token_eos_ends_on_the_context_leg(deployment, eos_bias):
    before = _sent(deployment)
    response = _complete(deployment, logit_bias=eos_bias)
    after = _sent(deployment)
    choice = response.choices[0]
    assert choice.finish_reason == "stop", response
    assert choice.text == "", response
    assert after["ctx"] == before["ctx"] + 1, (before, after)
    assert after["gen"] == before["gen"], (before, after)


def test_min_tokens_counts_across_both_legs(deployment, eos_bias):
    before = _sent(deployment)
    response = _complete(deployment, logit_bias=eos_bias, min_tokens=MIN_TOKENS)
    after = _sent(deployment)
    assert response.choices[0].finish_reason == "stop", response
    assert MIN_TOKENS <= response.usage.completion_tokens <= MIN_TOKENS + 1, response
    assert response.choices[0].text, response
    assert after["ctx"] == before["ctx"] + 1, (before, after)
    assert after["gen"] == before["gen"] + 1, (before, after)

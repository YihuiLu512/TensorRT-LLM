# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampling and stopping options of a disaggregated request (F-11 to F-17).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU), in-process or on the
OpenEngine chain. The context leg samples the first token and the generation
leg the rest. On the chain the tests also read the two legs'
GenerateRequests (``_test_openengine_disagg_common.legs``), which show an
option that one leg drops.

- NT-059 (F-11): ``top_k`` 1 and ``min_p`` 1 at temperature 1 are greedy; on
  the chain both legs get the request's ``temperature``, ``top_p``, ``top_k``
  and ``min_p``.
- NT-060 (F-12): a seed gives the same output again; on the chain both legs
  get the seed.
- NT-061 (F-13): each of the three penalties changes the output; on the
  chain both legs get it.
- NT-062 (F-15): ``min_tokens`` keeps the answer going past its end.
- NT-063 (F-17): a stop string that is the first token ends the request on
  the context leg with no text: whole, with ``finish_reason`` ``stop``;
  streamed, the orchestrator sends only the end of the stream, no chunk.
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

GREEDY_PROMPT = "The capital of France is"
REPEAT_PROMPT = "Write the word 'apple' again and again: apple apple apple"
SHORT_ANSWER = [{"role": "user", "content": "Reply with the single word OK."}]
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.fixture(scope="module")
def dump_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("disagg_sampling_dumps")


@pytest.fixture(scope="module")
def deployment(tmp_path_factory, dump_dir):
    with disagg_deployment(
        tmp_path_factory.mktemp("disagg_sampling"), env={DUMP_ENV: str(dump_dir)}
    ) as deployment:
        yield deployment


def _complete(deployment, prompt: str, **kwargs):
    kwargs.setdefault("max_tokens", 24)
    return deployment.client().completions.create(model=deployment.model, prompt=prompt, **kwargs)


def _text(deployment, prompt: str, **kwargs) -> str:
    return _complete(deployment, prompt, **kwargs).choices[0].text


def _sampling_of_both_legs(deployment, dump_dir: Path, **fields) -> dict:
    """On the chain: send a completion and return the sampling block both legs got."""
    before = dump_names(dump_dir)
    body = {"model": deployment.model, "prompt": GREEDY_PROMPT, "max_tokens": 8, **fields}
    response = requests.post(
        f"{deployment.url}/v1/completions", json=body, timeout=REQUEST_TIMEOUT_S
    )
    assert response.status_code == 200, response.text
    prefill, decode = legs(dump_dir, before)
    assert prefill.get("sampling") == decode.get("sampling"), (prefill, decode)
    return prefill.get("sampling", {})


def test_sampling_knobs_reach_both_legs(deployment, dump_dir):
    greedy = _text(deployment, GREEDY_PROMPT, temperature=0.0)
    assert greedy
    for knob in ({"top_k": 1}, {"min_p": 1.0}):
        assert _text(deployment, GREEDY_PROMPT, temperature=1.0, extra_body=knob) == greedy, knob
    if CHAIN:
        knobs = {"temperature": 0.8, "top_p": 0.9, "top_k": 5, "min_p": 0.5}
        sampling = _sampling_of_both_legs(deployment, dump_dir, **knobs)
        for name, value in knobs.items():
            assert float(sampling.get(name, "nan")) == pytest.approx(value), (name, sampling)


def test_seed_reproducible(deployment, dump_dir):
    outputs = {
        seed: [_text(deployment, GREEDY_PROMPT, temperature=1.0, seed=seed) for _ in range(2)]
        for seed in (1234, 4321)
    }
    for seed, (first, second) in outputs.items():
        assert first == second, seed
    if CHAIN:
        sampling = _sampling_of_both_legs(deployment, dump_dir, temperature=1.0, seed=1234)
        # Protobuf JSON writes the uint64 seed as a string.
        assert int(sampling.get("seed", -1)) == 1234, sampling


@pytest.mark.parametrize("penalty", ["repetition_penalty", "frequency_penalty", "presence_penalty"])
def test_penalties(deployment, dump_dir, penalty):
    plain = _text(deployment, REPEAT_PROMPT, temperature=0.0, max_tokens=64)
    penalized = _text(
        deployment, REPEAT_PROMPT, temperature=0.0, max_tokens=64, extra_body={penalty: 2.0}
    )
    assert penalized != plain
    if CHAIN:
        sampling = _sampling_of_both_legs(deployment, dump_dir, temperature=0.0, **{penalty: 2.0})
        assert float(sampling.get(penalty, "nan")) == pytest.approx(2.0), sampling


def test_min_tokens(deployment):
    client = deployment.client()
    plain = client.chat.completions.create(
        model=deployment.model,
        messages=SHORT_ANSWER,
        max_tokens=48,
        temperature=0.0,
        extra_body=NO_THINKING,
    )
    assert plain.choices[0].finish_reason == "stop"
    floor = plain.usage.completion_tokens + 8
    held = client.chat.completions.create(
        model=deployment.model,
        messages=SHORT_ANSWER,
        max_tokens=floor + 8,
        temperature=0.0,
        extra_body={**NO_THINKING, "min_tokens": floor},
    )
    assert held.usage.completion_tokens >= floor


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
def test_stop_word_on_first_token(deployment, stream):
    first = _complete(deployment, GREEDY_PROMPT, temperature=0.0, max_tokens=1)
    word = first.choices[0].text
    assert word.strip(), "the first token must have text to stop on"
    kwargs = dict(temperature=0.0, max_tokens=16, stop=[word])
    if stream:
        chunks = list(
            deployment.client().completions.create(
                model=deployment.model, prompt=GREEDY_PROMPT, stream=True, **kwargs
            )
        )
        # The context leg ended the request; the orchestrator has no chunk to relay.
        assert chunks == [], chunks
    else:
        choice = _complete(deployment, GREEDY_PROMPT, **kwargs).choices[0]
        assert choice.finish_reason == "stop"
        assert choice.text == "", choice.text

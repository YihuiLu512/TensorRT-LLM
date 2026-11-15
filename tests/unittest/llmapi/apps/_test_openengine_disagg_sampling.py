# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampling and stopping options of a disaggregated request (F-11 to F-17).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU), in-process or on the
OpenEngine chain. The context leg samples the first token and the generation
leg the rest, so an option that one leg drops changes the output.

- NT-059 (F-11): ``top_k`` 1 and ``min_p`` 1 at temperature 1 are greedy.
- NT-060 (F-12): a seed gives the same output again.
- NT-061 (F-13): a repetition penalty repeats less.
- NT-062 (F-15): ``min_tokens`` keeps the answer going past its end.
- NT-063 (F-17): a stop string that is the first token ends the output with
  nothing, as in the aggregated form.
"""

from collections import Counter

import pytest
from transformers import AutoTokenizer

from ._test_openengine_disagg_common import disagg_deployment, model_path

pytestmark = pytest.mark.threadleak(enabled=False)

GREEDY_PROMPT = "The capital of France is"
REPEAT_PROMPT = "Write the word 'apple' again and again: apple apple apple"
SHORT_ANSWER = [{"role": "user", "content": "Reply with the single word OK."}]
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    with disagg_deployment(tmp_path_factory.mktemp("disagg_sampling")) as deployment:
        yield deployment


@pytest.fixture(scope="module")
def tokenizer(deployment):
    return AutoTokenizer.from_pretrained(model_path(deployment.model))


def _complete(deployment, prompt: str, **kwargs):
    kwargs.setdefault("max_tokens", 24)
    return deployment.client().completions.create(model=deployment.model, prompt=prompt, **kwargs)


def _text(deployment, prompt: str, **kwargs) -> str:
    return _complete(deployment, prompt, **kwargs).choices[0].text


def test_sampling_knobs_reach_both_legs(deployment):
    greedy = _text(deployment, GREEDY_PROMPT, temperature=0.0)
    assert greedy
    for knob in ({"top_k": 1}, {"min_p": 1.0}):
        assert _text(deployment, GREEDY_PROMPT, temperature=1.0, extra_body=knob) == greedy, knob


def test_seed_reproducible(deployment):
    outputs = {
        seed: [_text(deployment, GREEDY_PROMPT, temperature=1.0, seed=seed) for _ in range(2)]
        for seed in (1234, 4321)
    }
    for seed, (first, second) in outputs.items():
        assert first == second, seed


def _repeated_trigrams(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    counts = Counter(zip(ids, ids[1:], ids[2:]))
    return sum(count - 1 for count in counts.values() if count > 1)


def test_penalties(deployment, tokenizer):
    plain = _text(deployment, REPEAT_PROMPT, temperature=0.0, max_tokens=64)
    penalized = _text(
        deployment,
        REPEAT_PROMPT,
        temperature=0.0,
        max_tokens=64,
        extra_body={"repetition_penalty": 2.0},
    )
    assert penalized != plain
    assert _repeated_trigrams(tokenizer, penalized) < _repeated_trigrams(tokenizer, plain)


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
def test_stop_word_on_first_token(deployment, tokenizer, stream):
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
        text = "".join(chunk.choices[0].text for chunk in chunks if chunk.choices)
        finish = [
            c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason
        ]
        assert finish == ["stop"], finish
    else:
        choice = _complete(deployment, GREEDY_PROMPT, **kwargs).choices[0]
        text = choice.text
        assert choice.finish_reason == "stop"
    assert text == "", text

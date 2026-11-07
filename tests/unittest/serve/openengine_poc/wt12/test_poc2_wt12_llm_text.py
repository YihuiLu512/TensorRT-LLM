# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""For the same token ids, the assembler's text is the text ``LLM`` gives.

``LLM`` detokenizes in ``DetokenizedGenerationResultBase._handle_response``
(``tensorrt_llm/executor/result.py``): ``tokenizer.decode`` of the output's
tokens when not streaming, ``decode_incrementally`` when streaming. Its engine
matches stop strings on their token ids and drops those tokens unless they are
kept (``GenerationResultBase._handle_sequence``); the text is then cut at the
first occurrence of the first stop string, in request order, that it holds,
whatever ended the output. :func:`_llm_output` restates those steps.
``min_tokens`` only bans single-token stop strings' tokens until the output is
that long, so for token ids holding no such token that early, ``LLM``'s output
does not depend on it.

The cases are built from pieces of text, so they contain the multi-byte
characters, special tokens and whitespace runs real outputs do.
"""

import os
import random
from pathlib import Path

import pytest

from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openengine_poc import response
from tensorrt_llm.serve.openengine_poc.engine_client import OutputFinished, TokenDelta
from tensorrt_llm.tokenizer import TransformersTokenizer

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

_PIECES = [
    "Hello",
    " world",
    "\n",
    "\n\n",
    "Answer:",
    " 42",
    "。",
    "你好",
    "<|im_end|>",
    "</think>",
    " stop",
    "STOP",
    "a",
    "b",
    "ab",
    " ",
    "😀",
    "x\n",
    "»",
    "naïve café",
]
_STOPS = [
    "\n\n",
    "Answer:",
    "。",
    "<|im_end|>",
    "</think>",
    "STOP",
    "b",
    "ab",
    "\n",
    " stop",
    "x\n",
    "😀",
]
_CASES = 400


@pytest.fixture(scope="module")
def tokenizer() -> TransformersTokenizer:
    from transformers import AutoTokenizer

    model_dir = Path(os.environ.get("LLM_MODELS_ROOT", "")) / "Qwen3" / "Qwen3-0.6B"
    if not (model_dir / "tokenizer.json").is_file():
        pytest.skip(f"no Qwen3-0.6B tokenizer at {model_dir}")
    return TransformersTokenizer(AutoTokenizer.from_pretrained(str(model_dir)))


def _cases(tokenizer):
    rng = random.Random(20261108)
    for _ in range(_CASES):
        token_ids = []
        for _ in range(rng.randint(1, 8)):
            token_ids += tokenizer.encode(rng.choice(_PIECES), add_special_tokens=False)
        yield (
            token_ids,
            rng.sample(_STOPS, rng.randint(1, 2)),
            rng.random() < 0.3,
            rng.random() < 0.7,
            [rng.randint(1, 3) for _ in token_ids],
        )


def _llm_output(tokenizer, token_ids, stop, include, skip):
    """What ``LLM`` ends with for an engine that generated ``token_ids``.

    Returns the tokens it keeps, its text, its ``stop_reason`` and the stop
    string its engine matched on token ids (None when the engine did not stop).
    """
    kept, engine_stop = list(token_ids), None
    words = [(s, tokenizer.encode(s, add_special_tokens=False)) for s in stop]
    for end in range(1, len(token_ids) + 1):
        for stop_str, word in words:
            if word and token_ids[max(0, end - len(word)) : end] == word:
                kept, engine_stop = (
                    token_ids[:end] if include else token_ids[: end - len(word)],
                    stop_str,
                )
                break
        if engine_stop is not None:
            break
    text, reason = tokenizer.decode(kept, skip_special_tokens=skip), engine_stop
    for stop_str in stop:
        position = text.find(stop_str)
        if position >= 0:
            text, reason = text[: position + (len(stop_str) if include else 0)], stop_str
            break
    return kept, text, reason, engine_stop


def _banned_before(tokenizer, token_ids, stop, min_tokens):
    """Whether a single-token stop string's token comes before ``min_tokens``.

    ``LLM`` could not have generated such token ids.
    """
    banned = set()
    for stop_str in stop:
        word = tokenizer.encode(stop_str, add_special_tokens=False)
        if len(word) == 1:
            banned.add(word[0])
    return any(token_id in banned for token_id in token_ids[:min_tokens])


def _assembled_text(tokenizer, params, token_ids, chunks, finished, *, engine_stop_texts):
    assembler = response.create_response_assembler(
        sampling_params=params,
        tokenizer=tokenizer,
        num_outputs=1,
        engine_stop_texts=engine_stop_texts,
    )
    streamed, start = "", 0
    for size in chunks:
        if assembler.finished or start >= len(token_ids):
            break
        if assembler.apply(
            TokenDelta(output_index=0, token_ids=tuple(token_ids[start : start + size]))
        ):
            streamed += assembler.outputs[0].text_diff
        start += size
    if not assembler.finished and assembler.apply(finished):
        streamed += assembler.outputs[0].text_diff
    output = assembler.outputs[0]
    assert streamed == output.text, "the streamed diffs add up to the text"
    return output


@pytest.mark.parametrize("skip_special_tokens", [True, False])
def test_text_without_stop_strings(tokenizer, skip_special_tokens):
    """Equal to LLM's non-streaming decode and to its streaming incremental decode."""
    for token_ids, _, _, _, chunks in _cases(tokenizer):
        params = SamplingParams(skip_special_tokens=skip_special_tokens)
        finished = OutputFinished(output_index=0, reason="length")
        output = _assembled_text(
            tokenizer, params, token_ids, chunks, finished, engine_stop_texts=True
        )

        expected = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        streamed, states, start = None, None, 0
        for size in chunks:
            if start >= len(token_ids):
                break
            streamed, states = tokenizer.decode_incrementally(
                token_ids[start : start + size],
                prev_text=streamed,
                states=states,
                flush=start + size >= len(token_ids),
                skip_special_tokens=skip_special_tokens,
            )
            start += size
        assert output.text == expected == streamed, token_ids


def test_engine_matched_stop_strings(tokenizer):
    """The engine (LLM's) sends the tokens it kept and how it stopped; the text is LLM's."""
    for token_ids, stop, include, skip, chunks in _cases(tokenizer):
        kept, text, reason, engine_stop = _llm_output(tokenizer, token_ids, stop, include, skip)
        params = SamplingParams(
            stop=stop, include_stop_str_in_output=include, skip_special_tokens=skip
        )
        finished = OutputFinished(
            output_index=0, reason="stop" if engine_stop else "length", stop_text=engine_stop
        )

        output = _assembled_text(tokenizer, params, kept, chunks, finished, engine_stop_texts=True)

        assert (output.text, output.token_ids, output.stop_reason, output.finish_reason) == (
            text,
            kept,
            reason,
            "stop" if reason else "length",
        ), (token_ids, stop, include)


def test_serving_side_stop_strings(tokenizer):
    """With one stop string, the serving side ends with LLM's text and stop string.

    With several, LLM picks the first one in request order that the text holds,
    while this one ends at the match that ends first (and so does LLM's
    streaming path when the matches complete in different steps); those cases
    are not compared. ``min_tokens`` is drawn too, for token ids LLM could have
    generated with it.
    """
    compared = 0
    rng = random.Random(7)
    for token_ids, stop, include, skip, chunks in _cases(tokenizer):
        min_tokens = rng.choice([None, 2, 5])
        if len(stop) != 1 or (
            min_tokens and _banned_before(tokenizer, token_ids, stop, min_tokens)
        ):
            continue
        compared += 1
        kept, text, reason, engine_stop = _llm_output(tokenizer, token_ids, stop, include, skip)
        params = SamplingParams(
            stop=stop,
            include_stop_str_in_output=include,
            skip_special_tokens=skip,
            min_tokens=min_tokens,
        )

        output = _assembled_text(
            tokenizer,
            params,
            token_ids,
            chunks,
            OutputFinished(output_index=0, reason="length"),
            engine_stop_texts=False,
        )

        assert (output.text, output.stop_reason) == (text, reason), (token_ids, stop, include)
        if reason is None:
            assert output.token_ids == token_ids
        elif engine_stop is not None:
            assert output.token_ids == kept
        else:
            # LLM's engine would have gone on: the output ends with the token whose text completed the match.
            count = len(output.token_ids)
            assert output.token_ids == token_ids[:count]
            assert reason in tokenizer.decode(token_ids[:count], skip_special_tokens=skip)
            assert reason not in tokenizer.decode(token_ids[: count - 1], skip_special_tokens=skip)
    assert compared > _CASES // 4

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the OpenEngine response assembler, driven by scripted engine streams."""

import asyncio

import pytest
from tokenizers import Regex, Tokenizer
from tokenizers.decoders import ByteFallback, Fuse, Sequence
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openengine_poc import contracts, response
from tensorrt_llm.serve.openengine_poc.engine_client import (
    EngineRequest,
    OutputFinished,
    PromptInfo,
    TokenDelta,
    TokenLogprob,
    Usage,
)
from tensorrt_llm.tokenizer import TransformersTokenizer

__extra_import_path__ = [".."]
from fakes import FakeEngineClient  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

_VOCAB = {
    "<0xC2>": 0,
    "<0xBB>": 1,
    "He": 2,
    "llo": 3,
    "</": 4,
    "ans": 5,
    ">": 6,
    " world": 7,
    "a": 8,
    "b": 9,
    "c": 10,
    "<unk>": 11,
    "</s>": 12,
    "\n": 13,
    "c.": 14,
}
C2, BB, HE, LLO, CLOSE, ANS, GT, WORLD, A, B, C, _, EOS, NEWLINE, C_DOT = range(15)


def _word_level(*, split: bool) -> TransformersTokenizer:
    backend = Tokenizer(WordLevel(_VOCAB, unk_token="<unk>"))
    backend.decoder = Sequence([ByteFallback(), Fuse()])
    backend.add_special_tokens(["</s>"])
    if split:
        backend.pre_tokenizer = Split(
            Regex(r"</|ans|>|He|llo| world|c\.|a|b|c|\n"), behavior="isolated"
        )
    return TransformersTokenizer(
        PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>")
    )


@pytest.fixture(scope="module")
def tokenizer() -> TransformersTokenizer:
    """Word-level pieces that decode by concatenation; ``<0xC2>`` ``<0xBB>`` make up ``»``.

    A stop string encodes to one token when it is one piece, else to ``<unk>``,
    so only single-piece stop strings match on their token ids.
    """
    return _word_level(split=False)


@pytest.fixture(scope="module")
def split_tokenizer() -> TransformersTokenizer:
    """The same vocabulary, encoding text piece by piece: ``</ans>`` is ``</`` ``ans`` ``>``."""
    return _word_level(split=True)


def _usage(cached: int | None = 3, completion: int = 0) -> Usage:
    return Usage(prompt_tokens=4, completion_tokens=completion, cached_prompt_tokens=cached)


def _assemble(tokenizer, sampling_params, steps, *, engine_stop_texts=False, num_outputs=None):
    """Play ``steps`` through a FakeEngineClient into a new assembler; returns it and the outputs it yielded.

    Stops reading once every output finished, as the facade does.
    """
    num_outputs = num_outputs or sampling_params.n
    assembler = response.create_response_assembler(
        sampling_params=sampling_params,
        tokenizer=tokenizer,
        num_outputs=num_outputs,
        engine_stop_texts=engine_stop_texts,
    )
    client = FakeEngineClient(script=lambda request: steps)
    request = EngineRequest(
        request_id="trtllm-1",
        prompt_token_ids=[1, 2, 3, 4],
        max_tokens=16,
        num_sequences=num_outputs,
    )
    text_diffs: list[list[str]] = []

    async def run():
        stream = client.generate(request)
        try:
            async for event in stream:
                if assembler.apply(event):
                    text_diffs.append([output.text_diff for output in assembler.outputs])
                if assembler.finished:
                    break
        finally:
            await stream.aclose()

    asyncio.run(run())
    return assembler, text_diffs


def test_the_slot_finds_this_module(tokenizer):
    contracts.reset_slot_cache()
    try:
        impl, source = contracts.resolve_slot("create_response_assembler")
        assert source == "tensorrt_llm.serve.openengine_poc.response:create_response_assembler"
        assembler = contracts.create_response_assembler(
            sampling_params=SamplingParams(), tokenizer=tokenizer, num_outputs=2
        )
        assert isinstance(assembler, response.OpenEngineResponseAssembler)
        assert len(assembler.outputs) == 2
    finally:
        contracts.reset_slot_cache()


# -- The engine matches the stop strings -------------------------------------------


@pytest.mark.parametrize(
    "include, tokens, expected_text",
    [
        pytest.param(False, [HE, LLO], "Hello", id="stop_tokens_left_out_by_the_engine"),
        pytest.param(True, [HE, LLO, CLOSE, ANS, GT], "Hello</ans>", id="kept"),
    ],
)
def test_engine_stop_text_sets_stop_reason_and_usage(tokenizer, include, tokens, expected_text):
    params = SamplingParams(stop=["</ans>"], include_stop_str_in_output=include)
    steps = [
        TokenDelta(output_index=0, token_ids=tuple(tokens)),
        OutputFinished(output_index=0, reason="stop", stop_text="</ans>", usage=_usage(cached=7)),
    ]

    assembler, _ = _assemble(tokenizer, params, steps, engine_stop_texts=True)

    output = assembler.outputs[0]
    assert (output.text, output.finish_reason, output.stop_reason) == (
        expected_text,
        "stop",
        "</ans>",
    )
    assert output.token_ids == tokens
    assert assembler.cached_tokens == 7


def test_engine_stop_text_inside_a_token_cut_as_llm_does(tokenizer):
    """The engine keeps ``c.`` (it ends with the stop string); LLM cuts the text at the ``.``."""
    params = SamplingParams(stop=["."])
    steps = [
        TokenDelta(output_index=0, token_ids=(A, C_DOT)),
        OutputFinished(output_index=0, reason="stop", stop_text=".", usage=_usage()),
    ]

    assembler, diffs = _assemble(tokenizer, params, steps, engine_stop_texts=True)

    output = assembler.outputs[0]
    assert output.text == "ac"
    assert "".join(diff[0] for diff in diffs) == "ac"
    assert output.token_ids == [A, C_DOT]


def test_engine_stop_does_not_end_the_output_early(tokenizer):
    """A stop string in the text does not end the output: the engine's last event, with usage, is read."""
    params = SamplingParams(stop=["b"])
    steps = [
        TokenDelta(output_index=0, token_ids=(A, B)),
        TokenDelta(output_index=0, token_ids=(C,)),
        OutputFinished(output_index=0, reason="stop", stop_text="b", usage=_usage(cached=5)),
    ]

    assembler, diffs = _assemble(tokenizer, params, steps, engine_stop_texts=True)

    assert len(diffs) == 3
    assert assembler.outputs[0].text == "a"
    assert assembler.cached_tokens == 5


def test_engine_stop_cuts_the_text_whatever_ended_the_output(tokenizer):
    """As LLM: an output that ends by length with a stop string in its text ends there, with ``stop``."""
    params = SamplingParams(stop=["b"])
    steps = [
        TokenDelta(output_index=0, token_ids=(A, B, C)),
        OutputFinished(output_index=0, reason="length", usage=_usage()),
    ]

    assembler, diffs = _assemble(tokenizer, params, steps, engine_stop_texts=True)

    output = assembler.outputs[0]
    assert (output.text, output.finish_reason, output.stop_reason) == ("a", "stop", "b")
    assert "".join(diff[0] for diff in diffs) == "a"
    assert output.token_ids == [A, B, C]


def test_engine_stop_cancelled_by_the_serving_side(tokenizer):
    """A cancelled output stays cancelled; its text is cut at the stop string all the same."""
    assembler = response.create_response_assembler(
        sampling_params=SamplingParams(stop=["b"]),
        tokenizer=tokenizer,
        num_outputs=1,
        engine_stop_texts=True,
    )
    assembler.apply(TokenDelta(output_index=0, token_ids=(A, B, C)))

    assembler.finish_cancelled()

    output = assembler.outputs[0]
    assert (output.text, output.finish_reason, output.stop_reason) == ("a", "cancelled", None)


@pytest.mark.parametrize(
    "finished, expected",
    [
        pytest.param(
            OutputFinished(output_index=0, reason="stop", eos_token_id=EOS),
            ("stop", None),
            id="eos",
        ),
        pytest.param(
            OutputFinished(output_index=0, reason="stop", stop_token_id=GT),
            ("stop", GT),
            id="stop_token",
        ),
        pytest.param(
            OutputFinished(output_index=0, reason="length"), ("length", None), id="length"
        ),
        pytest.param(
            OutputFinished(output_index=0, reason="cancelled"), ("cancelled", None), id="cancelled"
        ),
    ],
)
def test_finish_reasons(tokenizer, finished, expected):
    steps = [TokenDelta(output_index=0, token_ids=(HE, LLO)), finished]

    assembler, _ = _assemble(tokenizer, SamplingParams(), steps, engine_stop_texts=True)

    output = assembler.outputs[0]
    assert (output.finish_reason, output.stop_reason) == expected
    assert output.text == "Hello"


# -- The serving side matches the stop strings (fallback) --------------------------


def test_fallback_stop_ends_the_output_and_counts_tokens_up_to_it(tokenizer):
    """Tokens past the one completing the match are not part of the output (G-4)."""
    params = SamplingParams(stop=["</ans>"], logprobs=0)
    tokens = (HE, LLO, CLOSE, ANS, GT, WORLD, WORLD)
    steps = [
        TokenDelta(
            output_index=0,
            token_ids=tokens,
            logprobs=tuple(TokenLogprob(token_id=t, logprob=-0.5, rank=1) for t in tokens),
        ),
        OutputFinished(output_index=0, reason="length", usage=_usage()),
    ]

    assembler, _ = _assemble(tokenizer, params, steps)

    output = assembler.outputs[0]
    assert (output.text, output.finish_reason, output.stop_reason) == ("Hello", "stop", "</ans>")
    assert output.token_ids == [HE, LLO, CLOSE, ANS, GT]
    assert len(output.logprobs) == len(output.token_ids)
    assert assembler.finished
    assert assembler.cached_tokens == 0


@pytest.mark.parametrize(
    "include, expected_tokens, expected_text",
    [
        pytest.param(False, [A, C], "ac", id="stop_tokens_dropped"),
        pytest.param(True, [A, C, EOS], "ac", id="stop_tokens_kept"),
    ],
)
def test_fallback_stop_string_of_special_tokens(tokenizer, include, expected_tokens, expected_text):
    """``</s>`` matches on its token id though ``skip_special_tokens`` leaves it out of the text."""
    params = SamplingParams(stop=["</s>"], include_stop_str_in_output=include, logprobs=1)
    tokens = (A, C, EOS, B)
    steps = [
        TokenDelta(
            output_index=0,
            token_ids=tokens,
            logprobs=tuple(
                TokenLogprob(token_id=t, logprob=-1.0, rank=1, top=((t, -1.0),)) for t in tokens
            ),
        ),
        OutputFinished(output_index=0, reason="length", usage=_usage()),
    ]

    assembler, _ = _assemble(tokenizer, params, steps)

    output = assembler.outputs[0]
    assert (output.text, output.stop_reason) == (expected_text, "</s>")
    assert output.token_ids == expected_tokens
    assert [list(entry) for entry in output.logprobs] == [[t] for t in expected_tokens]


def test_fallback_stop_spanning_events_trims_earlier_tokens(split_tokenizer):
    """The stop tokens may have arrived in earlier events; they are all left out."""
    params = SamplingParams(stop=["</ans>"], logprobs=0)
    tokens = (HE, CLOSE, ANS, GT)
    steps = [
        TokenDelta(
            output_index=0,
            token_ids=(token_id,),
            logprobs=(TokenLogprob(token_id=token_id, logprob=-0.5, rank=1),),
        )
        for token_id in tokens
    ] + [OutputFinished(output_index=0, reason="length", usage=_usage())]

    assembler, diffs = _assemble(split_tokenizer, params, steps)

    output = assembler.outputs[0]
    assert "".join(diff[0] for diff in diffs) == output.text == "He"
    assert (output.token_ids, output.stop_reason) == ([HE], "</ans>")
    assert len(output.logprobs) == 1
    assert output.token_ids_diff == []


@pytest.mark.parametrize(
    "last, expected_diffs, expected_tokens",
    [
        pytest.param(B, ["a", ""], [A], id="stop_tokens_follow"),
        pytest.param(C, ["a", "\nc"], [A, EOS, NEWLINE, C], id="they_do_not"),
    ],
)
def test_fallback_holds_back_tokens_that_may_start_stop_tokens(
    split_tokenizer, last, expected_diffs, expected_tokens
):
    """``</s>`` is left out of the text, so only the token ids show that the newline may be a stop's."""
    params = SamplingParams(stop=["</s>\nb"])
    steps = [
        TokenDelta(output_index=0, token_ids=(A, EOS, NEWLINE)),
        TokenDelta(output_index=0, token_ids=(last,)),
        OutputFinished(output_index=0, reason="length", usage=_usage()),
    ]

    assembler, diffs = _assemble(split_tokenizer, params, steps)

    assert ["".join(diff) for diff in diffs][:2] == expected_diffs
    assert assembler.outputs[0].token_ids == expected_tokens


@pytest.mark.parametrize(
    "stop, tokens, expected",
    [
        pytest.param(["b"], (A, B, C, B, A), ("abc", [A, B, C], "b"), id="single_token"),
        pytest.param(["</ans>"], (HE, CLOSE, ANS, GT, A), ("He", [HE], "</ans>"), id="multi_token"),
    ],
)
def test_fallback_min_tokens(split_tokenizer, stop, tokens, expected):
    """As LLM: only a single-token stop string waits for ``min_tokens``."""
    params = SamplingParams(stop=stop, min_tokens=3)
    steps = [
        TokenDelta(output_index=0, token_ids=tokens),
        OutputFinished(output_index=0, reason="length", usage=_usage()),
    ]

    assembler, _ = _assemble(split_tokenizer, params, steps)

    output = assembler.outputs[0]
    assert (output.text, output.token_ids, output.stop_reason) == expected


def test_fallback_without_detokenizing_matches_token_ids(tokenizer):
    params = SamplingParams(stop=["b"], detokenize=False)
    steps = [
        TokenDelta(output_index=0, token_ids=(A, C, B, C)),
        OutputFinished(output_index=0, reason="length", usage=_usage()),
    ]

    assembler, _ = _assemble(tokenizer, params, steps)

    output = assembler.outputs[0]
    assert (output.text, output.token_ids, output.stop_reason) == ("", [A, C], "b")


def test_cancelled_by_the_serving_side(tokenizer):
    params = SamplingParams(stop=["</ans>"], n=2)
    assembler = response.create_response_assembler(
        sampling_params=params, tokenizer=tokenizer, num_outputs=2
    )
    assembler.apply(TokenDelta(output_index=0, token_ids=(HE, CLOSE, ANS)))
    assembler.apply(TokenDelta(output_index=1, token_ids=(HE, CLOSE, ANS, GT)))

    assembler.finish_cancelled()

    first, second = assembler.outputs
    assert (first.text, first.finish_reason) == ("He</ans", "cancelled")
    assert first.text_diff == "</ans"
    assert (second.text, second.finish_reason, second.stop_reason) == ("He", "stop", "</ans>")


# -- Several outputs, usage --------------------------------------------------------


def test_outputs_by_index_and_the_first_usage(tokenizer):
    params = SamplingParams(n=2)
    steps = [
        TokenDelta(output_index=1, token_ids=(A,)),
        TokenDelta(output_index=0, token_ids=(B,)),
        OutputFinished(output_index=1, reason="length", usage=_usage(cached=4)),
        TokenDelta(output_index=0, token_ids=(C,)),
        OutputFinished(output_index=0, reason="length", usage=_usage(cached=9)),
    ]

    assembler, diffs = _assemble(tokenizer, params, steps)

    assert [output.text for output in assembler.outputs] == ["bc", "a"]
    assert diffs[0] == ["", "a"]
    assert assembler.cached_tokens == 4


def test_no_cached_count_reads_as_zero(tokenizer):
    steps = [
        TokenDelta(output_index=0, token_ids=(A,)),
        OutputFinished(output_index=0, reason="length", usage=_usage(cached=None)),
    ]

    assembler, _ = _assemble(tokenizer, SamplingParams(), steps)

    assert assembler.cached_tokens == 0


# -- Logprobs ------------------------------------------------------------------------


def test_output_logprobs_put_the_sampled_token_first(tokenizer):
    """As LLM's sampler builds them: the sampled token, then the top candidates by rank."""
    params = SamplingParams(logprobs=2)
    steps = [
        TokenDelta(
            output_index=0,
            token_ids=(A, B),
            logprobs=(
                TokenLogprob(token_id=A, logprob=-0.1, rank=1, top=((A, -0.1), (B, -2.0))),
                TokenLogprob(token_id=B, logprob=-3.0, rank=5, top=((C, -0.2), (A, -1.0))),
            ),
        ),
        OutputFinished(output_index=0, reason="length", usage=_usage()),
    ]

    assembler, _ = _assemble(tokenizer, params, steps)

    first, second = assembler.outputs[0].logprobs
    assert list(first.items()) == [(A, Logprob(-0.1, 1)), (B, Logprob(-2.0, 2))]
    assert list(second.items()) == [
        (B, Logprob(-3.0, 5)),
        (C, Logprob(-0.2, 1)),
        (A, Logprob(-1.0, 2)),
    ]


def test_output_logprobs_of_the_sampled_token_only(tokenizer):
    steps = [
        TokenDelta(
            output_index=0,
            token_ids=(A,),
            logprobs=(TokenLogprob(token_id=A, logprob=-0.3, rank=2),),
        ),
        OutputFinished(output_index=0, reason="length", usage=_usage()),
    ]

    dict_format, _ = _assemble(tokenizer, SamplingParams(logprobs=0), steps)
    simple_format, _ = _assemble(
        tokenizer, SamplingParams(logprobs=0, logprobs_simple_format=True), steps
    )

    assert dict_format.outputs[0].logprobs == [{A: Logprob(-0.3, 2)}]
    assert simple_format.outputs[0].logprobs == [-0.3]


def test_no_logprobs_unless_requested(tokenizer):
    steps = [
        TokenDelta(
            output_index=0, token_ids=(A,), logprobs=(TokenLogprob(token_id=A, logprob=-0.3),)
        ),
        OutputFinished(output_index=0, reason="length", usage=_usage()),
    ]

    assembler, _ = _assemble(tokenizer, SamplingParams(), steps)

    assert assembler.outputs[0].logprobs == []


def test_prompt_logprobs(tokenizer):
    """LLM's prompt logprobs start at the second prompt token, top candidates first."""
    params = SamplingParams(prompt_logprobs=1, n=2)
    steps = [
        PromptInfo(
            token_logprobs=(
                TokenLogprob(token_id=HE, logprob=0.0),
                TokenLogprob(token_id=LLO, logprob=-0.5, rank=1, top=((LLO, -0.5),)),
                TokenLogprob(token_id=GT, logprob=-4.0, rank=3, top=((A, -0.1),)),
            )
        ),
        TokenDelta(output_index=0, token_ids=(A,)),
        TokenDelta(output_index=1, token_ids=(B,)),
        OutputFinished(output_index=0, reason="length"),
        OutputFinished(output_index=1, reason="length", usage=_usage()),
    ]

    assembler, _ = _assemble(tokenizer, params, steps)

    for output in assembler.outputs:
        assert [list(entry.items()) for entry in output.prompt_logprobs] == [
            [(LLO, Logprob(-0.5, 1))],
            [(A, Logprob(-0.1, 1)), (GT, Logprob(-4.0, 3))],
        ]


# -- Arguments -----------------------------------------------------------------------


def test_tokenizer_checked_when_the_assembler_is_created():
    with pytest.raises(TypeError, match="TransformersTokenizer"):
        response.create_response_assembler(
            sampling_params=SamplingParams(), tokenizer=object(), num_outputs=1
        )


def test_no_tokenizer_needed_without_text_or_serving_side_stop():
    params = SamplingParams(detokenize=False, stop=["b"])

    assembler = response.create_response_assembler(
        sampling_params=params, tokenizer=None, num_outputs=1, engine_stop_texts=True
    )

    assert assembler.apply(TokenDelta(output_index=0, token_ids=(A, B)))
    assert assembler.outputs[0].token_ids == [A, B]


def test_empty_stop_string_rejected(tokenizer):
    with pytest.raises(ValueError, match="must not be empty"):
        response.create_response_assembler(
            sampling_params=SamplingParams(stop=["", "b"]), tokenizer=tokenizer, num_outputs=1
        )

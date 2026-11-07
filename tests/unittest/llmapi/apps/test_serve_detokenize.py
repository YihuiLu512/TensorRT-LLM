# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the serving-side incremental detokenizer and stop-string matcher."""

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback, Fuse, Metaspace, Sequence
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from tensorrt_llm.serve.detokenize import DetokenizeDelta, IncrementalDetokenizer, StopMatch
from tensorrt_llm.tokenizer import TransformersTokenizer

# Pure-Python tokenizer checks with no GPU dependency. The l0_cpu stage runs pytest
# with ``-m cpu_only`` and conftest drops any file lacking this marker.
pytestmark = pytest.mark.cpu_only

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
}
C2, BB, HE, LLO, CLOSE, ANS, GT, WORLD, A, B, C, _, EOS, NEWLINE = range(14)


@pytest.fixture(scope="module")
def tokenizer() -> TransformersTokenizer:
    """Word-level pieces that decode by concatenation; ``<0xC2>`` ``<0xBB>`` make up ``»``."""
    backend = Tokenizer(WordLevel(_VOCAB, unk_token="<unk>"))
    backend.decoder = Sequence([ByteFallback(), Fuse()])
    backend.add_special_tokens(["</s>"])
    return TransformersTokenizer(
        PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>")
    )


@pytest.fixture(params=["decode_stream", "fallback"], autouse=True)
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Runs every test with ``DecodeStream`` and with the ``decode_incrementally`` fallback.

    The fallback is what a tokenizer without a ``tokenizers`` backend gets; here it is forced by
    hiding ``DecodeStream`` and selecting the TRT-LLM incremental detokenizer.
    """
    if request.param == "fallback":
        from tensorrt_llm.serve import detokenize as detokenize_module
        from tensorrt_llm.tokenizer import tokenizer as tokenizer_module

        monkeypatch.setattr(detokenize_module, "DecodeStream", None)
        monkeypatch.setattr(tokenizer_module, "TLLM_INCREMENTAL_DETOKENIZATION_BACKEND", "TRTLLM")
    return request.param


def _run(detokenizer: IncrementalDetokenizer, chunks: list[list[int]]) -> list[str]:
    """Feeds the chunks, then finishes, and returns the text released by each call."""
    deltas = [detokenizer.step(chunk).text for chunk in chunks]
    deltas.append(detokenizer.finish().text)
    return deltas


@pytest.mark.parametrize("skip_special_tokens", [True, False])
@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param([[HE], [LLO], [WORLD], [C2], [BB], [EOS]], id="one_token_per_step"),
        pytest.param([[HE, LLO, WORLD, C2, BB, EOS]], id="one_step"),
        pytest.param([[HE, LLO], [], [WORLD, C2], [BB, EOS]], id="mixed_steps"),
    ],
)
def test_released_text_matches_full_decode(
    tokenizer: TransformersTokenizer, chunks: list[list[int]], skip_special_tokens: bool
) -> None:
    """Holds for this tokenizer, which does not clean up tokenization spaces."""
    detokenizer = IncrementalDetokenizer(tokenizer, skip_special_tokens=skip_special_tokens)

    text = "".join(_run(detokenizer, chunks))

    token_ids = [token_id for chunk in chunks for token_id in chunk]
    assert text == tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    assert detokenizer.text == text
    assert detokenizer.stop_match is None
    assert detokenizer.finished


def test_multibyte_character_released_once_complete(tokenizer: TransformersTokenizer) -> None:
    detokenizer = IncrementalDetokenizer(tokenizer)

    assert detokenizer.step([A]).text == "a"
    assert detokenizer.step([C2]).text == ""
    assert detokenizer.step([BB]).text == "»"


def test_incomplete_multibyte_character_flushed_on_finish(tokenizer: TransformersTokenizer) -> None:
    detokenizer = IncrementalDetokenizer(tokenizer)

    assert _run(detokenizer, [[A], [C2]]) == ["a", "", "\N{REPLACEMENT CHARACTER}"]


def test_stop_string_split_across_steps(tokenizer: TransformersTokenizer) -> None:
    detokenizer = IncrementalDetokenizer(tokenizer, stop="</ans>")

    assert detokenizer.step([HE, LLO, CLOSE]).text == "Hello"
    assert detokenizer.step([ANS]).text == ""
    delta = detokenizer.step([GT])

    assert delta.text == ""
    assert delta.stop == StopMatch(stop_str="</ans>", token_index=4)
    assert detokenizer.text == "Hello"
    assert detokenizer.finished


@pytest.mark.parametrize(
    "include_stop_str_in_output, expected_text",
    [
        pytest.param(False, "Hello", id="stop_str_removed"),
        pytest.param(True, "Hello</ans>", id="stop_str_kept"),
    ],
)
def test_text_truncated_at_stop_string(
    tokenizer: TransformersTokenizer, include_stop_str_in_output: bool, expected_text: str
) -> None:
    """Tokens generated past the stop string do not reach the text."""
    detokenizer = IncrementalDetokenizer(
        tokenizer, stop=["</ans>"], include_stop_str_in_output=include_stop_str_in_output
    )

    delta = detokenizer.step([HE, LLO, CLOSE, ANS, GT, WORLD, WORLD])

    assert delta.text == expected_text
    assert delta.stop == StopMatch(stop_str="</ans>", token_index=4)


def test_stop_string_completed_by_second_byte_of_character(
    tokenizer: TransformersTokenizer,
) -> None:
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["»"])

    assert _run(detokenizer, [[A, C2], [BB, B]]) == ["a", "", ""]
    assert detokenizer.stop_match == StopMatch(stop_str="»", token_index=2)


@pytest.mark.parametrize(
    "stop, chunks, expected_deltas, expected_match",
    [
        pytest.param(
            ["aab"],
            [[A], [A], [A], [B]],
            ["", "", "a", "", ""],
            StopMatch("aab", 3),
            id="stop_string_overlapping_itself",
        ),
        pytest.param(
            ["abc", "b"],
            [[A, B, C]],
            ["a", ""],
            StopMatch("b", 1, num_stop_tokens=1),
            id="match_that_ends_first_wins",
        ),
        pytest.param(
            ["abc", "bc"],
            [[A], [B], [C]],
            ["", "", "", ""],
            StopMatch("abc", 2),
            id="same_end_earlier_start_wins",
        ),
        pytest.param(
            ["bc", "abc"],
            [[A], [B], [C]],
            ["", "", "", ""],
            StopMatch("abc", 2),
            id="same_end_earlier_start_wins_regardless_of_order",
        ),
        pytest.param(
            ["ab", "bc", "cc"],
            [[A], [C], [C]],
            ["", "a", "", ""],
            StopMatch("cc", 2),
            id="hold_back_follows_longest_prefix",
        ),
        pytest.param(
            ["abc", "cab"],
            [[A, B], [A], [B], [A]],
            ["", "ab", "", "ab", "a"],
            None,
            id="held_back_text_released_on_finish",
        ),
    ],
)
def test_several_stop_strings(
    tokenizer: TransformersTokenizer,
    stop: list[str],
    chunks: list[list[int]],
    expected_deltas: list[str],
    expected_match: StopMatch,
) -> None:
    detokenizer = IncrementalDetokenizer(tokenizer, stop=stop)

    assert _run(detokenizer, chunks) == expected_deltas
    assert detokenizer.stop_match == expected_match


@pytest.mark.parametrize("split", range(9))
def test_match_does_not_depend_on_chunking(tokenizer: TransformersTokenizer, split: int) -> None:
    """``Helloabc</ans>`` stops at ``bc`` (completed by token 4) wherever the ids are split."""
    token_ids = [HE, LLO, A, B, C, CLOSE, ANS, GT]
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["</ans>", "bc"])

    text = "".join(_run(detokenizer, [token_ids[:split], token_ids[split:]]))

    assert text == "Helloa"
    assert detokenizer.stop_match == StopMatch("bc", 4)


def test_calls_after_stop_release_nothing(tokenizer: TransformersTokenizer) -> None:
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["ll"])
    detokenizer.step([HE, LLO])

    assert detokenizer.step([WORLD]) == DetokenizeDelta("")
    assert detokenizer.finish() == DetokenizeDelta("")
    assert detokenizer.text == "He"
    assert detokenizer.stop_match == StopMatch("ll", 1)


@pytest.mark.parametrize("spaces_between_special_tokens", [True, False])
@pytest.mark.parametrize(
    "stop, token_ids, expected_text, expected_index",
    [
        pytest.param("a\n", [C, A, NEWLINE, B], "c", 2, id="one_newline"),
        pytest.param("a\n\n", [C, A, NEWLINE, NEWLINE, B], "c", 3, id="two_newlines"),
        pytest.param("a\n\n", [C, A, NEWLINE, B, NEWLINE], "ca\nb\n", None, id="no_match"),
    ],
)
def test_stop_string_ending_in_whitespace(
    tokenizer: TransformersTokenizer,
    spaces_between_special_tokens: bool,
    stop: str,
    token_ids: list[int],
    expected_text: str,
    expected_index: int | None,
) -> None:
    """Found with the token that adds the whitespace, on both decoders.

    The fallback decoder holds back text that only adds whitespace; text that
    completes a stop string is released anyway. The stop strings are not in
    the vocabulary, so they are found in the text. The setting that makes
    ``decode_incrementally`` pick its fallback for a tokenizer does not change
    which decoder this component uses.
    """
    detokenizer = IncrementalDetokenizer(
        tokenizer, stop=[stop], spaces_between_special_tokens=spaces_between_special_tokens
    )

    text = "".join(_run(detokenizer, [[token_id] for token_id in token_ids]))

    assert text == expected_text
    expected = None if expected_index is None else StopMatch(stop, expected_index)
    assert detokenizer.stop_match == expected


def test_kept_stop_string_is_not_held_back(tokenizer: TransformersTokenizer) -> None:
    """With the stop string kept in the output, a possible prefix is safe to release."""
    detokenizer = IncrementalDetokenizer(
        tokenizer, stop=["</ans>"], include_stop_str_in_output=True
    )

    assert detokenizer.step([HE, LLO, CLOSE]).text == "Hello</"
    assert detokenizer.step([ANS]).text == "ans"
    assert detokenizer.step([GT]) == DetokenizeDelta(">", StopMatch("</ans>", 4))
    assert detokenizer.text == "Hello</ans>"


@pytest.mark.parametrize(
    "token_ids",
    [
        pytest.param([0, 1, 2], id="words"),
        pytest.param([3, 0, 3, 1], id="lone_space_pieces"),
    ],
)
def test_sentencepiece_style_text_matches_full_decode(token_ids: list[int]) -> None:
    """A Metaspace decoder drops the leading space of the first piece only."""
    backend = Tokenizer(
        WordLevel(
            {"\u2581Hello": 0, "\u2581world": 1, "!": 2, "\u2581": 3, "<unk>": 4}, unk_token="<unk>"
        )
    )
    backend.decoder = Metaspace()
    tokenizer = TransformersTokenizer(PreTrainedTokenizerFast(tokenizer_object=backend))
    detokenizer = IncrementalDetokenizer(tokenizer)

    text = "".join(_run(detokenizer, [[token_id] for token_id in token_ids]))

    assert text == tokenizer.decode(token_ids)


def test_stop_string_completed_by_flushed_text(tokenizer: TransformersTokenizer) -> None:
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["\N{REPLACEMENT CHARACTER}"])

    assert _run(detokenizer, [[A, C2]]) == ["a", ""]
    assert detokenizer.stop_match == StopMatch("\N{REPLACEMENT CHARACTER}", 1)


def test_finish_without_tokens(tokenizer: TransformersTokenizer) -> None:
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["a"])

    assert detokenizer.finish().text == ""
    assert detokenizer.finished


def test_empty_stop_string_rejected(tokenizer: TransformersTokenizer) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        IncrementalDetokenizer(tokenizer, stop=["", "b"])


def test_single_stop_string_is_not_split_into_characters(
    tokenizer: TransformersTokenizer,
) -> None:
    detokenizer = IncrementalDetokenizer(tokenizer, stop="ab")

    assert _run(detokenizer, [[A, C, B]]) == ["acb", ""]
    assert detokenizer.stop_match is None


def test_non_str_stop_string_rejected(tokenizer: TransformersTokenizer) -> None:
    with pytest.raises(TypeError, match="stop strings must be str"):
        IncrementalDetokenizer(tokenizer, stop=["a", 1])


def test_tokenizer_of_another_type_rejected() -> None:
    with pytest.raises(TypeError, match="TransformersTokenizer"):
        IncrementalDetokenizer(object())


# -- Matching on the stop string's token ids, as LLM's engine does --------------


@pytest.mark.parametrize("skip_special_tokens", [True, False])
def test_stop_string_of_special_tokens_matched_on_token_ids(
    tokenizer: TransformersTokenizer, skip_special_tokens: bool
) -> None:
    """``</s>`` is found even when special tokens are left out of the text."""
    detokenizer = IncrementalDetokenizer(
        tokenizer, stop=["</s>"], skip_special_tokens=skip_special_tokens
    )

    deltas = _run(detokenizer, [[HE, LLO], [EOS, WORLD]])

    assert "".join(deltas) == "Hello"
    assert detokenizer.stop_match == StopMatch("</s>", 2, num_stop_tokens=1)


@pytest.mark.parametrize("skip_special_tokens", [True, False])
def test_kept_special_token_stop_string(
    tokenizer: TransformersTokenizer, skip_special_tokens: bool
) -> None:
    """Kept in the output, the stop tokens stay and their text is what decode gives."""
    detokenizer = IncrementalDetokenizer(
        tokenizer,
        stop=["</s>"],
        include_stop_str_in_output=True,
        skip_special_tokens=skip_special_tokens,
    )

    text = "".join(_run(detokenizer, [[HE, LLO, EOS, WORLD]]))

    assert text == tokenizer.decode([HE, LLO, EOS], skip_special_tokens=skip_special_tokens)
    assert detokenizer.stop_match == StopMatch("</s>", 2, num_stop_tokens=1)


def test_token_ids_match_before_text_match_on_the_same_token(
    tokenizer: TransformersTokenizer,
) -> None:
    """``b`` is a token and ends ``ab`` in the text; the token-id match wins, as in LLM.

    LLM's engine drops the ``b`` token, and the text of the rest no longer holds ``ab``.
    """
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["ab", "b"])

    assert "".join(_run(detokenizer, [[C, A, B]])) == "ca"
    assert detokenizer.stop_match == StopMatch("b", 2, num_stop_tokens=1)


def test_token_ids_match_without_detokenizing(tokenizer: TransformersTokenizer) -> None:
    """Without text, stop strings are still matched on their token ids, as in LLM."""
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["b", "</ans>"], detokenize=False)

    assert detokenizer.step([A, C]) == DetokenizeDelta("")
    assert detokenizer.step([B, C]) == DetokenizeDelta("", StopMatch("b", 2, num_stop_tokens=1))
    assert detokenizer.text == ""


# -- min_tokens --------------------------------------------------------------------


@pytest.mark.parametrize(
    "stop, chunks, min_tokens, expected_text, expected_match",
    [
        pytest.param(["b"], [[A, B, C, B]], 3, "abc", StopMatch("b", 3, 1), id="token_ids"),
        pytest.param(["b"], [[A, B, C, B]], 2, "a", StopMatch("b", 1, 1), id="token_ids_at_min"),
        pytest.param(["ab"], [[A, B, C, A, B]], 3, "abc", StopMatch("ab", 4), id="text"),
        pytest.param(["ab"], [[A, B], [C]], 3, "abc", None, id="text_never_counts"),
        pytest.param(["aab"], [[A, A], [A, B]], 3, "a", StopMatch("aab", 3), id="overlapping"),
    ],
)
def test_matches_before_min_tokens_do_not_count(
    tokenizer: TransformersTokenizer,
    stop: list[str],
    chunks: list[list[int]],
    min_tokens: int,
    expected_text: str,
    expected_match: StopMatch | None,
) -> None:
    detokenizer = IncrementalDetokenizer(tokenizer, stop=stop, min_tokens=min_tokens)

    assert "".join(_run(detokenizer, chunks)) == expected_text
    assert detokenizer.stop_match == expected_match


# -- engine_stop: the engine matches the stop strings ------------------------------


def test_engine_stop_never_ends_the_output(tokenizer: TransformersTokenizer) -> None:
    """The engine decides; the text is held back from the match on, then cut at finish."""
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["</ans>"], engine_stop=True)

    assert detokenizer.step([HE, LLO, CLOSE]) == DetokenizeDelta("Hello")
    assert detokenizer.step([ANS, GT, WORLD]) == DetokenizeDelta("")
    assert not detokenizer.finished
    assert detokenizer.finish(stop_text="</ans>") == DetokenizeDelta("")
    assert detokenizer.text == "Hello"
    assert detokenizer.stop_match is None


@pytest.mark.parametrize(
    "include_stop_str_in_output, stop_text, expected_text",
    [
        pytest.param(False, "b", "a", id="cut_at_first_occurrence"),
        pytest.param(True, "b", "ab", id="kept"),
        pytest.param(False, "</ans>", "abcab", id="not_in_text"),
        pytest.param(False, None, "abcab", id="no_stop_reported"),
    ],
)
def test_engine_stop_cuts_as_llm_does(
    tokenizer: TransformersTokenizer,
    include_stop_str_in_output: bool,
    stop_text: str | None,
    expected_text: str,
) -> None:
    """``LLM`` cuts the text at the first occurrence of the stop string it matched."""
    detokenizer = IncrementalDetokenizer(
        tokenizer,
        stop=["b"],
        include_stop_str_in_output=include_stop_str_in_output,
        engine_stop=True,
    )

    text = "".join(_run_finish(detokenizer, [[A, B, C], [A, B]], stop_text))

    assert text == expected_text
    assert detokenizer.text == expected_text


def test_engine_stop_holds_back_a_possible_stop_string(tokenizer: TransformersTokenizer) -> None:
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["ab"], engine_stop=True)

    assert detokenizer.step([C, A]) == DetokenizeDelta("c")
    assert detokenizer.step([C]) == DetokenizeDelta("ac")
    assert detokenizer.finish() == DetokenizeDelta("")


def test_engine_stop_does_not_match_token_ids(tokenizer: TransformersTokenizer) -> None:
    """The engine removes the stop tokens itself; this class does not look for them."""
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["</s>"], engine_stop=True)

    assert "".join(_run(detokenizer, [[HE, EOS, LLO]])) == "Hello"
    assert detokenizer.stop_match is None


def _run_finish(
    detokenizer: IncrementalDetokenizer, chunks: list[list[int]], stop_text: str | None
) -> list[str]:
    deltas = [detokenizer.step(chunk).text for chunk in chunks]
    deltas.append(detokenizer.finish(stop_text=stop_text).text)
    return deltas

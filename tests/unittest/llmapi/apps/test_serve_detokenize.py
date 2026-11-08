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
from tokenizers import Regex, Tokenizer
from tokenizers.decoders import ByteFallback, Fuse, Metaspace, Replace, Sequence
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
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


def _word_level(*, split: bool) -> TransformersTokenizer:
    backend = Tokenizer(WordLevel(_VOCAB, unk_token="<unk>"))
    backend.decoder = Sequence([ByteFallback(), Fuse()])
    backend.add_special_tokens(["</s>"])
    if split:
        backend.pre_tokenizer = Split(
            Regex(r"</|ans|>|He|llo| world|a|b|c|\n"), behavior="isolated"
        )
    return TransformersTokenizer(
        PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>")
    )


@pytest.fixture(scope="module")
def tokenizer() -> TransformersTokenizer:
    """Word-level pieces that decode by concatenation; ``<0xC2>`` ``<0xBB>`` make up ``»``.

    A stop string encodes to one token when it is one piece, else to ``<unk>``.
    """
    return _word_level(split=False)


@pytest.fixture(scope="module")
def split_tokenizer() -> TransformersTokenizer:
    """The same vocabulary, encoding text piece by piece: ``</ans>`` is ``</`` ``ans`` ``>``."""
    return _word_level(split=True)


@pytest.fixture(params=["decode_stream", "fallback"], autouse=True)
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Runs every test with ``DecodeStream`` and with the ``trtllm_decode_incrementally`` fallback.

    The fallback is what a tokenizer without a ``tokenizers`` backend gets; here it is forced by
    hiding ``DecodeStream``.
    """
    if request.param == "fallback":
        from tensorrt_llm.serve import detokenize as detokenize_module

        monkeypatch.setattr(detokenize_module, "DecodeStream", None)
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
            id="single_token_stop_matched_on_its_id_first",
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
    the vocabulary, so they are found in the text.
    """
    detokenizer = IncrementalDetokenizer(
        tokenizer, stop=[stop], spaces_between_special_tokens=spaces_between_special_tokens
    )

    text = "".join(_run(detokenizer, [[token_id] for token_id in token_ids]))

    assert text == expected_text
    expected = None if expected_index is None else StopMatch(stop, expected_index)
    assert detokenizer.stop_match == expected


def test_kept_stop_string_ending_in_whitespace_ends_at_its_token(
    tokenizer: TransformersTokenizer,
) -> None:
    """The stop string is matched on the token that adds its whitespace, however the text was released."""
    detokenizer = IncrementalDetokenizer(
        tokenizer, stop=["Hello\n"], include_stop_str_in_output=True
    )

    assert "".join(_run(detokenizer, [[HE], [LLO], [NEWLINE], [A], [B]])) == "Hello\n"
    assert detokenizer.stop_match == StopMatch("Hello\n", 2)


def test_stop_string_completed_by_whitespace_after_released_text(
    tokenizer: TransformersTokenizer,
) -> None:
    """Whitespace held back before some released text plays no part in a later match."""
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["a\n"])

    assert "".join(_run(detokenizer, [[NEWLINE], [A], [NEWLINE], [B]])) == "\n"
    assert detokenizer.stop_match == StopMatch("a\n", 2)


def test_stop_string_completed_by_whitespace_whose_text_changes(backend: str) -> None:
    """Whitespace the fallback decoder holds back is checked again when its text changes.

    Two spaces decode to a tab, so the held text goes from a space to a tab.
    """
    if backend == "decode_stream":
        pytest.skip("DecodeStream releases the first space before the second one changes it")
    backend_tokenizer = Tokenizer(
        WordLevel({"x": 0, " ": 1, "y": 2, "<unk>": 3}, unk_token="<unk>")
    )
    backend_tokenizer.decoder = Sequence([ByteFallback(), Fuse(), Replace("  ", "\t")])
    tokenizer = TransformersTokenizer(PreTrainedTokenizerFast(tokenizer_object=backend_tokenizer))
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["x\t"])

    assert "".join(_run(detokenizer, [[0], [1], [1], [2]])) == ""
    assert detokenizer.stop_match == StopMatch("x\t", 2)


def test_kept_stop_string_is_not_held_back(tokenizer: TransformersTokenizer) -> None:
    """With the stop string kept in the output, a possible prefix is safe to release."""
    detokenizer = IncrementalDetokenizer(
        tokenizer, stop=["</ans>"], include_stop_str_in_output=True
    )

    assert detokenizer.step([HE, LLO, CLOSE]).text == "Hello</"
    assert detokenizer.step([ANS]).text == "ans"
    assert detokenizer.step([GT]) == DetokenizeDelta(">", StopMatch("</ans>", 4))
    assert detokenizer.text == "Hello</ans>"


def test_kept_stop_string_holds_no_token_back(split_tokenizer: TransformersTokenizer) -> None:
    """With the stop string kept, the ids that may start its tokens are not held back either."""
    detokenizer = IncrementalDetokenizer(
        split_tokenizer, stop=["</ans>"], include_stop_str_in_output=True
    )

    detokenizer.step([HE, CLOSE, ANS])
    assert detokenizer.held_tokens == 0


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


def test_kept_stop_string_reported_as_llm_does(tokenizer: TransformersTokenizer) -> None:
    """With the stop string kept, LLM's text check reports the first stop string, in request order, the text holds."""
    detokenizer = IncrementalDetokenizer(
        tokenizer, stop=["ab", "b"], include_stop_str_in_output=True
    )

    assert "".join(_run(detokenizer, [[C, A, B]])) == "cab"
    assert detokenizer.stop_match == StopMatch("ab", 2, num_stop_tokens=1)


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
        pytest.param(["b"], [[A, B, C, B]], 2, "abc", StopMatch("b", 3, 1), id="token_ids_at_min"),
        pytest.param(["b"], [[A, C, B]], 2, "ac", StopMatch("b", 2, 1), id="token_ids_after_min"),
        pytest.param(["b", "cb"], [[A, C, B]], 3, "acb", None, id="banned_token_text"),
        pytest.param(["ab"], [[A, B, C, A, B]], 3, "", StopMatch("ab", 1), id="text_counts"),
    ],
)
def test_min_tokens_only_holds_back_single_token_stop_strings(
    tokenizer: TransformersTokenizer,
    stop: list[str],
    chunks: list[list[int]],
    min_tokens: int,
    expected_text: str,
    expected_match: StopMatch | None,
) -> None:
    """``LLM`` bans the token of a single-token stop string until the output has ``min_tokens`` tokens.

    Such a token generated anyway completes no match, on its token id or in the
    text; any other match counts, as ``LLM`` does not wait for ``min_tokens`` there.
    """
    detokenizer = IncrementalDetokenizer(tokenizer, stop=stop, min_tokens=min_tokens)

    assert "".join(_run(detokenizer, chunks)) == expected_text
    assert detokenizer.stop_match == expected_match


def test_min_tokens_holds_for_whitespace_the_decoder_holds(
    tokenizer: TransformersTokenizer,
) -> None:
    """A banned whitespace token completes no match, also when the decoder releases its text later."""
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["\n"], min_tokens=5)

    assert "".join(_run(detokenizer, [[A], [NEWLINE], [B]])) == "a\nb"
    assert detokenizer.stop_match is None


def test_min_tokens_holds_for_a_banned_last_token(tokenizer: TransformersTokenizer) -> None:
    """A banned token completes no match, also as the output's last token."""
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["\n"], min_tokens=5)

    assert "".join(_run(detokenizer, [[A, NEWLINE]])) == "a\n"
    assert detokenizer.stop_match is None


# -- engine_stop: the engine matches the stop strings ------------------------------


def test_engine_stop_never_ends_the_output(tokenizer: TransformersTokenizer) -> None:
    """The engine decides; the text is held back from the match on, then cut at finish."""
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["</ans>"], engine_stop=True)

    assert detokenizer.step([HE, LLO, CLOSE]) == DetokenizeDelta("Hello")
    assert detokenizer.step([ANS, GT, WORLD]) == DetokenizeDelta("")
    assert not detokenizer.finished
    assert detokenizer.finish() == DetokenizeDelta("", StopMatch("</ans>", 5))
    assert detokenizer.text == "Hello"


@pytest.mark.parametrize(
    "stop, include_stop_str_in_output, expected_text, expected_stop",
    [
        pytest.param(["b"], False, "a", "b", id="cut_at_first_occurrence"),
        pytest.param(["b"], True, "ab", "b", id="kept"),
        pytest.param(["c", "b"], False, "ab", "c", id="first_in_request_order"),
        pytest.param(["</ans>"], False, "abcab", None, id="not_in_text"),
    ],
)
def test_engine_stop_cuts_as_llm_does(
    tokenizer: TransformersTokenizer,
    stop: list[str],
    include_stop_str_in_output: bool,
    expected_text: str,
    expected_stop: str | None,
) -> None:
    """Whatever ended the output, ``LLM`` cuts the text at the first stop string, in request order, that it holds."""
    detokenizer = IncrementalDetokenizer(
        tokenizer,
        stop=stop,
        include_stop_str_in_output=include_stop_str_in_output,
        engine_stop=True,
    )

    text = "".join(_run(detokenizer, [[A, B, C], [A, B]]))

    assert text == expected_text
    assert detokenizer.text == expected_text
    match = detokenizer.stop_match
    assert (match.stop_str if match else None) == expected_stop
    assert match is None or (match.token_index, match.num_stop_tokens) == (4, 0)


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


def test_token_ids_match_when_the_full_decode_disagrees(
    tokenizer: TransformersTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The text held back is kept when the full decode does not continue the released text."""
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["b", "ca"])

    assert detokenizer.step([A, C]) == DetokenizeDelta("a")
    monkeypatch.setattr(tokenizer, "decode", lambda *args, **kwargs: "zz")
    assert detokenizer.step([B]) == DetokenizeDelta("c", StopMatch("b", 2, num_stop_tokens=1))
    assert detokenizer.text == "ac"


def test_multi_token_stop_cut_when_the_full_decode_disagrees(
    split_tokenizer: TransformersTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All the stop string's tokens are cut from the text held back, not just the last."""
    detokenizer = IncrementalDetokenizer(split_tokenizer, stop=["</ans>"])

    assert detokenizer.step([HE, CLOSE, ANS]) == DetokenizeDelta("He")
    monkeypatch.setattr(split_tokenizer, "decode", lambda *args, **kwargs: "zz")
    assert detokenizer.step([GT]) == DetokenizeDelta("", StopMatch("</ans>", 3, num_stop_tokens=3))
    assert detokenizer.text == "He"


def test_whitespace_held_back_kept_when_the_full_decode_disagrees(
    tokenizer: TransformersTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Text the decoder holds back before a stop string's token stays in the output."""
    detokenizer = IncrementalDetokenizer(tokenizer, stop=["b"])

    released = detokenizer.step([A, NEWLINE]).text
    monkeypatch.setattr(tokenizer, "decode", lambda *args, **kwargs: "zz")
    delta = detokenizer.step([B])
    assert released + delta.text == "a\n"
    assert delta.stop == StopMatch("b", 2, num_stop_tokens=1)


def test_finish_without_detokenizing_holding_part_of_a_stop(
    split_tokenizer: TransformersTokenizer,
) -> None:
    detokenizer = IncrementalDetokenizer(split_tokenizer, stop=["</ans>"], detokenize=False)

    assert detokenizer.step([HE, CLOSE, ANS]) == DetokenizeDelta("")
    assert detokenizer.held_tokens == 2
    assert detokenizer.finish() == DetokenizeDelta("")
    assert detokenizer.stop_match is None

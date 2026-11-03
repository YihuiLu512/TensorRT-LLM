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
"""Incremental detokenization and stop-string matching for one generated output.

An engine that returns only token ids leaves both jobs to the serving frontend.
:class:`IncrementalDetokenizer` turns the ids of one output into text as they
arrive and, unless matched stop strings are kept in the output, only releases
text that can no longer turn out to be the start of a stop string. It needs a
tokenizer but no executor, model weights or GPU.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from tensorrt_llm.tokenizer import TransformersTokenizer

try:
    from tokenizers.decoders import DecodeStream
except ImportError:  # tokenizers < 0.21
    DecodeStream = None


@dataclass(frozen=True)
class StopMatch:
    """A stop string found in the output.

    Attributes:
        stop_str: The stop string that matched.
        token_index: Index, within this output, of the token that released the
            text completing the match. That is the token whose text completed
            it, unless the decoder held that text back until a later token (see
            :class:`IncrementalDetokenizer`). Tokens after it were generated
            past the stop.
    """

    stop_str: str
    token_index: int


@dataclass(frozen=True)
class DetokenizeDelta:
    """Text released by one call to the detokenizer.

    Attributes:
        text: New text that is safe to send to the client.
        stop: The stop string match, set only on the call that found it.
    """

    text: str
    stop: StopMatch | None = None


class IncrementalDetokenizer:
    """Detokenizes one output incrementally and applies stop strings to its text.

    Use one instance per output sequence. Unless the stop string is to be kept
    in the output, text that could be the start of a stop string is held back
    until later text rules the match in or out, or until :meth:`finish`
    releases it. When several stop strings occur, the one whose match ends
    first wins, and among those the longest; the result therefore does not
    depend on how the token ids are split across calls.

    Tokens are decoded one at a time. When the tokenizer wraps a Hugging Face
    ``tokenizers`` backend (``tokenizer.tokenizer._tokenizer``) and
    ``DecodeStream`` is available, ``DecodeStream`` decodes them; otherwise they
    go through the tokenizer's ``decode_incrementally``. Both hold back decoded
    text while it ends in an incomplete multi-byte character; in byte-level
    vocabularies a token can carry complete characters before such a partial
    one, and those are held back with it. ``decode_incrementally`` also holds
    back text while the latest tokens add only whitespace, so on that path a
    stop string that ends in whitespace is not found while the output keeps
    adding only whitespace. A stop string that ends in held-back text is
    reported as completed by the token that released it; text still held back
    when :meth:`finish` is called counts as text of the last token.

    Args:
        tokenizer: Tokenizer whose vocabulary produced the token ids.
        stop: Stop strings to match in the decoded text. Empty strings are
            ignored.
        include_stop_str_in_output: Keep the matched stop string at the end of
            the released text instead of cutting it off.
        skip_special_tokens: Leave special tokens out of the decoded text.
        spaces_between_special_tokens: Passed to ``decode_incrementally``;
            ``DecodeStream`` ignores it.

    Raises:
        TypeError: If the tokenizer is not a ``TransformersTokenizer``, or a stop
            string is not a ``str``.
    """

    def __init__(
        self,
        tokenizer: TransformersTokenizer,
        *,
        stop: str | Iterable[str] | None = None,
        include_stop_str_in_output: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ) -> None:
        if not isinstance(tokenizer, TransformersTokenizer):
            raise TypeError(
                f"tokenizer must be a TransformersTokenizer, got {type(tokenizer).__name__}"
            )
        self._tokenizer = tokenizer
        # decode_incrementally also requires this attribute for its HF backend.
        self._use_decode_stream = DecodeStream is not None and hasattr(
            tokenizer.tokenizer, "_tokenizer"
        )
        self._skip_special_tokens = skip_special_tokens
        self._spaces_between_special_tokens = spaces_between_special_tokens
        stop_strs = _normalize_stop(stop)
        self._matcher = _StopMatcher(stop_strs) if stop_strs else None
        self._include_stop_str_in_output = include_stop_str_in_output
        self._decode_states: dict[str, Any] | None = None
        self._num_tokens = 0
        self._released: list[str] = []
        self._unreleased = ""
        self._stop_match: StopMatch | None = None
        self._finished = False

    @property
    def text(self) -> str:
        """All text released so far."""
        if len(self._released) > 1:
            self._released = ["".join(self._released)]
        return self._released[0] if self._released else ""

    @property
    def stop_match(self) -> StopMatch | None:
        """The stop string match, or None if no stop string has matched."""
        return self._stop_match

    @property
    def finished(self) -> bool:
        """Whether a stop string matched or :meth:`finish` was called."""
        return self._finished

    def step(self, token_ids: Sequence[int]) -> DetokenizeDelta:
        """Decodes newly generated token ids and returns the text that is safe to release.

        Args:
            token_ids: The ids generated since the previous call, in order.

        Returns:
            The released text, with the stop match if this call found one. Ids
            after the one that released the match are not decoded. Once the
            output has ended (a stop string matched or :meth:`finish` was
            called), the delta is empty and carries no match;
            :attr:`stop_match` keeps it.
        """
        if self._finished:
            return DetokenizeDelta("")
        released = []
        for token_id in token_ids:
            new_text = self._decode([token_id], flush=False)
            self._num_tokens += 1
            released.append(self._consume(new_text))
            if self._finished:
                break
        return DetokenizeDelta("".join(released), self._stop_match if self._finished else None)

    def finish(self) -> DetokenizeDelta:
        """Ends the output and releases all text still held back.

        Call it when the engine reports that the output finished. Text that the
        tokenizer was still holding (for example, an incomplete multi-byte
        character) is decoded and checked against the stop strings first; it
        counts as text of the last token.

        Returns:
            The remaining text, with the stop match if that text completed one.
            Once the output has ended, the delta is empty and carries no match.
        """
        if self._finished:
            return DetokenizeDelta("")
        released = ""
        if self._num_tokens:
            released = self._consume(self._decode([], flush=True))
        if self._finished:
            return DetokenizeDelta(released, self._stop_match)
        released += self._unreleased
        self._released.append(self._unreleased)
        self._unreleased = ""
        self._finished = True
        return DetokenizeDelta(released)

    def _decode(self, token_ids: list[int], *, flush: bool) -> str:
        """Returns only the text that ``token_ids`` (or the flush) adds."""
        if self._use_decode_stream:
            text, self._decode_states = self._tokenizer.hf_decode_incrementally(
                token_ids,
                states=self._decode_states,
                flush=flush,
                skip_special_tokens=self._skip_special_tokens,
            )
        else:
            text, self._decode_states = self._tokenizer.decode_incrementally(
                token_ids,
                states=self._decode_states,
                flush=flush,
                skip_special_tokens=self._skip_special_tokens,
                spaces_between_special_tokens=self._spaces_between_special_tokens,
            )
        return text

    def _consume(self, new_text: str) -> str:
        """Appends text of the latest token and returns the part that is safe to release."""
        match = self._matcher.feed(new_text) if self._matcher is not None else None
        pending = self._unreleased + new_text
        if match is not None:
            end_in_new_text, stop_str = match
            end = len(self._unreleased) + end_in_new_text
            release_end = end if self._include_stop_str_in_output else end - len(stop_str)
            self._stop_match = StopMatch(stop_str, self._num_tokens - 1)
            self._finished = True
            self._unreleased = ""
        else:
            held = 0
            if self._matcher is not None and not self._include_stop_str_in_output:
                held = self._matcher.held_length
            release_end = len(pending) - held
            self._unreleased = pending[release_end:]
        released = pending[:release_end]
        self._released.append(released)
        return released


class _StopMatcher:
    """Finds stop strings in a stream of text, one character at a time.

    Keeps, for every stop string, the length of the longest suffix of the text
    seen so far that is a prefix of it (Knuth-Morris-Pratt), so each character
    costs one amortized step per stop string.
    """

    def __init__(self, stop_strs: tuple[str, ...]) -> None:
        self._stop_strs = stop_strs
        self._failure = [_kmp_failure_table(stop_str) for stop_str in stop_strs]
        self._matched = [0] * len(stop_strs)

    @property
    def held_length(self) -> int:
        """Length of the longest suffix of the text that starts some stop string."""
        return max(self._matched, default=0)

    def feed(self, text: str) -> tuple[int, str] | None:
        """Scans ``text`` and returns ``(end, stop_str)`` for the first match that ends in it.

        ``end`` is the offset in ``text`` just past the match. Among matches that
        end at the same character, the longest stop string wins. The matcher must
        not be fed again after it returns a match.
        """
        for offset, char in enumerate(text):
            found = None
            for index, stop_str in enumerate(self._stop_strs):
                matched = self._matched[index]
                while matched and char != stop_str[matched]:
                    matched = self._failure[index][matched - 1]
                if char == stop_str[matched]:
                    matched += 1
                if matched == len(stop_str) and (found is None or len(stop_str) > len(found)):
                    found = stop_str
                self._matched[index] = matched
            if found is not None:
                return offset + 1, found
        return None


def _kmp_failure_table(pattern: str) -> list[int]:
    """``table[i]``: length of the longest proper prefix of ``pattern[:i + 1]`` that is its suffix."""
    table = [0] * len(pattern)
    matched = 0
    for index in range(1, len(pattern)):
        while matched and pattern[index] != pattern[matched]:
            matched = table[matched - 1]
        if pattern[index] == pattern[matched]:
            matched += 1
        table[index] = matched
    return table


def _normalize_stop(stop: str | Iterable[str] | None) -> tuple[str, ...]:
    if stop is None:
        return ()
    stop_strs = [stop] if isinstance(stop, str) else list(stop)
    for stop_str in stop_strs:
        if not isinstance(stop_str, str):
            raise TypeError(f"stop strings must be str, got {type(stop_str).__name__}")
    return tuple(dict.fromkeys(stop_str for stop_str in stop_strs if stop_str))

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

An engine that returns only token ids leaves detokenization, and sometimes stop
strings, to the serving frontend. :class:`IncrementalDetokenizer` turns the ids
of one output into text as they arrive and, unless matched stop strings are
kept in the output, only releases text that can no longer turn out to be the
start of a stop string. It needs a tokenizer but no executor, model weights or
GPU.
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

_REPLACEMENT_CHARACTER = "\N{REPLACEMENT CHARACTER}"


@dataclass(frozen=True)
class StopMatch:
    """A stop string found in the output.

    Attributes:
        stop_str: The stop string that matched.
        token_index: Index, within this output, of the token that completed
            the match. For a match found in the text, that is the token that
            released the text completing it (see :class:`IncrementalDetokenizer`);
            for the cut :meth:`IncrementalDetokenizer.finish` makes with
            ``engine_stop``, the last token. Tokens after it were generated
            past the stop.
        num_stop_tokens: For a match found as the stop string's own token ids
            (the way ``LLM`` matches stop strings), the number of those ids,
            which end at ``token_index``; 0 for a match found in the text.
            Unless the stop string is kept in the output, these tokens are not
            part of the output, as with ``LLM``.
    """

    stop_str: str
    token_index: int
    num_stop_tokens: int = 0


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
    """Detokenizes one output incrementally and applies stop strings to it.

    Use one instance per output sequence. The text released over all calls is
    the text ``LLM`` gives for the same token ids: the tokenizer's ``decode``
    of the output tokens, cut at the stop string, except where noted below.

    **Who matches stop strings.** With ``engine_stop`` the engine matches them
    and ends the output. :meth:`finish` then cuts the text as ``LLM`` cuts an
    output that was not cancelled, whatever ended it: at the first occurrence of the first stop
    string, in request order, that the text holds, keeping it when
    ``include_stop_str_in_output`` is set. Until then, text that could start a
    stop string is held back, and once the text holds a complete stop string
    nothing more is released before :meth:`finish`.

    Without ``engine_stop`` this class matches them and ends the output at the
    first match, found in either of the two ways ``LLM`` finds one:

    - the output's last token ids equal a stop string's token ids (the
      tokenizer's encoding of it without special tokens). This finds stop
      strings made of special tokens even when ``skip_special_tokens`` leaves
      them out of the text. When both ways match on the same token, this one
      wins. ``LLM``'s engine also matches token ids that start in the
      prompt's last tokens; this class sees only the output's;
    - the decoded text contains the stop string. When several stop strings
      occur, the one whose match ends first wins, and among those the longest;
      the result therefore does not depend on how the token ids are split
      across calls. ``LLM`` instead takes the first stop string, in request
      order, that its text holds: when not streaming, that text is the whole
      output, so a later match of an earlier stop string can win there.

    As with ``LLM``, ``min_tokens`` only keeps a stop string made of a single
    token from ending the output: a match completed by such a token generated
    while the output had fewer than ``min_tokens`` tokens does not count
    (``LLM``'s sampler bans those tokens until then). Unless the stop string is
    to be kept in the output, text that could be the start of a stop string is
    held back until later text rules the match in or out, or until
    :meth:`finish` releases it.

    **Decoding.** Tokens are decoded one at a time. When the tokenizer wraps a
    Hugging Face ``tokenizers`` backend (``tokenizer.tokenizer._tokenizer``) and
    ``DecodeStream`` is available, ``DecodeStream`` decodes them; otherwise the
    tokenizer's ``trtllm_decode_incrementally`` does. Both hold back decoded
    text while it ends in an incomplete multi-byte character; in byte-level
    vocabularies a token can carry complete characters before such a partial
    one, and those are held back with it. ``trtllm_decode_incrementally`` also
    holds back text while the latest tokens add only whitespace; when releasing
    that whitespace would complete a stop string, it is released with the
    token that added it, so stop strings ending in whitespace are found as on
    the other path. A stop string that ends in held-back text is reported as
    completed by the token that released it; text still held back when
    :meth:`finish` is called counts as text of the last token.
    ``DecodeStream`` does not apply the tokenizer's
    ``clean_up_tokenization_spaces``, which ``decode`` applies when the
    tokenizer sets it.

    Args:
        tokenizer: Tokenizer whose vocabulary produced the token ids.
        stop: Stop strings, in the request's order.
        include_stop_str_in_output: Keep the matched stop string (and, for a
            match on token ids, its tokens) in the output instead of cutting it off.
        skip_special_tokens: Leave special tokens out of the decoded text.
        spaces_between_special_tokens: Passed to ``trtllm_decode_incrementally``;
            ``DecodeStream`` ignores it.
        min_tokens: A single-token stop string's token generated before the
            output has this many tokens does not end it; see above.
        engine_stop: The engine matches the stop strings; see above.
        detokenize: Decode text. Without it no text is released, and stop
            strings are only matched on their token ids, as ``LLM`` does when
            not detokenizing.

    Raises:
        TypeError: If the tokenizer is not a ``TransformersTokenizer``, or a stop
            string is not a ``str``.
        ValueError: If a stop string is empty.
    """

    def __init__(
        self,
        tokenizer: TransformersTokenizer,
        *,
        stop: str | Iterable[str] | None = None,
        include_stop_str_in_output: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        min_tokens: int | None = None,
        engine_stop: bool = False,
        detokenize: bool = True,
    ) -> None:
        if not isinstance(tokenizer, TransformersTokenizer):
            raise TypeError(
                f"tokenizer must be a TransformersTokenizer, got {type(tokenizer).__name__}"
            )
        self._tokenizer = tokenizer
        # DecodeStream works on the Hugging Face tokenizers backend in this attribute.
        self._use_decode_stream = DecodeStream is not None and hasattr(
            tokenizer.tokenizer, "_tokenizer"
        )
        self._skip_special_tokens = skip_special_tokens
        self._spaces_between_special_tokens = spaces_between_special_tokens
        self._detokenize = detokenize
        self._include_stop_str_in_output = include_stop_str_in_output
        self._min_tokens = min_tokens or 0
        self._engine_stop = engine_stop
        stop_strs = normalize_stop(stop)
        self._stop_strs = stop_strs
        self._matcher = _StopMatcher(stop_strs) if stop_strs and detokenize else None
        # The token ids of each stop string, for matching the way LLM's engine does.
        words: list[tuple[str, tuple[int, ...]]] = []
        if not engine_stop:
            for stop_str in stop_strs:
                token_ids = tuple(tokenizer.encode(stop_str, add_special_tokens=False))
                if token_ids:
                    words.append((stop_str, token_ids))
        self._token_matcher = _TokenStopMatcher(words) if words else None
        # Tokens LLM's sampler bans until the output has min_tokens tokens.
        self._banned_before_min = {ids[0] for _, ids in words if len(ids) == 1}
        # trtllm_decode_incrementally holds back whitespace only, so only these
        # stop strings can end in the text it holds (see _whitespace_completing_a_stop).
        self._whitespace_stops = tuple(s for s in stop_strs if s[-1].isspace())
        self._longest_whitespace_stop = max(map(len, self._whitespace_stops), default=0)
        # Length of the held whitespace already checked for those stop strings.
        self._whitespace_checked = 0
        # With a token-level matcher only: the ids, and the text length (released
        # and held) after each token; index -1 - k is the length before the last k tokens.
        self._ids: list[int] = []
        self._token_ends: list[int] = [0]
        # Whether a match completed by the latest token counts (see min_tokens).
        self._last_counts = True
        self._decode_states: dict[str, Any] | None = None
        self._num_tokens = 0
        self._released: list[str] = []
        self._released_len = 0
        # The text released by the current call.
        self._delta: list[str] = []
        self._unreleased = ""
        self._stop_match: StopMatch | None = None
        # engine_stop: the text holds a complete stop string; release nothing until finish.
        self._frozen = False
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

    @property
    def held_tokens(self) -> int:
        """How many of the latest ids may start a stop string's token ids, so may be dropped.

        Always 0 when the stop string would be kept in the output, with
        ``engine_stop``, and once the output has ended.
        """
        if self._finished or self._token_matcher is None or self._include_stop_str_in_output:
            return 0
        return self._token_matcher.held_length

    def step(self, token_ids: Sequence[int]) -> DetokenizeDelta:
        """Decodes newly generated token ids and returns the text that is safe to release.

        Args:
            token_ids: The ids generated since the previous call, in order.

        Returns:
            The released text, with the stop match if this call found one. Ids
            after the one that completed the match are not decoded. Once the
            output has ended (a stop string matched or :meth:`finish` was
            called), the delta is empty and carries no match;
            :attr:`stop_match` keeps it.
        """
        if self._finished:
            return DetokenizeDelta("")
        self._delta = []
        for token_id in token_ids:
            self._step_one(token_id)
            if self._finished:
                break
        return DetokenizeDelta("".join(self._delta), self._stop_match if self._finished else None)

    def finish(self) -> DetokenizeDelta:
        """Ends the output and releases all text still held back.

        Call it when the output finished, or when the serving side gave up on
        it. Text that the tokenizer was still holding (for example, an
        incomplete multi-byte character) is decoded first; it counts as text of
        the last token. With ``engine_stop``, the text is then cut as ``LLM``
        cuts it (see the class description).

        Returns:
            The remaining text, with the stop match if that text completed one
            or, with ``engine_stop``, if the text was cut at a stop string; such
            a match has ``num_stop_tokens`` 0 and ``token_index`` of the last
            token. Once the output has ended, the delta is empty and carries no
            match.
        """
        if self._finished:
            return DetokenizeDelta("")
        self._delta = []
        new_text = self._decode([], flush=True) if self._num_tokens and self._detokenize else ""
        if self._engine_stop:
            self._unreleased += new_text
            full = self.text + self._unreleased
            end = len(full)
            for stop_str in self._stop_strs:
                position = full.find(stop_str)
                if position >= 0:
                    end = position + (len(stop_str) if self._include_stop_str_in_output else 0)
                    self._stop_match = StopMatch(stop_str, self._num_tokens - 1)
                    break
            self._release_to(end)
        elif self._detokenize:
            self._consume_text(new_text, self._num_tokens - 1, counts=self._last_counts)
            if not self._finished:
                self._release_to(self._released_len + len(self._unreleased))
        self._unreleased = ""
        self._finished = True
        return DetokenizeDelta("".join(self._delta), self._stop_match)

    # -- one token -----------------------------------------------------------

    def _step_one(self, token_id: int) -> None:
        index = self._num_tokens
        self._num_tokens += 1
        if self._token_matcher is not None:
            self._ids.append(token_id)
        counts = index >= self._min_tokens or token_id not in self._banned_before_min
        self._last_counts = counts
        word = self._token_matcher.feed(token_id) if self._token_matcher is not None else None
        if word is not None and counts:
            stop_str, num_stop_tokens = word
            end, stop_str = self._text_as_llm(stop_str, num_stop_tokens)
            self._end_at(StopMatch(stop_str, index, num_stop_tokens), end)
            return
        if not self._detokenize:
            return
        new_text = self._decode([token_id], flush=False)
        if new_text:
            self._whitespace_checked = 0
        elif counts:
            new_text = self._whitespace_completing_a_stop()
        self._consume_text(new_text, index, counts=counts)

    def _text_as_llm(self, stop_str: str, num_stop_tokens: int) -> tuple[int, str]:
        """Where ``LLM`` ends the text after matching ``stop_str`` on its token ids, and the stop it reports.

        ``LLM`` decodes the output tokens, without the stop tokens unless they
        are kept, then looks for the stop strings in the text in request order
        and cuts the text at the first occurrence of the first one found,
        which becomes the reported stop. Returns the end as a position in the
        text, which this method also makes the held text; already released
        text stays.
        """
        if not self._detokenize:
            return self._released_len, stop_str
        ids = self._ids if self._include_stop_str_in_output else self._ids[:-num_stop_tokens]
        full = self._tokenizer.decode(
            ids,
            skip_special_tokens=self._skip_special_tokens,
            spaces_between_special_tokens=self._spaces_between_special_tokens,
        )
        released = self.text
        if not full.startswith(released):
            # The incremental decode disagrees with the full one: cut the incremental text.
            full = released + self._unreleased
            if self._include_stop_str_in_output:
                full += self._decode([self._ids[-1]], flush=True)
            else:
                full = full[: max(self._token_ends[-num_stop_tokens], len(released))]
        for candidate in self._stop_strs:
            position = full.find(candidate)
            if position >= 0:
                stop_str = candidate
                keep = len(candidate) if self._include_stop_str_in_output else 0
                full = full[: max(position + keep, len(released))]
                break
        self._unreleased = full[len(released) :]
        return len(full), stop_str

    def _consume_text(self, new_text: str, index: int, *, counts: bool = True) -> None:
        """Adds text of token ``index`` and releases what is safe.

        Without ``counts``, a stop string the text completes does not count.
        """
        before = self._released_len + len(self._unreleased)
        self._unreleased += new_text
        if index >= 0 and self._token_matcher is not None:
            self._token_ends.append(before + len(new_text))
        if self._frozen:
            return
        if new_text and self._matcher is not None:
            match = self._matcher.feed(new_text, counts=counts)
            if match is not None:
                end_in_new_text, stop_str = match
                if self._engine_stop:
                    # The engine decides where the output ends; release nothing more.
                    self._frozen = True
                    return
                end = before + end_in_new_text
                if not self._include_stop_str_in_output:
                    end -= len(stop_str)
                self._end_at(StopMatch(stop_str, index), end)
                return
        end = self._released_len + len(self._unreleased)
        if self._matcher is not None and not self._include_stop_str_in_output:
            end -= self._matcher.held_length
        if self._token_matcher is not None and not self._include_stop_str_in_output:
            # The text of the latest tokens that could start a stop string's token
            # ids: if the rest of them follows, LLM drops these tokens, text included.
            held_tokens = self._token_matcher.held_length
            if held_tokens:
                end = min(end, self._token_ends[-1 - held_tokens])
        self._release_to(end)

    def _end_at(self, match: StopMatch, end: int) -> None:
        self._release_to(end)
        self._unreleased = ""
        self._stop_match = match
        self._finished = True

    def _release_to(self, end: int) -> None:
        """Releases the held text up to absolute position ``end``.

        Text already released stays released, so ``end`` before it releases nothing.
        """
        count = end - self._released_len
        if count <= 0:
            return
        released, self._unreleased = self._unreleased[:count], self._unreleased[count:]
        self._released.append(released)
        self._delta.append(released)
        self._released_len += len(released)

    # -- decoding ------------------------------------------------------------

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
            text, self._decode_states = self._tokenizer.trtllm_decode_incrementally(
                token_ids,
                states=self._decode_states,
                flush=flush,
                skip_special_tokens=self._skip_special_tokens,
                spaces_between_special_tokens=self._spaces_between_special_tokens,
            )
        return text

    def _whitespace_completing_a_stop(self) -> str:
        """The text ``trtllm_decode_incrementally`` holds back, if releasing it completes a stop string.

        That decoder holds back text while the latest tokens add only
        whitespace. When that text completes a stop string, it is released
        now; the decoder state then continues from the flushed state.
        """
        if self._use_decode_stream or self._frozen or not self._whitespace_stops:
            return ""
        # Given no ids, the decoder only pops keys of the state it is passed, so a
        # shallow copy leaves the current state as it is.
        text, flushed_states = self._tokenizer.trtllm_decode_incrementally(
            [],
            states=dict(self._decode_states or {}),
            flush=True,
            skip_special_tokens=self._skip_special_tokens,
            spaces_between_special_tokens=self._spaces_between_special_tokens,
        )
        if not text or text.endswith(_REPLACEMENT_CHARACTER):
            return ""
        # The text checked before completed no stop string, so a match must end
        # in what was added since.
        checked, self._whitespace_checked = self._whitespace_checked, len(text)
        context = self._longest_whitespace_stop - 1
        window = text[checked:]
        if context:
            window = (self._tail(context) + text[:checked])[-context:] + window
        if not any(stop_str in window for stop_str in self._whitespace_stops):
            return ""
        self._decode_states = flushed_states
        self._whitespace_checked = 0
        return text

    def _tail(self, length: int) -> str:
        """The last ``length`` characters of the text so far, released or held."""
        tail = self._unreleased[-length:]
        for piece in reversed(self._released):
            if len(tail) >= length:
                break
            tail = piece[len(piece) - (length - len(tail)) :] + tail
        return tail


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

    def feed(self, text: str, *, counts: bool = True) -> tuple[int, str] | None:
        """Scans ``text`` and returns ``(end, stop_str)`` for the first match that ends in it.

        ``end`` is the offset in ``text`` just past the match. Among matches that
        end at the same character, the longest stop string wins. Without
        ``counts``, matches are skipped and scanning goes on. The matcher must
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
                if matched == len(stop_str):
                    if found is None or len(stop_str) > len(found):
                        found = stop_str
                    if not counts:
                        matched = self._failure[index][matched - 1]
                self._matched[index] = matched
            if found is not None and counts:
                return offset + 1, found
        return None


class _TokenStopMatcher:
    """Finds stop strings' token ids at the end of a stream of token ids.

    Keeps, for every stop string, the length of the longest suffix of the ids
    seen so far that is a prefix of its token ids (Knuth-Morris-Pratt), so each
    id costs one amortized step per stop string.
    """

    def __init__(self, words: list[tuple[str, tuple[int, ...]]]) -> None:
        self._words = words
        self._failure = [_kmp_failure_table(token_ids) for _, token_ids in words]
        self._matched = [0] * len(words)

    @property
    def held_length(self) -> int:
        """Length of the longest suffix of the ids that is a proper prefix of some stop string's ids."""
        return max(self._matched, default=0)

    def feed(self, token_id: int) -> tuple[str, int] | None:
        """Adds one id.

        Returns ``(stop_str, number of its ids)`` for the first stop string, in
        request order, whose ids now end the stream, or None.
        """
        found = None
        for index, (stop_str, token_ids) in enumerate(self._words):
            matched = self._matched[index]
            while matched and token_id != token_ids[matched]:
                matched = self._failure[index][matched - 1]
            if token_id == token_ids[matched]:
                matched += 1
            if matched == len(token_ids):
                if found is None:
                    found = (stop_str, matched)
                matched = self._failure[index][matched - 1]
            self._matched[index] = matched
        return found


def _kmp_failure_table(pattern: Sequence[Any]) -> list[int]:
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


def normalize_stop(stop: str | Iterable[str] | None) -> tuple[str, ...]:
    """The request's stop strings as a tuple without repeats, in order.

    Raises:
        TypeError: A stop string is not a ``str``.
        ValueError: A stop string is empty.
    """
    if stop is None:
        return ()
    stop_strs = [stop] if isinstance(stop, str) else list(stop)
    for stop_str in stop_strs:
        if not isinstance(stop_str, str):
            raise TypeError(f"stop strings must be str, got {type(stop_str).__name__}")
        if not stop_str:
            raise ValueError("stop strings must not be empty")
    return tuple(dict.fromkeys(stop_strs))

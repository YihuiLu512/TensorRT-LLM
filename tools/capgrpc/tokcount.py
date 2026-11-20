# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token count of one serialized openengine.v1.GenerateResponse, without the generated module.

capgrpc.load --count-tokens (raw consume, OpenEngine requests) reads every response this way, so
a stream whose server merges adjacent token events (D1 stage S3: one TokenOutput with k >= 1
TokenInfo entries) is reconciled by tokens instead of by messages. Schema (openengine/v1/
generation.proto): GenerateResponse { string request_id = 1; oneof event { PromptOutput prompt = 2;
TokenOutput token = 3; PrefillReady prefill_ready = 4; GenerationFinished finished = 5;
EngineError error = 6; } Usage usage = 10; }, TokenOutput { optional uint32 output_index = 1;
repeated TokenInfo tokens = 2; string text = 3; }. The count is the number of field-2 entries of
the field-3 submessage; any other event counts 0 tokens.
"""

GENERATE_RESPONSE_TOKEN = 3
TOKEN_OUTPUT_TOKENS = 2


def _varint(b, i):
    x = s = 0
    while True:
        c = b[i]
        i += 1
        x |= (c & 0x7F) << s
        if c < 0x80:
            return x, i
        s += 7
        if s > 63:
            raise ValueError("varint too long")


def _skip(b, i, wt):
    if wt == 0:
        return _varint(b, i)[1]
    if wt == 1:
        return i + 8
    if wt == 5:
        return i + 4
    if wt == 2:
        ln, i = _varint(b, i)
        return i + ln
    raise ValueError("unsupported wire type %d" % wt)


def response_tokens(msg):
    """(tokens, is_token_event) of one serialized GenerateResponse. Raises ValueError when the
    bytes are not a well-formed message (truncated, overrun, group wire types)."""
    n, i, end, tok = 0, 0, len(msg), False
    try:
        while i < end:
            key, i = _varint(msg, i)
            wt = key & 7
            if key >> 3 == GENERATE_RESPONSE_TOKEN and wt == 2:
                tok = True
                ln, i = _varint(msg, i)
                j, e = i, i + ln
                if e > end:
                    raise ValueError("TokenOutput overruns the message")
                while j < e:
                    k2, j = _varint(msg, j)
                    if k2 >> 3 == TOKEN_OUTPUT_TOKENS and k2 & 7 == 2:
                        n += 1
                    j = _skip(msg, j, k2 & 7)
                if j != e:
                    raise ValueError("TokenOutput field overrun")
                i = e
            else:
                i = _skip(msg, i, wt)
    except IndexError:
        raise ValueError("truncated message")
    if i != end:
        raise ValueError("message overrun")
    return n, tok


class KHist:
    """Merge factor histogram: token messages by their token count k."""

    __slots__ = ("counts",)

    def __init__(self):
        self.counts = {}

    def add(self, k):
        self.counts[k] = self.counts.get(k, 0) + 1

    def summary(self):
        n = sum(self.counts.values())
        toks = sum(k * c for k, c in self.counts.items())
        return {"token_msgs": n, "tokens": toks,
                "k_mean": round(toks / n, 4) if n else None,
                "k_max": max(self.counts) if self.counts else None,
                "k_hist": {str(k): c for k, c in sorted(self.counts.items())}}

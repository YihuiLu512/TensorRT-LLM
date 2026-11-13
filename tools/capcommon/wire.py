# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire formats between capkit load generators and capkit servers (INTERFACES.md 4.4, 4.5).

Per-request parameters travel WITH the request; a server never looks them up from the tier table
by arrival time (a request planned at the end of a load tier may arrive in the next drain tier).

- HTTP: max_tokens from the JSON body; k, s, tier and request id from the x-capload-* headers.
- gRPC (request_format capkit): GRPC_REQ header + prompt bytes.
- SSE: nullsse writes content frames, then one usage frame if asked, then the [DONE] sentinel.
  SseCounter is the one implementation of the client-side counting rule.
"""

import re
import struct

from .exitcodes import ConfigError

HDR_ID = "x-capload-id"        # request id (decimal), unique within a launch
HDR_K = "x-capload-k"          # tokens_per_msg of the request's planned tier
HDR_S = "x-capload-s"          # stream_interval of the request's planned tier
HDR_TIER = "x-capload-tier"    # tier_seq of the request's planned time

# req_id, max_tokens, tokens_per_msg, stream_interval, tier_seq; then the prompt bytes
GRPC_REQ = struct.Struct("<qiiii")

DONE_FRAME = b"data: [DONE]\n\n"
_USAGE_RE = re.compile(rb'"completion_tokens":\s*(\d+)')


class WireError(ConfigError):
    pass


def pack_grpc_request(req_id, max_tokens, tokens_per_msg, stream_interval, tier_seq, prompt):
    return GRPC_REQ.pack(req_id, max_tokens, tokens_per_msg, stream_interval, tier_seq) + prompt


def unpack_grpc_request(buf):
    """-> (req_id, max_tokens, tokens_per_msg, stream_interval, tier_seq, prompt bytes)."""
    if len(buf) < GRPC_REQ.size:
        raise WireError("capkit gRPC request shorter than its header")
    return GRPC_REQ.unpack_from(buf, 0) + (bytes(buf[GRPC_REQ.size:]),)


def http_headers(req_id, tokens_per_msg, stream_interval, tier_seq):
    return {HDR_ID: str(req_id), HDR_K: str(tokens_per_msg), HDR_S: str(stream_interval),
            HDR_TIER: str(tier_seq)}


def content_frame(req_id, text):
    """One nullsse content frame (OpenAI completions chunk shape, minimal)."""
    body = ('{"id":"capkit-%d","object":"text_completion","choices":[{"index":0,"text":"%s",'
            '"finish_reason":null}]}' % (req_id, text))
    return b"data: " + body.encode("ascii") + b"\n\n"


def padded_content_frame(req_id, frame_bytes):
    """A content frame of exactly frame_bytes bytes (text is 'x' padding)."""
    base = len(content_frame(req_id, ""))
    if frame_bytes < base:
        raise WireError("chunk bytes %d < minimal frame %d for req %d"
                        % (frame_bytes, base, req_id))
    return content_frame(req_id, "x" * (frame_bytes - base))


def usage_frame(req_id, prompt_tokens, completion_tokens):
    body = ('{"id":"capkit-%d","object":"text_completion","choices":[],"usage":'
            '{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}'
            % (req_id, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens))
    return b"data: " + body.encode("ascii") + b"\n\n"


class SseCounter:
    """Incremental SSE frame counter for one response body (the count-mode receive path).

    feed(data) returns how many frames completed in this chunk (excluding [DONE]). Events are
    delimited by a blank line (b"\\n\\n"); the server side of this contract never emits CRLF.
    After the stream ends: .frames, .done_seen, .completion_tokens (from the last non-[DONE] frame
    if it carries a usage object, else None).
    """

    __slots__ = ("frames", "done_seen", "_cur", "_last", "_tail_nl")

    def __init__(self):
        self.frames = 0
        self.done_seen = False
        self._cur = bytearray()
        self._last = b""
        self._tail_nl = False

    def feed(self, data):
        new = 0
        start = 0
        n = len(data)
        if self._tail_nl and n and data[0:1] == b"\n":
            # a delimiter split across chunks: the previous chunk ended with '\n'
            new += self._close(bytes(self._cur[:-1]))
            self._cur.clear()
            start = 1
        while True:
            j = data.find(b"\n\n", start)
            if j < 0:
                break
            self._cur += data[start:j]
            new += self._close(bytes(self._cur))
            self._cur.clear()
            start = j + 2
        self._cur += data[start:]
        self._tail_nl = bool(self._cur) and self._cur[-1:] == b"\n"
        return new

    def _close(self, frame):
        if frame == b"data: [DONE]":
            self.done_seen = True
            return 0
        if not frame:
            return 0
        self.frames += 1
        self._last = frame
        return 1

    @property
    def completion_tokens(self):
        m = _USAGE_RE.search(self._last) if b'"usage"' in self._last else None
        return int(m.group(1)) if m else None

    @property
    def leftover(self):
        """Bytes after the last delimiter (non-empty at end of stream = truncated frame)."""
        return len(self._cur)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP/1.1 pieces of capload: pre-rendered request bytes, response head parsing, an incremental
chunked-transfer decoder, and the optional full SSE parser (cross-check of the counting rule).

Request bytes are rendered once at start-up per (tier, prompt item); the send path only joins
a fixed prefix, the decimal request id and that suffix (no JSON serialization while measuring).
The x-capload-* header names and values come from capcommon.wire.http_headers (INTERFACES 4.4).
"""

import json

from capcommon import wire
from capcommon.exitcodes import ConfigError
from capcommon.schedule import prompt_index

ENDPOINTS = ("/v1/completions", "/v1/chat/completions")
PROMPTS_SCHEMA = "capkit.prompts/1"
PROMPT_KINDS = ("token_ids", "text")


class ProtocolError(Exception):
    """Malformed HTTP / SSE from the peer (failure reason protocol_error)."""


def load_prompts(path):
    try:
        with open(path, "rb") as f:
            doc = json.loads(f.read().decode("utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError("%s: cannot read prompt pool: %s" % (path, e))
    if not isinstance(doc, dict) or doc.get("schema") != PROMPTS_SCHEMA:
        raise ConfigError("%s: schema must be %s" % (path, PROMPTS_SCHEMA))
    sets = doc.get("sets")
    if not isinstance(sets, dict) or not sets:
        raise ConfigError("%s: sets must be a non-empty object" % path)
    for name, s in sets.items():
        if not isinstance(s, dict) or s.get("kind") not in PROMPT_KINDS:
            raise ConfigError("%s: set %s needs kind in %s" % (path, name, PROMPT_KINDS))
        items = s.get("items")
        if not isinstance(items, list) or not items:
            raise ConfigError("%s: set %s has no items" % (path, name))
        for it in items:
            if s["kind"] == "text" and not isinstance(it, str):
                raise ConfigError("%s: set %s: text items must be strings" % (path, name))
            if s["kind"] == "token_ids" and not (
                    isinstance(it, list) and it
                    and all(isinstance(x, int) and not isinstance(x, bool) for x in it)):
                raise ConfigError("%s: set %s: token_ids items must be non-empty int lists"
                                  % (path, name))
    return sets


# Fields render_body writes itself (--body-extra may not name them).
BODY_KEYS = ("model", "max_tokens", "stream", "stream_options", "ignore_eos", "prompt",
             "messages", "prompt_token_ids")


def render_body(endpoint, model, max_tokens, kind, item, extra=None):
    """JSON request body. The fixed per-request content (INTERFACES 4.4) comes first, then the
    --body-extra fields; the prompt is the last key so a sink can find max_tokens /
    include_usage near the start."""
    d = {"model": model, "max_tokens": max_tokens, "stream": True,
         "stream_options": {"include_usage": True}, "ignore_eos": True}
    if extra:
        d.update(extra)
    if endpoint == "/v1/completions":
        d["prompt"] = item
    elif endpoint == "/v1/chat/completions":
        if kind == "text":
            d["messages"] = [{"role": "user", "content": item}]
        else:  # token ids travel in TRT-LLM's prompt_token_ids extension; messages is required
            d["messages"] = [{"role": "user", "content": ""}]
            d["prompt_token_ids"] = item
    else:
        raise ConfigError("endpoint must be one of %s" % (ENDPOINTS,))
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class RequestBuilder:
    """Pre-rendered request bytes for the (tier, prompt item) pairs one worker will send."""

    def __init__(self, endpoint, model, host_header, conn_close, tier_table, prompt_sets,
                 needed, body_extra=None):
        """needed: {tier_seq: set of request ids} -- only these (tier, item) pairs are built."""
        if endpoint not in ENDPOINTS:
            raise ConfigError("endpoint must be one of %s" % (ENDPOINTS,))
        conn = b"close" if conn_close else b"keep-alive"
        self.prefix = (b"POST " + endpoint.encode("ascii") + b" HTTP/1.1\r\nHost: "
                       + host_header.encode("ascii")
                       + b"\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\n"
                       + b"User-Agent: capload\r\nConnection: " + conn + b"\r\n"
                       + wire.HDR_ID.encode("ascii") + b": ")
        self.suffix = {}  # (tier_seq, item index) -> bytes
        self.set_sizes = {}  # tier_seq -> len(items)
        self.total_bytes = 0
        bodies = {}
        for seq, rids in sorted(needed.items()):
            t = tier_table.tiers[seq]
            if not t.is_load:
                raise ConfigError("schedule has requests in non-load tier %d" % seq)
            ps = prompt_sets.get(t.prompt_set)
            if ps is None:
                raise ConfigError("tier %d prompt_set %r not in the prompt pool"
                                  % (seq, t.prompt_set))
            n_items = len(ps["items"])
            self.set_sizes[seq] = n_items
            hdrs = wire.http_headers(0, t.tokens_per_msg, t.stream_interval, seq)
            lines = b"".join(("%s: %s\r\n" % (k, v)).encode("ascii")
                             for k, v in hdrs.items() if k != wire.HDR_ID)
            for idx in sorted({prompt_index(r, n_items) for r in rids}):
                key = (t.prompt_set, t.max_tokens, idx)
                body = bodies.get(key)
                if body is None:
                    body = render_body(endpoint, model, t.max_tokens, ps["kind"],
                                       ps["items"][idx], body_extra)
                    bodies[key] = body
                suf = (b"\r\n" + lines + b"Content-Length: " + str(len(body)).encode("ascii")
                       + b"\r\n\r\n" + body)
                self.suffix[(seq, idx)] = suf
                self.total_bytes += len(suf)

    def render(self, req_id, tier_seq):
        return b"".join((self.prefix, str(req_id).encode("ascii"),
                         self.suffix[(tier_seq, req_id % self.set_sizes[tier_seq])]))


def parse_head(buf):
    """Parse a response head from the start of buf.

    Returns None when the head is incomplete, else (status, headers {lower name: value},
    head_len). Raises ProtocolError on a malformed head.
    """
    end = buf.find(b"\r\n\r\n")
    if end < 0:
        if len(buf) > 65536:
            raise ProtocolError("response head longer than 64 KiB")
        return None
    lines = bytes(buf[:end]).split(b"\r\n")
    parts = lines[0].split(b" ", 2)
    if len(parts) < 2 or not parts[0].startswith(b"HTTP/1."):
        raise ProtocolError("bad status line %r" % lines[0][:80])
    try:
        status = int(parts[1])
    except ValueError:
        raise ProtocolError("bad status code %r" % parts[1][:16])
    headers = {}
    for ln in lines[1:]:
        k, sep, v = ln.partition(b":")
        if not sep:
            raise ProtocolError("bad header line %r" % ln[:80])
        headers[k.strip().lower().decode("latin-1")] = v.strip().decode("latin-1")
    return status, headers, end + 4


class ChunkedDecoder:
    """Incremental Transfer-Encoding: chunked decoder.

    feed(data) -> (payload bytes, consumed). After the terminal chunk and trailers, .done is True
    and `consumed` tells where the body ended inside `data`.
    """

    __slots__ = ("done", "_state", "_remain", "_line")

    def __init__(self):
        self.done = False
        self._state = 0  # 0 size line, 1 data, 2 CRLF after data, 3 trailers
        self._remain = 0
        self._line = b""

    def feed(self, data):
        out = []
        i = 0
        n = len(data)
        while i < n and not self.done:
            st = self._state
            if st == 1:
                take = self._remain if self._remain < n - i else n - i
                out.append(data[i:i + take])
                i += take
                self._remain -= take
                if self._remain == 0:
                    self._state = 2
                    self._remain = 2
            elif st == 2:
                c = data[i]
                if not ((self._remain == 2 and c == 13) or c == 10):
                    raise ProtocolError("missing CRLF after chunk data")
                i += 1
                if c == 10:
                    self._state = 0
                else:
                    self._remain = 1
            else:
                j = data.find(b"\n", i)
                if j < 0:
                    self._line += data[i:]
                    if len(self._line) > 4096:
                        raise ProtocolError("chunk size line too long")
                    i = n
                    break
                line = (self._line + data[i:j]).rstrip(b"\r")
                self._line = b""
                i = j + 1
                if st == 0:
                    try:
                        size = int(line.split(b";", 1)[0].strip(), 16)
                    except ValueError:
                        raise ProtocolError("bad chunk size line %r" % line[:40])
                    if size < 0:
                        raise ProtocolError("negative chunk size")
                    if size == 0:
                        self._state = 3
                    else:
                        self._state = 1
                        self._remain = size
                elif not line:  # st == 3: blank line ends the trailers
                    self.done = True
        if not out:
            return b"", i
        return (out[0] if len(out) == 1 else b"".join(out)), i


class FullSseParser:
    """Full-parse receive mode: split SSE events and json.loads every data payload.

    Used only to cross-check the cheap counting rule (wire.SseCounter): frames here must equal
    SseCounter.frames, and the usage completion_tokens must agree.
    """

    __slots__ = ("frames", "done_seen", "completion_tokens", "errors", "id_mismatch", "_buf",
                 "_expect_id")

    def __init__(self, req_id):
        self.frames = 0
        self.done_seen = False
        self.completion_tokens = None
        self.errors = 0
        self.id_mismatch = 0
        self._buf = b""
        self._expect_id = "capkit-%d" % req_id

    def feed(self, data):
        buf = self._buf + data
        start = 0
        while True:
            j = buf.find(b"\n\n", start)
            if j < 0:
                break
            self._event(buf[start:j])
            start = j + 2
        self._buf = buf[start:]

    def _event(self, ev):
        if not ev:
            return
        payload = []
        for line in ev.split(b"\n"):
            if line.startswith(b"data:"):
                payload.append(line[5:].lstrip(b" "))
        if not payload:
            self.frames += 1  # a non-data event still counts as a frame under SseCounter
            return
        data = b"\n".join(payload)
        if data == b"[DONE]":
            self.done_seen = True
            return
        self.frames += 1
        try:
            obj = json.loads(data)
        except ValueError:
            self.errors += 1
            return
        if not isinstance(obj, dict):
            self.errors += 1
            return
        oid = obj.get("id")
        if isinstance(oid, str) and oid.startswith("capkit-") and oid != self._expect_id:
            self.id_mismatch += 1
        usage = obj.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
            self.completion_tokens = usage["completion_tokens"]

    @property
    def leftover(self):
        return len(self._buf)

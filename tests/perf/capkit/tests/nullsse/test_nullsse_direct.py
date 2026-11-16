# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nullsse driven by raw sockets: foreign clients, errors, idle close, beat spacing, SO_REUSEPORT."""

import json
import socket
import time

from capcommon import wire
from capcommon.lifecycle import wait_port_file
from capload.httpwire import ChunkedDecoder, parse_head

from capload_support import MS, table_doc


def _doc(beat_ms=10):
    spec = [("warmup", 3000, 10.0, {"n": 5, "beat": beat_ms * MS}),
            ("drain", 500, 0.0, {"beat": beat_ms * MS})]
    return table_doc(spec)


def _wait_origin(lz):
    while time.monotonic_ns() < lz.origin + 20 * MS:
        time.sleep(0.01)


def _post(sock, body, headers=b""):
    sock.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                 + headers + b"Content-Length: %d\r\n\r\n" % len(body) + body)


def _read_response(sock):
    """-> (status, headers, payload bytes, [arrival ns per recv])"""
    buf = b""
    while True:
        d = sock.recv(65536)
        assert d, "connection closed before the response head"
        buf += d
        h = parse_head(buf)
        if h is not None:
            break
    st, hdrs, n = h
    rest = buf[n:]
    if "content-length" in hdrs:
        need = int(hdrs["content-length"])
        while len(rest) < need:
            rest += sock.recv(65536)
        return st, hdrs, rest, []
    dec = ChunkedDecoder()
    payload, _ = dec.feed(rest)
    times = []
    while not dec.done:
        d = sock.recv(65536)
        assert d
        times.append(time.monotonic_ns())
        p, _ = dec.feed(d)
        payload += p
    return st, hdrs, payload, times


def _start(lz, **kw):
    p = lz.start_sink(**kw)
    host, port, _ = wait_port_file(lz.port_file, 30)
    _wait_origin(lz)
    return p, host, port


def test_foreign_client_without_capload_headers(launch):
    lz = launch(_doc())
    p, host, port = _start(lz, workers=1)
    with socket.create_connection((host, port)) as s:
        _post(s, json.dumps({"prompt": "x", "stream": True, "max_tokens": 4}).encode())
        st, _h, payload, _t = _read_response(s)
        assert st == 200
        ctr = wire.SseCounter()
        ctr.feed(payload)
        assert ctr.frames == 4 and ctr.done_seen and ctr.completion_tokens is None  # no usage
        # GET /health is fine; an unknown path is a counted bad request
        s.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
        assert _read_response(s)[0] == 200
        s.sendall(b"POST /nope HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n")
        assert _read_response(s)[0] == 404
    assert lz.stop(p) == 0
    c = lz.summaries("nullsse", 0)[0]["counts"]
    assert c["foreign_requests"] == 1 and c["requests"] == 1
    assert c["frames_sent"] == 4 and c["bad_requests"] == 1


def test_idle_close(launch):
    lz = launch(_doc())
    p, host, port = _start(lz, workers=1, idle_close_s=0.2)
    with socket.create_connection((host, port)) as s:
        s.settimeout(3)
        t0 = time.monotonic()
        assert s.recv(10) == b""  # closed by the server
        assert time.monotonic() - t0 >= 0.15
    assert lz.stop(p) == 0
    assert lz.summaries("nullsse", 0)[0]["counts"]["idle_closed"] == 1


def test_beat_spacing_and_usage(launch):
    lz = launch(_doc(beat_ms=20))
    p, host, port = _start(lz, workers=1, pacing="beat")
    with socket.create_connection((host, port)) as s:
        body = json.dumps({"max_tokens": 6, "stream": True,
                           "stream_options": {"include_usage": True}, "prompt": [1]}).encode()
        hdr = b"x-capload-id: 42\r\nx-capload-k: 1\r\nx-capload-s: 1\r\nx-capload-tier: 0\r\n"
        t0 = time.monotonic_ns()
        _post(s, body, hdr)
        st, h, payload, _times = _read_response(s)
        t1 = time.monotonic_ns()
    assert st == 200 and h["x-capload-id"] == "42"
    ctr = wire.SseCounter()
    ctr.feed(payload)
    assert ctr.frames == 7 and ctr.completion_tokens == 6 and ctr.done_seen
    assert t1 - t0 >= 5 * 20 * MS  # 6 messages on 6 consecutive 20 ms steps
    assert lz.stop(p) == 0
    c = lz.summaries("nullsse", 0)[0]["counts"]
    assert c["frames_sent"] == 7 and c["requests"] == 1 and c["foreign_requests"] == 0


def test_reuseport_spreads_accepts(launch):
    lz = launch(_doc())
    p, host, port = _start(lz, workers=3)
    for _ in range(30):
        socket.create_connection((host, port)).close()
    time.sleep(0.3)
    assert lz.stop(p) == 0
    per_worker = [s["counts"]["accepts"] for s in lz.summaries("nullsse", 0)]
    assert len(per_worker) == 3 and sum(per_worker) == 30
    assert sum(1 for a in per_worker if a > 0) >= 2

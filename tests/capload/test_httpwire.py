# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of the HTTP pieces (no network)."""

import json

import pytest

from capcommon import wire
from capload.httpwire import (ChunkedDecoder, FullSseParser, ProtocolError, RequestBuilder,
                              parse_head, render_body)
from capload.sink import chunk, min_chunk_bytes

from capload_support import basic_spec, table_doc
from capcommon.tiers import TierTable


def _stream(rid=7, m=3, usage=True):
    body = b"".join(chunk(wire.padded_content_frame(rid, 120)) for _ in range(m))
    if usage:
        body += chunk(wire.usage_frame(rid, 0, m))
    return body + chunk(wire.DONE_FRAME) + b"0\r\n\r\n"


@pytest.mark.parametrize("step", [1, 2, 3, 7, 64, 10000])
def test_chunked_decoder_any_split(step):
    raw = _stream()
    dec = ChunkedDecoder()
    ctr = wire.SseCounter()
    out = b""
    for i in range(0, len(raw), step):
        payload, used = dec.feed(raw[i:i + step])
        out += payload
        ctr.feed(payload)
        if dec.done:
            assert i + used == len(raw)
    assert dec.done
    assert ctr.frames == 4 and ctr.done_seen and ctr.leftover == 0
    assert ctr.completion_tokens == 3


def test_chunked_decoder_rejects_garbage():
    with pytest.raises(ProtocolError):
        ChunkedDecoder().feed(b"zz\r\nabc")
    with pytest.raises(ProtocolError):
        ChunkedDecoder().feed(b"3\r\nabcXY")


def test_parse_head():
    assert parse_head(b"HTTP/1.1 200 OK\r\nX-A: 1") is None
    raw = b"HTTP/1.1 503 Busy\r\nContent-Length: 4\r\nX-Capload-Id: 9\r\n\r\nbusy"
    st, h, n = parse_head(raw)
    assert st == 503 and h["content-length"] == "4" and h["x-capload-id"] == "9"
    assert raw[n:] == b"busy"
    with pytest.raises(ProtocolError):
        parse_head(b"FOO\r\n\r\n")


def test_full_parser_matches_counter():
    raw = _stream(rid=5, m=4)
    dec = ChunkedDecoder()
    payload, _ = dec.feed(raw)
    full = FullSseParser(5)
    ctr = wire.SseCounter()
    full.feed(payload)
    ctr.feed(payload)
    assert full.frames == ctr.frames == 5
    assert full.completion_tokens == ctr.completion_tokens == 4
    assert full.done_seen and not full.errors and not full.id_mismatch
    other = FullSseParser(6)
    other.feed(payload)
    assert other.id_mismatch == 5  # 4 content frames + the usage frame carry capkit-5


def test_request_builder_headers_and_body(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(table_doc(basic_spec(up_kw={"k": 2, "s": 2}))))
    tt = TierTable.load(str(p))
    sets = {"p": {"kind": "token_ids", "items": [[1, 2], [3]]}}
    b = RequestBuilder("/v1/completions", "m", "h:1", False, tt, sets, {2: {10, 11}})
    req = b.render(11, 2)
    head, _, body = req.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    assert lines[0] == b"POST /v1/completions HTTP/1.1"
    hd = dict(ln.split(b": ", 1) for ln in lines[1:])
    assert hd[b"x-capload-id"] == b"11" and hd[b"x-capload-k"] == b"2"
    assert hd[b"x-capload-s"] == b"2" and hd[b"x-capload-tier"] == b"2"
    assert int(hd[b"Content-Length"]) == len(body)
    d = json.loads(body)
    assert d["prompt"] == [3] and d["max_tokens"] == 5 and d["ignore_eos"] is True
    assert d["stream"] is True and d["stream_options"] == {"include_usage": True}
    chat = json.loads(render_body("/v1/chat/completions", "m", 4, "token_ids", [1, 2]))
    assert chat["prompt_token_ids"] == [1, 2] and chat["messages"][0]["role"] == "user"


def test_min_chunk_bytes_fits_any_id():
    assert len(wire.padded_content_frame(2 ** 63 - 1, min_chunk_bytes())) == min_chunk_bytes()


@pytest.mark.parametrize("target,ok", [("h:8000", True), ("h:\u00b2", False),
                                       ("h:\u0661\u0662", False), ("h:0", False)])
def test_target_port_is_ascii_digits(target, ok):
    """--target HOST:PORT: the port is ASCII digits in (0, 65536); '\u00b2' (str.isdigit() but
    not int()) is a ConfigError (exit 3), never an uncaught ValueError."""
    import types
    from capcommon.exitcodes import ConfigError
    from capload.main import config_from_args
    args = types.SimpleNamespace(workers=1, chunk_sample_mod=1, fionread_every_ms=1,
                                 lag_period_us=1, max_records_per_worker=0,
                                 slow_read_pause_us=0, wave_gap_ns=0, host_header=None,
                                 target=target)
    common = types.SimpleNamespace(cpus=None)
    if ok:
        try:
            config_from_args(args, common)
        except ConfigError as e:
            assert "--target" not in str(e)
        except AttributeError:
            pass  # past the target check (the namespace has only what precedes it)
    else:
        with pytest.raises(ConfigError, match="--target must be HOST:PORT"):
            config_from_args(args, common)


def test_body_extra_fields_and_refusals():
    from capcommon.exitcodes import ConfigError
    from capload.main import parse_body_extra
    d = json.loads(render_body("/v1/completions", "m", 4, "token_ids", [1, 2],
                               {"detokenize": False}))
    assert d["detokenize"] is False and d["max_tokens"] == 4 and list(d)[-1] == "prompt"
    assert parse_body_extra(None) is None
    assert parse_body_extra('{"detokenize": false}') == {"detokenize": False}
    for bad in ('[1]', '{}', 'x', '{"max_tokens": 3}', '{"prompt": "p"}'):
        with pytest.raises(ConfigError):
            parse_body_extra(bad)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capstubs.prebaked: the protobuf wire reader (checked against real protobuf parsing as the
oracle), the builder (pb2 test double, and the product's generated module from the probe tree),
the loader's checks (one broken document per check) and the container-only command line."""

import base64
import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest

import oe_double
from capcommon import expect
from capcommon.exitcodes import ConfigError
from capcommon.inject import InjectSpec
from capcommon.tiers import TierTable
from capgrpc.reqbytes import build_openengine, encode_pb_string_field
from capstubs import prebaked as P

from conftest import HERE, TOOLS, real_pb2, table_doc

MS = 1_000_000


def _tt(tmp_path, spec, launch_inject=None):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(table_doc(spec, launch_inject=launch_inject)))
    return TierTable.load(str(p))


def _two_tiers(tmp_path, k=2, s=1, inject=None):
    return _tt(tmp_path, [("warmup", 200, 10.0, {"n": 5, "k": k, "s": s, "inject": inject}),
                          ("drain", 200, 0.0, {"k": k, "s": s}),
                          ("up", 200, 20.0, {"n": 7, "k": k, "s": s}),
                          ("drain", 200, 0.0, {"k": k, "s": s})])


# ---- wire reader ---------------------------------------------------------------------------------

def test_parse_request_matches_protobuf_merge_semantics():
    pb2 = oe_double
    req = pb2.GenerateRequest(request_id="7", model="m")
    req.token_ids.ids.extend([1, 2, 300000])
    req.stopping.max_tokens = 9
    req.stopping.ignore_eos = True
    body = req.SerializeToString()
    assert P.parse_generate_request(body) == (b"7", 9)
    # a prefix request_id loses to the body's (last one wins) ...
    cat = encode_pb_string_field(1, "5") + body
    assert P.parse_generate_request(cat)[0] == pb2.GenerateRequest.FromString(cat).request_id \
        .encode()
    # ... a body without one keeps the prefix (capgrpc.load's openengine format)
    req.ClearField("request_id")
    cat = encode_pb_string_field(1, "123") + req.SerializeToString()
    assert P.parse_generate_request(cat) == (b"123", 9)
    # a later stopping message merges: max_tokens from it wins only when it sets one
    extra = pb2.GenerateRequest()
    extra.stopping.ignore_eos = False
    cat2 = cat + extra.SerializeToString()
    assert P.parse_generate_request(cat2)[1] == pb2.GenerateRequest.FromString(cat2) \
        .stopping.max_tokens == 9
    extra.stopping.max_tokens = 3
    cat3 = cat + extra.SerializeToString()
    assert P.parse_generate_request(cat3)[1] == pb2.GenerateRequest.FromString(cat3) \
        .stopping.max_tokens == 3


def test_parse_request_without_max_tokens():
    req = oe_double.GenerateRequest(request_id="1", prompt="hi")
    assert P.parse_generate_request(req.SerializeToString()) == (b"1", None)
    req.stopping.ignore_eos = True
    assert P.parse_generate_request(req.SerializeToString()) == (b"1", None)
    assert P.parse_generate_request(b"") == (b"", None)


@pytest.mark.parametrize("bad", [
    b"\x08",                      # truncated varint value
    b"\x0a\x05abc",               # length-delimited field past the end
    b"\x0b",                      # group (wire type 3)
    b"\x00\x01",                  # field number 0
    b"\x30\x05",                  # stopping (6) as a varint
    b"\x32\x02\x0a\x00",          # stopping.max_tokens as length-delimited
    b"\x08" + b"\xff" * 10 + b"\x01",  # varint longer than 10 bytes
    b"\x0d\x01\x02",              # fixed32 past the end
])
def test_parse_request_rejects_malformed(bad):
    with pytest.raises(P.WireFormatError):
        P.parse_generate_request(bad)


def test_parse_request_skips_fixed_fields():
    body = b"\x79" + b"\x00" * 8 + b"\x7d" + b"\x00" * 4  # field 15 fixed64, field 15 fixed32
    assert P.parse_generate_request(body + b"\x32\x02\x08\x04") == (b"", 4)


@pytest.mark.parametrize("rid,want", [
    (b"0", 0), (b"123", 123), (b"999999999999999999", 999999999999999999),
    (b"", -1), (b"007", -1), (b"12a", -1), (b"-1", -1), (b"+1", -1), (b" 1", -1),
    (b"1234567890123456789", -1), ("١٢".encode(), -1), ("²".encode(), -1), (b"cmpl-1", -1),
])
def test_decimal_request_id(rid, want):
    assert P.decimal_request_id(rid) == want


# ---- builder ---------------------------------------------------------------------------------

def test_build_shapes_tokens_and_usage(tmp_path):
    tt = _two_tiers(tmp_path, k=2, s=1)
    doc = P.build_prebaked(oe_double, tt, [11, 12, 13], 4)
    assert [(x["max_tokens"], x["tokens_per_msg"], x["stream_interval"])
            for x in doc["streams"]] == [(5, 2, 1), (7, 2, 1)]
    for st in doc["streams"]:
        n = st["max_tokens"]
        assert [m["tokens"] for m in st["messages"]] == expect.msg_tokens(n, 2, 1, "product")
        ids = []
        for m in st["messages"]:
            r = oe_double.GenerateResponse.FromString(base64.b64decode(m["b64"]))
            assert r.WhichOneof("event") == "token" and r.token.HasField("output_index")
            assert r.request_id == "" and not r.HasField("usage") and r.token.text == ""
            ids += [t.token_id for t in r.token.tokens]
        assert ids == [[11, 12, 13][i % 3] for i in range(n)]  # one walk over the whole stream
        f = oe_double.GenerateResponse.FromString(base64.b64decode(st["finished"]["b64"]))
        assert f.WhichOneof("event") == "finished" and f.finished.HasField("output_index")
        assert f.finished.reason == oe_double.FINISH_REASON_LENGTH and f.request_id == ""
        assert (f.usage.prompt_tokens, f.usage.completion_tokens, f.usage.total_tokens) == \
            (4, n, 4 + n)
    pb = P.Prebaked(doc)
    sb = pb.lookup(7, 2, 1)
    offs = expect.msg_step_offsets(7, 2, 1, "product")
    assert sb.offs == tuple(offs) + (offs[-1],) and len(sb.msgs) == len(offs) + 1
    assert sb.cum[-1] == sb.cum[-2] == 7 and sb.n_token_msgs == len(offs)
    assert pb.lookup(7, 1, 1) is None and pb.lookup(6, 2, 1) is None


def test_build_product_gate_with_stream_interval(tmp_path):
    tt = _tt(tmp_path, [("warmup", 200, 10.0, {"n": 10, "k": 4, "s": 2}),
                        ("drain", 200, 0.0, {"k": 4, "s": 2})])
    pb = P.Prebaked(P.build_prebaked(oe_double, tt, [5], 0))
    sb = pb.lookup(10, 4, 2)
    assert [sb.cum[j + 1] - sb.cum[j] for j in range(sb.n_token_msgs)] == \
        expect.msg_tokens(10, 4, 2, "product") == [2, 2, 4, 2]


def test_short_shapes_from_table_and_command_line(tmp_path):
    short = [{"site": "short.req", "mode": "short", "short_tokens": 2, "role": "stub"}]
    tt = _two_tiers(tmp_path, k=1, s=1, inject=short)
    assert P.table_short_amounts(tt) == [2]
    doc = P.build_prebaked(oe_double, tt, [1], 0, short_tokens=[9])
    got = sorted(x["max_tokens"] for x in doc["streams"])
    # table N 5, 7; short 2 -> 3, 5; short 9 -> 1 (nullsse's rule: never below 1)
    assert got == [1, 3, 5, 7] and doc["short_tokens"] == [2, 9]
    assert P.shorted(7, 2) == 5 and P.shorted(3, 5) == 1 and P.shorted(3, 3) == 1
    other = [InjectSpec("short.req", "busy", delta_ns=5), InjectSpec("send.msg", "busy",
                                                                       delta_ns=5)]
    assert P.short_amounts(other) == []


def test_short_shapes_are_not_compounded(tmp_path):
    tt = _tt(tmp_path, [("warmup", 200, 10.0, {"n": 7}), ("drain", 200, 0.0, {})])
    # each short amount applies to the table's N only: 7-2, 7-3, never 7-2-3
    assert P.required_shapes(tt, [2, 3]) == [(4, 1, 1), (5, 1, 1), (7, 1, 1)]
    assert P.required_shapes(tt) == [(7, 1, 1)]


def test_uniform_shape_required(tmp_path):
    tt = _tt(tmp_path, [("warmup", 200, 10.0, {"n": 5, "k": 1}), ("drain", 200, 0.0, {}),
                        ("up", 200, 20.0, {"n": 5, "k": 2, "s": 1}), ("drain", 200, 0.0, {})])
    with pytest.raises(ConfigError, match="one \\(tokens_per_msg, stream_interval\\)"):
        P.uniform_msg_shape(tt)
    with pytest.raises(ConfigError):
        P.build_prebaked(oe_double, tt, [1], 0)
    # drain tiers do not count
    tt2 = _tt(tmp_path, [("warmup", 200, 10.0, {"n": 5, "k": 2}), ("drain", 200, 0.0, {"k": 1})])
    assert P.uniform_msg_shape(tt2) == (2, 1)


@pytest.mark.parametrize("ov", [{("Usage", "completion_tokens"): 7},
                                {("GenerateRequest", "stopping"): 8},
                                {("TokenOutput", "tokens"): 4}])
def test_field_numbers_checked(tmp_path, ov):
    bad = oe_double.make(ov, pkg="fakeoe_bad%d" % len(str(ov)))
    with pytest.raises(ConfigError, match="disagrees with the wire reader"):
        P.check_field_numbers(bad)
    with pytest.raises(ConfigError):
        P.build_prebaked(bad, _two_tiers(tmp_path), [1], 0)
    P.check_field_numbers(oe_double)


@pytest.mark.parametrize("ids,prompt", [([], 0), ([-1], 0), ([1 << 32], 0), ([True], 0),
                                        ([1], -1), ([1], 1 << 32)])
def test_build_rejects_bad_token_ids_and_prompt(tmp_path, ids, prompt):
    with pytest.raises(ConfigError):
        P.build_prebaked(oe_double, _two_tiers(tmp_path), ids, prompt)


# ---- loader checks -------------------------------------------------------------------------------

def _b64(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "b64": base64.b64encode(raw).decode()}


def _good(tmp_path):
    return P.build_prebaked(oe_double, _two_tiers(tmp_path, k=2, s=1), [11, 12, 13], 4)


def _rsp(**kw):
    return oe_double.GenerateResponse(**kw).SerializeToString()


def _tok(n, rid=""):
    return _rsp(request_id=rid, token=oe_double.TokenOutput(
        output_index=0, tokens=[oe_double.TokenInfo(token_id=1)] * n))


def _break(doc, what):
    d = copy.deepcopy(doc)
    st = d["streams"][0]
    if what == "schema":
        d["schema"] = "capkit.prebaked/0"
    elif what == "method":
        d["method"] = "Generate"
    elif what == "gate":
        d["gate"] = "every_step_plus_final"
    elif what == "token_ids":
        d["token_ids"] = []
    elif what == "reason0":
        d["finish_reason_value"] = 0
    elif what == "no_streams":
        d["streams"] = []
    elif what == "dup":
        d["streams"].append(copy.deepcopy(st))
    elif what == "bad_shape":
        st["tokens_per_msg"] = 3  # k % s == 0 but the messages were built for k = 2
    elif what == "count":
        st["messages"].pop()
    elif what == "sha":
        st["messages"][0]["sha256"] = "0" * 64
    elif what == "b64":
        st["messages"][0]["b64"] = "!!!"
    elif what == "declared_tokens":
        st["messages"][0]["tokens"] += 1
    elif what == "wire_tokens":
        st["messages"][0] = dict(_b64(_tok(st["messages"][0]["tokens"] + 1)),
                                 tokens=st["messages"][0]["tokens"])
    elif what == "request_id":
        st["messages"][0] = dict(_b64(_tok(st["messages"][0]["tokens"], rid="x")),
                                 tokens=st["messages"][0]["tokens"])
    elif what == "two_events":
        raw = _tok(st["messages"][0]["tokens"]) + base64.b64decode(st["finished"]["b64"])
        st["messages"][0] = dict(_b64(raw), tokens=st["messages"][0]["tokens"])
    elif what == "no_output_index":
        nop = oe_double.make(presence=False, pkg="fakeoe_nop")
        raw = nop.GenerateResponse(token=nop.TokenOutput(
            output_index=0, tokens=[nop.TokenInfo(token_id=1)] * st["messages"][0]["tokens"]))
        st["messages"][0] = dict(_b64(raw.SerializeToString()), tokens=st["messages"][0]["tokens"])
    elif what == "usage_on_token":
        raw = oe_double.GenerateResponse(
            token=oe_double.TokenOutput(output_index=0, tokens=[oe_double.TokenInfo()] * 2),
            usage=oe_double.Usage(completion_tokens=2)).SerializeToString()
        st["messages"][0] = dict(_b64(raw), tokens=2)
    elif what == "finished_completion":
        raw = _rsp(finished=oe_double.GenerationFinished(output_index=0, reason=2),
                   usage=oe_double.Usage(prompt_tokens=4, completion_tokens=st["max_tokens"] - 1,
                                         total_tokens=3 + st["max_tokens"]))
        st["finished"] = _b64(raw)
    elif what == "finished_reason":
        raw = _rsp(finished=oe_double.GenerationFinished(output_index=0, reason=1),
                   usage=oe_double.Usage(prompt_tokens=4, completion_tokens=st["max_tokens"],
                                         total_tokens=4 + st["max_tokens"]))
        st["finished"] = _b64(raw)
    elif what == "finished_no_usage":
        st["finished"] = _b64(_rsp(finished=oe_double.GenerationFinished(output_index=0,
                                                                          reason=2)))
    elif what == "finished_is_token":
        st["finished"] = _b64(_tok(1))
    elif what == "malformed":
        st["finished"] = _b64(b"\x0a\x09abc")
    elif what == "rid_varint":
        st["finished"] = _b64(b"\x08\x01" + base64.b64decode(st["finished"]["b64"]))
    else:
        raise AssertionError(what)
    return d


BREAKS = ["schema", "method", "gate", "token_ids", "reason0", "no_streams", "dup", "bad_shape",
          "count", "sha", "b64", "declared_tokens", "wire_tokens", "request_id", "two_events",
          "no_output_index", "usage_on_token", "finished_completion", "finished_reason",
          "finished_no_usage", "finished_is_token", "malformed", "rid_varint"]


def test_loader_accepts_the_built_document(tmp_path):
    doc = _good(tmp_path)
    p = tmp_path / "pb.json"
    p.write_text(json.dumps(doc))
    pb = P.load_prebaked(str(p))
    assert pb.sha256 == hashlib.sha256(p.read_bytes()).hexdigest() and pb.prompt_tokens == 4
    assert sorted(pb.table) == [(5, 2, 1), (7, 2, 1)] and pb.method == P.GENERATE_METHOD


@pytest.mark.parametrize("what", BREAKS)
def test_loader_rejects(tmp_path, what):
    with pytest.raises(ConfigError):
        P.Prebaked(_break(_good(tmp_path), what))


def test_load_prebaked_unreadable(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        P.load_prebaked(str(tmp_path / "missing.json"))
    (tmp_path / "x.json").write_text("{")
    with pytest.raises(ConfigError, match="not JSON"):
        P.load_prebaked(str(tmp_path / "x.json"))


# ---- the product's generated module (probe tree) -------------------------------------------------

def test_real_generated_module(tmp_path):
    gen, oe = real_pb2()
    P.check_field_numbers(gen)
    svc = oe.DESCRIPTOR.services_by_name["Inference"].methods_by_name["Generate"]
    assert P.GENERATE_METHOD == "/%s/%s" % (svc.containing_service.full_name, svc.name)
    tt = _tt(tmp_path, [("warmup", 200, 10.0, {"n": 9, "k": 3, "s": 1}),
                        ("drain", 200, 0.0, {"k": 3, "s": 1})])
    doc = P.build_prebaked(gen, tt, [0, 7], 5)
    sb = P.Prebaked(doc).lookup(9, 3, 1)
    # the product client's per-response contract (openengine_client/stream.py _validate)
    for j, raw in enumerate(sb.msgs):
        r = gen.GenerateResponse.FromString(raw)
        ev = r.WhichOneof("event")
        assert r.request_id == ""
        payload = r.token if ev == "token" else r.finished
        assert payload.HasField("output_index") and payload.output_index == 0
        assert r.HasField("usage") == (j == len(sb.msgs) - 1) == (ev == "finished")
    assert gen.GenerateResponse.FromString(sb.msgs[-1]).finished.reason == \
        gen.FINISH_REASON_LENGTH
    assert [t.token_id for raw in sb.msgs[:-1]
            for t in gen.GenerateResponse.FromString(raw).token.tokens] == [0, 7] * 4 + [0]
    # request side: capgrpc.reqbytes templates built with the real module parse back
    from conftest import prompts_doc
    rq = build_openengine(gen, prompts_doc(), tt, "m0")
    body = encode_pb_string_field(1, "42") + base64.b64decode(rq["entries"][0]["b64"])
    assert P.parse_generate_request(body) == (b"42", 9)


# ---- command line ---------------------------------------------------------------------------------

def _cli(tmp_path, *args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([TOOLS, HERE]))
    return subprocess.run([sys.executable, "-m", "capstubs.prebaked"] + list(args), env=env,
                          capture_output=True, text=True, timeout=120, cwd=str(tmp_path))


def test_cli(tmp_path):
    tt = tmp_path / "t.json"
    tt.write_text(json.dumps(table_doc([("warmup", 200, 10.0, {"n": 6, "k": 2}),
                                        ("drain", 200, 0.0, {"k": 2})])))
    out = tmp_path / "pb.json"
    common = ["--tier-table", str(tt), "--token-ids", "3,4", "--prompt-tokens", "2",
              "--out", str(out)]
    r = _cli(tmp_path, "--pb2-module", "no_such_module_capkit_test", *common)
    assert r.returncode == 3 and "cannot import" in r.stderr and not out.exists()
    r = _cli(tmp_path, "--pb2-module", "oe_double", "--short-tokens", "5", *common)
    assert r.returncode == 0, r.stderr
    pb = P.load_prebaked(str(out))
    assert sorted(pb.table) == [(1, 2, 1), (6, 2, 1)]
    doc = json.loads(out.read_text())
    assert doc["source"]["tier_table_sha256"] == TierTable.load(str(tt)).sha256
    for bad in (["--token-ids", "3,x"], ["--token-ids", "4294967296"], ["--prompt-tokens", "-1"],
                ["--short-tokens", "0"], ["--prompt-tok", "2"]):
        r = _cli(tmp_path, "--pb2-module", "oe_double", *(common + bad))
        assert r.returncode == 2, (bad, r.stderr)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prebaked OpenEngine GenerateResponse bytes for the S-PB stub (design 2.3 "S-PB").

A prebaked file (`schema: capkit.prebaked/1`, JSON) holds, for every stream shape
(max_tokens N, tokens_per_msg k, stream_interval s) the stub may serve, the serialized responses
of one whole stream, so the stub never builds or serializes a protobuf object:

    {"schema": "capkit.prebaked/1", "method": "/openengine.v1.Inference/Generate",
     "gate": "product", "token_ids": [int, ...], "prompt_tokens": int,
     "finish_reason": "FINISH_REASON_LENGTH", "source": {...how it was built...},
     "streams": [{"max_tokens": N, "tokens_per_msg": k, "stream_interval": s,
                  "messages": [{"tokens": t, "sha256": hex, "b64": base64}, ...],
                  "finished": {"sha256": hex, "b64": base64}}]}

- messages: one TokenOutput response (output_index 0, `tokens` TokenInfo entries with token_id
  only, no text) per message of capcommon.expect's product gate, carrying
  expect.msg_tokens(N, k, s, "product")[j] tokens; the token ids walk `token_ids` cyclically over
  the whole stream.
- finished: one GenerationFinished response (output_index 0, reason LENGTH) with usage
  (prompt_tokens, completion_tokens = N, total_tokens); the stub sends it in the same beat step
  as the last token message.
- No response carries request_id: the product client checks it only when it is non-empty
  (tensorrt_llm/grpc/openengine_client/stream.py, `_validate`), so the same bytes serve any request.

The wire reader below (`parse_generate_request`, `inspect_response`) is the only protobuf parsing
capkit does on these messages; it reads the field numbers in FIELDS, which `check_field_numbers`
verifies against the product's generated module whenever the file is built.

Command (container-only: imports the product's generated module):
    python -m capstubs.prebaked --pb2-module MODULE --tier-table T --token-ids LIST
        --prompt-tokens P --out PATH [--short-tokens D ...] [--method M]
"""

import argparse
import base64
import hashlib
import json
import re
import sys

from capcommon import expect
from capcommon.exitcodes import CapkitError, ConfigError
from capcommon.jsonio import atomic_write_bytes, dumps

SCHEMA = "capkit.prebaked/1"
GATE = "product"
GENERATE_METHOD = "/openengine.v1.Inference/Generate"
FINISH_REASON = "FINISH_REASON_LENGTH"
SHORT_SITE = "short.req"
_METHOD_RE = re.compile(r"^/[A-Za-z0-9_.]+/[A-Za-z0-9_]+$")
UINT32_MAX = (1 << 32) - 1

# Field numbers the wire reader uses (openengine/v1/generation.proto):
# (message, field) -> number. check_field_numbers() compares them with the generated module.
FIELDS = {
    ("GenerateRequest", "request_id"): 1,
    ("GenerateRequest", "stopping"): 6,
    ("StoppingOptions", "max_tokens"): 1,
    ("GenerateResponse", "request_id"): 1,
    ("GenerateResponse", "prompt"): 2,
    ("GenerateResponse", "token"): 3,
    ("GenerateResponse", "prefill_ready"): 4,
    ("GenerateResponse", "finished"): 5,
    ("GenerateResponse", "error"): 6,
    ("GenerateResponse", "usage"): 10,
    ("TokenOutput", "output_index"): 1,
    ("TokenOutput", "tokens"): 2,
    ("GenerationFinished", "output_index"): 1,
    ("GenerationFinished", "reason"): 2,
    ("Usage", "prompt_tokens"): 1,
    ("Usage", "completion_tokens"): 2,
    ("Usage", "total_tokens"): 3,
}
_RSP_EVENTS = {FIELDS[("GenerateResponse", e)]: e
               for e in ("prompt", "token", "prefill_ready", "finished", "error")}
_F_TOKEN = FIELDS[("GenerateResponse", "token")]
_F_FINISHED = FIELDS[("GenerateResponse", "finished")]
_F_OUTPUT_INDEX = {_F_TOKEN: FIELDS[("TokenOutput", "output_index")],
                   _F_FINISHED: FIELDS[("GenerationFinished", "output_index")]}


class WireFormatError(ValueError):
    """Bytes that are not a well-formed protobuf message (as far as the reader looks)."""


# ---- protobuf wire reader -----------------------------------------------------------------------

def _varint(buf, i, end):
    out = shift = 0
    while True:
        if i >= end:
            raise WireFormatError("truncated varint")
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, i
        shift += 7
        if shift >= 70:
            raise WireFormatError("varint longer than 10 bytes")


def iter_fields(buf, start=0, end=None):
    """Yield (number, wire_type, value) for each field of buf[start:end]. value is the integer for
    varint fields and (begin, stop) offsets into buf for length-delimited and fixed fields.
    Groups (wire types 3, 4) and field number 0 are rejected."""
    i = start
    end = len(buf) if end is None else end
    while i < end:
        key, i = _varint(buf, i, end)
        num, wt = key >> 3, key & 7
        if num == 0:
            raise WireFormatError("field number 0")
        if wt == 0:
            v, i = _varint(buf, i, end)
            yield num, wt, v
        elif wt == 2:
            n, i = _varint(buf, i, end)
            if i + n > end:
                raise WireFormatError("length-delimited field runs past the end")
            yield num, wt, (i, i + n)
            i += n
        elif wt in (1, 5):
            w = 8 if wt == 1 else 4
            if i + w > end:
                raise WireFormatError("fixed field runs past the end")
            yield num, wt, (i, i + w)
            i += w
        else:
            raise WireFormatError("unsupported wire type %d" % wt)


def parse_generate_request(buf):
    """(request_id bytes, stopping.max_tokens or None) of a serialized GenerateRequest.

    Protobuf merge semantics: the last request_id wins; every stopping occurrence merges, so the
    last max_tokens seen wins. Nothing else in the request is looked at."""
    rid, n = b"", None
    f_rid = FIELDS[("GenerateRequest", "request_id")]
    f_stop = FIELDS[("GenerateRequest", "stopping")]
    f_max = FIELDS[("StoppingOptions", "max_tokens")]
    for num, wt, v in iter_fields(buf):
        if num == f_rid:
            if wt != 2:
                raise WireFormatError("request_id is not length-delimited")
            rid = bytes(buf[v[0]:v[1]])
        elif num == f_stop:
            if wt != 2:
                raise WireFormatError("stopping is not length-delimited")
            for num2, wt2, v2 in iter_fields(buf, v[0], v[1]):
                if num2 == f_max:
                    if wt2 != 0:
                        raise WireFormatError("stopping.max_tokens is not a varint")
                    n = v2
    return rid, n


def decimal_request_id(rid):
    """int(rid) when rid is canonical ASCII decimal (no sign, no leading zero, <= 18 digits;
    capgrpc.load writes str(req_id)), else -1."""
    if not rid or len(rid) > 18 or not rid.isdigit() or (rid[:1] == b"0" and len(rid) > 1):
        return -1
    return int(rid)


def inspect_response(buf):
    """What the stub's checks need to know about one serialized GenerateResponse:
    {"request_id": bool, "event": name | None, "events": n, "output_index": bool,
     "tokens": n, "reason": int | None, "usage": {prompt_tokens, completion_tokens,
     total_tokens} | None}."""
    out = {"request_id": False, "event": None, "events": 0, "output_index": False, "tokens": 0,
           "reason": None, "usage": None}
    for num, wt, v in iter_fields(buf):
        if num == FIELDS[("GenerateResponse", "request_id")]:
            if wt != 2:
                raise WireFormatError("request_id is not length-delimited")
            out["request_id"] = out["request_id"] or v[1] > v[0]
        elif num in _RSP_EVENTS:
            if wt != 2:
                raise WireFormatError("event field %d is not length-delimited" % num)
            out["event"] = _RSP_EVENTS[num]
            out["events"] += 1
            if num in (_F_TOKEN, _F_FINISHED):
                for num2, wt2, v2 in iter_fields(buf, v[0], v[1]):
                    if num2 == _F_OUTPUT_INDEX[num] and wt2 == 0:
                        out["output_index"] = True
                    elif num == _F_TOKEN and num2 == FIELDS[("TokenOutput", "tokens")] \
                            and wt2 == 2:
                        out["tokens"] += 1
                    elif num == _F_FINISHED and wt2 == 0 \
                            and num2 == FIELDS[("GenerationFinished", "reason")]:
                        out["reason"] = v2
        elif num == FIELDS[("GenerateResponse", "usage")]:
            if wt != 2:
                raise WireFormatError("usage is not length-delimited")
            u = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            names = {FIELDS[("Usage", k)]: k for k in u}
            for num2, wt2, v2 in iter_fields(buf, v[0], v[1]):
                if num2 in names and wt2 == 0:
                    u[names[num2]] = v2
            out["usage"] = u
    return out


# ---- stream shapes ------------------------------------------------------------------------------

def uniform_msg_shape(tier_table):
    """(k, s) shared by every load tier. A GenerateRequest carries max_tokens but not k or s, so
    the stub can serve a tier table only when its load tiers agree on them (a K-axis sweep is one
    launch per k)."""
    ks = sorted({(t.tokens_per_msg, t.stream_interval) for t in tier_table.tiers if t.is_load})
    if len(ks) != 1:
        raise ConfigError("S-PB needs one (tokens_per_msg, stream_interval) over all load tiers; "
                          "the tier table has %s" % (ks or "no load tier"))
    return ks[0]


def shorted(max_tokens, short_tokens):
    """max_tokens of a request whose short injection omits short_tokens tokens (as nullsse)."""
    return max_tokens - short_tokens if short_tokens < max_tokens else 1


def short_amounts(specs):
    """short_tokens of every short.req spec in specs (any role)."""
    return sorted({sp.short_tokens for sp in specs if sp.site == SHORT_SITE
                   and sp.mode == "short"})


def required_shapes(tier_table, short_tokens=()):
    """Every (N, k, s) a stub serving tier_table may have to send: the load tiers' max_tokens and,
    for each amount in short_tokens, the shortened max_tokens."""
    k, s = uniform_msg_shape(tier_table)
    base = {t.max_tokens for t in tier_table.tiers if t.is_load}
    ns = base | {shorted(n, d) for n in base for d in short_tokens}
    return sorted((n, k, s) for n in ns)


def table_short_amounts(tier_table):
    """short_tokens of every short.req spec in the tier table (launch_inject and tiers)."""
    specs = list(tier_table.launch_inject)
    for t in tier_table.tiers:
        specs.extend(t.inject)
    return short_amounts(specs)


# ---- building (container-only: needs the product's generated module) --------------------------

def check_field_numbers(pb2):
    """Every field number in FIELDS must match the generated module (a stand-in in tests)."""
    bad = []
    for (msg, field), num in sorted(FIELDS.items()):
        cls = getattr(pb2, msg, None)
        fd = None if cls is None else cls.DESCRIPTOR.fields_by_name.get(field)
        if fd is None or fd.number != num:
            bad.append("%s.%s: expected %d, module has %s"
                       % (msg, field, num, None if fd is None else fd.number))
    if bad:
        raise ConfigError("generated module disagrees with the wire reader: %s" % "; ".join(bad))


def _entry(raw, tokens=None):
    e = {"sha256": hashlib.sha256(raw).hexdigest(), "b64": base64.b64encode(raw).decode("ascii")}
    if tokens is not None:
        e["tokens"] = tokens
    return e


def build_prebaked(pb2, tier_table, token_ids, prompt_tokens, short_tokens=(),
                   method=GENERATE_METHOD, source=None):
    """The prebaked document for every shape of required_shapes(tier_table, short amounts of the
    table plus short_tokens). pb2: the product's generation_pb2 (or a stand-in)."""
    check_field_numbers(pb2)
    if not token_ids or any(not isinstance(t, int) or isinstance(t, bool) or not
                            0 <= t <= UINT32_MAX for t in token_ids):
        raise ConfigError("token_ids must be a non-empty list of uint32")
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or \
            not 0 <= prompt_tokens <= UINT32_MAX:
        raise ConfigError("prompt_tokens must be a uint32")
    shorts = sorted(set(short_tokens) | set(table_short_amounts(tier_table)))
    reason = getattr(pb2, FINISH_REASON)
    streams = []
    for n, k, s in required_shapes(tier_table, shorts):
        if prompt_tokens + n > UINT32_MAX:
            raise ConfigError("prompt_tokens + max_tokens overflows uint32")
        msgs, pos = [], 0
        for t in expect.msg_tokens(n, k, s, GATE):
            ids = [token_ids[(pos + i) % len(token_ids)] for i in range(t)]
            pos += t
            rsp = pb2.GenerateResponse(token=pb2.TokenOutput(
                output_index=0, tokens=[pb2.TokenInfo(token_id=x) for x in ids]))
            msgs.append(_entry(rsp.SerializeToString(deterministic=True), t))
        fin = pb2.GenerateResponse(
            finished=pb2.GenerationFinished(output_index=0, reason=reason),
            usage=pb2.Usage(prompt_tokens=prompt_tokens, completion_tokens=n,
                            total_tokens=prompt_tokens + n))
        streams.append({"max_tokens": n, "tokens_per_msg": k, "stream_interval": s,
                        "messages": msgs,
                        "finished": _entry(fin.SerializeToString(deterministic=True))})
    doc = {"schema": SCHEMA, "method": method, "gate": GATE, "token_ids": list(token_ids),
           "prompt_tokens": prompt_tokens, "finish_reason": FINISH_REASON,
           "finish_reason_value": int(reason), "short_tokens": shorts,
           "source": source or {}, "streams": streams}
    Prebaked(doc)  # the same checks the stub runs at start-up
    return doc


# ---- loading (the stub; no protobuf import) -----------------------------------------------------

class StreamBytes:
    """One prebaked stream: msgs = token messages + the finished message; offs[j] = beat step
    offset of msgs[j] from the stream's first step (the finished message shares the last token
    message's step); cum[j] = tokens carried by msgs[:j]."""
    __slots__ = ("max_tokens", "msgs", "offs", "cum", "n_token_msgs")

    def __init__(self, max_tokens, msgs, offs, cum):
        self.max_tokens = max_tokens
        self.msgs, self.offs, self.cum = msgs, offs, cum
        self.n_token_msgs = len(msgs) - 1


def _uint(v):
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= UINT32_MAX


def _decode(e, where):
    if not isinstance(e, dict) or not isinstance(e.get("b64"), str):
        raise ConfigError("%s: needs b64" % where)
    try:
        raw = base64.b64decode(e["b64"], validate=True)
    except ValueError as err:
        raise ConfigError("%s: bad base64: %s" % (where, err))
    if hashlib.sha256(raw).hexdigest() != e.get("sha256"):
        raise ConfigError("%s: sha256 mismatch" % where)
    try:
        info = inspect_response(raw)
    except WireFormatError as err:
        raise ConfigError("%s: not a GenerateResponse: %s" % (where, err))
    if info["request_id"]:
        raise ConfigError("%s: carries a request_id" % where)
    if info["events"] != 1 or not info["output_index"]:
        raise ConfigError("%s: needs exactly one event with output_index set" % where)
    return raw, info


class Prebaked:
    """A validated prebaked document. lookup(N, k, s) -> StreamBytes or None."""

    def __init__(self, doc, path=None, sha256=None):
        w = path or "prebaked"
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise ConfigError("%s: schema must be %s" % (w, SCHEMA))
        if not isinstance(doc.get("method"), str) or not _METHOD_RE.match(doc["method"]):
            raise ConfigError("%s: method must look like /pkg.Service/Method" % w)
        if doc.get("gate") != GATE:
            raise ConfigError("%s: gate must be %s" % (w, GATE))
        ids = doc.get("token_ids")
        if not isinstance(ids, list) or not ids or not all(_uint(t) for t in ids):
            raise ConfigError("%s: token_ids must be a non-empty list of uint32" % w)
        p = doc.get("prompt_tokens")
        reason = doc.get("finish_reason_value")
        if not _uint(p) or not _uint(reason) or reason == 0:
            raise ConfigError("%s: prompt_tokens and finish_reason_value must be uint32 "
                              "(reason != 0)" % w)
        streams = doc.get("streams")
        if not isinstance(streams, list) or not streams:
            raise ConfigError("%s: streams must be a non-empty list" % w)
        self.method, self.prompt_tokens, self.path, self.sha256 = doc["method"], p, path, sha256
        self.table = {}
        for i, st in enumerate(streams):
            where = "%s: streams[%d]" % (w, i)
            try:
                n, k, s = st["max_tokens"], st["tokens_per_msg"], st["stream_interval"]
                toks = expect.msg_tokens(n, k, s, GATE)
                offs = expect.msg_step_offsets(n, k, s, GATE)
            except (KeyError, TypeError, expect.ExpectError) as err:
                raise ConfigError("%s: bad shape: %s" % (where, err))
            if (n, k, s) in self.table:
                raise ConfigError("%s: duplicate shape %s" % (where, (n, k, s)))
            msgs = st.get("messages")
            if not isinstance(msgs, list) or len(msgs) != len(toks):
                raise ConfigError("%s: needs %d token messages (product gate)" % (where, len(toks)))
            out, cum = [], [0]
            for j, (e, t) in enumerate(zip(msgs, toks)):
                raw, info = _decode(e, "%s.messages[%d]" % (where, j))
                if info["event"] != "token" or e.get("tokens") != t or info["tokens"] != t \
                        or info["usage"] is not None:
                    raise ConfigError("%s.messages[%d]: expected a token event with %d tokens "
                                      "and no usage" % (where, j, t))
                out.append(raw)
                cum.append(cum[-1] + t)
            raw, info = _decode(st.get("finished"), "%s.finished" % where)
            u = info["usage"]
            if info["event"] != "finished" or info["reason"] != reason or u is None or \
                    u != {"prompt_tokens": p, "completion_tokens": n, "total_tokens": p + n}:
                raise ConfigError("%s.finished: expected finished (reason %d) with usage "
                                  "completion_tokens %d, prompt_tokens %d" % (where, reason, n, p))
            out.append(raw)
            cum.append(cum[-1])
            self.table[(n, k, s)] = StreamBytes(n, tuple(out), tuple(offs) + (offs[-1],),
                                                tuple(cum))

    def lookup(self, max_tokens, tokens_per_msg, stream_interval):
        return self.table.get((max_tokens, tokens_per_msg, stream_interval))


def load_prebaked(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise ConfigError("cannot read prebaked file %s: %s" % (path, e))
    try:
        doc = json.loads(data.decode("utf-8"))
    except ValueError as e:
        raise ConfigError("%s: not JSON: %s" % (path, e))
    return Prebaked(doc, path, hashlib.sha256(data).hexdigest())


# ---- command line --------------------------------------------------------------------------------

def _uint_list(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part or not part.isascii() or not part.isdigit() or int(part) > UINT32_MAX:
            raise argparse.ArgumentTypeError("not a uint32 list: %r" % text)
        out.append(int(part))
    return out


def _uint_arg(text):
    if not text.isascii() or not text.isdigit() or int(text) > UINT32_MAX:
        raise argparse.ArgumentTypeError("not a uint32: %r" % text)
    return int(text)


def _pos_int(text):
    if not text.isascii() or not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError("not a positive integer: %r" % text)
    return int(text)


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capstubs.prebaked", allow_abbrev=False,
                                description=__doc__.split("\n")[0])
    p.add_argument("--pb2-module", required=True,
                   help="the product's generated module, e.g. "
                        "tensorrt_llm.grpc.openengine._generated.generation_pb2")
    p.add_argument("--tier-table", required=True)
    p.add_argument("--token-ids", required=True, type=_uint_list,
                   help="comma-separated token ids, walked cyclically over every stream")
    p.add_argument("--prompt-tokens", required=True, type=_uint_arg,
                   help="usage.prompt_tokens on the finished message")
    p.add_argument("--short-tokens", action="append", type=_pos_int, default=[],
                   help="also build the shapes a command-line short.req injection of this many "
                        "tokens needs (the tier table's own short specs are always included)")
    p.add_argument("--method", default=GENERATE_METHOD)
    p.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        from capcommon.tiers import TierTable  # noqa: PLC0415
        from capgrpc.reqbytes import import_pb2  # noqa: PLC0415
        tt = TierTable.load(args.tier_table)
        pb2 = import_pb2(args.pb2_module)
        src = {"pb2_module": args.pb2_module, "pb2_file": getattr(pb2, "__file__", None),
               "tier_table": args.tier_table, "tier_table_sha256": tt.sha256}
        doc = build_prebaked(pb2, tt, args.token_ids, args.prompt_tokens, args.short_tokens,
                             args.method, src)
        atomic_write_bytes(args.out, (dumps(doc) + "\n").encode("utf-8"))
        print("%s: %d streams, method %s" % (args.out, len(doc["streams"]), doc["method"]))
        return 0
    except CapkitError as e:
        print("prebaked: %s" % e, file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())

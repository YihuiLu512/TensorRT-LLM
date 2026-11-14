# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-bytes files for capgrpc --request-format {openengine,raw}.

A request-bytes file (`schema: capkit.reqbytes/1`, JSON) holds pre-serialized request bodies so
the load generator never builds protobuf objects:

    {"schema": "capkit.reqbytes/1", "format": "openengine" | "raw",
     "method": "/openengine.v1.Inference/Generate" | null,      (informational)
     "id_field": {"kind": "pb_string", "number": 1} | null,
     "source": {...how it was built...},
     "entries": [{"prompt_set": str|null, "prompt_index": int|null, "max_tokens": int|null,
                  "sha256": "<hex of the bytes>", "b64": "<base64 bytes>"}]}

- raw: exactly one entry, all keys null; sent unchanged for every request (no request id).
- openengine: one entry per (prompt_set, prompt_index, max_tokens) of the tier table's load tiers,
  each a GenerateRequest WITHOUT request_id. The loader prepends the request_id field
  (`encode_pb_string_field(number, str(req_id))`); protobuf parsing merges concatenated
  serializations, so the result is the template with request_id set.

Commands:
    python -m capgrpc.reqbytes raw --in FILE --out PATH [--method M]
    python -m capgrpc.reqbytes openengine --pb2-module MODULE --prompts P --tier-table T
        --model NAME --out PATH [--method M]
The openengine builder imports the product's generated module only when asked (container-only);
when it cannot be imported the command exits 3.
"""

import argparse
import base64
import hashlib
import importlib
import json
import sys

from capcommon.exitcodes import CapkitError, ConfigError
from capcommon.jsonio import atomic_write_bytes, dumps

SCHEMA = "capkit.reqbytes/1"
FORMATS = ("openengine", "raw")
PROMPTS_SCHEMA = "capkit.prompts/1"


def encode_varint(n):
    if n < 0:
        raise ValueError("varint must be >= 0")
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def encode_pb_string_field(number, text):
    """Wire bytes of one length-delimited protobuf field (string) `number` = text."""
    data = text.encode("utf-8")
    return encode_varint((number << 3) | 2) + encode_varint(len(data)) + data


class ReqBytes:
    """A loaded request-bytes file. lookup(prompt_set, prompt_index, max_tokens) -> bytes."""

    def __init__(self, doc, path=None):
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise ConfigError("%s: schema must be %s" % (path, SCHEMA))
        fmt = doc.get("format")
        if fmt not in FORMATS:
            raise ConfigError("%s: format must be one of %s" % (path, FORMATS))
        self.format = fmt
        self.method = doc.get("method")
        idf = doc.get("id_field")
        if fmt == "openengine":
            if not (isinstance(idf, dict) and idf.get("kind") == "pb_string"
                    and isinstance(idf.get("number"), int) and idf["number"] > 0):
                raise ConfigError("%s: openengine needs id_field {kind: pb_string, number}" % path)
            self.id_field_number = idf["number"]
        else:
            if idf is not None:
                raise ConfigError("%s: raw request bytes carry no id_field" % path)
            self.id_field_number = None
        entries = doc.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ConfigError("%s: entries must be a non-empty list" % path)
        self.table = {}
        self.wildcard = None
        for e in entries:
            raw = base64.b64decode(e["b64"], validate=True)
            if hashlib.sha256(raw).hexdigest() != e.get("sha256"):
                raise ConfigError("%s: entry sha256 mismatch" % path)
            key = (e.get("prompt_set"), e.get("prompt_index"), e.get("max_tokens"))
            if key == (None, None, None):
                if fmt != "raw" or self.wildcard is not None or len(entries) != 1:
                    raise ConfigError("%s: a wildcard entry is only valid as the single raw "
                                      "entry" % path)
                self.wildcard = raw
                continue
            if fmt == "raw" or None in key:
                raise ConfigError("%s: openengine entries need prompt_set, prompt_index, "
                                  "max_tokens" % path)
            if key in self.table:
                raise ConfigError("%s: duplicate entry %s" % (path, key))
            self.table[key] = raw
        if fmt == "raw" and self.wildcard is None:
            raise ConfigError("%s: raw needs exactly one wildcard entry" % path)

    def lookup(self, prompt_set, prompt_index, max_tokens):
        if self.wildcard is not None:
            return self.wildcard
        return self.table.get((prompt_set, prompt_index, max_tokens))


def load_reqbytes(path):
    with open(path, "rb") as f:
        try:
            doc = json.loads(f.read().decode("utf-8"))
        except ValueError as e:
            raise ConfigError("%s: not JSON: %s" % (path, e))
    return ReqBytes(doc, path)


def _entry(prompt_set, prompt_index, max_tokens, raw):
    return {"prompt_set": prompt_set, "prompt_index": prompt_index, "max_tokens": max_tokens,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "b64": base64.b64encode(raw).decode("ascii")}


def build_raw(data, method=None, source=None):
    return {"schema": SCHEMA, "format": "raw", "method": method, "id_field": None,
            "source": source or {}, "entries": [_entry(None, None, None, data)]}


def load_prompts(path):
    with open(path, "rb") as f:
        doc = json.loads(f.read().decode("utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != PROMPTS_SCHEMA:
        raise ConfigError("%s: schema must be %s" % (path, PROMPTS_SCHEMA))
    sets = doc.get("sets")
    if not isinstance(sets, dict) or not sets:
        raise ConfigError("%s: sets must be a non-empty object" % path)
    for name, s in sets.items():
        if not isinstance(s, dict) or s.get("kind") not in ("token_ids", "text") \
                or not isinstance(s.get("items"), list) or not s["items"]:
            raise ConfigError("%s: set %s needs kind token_ids|text and non-empty items"
                              % (path, name))
    return doc


def build_openengine(pb2, prompts_doc, tier_table, model, method=None, source=None):
    """Templates for every (prompt_set, item, max_tokens) used by the tier table's load tiers.

    pb2: the product's generated module exposing GenerateRequest (passed in, so a test can hand
    over a stand-in module built from a descriptor)."""
    req_cls = pb2.GenerateRequest
    fd = req_cls.DESCRIPTOR.fields_by_name.get("request_id")
    if fd is None or fd.type != fd.TYPE_STRING:
        raise ConfigError("GenerateRequest.request_id is not a string field")
    sets = prompts_doc["sets"]
    keys = sorted({(t.prompt_set, t.max_tokens) for t in tier_table.tiers if t.is_load})
    entries = []
    for set_name, max_tokens in keys:
        if set_name not in sets:
            raise ConfigError("prompt set %r (tier table) missing from the prompts file"
                              % set_name)
        kind, items = sets[set_name]["kind"], sets[set_name]["items"]
        for idx, item in enumerate(items):
            msg = req_cls(model=model)
            if kind == "text":
                msg.prompt = item
            else:
                msg.token_ids.ids.extend(item)
            msg.stopping.max_tokens = max_tokens
            msg.stopping.ignore_eos = True
            entries.append(_entry(set_name, idx, max_tokens,
                                  msg.SerializeToString(deterministic=True)))
    return {"schema": SCHEMA, "format": "openengine", "method": method,
            "id_field": {"kind": "pb_string", "number": fd.number},
            "source": source or {}, "entries": entries}


def import_pb2(module_name):
    """Import the product's generated module; any exception while importing it (not only a
    missing product package) is exit 3 (CONFIG). A failed import leaves no module behind."""
    try:
        return importlib.import_module(module_name)
    except (Exception, SystemExit) as e:  # noqa: BLE001 - whatever it raises, exit 3
        sys.modules.pop(module_name, None)
        raise ConfigError("cannot import %s (the openengine builder runs only where the "
                          "product package is installed): %s: %s"
                          % (module_name, type(e).__name__, e))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m capgrpc.reqbytes")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("raw", help="one fixed body for every request")
    r.add_argument("--in", dest="inp", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--method")
    o = sub.add_parser("openengine", help="GenerateRequest templates (container-only)")
    o.add_argument("--pb2-module", required=True)
    o.add_argument("--prompts", required=True)
    o.add_argument("--tier-table", required=True)
    o.add_argument("--model", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--method")
    args = p.parse_args(argv)
    try:
        if args.cmd == "raw":
            with open(args.inp, "rb") as f:
                data = f.read()
            doc = build_raw(data, args.method, {"in": args.inp})
        else:
            from capcommon.hashing import sha256_file  # noqa: PLC0415
            from capcommon.tiers import TierTable  # noqa: PLC0415
            tt = TierTable.load(args.tier_table)
            prompts = load_prompts(args.prompts)
            pb2 = import_pb2(args.pb2_module)
            src = {"pb2_module": args.pb2_module, "pb2_file": getattr(pb2, "__file__", None),
                   "prompts": args.prompts, "prompts_sha256": sha256_file(args.prompts),
                   "tier_table": args.tier_table, "tier_table_sha256": tt.sha256,
                   "model": args.model}
            doc = build_openengine(pb2, prompts, tt, args.model, args.method, src)
        ReqBytes(doc, args.out)  # validate what we are about to write
        atomic_write_bytes(args.out, (dumps(doc) + "\n").encode("utf-8"))
        print("%s: %d entries (%s)" % (args.out, len(doc["entries"]), doc["format"]))
        return 0
    except CapkitError as e:
        print("reqbytes: %s" % e, file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())

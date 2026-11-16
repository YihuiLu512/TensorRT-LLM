# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic writes, identity-stamped JSON, and capkit JSONL (INTERFACES.md sections 0, 5.2, 5.6)."""

import json
import os

from .clock import now_ns, wall_iso
from .exitcodes import ConfigError
from .identity import Identity

RESERVED_KEYS = ("schema", "identity", "written_mono_ns", "written_wall")


def atomic_write_bytes(path, data):
    """Write data to path via a temp file in the same directory and os.replace."""
    d = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(d, ".%s.tmp.%d" % (os.path.basename(path), os.getpid()))
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
    os.replace(tmp, path)


def _identity_dict(identity, allow_null_role=False):
    if identity is None:
        return None
    if isinstance(identity, Identity):
        return identity.to_dict()
    return Identity.from_dict(identity, allow_null_role=allow_null_role).to_dict()


def dumps(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=1)


def write_json_stamped(path, schema, identity, payload):
    """Write {"schema", "identity", "written_mono_ns", "written_wall", **payload} atomically.

    identity must be an Identity (or a dict that validates as one); payload must not use the
    reserved keys.
    """
    if not isinstance(schema, str) or not schema.startswith("capkit."):
        raise ConfigError("schema must be a capkit.* name, got %r" % (schema,))
    if identity is None:
        raise ConfigError("identity is required in every capkit output file")
    clash = [k for k in RESERVED_KEYS if k in payload]
    if clash:
        raise ConfigError("payload uses reserved keys %s" % clash)
    doc = {"schema": schema, "identity": _identity_dict(identity),
           "written_mono_ns": now_ns(), "written_wall": wall_iso()}
    doc.update(payload)
    atomic_write_bytes(path, (dumps(doc) + "\n").encode("utf-8"))
    return doc


def read_json(path):
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def read_json_stamped(path, schema, allow_null_role=False):
    """Read a stamped JSON file; raise ConfigError if the schema differs or identity is invalid.

    Returns (Identity, doc).
    """
    doc = read_json(path)
    if not isinstance(doc, dict) or doc.get("schema") != schema:
        raise ConfigError("%s: schema %r != expected %r" % (
            path, doc.get("schema") if isinstance(doc, dict) else None, schema))
    ident = Identity.from_dict(doc.get("identity"), allow_null_role=allow_null_role)
    return ident, doc


class JsonlWriter:
    """Streaming capkit JSONL: header line, then one JSON array per row (column order fixed).

    Only for independent sampler processes (sidecar, canary), never on a measured hot path.
    """

    def __init__(self, path, schema, identity, columns, flush_every=1):
        if not columns or len(set(columns)) != len(columns):
            raise ConfigError("columns must be non-empty and unique")
        self.path = path
        self.columns = list(columns)
        self._n = len(columns)
        self._flush_every = max(1, int(flush_every))
        self._pending = 0
        self.rows = 0
        self._f = open(path, "w", encoding="utf-8")
        header = {"_header": True, "schema": schema, "identity": _identity_dict(identity),
                  "columns": self.columns, "written_wall": wall_iso()}
        self._f.write(json.dumps(header, sort_keys=True, allow_nan=False) + "\n")
        self._f.flush()

    def write_row(self, row):
        if len(row) != self._n:
            raise ValueError("row has %d values, expected %d" % (len(row), self._n))
        self._f.write(json.dumps(list(row), allow_nan=False, separators=(",", ":")) + "\n")
        self.rows += 1
        self._pending += 1
        if self._pending >= self._flush_every:
            self._f.flush()
            self._pending = 0

    def close(self):
        if self._f is not None:
            self._f.flush()
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_jsonl(path, schema=None):
    """Return (header, rows). Truncated last line (writer killed) is reported, not hidden:
    header["_truncated_tail"] is set to True and the partial line is dropped.
    """
    rows = []
    header = None
    truncated = False
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    else:
        truncated = bool(lines)
    for i, line in enumerate(lines):
        is_last = i == len(lines) - 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if is_last and truncated:
                break
            raise ConfigError("%s:%d: invalid JSON line" % (path, i + 1))
        if i == 0:
            if not isinstance(obj, dict) or obj.get("_header") is not True:
                raise ConfigError("%s: first line is not a capkit JSONL header" % path)
            header = obj
            continue
        if not isinstance(obj, list) or len(obj) != len(header["columns"]):
            raise ConfigError("%s:%d: row width mismatch" % (path, i + 1))
        rows.append(obj)
    if header is None:
        raise ConfigError("%s: empty JSONL" % path)
    if schema is not None and header.get("schema") != schema:
        raise ConfigError("%s: schema %r != %r" % (path, header.get("schema"), schema))
    header["_truncated_tail"] = truncated
    return header, rows

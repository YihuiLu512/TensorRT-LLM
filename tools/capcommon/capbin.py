# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capbin: columnar binary container for row records and schedules (INTERFACES.md section 4.1).

Layout (little endian): b"CAPBIN01" | u64 header_len | JSON header padded with spaces to a
multiple of 8 | columns, each the raw array bytes padded with zeros to a multiple of 8.
"""

import array
import json
import mmap
import os
import struct
import sys

from .exitcodes import ConfigError
from .identity import Identity
from .jsonio import atomic_write_bytes

MAGIC = b"CAPBIN01"
SCHEMA = "capkit.capbin/1"
TYPECODES = "bBhHiIqQfd"
_U64 = struct.Struct("<Q")


class CapbinError(ConfigError):
    pass


def _pad8(n):
    return (-n) % 8


def _le_bytes(arr):
    if sys.byteorder == "little":
        return arr.tobytes()
    a = array.array(arr.typecode, arr)
    a.byteswap()
    return a.tobytes()


def _identity_dict(identity):
    if identity is None:
        return None
    if isinstance(identity, Identity):
        return identity.to_dict()
    return Identity.from_dict(identity, allow_null_role=True).to_dict()


def encode_capbin(kind, identity, columns, meta=None):
    """columns: list of (name, array.array). Returns the file bytes."""
    if not columns:
        raise CapbinError("capbin needs at least one column")
    names = [c[0] for c in columns]
    if len(set(names)) != len(names):
        raise CapbinError("duplicate column names")
    n_rows = len(columns[0][1])
    col_meta = []
    off = 0
    blobs = []
    for name, arr in columns:
        if not isinstance(arr, array.array) or arr.typecode not in TYPECODES:
            raise CapbinError("column %s must be an array.array with typecode in %s"
                              % (name, TYPECODES))
        if len(arr) != n_rows:
            raise CapbinError("column %s has %d rows, expected %d" % (name, len(arr), n_rows))
        b = _le_bytes(arr)
        col_meta.append({"name": name, "typecode": arr.typecode, "itemsize": arr.itemsize,
                         "offset": off, "nbytes": len(b)})
        blobs.append(b + b"\0" * _pad8(len(b)))
        off += len(b) + _pad8(len(b))
    header = {"schema": SCHEMA, "kind": kind, "identity": _identity_dict(identity),
              "n_rows": n_rows, "columns": col_meta, "meta": meta or {}, "byteorder": "little"}
    hb = json.dumps(header, sort_keys=True, allow_nan=False).encode("utf-8")
    hb += b" " * _pad8(len(hb))
    return b"".join([MAGIC, _U64.pack(len(hb)), hb] + blobs)


def write_capbin(path, kind, identity, columns, meta=None):
    atomic_write_bytes(path, encode_capbin(kind, identity, columns, meta))


def _parse_header(buf, path):
    if len(buf) < 16 or bytes(buf[:8]) != MAGIC:
        raise CapbinError("%s: not a capbin file" % path)
    (hlen,) = _U64.unpack(bytes(buf[8:16]))
    if 16 + hlen > len(buf):
        raise CapbinError("%s: truncated header" % path)
    try:
        header = json.loads(bytes(buf[16:16 + hlen]).decode("utf-8"))
    except ValueError as e:
        raise CapbinError("%s: bad header JSON: %s" % (path, e))
    if header.get("schema") != SCHEMA or header.get("byteorder") != "little":
        raise CapbinError("%s: unsupported capbin header" % path)
    data_start = 16 + hlen
    for c in header["columns"]:
        tc = c["typecode"]
        if tc not in TYPECODES or array.array(tc).itemsize != c["itemsize"]:
            raise CapbinError("%s: column %s typecode %s itemsize %s does not match this platform"
                              % (path, c["name"], tc, c["itemsize"]))
        if c["nbytes"] != header["n_rows"] * c["itemsize"]:
            raise CapbinError("%s: column %s size mismatch" % (path, c["name"]))
        if data_start + c["offset"] + c["nbytes"] > len(buf):
            raise CapbinError("%s: column %s truncated" % (path, c["name"]))
    return header, data_start


def read_capbin(path, columns=None, stride=None):
    """Return (header, {name: array}).

    columns: subset of column names (None = all). stride: (k, W) keeps rows k, k+W, k+2W, ...
    Uses mmap so only the requested columns are touched.
    """
    if stride is not None:
        k, w = stride
        if not (isinstance(w, int) and w >= 1 and isinstance(k, int) and 0 <= k < w):
            raise CapbinError("stride must be (k, W) with 0 <= k < W")
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if size == 0:
            raise CapbinError("%s: empty file" % path)
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            mv = memoryview(mm)
            try:
                header, data_start = _parse_header(mv, path)
                want = [c["name"] for c in header["columns"]] if columns is None else list(columns)
                by_name = {c["name"]: c for c in header["columns"]}
                missing = [n for n in want if n not in by_name]
                if missing:
                    raise CapbinError("%s: no column(s) %s" % (path, missing))
                out = {}
                for n in want:
                    c = by_name[n]
                    lo = data_start + c["offset"]
                    seg = mv[lo:lo + c["nbytes"]]
                    if stride is None:
                        arr = array.array(c["typecode"], bytes(seg))
                    else:
                        view = seg.cast(c["typecode"]) if c["nbytes"] else None
                        arr = (array.array(c["typecode"], view[stride[0]::stride[1]].tobytes())
                               if view is not None else array.array(c["typecode"]))
                        if view is not None:
                            view.release()
                    if sys.byteorder != "little":
                        arr.byteswap()
                    out[n] = arr
                    seg.release()
            finally:
                mv.release()
        finally:
            mm.close()
    return header, out


class RecordArray:
    """Preallocated struct-of-arrays row buffer. Rows beyond capacity are dropped and counted.

    fields: list of (name, typecode). append(*values) is the hot-path call (no allocation).
    """

    __slots__ = ("fields", "names", "capacity", "n", "overflow", "cols", "_cols_t", "_nf")

    def __init__(self, fields, capacity):
        if not isinstance(capacity, int) or capacity < 0:
            raise CapbinError("capacity must be an int >= 0")
        names = [f[0] for f in fields]
        if not names or len(set(names)) != len(names):
            raise CapbinError("fields must be non-empty and unique")
        for name, tc in fields:
            if tc not in TYPECODES:
                raise CapbinError("field %s typecode %r not allowed" % (name, tc))
        self.fields = list(fields)
        self.names = names
        self.capacity = capacity
        self.n = 0
        self.overflow = 0
        self.cols = {}
        for name, tc in fields:
            a = array.array(tc)
            if capacity:
                a.frombytes(bytes(capacity * a.itemsize))
            self.cols[name] = a
        self._cols_t = tuple(self.cols[n] for n in names)
        self._nf = len(names)

    def append(self, *values):
        if len(values) != self._nf:
            raise CapbinError("append got %d values for %d fields" % (len(values), self._nf))
        i = self.n
        if i >= self.capacity:
            self.overflow += 1
            return False
        for col, v in zip(self._cols_t, values):
            col[i] = v
        self.n = i + 1
        return True

    def __len__(self):
        return self.n

    def column(self, name):
        """A trimmed copy of one column (n rows)."""
        return self.cols[name][:self.n]

    def _trimmed_columns(self, runs=None):
        """runs: None (all n rows) or a list of (start, stop) row ranges to concatenate."""
        out = []
        for name in self.names:
            col = self.cols[name]
            if runs is None:
                out.append((name, col[:self.n]))
            else:
                a = array.array(col.typecode)
                for lo, hi in runs:
                    a.extend(col[lo:hi])
                out.append((name, a))
        return out

    def dump(self, path, kind, identity, meta=None):
        m = dict(meta or {})
        m["overflow"] = self.overflow
        m["capacity"] = self.capacity
        write_capbin(path, kind, identity, self._trimmed_columns(), m)

    def split_runs(self, key):
        """{key value: [(start, stop), ...]} contiguous row ranges per value, in row order."""
        col = self.cols[key]
        groups = {}
        n = self.n
        i = 0
        while i < n:
            v = col[i]
            j = i + 1
            while j < n and col[j] == v:
                j += 1
            groups.setdefault(v, []).append((i, j))
            i = j
        return groups

    def dump_split(self, path_for, kind, identity_for, key="tier_seq", meta=None):
        """Write one capbin per distinct value of column `key`.

        path_for(value) -> path (its directory must exist or will be created);
        identity_for(value) -> Identity for that file. overflow/capacity go into every file's
        meta (overflow is per buffer, so readers must treat any overflow > 0 as a failure).
        Returns {value: path}.
        """
        written = {}
        for value, runs in sorted(self.split_runs(key).items()):
            path = path_for(value)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            m = dict(meta or {})
            m["overflow"] = self.overflow
            m["capacity"] = self.capacity
            m["split_key"] = key
            m["split_value"] = value
            write_capbin(path, kind, identity_for(value), self._trimmed_columns(runs), m)
            written[value] = path
        return written

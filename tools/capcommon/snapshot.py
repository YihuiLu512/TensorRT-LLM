# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live counter snapshots shared by capprobe and every capkit tool (INTERFACES.md section 6).

A process registers slots in a SlotRegistry (one int64 array; each slot written by one thread).
A SnapshotWriter thread copies the array to <shm_dir>/<pid>.bin every interval (atomic replace).
"""

import array
import json
import os
import re
import stat
import struct
import sys
import threading
import time

from . import hist as _hist
from .exitcodes import ConfigError

MAGIC = b"CAPSNAP1"
VERSION = 1
SCHEMA = "capkit.snapshot/1"
_HDR = struct.Struct("<8sIIQqII")  # magic, version, meta_len, seq, t_mono_ns, n_values, meta_gen
HEADER_SIZE = _HDR.size  # 40
SLOT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[A-Za-z0-9_]+)+$")
KINDS = ("counter", "gauge", "hist")
_FILE_RE = re.compile(r"^(\d+)\.bin$")


class SnapshotError(ConfigError):
    pass


def proc_starttime_ticks(pid=None):
    """Field 22 of /proc/<pid>/stat (process start time in clock ticks), or None."""
    path = "/proc/%s/stat" % ("self" if pid is None else pid)
    try:
        with open(path, "rb") as f:
            raw = f.read().decode(errors="replace")
    except OSError:
        return None
    rest = raw[raw.rfind(")") + 2:].split()
    return int(rest[19])


class HistSlot:
    """A loghist/1 histogram living inside a SlotRegistry (plus .n/.sum/.max/.neg counters)."""

    __slots__ = ("_v", "base", "i_n", "i_sum", "i_max", "i_neg")

    def __init__(self, values, base, i_n, i_sum, i_max, i_neg):
        self._v = values
        self.base, self.i_n, self.i_sum, self.i_max, self.i_neg = base, i_n, i_sum, i_max, i_neg

    def add(self, v):
        vals = self._v
        if v < 0:
            vals[self.i_neg] += 1
            vals[self.base] += 1
        else:
            vals[self.base + (0 if v < 1024 else _hist.bucket_index(v))] += 1
            vals[self.i_sum] += v
        vals[self.i_n] += 1
        if v > vals[self.i_max]:
            vals[self.i_max] = v

    def counts(self):
        return list(self._v[self.base:self.base + _hist.NBINS])


class SlotRegistry:
    """Named int64 slots. Registration is idempotent per (name, kind)."""

    def __init__(self):
        self.values = array.array("q")
        self.slots = []  # dicts: name, kind, offset, length[, bins]
        self._by_name = {}
        self.meta_gen = 0
        self._lock = threading.Lock()  # registration only; never taken on the hot path

    def _alloc(self, name, kind, length):
        if not isinstance(name, str) or not SLOT_NAME_RE.match(name):
            raise SnapshotError("slot name invalid: %r" % (name,))
        with self._lock:
            got = self._by_name.get(name)
            if got is not None:
                if got["kind"] != kind:
                    raise SnapshotError("slot %s already registered as %s" % (name, got["kind"]))
                return got["offset"]
            off = len(self.values)
            self.values.frombytes(bytes(8 * length))
            d = {"name": name, "kind": kind, "offset": off, "length": length}
            if kind == "hist":
                d["bins"] = _hist.NAME
            self.slots.append(d)
            self._by_name[name] = d
            self.meta_gen += 1
            return off

    def counter(self, name):
        return self._alloc(name, "counter", 1)

    def gauge(self, name):
        return self._alloc(name, "gauge", 1)

    def hist(self, name):
        base = self._alloc(name, "hist", _hist.NBINS)
        return HistSlot(self.values, base, self.counter(name + ".n"),
                        self.counter(name + ".sum"), self.counter(name + ".max"),
                        self.counter(name + ".neg"))

    def get(self, name):
        d = self._by_name[name]
        if d["length"] == 1:
            return self.values[d["offset"]]
        return list(self.values[d["offset"]:d["offset"] + d["length"]])

    def names(self):
        return [s["name"] for s in self.slots]

    def register_decls(self, decls, groups=None):
        """Register every declared slot now (eagerly, at import / start-up), so a slot that is
        never passed still appears in snapshots with value 0 and an unwired site is visible as
        "slot present, count 0" rather than indistinguishable from "slot never declared".

        decls: a package's SLOTS mapping (validate_slot_decls). groups: register only slots
        whose group is in this collection (None = all). Returns {name: offset or HistSlot}.
        """
        validate_slot_decls(decls)
        out = {}
        for name in sorted(decls):
            d = decls[name]
            if groups is not None and d["group"] not in groups:
                continue
            kind = d["kind"]
            out[name] = (self.hist(name) if kind == "hist" else
                         self.counter(name) if kind == "counter" else self.gauge(name))
        return out


SLOT_DECL_KEYS = ("kind", "site", "group", "per")
SLOT_PER = ("request", "message", "step", "event", "sample", "none")


def validate_slot_decls(decls):
    """A package's machine-readable SLOTS: {slot name: {kind, site, group, per}}.

    site: the injection/probe site the slot belongs to, or null; group: the probe/tool group
    (count, bytes, lag, queue, ts, pin, inject, backlog, self); per: what one increment counts.
    """
    if not isinstance(decls, dict) or not decls:
        raise SnapshotError("SLOTS must be a non-empty dict")
    for name, d in decls.items():
        if not isinstance(name, str) or not SLOT_NAME_RE.match(name):
            raise SnapshotError("slot name invalid: %r" % (name,))
        if not isinstance(d, dict) or set(d) != set(SLOT_DECL_KEYS):
            raise SnapshotError("slot %s: declaration needs exactly %s" % (name, SLOT_DECL_KEYS))
        if d["kind"] not in KINDS:
            raise SnapshotError("slot %s: kind must be one of %s" % (name, KINDS))
        if d["per"] not in SLOT_PER:
            raise SnapshotError("slot %s: per must be one of %s" % (name, SLOT_PER))
        if not isinstance(d["group"], str) or not d["group"]:
            raise SnapshotError("slot %s: group must be a non-empty string" % name)
        if name.split(".", 1)[0] != d["group"]:
            raise SnapshotError("slot %s: name must start with its group %r" % (name, d["group"]))
        if d["site"] is not None and not isinstance(d["site"], str):
            raise SnapshotError("slot %s: site must be a string or null" % name)
    return decls


def encode_snapshot(registry, seq, t_mono_ns, meta_bytes):
    payload = registry.values.tobytes()
    if sys.byteorder != "little":
        a = array.array("q", payload)
        a.byteswap()
        payload = a.tobytes()
    n_values = len(payload) // 8
    return _HDR.pack(MAGIC, VERSION, len(meta_bytes), seq, t_mono_ns, n_values,
                     registry.meta_gen) + meta_bytes + payload


class Snapshot:
    __slots__ = ("seq", "t_mono_ns", "meta_gen", "meta", "raw", "values")

    def __init__(self, seq, t_mono_ns, meta_gen, meta, raw, values):
        self.seq, self.t_mono_ns, self.meta_gen = seq, t_mono_ns, meta_gen
        self.meta, self.raw, self.values = meta, raw, values

    @property
    def pid(self):
        return self.meta.get("pid")


def decode_snapshot(buf):
    if len(buf) < HEADER_SIZE:
        raise SnapshotError("snapshot too short")
    magic, version, meta_len, seq, t_mono, n_values, meta_gen = _HDR.unpack_from(buf, 0)
    if magic != MAGIC or version != VERSION:
        raise SnapshotError("bad snapshot magic/version")
    end_meta = HEADER_SIZE + meta_len
    if len(buf) != end_meta + 8 * n_values:
        raise SnapshotError("snapshot length %d != expected %d" % (len(buf),
                                                                     end_meta + 8 * n_values))
    try:
        meta = json.loads(bytes(buf[HEADER_SIZE:end_meta]).decode("utf-8"))
    except ValueError as e:
        raise SnapshotError("bad snapshot meta: %s" % e)
    if meta.get("schema") != SCHEMA:
        raise SnapshotError("snapshot meta schema %r" % meta.get("schema"))
    raw = array.array("q", bytes(buf[end_meta:]))
    if sys.byteorder != "little":
        raw.byteswap()
    values = {}
    for s in meta["slots"]:
        off, ln = s["offset"], s["length"]
        if off + ln > n_values:
            # slot registered after this payload was copied cannot happen with meta cached at
            # the same time; treat as corruption
            raise SnapshotError("slot %s outside payload" % s["name"])
        values[s["name"]] = raw[off] if ln == 1 else list(raw[off:off + ln])
    return Snapshot(seq, t_mono, meta_gen, meta, raw, values)


def read_snapshot_file(path):
    with open(path, "rb") as f:
        return decode_snapshot(f.read())


def list_snapshot_files(shm_dir):
    """{pid: path} for every <pid>.bin in shm_dir (missing dir -> {})."""
    out = {}
    try:
        names = os.listdir(shm_dir)
    except FileNotFoundError:
        return out
    for n in names:
        m = _FILE_RE.match(n)
        if m:
            out[int(m.group(1))] = os.path.join(shm_dir, n)
    return out


def snapshot_path(shm_dir, pid=None):
    return os.path.join(shm_dir, "%d.bin" % (os.getpid() if pid is None else pid))


def ensure_own_dir(path):
    """Create the snapshot directory `path` (mode 0700, parents as needed) unless it exists, then
    require it to be a real directory owned by this user: a symlink, a non-directory or another
    user's directory is refused, so snapshot files are never written where someone else decides.
    """
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
    except FileExistsError:
        pass  # a non-directory (or dangling symlink) is there: refused just below
    st = os.lstat(os.path.normpath(path))
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise SnapshotError("snapshot dir %s is not a directory (symlinks are refused)" % path)
    if st.st_uid != os.geteuid():
        raise SnapshotError("snapshot dir %s is owned by uid %d, not by this user (uid %d)"
                            % (path, st.st_uid, os.geteuid()))
    return path


_TMP_FLAGS = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
              | getattr(os, "O_CLOEXEC", 0))


class SnapshotWriter:
    """Background thread writing the registry to <shm_dir>/<pid>.bin every interval_ns.

    meta_extra: dict merged into the meta (identity, versions, role_hint, ...). Call stop() at the
    end; it writes one final snapshot so the last counts are never lost.

    The meta also carries "threads": {native_id: Python thread name} for every live Python
    thread of the process, refreshed whenever that set changes. This is how a process declares
    its own threads to the orchestrator's thread inventory (INTERFACES 5.3, C13).
    """

    def __init__(self, registry, shm_dir, interval_ns, meta_extra=None, pid=None):
        if not isinstance(interval_ns, int) or interval_ns <= 0:
            raise SnapshotError("interval_ns must be an int > 0")
        if not shm_dir:
            raise SnapshotError("shm_dir is required")
        ensure_own_dir(shm_dir)
        self.registry = registry
        self.pid = os.getpid() if pid is None else pid
        self.path = snapshot_path(shm_dir, self.pid)
        self._tmp = os.path.join(shm_dir, ".%d.bin.tmp" % self.pid)
        self.interval_ns = interval_ns
        self._extra = dict(meta_extra or {})
        self._meta_gen = -1
        self._meta_threads = None
        self._meta_bytes = b""
        self.seq = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread = None
        self._wlock = threading.Lock()

    def _meta_locked(self):
        """Meta bytes for the current slot table; caller holds registry._lock."""
        reg = self.registry
        threads = tuple(sorted((t.native_id, t.name) for t in threading.enumerate()
                               if t.native_id is not None))
        if reg.meta_gen != self._meta_gen or threads != self._meta_threads:
            meta = {"schema": SCHEMA, "pid": self.pid, "ppid": os.getppid(),
                    "starttime_ticks": proc_starttime_ticks(self.pid), "argv": list(sys.argv),
                    "interval_ns": self.interval_ns, "slots": [dict(x) for x in reg.slots],
                    "threads": {str(tid): name for tid, name in threads}}
            for k, v in self._extra.items():
                meta.setdefault(k, v)
            for k in ("identity", "role_hint", "versions"):
                meta.setdefault(k, None)
            mb = json.dumps(meta, sort_keys=True, allow_nan=False).encode("utf-8")
            mb += b" " * ((-len(mb)) % 8)
            self._meta_bytes, self._meta_gen, self._meta_threads = mb, reg.meta_gen, threads
        return self._meta_bytes

    def write_now(self):
        """Copy the registry and write one snapshot file (atomic replace)."""
        with self._wlock:
            reg = self.registry
            with reg._lock:  # registration is excluded, so meta and payload agree
                meta = self._meta_locked()
                self.seq += 1
                data = encode_snapshot(reg, self.seq, time.monotonic_ns(), meta)
            # O_NOFOLLOW: a symlink planted at the temporary name is refused, never followed.
            with os.fdopen(os.open(self._tmp, _TMP_FLAGS, 0o644), "wb") as f:
                f.write(data)
            os.replace(self._tmp, self.path)

    def _run(self):
        nxt = time.monotonic_ns() + self.interval_ns
        while not self._stop.is_set():
            delay = (nxt - time.monotonic_ns()) / 1e9
            if delay > 0 and self._stop.wait(delay):
                break
            try:
                self.write_now()
            except OSError:
                self.errors += 1
            nxt += self.interval_ns
            now = time.monotonic_ns()
            if nxt < now:  # fell behind (e.g. starved); do not burst
                nxt = now + self.interval_ns

    def start(self):
        self.write_now()
        self._thread = threading.Thread(target=self._run, name="capkit-snapshot", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.write_now()

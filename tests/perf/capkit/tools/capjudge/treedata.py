# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read one launch tree's raw data (INTERFACES 5). Roles, workers and tiers are taken from the
`identity` / `schema` / `kind` inside each file, never from file or directory names. The only
name-based choices are the file type suffix (.json, .capbin, .bin, .jsonl), the fixed paths
INTERFACES assigns (raw/launch.json, raw/sidecar/threads.jsonl, ...) and FRAME_CAPTURE_FILES.
"""

import json
import os

from capcommon.capbin import CapbinError, read_capbin
from capcommon.exitcodes import ConfigError
from capcommon.jsonio import read_json, read_jsonl
from capcommon.lifecycle import inprocess_tool_info
from capcommon.snapshot import SnapshotError, read_snapshot_file

from . import stats

NS = 1_000_000_000
SUMMARY_SCHEMA = "capkit.tier_summary/1"
TIER_SCHEMA = "capkit.tier/1"
DRAIN_GATE_SCHEMA = "capkit.drain_gate/1"
# The frame files capstubs.frame_capture writes (nx_role --frame-capture-dir, anywhere under
# raw/): .bin by suffix but not snapshots and without an identity, so no reader of the judge
# opens them. Matched by exact file name only; every other .bin is still read as a snapshot.
FRAME_CAPTURE_FILES = frozenset(("frames.result.bin", "frames.request.bin"))


class TierData:
    """Everything filed under raw/tiers/<seq>_<label>/ for one tier."""

    def __init__(self, ctx, tier):
        self.tier = tier
        self.dir = ctx.tree.tier_dir(tier.seq, tier.label)
        self.exists = os.path.isdir(self.dir)
        self.summaries = {}  # role -> {worker: doc}
        self.capbins = {}  # kind -> [(identity, header, cols)]
        self.tier_json = None
        self.drain_gate = None
        self.snaps = {}  # pid -> [Snapshot] sorted by t_mono_ns
        self.errors = []  # record-integrity problems (misfiled, unreadable, duplicate)
        if self.exists:
            self._load(ctx)

    def _err(self, msg):
        self.errors.append(msg)

    def _check_tier_ident(self, ident, what):
        if not isinstance(ident, dict) or ident.get("tier_seq") != self.tier.seq \
                or ident.get("tier_label") != self.tier.label:
            self._err("%s: identity tier %r/%r is not tier %d/%s" % (
                what, (ident or {}).get("tier_seq"), (ident or {}).get("tier_label"),
                self.tier.seq, self.tier.label))
            return False
        return True

    def _load(self, ctx):
        for d, dirs, files in os.walk(self.dir):
            dirs.sort()
            for name in sorted(files):
                if name in FRAME_CAPTURE_FILES:
                    continue
                p = os.path.join(d, name)
                rel = os.path.relpath(p, ctx.root)
                try:
                    if name.endswith(".json"):
                        self._load_json(p, rel)
                    elif name.endswith(".capbin"):
                        self._load_capbin(p, rel)
                    elif name.endswith(".bin"):
                        snap = read_snapshot_file(p)
                        self.snaps.setdefault(snap.pid, []).append(snap)
                    else:
                        continue
                except (OSError, ValueError, ConfigError, CapbinError, SnapshotError) as e:
                    self._err("%s unreadable: %s" % (rel, e))
                    continue
                ctx.note_file(p)
        for pid in self.snaps:
            self.snaps[pid].sort(key=lambda s: s.t_mono_ns)

    def _load_json(self, p, rel):
        doc = read_json(p)
        schema = doc.get("schema") if isinstance(doc, dict) else None
        if schema == SUMMARY_SCHEMA:
            ident = doc.get("identity") or {}
            if not self._check_tier_ident(ident, rel):
                return
            per_role = self.summaries.setdefault(ident.get("role"), {})
            if ident.get("worker") in per_role:
                self._err("%s: duplicate summary for %s worker %s" % (rel, ident.get("role"),
                                                                     ident.get("worker")))
                return
            per_role[ident.get("worker")] = doc
        elif schema == TIER_SCHEMA:
            self.tier_json = doc
        elif schema == DRAIN_GATE_SCHEMA:
            self.drain_gate = doc
        else:
            self._err("%s: unexpected schema %r in a tier directory" % (rel, schema))

    def _load_capbin(self, p, rel):
        header, cols = read_capbin(p)
        ident = header.get("identity") or {}
        if not self._check_tier_ident(ident, rel):
            return
        ov = (header.get("meta") or {}).get("overflow", 0)
        if ov:
            self._err("%s: meta.overflow = %s rows dropped" % (rel, ov))
        self.capbins.setdefault(header.get("kind"), []).append((ident, header, cols))

    # ---- merged views ----------------------------------------------------------------------
    def rows(self, kind, role=None):
        """Merge every capbin of `kind` (optionally only one role) into {col: list}; also returns
        the list of (role, worker, meta) parts merged."""
        out, parts = {}, []
        for ident, header, cols in self.capbins.get(kind, []):
            if role is not None and ident.get("role") != role:
                continue
            parts.append((ident.get("role"), ident.get("worker"), header.get("meta") or {}))
            for k, arr in cols.items():
                out.setdefault(k, []).extend(arr)
        return out, parts

    def roles_with(self, kind):
        return sorted({ident.get("role") for ident, _h, _c in self.capbins.get(kind, [])})


def summary_value(doc, dotted):
    """Value at a dotted path inside a summary ('counts.ok'); None if absent."""
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def per_second(times, values, start_ns, end_ns, reducer=stats.median):
    """Bucket (time, value) pairs into whole seconds of [start, start + k*1s) (k = floor of the
    window length in s; a partial last second is dropped) and reduce each bucket. Empty seconds
    are None. Returns the list (len k)."""
    k = (end_ns - start_ns) // NS
    if k <= 0:
        return []
    buckets = [[] for _ in range(k)]
    for t, v in zip(times, values):
        if start_ns <= t < start_ns + k * NS:
            buckets[(t - start_ns) // NS].append(v)
    return [reducer(b) if b else None for b in buckets]


STAT_COLS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def first_last_in_window(rows, keyf, ws, we):
    """({key: first row}, {key: last row}) by t_rel_ns over the rows with ws <= t_rel_ns < we;
    keyf(row) gives the key, None skips the row (it is only called on rows inside the window)."""
    first, last = {}, {}
    for row in rows:
        t = row.get("t_rel_ns")
        if not _is_int(t) or not ws <= t < we:
            continue
        k = keyf(row)
        if k is None:
            continue
        if k not in first or t < first[k]["t_rel_ns"]:
            first[k] = row
        if k not in last or t > last[k]["t_rel_ns"]:
            last[k] = row
    return first, last


def cpu_tick_deltas(cpu_rows, cores, ws, we):
    """/proc/stat tick deltas of `cores` over [ws, we) from sidecar cpus.jsonl rows: the last
    minus the first row of each core inside the window. Returns ({core: {<STAT_COLS>: delta,
    "t_first_rel_ns", "t_last_rel_ns"}}, None) or (None, why). Unusable (why) when `cores` is
    empty, a core has fewer than two rows in the window, or any of its rows in the window has a
    non-integer tick column (the sidecar writes a row of nulls while a watched core is offline;
    such a core is not counted, and the reading is incomplete rather than partial)."""
    if not cores:
        return None, "no cores"
    bad = {}

    def key(row):
        c = row.get("cpu")
        if c not in cores:
            return None
        if not all(_is_int(row.get(k)) for k in STAT_COLS):
            bad.setdefault(c, []).append(row.get("t_rel_ns"))
            return None
        return c

    first, last = first_last_in_window(cpu_rows, key, ws, we)
    if bad:
        return None, "core(s) %s offline or unreadable in the window (rows of nulls at " \
                     "t_rel_ns %s)" % (sorted(bad), {c: v[:3] for c, v in sorted(bad.items())})
    missing = sorted(c for c in cores if c not in first or first[c] is last[c])
    if missing:
        return None, "fewer than two sidecar cpu rows in the window for cores %s" % missing
    out = {}
    for c in sorted(cores):
        d = {k: last[c][k] - first[c][k] for k in STAT_COLS}
        d.update(t_first_rel_ns=first[c]["t_rel_ns"], t_last_rel_ns=last[c]["t_rel_ns"])
        out[c] = d
    return out, None


def role_cpus(pin_table):
    """{role: set of cpus} from pin_table.json (None -> {})."""
    return {role: set(v.get("cpus") or [])
            for role, v in ((pin_table or {}).get("roles") or {}).items()}


class LaunchData:
    """Launch-level raw files, loaded lazily and cached."""

    def __init__(self, ctx):
        self.ctx = ctx
        self._cache = {}

    def _get(self, key, loader):
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    def _json(self, path):
        if not os.path.isfile(path):
            return None
        doc = read_json(path)
        self.ctx.note_file(path)
        return doc

    def _jsonl(self, path):
        if not os.path.isfile(path):
            return None
        header, rows = read_jsonl(path)
        self.ctx.note_file(path)
        cols = header["columns"]
        return header, [dict(zip(cols, r)) for r in rows]

    @property
    def roles_json(self):
        return self._get("roles", lambda: self._json(self.ctx.tree.roles_json))

    def procs_by_pid(self):
        rj = self.roles_json or {}
        return {p["pid"]: p for p in rj.get("procs", [])}

    def procs_of_role(self, role):
        return [p for p in (self.roles_json or {}).get("procs", []) if p.get("role") == role]

    @property
    def pin_table(self):
        return self._get("pin", lambda: self._json(self.ctx.tree.env("pin_table.json")))

    @property
    def gpu_idle(self):
        return self._get("gpu_idle", lambda: self._json(self.ctx.tree.env("gpu_idle.json")))

    @property
    def overlay(self):
        return self._get("overlay",
                         lambda: self._json(self.ctx.tree.env("overlay_manifest.json")))

    @property
    def sidecar_meta(self):
        return self._get("sc_meta", lambda: self._json(self.ctx.tree.sidecar("meta.json")))

    def sidecar(self, name):
        return self._get("sc_" + name,
                         lambda: self._jsonl(self.ctx.tree.sidecar(name + ".jsonl")))

    def sidecar_interval_ns(self):
        m = self.sidecar_meta or {}
        v = m.get("interval_ns")  # capenv.sidecar writes the sampling period as int ns
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
        return None

    def thread_inventories(self):
        """[(doc, path)] for every procs/threads/*.json (process identity from content)."""
        def load():
            d = self.ctx.tree.procs("threads")
            out = []
            if os.path.isdir(d):
                for name in sorted(os.listdir(d)):
                    if name.endswith(".json"):
                        p = os.path.join(d, name)
                        out.append((self._json(p), p))
            return out
        return self._get("threads_inv", load)

    def version_records(self):
        """{pid: record} from raw/env/versions/*.json; the pid comes from the record."""
        def load():
            d = self.ctx.tree.env("versions")
            out, errors = {}, []
            if os.path.isdir(d):
                for name in sorted(os.listdir(d)):
                    if not name.endswith(".json"):
                        continue
                    p = os.path.join(d, name)
                    doc = self._json(p)
                    rec = doc.get("versions") if isinstance(doc, dict) and \
                        isinstance(doc.get("versions"), dict) else doc
                    pid = rec.get("pid") if isinstance(rec, dict) else None
                    if not isinstance(pid, int) or pid in out:
                        errors.append("%s: missing or duplicate pid" % name)
                        continue
                    out[pid] = rec
            return out, errors
        return self._get("versions", load)

    def ready_doc(self, role):
        """procs/ready/<role>.json (single-process role), or None."""
        return self._get("ready_" + role, lambda: self._json(self.ctx.tree.ready_file(role)))

    def ready_tool_info(self, role):
        """(info, None): what the in-process tool of `role` reported in its ready file
        (info.workers[0].info, capcommon.lifecycle.inprocess_tool_info); (None, why) when the
        ready file is missing or not of that shape."""
        doc = self.ready_doc(role)
        if not isinstance(doc, dict):
            return None, "no ready file"
        return inprocess_tool_info(doc)

    @property
    def launch_end(self):
        return self._get("launch_end", lambda: self._json(self.ctx.tree.launch_end_json))


def dumps_small(obj):
    return json.dumps(obj, sort_keys=True, allow_nan=False)

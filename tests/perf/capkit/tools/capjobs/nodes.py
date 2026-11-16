# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node lists for pinning and exclusion (calib-v2 DESIGN-r7a 3.1 P2-08a, 3.2, 4.5; INTERFACES
erratum calib-v2-A4): registered node_exclusions / node_list files, the live `sinfo` node set of
the partitions, exclude --all-but, the pin-next rotation, and the `scontrol show reservation`
parser of the maintenance gate. Every cluster query goes through capjobs.remote (ssh,
allow-listed, read-only)."""

import hashlib
import json
import os
import re

from capcommon import ledger
from capcommon.exitcodes import ConfigError
from capcommon.hashing import sha256_file

from . import remote

NODE_EXCLUSIONS_SCHEMA = "capkit.node_exclusions/1"
NODE_LIST_SCHEMA = "capkit.node_list/1"
NODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
# The only sinfo %t states a pinned job may be sent to (DESIGN 4.5 names down / drain; every
# other base state - maint, resv, unk, inval, boot, npc, plnd, ... - is unusable too, and so is
# any flag suffix but "-" (planned by the backfill scheduler): * not responding, ~ powered off,
# # powering up, ! pending power down, % powering down, $ maintenance reservation, @ pending
# reboot, ^ rebooting).
USABLE_STATES = frozenset({"idle", "mix", "mixed", "alloc", "allocated", "comp", "completing"})
USABLE_STATE_SUFFIX = "-"


class NodesError(ConfigError):
    pass


def registered_path(workdir, path, kind):
    """(relpath, sha256) of a file REGISTRY.tsv lists under kind with this sha at this path."""
    sha = sha256_file(path)
    row = ledger.registry_lookup(os.path.join(workdir, "REGISTRY.tsv"), sha)
    if row is None or ledger.registry_kind(row) != kind:
        raise NodesError("%s (sha %s) is not in REGISTRY.tsv as kind=%s" % (path, sha, kind))
    if os.path.realpath(os.path.join(workdir, row["relpath"])) != os.path.realpath(path):
        raise NodesError("%s is not the registered relpath %s" % (path, row["relpath"]))
    return row["relpath"], sha


def registered_doc(workdir, path, kind, schema):
    """(relpath, sha256, doc) of a registered JSON file of this schema."""
    rel, sha = registered_path(workdir, path, kind)
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or doc.get("schema") != schema:
        raise NodesError("%s is not %s" % (path, schema))
    return rel, sha, doc


def latest_registered(workdir, kind):
    """The REGISTRY row of kind registered last (file order), or None."""
    rows = [r for r in ledger.read_tsv(os.path.join(workdir, "REGISTRY.tsv"),
                                       ledger.REGISTRY_COLUMNS) if ledger.registry_kind(r) == kind]
    return rows[-1] if rows else None


def node_exclusions(workdir, path, require_latest=True):
    """{"relpath", "sha256", "version", "nodes": {node: reason}} of a registered
    capkit.node_exclusions/1 file; with require_latest it must be the last registered version."""
    rel, sha, doc = registered_doc(workdir, path, "node_exclusions", NODE_EXCLUSIONS_SCHEMA)
    if require_latest:
        last = latest_registered(workdir, "node_exclusions")
        if last["sha256"] != sha:
            raise NodesError("%s is not the latest registered node_exclusions (%s)"
                             % (rel, last["relpath"]))
    nodes = doc.get("nodes")
    prev = doc.get("previous")
    if not isinstance(nodes, list) or not all(
            isinstance(n, dict) and isinstance(n.get("node"), str) and NODE_RE.match(n["node"])
            and isinstance(n.get("reason"), str) and n["reason"] for n in nodes) \
            or not (prev is None or (isinstance(prev, dict)
                                     and set(prev) == {"relpath", "sha256"})):
        raise NodesError("%s: needs version, nodes [{node, reason, evidence}] and previous "
                         "{relpath, sha256} or null" % rel)
    names = [n["node"] for n in nodes]
    if len(set(names)) != len(names):
        raise NodesError("%s lists a node twice" % rel)
    return {"relpath": rel, "sha256": sha, "version": doc.get("version"),
            "nodes": {n["node"]: n["reason"] for n in nodes}}


def latest_node_exclusions(workdir):
    """node_exclusions() of the last registered version, or None when none is registered."""
    row = latest_registered(workdir, "node_exclusions")
    if row is None:
        return None
    return node_exclusions(workdir, os.path.join(workdir, row["relpath"]))


def node_list(workdir, path):
    """{"relpath", "sha256", "nodes"} of a registered capkit.node_list/1 file (nodes sorted)."""
    rel, sha, doc = registered_doc(workdir, path, "node_list", NODE_LIST_SCHEMA)
    nodes = doc.get("nodes")
    if not isinstance(nodes, list) or not nodes or not all(
            isinstance(n, str) and NODE_RE.match(n) for n in nodes) \
            or nodes != sorted(set(nodes)):
        raise NodesError("%s: nodes must be a non-empty sorted list of distinct node names" % rel)
    return {"relpath": rel, "sha256": sha, "nodes": list(nodes)}


def sinfo_nodes(runner, login_host, partitions, states=False):
    """{node: state or None} of the partitions from a live `sinfo -N -h -o %N[ %t] -p <p>` per
    partition, and the raw results. Any failed query raises (nothing is computed from a partial
    view)."""
    if not partitions or not all(isinstance(p, str) and p for p in partitions):
        raise NodesError("sinfo needs a non-empty list of partitions")
    fmt = "%N %t" if states else "%N"
    out, raw = {}, []
    for part in partitions:
        r = remote.run_remote(runner, login_host, ["sinfo", "-N", "-h", "-o", fmt, "-p", part])
        raw.append({"remote_argv": r["remote_argv"], "rc": r["rc"], "stdout": r["stdout"]})
        if r["rc"] != 0:
            raise NodesError("sinfo -p %s failed rc=%r: %s" % (part, r["rc"],
                                                              (r["stderr"] or "").strip()))
        for line in (r["stdout"] or "").splitlines():
            cells = line.split()
            if not cells:
                continue
            if len(cells) != (2 if states else 1) or not NODE_RE.match(cells[0]):
                raise NodesError("sinfo -p %s: unparsable line %r" % (part, line))
            out[cells[0]] = cells[1] if states else None
    if not out:
        raise NodesError("sinfo listed no node of %s" % partitions)
    return out, raw


def usable_state(state):
    """An allow-list: a base state in USABLE_STATES with no flag suffix or the "-" one."""
    s = (state or "").lower()
    if s.endswith(USABLE_STATE_SUFFIX):
        s = s[:-len(USABLE_STATE_SUFFIX)]
    return s in USABLE_STATES


def rotation(nodes, seed_base, key):
    """DESIGN 4.5: the registered nodes sorted by sha256(seed_base|key|node)."""
    return sorted(nodes, key=lambda n: hashlib.sha256(
        ("%s|%s|%s" % (seed_base, key, n)).encode("utf-8")).hexdigest())


def pin_choice(order, previous, excluded, live):
    """First usable node after previous (from the start without one), wrapping around once.
    excluded {node: reason}; live {node: sinfo state}. Returns (node or None, skipped)."""
    if previous is not None and previous not in order:
        raise NodesError("previous target %s is not in the registered node_list" % previous)
    start = 0 if previous is None else order.index(previous) + 1
    skipped = []
    for i in range(len(order)):
        node = order[(start + i) % len(order)]
        if node in excluded:
            skipped.append({"node": node, "reason": "node_exclusions: %s" % excluded[node]})
        elif node not in live:
            skipped.append({"node": node, "reason": "not in the partitions (sinfo)"})
        elif not usable_state(live[node]):
            skipped.append({"node": node, "reason": "sinfo state %s" % live[node]})
        else:
            return node, skipped
    return None, skipped


# ---- hostlists and reservations -----------------------------------------------------------------

def _split_top(text):
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth < 0:
                raise NodesError("unbalanced hostlist %r" % text)
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if depth:
        raise NodesError("unbalanced hostlist %r" % text)
    return parts + [cur]


def _expand_one(piece):
    m = re.match(r"^([^\[\]]*)\[([^\[\]]+)\](.*)$", piece)
    if not m:
        if not NODE_RE.match(piece):
            raise NodesError("bad host name %r" % piece)
        return [piece]
    head, body, tail = m.groups()
    out = []
    for rng in body.split(","):
        lo, _, hi = rng.partition("-")
        if not lo.isdigit() or (hi and not hi.isdigit()):
            raise NodesError("bad hostlist range %r" % rng)
        hi = hi or lo
        if int(hi) < int(lo):
            raise NodesError("bad hostlist range %r" % rng)
        out += ["%s%s" % (head, str(n).zfill(len(lo))) for n in range(int(lo), int(hi) + 1)]
    return [x for h in out for x in _expand_one(h + tail)]


def expand_hostlist(text):
    """'a-[001-003,007],b' -> ['a-001', 'a-002', 'a-003', 'a-007', 'b']."""
    if not text or text == "(null)":
        return []
    return [h for piece in _split_top(text) for h in _expand_one(piece)]


RESV_START = re.compile(r"\s+(?=ReservationName=)")
RESV_KEYS = ("ReservationName", "StartTime", "EndTime", "Nodes", "PartitionName", "Flags")


def _resv_value(rec, key):
    """The value of `key=` in one record (whitespace before the key or at the start; the value
    runs to the next whitespace), None when absent; two occurrences raise."""
    vals = re.findall(r"(?:^|\s)%s=(\S*)" % re.escape(key), rec)
    if len(vals) > 1:
        raise NodesError("reservation record has %s twice: %r" % (key, rec[:80]))
    return vals[0] if vals else None


def parse_reservations(text):
    """Records of `scontrol show reservation` as [{name, start, end, nodes (list or "ALL"),
    partition, flags}]; times are the controller's local wall clock strings. A record runs from
    one `ReservationName=` to the next; inside it the keys are read by name, so other fields may
    hold spaces (Comment=weekly maintenance window). Raises on anything it cannot read (the gate
    then defers)."""
    body = (text or "").strip()
    if not body:
        raise NodesError("empty scontrol show reservation output")
    if body.startswith("No reservations"):
        return []
    if not body.startswith("ReservationName="):
        raise NodesError("scontrol show reservation output does not start with a record: %r"
                         % body[:80])
    recs = []
    for chunk in RESV_START.split(body):
        if chunk.strip():
            recs.append({k: v for k in RESV_KEYS for v in [_resv_value(chunk, k)]
                         if v is not None})
    out = []
    for r in recs:
        if not all(k in r for k in ("ReservationName", "StartTime", "EndTime", "Nodes")):
            raise NodesError("reservation %r lacks StartTime / EndTime / Nodes"
                             % r.get("ReservationName"))
        flags = r.get("Flags", "").split(",")
        nodes = "ALL" if r["Nodes"] == "ALL" or "ALL_NODES" in flags \
            else expand_hostlist(r["Nodes"])
        out.append({"name": r["ReservationName"], "start": r["StartTime"], "end": r["EndTime"],
                    "nodes": nodes, "partition": r.get("PartitionName", "(null)"),
                    "flags": flags})
    return out

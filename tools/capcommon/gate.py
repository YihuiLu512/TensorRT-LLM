# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drain-gate reading arithmetic (INTERFACES.md section 8), shared by the orchestrator (live
polling) and the judge (re-checking drain_gate.json), so both compute readings identically.
"""

import time

from .exitcodes import ConfigError
from .identity import LAUNCH_KEY_FIELDS
from .snapshot import SnapshotError, list_snapshot_files, read_snapshot_file


class GateError(ConfigError):
    pass


def snapshot_fields(snap):
    """The per-snapshot facts a poll records, so the judge can re-apply check_snapshot offline."""
    ident = snap.meta.get("identity") or None
    lk = None if not isinstance(ident, dict) else [ident.get(k) for k in LAUNCH_KEY_FIELDS]
    return {"pid": snap.pid, "seq": snap.seq, "t_mono_ns": snap.t_mono_ns,
            "starttime_ticks": snap.meta.get("starttime_ticks"), "launch_key": lk}


def check_snapshot(fields, proc, now_ns, max_age_ns, prev_seq, launch_key):
    """Is this snapshot a usable reading of `proc` (a roles.json entry) at now_ns?

    Returns (ok, reason). Not usable: pid or starttime_ticks differ from the registration (pid
    reuse / another process), identity is not this launch's, older than max_age_ns (writer dead
    or starved), or seq did not advance since the previous poll (prev_seq given).
    """
    if fields["pid"] != proc["pid"]:
        return False, "pid %r != registered %r" % (fields["pid"], proc["pid"])
    st = proc.get("starttime_ticks")
    if st is None or fields["starttime_ticks"] != st:
        return False, "starttime_ticks %r != registered %r" % (fields["starttime_ticks"], st)
    if fields["launch_key"] != list(launch_key):
        return False, "identity launch key %r is not this launch's" % (fields["launch_key"],)
    age = now_ns - fields["t_mono_ns"]
    if age > max_age_ns:
        return False, "stale: age %d ns > %d" % (age, max_age_ns)
    if prev_seq is not None and fields["seq"] <= prev_seq:
        return False, "seq %d did not advance (previous poll %d)" % (fields["seq"], prev_seq)
    return True, None


def snapshots_by_role(shm_dir, procs, launch_key, max_age_ns, prev_seqs=None, now_ns=None):
    """Read and vet the current snapshot of every registered process.

    procs: the "procs" list of roles.json (dicts with pid, role, starttime_ticks). launch_key:
    the six launch-key values of this launch (Identity.launch_key()). prev_seqs: {pid: seq} from
    the previous poll of the same gate, or None on the first poll. Returns
    (by_role {role: [Snapshot]}, errors {pid: reason}, fields {pid: snapshot_fields}) --
    unusable snapshots are reported in errors and left out of by_role, never read as zero.
    """
    files = list_snapshot_files(shm_dir)
    snaps = {}
    out, errors, fields = {}, {}, {}
    for p in procs:
        path = files.get(p["pid"])
        if path is None:
            errors[p["pid"]] = "no snapshot file"
            continue
        try:
            snaps[p["pid"]] = read_snapshot_file(path)
        except (OSError, SnapshotError) as e:
            errors[p["pid"]] = str(e)
    if now_ns is None:
        now_ns = time.monotonic_ns()  # read after the files, so ages are never understated
    for p in procs:
        snap = snaps.get(p["pid"])
        if snap is None:
            continue
        f = snapshot_fields(snap)
        fields[p["pid"]] = f
        prev = None if prev_seqs is None else prev_seqs.get(p["pid"])
        ok, why = check_snapshot(f, p, now_ns, max_age_ns, prev, launch_key)
        if not ok:
            errors[p["pid"]] = why
            continue
        out.setdefault(p["role"], []).append(snap)
    return out, errors, fields


def planned_polls(start_rel_ns, end_rel_ns, poll_ns):
    """Planned poll times start + i*poll_ns that are <= end_rel_ns (absolute schedule).

    The orchestrator polls at exactly these planned times until the gate passes or the deadline;
    the judge recomputes this list (end = t_pass or deadline) and reconciles it with the polls
    recorded in drain_gate.json (C11: expected = len(planned_polls(...)), never zero-trusting).
    """
    if poll_ns <= 0:
        raise GateError("poll_ns must be > 0")
    if end_rel_ns < start_rel_ns:
        return []
    return list(range(start_rel_ns, end_rel_ns + 1, poll_ns))


def evaluate_reading(reading, snaps):
    """Value of one gate reading: sum over terms of sign * agg(slot over the role's snapshots).

    Raises GateError when a role has no snapshots or a snapshot lacks the slot (missing data must
    never count as zero).
    """
    total = 0
    detail = []
    for t in reading["terms"]:
        role_snaps = snaps.get(t["role"])
        if not role_snaps:
            raise GateError("reading %s: no snapshots for role %s" % (reading["name"], t["role"]))
        vals = []
        for s in role_snaps:
            if t["slot"] not in s.values:
                raise GateError("reading %s: pid %s has no slot %s"
                                % (reading["name"], s.pid, t["slot"]))
            v = s.values[t["slot"]]
            if not isinstance(v, int):
                raise GateError("reading %s: slot %s is not scalar" % (reading["name"], t["slot"]))
            vals.append(v)
        agg = sum(vals) if t["agg"] == "sum" else max(vals)
        total += t["sign"] * agg
        detail.append({"role": t["role"], "slot": t["slot"], "agg": t["agg"], "sign": t["sign"],
                       "value": agg, "n_procs": len(vals),
                       "seqs": [s.seq for s in role_snaps]})
    return total, detail


def evaluate_gate(gate, snaps, band=None):
    """One poll. Returns (satisfied, {name: value or None}, {name: detail or error}).

    rule abs_max: every value <= reading["abs_max"]. rule baseline_band: band[name] = (lo, hi)
    from a registered calibration file; value must lie in [lo, hi]. A reading that cannot be
    computed makes the poll unsatisfied.
    """
    values, details = {}, {}
    ok = True
    for r in gate["readings"]:
        try:
            v, d = evaluate_reading(r, snaps)
        except GateError as e:
            values[r["name"]], details[r["name"]] = None, {"error": str(e)}
            ok = False
            continue
        values[r["name"]], details[r["name"]] = v, d
        if gate["rule"] == "abs_max":
            ok = ok and v <= r["abs_max"]
        elif gate["rule"] == "baseline_band":
            if band is None or r["name"] not in band:
                details[r["name"]] = {"error": "no registered band for this reading", "terms": d}
                ok = False
            else:
                lo, hi = band[r["name"]]
                ok = ok and lo <= v <= hi
        else:
            raise GateError("unknown gate rule %r" % gate["rule"])
    return ok, values, details


class GateTracker:
    """Applies settle_polls: the gate passes after that many consecutive satisfied polls."""

    def __init__(self, gate):
        self.gate = gate
        self.consecutive = 0
        self.n_polls = 0
        self.passed = False
        self.t_pass_rel_ns = None

    def update(self, satisfied, t_rel_ns):
        self.n_polls += 1
        self.consecutive = self.consecutive + 1 if satisfied else 0
        if not self.passed and self.consecutive >= self.gate["settle_polls"]:
            self.passed = True
            self.t_pass_rel_ns = t_rel_ns
        return self.passed

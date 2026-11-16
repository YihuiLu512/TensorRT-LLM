# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capenv.canary: concurrent frequency canary (design 2.5 D5; INTERFACES.md 5.6, 11).

One thread, pinned to exactly one core (--cpus), runs a fixed pure-Python work unit forever and
counts completed units per interval. The rate is a reading of the core's frequency state during
the measurement window; the canary's own presence is a constant shared by all tiers.

Fixed workload: UNIT_OPS iterations of an integer LCG per unit (code constant, recorded in the
header and meta; --unit-ops may override it, the value used is recorded). Interval boundaries
follow the absolute schedule origin + k * interval; each row is written at the first unit
boundary after the planned tick and carries the actual time (t_rel_ns), so a late boundary is
visible, never hidden.

canary.jsonl columns: t_rel_ns, iters (units completed since the previous row), thread_cpu_ns
(thread CPU time consumed since the previous row), cpu (last-run CPU read from
/proc/thread-self/stat). With --shm-dir the canary also publishes a capcommon snapshot (written
inline at each boundary, no extra thread) with counters canary.loop.iters and
canary.loop.thread_cpu_ns, so the sidecar can put it into the same row as /proc readings.

No other threads are created (signal handling only sets a flag checked once per unit).
"""

import argparse
import os
import sys
import time

from capcommon import exitcodes
from capcommon.cli import add_common_args, common_from_args
from capcommon.exitcodes import ConfigError
from capcommon.jsonio import JsonlWriter, write_json_stamped
from capcommon.lifecycle import StopController, pin_self, write_ready_file
from capcommon.snapshot import SlotRegistry, SnapshotWriter
from capcommon.versions import version_record

SCHEMA = "capkit.canary/1"
META_SCHEMA = "capkit.canary.meta/1"
COLUMNS = ("t_rel_ns", "iters", "thread_cpu_ns", "cpu")
UNIT_OPS = 2000


def work_unit(x, n):
    """The fixed workload: n steps of a 31-bit LCG. Pure Python, holds the GIL."""
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
    return x


def last_cpu():
    try:
        with open("/proc/thread-self/stat", "rb") as f:
            raw = f.read().decode()
        return int(raw[raw.rfind(")") + 2:].split()[36])
    except (OSError, ValueError, IndexError):
        return None


def run(origin_ns, interval_ns, unit_ops, writer, stop, snap=None, counters=None,
        max_rows=None):
    """Main loop. Returns stats dict. max_rows (tests only) stops after that many rows."""
    now = time.monotonic_ns
    tcpu = time.thread_time_ns
    t = now()
    k = -((origin_ns - t) // interval_ns)
    nxt = origin_ns + k * interval_ns
    iters = 0
    total = 0
    cpu0 = tcpu()
    x = 1
    late_max = 0
    rows = 0
    while not stop.requested:
        x = work_unit(x, unit_ops)
        iters += 1
        t = now()
        if t >= nxt:
            c = tcpu()
            writer.write_row((t - origin_ns, iters, c - cpu0, last_cpu()))
            total += iters
            if counters is not None:
                counters[0][counters[1]] = total
                counters[0][counters[2]] = c
                snap.write_now()
            late_max = max(late_max, t - nxt)
            rows += 1
            iters = 0
            cpu0 = c
            nxt += interval_ns
            if nxt <= t:  # a whole interval was missed: realign, do not emit catch-up rows
                nxt = origin_ns + (-((origin_ns - t) // interval_ns)) * interval_ns
                if nxt <= t:
                    nxt += interval_ns
            if max_rows is not None and rows >= max_rows:
                break
    return {"rows": rows, "total_units": total + iters, "max_boundary_lateness_ns": late_max,
            "lcg_state": x}


def build_parser():
    ap = argparse.ArgumentParser(prog="capenv.canary", description=__doc__.split("\n\n")[0])
    add_common_args(ap, cpus=True, inject=False, stop=True)
    g = ap.add_argument_group("canary (INTERFACES.md 11)")
    g.add_argument("--origin-ns", type=int, required=True)
    g.add_argument("--interval-ms", type=int, required=True)
    g.add_argument("--out", required=True, help="raw/canary/canary.jsonl (node-local tree)")
    o = ap.add_argument_group("optional")
    o.add_argument("--ready-file")
    o.add_argument("--unit-ops", type=int, default=UNIT_OPS)
    o.add_argument("--shm-dir", help="publish a snapshot here (read by the sidecar)")
    o.add_argument("--max-rows", type=int, default=None, help=argparse.SUPPRESS)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        common = common_from_args(args)
        if len(common.cpus) != 1:
            raise ConfigError("--cpus must name exactly one cpu, got %s" % common.cpus)
        if args.interval_ms <= 0 or args.unit_ops <= 0:
            raise ConfigError("--interval-ms and --unit-ops must be > 0")
        affinity = pin_self(common.cpus)
    except ConfigError as e:
        print("capenv.canary: %s" % e, file=sys.stderr)
        return exitcodes.CONFIG
    stop = StopController()
    interval_ns = args.interval_ms * 1_000_000
    snap = counters = None
    if args.shm_dir:
        reg = SlotRegistry()
        off_iters = reg.counter("canary.loop.iters")
        off_cpu = reg.counter("canary.loop.thread_cpu_ns")
        snap = SnapshotWriter(reg, args.shm_dir, interval_ns,
                              meta_extra={"identity": common.identity.to_dict(),
                                          "role_hint": common.identity.role,
                                          "versions": version_record(include_grpc=False)})
        counters = (reg.values, off_iters, off_cpu)
        snap.write_now()
    writer = JsonlWriter(args.out, SCHEMA, common.identity, COLUMNS, flush_every=1)
    if args.ready_file:
        write_ready_file(args.ready_file, common.identity, version_record(include_grpc=False),
                         affinity=affinity, info={"unit_ops": args.unit_ops, "threads": []})
    t_start = time.monotonic_ns()
    try:
        stats = run(common.origin_ns if common.origin_ns is not None else args.origin_ns,
                    interval_ns, args.unit_ops, writer, stop, snap, counters, args.max_rows)
    finally:
        writer.close()
    meta_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "canary.meta.json")
    write_json_stamped(meta_path, META_SCHEMA, common.identity, {
        "unit_ops": args.unit_ops, "unit_ops_default": UNIT_OPS, "interval_ns": interval_ns,
        "origin_ns": args.origin_ns, "affinity": affinity, "started_mono_ns": t_start,
        "ended_mono_ns": time.monotonic_ns(), "stats": stats, "workload": "31-bit LCG",
        "python": sys.version})
    return exitcodes.OK


if __name__ == "__main__":
    sys.exit(main())

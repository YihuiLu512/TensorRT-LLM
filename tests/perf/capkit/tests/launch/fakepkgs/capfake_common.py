# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared logic of the fake capkit tools used only by tests/launch (never deployed).

A fake tool parses the real INTERFACES.md 2 common arguments with capcommon.cli, publishes one
backlog gauge in its snapshot, and drives it from the tier table: `busy` during load tiers,
0 in drain tiers after --drain-delay-ms (or never with --never-drain).
"""

import argparse
import os
import socket
import subprocess
import sys
import time

from capcommon.cli import add_common_args, common_from_args
from capcommon.lifecycle import StopController, write_port_file, write_ready_file
from capcommon.snapshot import SlotRegistry, SnapshotWriter
from capcommon.versions import version_record


def parser(port, workers):
    p = argparse.ArgumentParser()
    add_common_args(p, loop=True, tiers=True, snapshot=True, ready=True, port=port)
    if workers:
        p.add_argument("--workers", type=int, required=True)
    p.add_argument("--slot", required=True)
    p.add_argument("--busy", type=int, default=3)
    p.add_argument("--drain-delay-ms", type=int, default=0)
    p.add_argument("--never-drain", action="store_true")
    p.add_argument("--crash-at-rel-ms", type=int, default=None)
    p.add_argument("--exit-after-end", action="store_true", help="client: exit after last tier")
    p.add_argument("--child-of", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--target-port-file", default=None)
    p.add_argument("--peer", default=None)
    p.add_argument("--rogue-snapshot-child", action="store_true",
                   help="start a snapshot-writing child that is not reported in the ready file")
    p.add_argument("--rogue", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--mp-helper", action="store_true",
                   help="start multiprocessing's resource tracker (a runtime helper process)")
    p.add_argument("--plain-child", action="store_true",
                   help="start an unreported long-lived child that is not a known helper")
    return p


def rogue_child(a, c):
    """Write snapshots under this process's own pid until SIGTERM (never reported)."""
    import signal  # noqa: PLC0415
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    reg = SlotRegistry()
    reg.counter("count.rogue")
    w = SnapshotWriter(reg, c.shm_dir, c.snapshot_interval_ns,
                       meta_extra={"identity": c.identity.to_dict(), "role_hint": "rogue",
                                   "versions": version_record(include_grpc=False)}).start()
    while not stop:
        time.sleep(0.02)
    w.stop()
    return 0


def gauge_value(tt, rel, a):
    t = tt.tier_at(rel)
    if t is None:
        return 0
    if t.kind != "drain":
        return a.busy
    if a.never_drain:
        return a.busy
    return a.busy if rel - t.start_ns < a.drain_delay_ms * 1_000_000 else 0


def run(port, workers):
    a = parser(port, workers).parse_args()
    c = common_from_args(a)
    if a.rogue:
        return rogue_child(a, c)
    tt = c.load_tier_table()
    stop = StopController()
    ident = c.identity
    reg = SlotRegistry()
    g = reg.gauge(a.slot)  # offset into reg.values
    meta = {"identity": ident.to_dict(), "versions": version_record(include_grpc=False),
            "role_hint": ident.role}
    w = SnapshotWriter(reg, c.shm_dir, c.snapshot_interval_ns, meta_extra=meta).start()
    children = []
    if a.child_of is None and workers and a.workers > 1:
        for k in range(1, a.workers):
            argv = [sys.executable, "-m", sys.modules["__main__"].__spec__.name] + [
                x for x in sys.argv[1:]] + ["--worker", str(k), "--child-of", str(os.getpid())]
            children.append((k, subprocess.Popen(argv)))
    rogues = []
    if a.mp_helper and a.child_of is None:
        from multiprocessing import resource_tracker  # noqa: PLC0415
        resource_tracker.ensure_running()
    if a.plain_child and a.child_of is None:
        rogues.append(subprocess.Popen(["sleep", "600"]))
    if a.rogue_snapshot_child and a.child_of is None:
        argv = [sys.executable, "-m", sys.modules["__main__"].__spec__.name] + [
            x for x in sys.argv[1:] if x != "--rogue-snapshot-child"] + ["--rogue"]
        rogues.append(subprocess.Popen(argv))
    sock = None
    if port and a.child_of is None:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(8)
        write_port_file(c.port_file, ident, "127.0.0.1", sock.getsockname()[1])
    if a.child_of is None:
        time.sleep(0.2)  # children start their snapshot writers
        write_ready_file(c.ready_file, ident, version_record(include_grpc=False),
                         children=[{"pid": p.pid, "worker": k, "role": ident.role}
                                   for k, p in children],
                         info={"threads": {}, "peer": a.peer,
                               "target_port_file": a.target_port_file})
    while not stop.requested:
        rel = time.monotonic_ns() - c.origin_ns
        if a.crash_at_rel_ms is not None and rel >= a.crash_at_rel_ms * 1_000_000 \
                and a.child_of is None:
            os._exit(9)
        reg.values[g] = gauge_value(tt, rel, a)
        if a.exit_after_end and rel >= tt.end_ns:
            break
        time.sleep(0.01)
    for p in [p for _k, p in children] + rogues:
        p.terminate()
    for p in [p for _k, p in children] + rogues:
        p.wait()
    w.stop()
    return 0

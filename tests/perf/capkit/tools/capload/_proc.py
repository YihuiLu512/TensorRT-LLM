# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parent / worker process plumbing shared by capload and capload.nullsse.

The parent (the one process the orchestrator starts per role) spawns W workers with the
multiprocessing "spawn" start method, talks to each over a Pipe, and is the only process that
handles SIGTERM / SIGINT (capcommon.lifecycle.StopController): the first signal is forwarded as
"stop", the second as "hard". Workers ignore both signals so a process-group signal from the
orchestrator is not counted twice; a closed pipe (parent gone) is treated as "stop".

The parent runs an event loop of the requested implementation (so its version record carries a
loop_impl like every other process of the role) and publishes a snapshot with the role's slots
registered (values stay 0): the drain gate reads every registered pid of a role and treats a
missing snapshot as "not satisfied" (INTERFACES.md 8.3).

Messages worker -> parent: ("ready", info), ("error", text, exit_code), ("done", exit_code).
Messages parent -> worker: "stop", "hard".
"""

import multiprocessing
import os
import signal
import sys
import traceback

from capcommon import exitcodes
from capcommon.aio import run as aio_run
from capcommon.exitcodes import CapkitError
from capcommon.lifecycle import StopController, pin_self
from capcommon.snapshot import SlotRegistry, SnapshotWriter
from capcommon.versions import version_record

# Worse exit codes win when the parent folds its workers' codes into its own.
_SEVERITY = {exitcodes.OK: 0, exitcodes.STOPPED_EARLY: 1, exitcodes.CONFIG: 2,
             exitcodes.RUNTIME: 3}


def fold_exit_codes(codes):
    worst, rank = exitcodes.OK, 0
    for c in codes:
        r = _SEVERITY.get(c, 4)  # unknown / crash (1, negative signal codes) is the worst
        if r > rank:
            worst, rank = (c if c is not None and c >= 0 else exitcodes.INTERNAL), r
    return worst


def worker_bootstrap(entry, cfg, worker, conn):
    """Body of every spawned worker: ignore signals, run entry(cfg, worker, conn) -> exit code."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    code = exitcodes.INTERNAL
    try:
        code = entry(cfg, worker, conn)
    except CapkitError as e:
        print("worker %d: %s" % (worker, e), file=sys.stderr)
        code = e.exit_code
        _safe_send(conn, ("error", str(e), code))
    except BaseException as e:  # noqa: BLE001 - reported to the parent, then re-raised as code 1
        traceback.print_exc()
        _safe_send(conn, ("error", "%s: %s" % (type(e).__name__, e), exitcodes.INTERNAL))
        code = exitcodes.INTERNAL
    _safe_send(conn, ("done", code))
    try:
        conn.close()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)  # skip multiprocessing's atexit join logic; parent already has the code


def _safe_send(conn, msg):
    try:
        conn.send(msg)
    except (OSError, EOFError, BrokenPipeError):
        pass


class ParentRunner:
    """Spawn workers, wait for them, forward stop requests, fold exit codes.

    entry: module-level function entry(cfg, worker, conn) -> exit code (picklable by name).
    on_ready(infos): called on the parent's loop once every worker reported ready; infos is the
    list of ready dicts in worker order (each has at least "pid"). It writes port / ready files.
    """

    def __init__(self, common, n_workers, entry, cfg, slots, role_hint):
        self.common = common
        self.n = n_workers
        self.entry = entry
        self.cfg = cfg
        self.slots = slots
        self.role_hint = role_hint
        self.procs = []
        self.conns = []
        self.infos = [None] * n_workers
        self.codes = [None] * n_workers
        self.errors = []
        self.ready_fail_code = None  # set when on_ready raised: the role never became ready
        self.all_ready = False
        self.on_ready = None
        self.versions = None
        self.affinity = None

    def run(self, on_ready):
        self.on_ready = on_ready
        if self.common.cpus:
            self.affinity = pin_self(self.common.cpus)
        return aio_run(self._main, self.common.loop)

    async def _main(self, loop):
        import asyncio  # noqa: PLC0415
        self.loop = loop
        self.versions = version_record(loop=loop, include_grpc=False)
        reg = SlotRegistry()
        reg.register_decls(self.slots)
        ident = self.common.identity
        writer = SnapshotWriter(reg, self.common.shm_dir, self.common.snapshot_interval_ns,
                                meta_extra={"identity": ident.to_dict(), "versions": self.versions,
                                            "role_hint": self.role_hint})
        writer.start()
        stopper = StopController(install=True)
        stopper.attach_loop(loop)
        self._finished = loop.create_future()
        stopper.on_stop(self._forward_stop)
        ctx = multiprocessing.get_context("spawn")
        try:
            for k in range(self.n):
                parent_conn, child_conn = ctx.Pipe(duplex=True)
                p = ctx.Process(target=worker_bootstrap, args=(self.entry, self.cfg, k, child_conn),
                                name="%s-w%d" % (ident.role, k), daemon=False)
                p.start()
                child_conn.close()
                self.procs.append(p)
                self.conns.append(parent_conn)
                loop.add_reader(parent_conn.fileno(), self._on_msg, k)
            await self._finished
        finally:
            for c in self.conns:
                try:
                    loop.remove_reader(c.fileno())
                except (OSError, ValueError):
                    pass
            for k, p in enumerate(self.procs):
                await loop.run_in_executor(None, p.join)
                if self.codes[k] is None:
                    self.codes[k] = p.exitcode
            writer.stop()
            await asyncio.sleep(0)
        for e in self.errors:
            print("capload parent: %s" % (e,), file=sys.stderr)
        extra = [] if self.ready_fail_code is None else [self.ready_fail_code]
        return fold_exit_codes(list(self.codes) + extra)

    def _forward_stop(self, hard):
        msg = "hard" if hard else "stop"
        for k, c in enumerate(self.conns):
            if self.codes[k] is None:
                _safe_send(c, msg)

    def _on_msg(self, k):
        conn = self.conns[k]
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            self.loop.remove_reader(conn.fileno())
            if self.codes[k] is None:
                self.codes[k] = -1  # resolved from p.exitcode after join
                self.errors.append("worker %d closed its pipe without a done message" % k)
                if not self.all_ready:
                    self._forward_stop(False)
            self._check_finished()
            return
        kind = msg[0]
        if kind == "ready":
            self.infos[k] = msg[1]
            if not self.all_ready and all(i is not None for i in self.infos):
                self.all_ready = True
                try:
                    self.on_ready(self.infos)
                except (CapkitError, OSError) as e:
                    self.errors.append("on_ready failed: %s" % e)
                    self.ready_fail_code = getattr(e, "exit_code", exitcodes.RUNTIME)
                    self._forward_stop(True)
        elif kind == "error":
            self.errors.append("worker %d: %s" % (k, msg[1]))
            if not self.all_ready:
                self._forward_stop(False)  # a worker failed before start: stop the others
        elif kind == "done":
            self.codes[k] = msg[1]
            self.loop.remove_reader(conn.fileno())
            self._check_finished()

    def _check_finished(self):
        if all(c is not None for c in self.codes) and not self._finished.done():
            for k, c in enumerate(self.codes):
                if c == -1:
                    self.codes[k] = None
            self._finished.set_result(None)

    def children(self):
        return [{"pid": p.pid, "worker": k, "role": self.common.identity.role}
                for k, p in enumerate(self.procs)]


class WorkerChannel:
    """Worker side of the pipe, driven from the worker's event loop."""

    def __init__(self, conn, loop, on_stop):
        self.conn = conn
        self.loop = loop
        self.on_stop = on_stop
        loop.add_reader(conn.fileno(), self._on_msg)

    def _on_msg(self):
        try:
            msg = self.conn.recv()
        except (EOFError, OSError):
            self.loop.remove_reader(self.conn.fileno())
            self.on_stop(False, "parent_pipe_closed")
            return
        if msg == "hard":
            self.on_stop(True, "signal")
        elif msg == "stop":
            self.on_stop(False, "signal")

    def send(self, msg):
        _safe_send(self.conn, msg)

    def close(self):
        try:
            self.loop.remove_reader(self.conn.fileno())
        except (OSError, ValueError):
            pass

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Port files, ready files, stop signalling (INTERFACES.md section 2.3)."""

import os
import signal
import threading
import time

from .clock import now_ns
from .exitcodes import ConfigError, RUNTIME, CapkitError
from .jsonio import read_json, read_json_stamped, write_json_stamped

PORTFILE_SCHEMA = "capkit.portfile/1"
READY_SCHEMA = "capkit.ready/1"
STOP_SCHEMA = "capkit.stop/1"


class WaitTimeout(CapkitError):
    exit_code = RUNTIME


def write_port_file(path, identity, host, port):
    if not isinstance(port, int) or not 0 < port < 65536:
        raise ConfigError("port must be an int in (0, 65536), got %r" % (port,))
    return write_json_stamped(path, PORTFILE_SCHEMA, identity,
                              {"host": host, "port": port, "pid": os.getpid(),
                               "t_mono_ns": now_ns()})


def _wait_for(path, timeout_s, poll_s):
    deadline = time.monotonic() + timeout_s
    while True:
        if os.path.exists(path):
            return
        if time.monotonic() >= deadline:
            raise WaitTimeout("timed out after %.1fs waiting for %s" % (timeout_s, path))
        time.sleep(poll_s)


def wait_port_file(path, timeout_s, poll_s=0.05):
    """Block until the port file exists; return (host, port, doc)."""
    _wait_for(path, timeout_s, poll_s)
    _ident, doc = read_json_stamped(path, PORTFILE_SCHEMA)
    return doc["host"], doc["port"], doc


def read_port_file(path):
    _ident, doc = read_json_stamped(path, PORTFILE_SCHEMA)
    return doc["host"], doc["port"], doc


def write_ready_file(path, identity, versions, affinity=None, children=None, info=None):
    """children: list of {"pid", "worker", "role"} for every child process of this tool."""
    kids = list(children or [])
    for c in kids:
        if set(c) != {"pid", "worker", "role"}:
            raise ConfigError("ready children entries need exactly pid/worker/role: %r" % (c,))
    if affinity is None:
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            affinity = None
    return write_json_stamped(path, READY_SCHEMA, identity,
                              {"t_mono_ns": now_ns(), "versions": versions,
                               "affinity": affinity, "children": kids, "info": info or {}})


def inprocess_ready_info(tool, worker, threads):
    """The ready file `info` of an in-process tool (capgrpc.common.write_inprocess_ready): its
    one worker entry (pid, worker, affinity, versions, aux_threads, info) under `workers`. Read
    the tool's own info back with inprocess_tool_info."""
    return {"tool": tool, "workers": [worker], "threads": threads}


def inprocess_tool_info(ready_doc):
    """(info, None): the tool info an in-process tool wrote (ready_doc info.workers[0].info,
    see inprocess_ready_info); (None, why) when the ready file does not have exactly one worker
    entry carrying an info object."""
    workers = ((ready_doc or {}).get("info") or {}).get("workers")
    if not isinstance(workers, list):
        return None, "ready file has no info.workers list"
    if len(workers) != 1:
        return None, "ready file info.workers has %d entries, not 1" % len(workers)
    info = workers[0].get("info") if isinstance(workers[0], dict) else None
    if not isinstance(info, dict):
        return None, "ready file info.workers[0].info is not an object"
    return info, None


def wait_ready_file(path, timeout_s, poll_s=0.05):
    _wait_for(path, timeout_s, poll_s)
    return read_json_stamped(path, READY_SCHEMA)


def read_stop_file(path):
    """Contents of control/stop.json (written by the orchestrator before signalling) or None."""
    if not path or not os.path.exists(path):
        return None
    doc = read_json(path)
    if not isinstance(doc, dict) or doc.get("schema") != STOP_SCHEMA:
        return {"schema": None, "unparsable": True}
    return doc


def pin_self(cpus):
    """sched_setaffinity(0, cpus) and return the read-back list (raises on mismatch)."""
    os.sched_setaffinity(0, set(cpus))
    got = sorted(os.sched_getaffinity(0))
    if got != sorted(set(cpus)):
        raise CapkitError("affinity read-back %s != requested %s" % (got, sorted(set(cpus))))
    return got


class StopController:
    """SIGTERM/SIGINT handling. First signal -> graceful stop; second -> hard stop.

    Use .requested / .hard flags, wait(timeout), or attach to an asyncio loop with
    attach_loop(loop, on_stop) so callbacks run on the loop thread.
    """

    def __init__(self, install=True):
        self.requested = False
        self.hard = False
        self.signals = []
        self._event = threading.Event()
        self._callbacks = []
        self._loop = None
        if install:
            signal.signal(signal.SIGTERM, self._handler)
            signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, _frame):
        self.trigger(signum)

    def trigger(self, signum=None):
        self.signals.append(signum)
        if self.requested:
            self.hard = True
        self.requested = True
        self._event.set()
        cbs = list(self._callbacks)
        if self._loop is not None:
            for cb in cbs:
                self._loop.call_soon_threadsafe(cb, self.hard)
        else:
            for cb in cbs:
                cb(self.hard)

    def on_stop(self, callback):
        """callback(hard: bool) is invoked on each stop signal."""
        self._callbacks.append(callback)

    def attach_loop(self, loop):
        """Deliver callbacks via loop.call_soon_threadsafe (signal handlers run on the main
        thread between bytecodes; this keeps loop state mutations on the loop)."""
        self._loop = loop

    def wait(self, timeout=None):
        return self._event.wait(timeout)

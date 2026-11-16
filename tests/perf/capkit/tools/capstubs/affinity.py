# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-NX core-set split inside one product role (design E9 parity: rank 0 against rank 0).

A role's pinned core set A (sorted) is cut from the end: the last `worker` cpus go to the rank 0
worker (NullWorker pins its own threads to them, from the S-NX config's worker_cpus), the
`monitor` cpus before them to the procmon sampler, and the rest stay with the role's own
processes (the frontends or the engine host, and every child they start afterwards).
"""

import os


def split_cpus(cpus, worker=0, monitor=0):
    """(rest, monitor_cpus, worker_cpus) of the sorted cpu list; ValueError if rest would be
    empty or a count is negative."""
    a = sorted(cpus)
    if worker < 0 or monitor < 0:
        raise ValueError("core counts must be >= 0")
    if worker + monitor >= len(a):
        raise ValueError("core set of %d cpus cannot give %d to the worker and %d to the monitor "
                         "and keep one" % (len(a), worker, monitor))
    w = a[len(a) - worker:] if worker else []
    m = a[len(a) - worker - monitor:len(a) - worker] if monitor else []
    return a[:len(a) - worker - monitor], m, w


def task_ids(pid=None):
    base = "/proc/self/task" if pid is None else "/proc/%d/task" % pid
    try:
        return [int(t) for t in os.listdir(base)]
    except OSError:
        return []


def pin_all_tasks(cpus, pid=None):
    """Set the affinity of every thread of a process (default: this one) to cpus. Returns
    {"before": [...], "after": [...], "tasks": n} (before / after of the process's main thread)."""
    target = os.getpid() if pid is None else pid
    before = sorted(os.sched_getaffinity(target))
    n = 0
    for tid in task_ids(pid):
        try:
            os.sched_setaffinity(tid, cpus)
            n += 1
        except (ProcessLookupError, PermissionError):
            continue
    return {"before": before, "after": sorted(os.sched_getaffinity(target)), "tasks": n}


def thread_names():
    """{native_id: name} of this process's live Python threads."""
    import threading  # noqa: PLC0415
    return {str(t.native_id): t.name for t in threading.enumerate() if t.native_id is not None}


def add_split_args(g):
    """Core-set split and process-tree sampler options (shared with serve_nx)."""
    g.add_argument("--worker-cores", type=int, default=0,
                   help="give the last N cpus of this role's core set to the rank 0 worker "
                        "(S-NX config worker_cpus; NullWorker pins its threads there)")
    g.add_argument("--procmon-out", help="run capstubs.procmon over this process tree, to here")
    g.add_argument("--procmon-interval-ms", type=int, default=1000)
    g.add_argument("--procmon-cores", type=int, default=1,
                   help="cpus (before the worker's) given to the sampler; 0: share the role's")


def split_role_cpus(a, python):
    """Apply --worker-cores / --procmon-*: start the sampler on its cpus, pin every thread of
    this process to the rest. Returns (worker cpus, record dict)."""
    from capcommon.exitcodes import ConfigError  # noqa: PLC0415

    from .procmon import start as procmon_start  # noqa: PLC0415
    cur = sorted(os.sched_getaffinity(0))
    try:
        rest, mon, wk = split_cpus(cur, a.worker_cores,
                                   a.procmon_cores if a.procmon_out else 0)
    except ValueError as e:
        raise ConfigError(str(e)) from e
    pm = None
    if a.procmon_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.procmon_out)), exist_ok=True)
        pm = procmon_start(os.getpid(), a.procmon_out, a.procmon_interval_ms, python,
                           cpus=mon or None)
    pin = pin_all_tasks(rest)
    return wk, {"role_cpus": cur, "own_cpus": rest, "procmon_cpus": mon, "worker_cpus": wk,
                "pin": pin, "procmon_pid": None if pm is None else pm.pid,
                "procmon_out": a.procmon_out, "procmon_interval_ms": a.procmon_interval_ms}

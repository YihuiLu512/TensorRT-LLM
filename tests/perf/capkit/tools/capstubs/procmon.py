# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-NX process-tree thread sampler: per-thread CPU of a product role's whole process tree.

    python -m capstubs.procmon --root-pid P --out FILE [--interval-ms 1000]

The capkit sidecar samples registered role processes only. The S-NX rank 0 worker (an MPI spawn
child) and the attached frontends of a multi-frontend trtllm-serve are unregistered descendants of
the product role, so their threads are invisible to it. This sampler runs as a separate process
(it takes no lock in the processes it reads) and, every interval, walks the descendants of
--root-pid (ppid chains over /proc) and writes one JSONL row per thread:

    [t_ns, pid, tid, comm, utime_ticks, stime_ticks, run_delay_ns, last_cpu]

t_ns is time.monotonic_ns() (the S-NX / caplaunch origin clock). The first time a process is seen
a {"proc": pid, ...} line gives its ppid, starttime, command line and Cpus_allowed_list; the first
time a thread is seen a {"task": tid, ...} line gives its Cpus_allowed_list. The sampler leaves
itself out. It ends when the root process is gone (or its start time changed), or on SIGTERM /
SIGINT, after one last sample. Its own CPU time is written every 30 samples and at the end
({"self": pid, "t_ns", "cpu_s", "cpus_allowed"}), so the readout can report the sampler's cost.
"""

import argparse
import json
import os
import signal
import sys
import time

SCHEMA = "capkit.snx_procmon/1"
COLUMNS = ["t_ns", "pid", "tid", "comm", "utime_ticks", "stime_ticks", "run_delay_ns",
           "last_cpu"]


def read_text(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace")
    except OSError:
        return None


def parse_stat(text):
    """(comm, fields after comm) of a /proc stat line; fields[0] is the state (field 3)."""
    if not text:
        return None
    lp, rp = text.find("("), text.rfind(")")
    if lp < 0 or rp < lp:
        return None
    return text[lp + 1:rp], text[rp + 2:].split()


def stat_of(path):
    """dict(comm, ppid, utime, stime, starttime, last_cpu) from a stat file, or None."""
    p = parse_stat(read_text(path))
    if p is None:
        return None
    comm, f = p
    try:
        # fields after comm: [0]=state [1]=ppid ... [11]=utime [12]=stime ... [19]=starttime
        # ... [36]=processor
        return {"comm": comm, "state": f[0], "ppid": int(f[1]), "utime": int(f[11]), "stime": int(f[12]),
                "starttime": int(f[19]), "last_cpu": int(f[36]) if len(f) > 36 else -1}
    except (IndexError, ValueError):
        return None


def run_delay(path):
    t = read_text(path)
    if not t:
        return -1
    try:
        return int(t.split()[1])
    except (IndexError, ValueError):
        return -1


def cpus_allowed(path):
    t = read_text(path)
    if not t:
        return None
    for line in t.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return None


def descendants(root, proc="/proc", skip=()):
    """[root] + every live descendant pid of root (ppid chains over one /proc pass)."""
    children = {}
    try:
        names = os.listdir(proc)
    except OSError:
        return []
    for n in names:
        if not n.isdigit():
            continue
        st = stat_of(os.path.join(proc, n, "stat"))
        if st is not None:
            children.setdefault(st["ppid"], []).append(int(n))
    out, todo = [], [root]
    while todo:
        pid = todo.pop()
        if pid in skip:
            continue
        out.append(pid)
        todo.extend(children.get(pid, ()))
    return out


class Sampler:
    def __init__(self, root, out, proc="/proc", clock=time.monotonic_ns):
        self.root, self.proc, self.clock = root, proc, clock
        self.f = out
        self.seen_procs, self.seen_tasks = {}, set()
        self.me = os.getpid()
        st = stat_of(os.path.join(proc, str(root), "stat"))
        self.root_start = st["starttime"] if st else None
        self.rows = 0

    def root_alive(self):
        """The root still runs (a zombie waiting for its parent's wait() counts as gone)."""
        st = stat_of(os.path.join(self.proc, str(self.root), "stat"))
        return st is not None and st["starttime"] == self.root_start and st["state"] != "Z"

    def _line(self, obj):
        self.f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def sample(self):
        t = self.clock()
        for pid in descendants(self.root, self.proc, skip={self.me}):
            pdir = os.path.join(self.proc, str(pid))
            pst = stat_of(os.path.join(pdir, "stat"))
            if pst is None or pst["state"] == "Z":
                continue
            key = (pid, pst["starttime"])
            if key not in self.seen_procs:
                cmd = (read_text(os.path.join(pdir, "cmdline")) or "").replace("\0", " ")
                self.seen_procs[key] = True
                self._line({"proc": pid, "ppid": pst["ppid"], "starttime": pst["starttime"],
                            "comm": pst["comm"], "cmdline": cmd.strip()[:400], "t_ns": t,
                            "cpus_allowed": cpus_allowed(os.path.join(pdir, "status"))})
            try:
                tids = os.listdir(os.path.join(pdir, "task"))
            except OSError:
                continue
            for tid in tids:
                tdir = os.path.join(pdir, "task", tid)
                st = stat_of(os.path.join(tdir, "stat"))
                if st is None:
                    continue
                tkey = (int(tid), st["starttime"])
                if tkey not in self.seen_tasks:
                    self.seen_tasks.add(tkey)
                    self._line({"task": int(tid), "pid": pid, "t_ns": t,
                                "cpus_allowed": cpus_allowed(os.path.join(tdir, "status"))})
                self._line([t, pid, int(tid), st["comm"], st["utime"], st["stime"],
                            run_delay(os.path.join(tdir, "schedstat")), st["last_cpu"]])
                self.rows += 1
        self.f.flush()


def self_line():
    """The sampler's own cost so far: {"self": pid, t_ns, cpu_s (user + system), cpus}."""
    t = os.times()
    return {"self": os.getpid(), "t_ns": time.monotonic_ns(), "cpu_s": t.user + t.system,
            "cpus_allowed": sorted(os.sched_getaffinity(0))}


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capstubs.procmon", description=__doc__.split("\n")[0])
    p.add_argument("--root-pid", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--interval-ms", type=int, default=1000)
    p.add_argument("--max-s", type=float, default=0.0, help="stop after this long (0: never)")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.interval_ms < 10:
        print("procmon: --interval-ms must be >= 10", file=sys.stderr)
        return 2
    stop = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.append(1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "x", encoding="utf-8") as f:
        f.write(json.dumps({"_header": True, "schema": SCHEMA, "root_pid": a.root_pid,
                            "pid": os.getpid(), "clock": "time.monotonic_ns",
                            "clk_tck": os.sysconf("SC_CLK_TCK"), "interval_ms": a.interval_ms,
                            "columns": COLUMNS}) + "\n")
        s = Sampler(a.root_pid, f)
        if s.root_start is None:
            print("procmon: root pid %d not found" % a.root_pid, file=sys.stderr)
            return 3
        t0 = time.monotonic()
        nxt = t0
        n = 0
        while True:
            s.sample()
            n += 1
            if n % 30 == 1:
                f.write(json.dumps(self_line()) + "\n")
            if stop or not s.root_alive() or (a.max_s and time.monotonic() - t0 >= a.max_s):
                break
            nxt += a.interval_ms / 1000.0
            dt = nxt - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                nxt = time.monotonic()
        end = self_line()
        end.update({"_end": True, "rows": s.rows, "samples": n,
                    "reason": "signal" if stop else "root_gone"})
        f.write(json.dumps(end) + "\n")
    return 0


def start(root_pid, out, interval_ms=1000, python=None, cpus=None):
    """Popen of a sampler for root_pid (it ends with the root); cpus: run it under
    `taskset -c <cpus>` (no preexec function: the caller may have threads)."""
    import subprocess  # noqa: PLC0415
    argv = [python or sys.executable, "-m", "capstubs.procmon", "--root-pid", str(root_pid),
            "--out", out, "--interval-ms", str(interval_ms)]
    if cpus:
        argv = ["taskset", "-c", ",".join(str(c) for c in cpus)] + argv
    return subprocess.Popen(argv)  # nosec B603


def load(path):
    """(header, rows, procs, tasks, selfs) of a procmon file; rows are lists in COLUMNS order,
    selfs the sampler's own-cost lines (the end line included)."""
    hdr, rows, procs, tasks, selfs = None, [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                rows.append(obj)
            elif obj.get("_header"):
                hdr = obj
            elif "proc" in obj:
                procs.append(obj)
            elif "task" in obj:
                tasks.append(obj)
            elif "self" in obj:
                selfs.append(obj)
    return hdr, rows, procs, tasks, selfs


if __name__ == "__main__":
    sys.exit(main())

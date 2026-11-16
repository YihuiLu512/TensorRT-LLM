# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readers for /proc and /sys used by the sidecar, the gate and the pin verifier.

Stdlib only. Every reader takes an optional root prefix (proc_root / sys_root) so tests can point
it at a synthetic tree. Readers return None (or raise ProcGone) instead of guessing when a file is
missing; callers record the reason.
"""

import os

PROC = "/proc"
SYS = "/sys"


class ProcGone(Exception):
    """The process or thread disappeared while being read."""


def read_text(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def try_read_text(path):
    """(text, None) or (None, reason)."""
    try:
        return read_text(path), None
    except OSError as e:
        return None, "%s: %s" % (type(e).__name__, e.strerror or e)


def parse_stat(raw):
    """Parse a /proc/<pid>[/task/<tid>]/stat line.

    Returns comm, state, ppid, utime, stime (clock ticks), nthreads, starttime (ticks),
    rss_pages, processor (last CPU the task ran on).
    """
    lp, rp = raw.find("("), raw.rfind(")")
    if lp < 0 or rp < 0:
        raise ValueError("malformed stat line")
    comm = raw[lp + 1:rp]
    rest = raw[rp + 2:].split()
    # rest[0] is field 3 (state); field n is rest[n - 3]
    return {"comm": comm, "state": rest[0], "ppid": int(rest[1]), "utime": int(rest[11]),
            "stime": int(rest[12]), "nthreads": int(rest[17]), "starttime": int(rest[19]),
            "rss_pages": int(rest[21]), "processor": int(rest[36])}


def read_stat(path):
    try:
        raw = read_text(path)
    except (FileNotFoundError, ProcessLookupError):
        raise ProcGone(path)
    return parse_stat(raw)


def parse_status(raw):
    """Selected fields of /proc/<pid>/status (text values kept verbatim)."""
    out = {}
    for line in raw.splitlines():
        k, _, v = line.partition(":")
        if k in ("Cpus_allowed_list", "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches",
                 "VmRSS", "Threads", "Name", "Tgid", "PPid"):
            out[k] = v.strip()
    return out


def read_status(path):
    try:
        raw = read_text(path)
    except (FileNotFoundError, ProcessLookupError):
        raise ProcGone(path)
    return parse_status(raw)


def parse_schedstat(raw):
    """/proc/<pid>/task/<tid>/schedstat: run_ns, run_delay_ns (run-queue wait), timeslices."""
    parts = raw.split()
    if len(parts) < 3:
        raise ValueError("malformed schedstat %r" % raw)
    return int(parts[0]), int(parts[1]), int(parts[2])


def list_tids(pid, proc_root=PROC):
    try:
        names = os.listdir(os.path.join(proc_root, str(pid), "task"))
    except (FileNotFoundError, ProcessLookupError, NotADirectoryError):
        raise ProcGone("pid %s" % pid)
    return sorted(int(n) for n in names if n.isdigit())


def task_dir(pid, tid, proc_root=PROC):
    return os.path.join(proc_root, str(pid), "task", str(tid))


def read_thread(pid, tid, proc_root=PROC):
    """One thread's sample: comm, utime, stime, last_cpu, run_delay_ns (None if no schedstat),
    voluntary / nonvoluntary ctx switches (None if absent), cpus_allowed (verbatim text)."""
    d = task_dir(pid, tid, proc_root)
    st = read_stat(os.path.join(d, "stat"))
    status = read_status(os.path.join(d, "status"))
    run_delay = None
    try:
        _run, run_delay, _slices = parse_schedstat(read_text(os.path.join(d, "schedstat")))
    except (OSError, ValueError):
        run_delay = None
    vol = status.get("voluntary_ctxt_switches")
    nvol = status.get("nonvoluntary_ctxt_switches")
    return {"comm": st["comm"], "utime": st["utime"], "stime": st["stime"],
            "last_cpu": st["processor"], "run_delay_ns": run_delay,
            "voluntary_ctxt": int(vol) if vol is not None else None,
            "nonvoluntary_ctxt": int(nvol) if nvol is not None else None,
            "cpus_allowed": status.get("Cpus_allowed_list")}


def starttime_ticks(pid, proc_root=PROC):
    try:
        return read_stat(os.path.join(proc_root, str(pid), "stat"))["starttime"]
    except (ProcGone, OSError, ValueError):
        return None


STAT_CPU_COLS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
BUSY_COLS = ("user", "nice", "system", "irq", "softirq", "steal")


def parse_proc_stat_cpus(raw):
    """{cpu_index: {col: ticks}} for every 'cpuN' line of /proc/stat (aggregate line skipped)."""
    out = {}
    for line in raw.splitlines():
        if not line.startswith("cpu") or line.startswith("cpu "):
            continue
        parts = line.split()
        name = parts[0][3:]
        if not name.isdigit():
            continue
        vals = [int(x) for x in parts[1:1 + len(STAT_CPU_COLS)]]
        vals += [0] * (len(STAT_CPU_COLS) - len(vals))
        out[int(name)] = dict(zip(STAT_CPU_COLS, vals))
    return out


def read_proc_stat_cpus(proc_root=PROC):
    return parse_proc_stat_cpus(read_text(os.path.join(proc_root, "stat")))


def busy_ticks(row):
    return sum(row[c] for c in BUSY_COLS)


def parse_cpu_list_text(text):
    """Kernel cpu-list syntax '0-3,8,10-11' -> sorted list of ints ('' -> [])."""
    out = set()
    text = (text or "").strip()
    if not text:
        return []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def online_cpus(sys_root=SYS):
    return parse_cpu_list_text(read_text(os.path.join(sys_root, "devices/system/cpu/online")))


class FreqReader:
    """Per-cpu current frequency in kHz.

    Source order: /sys/devices/system/cpu/cpuN/cpufreq/scaling_cur_freq, then the 'cpu MHz' lines
    of /proc/cpuinfo. The source actually used is reported (.source); unreadable -> None values.
    """

    def __init__(self, cpus, proc_root=PROC, sys_root=SYS):
        self.cpus = list(cpus)
        self.proc_root, self.sys_root = proc_root, sys_root
        self.source = None
        self.reason = None
        probe = self.cpus[0] if self.cpus else 0
        p = self._sys_path(probe)
        if os.path.exists(p):
            self.source = "sysfs:scaling_cur_freq"
        else:
            try:
                if self._cpuinfo_mhz():
                    self.source = "proc:cpuinfo_mhz"
                else:
                    self.reason = "no scaling_cur_freq and no 'cpu MHz' in cpuinfo"
            except OSError as e:
                self.reason = "no scaling_cur_freq; cpuinfo unreadable: %s" % e

    def _sys_path(self, cpu):
        return os.path.join(self.sys_root, "devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq"
                            % cpu)

    def _cpuinfo_mhz(self):
        out, cur = {}, None
        for line in read_text(os.path.join(self.proc_root, "cpuinfo")).splitlines():
            k, _, v = line.partition(":")
            k = k.strip()
            if k == "processor":
                cur = int(v)
            elif k == "cpu MHz" and cur is not None:
                out[cur] = int(round(float(v) * 1000))
        return out

    def read(self):
        if self.source == "sysfs:scaling_cur_freq":
            res = {}
            for c in self.cpus:
                try:
                    res[c] = int(read_text(self._sys_path(c)).strip())
                except (OSError, ValueError):
                    res[c] = None
            return res
        if self.source == "proc:cpuinfo_mhz":
            try:
                mhz = self._cpuinfo_mhz()
            except OSError:
                mhz = {}
            return {c: mhz.get(c) for c in self.cpus}
        return {c: None for c in self.cpus}


def read_rss_kb(pid, proc_root=PROC):
    st = read_stat(os.path.join(proc_root, str(pid), "stat"))
    return st["rss_pages"] * (os.sysconf("SC_PAGE_SIZE") // 1024), st


def numa_pages(pid, proc_root=PROC):
    """{node: pages} summed over the N<k>=<pages> fields of /proc/<pid>/numa_maps."""
    raw = read_text(os.path.join(proc_root, str(pid), "numa_maps"))
    out = {}
    for line in raw.splitlines():
        for tok in line.split():
            if tok.startswith("N") and "=" in tok:
                k, _, v = tok[1:].partition("=")
                if k.isdigit() and v.isdigit():
                    out[k] = out.get(k, 0) + int(v)
    return out


def cgroup_lines(pid, proc_root=PROC):
    """[(hierarchy_id, controllers, path)] from /proc/<pid>/cgroup."""
    raw = read_text(os.path.join(proc_root, str(pid), "cgroup"))
    out = []
    for line in raw.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            out.append((parts[0], parts[1], parts[2]))
    return out

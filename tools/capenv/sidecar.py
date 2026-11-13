# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capenv.sidecar: sidecar v4 (design 2.4 "sidecar v4"; INTERFACES.md 5.6, 11).

Rewritten from WT-21 sidecar_v3.py (copied, not modified in place). Differences from v3:

* processes are identified ONLY by the pid -> role registry procs/roles.json (pid +
  starttime_ticks); no command-line substring matching anywhere;
* every thread of every registered process, every tick: utime / stime split, schedstat run-queue
  wait, voluntary / nonvoluntary context switches, last-run CPU, Cpus_allowed_list (D1/D2/C9);
* per-core /proc/stat (all columns) and current frequency for the watched cores (D5, C13);
* thread inventory: tid appear / disappear events with comm, compared with the orchestrator's
  procs/threads/<role>.<pid>.json inventory when it exists (unknown threads are reported);
* capprobe / capkit snapshots of registered pids (including the canary's) decoded into the same
  tick timestamp as the /proc readings;
* GPU utilization + compute pids (positive control first) and numa_maps on a separate slow
  thread so nvidia-smi never delays the per-second /proc tick;
* times are monotonic_ns relative to the launch origin; ticks follow an absolute schedule
  origin + k * interval (missed ticks are counted, never back-filled).

Output files (all capkit JSONL in --out-dir, node-local): threads, cpus, procs, snapshots, gpu,
numa, logs, tids (.jsonl) and meta.json. Runs until SIGTERM / SIGINT; exit 0.
"""

import argparse
import os
import sys
import threading
import time

from capcommon import exitcodes
from capcommon.cli import add_common_args, common_from_args, parse_cpu_list
from capcommon.clock import now_ns, wall_iso
from capcommon.exitcodes import ConfigError
from capcommon.jsonio import JsonlWriter, read_json_stamped, write_json_stamped
from capcommon.lifecycle import StopController, pin_self, read_stop_file, write_ready_file
from capcommon.runtree import RunTree
from capcommon.snapshot import SnapshotError, list_snapshot_files, read_snapshot_file
from capcommon.versions import version_record

from capenv import gpu as gpu_mod
from capenv import procfs
from capenv.roles import RolesWatcher

S_THREADS = "capkit.sidecar.threads/1"
S_CPUS = "capkit.sidecar.cpus/1"
S_PROCS = "capkit.sidecar.procs/1"
S_SNAPS = "capkit.sidecar.snapshots/1"
S_GPU = "capkit.sidecar.gpu/1"
S_NUMA = "capkit.sidecar.numa/1"
S_LOGS = "capkit.sidecar.logs/1"
S_TIDS = "capkit.sidecar.tids/1"
S_META = "capkit.sidecar.meta/1"
PIN_SCHEMA = "capkit.pin_table/1"

COLS = {
    "threads": ("t_rel_ns", "pid", "tid", "comm", "utime_ticks", "stime_ticks", "run_delay_ns",
                "voluntary_ctxt", "nonvoluntary_ctxt", "last_cpu", "cpus_allowed"),
    "cpus": ("t_rel_ns", "cpu", "user", "nice", "system", "idle", "iowait", "irq", "softirq",
             "steal", "freq_khz"),
    "procs": ("t_rel_ns", "pid", "role", "worker", "rss_kb", "nthreads", "cpus_allowed"),
    "snapshots": ("t_rel_ns", "pid", "seq", "snap_t_rel_ns", "counters", "hists"),
    "gpu": ("t_rel_ns", "gpu_index", "util_pct", "mem_used_mib", "compute_pids", "query_rc"),
    "numa": ("t_rel_ns", "pid", "pages_by_node"),
    "logs": ("t_rel_ns", "path", "bytes"),
    # not in INTERFACES 5.6 (added file): thread inventory events
    "tids": ("t_rel_ns", "pid", "role", "tid", "event", "comm", "in_inventory",
             "identified_as"),
}
SCHEMAS = {"threads": S_THREADS, "cpus": S_CPUS, "procs": S_PROCS, "snapshots": S_SNAPS,
           "gpu": S_GPU, "numa": S_NUMA, "logs": S_LOGS, "tids": S_TIDS}


class Writer(JsonlWriter):
    """capkit JSONL writer flushed once per tick (not per row), with optional extra header keys
    (gpu.jsonl carries gpu_count in its header, INTERFACES 5.6). The header keys of
    JsonlWriter are kept; extras may not override them."""

    def __init__(self, path, schema, identity, columns, header_extra=None):
        if not header_extra:
            super().__init__(path, schema, identity, columns, flush_every=1 << 30)
            return
        # same file format as JsonlWriter, header extended with the extra keys
        import json  # noqa: PLC0415
        from capcommon.jsonio import _identity_dict  # noqa: PLC0415
        self.path = path
        self.columns = list(columns)
        self._n = len(columns)
        self._flush_every = 1 << 30
        self._pending = 0
        self.rows = 0
        header = {"_header": True, "schema": schema, "identity": _identity_dict(identity),
                  "columns": self.columns, "written_wall": wall_iso()}
        clash = set(header) & set(header_extra)
        if clash:
            raise ConfigError("header_extra overrides %s" % sorted(clash))
        header.update(header_extra)
        self._f = open(path, "w", encoding="utf-8")
        self._f.write(json.dumps(header, sort_keys=True, allow_nan=False) + "\n")
        self._f.flush()

    def flush(self):
        if self._f is not None:
            self._f.flush()
            self._pending = 0



def _gkey(p):
    return (p["pid"], p.get("starttime_ticks"))


class Inventory:
    """The orchestrator's thread inventory procs/threads/<role>.<pid>.json, if it exists."""

    def __init__(self, run_dir):
        self.tree = RunTree(run_dir) if run_dir else None
        self.cache = {}  # pid -> {tid: identified_as} once the file exists

    def lookup(self, role, pid, tid):
        """(in_inventory, identified_as); (None, None) while no inventory file exists."""
        inv = self.cache.get(pid)
        if inv is None and self.tree is not None:
            path = self.tree.threads_file(role, pid)
            if os.path.exists(path):
                try:
                    _ident, doc = read_json_stamped(path, "capkit.threads/1")
                    inv = {t["tid"]: t.get("identified_as") for t in doc.get("threads", [])}
                except (ConfigError, ValueError, OSError, KeyError, TypeError):
                    inv = None
                if inv is not None:
                    self.cache[pid] = inv
        if inv is None:
            return None, None
        if tid in inv:
            return True, inv[tid]
        return False, None


class Sidecar:
    def __init__(self, common, out_dir, shm_dir, roles_file, interval_ns, cpus_watch,
                 gpu_every_ns, numa_every_ns, logs_dir, gpu_timeout_s, proc_root=procfs.PROC,
                 sys_root=procfs.SYS, nvidia_smi=None):
        self.common = common
        self.ident = common.identity
        self.origin = common.origin_ns
        self.out_dir = out_dir
        self.shm_dir = shm_dir
        self.interval_ns = interval_ns
        self.cpus_watch = cpus_watch
        self.gpu_every_ns = gpu_every_ns
        self.numa_every_ns = numa_every_ns
        self.logs_dir = logs_dir
        self.gpu_timeout_s = gpu_timeout_s
        self.nvidia_smi = nvidia_smi  # None: GPU positive control not run ("path not given")
        self.proc_root, self.sys_root = proc_root, sys_root
        self.watcher = RolesWatcher(roles_file)
        self.inventory = Inventory(common.run_dir)
        self.freq = procfs.FreqReader(cpus_watch, proc_root, sys_root)
        self.lock = threading.Lock()  # guards meta counters shared with the slow thread
        self.tids = {}  # pid -> {tid: comm}
        self.inv_checked = set()  # pids whose live threads were checked against the inventory
        self.gone = {}  # (pid, registered starttime) -> reason; a re-registration is new
        self.stats = {"ticks": 0, "missed_ticks": 0, "max_lateness_ns": 0, "sum_tick_ns": 0,
                      "max_tick_ns": 0, "thread_read_errors": 0, "snapshot_decode_errors": 0,
                      "snapshot_identity_mismatch": 0, "schedstat_missing_rows": 0,
                      "ctxt_missing_rows": 0, "freq_missing_rows": 0, "gpu_samples": 0,
                      "gpu_failed_samples": 0, "gpu_util_unreadable_rows": 0, "numa_samples": 0, "numa_unreadable": 0}
        self.unregistered_snapshot_pids = set()
        self.roles_generations = []
        self.gpu_pc = None
        self.w = {}
        self.stop_evt = threading.Event()
        self.slow_thread = None
        self.started_wall = wall_iso()

    # -- files ------------------------------------------------------------------------------
    def open_writers(self):
        os.makedirs(self.out_dir, exist_ok=True)
        for name, cols in COLS.items():
            if name == "gpu" and self.gpu_every_ns == 0:
                continue
            if name == "numa" and self.numa_every_ns == 0:
                continue
            extra = None
            if name == "gpu":
                ok, count, rec = gpu_mod.positive_control(self.gpu_timeout_s,
                                                          nvsmi=self.nvidia_smi)
                self.gpu_pc = {"ok": ok, "gpu_count": count, "rc": rec.get("rc"),
                               "reason": rec.get("reason"), "cmd": rec.get("cmd")}
                extra = {"gpu_count": count, "positive_control_ok": ok}
            self.w[name] = Writer(os.path.join(self.out_dir, name + ".jsonl"), SCHEMAS[name],
                                  self.ident, cols, header_extra=extra)

    def close_writers(self):
        for w in self.w.values():
            w.close()

    def rel(self, t):
        return t - self.origin

    # -- one /proc tick ---------------------------------------------------------------------
    def sample_proc(self, t_rel, reg):
        wt, wp, wtid = self.w["threads"], self.w["procs"], self.w["tids"]
        for p in reg.procs:
            pid, role = p["pid"], p["role"]
            if _gkey(p) in self.gone:
                continue
            try:
                rss_kb, st = procfs.read_rss_kb(pid, self.proc_root)
                status = procfs.read_status(os.path.join(self.proc_root, str(pid), "status"))
            except (procfs.ProcGone, OSError, ValueError) as e:
                self._mark_gone(t_rel, p, "gone: %s" % type(e).__name__)
                continue
            reg_start = p.get("starttime_ticks")
            if reg_start is not None and st["starttime"] != reg_start:
                self._mark_gone(t_rel, p, "pid reused (starttime %s != registered %s)"
                                % (st["starttime"], reg_start))
                continue
            wp.write_row((t_rel, pid, role, p.get("worker"), rss_kb, st["nthreads"],
                          status.get("Cpus_allowed_list")))
            try:
                tids = procfs.list_tids(pid, self.proc_root)
            except procfs.ProcGone:
                self._mark_gone(t_rel, p, "gone while listing tasks")
                continue
            known = self.tids.setdefault(pid, {})
            seen = set()
            for tid in tids:
                try:
                    t = procfs.read_thread(pid, tid, self.proc_root)
                except (procfs.ProcGone, OSError, ValueError):
                    self.stats["thread_read_errors"] += 1
                    continue
                seen.add(tid)
                if t["run_delay_ns"] is None:
                    self.stats["schedstat_missing_rows"] += 1
                if t["voluntary_ctxt"] is None:
                    self.stats["ctxt_missing_rows"] += 1
                wt.write_row((t_rel, pid, tid, t["comm"], t["utime"], t["stime"],
                              t["run_delay_ns"], t["voluntary_ctxt"], t["nonvoluntary_ctxt"],
                              t["last_cpu"], t["cpus_allowed"]))
                if tid not in known or known[tid] != t["comm"]:
                    inv, ident_as = self.inventory.lookup(role, pid, tid)
                    wtid.write_row((t_rel, pid, role, tid,
                                    "appear" if tid not in known else "rename", t["comm"], inv,
                                    ident_as))
                    known[tid] = t["comm"]
            for tid in sorted(set(known) - seen):
                wtid.write_row((t_rel, pid, role, tid, "disappear", known.pop(tid), None, None))
            if pid not in self.inv_checked and self.inventory.lookup(role, pid, -1)[0] is not None:
                # the orchestrator's inventory just became readable: check every live thread
                # against it once (appear events written before it existed carry null)
                self.inv_checked.add(pid)
                for tid in sorted(known):
                    inv, ident_as = self.inventory.lookup(role, pid, tid)
                    wtid.write_row((t_rel, pid, role, tid, "inventory", known[tid], inv,
                                    ident_as))

    def _mark_gone(self, t_rel, p, why):
        pid, role = p["pid"], p["role"]
        self.gone[_gkey(p)] = {"t_rel_ns": t_rel, "pid": pid, "role": role, "why": why,
                               "starttime_registered": p.get("starttime_ticks")}
        for tid, comm in sorted(self.tids.pop(pid, {}).items()):
            self.w["tids"].write_row((t_rel, pid, role, tid, "disappear", comm, None, None))

    def sample_cpus(self, t_rel):
        stat = procfs.read_proc_stat_cpus(self.proc_root)
        freq = self.freq.read()
        w = self.w["cpus"]
        for c in self.cpus_watch:
            row = stat.get(c)
            if row is None:  # watched cpu went offline: write a row of nulls, do not skip
                w.write_row((t_rel, c) + (None,) * 8 + (freq.get(c),))
                continue
            f = freq.get(c)
            if f is None:
                self.stats["freq_missing_rows"] += 1
            w.write_row((t_rel, c, row["user"], row["nice"], row["system"], row["idle"],
                         row["iowait"], row["irq"], row["softirq"], row["steal"], f))

    def sample_snapshots(self, t_rel, reg):
        files = list_snapshot_files(self.shm_dir)
        regmap = reg.by_pid()
        for pid in files:
            if pid not in regmap:
                self.unregistered_snapshot_pids.add(pid)
        w = self.w["snapshots"]
        for pid, p in regmap.items():
            path = files.get(pid)
            if path is None or _gkey(p) in self.gone:
                continue
            try:
                snap = read_snapshot_file(path)
            except (SnapshotError, OSError, ValueError):
                self.stats["snapshot_decode_errors"] += 1
                continue
            m = snap.meta
            reg_start = p.get("starttime_ticks")
            if m.get("pid") != pid or (reg_start is not None and m.get("starttime_ticks")
                                       is not None and m["starttime_ticks"] != reg_start):
                self.stats["snapshot_identity_mismatch"] += 1
                continue
            counters, hists = {}, {}
            for s in m.get("slots", []):
                v = snap.values.get(s["name"])
                if s.get("kind") == "hist":
                    hists[s["name"]] = v
                else:
                    counters[s["name"]] = v
            w.write_row((t_rel, pid, snap.seq, self.rel(snap.t_mono_ns), counters, hists))

    def sample_logs(self, t_rel):
        if not self.logs_dir:
            return
        try:
            entries = sorted(os.scandir(self.logs_dir), key=lambda e: e.name)
        except FileNotFoundError:
            return
        w = self.w["logs"]
        for e in entries:
            try:
                if e.is_file(follow_symlinks=False):
                    w.write_row((t_rel, e.name, e.stat(follow_symlinks=False).st_size))
            except FileNotFoundError:
                continue

    def tick(self, t_abs):
        t_rel = self.rel(t_abs)
        reg, changed = self.watcher.poll()
        if changed and reg is not None:
            self.roles_generations.append({"t_rel_ns": t_rel, "generation": reg.generation,
                                           "procs": len(reg.procs),
                                           "unattributed": reg.unattributed})
        self.sample_cpus(t_rel)
        if reg is not None:
            self.sample_proc(t_rel, reg)
            self.sample_snapshots(t_rel, reg)
        self.sample_logs(t_rel)
        for name in ("threads", "cpus", "procs", "snapshots", "logs", "tids"):
            self.w[name].flush()

    # -- slow thread: GPU and numa_maps -----------------------------------------------------
    def _slow_loop(self):
        nxt_gpu = self._first_tick(self.gpu_every_ns) if self.gpu_every_ns else None
        nxt_numa = self._first_tick(self.numa_every_ns) if self.numa_every_ns else None
        while not self.stop_evt.is_set():
            due = min(x for x in (nxt_gpu, nxt_numa) if x is not None)
            delay = (due - now_ns()) / 1e9
            if delay > 0 and self.stop_evt.wait(delay):
                break
            t = now_ns()
            if nxt_gpu is not None and t >= nxt_gpu:
                self._gpu_once(self.rel(t))
                nxt_gpu = self._next_after(nxt_gpu, self.gpu_every_ns)
            if nxt_numa is not None and t >= nxt_numa:
                self._numa_once(self.rel(t))
                nxt_numa = self._next_after(nxt_numa, self.numa_every_ns)

    def _gpu_once(self, t_rel):
        rc, rows = gpu_mod.sample(self.gpu_timeout_s, nvsmi=self.nvidia_smi)
        w = self.w["gpu"]
        with self.lock:
            self.stats["gpu_samples"] += 1
            if rc != 0:
                self.stats["gpu_failed_samples"] += 1
            self.stats["gpu_util_unreadable_rows"] += sum(
                1 for r in rows if r["util_pct"] is None and r["gpu_index"] is not None)
        if rc != 0:
            n = (self.gpu_pc or {}).get("gpu_count") or 0
            for i in (range(n) if n else [None]):
                w.write_row((t_rel, i, None, None, None, rc))
        else:
            for r in rows:
                w.write_row((t_rel, r["gpu_index"], r["util_pct"], r["mem_used_mib"],
                             r["compute_pids"], 0))
        w.flush()

    def _numa_once(self, t_rel):
        reg = self.watcher.current
        if reg is None:
            return
        w = self.w["numa"]
        for p in reg.procs:
            if _gkey(p) in self.gone:
                continue
            try:
                pages = procfs.numa_pages(p["pid"], self.proc_root)
            except OSError as e:
                pages = {"unreadable": "%s: %s" % (type(e).__name__, e.strerror or e)}
                with self.lock:
                    self.stats["numa_unreadable"] += 1
            w.write_row((t_rel, p["pid"], pages))
        with self.lock:
            self.stats["numa_samples"] += 1
        w.flush()

    def _first_tick(self, every_ns):
        """First origin-aligned tick at or after now."""
        t = now_ns()
        k = -((self.origin - t) // every_ns)  # ceil((t - origin) / every)
        return self.origin + k * every_ns

    def _next_after(self, prev, every_ns):
        nxt = prev + every_ns
        t = now_ns()
        if nxt <= t:  # fell behind: skip to the next future tick
            nxt = self._first_tick(every_ns)
            if nxt <= t:
                nxt += every_ns
        return nxt

    # -- main loop --------------------------------------------------------------------------
    def run(self, stop, on_first_tick=None):
        self.open_writers()
        self.write_meta(final=False)
        if self.gpu_every_ns or self.numa_every_ns:
            self.slow_thread = threading.Thread(target=self._slow_loop, name="capenv-sidecar-slow",
                                                daemon=True)
            self.slow_thread.start()
        target = self._first_tick(self.interval_ns)
        first = True
        try:
            while not stop.requested:
                delay = (target - now_ns()) / 1e9
                if delay > 0 and stop.wait(delay):
                    break
                t0 = now_ns()
                late = t0 - target
                self.tick(t0)
                t1 = now_ns()
                st = self.stats
                st["ticks"] += 1
                st["max_lateness_ns"] = max(st["max_lateness_ns"], late)
                st["sum_tick_ns"] += t1 - t0
                st["max_tick_ns"] = max(st["max_tick_ns"], t1 - t0)
                if first:
                    first = False
                    if on_first_tick is not None:
                        on_first_tick()
                nxt = target + self.interval_ns
                if nxt <= t1:
                    skipped = (t1 - target) // self.interval_ns
                    st["missed_ticks"] += skipped
                    nxt = target + (skipped + 1) * self.interval_ns
                target = nxt
        finally:
            self.stop_evt.set()
            if self.slow_thread is not None:
                self.slow_thread.join(timeout=max(5.0, self.gpu_timeout_s * 2 + 1))
            self.close_writers()
            self.write_meta(final=True)

    def declared_threads(self):
        """{native_id: name} of every live Python thread other than the main thread (the slow
        GPU/numa_maps thread when it runs), for the ready file's info.threads: the orchestrator
        identifies the sidecar's threads only from what the sidecar declares (INTERFACES 5.3)."""
        main = threading.main_thread()
        return {str(t.native_id): t.name for t in threading.enumerate()
                if t is not main and t.native_id is not None}

    def write_meta(self, final):
        with self.lock:
            stats = dict(self.stats)
        payload = {
            "clk_tck": os.sysconf("SC_CLK_TCK"),
            "page_kb": os.sysconf("SC_PAGE_SIZE") // 1024,
            "interval_ns": self.interval_ns,
            "origin_ns": self.origin,
            "cpus_watch": self.cpus_watch,
            "gpu_every_ns": self.gpu_every_ns,
            "numa_every_ns": self.numa_every_ns,
            "gpu_positive_control": self.gpu_pc,
            "logs_dir": self.logs_dir,
            "shm_dir": self.shm_dir,
            "roles_file": self.watcher.path,
            "freq_source": self.freq.source,
            "unreadable": self._unreadable(stats),
            "stats": stats,
            "pids_gone": [v for _k, v in sorted(self.gone.items(), key=lambda kv: str(kv[0]))],
            "unregistered_snapshot_pids": sorted(self.unregistered_snapshot_pids),
            "roles_generations": self.roles_generations,
            "roles_errors": list(self.watcher.errors),
            "rows": {k: w.rows for k, w in self.w.items()},
            "own_cpu_ns": time.process_time_ns(),
            "started_wall": self.started_wall,
            "final": final,
            "stop": read_stop_file(RunTree(self.common.run_dir).stop_json) if final else None,
        }
        write_json_stamped(os.path.join(self.out_dir, "meta.json"), S_META, self.ident, payload)

    def _unreadable(self, stats):
        out = {}
        if self.freq.source is None:
            out["freq_khz"] = self.freq.reason
        if stats["schedstat_missing_rows"]:
            out["schedstat"] = "%d thread rows without schedstat (kernel without "\
                "CONFIG_SCHED_INFO / schedstats?)" % stats["schedstat_missing_rows"]
        if stats["ctxt_missing_rows"]:
            out["ctxt_switches"] = "%d thread rows without *_ctxt_switches in status" \
                % stats["ctxt_missing_rows"]
        if self.gpu_pc is not None and not self.gpu_pc["ok"]:
            out["gpu"] = "positive control failed: %s" % (self.gpu_pc.get("reason"),)
        if stats["gpu_util_unreadable_rows"]:
            out["gpu_util"] = "%d gpu rows with util_pct null ([N/A]): not idle readings" \
                % stats["gpu_util_unreadable_rows"]
        return out


def resolve_cpus_watch(text, sys_root=procfs.SYS):
    if text == "all":
        return procfs.online_cpus(sys_root)
    return parse_cpu_list(text)


def check_watch_covers_pins(cpus_watch, pin_table_path):
    _ident, doc = read_json_stamped(pin_table_path, PIN_SCHEMA)
    need = sorted({c for e in doc["roles"].values() for c in e["cpus"]})
    missing = sorted(set(need) - set(cpus_watch))
    if missing:
        raise ConfigError("--cpus-watch does not cover pin table cpus %s" % missing)


def build_parser():
    ap = argparse.ArgumentParser(prog="capenv.sidecar", description=__doc__.split("\n\n")[0])
    add_common_args(ap, cpus=True, inject=False, stop=True)
    g = ap.add_argument_group("sidecar (INTERFACES.md 11)")
    g.add_argument("--shm-dir", required=True, help="this launch's snapshot directory")
    g.add_argument("--origin-ns", type=int, required=True)
    g.add_argument("--roles-file", required=True, help="procs/roles.json")
    g.add_argument("--interval-ms", type=int, required=True)
    g.add_argument("--out-dir", required=True, help="raw/sidecar of the node-local tree")
    g.add_argument("--cpus-watch", required=True, help="'all' or a cpu list")
    g.add_argument("--gpu-every-s", type=float, required=True, help="0 disables gpu.jsonl")
    g.add_argument("--numa-every-s", type=float, required=True, help="0 disables numa.jsonl")
    g.add_argument("--logs-dir", required=True, help="raw/logs of the node-local tree")
    o = ap.add_argument_group("optional")
    o.add_argument("--ready-file", help="write a capkit ready file after the first tick")
    o.add_argument("--pin-table", help="exit 3 unless --cpus-watch covers every pinned cpu")
    o.add_argument("--gpu-timeout-s", type=float, default=None,
                   help="nvidia-smi timeout; default = --gpu-every-s")
    o.add_argument("--nvidia-smi", default=None,
                   help="nvidia-smi executable; absent with --gpu-every-s > 0 -> the GPU positive "
                        "control is not run (gpu.jsonl header positive_control_ok false)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        common = common_from_args(args)
        common.origin_ns = args.origin_ns
        if args.interval_ms <= 0:
            raise ConfigError("--interval-ms must be > 0")
        if args.gpu_every_s < 0 or args.numa_every_s < 0:
            raise ConfigError("--gpu-every-s / --numa-every-s must be >= 0")
        cpus_watch = resolve_cpus_watch(args.cpus_watch)
        if not cpus_watch:
            raise ConfigError("--cpus-watch resolved to no cpus")
        if args.pin_table:
            check_watch_covers_pins(cpus_watch, args.pin_table)
        affinity = pin_self(common.cpus)
        gpu_every_ns = int(args.gpu_every_s * 1e9)
        sc = Sidecar(common, args.out_dir, args.shm_dir, args.roles_file,
                     args.interval_ms * 1_000_000, cpus_watch, gpu_every_ns,
                     int(args.numa_every_s * 1e9), args.logs_dir,
                     args.gpu_timeout_s if args.gpu_timeout_s is not None
                     else max(1.0, args.gpu_every_s), nvidia_smi=args.nvidia_smi)
    except ConfigError as e:
        print("capenv.sidecar: %s" % e, file=sys.stderr)
        return exitcodes.CONFIG
    stop = StopController()

    def ready():
        if args.ready_file:
            write_ready_file(args.ready_file, common.identity,
                             version_record(include_grpc=False), affinity=affinity,
                             info={"out_dir": args.out_dir, "cpus_watch": cpus_watch,
                                   "threads": sc.declared_threads()})

    sc.run(stop, on_first_tick=ready)
    return exitcodes.OK


if __name__ == "__main__":
    sys.exit(main())

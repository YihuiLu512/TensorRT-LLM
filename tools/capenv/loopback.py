# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capenv.loopback: D14, where does loopback-TCP kernel time get accounted (design 2.5 D14).

A sender process pinned to --sender-cpu pushes --msg-bytes writes over a loopback TCP
connection as fast as it can for --duration-s; a receiver pinned to --receiver-cpu drains it.
The coordinator (this process, which should run on a third core) reads, right before "go" and
right after both peers report done (both stay alive until told to quit):
  * utime / stime of the sender's and the receiver's single thread,
  * every /proc/stat column of both cores.
It reports, per core: busy ticks, softirq / irq ticks, the pinned thread's utime + stime, and
the remainder busy - (utime + stime) that belongs to no thread of the pair. No interpretation is
applied; pilot.json decides how the "close to one core" thresholds use it (design 2.6 I1).

    python -m capenv.loopback <identity> --sender-cpu A --receiver-cpu B --duration-s S
        --msg-bytes N --out env/loopback_softirq.json
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time

from capcommon import exitcodes
from capcommon.cli import add_identity_args
from capcommon.exitcodes import ConfigError
from capcommon.identity import Identity
from capcommon.jsonio import write_json_stamped

from . import procfs

SCHEMA = "capkit.loopback_softirq/1"


def _peer(argv):
    ap = argparse.ArgumentParser(prog="capenv.loopback _peer")
    ap.add_argument("kind", choices=("sender", "receiver"))
    ap.add_argument("--cpu", type=int, required=True)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--duration-s", type=float, default=0.0)
    ap.add_argument("--msg-bytes", type=int, default=0)
    a = ap.parse_args(argv)
    os.sched_setaffinity(0, {a.cpu})
    out = sys.stdout
    if a.kind == "receiver":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        out.write(json.dumps({"port": srv.getsockname()[1], "pid": os.getpid()}) + "\n")
        out.flush()
        conn, _ = srv.accept()
        buf = bytearray(1 << 20)
        mv = memoryview(buf)
        total = 0
        while True:
            n = conn.recv_into(mv)
            if n == 0:
                break
            total += n
        out.write(json.dumps({"done": True, "bytes": total}) + "\n")
        out.flush()
    else:
        s = socket.create_connection(("127.0.0.1", a.port))
        payload = b"x" * a.msg_bytes
        out.write(json.dumps({"ready": True, "pid": os.getpid()}) + "\n")
        out.flush()
        sys.stdin.readline()  # "go"
        end = time.monotonic_ns() + int(a.duration_s * 1e9)
        total = 0
        sendall = s.sendall
        while time.monotonic_ns() < end:
            sendall(payload)
            total += len(payload)
        s.close()
        out.write(json.dumps({"done": True, "bytes": total}) + "\n")
        out.flush()
    sys.stdin.readline()  # "quit": stay alive so the coordinator can read our counters
    return 0


def _thread(pid):
    st = procfs.read_stat("/proc/%d/stat" % pid)  # single-threaded peer: process == thread
    return {"utime": st["utime"], "stime": st["stime"], "last_cpu": st["processor"],
            "nthreads": st["nthreads"]}


def measure(sender_cpu, receiver_cpu, duration_s, msg_bytes, env=None):
    if sender_cpu == receiver_cpu:
        raise ConfigError("sender and receiver must be on different cpus")
    base = [sys.executable, "-m", "capenv.loopback", "_peer"]
    rx = subprocess.Popen(base + ["receiver", "--cpu", str(receiver_cpu)],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    tx = None
    try:
        hello = json.loads(rx.stdout.readline())
        tx = subprocess.Popen(base + ["sender", "--cpu", str(sender_cpu), "--port",
                                      str(hello["port"]), "--duration-s", str(duration_s),
                                      "--msg-bytes", str(msg_bytes)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        json.loads(tx.stdout.readline())  # ready
        cores = (sender_cpu, receiver_cpu)
        s0 = procfs.read_proc_stat_cpus()
        a0, b0 = _thread(tx.pid), _thread(rx.pid)
        t0 = time.monotonic_ns()
        tx.stdin.write("go\n")
        tx.stdin.flush()
        tx_done = json.loads(tx.stdout.readline())
        rx_done = json.loads(rx.stdout.readline())
        s1 = procfs.read_proc_stat_cpus()
        a1, b1 = _thread(tx.pid), _thread(rx.pid)
        t1 = time.monotonic_ns()
    finally:
        for p in (tx, rx):
            if p is None:
                continue
            try:
                p.stdin.write("quit\n")
                p.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        for p in (tx, rx):
            if p is None:
                continue
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=10)
    clk = os.sysconf("SC_CLK_TCK")
    window_ticks = (t1 - t0) * clk / 1e9
    sides = {}
    for name, cpu, th0, th1 in (("sender", cores[0], a0, a1), ("receiver", cores[1], b0, b1)):
        d = {k: s1[cpu][k] - s0[cpu][k] for k in procfs.STAT_CPU_COLS}
        busy = sum(d[k] for k in procfs.BUSY_COLS)
        u, s = th1["utime"] - th0["utime"], th1["stime"] - th0["stime"]
        sides[name] = {"cpu": cpu, "core_ticks": d, "core_busy_ticks": busy,
                       "thread_utime_ticks": u, "thread_stime_ticks": s,
                       "thread_cpu_frac": (u + s) / window_ticks if window_ticks else None,
                       "core_busy_minus_thread_ticks": busy - (u + s),
                       "thread_last_cpu": th1["last_cpu"]}
    return {"clk_tck": clk, "window_ns": t1 - t0, "window_ticks": window_ticks,
            "duration_s": duration_s, "msg_bytes": msg_bytes,
            "bytes_sent": tx_done["bytes"], "bytes_received": rx_done["bytes"],
            "sides": sides,
            "how_to_read": "core_busy_minus_thread_ticks > 0 on a core = busy time (mostly "
                           "softirq/irq columns) not charged to the pinned thread"}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_peer":
        return _peer(argv[1:])
    ap = argparse.ArgumentParser(prog="capenv.loopback", description=__doc__.split("\n\n")[0])
    add_identity_args(ap)
    ap.add_argument("--sender-cpu", type=int, required=True)
    ap.add_argument("--receiver-cpu", type=int, required=True)
    ap.add_argument("--duration-s", type=float, required=True)
    ap.add_argument("--msg-bytes", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        if args.duration_s <= 0 or args.msg_bytes <= 0:
            raise ConfigError("--duration-s and --msg-bytes must be > 0")
        ident = Identity(experiment=args.experiment, arm=args.arm, launch_seq=args.launch_seq,
                         job_id=args.job_id, restart_count=args.restart_count,
                         register_sha=args.register_sha, role=args.role, worker=args.worker,
                         pid=os.getpid())
        res = measure(args.sender_cpu, args.receiver_cpu, args.duration_s, args.msg_bytes)
        res["coordinator_affinity"] = sorted(os.sched_getaffinity(0))
        write_json_stamped(args.out, SCHEMA, ident, res)
    except ConfigError as e:
        print("capenv.loopback: %s" % e, file=sys.stderr)
        return exitcodes.CONFIG
    print(json.dumps({"out": args.out, "bytes": res["bytes_received"]}))
    return exitcodes.OK


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capenv.clockcheck: is CLOCK_MONOTONIC shared between two processes (INTERFACES.md 5.5, 12 U-4).

Two processes (possibly in different containers / namespaces) exchange monotonic_ns stamps over
loopback TCP. The client sends its send time; the server answers with its own monotonic_ns; the
client records its receive time. If both read the same clock, every one-way difference
(peer stamp - send time) lies inside [0, rtt]. Also records back-to-back monotonic_ns deltas
(clock read noise) and both processes' /proc/self/ns/{pid,mnt} and cgroup (namespace evidence).

    server: python -m capenv.clockcheck <common> --peer-mode server --port-file P
              --round-trips N --back-to-back M --out unused
    client: python -m capenv.clockcheck <common> --peer-mode client --port-file P
              --round-trips N --back-to-back M --out env/clock_check.json
The client writes the result; the server exits after the client disconnects.
"""

import argparse
import json
import os
import socket
import struct
import sys
import time

from capcommon import exitcodes
from capcommon.cli import add_common_args, common_from_args
from capcommon.exitcodes import ConfigError
from capcommon.jsonio import write_json_stamped
from capcommon.lifecycle import pin_self, wait_port_file, write_port_file

SCHEMA = "capkit.clock_check/1"
_Q = struct.Struct("<q")


def ns_info():
    out = {}
    for k in ("pid", "mnt"):
        try:
            out[k] = os.readlink("/proc/self/ns/%s" % k)
        except OSError as e:
            out[k] = {"unreadable": str(e)}
    try:
        with open("/proc/self/cgroup") as f:
            out["cgroup"] = f.read()
    except OSError as e:
        out["cgroup"] = {"unreadable": str(e)}
    return out


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def summary(values):
    if not values:
        return {"n": 0}
    v = sorted(values)
    def q(p):
        return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]
    return {"n": len(v), "min": v[0], "p50": q(0.5), "p99": q(0.99), "max": v[-1]}


def server(common, port_file):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    write_port_file(port_file, common.identity, "127.0.0.1", srv.getsockname()[1])
    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.sendall((json.dumps({"ns": ns_info(), "pid": os.getpid()}) + "\n").encode())
    now = time.monotonic_ns
    while True:
        b = _recv_exact(conn, 8)
        if b is None:
            break
        conn.sendall(_Q.pack(now()))
    conn.close()
    srv.close()


def client(common, port_file, round_trips, back_to_back, timeout_s=60.0):
    host, port, _doc = wait_port_file(port_file, timeout_s)
    s = socket.create_connection((host, port))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    line = b""
    while not line.endswith(b"\n"):
        c = s.recv(1)
        if not c:
            raise ConfigError("server closed before hello")
        line += c
    peer = json.loads(line)
    now = time.monotonic_ns
    one_way, rtt, outside = [], [], 0
    for _ in range(round_trips):
        t_send = now()
        s.sendall(_Q.pack(t_send))
        b = _recv_exact(s, 8)
        t_recv = now()
        t_peer = _Q.unpack(b)[0]
        d = t_peer - t_send
        one_way.append(d)
        rtt.append(t_recv - t_send)
        if d < 0 or d > t_recv - t_send:
            outside += 1
    s.close()
    b2b = []
    prev = now()
    for _ in range(back_to_back):
        t = now()
        b2b.append(t - prev)
        prev = t
    return {"round_trips": round_trips, "one_way_ns": summary(one_way), "rtt_ns": summary(rtt),
            "one_way_outside_0_rtt": outside, "back_to_back_ns": summary(b2b),
            "self": {"pid": os.getpid(), "ns": ns_info()}, "peer": peer,
            "same_pid_ns": peer["ns"].get("pid") == ns_info().get("pid"),
            "same_mnt_ns": peer["ns"].get("mnt") == ns_info().get("mnt"),
            "consistent": round_trips > 0 and outside == 0}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="capenv.clockcheck", description=__doc__.split("\n\n")[0])
    add_common_args(ap, cpus=True, inject=False, stop=True)
    ap.add_argument("--peer-mode", required=True, choices=("server", "client"))
    ap.add_argument("--port-file", required=True)
    ap.add_argument("--round-trips", type=int, required=True)
    ap.add_argument("--back-to-back", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        common = common_from_args(args)
        if args.round_trips < 1 or args.back_to_back < 1:
            raise ConfigError("--round-trips and --back-to-back must be >= 1")
        pin_self(common.cpus)
        if args.peer_mode == "server":
            server(common, args.port_file)
            return exitcodes.OK
        res = client(common, args.port_file, args.round_trips, args.back_to_back)
        write_json_stamped(args.out, SCHEMA, common.identity, res)
    except ConfigError as e:
        print("capenv.clockcheck: %s" % e, file=sys.stderr)
        return exitcodes.CONFIG
    print(json.dumps({"consistent": res["consistent"], "out": args.out}))
    return exitcodes.OK


if __name__ == "__main__":
    sys.exit(main())

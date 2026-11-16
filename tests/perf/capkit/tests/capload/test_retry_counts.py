# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep-alive reconnect retries count each request once; a failed on_ready stops the role."""

import os
import socket
import threading
import time

from capload_support import table_doc

from capcommon.snapshot import list_snapshot_files, read_snapshot_file
from capcommon.wire import DONE_FRAME, usage_frame

N_TOKENS = 5
DELAY_S = 0.5


class _DropSecondServer:
    """Answers the first request on every connection with a valid stream; on the second request
    of a connection it waits DELAY_S and closes it, either silently (mode "silent") or after an
    incomplete response head (mode "partial"). `triggers` counts the closes."""

    def __init__(self, mode):
        self.mode = mode
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        self.stop = False
        self.triggers = 0
        self.lock = threading.Lock()
        self.clients = []
        self.threads = []
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        self.sock.settimeout(0.2)
        while not self.stop:
            try:
                c, _ = self.sock.accept()
            except socket.timeout:
                continue
            self.clients.append(c)
            th = threading.Thread(target=self._serve, args=(c,), daemon=True)
            self.threads.append(th)
            th.start()

    def _read_request(self, c, buf):
        while b"\r\n\r\n" not in buf:
            d = c.recv(65536)
            if not d:
                return None, None
            buf += d
        head, _, rest = buf.partition(b"\r\n\r\n")
        hdrs = {}
        for ln in head.split(b"\r\n")[1:]:
            k, _, v = ln.partition(b":")
            hdrs[k.strip().lower()] = v.strip()
        clen = int(hdrs[b"content-length"])
        while len(rest) < clen:
            d = c.recv(65536)
            if not d:
                return None, None
            rest += d
        return int(hdrs[b"x-capload-id"]), rest[clen:]

    def _serve(self, c):
        c.settimeout(10)
        buf = b""
        n = 0
        try:
            while not self.stop:
                rid, buf = self._read_request(c, buf)
                if rid is None:
                    return
                if n == 1:
                    if self.mode == "partial":
                        c.sendall(b"HTTP/1.1 200 OK\r\n")
                    time.sleep(DELAY_S)
                    with self.lock:
                        self.triggers += 1
                    return
                n += 1
                body = usage_frame(rid, 0, N_TOKENS) + DONE_FRAME
                c.sendall(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                          b"x-capload-id: %d\r\ntransfer-encoding: chunked\r\n\r\n" % rid
                          + b"%x\r\n" % len(body) + body + b"\r\n0\r\n\r\n")
        except OSError:
            return
        finally:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            c.close()

    def close(self):
        self.stop = True
        self.t.join(2)
        self.sock.close()
        for th in self.threads:
            th.join(DELAY_S + 2)


def _spec():
    return [("warmup", 300, 5.0, {"n": N_TOKENS}), ("drain", 300, 0.0, {}),
            ("up", 2000, 5.0, {"n": N_TOKENS, "baseline": True}), ("drain", 1500, 0.0, {})]


def _load_argv(lz, port):
    return (["-m", "capload"] + lz.common("capload")
            + ["--cpus", str(lz.cpus[-1]), "--target", "127.0.0.1:%d" % port,
               "--endpoint", "/v1/completions", "--model", "m", "--workers", "1",
               "--schedule", lz.sched_path, "--prompts", lz.prompts_path,
               "--conn-policy", "keepalive", "--pool-pick", "roundrobin", "--pool-seed", "1",
               "--pool-reset", "never", "--recv-mode", "count", "--chunk-sample-mod", "1",
               "--fionread-every-ms", "20", "--lag-period-us", "1000",
               "--max-records-per-worker", "10000"])


def _run(lz, mode):
    srv = _DropSecondServer(mode)
    try:
        p = lz._spawn(_load_argv(lz, srv.port), "capload")
        rc = lz.wait(p, (lz.origin - time.monotonic_ns()) / 1e9 + lz.table.end_ns / 1e9 + 20)
    finally:
        srv.close()
    assert rc == 0, lz.errtext("capload")
    return srv.triggers


def _tiers(lz):
    return range(len(lz.table.tiers))


def _total(lz, path):
    return sum(lz.sum_count("capload", s, path) for s in _tiers(lz))


def _slots(lz, names):
    tot = dict.fromkeys(names, 0)
    for path in list_snapshot_files(lz.shm).values():
        snap = read_snapshot_file(path)
        if (snap.meta.get("identity") or {}).get("role") != "capload":
            continue
        for n in names:
            tot[n] += snap.values.get(n, 0)
    return tot


def _failures(lz, reason):
    return sum(f["count"] for s in _tiers(lz) for summ in lz.summaries("capload", s)
               for f in summ["failures"] if f["reason"] == reason)


def test_reconnect_retry_counts_written_and_send_lag_once(launch):
    lz = launch(table_doc(_spec()))
    triggers = _run(lz, "silent")
    planned = _total(lz, "counts.planned")
    assert triggers > 0 and planned > 0  # the retry path was exercised
    assert _total(lz, "conn_pool.reconnects") == triggers
    assert _total(lz, "counts.ok") == planned and _total(lz, "counts.failed") == 0
    s = _slots(lz, ["self.dispatched", "self.written", "self.send_lag.n",
                    "self.conn_reconnect", "self.send_lag.max"])
    assert s["self.conn_reconnect"] == triggers
    assert s["self.dispatched"] == planned
    assert s["self.written"] == planned
    assert s["self.send_lag.n"] == planned
    # the reconnect (>= DELAY_S after the first write) is not part of the send lag
    assert 0 <= s["self.send_lag.max"] < DELAY_S * 1e9
    lags = [sent - pl for seq in _tiers(lz) for _h, cols in lz.capbins("capload", "requests", seq)
            for pl, sent in zip(cols["planned_rel_ns"], cols["sent_rel_ns"])]
    assert len(lags) == planned
    assert all(0 <= x < DELAY_S * 1e9 for x in lags), max(lags)


def test_no_retry_after_partial_response_head(launch):
    lz = launch(table_doc(_spec()))
    triggers = _run(lz, "partial")
    assert triggers > 0
    assert _total(lz, "conn_pool.reconnects") == 0
    assert _slots(lz, ["self.conn_reconnect"])["self.conn_reconnect"] == 0
    assert _failures(lz, "conn_reset") == triggers
    planned = _total(lz, "counts.planned")
    assert _total(lz, "counts.ok") + _total(lz, "counts.failed") == planned


def test_on_ready_oserror_stops_role_nonzero(launch):
    lz = launch()
    blocker = os.path.join(lz.root, "not-a-dir")
    with open(blocker, "w") as f:
        f.write("x")
    argv = (["-m", "capload.nullsse"] + lz.common("nullsse")
            + ["--cpus", str(lz.cpus[0]), "--port-file", lz.port_file, "--host", "127.0.0.1",
               "--port", "0", "--workers", "1", "--chunk-bytes", "300", "--pacing", "asap",
               "--idle-close-s", "1", "--lag-period-us", "1000"])
    argv[argv.index("--ready-file") + 1] = os.path.join(blocker, "ready.json")
    p = lz._spawn(argv, "nullsse")
    rc = lz.wait(p, 30)
    assert rc == 4, lz.errtext("nullsse")
    assert "on_ready failed" in lz.errtext("nullsse")

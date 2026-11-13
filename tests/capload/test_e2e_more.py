# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Injection heartbeats, wave mode, full receive mode, timeouts, idle close, errors."""

import socket
import subprocess
import sys
import threading

import pytest

from capload_support import MS, TOOLS, basic_spec, table_doc


def _all(lz):
    return range(len(lz.table.tiers))


def _total(lz, role, path):
    return sum(lz.sum_count(role, s, path) for s in _all(lz))


def _failures(lz, reason, cls=None):
    n = 0
    for s in _all(lz):
        for summ in lz.summaries("capload", s):
            for f in summ["failures"]:
                if f["reason"] == reason:
                    if cls is not None:
                        assert f["class_hint"] == cls
                    n += f["count"]
    return n


def test_inject_heartbeat_and_short(launch):
    inj = [{"role": "capload", "site": "recv.chunk", "mode": "busy", "delta_ns": 20000}]
    lz = launch(table_doc(basic_spec(up_kw={"inject": inj})))
    rc_l, rc_s = lz.run_pair(
        sink_kw={"workers": 1, "extra": ["--inject", "site=short.req,mode=short,short_tokens=2,"
                                         "every_n=3"]},
        load_kw={"workers": 1})
    assert (rc_l, rc_s) == (0, 0), lz.errtext("capload")
    entry = [e for e in lz.summaries("capload", 2)[0]["inject"] if e["site"] == "recv.chunk"][0]
    assert entry["spec"]["mode"] == "busy" and entry["activation_unchanged"]
    assert entry["hits_delta"] > 0
    assert entry["applied_delta"] == entry["expected_applied_delta"] == entry["hits_delta"]
    assert entry["applied_ns_delta"] >= 20000 * entry["applied_delta"]
    # the spec is tier-only: the warmup tier must show hits but nothing applied
    w = [e for e in lz.summaries("capload", 0)[0]["inject"] if e["site"] == "recv.chunk"][0]
    assert w["hits_delta"] > 0 and w["applied_delta"] == 0 and w["spec"] is None
    # short.req (launch-level on the sink): every 3rd request is 2 tokens short
    shorts = _total(lz, "nullsse", "counts.short_requests")
    applied = sum(e["applied_delta"] for s in _all(lz) for summ in lz.summaries("nullsse", s)
                  for e in summ["inject"] if e["site"] == "short.req")
    n_req = lz.sched_meta["n_requests"]
    assert shorts == applied == (n_req + 2) // 3 > 0
    assert _failures(lz, "short_output", "non_overload") == shorts
    assert _total(lz, "capload", "counts.ok") == n_req - shorts


def test_wave_mode(launch):
    lz = launch()
    delays = [0, 1 * MS, 2 * MS, 3 * MS]
    rc_l, rc_s = lz.run_pair(load_kw={"workers": 1, "extra": [
        "--mode", "wave", "--wave-size", "4", "--wave-delays-ns", ",".join(map(str, delays)),
        "--wave-gap-ns", str(2 * MS)]})
    assert (rc_l, rc_s) == (0, 0), lz.errtext("capload")
    for seq in (0, 2):
        s = lz.summaries("capload", seq)[0]
        assert s["counts"]["sent"] == s["counts"]["planned"] == s["counts"]["ok"] > 4
        rows = []
        for _h, c in lz.capbins("capload", "requests", seq):
            rows += [(c["req_id"][i], c["planned_rel_ns"][i], c["done_rel_ns"][i])
                     for i in range(len(c["req_id"]))]
        rows.sort()
        waves = [rows[i:i + 4] for i in range(0, len(rows), 4)]
        assert waves[0][0][1] >= lz.table.tiers[seq].start_ns
        prev_done = None
        for wv in waves:
            t0 = wv[0][1]
            assert [r[1] - t0 for r in wv] == delays[:len(wv)]
            if prev_done is not None:
                assert t0 >= prev_done + 2 * MS  # closed loop: after the previous wave ended
            prev_done = max(r[2] for r in wv)


def test_full_mode_chat_text_prompts(launch):
    prompts = {"schema": "capkit.prompts/1",
               "sets": {"p": {"kind": "text", "items": ["hello \"quoted\" max_tokens", "x"]}}}
    lz = launch(prompts=prompts)
    rc_l, rc_s = lz.run_pair(load_kw={"workers": 1, "recv_mode": "full",
                                      "endpoint": "/v1/chat/completions"})
    assert (rc_l, rc_s) == (0, 0), lz.errtext("capload")
    n = lz.sched_meta["n_requests"]
    assert _total(lz, "capload", "counts.ok") == n
    assert _total(lz, "nullsse", "counts.foreign_requests") == 0


def test_registered_timeout_is_overload(launch):
    spec = [("warmup", 300, 10.0, {"n": 100}), ("drain", 1500, 0.0, {}),
            ("up", 100, 10.0, {"n": 2}), ("drain", 300, 0.0, {})]
    lz = launch(table_doc(spec, timeout_ns=300 * MS))
    rc_l, rc_s = lz.run_pair(sink_kw={"workers": 1, "pacing": "beat"}, load_kw={"workers": 1})
    assert (rc_l, rc_s) == (0, 0), lz.errtext("capload")
    w = lz.summaries("capload", 0)[0]
    assert w["counts"]["planned"] == w["counts"]["sent"] > 0
    assert _failures(lz, "timeout_registered", "overload") == w["counts"]["planned"]
    assert lz.summaries("capload", 2)[0]["counts"]["ok"] > 0
    assert _total(lz, "nullsse", "counts.streams_aborted") == w["counts"]["planned"]


def test_peer_idle_close_counted_apart(launch):
    lz = launch(table_doc(basic_spec(drain_ms=600)))
    rc_l, rc_s = lz.run_pair(sink_kw={"workers": 1, "idle_close_s": 0.2},
                             load_kw={"workers": 1, "pool_reset": "never"})
    assert (rc_l, rc_s) == (0, 0), lz.errtext("capload")
    closed = _total(lz, "nullsse", "counts.idle_closed")
    seen = _total(lz, "capload", "conn_pool.peer_idle_closed")
    rec = _total(lz, "capload", "conn_pool.reconnects")
    assert closed > 0
    assert seen + rec == closed
    assert _total(lz, "capload", "counts.failed") == 0


def _run_rc(argv, timeout=60):
    env = {"PYTHONPATH": TOOLS, "PATH": "/usr/bin:/bin"}
    p = subprocess.run([sys.executable] + argv, env=env, capture_output=True, timeout=timeout)
    return p.returncode, p.stderr.decode(errors="replace")


def _load_argv(lz, extra=(), drop=()):
    argv = (["-m", "capload"] + lz.common("capload")
            + ["--cpus", "0", "--target", "127.0.0.1:9", "--endpoint", "/v1/completions",
               "--model", "m", "--workers", "1", "--schedule", lz.sched_path,
               "--prompts", lz.prompts_path, "--conn-policy", "keepalive", "--pool-pick",
               "roundrobin", "--pool-seed", "1", "--pool-reset", "never", "--recv-mode",
               "count", "--chunk-sample-mod", "1", "--fionread-every-ms", "20",
               "--lag-period-us", "1000", "--max-records-per-worker", "100"] + list(extra))
    for d in drop:
        i = argv.index(d)
        del argv[i:i + 2]
    return argv


def test_config_errors_exit_3(launch):
    lz = launch()
    assert _run_rc(_load_argv(lz, ["--inject", "site=nope,mode=busy,delta_ns=5"]))[0] == 3
    assert _run_rc(_load_argv(lz, ["--inject", "site=send.req,mode=drop"]))[0] == 3
    assert _run_rc(_load_argv(lz, ["--wave-size", "3"]))[0] == 3
    assert _run_rc(_load_argv(lz, ["--mode", "wave"]))[0] == 3
    rc, err = _run_rc(_load_argv(lz, drop=["--model"]))
    assert rc == 2  # argparse: a required parameter has no default
    try:
        import uvloop  # noqa: F401
    except ImportError:
        argv = _load_argv(lz)
        argv[argv.index("--loop") + 1] = "uvloop"
        rc, err = _run_rc(argv)
        assert rc == 3, err
        assert "uvloop" in err


class _FakeServer:
    """Odd x-capload-id -> 503; even -> a complete-looking SSE stream (usage = max_tokens)
    that ends without [DONE]."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.stop = False
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

    def _serve(self, c):
        buf = b""
        c.settimeout(5)
        try:
            while not self.stop:
                while b"\r\n\r\n" not in buf:
                    d = c.recv(65536)
                    if not d:
                        return
                    buf += d
                head, _, rest = buf.partition(b"\r\n\r\n")
                hdrs = dict(ln.split(b": ", 1) for ln in head.split(b"\r\n")[1:])
                clen = int(hdrs[b"Content-Length"])
                while len(rest) < clen:
                    rest += c.recv(65536)
                buf = rest[clen:]
                rid = int(hdrs[b"x-capload-id"])
                if rid % 2:
                    c.sendall(b"HTTP/1.1 503 Busy\r\ncontent-length: 4\r\n\r\nbusy")
                else:
                    # usage matches max_tokens (5), so only the missing [DONE] makes it bad
                    frame = b'data: {"choices":[],"usage":{"completion_tokens":5}}\n\n'
                    c.sendall(b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n"
                              + b"%x\r\n" % len(frame) + frame + b"\r\n0\r\n\r\n")
        except OSError:
            return
        finally:
            c.close()

    def close(self):
        self.stop = True
        self.t.join(2)
        self.sock.close()
        for c in self.clients:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for th in self.threads:
            th.join(2)


def test_http_error_and_protocol_error(launch):
    lz = launch()
    srv = _FakeServer()
    try:
        argv = _load_argv(lz)
        argv[argv.index("--target") + 1] = "127.0.0.1:%d" % srv.port
        argv[argv.index("--cpus") + 1] = str(lz.cpus[-1])
        p = lz._spawn(argv[0:], "capload")
        rc = lz.wait(p, 30)
    finally:
        srv.close()
    assert rc == 0, lz.errtext("capload")
    n = lz.sched_meta["n_requests"]
    odd = sum(1 for r in range(n) if r % 2)
    assert _failures(lz, "http_503", "unclassified") == odd
    assert _failures(lz, "protocol_error", "non_overload") == n - odd
    assert _total(lz, "capload", "counts.ok") == 0


@pytest.mark.parametrize("bad", ["--chunk-bytes=10", "--workers=0"])
def test_sink_config_errors(launch, bad):
    lz = launch()
    name, val = bad.split("=")
    argv = (["-m", "capload.nullsse"] + lz.common("nullsse")
            + ["--cpus", "0", "--port-file", lz.port_file, "--host", "127.0.0.1", "--port", "0",
               "--workers", "1", "--chunk-bytes", "300", "--pacing", "asap",
               "--idle-close-s", "1", "--lag-period-us", "1000"])
    argv[argv.index(name) + 1] = val
    assert _run_rc(argv)[0] == 3

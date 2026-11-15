# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A product-like dummy for tests/launch: HTTP /health, a plain (S-FE style) port file or a
trtllm-serve style --report_addr file, optional children. Never deployed."""

import argparse
import http.server
import json
import os
import signal
import subprocess
import sys
import threading
import time


def probe_child():
    """Emulate a capprobe'd child process: snapshot with identity from CAPPROBE_IDENTITY_FILE."""
    from capcommon.snapshot import SlotRegistry, SnapshotWriter  # noqa: PLC0415
    with open(os.environ["CAPPROBE_IDENTITY_FILE"]) as f:
        ident = json.load(f)["identity"]
    ident["pid"] = os.getpid()
    reg = SlotRegistry()
    reg.counter("count.F2.msgs")
    w = SnapshotWriter(reg, os.environ["CAPPROBE_SHM_DIR"],
                       int(os.environ["CAPPROBE_INTERVAL_MS"]) * 1_000_000,
                       meta_extra={"identity": ident, "role_hint": os.environ.get(
                           "CAPPROBE_ROLE_HINT"), "versions": {"python": sys.version}}).start()
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    while not done.wait(0.05):
        pass
    w.stop()


def probe_sites_loop(sites, every_ms, stop):
    """Emulate a probed product path (errata FT1-18): import the real capprobe (it reads the
    CAPPROBE_* switches the orchestrator set, once) and pass each named count site every
    `every_ms` ms through its guarded call, as the probe patch does."""
    import capprobe  # noqa: PLC0415

    calls = [(getattr(capprobe, "ON_" + s), getattr(capprobe, s)) for s in sites]
    while not stop.wait(every_ms / 1000):
        for on, fn in calls:
            if on:
                fn()


def entry_report(path):
    """Record whether this process is the CAPPROBE_ENTRY process (capcommon.probe_entry, the
    rule capprobe applies at import)."""
    from capcommon import probe_entry  # noqa: PLC0415
    raw = os.environ.get(probe_entry.ENV_VAR)
    rule = probe_entry.parse(raw) if raw else None
    argv = probe_entry.process_argv()
    doc = {"pid": os.getpid(), "argv": argv, "rule": rule,
           "match": None if rule is None else probe_entry.match(rule, argv)}
    with open(path + ".tmp", "w") as f:
        json.dump(doc, f)
    os.replace(path + ".tmp", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--port-file")
    p.add_argument("--report-addr")
    p.add_argument("--health-status", type=int, default=200)
    p.add_argument("--delay-ready-s", type=float, default=0.0)
    p.add_argument("--spawn-child", action="store_true")
    p.add_argument("--probe-child", action="store_true")
    p.add_argument("--orphan", action="store_true", help="setsid grandchild ignoring SIGTERM")
    p.add_argument("--ignore-term", action="store_true")
    p.add_argument("--peer-label", help="ignored (lets a test put {addr:...} in the argv)")
    p.add_argument("--churn-every-ms", type=int, default=0,
                   help="start a short-lived child this often")
    p.add_argument("--role-child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--entry-report", help="dir: main.json (this process), spawn.json (a "
                   "multiprocessing spawn child)")
    p.add_argument("--probe-sites", help="comma list of capprobe count sites to pass")
    p.add_argument("--probe-every-ms", type=float, default=5.0)
    a = p.parse_args()
    if a.role_child:
        return probe_child()
    if a.entry_report:
        import multiprocessing  # noqa: PLC0415
        entry_report(os.path.join(a.entry_report, "main.json"))
        sp = multiprocessing.get_context("spawn").Process(
            target=entry_report, args=(os.path.join(a.entry_report, "spawn.json"),))
        sp.start()
        sp.join(60)
    stop = threading.Event()
    if a.ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    status = a.health_status

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(status if self.path == "/health" else 404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer((a.host, 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    if a.probe_sites:
        import capprobe  # noqa: F401, PLC0415  (switches parsed before the role is ready)
        threading.Thread(target=probe_sites_loop, name="dummy-probe-sites", daemon=True,
                         args=(a.probe_sites.split(","), a.probe_every_ms, stop)).start()
    kids = []
    if a.spawn_child:
        kids.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"]))
    if a.probe_child:
        kids.append(subprocess.Popen([sys.executable, __file__, "--host", a.host,
                                      "--role-child"]))
    if a.orphan:
        subprocess.Popen([sys.executable, "-c",
                          "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                          "time.sleep(600)"], start_new_session=True)
    time.sleep(a.delay_ready_s)
    port = srv.server_address[1]
    if a.port_file:
        with open(a.port_file + ".partial", "w") as f:
            f.write("%d\n" % port)
        os.replace(a.port_file + ".partial", a.port_file)
    if a.report_addr:
        with open(a.report_addr + ".tmp", "w") as f:
            f.write("%s:%d\n" % (a.host, port))
        os.replace(a.report_addr + ".tmp", a.report_addr)
    churn, next_kid = [], time.monotonic()
    while not stop.wait(min(0.05, (a.churn_every_ms or 50) / 1000)):
        if a.churn_every_ms and time.monotonic() >= next_kid:
            churn = [k for k in churn if k.poll() is None]
            churn.append(subprocess.Popen(["sleep", "0.2"]))
            next_kid = time.monotonic() + a.churn_every_ms / 1000
    kids += churn
    for k in kids:
        k.terminate()
    for k in kids:
        k.wait()
    srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

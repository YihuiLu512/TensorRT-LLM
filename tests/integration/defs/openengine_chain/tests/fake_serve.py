# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stand-in for ``trtllm-serve`` in the launch tests; installed as an executable named trtllm-serve.

With ``--grpc`` it is an engine: a gRPC health server whose ``inference``
service turns SERVING. Otherwise it is a front end (role ``frontend-check``
when the harness's front-end check variable is set): it refuses a
``--max_beam_width`` above 1 as the real one does (exit 2), connects once to its
``--openengine-endpoint`` and is then an HTTP server answering ``/health``.
Each run records its arguments, environment and signal times in
``$FAKE_RECORD_DIR``. ``FAKE_<ROLE>_MODE`` (``FAKE_FRONTEND_CHECK_MODE`` for the
check) is ``ok``, ``exit`` (exit 3 at once), ``done`` (exit 0 at once),
``killed`` (SIGKILL itself at once), ``busy_once`` (the first run fails to bind
its port, later runs are ``ok``) or ``never_ready`` (an engine never turns
SERVING, a front end never connects); ``FAKE_<ROLE>_IGNORE_TERM`` ignores
SIGTERM; options take ``--name value`` and ``--name=value`` alike; a front end
whose port is taken logs uvicorn's bind error and exits 1;
``FAKE_ENGINE_CHILD`` starts a child in the same session that ignores SIGTERM.
"""

import errno
import json
import os
import signal
import socket
import subprocess
import sys
import time

args = sys.argv[1:]
if "--grpc" in args:
    role = "engine"
elif os.environ.get("TRTLLM_OPENENGINE_CHAIN_FRONTEND_CHECK"):
    role = "frontend-check"
else:
    role = "frontend"
record_dir = os.environ["FAKE_RECORD_DIR"]
env_prefix = f"FAKE_{role.upper().replace('-', '_')}_"
mode = os.environ.get(env_prefix + "MODE", "ok")


def option(name, default=None):
    """The value of option ``name``, given as ``name value`` or ``name=value``."""
    for i, arg in enumerate(args):
        if arg == name and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return default


def record(name, data):
    path = os.path.join(record_dir, f"{role}-{name}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(data, f)
    os.replace(path + ".tmp", path)


def on_term(signum, frame):
    record("term", {"time": time.monotonic()})
    if not os.environ.get(env_prefix + "IGNORE_TERM"):
        sys.exit(0)


signal.signal(signal.SIGTERM, on_term)
port = int(option("--port", 8000))
child = None
if role == "engine" and os.environ.get("FAKE_ENGINE_CHILD"):
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)",
        ]
    )
record(
    "start",
    {
        "argv": args,
        "pid": os.getpid(),
        "time": time.monotonic(),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("TRTLLM_", "TLLM_"))},
        "child": child.pid if child else None,
    },
)
print(f"{role} started pid {os.getpid()}", flush=True)
if mode == "exit":
    sys.exit(3)
if mode == "done":
    sys.exit(0)
if mode == "killed":
    os.kill(os.getpid(), signal.SIGKILL)
if mode == "busy_once":
    marker = os.path.join(record_dir, f"{role}-busy")
    if not os.path.exists(marker):
        open(marker, "w").close()
        print(f"RuntimeError: Failed to bind 127.0.0.1:{port}: address already in use", flush=True)
        sys.exit(1)

if role == "engine":
    from concurrent.futures import ThreadPoolExecutor

    import grpc
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc

    server = grpc.server(ThreadPoolExecutor(2))
    servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(servicer, server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    servicer.set("inference", health_pb2.HealthCheckResponse.NOT_SERVING)
    time.sleep(float(os.environ.get("FAKE_ENGINE_DELAY", "0.5")))
    if mode != "never_ready":
        servicer.set("inference", health_pb2.HealthCheckResponse.SERVING)
    while True:
        time.sleep(0.1)
else:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    if int(option("--max_beam_width", 1)) > 1:
        print(
            "Error: max_beam_width > 1 is not supported: OpenEngine has no beam search",
            flush=True,
        )
        sys.exit(2)
    if mode != "never_ready" and option("--openengine-endpoint") is not None:
        engine_host, _, engine_port = option("--openengine-endpoint").rpartition(":")
        try:
            socket.create_connection((engine_host, int(engine_port)), timeout=1).close()
        except OSError:
            pass

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def log_message(self, *a):
            pass

    try:
        httpd = HTTPServer(("127.0.0.1", port), Handler)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        print(
            "ERROR:    [Errno 98] error while attempting to bind on address "
            f"('127.0.0.1', {port}): address already in use",
            flush=True,
        )
        sys.exit(1)
    httpd.timeout = 0.1
    while True:
        httpd.handle_request()

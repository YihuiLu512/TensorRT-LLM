# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared test helpers for the B4 (capenv) tests."""

import os
import signal
import subprocess
import sys
import time

C = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PATHS = [os.path.join(C, "tools")]
SHA = "b" * 64
PY = sys.executable


def capenv_env(extra=None, path=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(PATHS)
    if path is not None:
        env["PATH"] = path
    if extra:
        env.update(extra)
    return env


def id_args(role, launch_seq=1):
    return ["--experiment", "E0", "--arm", "unit-a", "--launch-seq", str(launch_seq),
            "--job-id", "local-b4test", "--restart-count", "0", "--register-sha", SHA,
            "--role", role]


def make_ident(role="launch"):
    from capcommon.identity import Identity
    return Identity(experiment="E0", arm="unit-a", launch_seq=1, job_id="local-b4test",
                    restart_count=0, register_sha=SHA, role=role, worker=0, pid=os.getpid())


SPIN_SRC = "import time\nwhile True:\n    pass\n"
IDLE_SRC = "import time\nwhile True:\n    time.sleep(0.5)\n"
THREADS_SRC = ("import threading, time\n"
               "def spin():\n    while True:\n        pass\n"
               "t = threading.Thread(target=spin, name='b4spin', daemon=True); t.start()\n"
               "while True:\n    time.sleep(0.5)\n")


def spawn_py(src, cpus=None):
    """Start a child python running src; optionally pinned. Caller must kill_all()."""
    p = subprocess.Popen([PY, "-c", src], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if cpus is not None:
        os.sched_setaffinity(p.pid, set(cpus))
    return p


def kill_all(procs, sig=signal.SIGKILL):
    for p in procs:
        if p.poll() is None:
            try:
                p.send_signal(sig)
            except ProcessLookupError:
                pass
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=10)


def wait_for(pred, timeout_s, poll_s=0.05):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(poll_s)
    return pred()

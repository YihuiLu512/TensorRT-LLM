# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E15-B positive-control arm H-inproc+busy (design 13.3): one pure-Python busy thread with a
configurable duty cycle inside the engine process, so the judge has a reading of "GIL contention
present". Started only by capstubs/e15_busy/sitecustomize.py, and only when the environment says
so:

  CAPKIT_E15_BUSY_DUTY       duty cycle in (0, 1) (busy share of every period)
  CAPKIT_E15_BUSY_PERIOD_MS  period in ms (1..1000)
  CAPKIT_E15_BUSY_OUT        directory for the heartbeat file e15_busy.<pid>.json

The first process that loads the hook claims it (CAPKIT_E15_BUSY_PID = its pid, set in
os.environ before anything else starts); a child that inherits the environment sees another pid
there and does nothing. The heartbeat file is rewritten every 2 s: periods, busy_ns, loop count.
"""

import json
import os
import threading
import time

ENV_DUTY = "CAPKIT_E15_BUSY_DUTY"
ENV_PERIOD = "CAPKIT_E15_BUSY_PERIOD_MS"
ENV_OUT = "CAPKIT_E15_BUSY_OUT"
ENV_PID = "CAPKIT_E15_BUSY_PID"
THREAD_NAME = "capkit_e15_busy"
SCHEMA = "capkit.e15_busy/1"


def settings(environ):
    """(duty, period_ns, out_dir) or None when the hook is off; ValueError on bad values."""
    if not environ.get(ENV_DUTY):
        return None
    duty = float(environ[ENV_DUTY])
    period_ms = int(environ.get(ENV_PERIOD, ""))
    out = environ.get(ENV_OUT) or ""
    if not 0 < duty < 1 or not 1 <= period_ms <= 1000 or not out.startswith("/"):
        raise ValueError("bad e15 busy settings duty=%r period_ms=%r out=%r"
                         % (duty, period_ms, out))
    return duty, period_ms * 1_000_000, out


class Busy:
    def __init__(self, duty, period_ns, out_dir, clock=time.monotonic_ns):
        self.duty, self.period_ns, self.out_dir = duty, period_ns, out_dir
        self.clock = clock
        self.periods = self.busy_ns = self.loops = 0
        self.t0 = clock()
        self.stop = threading.Event()

    def one_period(self):
        start = self.clock()
        end_busy = start + int(self.duty * self.period_ns)
        x = 0
        while self.clock() < end_busy:
            x += 1
        busy_end = self.clock()
        self.loops += x
        self.busy_ns += busy_end - start
        self.periods += 1
        rest = start + self.period_ns - self.clock()
        if rest > 0:
            self.stop.wait(rest / 1e9)

    def write(self):
        os.makedirs(self.out_dir, exist_ok=True)
        p = os.path.join(self.out_dir, "e15_busy.%d.json" % os.getpid())
        doc = {"schema": SCHEMA, "pid": os.getpid(), "duty": self.duty,
               "period_ns": self.period_ns, "periods": self.periods, "busy_ns": self.busy_ns,
               "loops": self.loops, "elapsed_ns": self.clock() - self.t0,
               "thread": THREAD_NAME}
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(doc, f, sort_keys=True)
        os.replace(p + ".tmp", p)
        return p

    def run(self):
        last = 0
        while not self.stop.is_set():
            self.one_period()
            now = self.clock()
            if now - last >= 2_000_000_000:
                self.write()
                last = now


def start_from_env(environ=None):
    """Start the busy thread when the environment asks for it and this process claims the hook;
    returns the Busy object or None."""
    env = os.environ if environ is None else environ
    cfg = settings(env)
    if cfg is None:
        return None
    owner = env.get(ENV_PID)
    if owner and owner != str(os.getpid()):
        return None
    env[ENV_PID] = str(os.getpid())
    b = Busy(*cfg)
    b.write()
    threading.Thread(target=b.run, name=THREAD_NAME, daemon=True).start()
    return b

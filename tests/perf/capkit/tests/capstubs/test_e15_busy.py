# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E15-B busy-thread hook: settings, ownership by the first process, duty cycle, heartbeat."""

import json
import os
import subprocess
import sys

import pytest

from capstubs import e15_busy_core as B

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, "..", "..", "tools"))


def test_settings():
    assert B.settings({}) is None
    assert B.settings({B.ENV_DUTY: "0.5", B.ENV_PERIOD: "10", B.ENV_OUT: "/o"}) == \
        (0.5, 10_000_000, "/o")
    for bad in ({B.ENV_DUTY: "1", B.ENV_PERIOD: "10", B.ENV_OUT: "/o"},
                {B.ENV_DUTY: "0.5", B.ENV_PERIOD: "0", B.ENV_OUT: "/o"},
                {B.ENV_DUTY: "0.5", B.ENV_PERIOD: "10", B.ENV_OUT: "rel"},
                {B.ENV_DUTY: "0.5", B.ENV_OUT: "/o"}):
        with pytest.raises(ValueError):
            B.settings(bad)


def test_duty_cycle_and_heartbeat(tmp_path):
    b = B.Busy(0.5, 20_000_000, str(tmp_path))
    for _ in range(10):
        b.one_period()
    assert b.periods == 10 and b.loops > 0
    assert 0.4 * 10 * 20e6 <= b.busy_ns <= 0.7 * 10 * 20e6
    doc = json.load(open(b.write()))
    assert doc["periods"] == 10 and doc["schema"] == B.SCHEMA


def test_only_the_claiming_process_starts(tmp_path):
    env = {B.ENV_DUTY: "0.2", B.ENV_PERIOD: "10", B.ENV_OUT: str(tmp_path),
           B.ENV_PID: "1"}
    assert B.start_from_env(env) is None  # another pid owns the hook
    env.pop(B.ENV_PID)
    b = B.start_from_env(env)
    assert b is not None and env[B.ENV_PID] == str(os.getpid())
    b.stop.set()
    assert os.path.exists(os.path.join(str(tmp_path), "e15_busy.%d.json" % os.getpid()))


def test_sitecustomize_hook_in_a_fresh_interpreter(tmp_path):
    hook = os.path.join(TOOLS, "capstubs", "e15_busy")
    env = dict(os.environ, PYTHONPATH="%s:%s" % (hook, TOOLS))
    code = "import threading,time; time.sleep(0.3); print([t.name for t in threading.enumerate()])"
    off = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert "started=False" in off.stderr and "capkit_e15_busy" not in off.stdout
    env.update({B.ENV_DUTY: "0.3", B.ENV_PERIOD: "10", B.ENV_OUT: str(tmp_path)})
    on = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert "started=True" in on.stderr and "capkit_e15_busy" in on.stdout, on.stderr

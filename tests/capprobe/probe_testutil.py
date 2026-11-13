# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for the capprobe tests (a uniquely named module: test files never import `conftest`,
so running several packages' test directories in one pytest call cannot mix their helpers)."""

import json
import os
import subprocess
import sys

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
SHA = "b" * 64
IDENTITY = {"experiment": "E0", "arm": "probe-unit", "launch_seq": 3, "job_id": "local-b3",
            "restart_count": 0, "register_sha": SHA, "role": None, "worker": 0, "pid": None,
            "tier_seq": None, "tier_label": None}


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAPPROBE_")}
    env["PYTHONPATH"] = TOOLS
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run_py(code, env, timeout=60, args=()):
    """Run `python -c code` in a fresh interpreter; returns CompletedProcess (text)."""
    return subprocess.run([sys.executable, "-c", code, *args], env=env, capture_output=True,
                          text=True, timeout=timeout, cwd=os.path.dirname(__file__))


def run_json(code, env, timeout=60):
    r = run_py(code, env, timeout=timeout)
    assert r.returncode == 0, "rc=%d\nstdout=%s\nstderr=%s" % (r.returncode, r.stdout, r.stderr)
    return json.loads(r.stdout.strip().splitlines()[-1])


# Prologue for subprocess snippets: decode this process's own snapshot after flush().
DECODE_SELF = """
import json, os, capprobe
from capcommon.snapshot import read_snapshot_file
def snap():
    capprobe.flush()
    s = read_snapshot_file(capprobe.snapshot_path())
    return {"seq": s.seq, "meta": s.meta, "values": s.values}
"""

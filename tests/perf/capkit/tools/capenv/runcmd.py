# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an external command and keep the evidence (argv, return code, stdout and its sha256).

A missing executable is not an exception and not an empty success: it is a record with
rc = None and a reason, so callers can mark the item unreadable / NOT RUN.
"""

import os
import shutil
import subprocess

from capcommon.hashing import sha256_bytes


def run(argv, timeout_s, env=None):
    """Return {"cmd", "rc", "stdout", "stdout_sha256", "stderr", "reason"}.

    rc is the process exit code, or None when the command could not be run (not found, timeout,
    OSError); reason says why. stdout is decoded as UTF-8 with replacement.
    """
    rec = {"cmd": list(argv), "rc": None, "stdout": None, "stdout_sha256": None, "stderr": None,
           "reason": None}
    exe = argv[0]
    if os.sep not in exe and shutil.which(exe, path=(env or os.environ).get("PATH")) is None:
        rec["reason"] = "not found: %s" % exe
        return rec
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s,
                           env=env, check=False)
    except subprocess.TimeoutExpired:
        rec["reason"] = "timeout after %ss" % timeout_s
        return rec
    except OSError as e:
        rec["reason"] = "%s: %s" % (type(e).__name__, e)
        return rec
    rec["rc"] = p.returncode
    rec["stdout"] = p.stdout.decode("utf-8", errors="replace")
    rec["stdout_sha256"] = sha256_bytes(p.stdout)
    rec["stderr"] = p.stderr.decode("utf-8", errors="replace")[-4000:]
    if p.returncode != 0:
        rec["reason"] = "exit code %d" % p.returncode
    return rec


def ok(rec):
    return rec is not None and rec.get("rc") == 0


def why(rec):
    if rec is None:
        return "not run"
    return rec.get("reason") or ("exit code %s" % rec.get("rc"))

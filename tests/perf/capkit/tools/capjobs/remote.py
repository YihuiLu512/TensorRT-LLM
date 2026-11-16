# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Front-end command construction and the one place that executes ssh.

Every cluster command is built as an argv list for the *remote* side first (so policy checks such
as policy.check_test_only_argv see real tokens), then wrapped into an ssh argv whose last element
is the shell-quoted remote command. The Runner is injectable: tests pass a stub and never execute
ssh. Only read-only Slurm commands (sinfo included) plus sbatch are allowed (plan CONSTRAINTS
section 7).
"""

import re
import shlex
import subprocess

from capcommon.exitcodes import ConfigError

HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,253}$")

# First token of every remote command this package may run (CONSTRAINTS 7: front end only does
# job management). scancel is deliberately absent: cancelling is a human decision. sinfo only
# queries (exclude --all-but, pin-next; erratum calib-v2-A4), so its arguments are not limited.
ALLOWED_REMOTE = ("sbatch", "squeue", "sacct", "scontrol", "cdb", "sinfo")
SCONTROL_ALLOWED_SUB = ("show",)


class RemoteError(ConfigError):
    pass


def check_remote_argv(remote_argv):
    """Refuse anything but the allow-listed read-only commands and sbatch."""
    if not isinstance(remote_argv, list) or not remote_argv \
            or not all(isinstance(a, str) for a in remote_argv):
        raise RemoteError("remote argv must be a non-empty list of strings")
    head = remote_argv[0]
    if head not in ALLOWED_REMOTE:
        raise RemoteError("remote command %r is not allowed (allowed: %s)" % (head, ALLOWED_REMOTE))
    if head == "scontrol" and (len(remote_argv) < 2 or remote_argv[1] not in SCONTROL_ALLOWED_SUB):
        raise RemoteError("only 'scontrol show ...' is allowed")
    if head == "cdb" and (len(remote_argv) < 2 or remote_argv[1] != "find"):
        raise RemoteError("only 'cdb find ...' is allowed")
    return remote_argv


def ssh_argv(login_host, remote_argv):
    """ssh argv for one remote command. login_host is always passed explicitly by the caller."""
    if not isinstance(login_host, str) or not HOST_RE.match(login_host):
        raise RemoteError("bad login host %r" % (login_host,))
    check_remote_argv(remote_argv)
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", login_host,
            shlex.join(remote_argv)]


class Runner:
    """Executes argv lists. run() returns (rc, stdout, stderr); rc is None when the command could
    not be started or timed out (callers treat None as 'unknown', never as success)."""

    def __init__(self, timeout_s):
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise RemoteError("timeout_s must be > 0")
        self.timeout_s = timeout_s

    def run(self, argv):
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout_s,
                               check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            return None, "", "%s: %s" % (type(e).__name__, e)
        return r.returncode, r.stdout, r.stderr


def run_remote(runner, login_host, remote_argv):
    """Run one remote command through ssh. Returns dict with the ssh argv and the raw result."""
    argv = ssh_argv(login_host, remote_argv)
    rc, out, err = runner.run(argv)
    return {"remote_argv": list(remote_argv), "argv": argv, "rc": rc, "stdout": out,
            "stderr": err}

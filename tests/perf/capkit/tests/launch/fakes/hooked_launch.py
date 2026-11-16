# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`python hooked_launch.py <caplaunch arguments>`: one launch (as `python -m caplaunch`) with test
hooks chosen by environment variables of this process.

HOOKED_LAUNCH_SIGNAL_ON_PID_READ=1: a stop signal lands between `subprocess.Popen(...)` returning
and the orchestrator reading the pid of the process it just started. The first process started
with CAPPROBE_ENTRY in its environment (the first product role) gets a Popen whose `pid`
attribute raises SIGTERM in this process on its first read after the constructor returned.
`signal.raise_signal` runs the Python handler before it returns, so the orchestrator's handler
raises inside that read, exactly in the window. That child is also created with SIGTERM ignored
(inherited across exec), so it outlives the stop grace and the cleanup has to SIGKILL its
process group.

HOOKED_LAUNCH_ROLES_WRITES=<file>: every payload the orchestrator writes to roles.json is
appended to <file> as one JSON line, in write order (what a reader polling roles.json could see).
"""

import json
import os
import signal
import subprocess
import sys

_Popen = subprocess.Popen


class _SignalOnPidReadPopen(_Popen):
    fired = False

    def __init__(self, *args, **kw):
        self.__dict__["_signal_on_pid_read"] = False
        hook = (not _SignalOnPidReadPopen.fired
                and "CAPPROBE_ENTRY" in (kw.get("env") or {}))
        old = signal.signal(signal.SIGTERM, signal.SIG_IGN) if hook else None
        try:
            super().__init__(*args, **kw)
        finally:
            if hook:
                signal.signal(signal.SIGTERM, old)
        if hook:
            _SignalOnPidReadPopen.fired = True
            self.__dict__["_signal_on_pid_read"] = True

    @property
    def pid(self):
        if self.__dict__.get("_signal_on_pid_read"):
            self.__dict__["_signal_on_pid_read"] = False
            signal.raise_signal(signal.SIGTERM)
        return self.__dict__.get("_pid_value")

    @pid.setter
    def pid(self, value):
        self.__dict__["_pid_value"] = value


def _record_roles_writes(orchestrator, out):
    real = orchestrator.write_json_stamped

    def write(path, schema, identity, payload):
        real(path, schema, identity, payload)
        if schema == "capkit.roles/1":
            with open(out, "a") as f:
                f.write(json.dumps(payload) + "\n")

    orchestrator.write_json_stamped = write


def main(argv):
    if os.environ.get("HOOKED_LAUNCH_SIGNAL_ON_PID_READ") == "1":
        subprocess.Popen = _SignalOnPidReadPopen
    from caplaunch import orchestrator  # noqa: PLC0415
    from caplaunch.__main__ import main as launch_main  # noqa: PLC0415
    if os.environ.get("HOOKED_LAUNCH_ROLES_WRITES"):
        _record_roles_writes(orchestrator, os.environ["HOOKED_LAUNCH_ROLES_WRITES"])
    return launch_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

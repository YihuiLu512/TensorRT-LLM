# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake capenv.sidecar for tests/launch: takes the INTERFACES.md 11 sidecar arguments, writes
meta.json once and records each roles.json generation it sees, until SIGTERM."""
import argparse
import json
import os
import subprocess
import sys
import time

from capcommon.cli import add_identity_args, common_from_args, parse_cpu_list
from capcommon.lifecycle import StopController, write_ready_file
from capcommon.versions import version_record


def main():
    p = argparse.ArgumentParser()
    add_identity_args(p)
    for k in ("--run-dir", "--shm-dir", "--roles-file", "--out-dir", "--logs-dir",
              "--cpus-watch"):
        p.add_argument(k, required=True)
    p.add_argument("--stop-grace-s", type=float, required=True)
    p.add_argument("--cpus", type=parse_cpu_list, required=True)
    p.add_argument("--origin-ns", type=int, required=True)
    p.add_argument("--interval-ms", type=int, required=True)
    p.add_argument("--gpu-every-s", type=float, required=True)
    p.add_argument("--numa-every-s", type=float, required=True)
    p.add_argument("--ready-file", required=True)
    p.add_argument("--nvidia-smi", required=True)
    p.add_argument("--churn-every-ms", type=int, default=0,
                   help="test only: run --nvidia-smi this often (the real sidecar's GPU query)")
    a = p.parse_args()
    stop = StopController()
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump({"fake": True, "cpus_watch": a.cpus_watch}, f)
    write_ready_file(a.ready_file, common_from_args(a).identity,
                     version_record(include_grpc=False))
    seen, kids, next_kid = [], [], time.monotonic()
    while not stop.requested:
        if a.churn_every_ms and time.monotonic() >= next_kid:
            kids = [k for k in kids if k.poll() is None]
            kids.append(subprocess.Popen([a.nvidia_smi, "0.2"]))
            next_kid = time.monotonic() + a.churn_every_ms / 1000
        try:
            with open(a.roles_file) as f:
                g = json.load(f)["generation"]
            if not seen or seen[-1] != g:
                seen.append(g)
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(min(a.interval_ms, a.churn_every_ms or a.interval_ms) / 1000)
    for k in kids:
        k.wait()
    with open(os.path.join(a.out_dir, "generations.json"), "w") as f:
        json.dump(seen, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())

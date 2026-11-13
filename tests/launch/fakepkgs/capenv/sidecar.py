# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake capenv.sidecar for tests/launch: takes the INTERFACES.md 11 sidecar arguments, writes
meta.json once and records each roles.json generation it sees, until SIGTERM."""
import argparse
import json
import os
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
    a = p.parse_args()
    stop = StopController()
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump({"fake": True, "cpus_watch": a.cpus_watch}, f)
    write_ready_file(a.ready_file, common_from_args(a).identity,
                     version_record(include_grpc=False))
    seen = []
    while not stop.requested:
        try:
            with open(a.roles_file) as f:
                g = json.load(f)["generation"]
            if not seen or seen[-1] != g:
                seen.append(g)
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(a.interval_ms / 1000)
    with open(os.path.join(a.out_dir, "generations.json"), "w") as f:
        json.dump(seen, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())

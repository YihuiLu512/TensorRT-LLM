# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake capenv.canary for tests/launch: canary.jsonl rows every interval until SIGTERM."""
import argparse
import sys
import time

from capcommon.cli import add_identity_args, common_from_args, parse_cpu_list
from capcommon.jsonio import JsonlWriter
from capcommon.lifecycle import StopController, write_ready_file
from capcommon.versions import version_record


def main():
    p = argparse.ArgumentParser()
    add_identity_args(p)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--stop-grace-s", type=float, required=True)
    p.add_argument("--cpus", type=parse_cpu_list, required=True)
    p.add_argument("--origin-ns", type=int, required=True)
    p.add_argument("--interval-ms", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ready-file", required=True)
    a = p.parse_args()
    ident = common_from_args(a).identity
    stop = StopController()
    write_ready_file(a.ready_file, ident, version_record(include_grpc=False))
    with JsonlWriter(a.out, "capkit.canary/1", ident,
                     ["t_rel_ns", "iters", "thread_cpu_ns", "cpu"]) as w:
        while not stop.requested:
            t0 = time.monotonic_ns()
            n = 0
            while time.monotonic_ns() - t0 < a.interval_ms * 1_000_000 // 4:
                n += 1
            w.write_row([time.monotonic_ns() - a.origin_ns, n, time.thread_time_ns(), 0])
            time.sleep(a.interval_ms / 1000 * 0.75)
    return 0


if __name__ == "__main__":
    sys.exit(main())

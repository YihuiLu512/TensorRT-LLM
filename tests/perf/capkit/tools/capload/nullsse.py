# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``python -m capload.nullsse``: multi-process SO_REUSEPORT SSE null sink (INTERFACES.md 11).

The parent binds (without listening) to reserve the port when --port is 0, spawns Z workers that
each bind + listen on the same port with SO_REUSEPORT, and writes the port file only after every
worker is listening, then the ready file (children = the worker pids).
"""

import argparse
import math
import socket
import sys
import time

from capcommon.cli import add_common_args, common_from_args
from capcommon.exitcodes import CapkitError, ConfigError
from capcommon.lifecycle import write_port_file, write_ready_file
from capcommon.schedule import count_schedule_rows

from . import SINK_SLOTS
from ._proc import ParentRunner
from .sink import min_chunk_bytes


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capload.nullsse",
                                description="multi-process SSE null sink")
    add_common_args(p, loop=True, tiers=True, snapshot=True, ready=True, port=True)
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True, help="0 = any free port")
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--chunk-bytes", type=int, required=True)
    p.add_argument("--pacing", required=True, choices=("beat", "asap"))
    p.add_argument("--idle-close-s", type=float, required=True,
                   help="close keep-alive connections idle this long (<= 0: never)")
    p.add_argument("--lag-period-us", type=int, required=True)
    p.add_argument("--max-records-per-worker", type=int, default=None,
                   help="optional; default = schedule rows implied by the tier table (streams)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    reserve = None
    try:
        common = common_from_args(args)
        if args.workers < 1:
            raise ConfigError("--workers must be >= 1")
        if common.cpus is not None and len(common.cpus) < args.workers:
            raise ConfigError("--cpus lists %d cpus for %d workers"
                              % (len(common.cpus), args.workers))
        if args.lag_period_us < 1:
            raise ConfigError("--lag-period-us must be >= 1")
        if args.chunk_bytes < min_chunk_bytes():
            raise ConfigError("--chunk-bytes must be >= %d" % min_chunk_bytes())
        if not 0 <= args.port < 65536:
            raise ConfigError("--port must be in [0, 65536)")
        table = common.load_tier_table()
        if args.pacing == "beat" and not any(t.beat_ns > 0 for t in table.tiers):
            raise ConfigError("--pacing beat needs beat_ns > 0 in the tier table")
        cap = args.max_records_per_worker
        if cap is None:
            cap = count_schedule_rows(table)
        lag_cap = math.ceil((table.end_ns + max(0, common.origin_ns - time.monotonic_ns()))
                            / (args.lag_period_us * 1000)) + 2
        family = socket.AF_INET6 if ":" in args.host else socket.AF_INET
        reserve = socket.socket(family, socket.SOCK_STREAM)
        reserve.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        reserve.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        try:
            reserve.bind((args.host, args.port))
        except OSError as e:
            raise CapkitError("cannot bind %s:%d: %s" % (args.host, args.port, e))
        port = reserve.getsockname()[1]
        cfg = {
            "identity": common.identity.to_dict(), "run_dir": common.run_dir, "loop": common.loop,
            "tier_table": common.tier_table_path, "origin_ns": common.origin_ns,
            "shm_dir": common.shm_dir, "snapshot_interval_ns": common.snapshot_interval_ns,
            "cpus": common.cpus, "inject": [s.to_text() for s in common.inject_specs],
            "stop_grace_s": common.stop_grace_s, "host": args.host, "port": port,
            "family": int(family), "workers": args.workers, "chunk_bytes": args.chunk_bytes,
            "pacing": args.pacing, "idle_close_s": args.idle_close_s,
            "lag_period_us": args.lag_period_us, "max_records": cap, "lag_records": lag_cap,
        }
        from .sink import sink_entry  # noqa: PLC0415
        runner = ParentRunner(common, args.workers, sink_entry, cfg, SINK_SLOTS, "nullsse")

        def on_ready(infos):
            write_port_file(common.port_file, common.identity, args.host, port)
            write_ready_file(common.ready_file, common.identity, runner.versions,
                             affinity=runner.affinity, children=runner.children(),
                             info={"workers": infos, "tool": "nullsse", "port": port,
                                   "tier_table_sha256": table.sha256,
                                   "max_records_per_worker": cap, "lag_records": lag_cap,
                                   "threads": sorted({n for i in infos
                                                      for n in i.get("aux_threads", [])})})
            reserve.close()

        return runner.run(on_ready)
    except CapkitError as e:
        print("capload.nullsse: %s" % e, file=sys.stderr)
        return e.exit_code
    finally:
        if reserve is not None:
            reserve.close()


if __name__ == "__main__":
    sys.exit(main())

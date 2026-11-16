# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``python -m capload``: command line (INTERFACES.md 11) and the parent process."""

import argparse
import os
import sys

from capcommon.cli import add_common_args, common_from_args
from capcommon.exitcodes import CapkitError, ConfigError
from capcommon.hashing import sha256_file
from capcommon.lifecycle import write_ready_file

from . import LOADGEN_SLOTS
from ._proc import ParentRunner
from .httpwire import ENDPOINTS


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capload",
                                description="open-loop multi-process HTTP SSE load generator")
    add_common_args(p, loop=True, tiers=True, snapshot=True, ready=True, port=False)
    tg = p.add_mutually_exclusive_group(required=True)
    tg.add_argument("--target", help="HOST:PORT")
    tg.add_argument("--target-port-file", help="port file of the target (INTERFACES 2.3)")
    p.add_argument("--endpoint", required=True, choices=ENDPOINTS)
    p.add_argument("--model", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--schedule", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--conn-policy", required=True, choices=("keepalive", "new-per-request"))
    p.add_argument("--pool-pick", required=True, choices=("roundrobin", "random"))
    p.add_argument("--pool-seed", type=int, required=True)
    p.add_argument("--pool-reset", required=True, choices=("per-tier", "never"))
    p.add_argument("--recv-mode", required=True, choices=("count", "full"))
    p.add_argument("--chunk-sample-mod", type=int, required=True)
    p.add_argument("--fionread-every-ms", type=int, required=True,
                   help="housekeeping tick: FIONREAD sampling and registered-timeout sweep")
    p.add_argument("--lag-period-us", type=int, required=True)
    p.add_argument("--max-records-per-worker", type=int, required=True)
    o = p.add_argument_group("optional (capload/README.md)")
    o.add_argument("--host-header", default=None,
                   help="Host header value (default: the target host:port)")
    o.add_argument("--mode", choices=("open", "wave"), default="open",
                   help="open = absolute schedule (default); wave = closed-loop waves")
    o.add_argument("--wave-size", type=int, default=None, help="requests per wave (wave mode)")
    o.add_argument("--wave-delays-ns", default=None,
                   help="comma list, per-slot send delay from the wave start (wave mode)")
    o.add_argument("--wave-gap-ns", type=int, default=0,
                   help="pause between the end of a wave and the next wave (wave mode)")
    o.add_argument("--slow-read-pause-us", type=int, default=0,
                   help="positive control: pause reading this long after every read")
    return p


def config_from_args(args, common):
    if args.workers < 1:
        raise ConfigError("--workers must be >= 1")
    if common.cpus is not None and len(common.cpus) < args.workers:
        raise ConfigError("--cpus lists %d cpus for %d workers" % (len(common.cpus), args.workers))
    for name in ("chunk_sample_mod", "fionread_every_ms", "lag_period_us"):
        if getattr(args, name) < 1:
            raise ConfigError("--%s must be >= 1" % name.replace("_", "-"))
    if args.max_records_per_worker < 0:
        raise ConfigError("--max-records-per-worker must be >= 0")
    if args.slow_read_pause_us < 0 or args.wave_gap_ns < 0:
        raise ConfigError("--slow-read-pause-us / --wave-gap-ns must be >= 0")
    host_header = args.host_header
    if args.target:
        host, sep, port = args.target.rpartition(":")
        if not sep or not host or not (port.isascii() and port.isdigit()) \
                or not 0 < int(port) < 65536:
            raise ConfigError("--target must be HOST:PORT, got %r" % args.target)
        host_header = host_header or args.target
    delays = None
    if args.mode == "wave":
        if not args.wave_size or args.wave_size < 1:
            raise ConfigError("--mode wave needs --wave-size >= 1")
        if args.wave_delays_ns:
            try:
                delays = [int(x) for x in args.wave_delays_ns.split(",")]
            except ValueError:
                raise ConfigError("--wave-delays-ns must be a comma list of integers")
            if len(delays) != args.wave_size or min(delays) < 0:
                raise ConfigError("--wave-delays-ns needs %d values >= 0" % args.wave_size)
    elif args.wave_size is not None or args.wave_delays_ns or args.wave_gap_ns:
        raise ConfigError("--wave-* options need --mode wave")
    for path, what in ((args.schedule, "--schedule"), (args.prompts, "--prompts")):
        if not os.path.isfile(path):
            raise ConfigError("%s %s does not exist" % (what, path))
    table = common.load_tier_table()  # validate now (exit 3), workers load it again
    return {
        "identity": common.identity.to_dict(), "run_dir": common.run_dir, "loop": common.loop,
        "tier_table": common.tier_table_path, "tier_table_sha256": table.sha256,
        "origin_ns": common.origin_ns, "shm_dir": common.shm_dir,
        "snapshot_interval_ns": common.snapshot_interval_ns, "cpus": common.cpus,
        "inject": [s.to_text() for s in common.inject_specs], "stop_grace_s": common.stop_grace_s,
        "target": args.target, "target_port_file": args.target_port_file,
        "host_header": host_header or "localhost", "endpoint": args.endpoint, "model": args.model,
        "workers": args.workers, "schedule": args.schedule,
        "schedule_sha256": sha256_file(args.schedule), "prompts": args.prompts,
        "prompts_sha256": sha256_file(args.prompts), "conn_policy": args.conn_policy,
        "pool_pick": args.pool_pick, "pool_seed": args.pool_seed, "pool_reset": args.pool_reset,
        "recv_mode": args.recv_mode, "chunk_sample_mod": args.chunk_sample_mod,
        "fionread_every_ms": args.fionread_every_ms, "lag_period_us": args.lag_period_us,
        "max_records": args.max_records_per_worker, "mode": args.mode,
        "wave_size": args.wave_size, "wave_delays_ns": delays, "wave_gap_ns": args.wave_gap_ns,
        "slow_read_pause_us": args.slow_read_pause_us,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        common = common_from_args(args)
        if not common.shm_dir:
            raise ConfigError("--shm-dir is required")
        cfg = config_from_args(args, common)
        from .loadgen import worker_entry  # noqa: PLC0415
        runner = ParentRunner(common, args.workers, worker_entry, cfg, LOADGEN_SLOTS, "capload")

        def on_ready(infos):
            write_ready_file(common.ready_file, common.identity, runner.versions,
                             affinity=runner.affinity, children=runner.children(),
                             info={"workers": infos, "tool": "capload",
                                   "schedule_sha256": cfg["schedule_sha256"],
                                   "tier_table_sha256": cfg["tier_table_sha256"],
                                   "prompts_sha256": cfg["prompts_sha256"],
                                   "threads": sorted({n for i in infos
                                                      for n in i.get("aux_threads", [])})})

        return runner.run(on_ready)
    except CapkitError as e:
        print("capload: %s" % e, file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())

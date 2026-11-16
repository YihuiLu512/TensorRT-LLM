# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common command-line arguments (INTERFACES.md section 2).

    parser = argparse.ArgumentParser(...)
    add_common_args(parser, loop=True, tiers=True, snapshot=True, ready=True, port=False)
    args = parser.parse_args()
    common = common_from_args(args)   # validated; raises ConfigError (exit 3)
"""

import os
import re

from .aio import LOOP_CHOICES
from .exitcodes import ConfigError
from .identity import Identity
from .inject import parse_inject_spec, check_unique_sites, InjectSpecError

_CPU_RE = re.compile(r"^\d+(-\d+)?$")


def parse_cpu_list(text):
    """'3,5,7-9' -> [3, 5, 7, 8, 9] (order preserved, duplicates rejected)."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if not _CPU_RE.match(part):
            raise ConfigError("bad cpu list item %r in %r" % (part, text))
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            if b < a:
                raise ConfigError("bad cpu range %r" % part)
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    if len(set(out)) != len(out):
        raise ConfigError("duplicate cpus in %r" % text)
    return out


def format_cpu_list(cpus):
    return ",".join(str(c) for c in cpus)


def add_identity_args(parser):
    g = parser.add_argument_group("identity (INTERFACES.md 2.1; all required)")
    g.add_argument("--experiment", required=True)
    g.add_argument("--arm", required=True)
    g.add_argument("--launch-seq", type=int, required=True)
    g.add_argument("--job-id", required=True)
    g.add_argument("--restart-count", type=int, required=True)
    g.add_argument("--register-sha", required=True)
    g.add_argument("--role", required=True)
    g.add_argument("--worker", type=int, default=0,
                   help="worker index within the role (set by the parent/orchestrator)")
    return g


def add_common_args(parser, loop=False, tiers=False, snapshot=False, ready=False, port=False,
                    cpus=True, inject=True, stop=True):
    """Register the INTERFACES.md section 2 arguments that this tool needs (all required)."""
    add_identity_args(parser)
    g = parser.add_argument_group("common (INTERFACES.md 2.2)")
    g.add_argument("--run-dir", required=True, help="launch tree root <T>")
    if loop:
        g.add_argument("--loop", required=True, choices=LOOP_CHOICES)
    if tiers:
        g.add_argument("--tier-table", required=True)
        g.add_argument("--origin-ns", type=int, required=True,
                       help="monotonic_ns of the launch origin (rel 0)")
    if snapshot:
        g.add_argument("--shm-dir", required=True)
        g.add_argument("--snapshot-interval-ms", type=int, required=True)
    if ready:
        g.add_argument("--ready-file", required=True)
    if port:
        g.add_argument("--port-file", required=True)
    if cpus:
        g.add_argument("--cpus", required=True, type=parse_cpu_list,
                       help="logical cpus, e.g. 3,5,7-9 (worker k uses item k)")
    if inject:
        g.add_argument("--inject", action="append", default=[], metavar="SPEC",
                       help="launch-level injection, e.g. site=recv.chunk,mode=busy,delta_ns=20000")
    if stop:
        g.add_argument("--stop-grace-s", type=float, required=True)
    return g


class Common:
    """Validated common arguments."""

    def __init__(self, identity, run_dir, loop=None, tier_table_path=None, origin_ns=None,
                 shm_dir=None, snapshot_interval_ns=None, ready_file=None, port_file=None,
                 cpus=None, inject_specs=(), stop_grace_s=None):
        self.identity = identity
        self.run_dir = run_dir
        self.loop = loop
        self.tier_table_path = tier_table_path
        self.origin_ns = origin_ns
        self.shm_dir = shm_dir
        self.snapshot_interval_ns = snapshot_interval_ns
        self.ready_file = ready_file
        self.port_file = port_file
        self.cpus = cpus
        self.inject_specs = list(inject_specs)
        self.stop_grace_s = stop_grace_s

    def load_tier_table(self):
        from .tiers import TierTable  # noqa: PLC0415
        return TierTable.load(self.tier_table_path)


def common_from_args(args):
    ident = Identity(experiment=args.experiment, arm=args.arm, launch_seq=args.launch_seq,
                     job_id=args.job_id, restart_count=args.restart_count,
                     register_sha=args.register_sha, role=args.role, worker=args.worker,
                     pid=os.getpid())
    if not os.path.isdir(args.run_dir):
        raise ConfigError("--run-dir %s is not a directory" % args.run_dir)
    snap_ns = None
    if getattr(args, "snapshot_interval_ms", None) is not None:
        if args.snapshot_interval_ms <= 0:
            raise ConfigError("--snapshot-interval-ms must be > 0")
        snap_ns = args.snapshot_interval_ms * 1_000_000
    specs = []
    try:
        for text in getattr(args, "inject", []) or []:
            specs.append(parse_inject_spec(text))
        check_unique_sites(specs)
    except InjectSpecError as e:
        raise ConfigError(str(e))
    grace = getattr(args, "stop_grace_s", None)
    if grace is not None and grace < 0:
        raise ConfigError("--stop-grace-s must be >= 0")
    tt = getattr(args, "tier_table", None)
    if tt is not None and not os.path.isfile(tt):
        raise ConfigError("--tier-table %s does not exist" % tt)
    return Common(identity=ident, run_dir=args.run_dir, loop=getattr(args, "loop", None),
                  tier_table_path=tt, origin_ns=getattr(args, "origin_ns", None),
                  shm_dir=getattr(args, "shm_dir", None), snapshot_interval_ns=snap_ns,
                  ready_file=getattr(args, "ready_file", None),
                  port_file=getattr(args, "port_file", None), cpus=getattr(args, "cpus", None),
                  inject_specs=specs, stop_grace_s=grace)

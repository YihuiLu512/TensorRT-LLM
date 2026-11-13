# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory guard read-back (RESOURCES.md 5 item 4; INTERFACES.md 5.5 memguard.json).

For each pid: /proc/<pid>/cgroup -> the memory cgroup path -> memory.max (v2) or
memory.limit_in_bytes (v1) at that level and every ancestor up to the mount root. The effective
guard is the smallest finite limit on the path. "max" (v2) and the v1 "unlimited" sentinel
(>= 2**62) are not guards. A pid passes only with at least one finite limit read from a file.

Inside a container the cgroup path in /proc/<pid>/cgroup may be relative to the namespace root,
which is what /sys/fs/cgroup shows; both the full path and the namespace root are tried and the
file actually read is recorded.

    python -m capenv.memguard <identity> --out PATH (--pids LIST | --roles-file PATH)
        exit 0 PASS, 6 RED / NOT_RUN (no pid to check)
"""

import argparse
import json
import os
import sys

from capcommon import exitcodes
from capcommon.cli import add_identity_args, parse_cpu_list
from capcommon.exitcodes import ConfigError
from capcommon.identity import Identity
from capcommon.jsonio import write_json_stamped

from . import procfs, roles as roles_mod

SCHEMA = "capkit.memguard/1"
UNLIMITED_V1 = 1 << 62


def _candidates(rel_path, mount):
    """Directories from the cgroup's own dir up to the mount root (deepest first)."""
    rel = rel_path.strip("/")
    parts = rel.split("/") if rel else []
    out = []
    for i in range(len(parts), -1, -1):
        out.append(os.path.join(mount, *parts[:i]))
    return out


def _read_limit(path):
    try:
        raw = procfs.read_text(path).strip()
    except OSError as e:
        return None, "%s: %s" % (type(e).__name__, e.strerror or e)
    if raw == "max":
        return None, "max (unlimited)"
    try:
        v = int(raw)
    except ValueError:
        return None, "unparsable %r" % raw
    if v >= UNLIMITED_V1:
        return None, "unlimited sentinel %d" % v
    return v, None


def read_pid(pid, proc_root=procfs.PROC, cgroup_root="/sys/fs/cgroup"):
    rec = {"pid": pid, "cgroup": None, "version": None, "levels": [], "effective_bytes": None,
           "reason": None}
    try:
        lines = procfs.cgroup_lines(pid, proc_root)
    except OSError as e:
        rec["reason"] = "cannot read /proc/%d/cgroup: %s" % (pid, e)
        return rec
    rec["cgroup"] = ["%s:%s:%s" % ln for ln in lines]
    v1 = [ln for ln in lines if "memory" in ln[1].split(",")]
    v2 = [ln for ln in lines if ln[0] == "0" and ln[1] == ""]
    if v1:
        rec["version"] = 1
        rel, mount, fname = v1[0][2], os.path.join(cgroup_root, "memory"), \
            "memory.limit_in_bytes"
    elif v2:
        rec["version"] = 2
        rel, mount, fname = v2[0][2], cgroup_root, "memory.max"
    else:
        rec["reason"] = "no memory controller line in /proc/%d/cgroup" % pid
        return rec
    tried = set()
    for base in _candidates(rel, mount) + [mount]:
        f = os.path.join(base, fname)
        if f in tried:
            continue
        tried.add(f)
        if not os.path.exists(f):
            continue
        val, why = _read_limit(f)
        rec["levels"].append({"file": f, "bytes": val, "note": why})
    finite = [lv["bytes"] for lv in rec["levels"] if lv["bytes"] is not None]
    if finite:
        rec["effective_bytes"] = min(finite)
    elif not rec["levels"]:
        rec["reason"] = "no %s file found under %s for %s" % (fname, mount, rel)
    else:
        rec["reason"] = "no finite limit on the cgroup path"
    return rec


def check(pids, proc_root=procfs.PROC, cgroup_root="/sys/fs/cgroup"):
    recs = [read_pid(p, proc_root, cgroup_root) for p in pids]
    red = [r for r in recs if r["effective_bytes"] is None]
    if not recs:
        verdict = "NOT_RUN"
    elif red:
        verdict = "RED"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "expected_pids": len(pids), "checked": len(recs),
            "red_pids": [r["pid"] for r in red], "pids": recs}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="capenv.memguard")
    add_identity_args(ap)
    ap.add_argument("--out", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pids", type=parse_cpu_list, help="comma list of pids")
    src.add_argument("--roles-file", help="check every pid registered in roles.json")
    args = ap.parse_args(argv)
    try:
        ident = Identity(experiment=args.experiment, arm=args.arm, launch_seq=args.launch_seq,
                         job_id=args.job_id, restart_count=args.restart_count,
                         register_sha=args.register_sha, role=args.role, worker=args.worker,
                         pid=os.getpid())
        if args.roles_file:
            pids = [p["pid"] for p in roles_mod.load(args.roles_file).procs]
            source = "roles.json"
        else:
            pids, source = args.pids, "--pids"
        payload = check(pids)
        payload["source_of_expected"] = source
        write_json_stamped(args.out, SCHEMA, ident, payload)
    except ConfigError as e:
        print("capenv.memguard: %s" % e, file=sys.stderr)
        return exitcodes.CONFIG
    print(json.dumps({"verdict": payload["verdict"], "red_pids": payload["red_pids"]}))
    return exitcodes.OK if payload["verdict"] == "PASS" else exitcodes.ENV_INVALID


if __name__ == "__main__":
    sys.exit(main())

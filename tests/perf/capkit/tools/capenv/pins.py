# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin table generator and read-back verifier (design 2.5 D1/D3/D4; INTERFACES.md 5.5.1).

generate: one logical CPU per physical core (the lowest id of the core; its hyper-thread
siblings stay idle), role core sets disjoint, every role on one NUMA node; roles without an
explicit node all go to one "primary" node. If the demand does not fit, exit 3 and list the gap
(never relax D1/D3, never spill a role across nodes).

verify: read back every thread of every registered process (roles.json) and compare its
Cpus_allowed_list and last-run CPU with the table. Expected count = registered processes whose
role is in the table; 0 -> NOT_RUN. Exit 0 PASS, 6 RED or NOT_RUN.

    python -m capenv.pins generate <identity> --demand FILE --out PATH --source-sha SHA
        [--lscpu-file PATH] [--avoid-cpus LIST] [--only-affinity] [--primary-numa N]
    python -m capenv.pins verify <identity> --pin-table PATH --roles-file PATH --out PATH
"""

import argparse
import json
import os
import sys

from capcommon import exitcodes
from capcommon.cli import add_identity_args, parse_cpu_list
from capcommon.exitcodes import ConfigError
from capcommon.hashing import canonical_json_sha256, is_sha256_hex
from capcommon.identity import Identity
from capcommon.jsonio import read_json, read_json_stamped, write_json_stamped

from . import procfs, roles as roles_mod, topology

PIN_SCHEMA = "capkit.pin_table/1"
READBACK_SCHEMA = "capkit.pin_readback/1"
RULES = "one-logical-per-physical; roles disjoint; one NUMA node per role; siblings idle"


class PinGap(ConfigError):
    """The demand does not fit the topology (exit 3); .gap lists what is missing."""

    def __init__(self, msg, gap):
        super().__init__(msg)
        self.gap = gap


def validate_demand(demand):
    if not isinstance(demand, list) or not demand:
        raise ConfigError("demand must be a non-empty list")
    seen = set()
    for d in demand:
        if not isinstance(d, dict) or set(d) - {"role", "cpus_needed", "numa"} or \
                "role" not in d or "cpus_needed" not in d:
            raise ConfigError("demand entry needs role, cpus_needed[, numa]: %r" % (d,))
        if d["role"] in seen:
            raise ConfigError("role %r appears twice in demand" % d["role"])
        seen.add(d["role"])
        n = d["cpus_needed"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ConfigError("cpus_needed must be an int >= 1 for %r" % d["role"])
        numa = d.get("numa")
        if numa is not None and (not isinstance(numa, int) or isinstance(numa, bool)):
            raise ConfigError("numa must be an int or null for %r" % d["role"])
    return demand


def generate(topo, demand, avoid_cpus=(), primary_numa=None, source_sha=None):
    """Return the pin table payload (dict) or raise PinGap."""
    validate_demand(demand)
    avoid = set(avoid_cpus)
    known = {c["cpu"] for c in topo.cpus}
    unknown = sorted(avoid - known)
    if unknown:
        raise ConfigError("--avoid-cpus lists CPUs not in the topology: %s" % unknown)
    by_node = topo.cores_by_node()
    free = {}
    avoided_cores = []
    for node, cores in by_node.items():
        ok = []
        for c in cores:
            if avoid & set([c["rep"]] + c["siblings"]):
                avoided_cores.append(c)
            else:
                ok.append(c)
        free[node] = ok
    if primary_numa is None:
        explicit = [d["numa"] for d in demand if d.get("numa") is not None]
        if explicit:
            primary_numa = min(set(explicit), key=lambda n: (-explicit.count(n), n))
        else:
            # lowest-numbered node that holds the whole demand; if none does, the largest node
            # (the gap check below then reports the shortfall)
            total = sum(d["cpus_needed"] for d in demand)
            order = sorted(free, key=lambda n: (n is None, n))
            fits = [n for n in order if len(free[n]) >= total]
            primary_numa = fits[0] if fits else min(order, key=lambda n: -len(free[n]))
    if primary_numa not in free:
        raise ConfigError("primary NUMA node %r not in topology nodes %s"
                          % (primary_numa, sorted(map(str, free))))
    need_by_node = {}
    for d in demand:
        node = d.get("numa")
        node = primary_numa if node is None else node
        if node not in free:
            raise ConfigError("role %r asks for NUMA node %r, not in topology" % (d["role"], node))
        need_by_node[node] = need_by_node.get(node, 0) + d["cpus_needed"]
    gap = []
    for node, need in sorted(need_by_node.items(), key=lambda kv: (kv[0] is None, kv[0])):
        have = len(free[node])
        if need > have:
            gap.append({"numa_node": node, "physical_cores_needed": need,
                        "physical_cores_free": have, "short_by": need - have})
    if gap:
        raise PinGap("pin demand does not fit: %s" % json.dumps(gap), gap)
    cursor = {n: 0 for n in free}
    roles_out = {}
    used_cores = []
    for d in demand:
        node = primary_numa if d.get("numa") is None else d["numa"]
        take = free[node][cursor[node]:cursor[node] + d["cpus_needed"]]
        cursor[node] += d["cpus_needed"]
        used_cores.extend(take)
        roles_out[d["role"]] = {"cpus": [c["rep"] for c in take], "numa_node": node,
                                "numa_requested": d.get("numa")}
    used_reps = {c["rep"] for c in used_cores}
    payload = {
        "source_sha": source_sha,
        "rules": RULES,
        "primary_numa": primary_numa,
        "avoid_cpus": sorted(avoid),
        "demand": demand,
        "siblings_left_idle": [[c["rep"]] + c["siblings"] for c in used_cores if c["siblings"]],
        "roles": roles_out,
        "unassigned": sorted(c["rep"] for cs in free.values() for c in cs
                             if c["rep"] not in used_reps),
        "avoided_cores": [[c["rep"]] + c["siblings"] for c in avoided_cores],
        "topology": topo.summary(),
    }
    check_table(payload)
    payload["table_sha256"] = canonical_json_sha256(payload["roles"])
    return payload


def check_table(payload):
    """Structural invariants of a pin table: disjoint role sets, no sibling of a used CPU used."""
    owner = {}
    for role, ent in payload["roles"].items():
        for c in ent["cpus"]:
            if c in owner:
                raise ConfigError("cpu %d assigned to both %s and %s" % (c, owner[c], role))
            owner[c] = role
    for group in payload.get("siblings_left_idle", []):
        for sib in group[1:]:
            if sib in owner:
                raise ConfigError("sibling cpu %d of %d is assigned to %s"
                                  % (sib, group[0], owner[sib]))
    return owner


def load_table(path):
    _ident, doc = read_json_stamped(path, PIN_SCHEMA)
    check_table(doc)
    return doc


def verify(table, registry, proc_root=procfs.PROC):
    """Read back affinity of every thread of every registered process. Returns the payload."""
    role_cpus = {r: set(e["cpus"]) for r, e in table["roles"].items()}
    owner = {}
    for r, cs in role_cpus.items():
        for c in cs:
            owner[c] = r
    procs = registry.procs if registry is not None else []
    expected = [p for p in procs if p["role"] in role_cpus]
    unpinned = sorted({p["role"] for p in procs if p["role"] not in role_cpus})
    roles_without_procs = sorted(set(role_cpus) - {p["role"] for p in procs})
    rows, red = [], []
    for p in expected:
        pid, role = p["pid"], p["role"]
        st = procfs.starttime_ticks(pid, proc_root)
        if st is None or (p.get("starttime_ticks") is not None and st != p["starttime_ticks"]):
            red.append({"pid": pid, "role": role, "why": "process gone or pid reused",
                        "starttime_now": st, "starttime_registered": p.get("starttime_ticks")})
            continue
        try:
            tids = procfs.list_tids(pid, proc_root)
        except procfs.ProcGone:
            red.append({"pid": pid, "role": role, "why": "process gone"})
            continue
        for tid in tids:
            try:
                t = procfs.read_thread(pid, tid, proc_root)
            except (procfs.ProcGone, OSError, ValueError):
                continue  # thread exited between listdir and read
            allowed = set(procfs.parse_cpu_list_text(t["cpus_allowed"]))
            outside = sorted(allowed - role_cpus[role])
            foreign = sorted(c for c in allowed if c in owner and owner[c] != role)
            last_ok = t["last_cpu"] in role_cpus[role]
            row = {"pid": pid, "tid": tid, "role": role, "comm": t["comm"],
                   "cpus_allowed": t["cpus_allowed"], "last_cpu": t["last_cpu"],
                   "outside_role": outside, "other_roles_cpus": foreign,
                   "last_cpu_in_role": last_ok}
            rows.append(row)
            if outside or foreign or not last_ok:
                red.append({"pid": pid, "tid": tid, "role": role,
                            "why": "affinity outside role cpus" if outside else
                            ("last cpu outside role cpus" if not last_ok else
                             "affinity intersects other roles")})
    for r in unpinned:
        red.append({"role": r, "why": "registered role has no pin table entry"})
    for r in roles_without_procs:
        red.append({"role": r, "why": "pin table role has no registered process"})
    if not expected:
        verdict = "NOT_RUN"
    elif red:
        verdict = "RED"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "expected_procs": len(expected),
            "source_of_expected": "roles.json procs whose role is in pin_table.json",
            "checked_threads": len(rows), "red": red, "threads": rows,
            "pin_table_sha256": table.get("table_sha256"),
            "roles_generation": registry.generation if registry is not None else None}


def _identity(args):
    return Identity(experiment=args.experiment, arm=args.arm, launch_seq=args.launch_seq,
                    job_id=args.job_id, restart_count=args.restart_count,
                    register_sha=args.register_sha, role=args.role, worker=args.worker,
                    pid=os.getpid())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="capenv.pins", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    add_identity_args(g)
    g.add_argument("--demand", required=True, help="JSON list [{role, cpus_needed, numa}]")
    g.add_argument("--out", required=True)
    g.add_argument("--source-sha", required=True, help="hw_profile sha256 this table is for")
    g.add_argument("--lscpu-file", help="read `lscpu -p=CPU,CORE,SOCKET,NODE` text from a file "
                                        "instead of running lscpu")
    g.add_argument("--avoid-cpus", type=parse_cpu_list, default=[],
                   help="CPUs whose whole physical core must not be assigned")
    g.add_argument("--only-affinity", action="store_true",
                   help="also avoid every CPU outside this process's affinity mask (the job "
                        "step's cpuset when run inside the job)")
    g.add_argument("--primary-numa", type=int, default=None)
    v = sub.add_parser("verify")
    add_identity_args(v)
    v.add_argument("--pin-table", required=True)
    v.add_argument("--roles-file", required=True)
    v.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        ident = _identity(args)
        if args.cmd == "generate":
            if not is_sha256_hex(args.source_sha):
                raise ConfigError("--source-sha must be 64 lowercase hex")
            if args.lscpu_file:
                with open(args.lscpu_file, "r", encoding="utf-8") as f:
                    text = f.read()
                topo, src = topology.parse_lscpu_p(text), {"file": args.lscpu_file}
            else:
                topo, rec = topology.probe()
                src = {"cmd": rec["cmd"], "stdout_sha256": rec["stdout_sha256"]}
            demand = read_json(args.demand)
            avoid = list(args.avoid_cpus)
            if args.only_affinity:
                allowed = os.sched_getaffinity(0)
                avoid += [c["cpu"] for c in topo.cpus
                          if c["cpu"] not in allowed and c["cpu"] not in avoid]
            try:
                payload = generate(topo, demand, avoid, args.primary_numa,
                                   args.source_sha)
            except PinGap as e:
                print(json.dumps({"verdict": "DOES_NOT_FIT", "gap": e.gap}), file=sys.stderr)
                return exitcodes.CONFIG
            payload["topology_source"] = src
            write_json_stamped(args.out, PIN_SCHEMA, ident, payload)
            print(json.dumps({"verdict": "OK", "out": args.out,
                              "table_sha256": payload["table_sha256"]}))
            return exitcodes.OK
        table = load_table(args.pin_table)
        reg = roles_mod.load(args.roles_file)
        payload = verify(table, reg)
        write_json_stamped(args.out, READBACK_SCHEMA, ident, payload)
        print(json.dumps({"verdict": payload["verdict"], "red": len(payload["red"]),
                          "expected_procs": payload["expected_procs"]}))
        return exitcodes.OK if payload["verdict"] == "PASS" else exitcodes.ENV_INVALID
    except ConfigError as e:
        print("capenv.pins: %s" % e, file=sys.stderr)
        return exitcodes.CONFIG


if __name__ == "__main__":
    sys.exit(main())

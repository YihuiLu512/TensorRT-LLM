# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU topology from `lscpu -p=CPU,CORE,SOCKET,NODE` (parseable output, stable across versions).

A physical core is keyed by (socket, core). Its representative logical CPU is the lowest CPU id
in it; the others are its hyper-thread siblings. Nothing about the machine is assumed: socket,
core, node counts all come from the text.
"""

from capcommon.exitcodes import ConfigError

from . import runcmd

LSCPU_ARGV = ["lscpu", "-p=CPU,CORE,SOCKET,NODE"]


class Topology:
    def __init__(self, cpus):
        """cpus: list of {"cpu", "core", "socket", "node"} (node may be None)."""
        if not cpus:
            raise ConfigError("topology has no CPUs")
        ids = [c["cpu"] for c in cpus]
        if len(set(ids)) != len(ids):
            raise ConfigError("duplicate CPU ids in topology")
        self.cpus = sorted(cpus, key=lambda c: c["cpu"])
        cores = {}
        for c in self.cpus:
            cores.setdefault((c["socket"], c["core"]), []).append(c)
        self.cores = []
        for key, members in cores.items():
            nodes = {m["node"] for m in members}
            if len(nodes) != 1:
                raise ConfigError("core %s spans NUMA nodes %s" % (key, sorted(map(str, nodes))))
            ids = sorted(m["cpu"] for m in members)
            self.cores.append({"socket": key[0], "core": key[1], "node": members[0]["node"],
                               "rep": ids[0], "siblings": ids[1:]})
        self.cores.sort(key=lambda c: c["rep"])

    @property
    def sockets(self):
        return sorted({c["socket"] for c in self.cpus})

    @property
    def nodes(self):
        return sorted({c["node"] for c in self.cpus}, key=lambda n: (n is None, n))

    def threads_per_core(self):
        counts = sorted({1 + len(c["siblings"]) for c in self.cores})
        return counts[0] if len(counts) == 1 else counts

    def cores_by_node(self):
        out = {}
        for c in self.cores:
            out.setdefault(c["node"], []).append(c)
        return out

    def summary(self):
        return {"logical_cpus": len(self.cpus), "physical_cores": len(self.cores),
                "sockets": len(self.sockets), "nodes": [n for n in self.nodes],
                "threads_per_core": self.threads_per_core(),
                "sibling_table": [[c["rep"]] + c["siblings"] for c in self.cores]}

    def cpu_node(self, cpu):
        for c in self.cpus:
            if c["cpu"] == cpu:
                return c["node"]
        raise KeyError(cpu)


def parse_lscpu_p(text):
    """Parse `lscpu -p=CPU,CORE,SOCKET,NODE` text into a Topology.

    Comment lines start with '#'. An empty NODE field (no NUMA support) becomes None.
    """
    cpus = []
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 4:
            raise ConfigError("lscpu -p line %d has %d fields, expected 4: %r"
                              % (ln, len(parts), line))
        try:
            cpu, core, sock = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            raise ConfigError("lscpu -p line %d: non-integer CPU/CORE/SOCKET: %r" % (ln, line))
        node = int(parts[3]) if parts[3].strip() != "" else None
        cpus.append({"cpu": cpu, "core": core, "socket": sock, "node": node})
    return Topology(cpus)


def probe(timeout_s=30.0):
    """Run lscpu on this machine. Returns (Topology, run record)."""
    rec = runcmd.run(LSCPU_ARGV, timeout_s, env=_c_locale_env())
    if not runcmd.ok(rec):
        raise ConfigError("lscpu -p failed: %s" % runcmd.why(rec))
    return parse_lscpu_p(rec["stdout"]), rec


def _c_locale_env():
    import os  # noqa: PLC0415
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    return env

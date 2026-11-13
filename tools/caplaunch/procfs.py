# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""/proc readers used by the orchestrator (process identity, descendants, threads, markers).

Every reader takes an explicit `proc_root` so tests can point it at a fake tree; the
orchestrator passes "/proc". Nothing here infers a role from a command line.
"""

import json
import os
import re

from capcommon.hashing import canonical_json_sha256

# Thread identification rules (INTERFACES.md 5.3; C13 first half). Evaluated in order; the first
# match wins. "comm" is a regex on /proc/<pid>/task/<tid>/comm. A thread no rule identifies keeps
# identified_as = null and the judge counts it red. Python threads other than the main thread are
# identified only from names the process itself declared (ready-file info.threads as a
# {tid: name} map, or snapshot meta "threads"), never guessed.
THREAD_RULES = (
    {"class": "grpc_core", "comm": r"^grpc_global_tim", "identified_as": "grpc timer"},
    {"class": "grpc_core", "comm": r"^default-executo", "identified_as": "grpc default executor"},
    {"class": "grpc_core", "comm": r"^resolver-execut", "identified_as": "grpc resolver executor"},
    {"class": "grpc_core", "comm": r"^grpcpp_sync_ser", "identified_as": "grpc sync server"},
    {"class": "grpc_core", "comm": r"^event_engine$", "identified_as": "grpc event engine pool"},
    {"class": "grpc_core", "comm": r"^lifeguard$", "identified_as": "grpc event engine lifeguard"},
    {"class": "grpc_core", "comm": r"^grpc_", "identified_as": "grpc core (other)"},
    {"class": "zmq_io", "comm": r"^ZMQbg/IO/", "identified_as": "zmq io"},
    {"class": "zmq_io", "comm": r"^ZMQbg/Reaper", "identified_as": "zmq reaper"},
)
RULES_SHA = canonical_json_sha256(list(THREAD_RULES))
_RULES_RE = [(r, re.compile(r["comm"])) for r in THREAD_RULES]


def _read(path, binary=False):
    with open(path, "rb" if binary else "r", encoding=None if binary else "utf-8",
              errors=None if binary else "replace") as f:
        return f.read()


def stat_fields(pid, proc_root="/proc"):
    """(comm, fields-after-comm list) of /proc/<pid>/stat, or None if the process is gone."""
    try:
        raw = _read(os.path.join(proc_root, str(pid), "stat"))
    except OSError:
        return None
    lp, rp = raw.find("("), raw.rfind(")")
    return raw[lp + 1:rp], raw[rp + 2:].split()


def starttime_ticks(pid, proc_root="/proc"):
    """Field 22 of /proc/<pid>/stat (starttime) or None."""
    sf = stat_fields(pid, proc_root)
    return None if sf is None else int(sf[1][19])


def ppid_of(pid, proc_root="/proc"):
    sf = stat_fields(pid, proc_root)
    return None if sf is None else int(sf[1][1])


def is_zombie(pid, proc_root="/proc"):
    sf = stat_fields(pid, proc_root)
    return sf is not None and sf[1][0] == "Z"


def alive(pid, starttime, proc_root="/proc"):
    """True if a non-zombie process with this pid AND starttime exists (pid reuse safe)."""
    sf = stat_fields(pid, proc_root)
    if sf is None or sf[1][0] == "Z":
        return False
    return starttime is None or int(sf[1][19]) == starttime


def cmdline(pid, proc_root="/proc"):
    try:
        raw = _read(os.path.join(proc_root, str(pid), "cmdline"), binary=True)
    except OSError:
        return None
    return [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]


def all_pids(proc_root="/proc"):
    try:
        names = os.listdir(proc_root)
    except OSError:
        return []
    return sorted(int(n) for n in names if n.isdigit())


def descendants(root_pids, proc_root="/proc"):
    """All live pids whose ancestor chain reaches one of root_pids (roots excluded)."""
    parent = {}
    for p in all_pids(proc_root):
        pp = ppid_of(p, proc_root)
        if pp is not None:
            parent[p] = pp
    roots = set(root_pids)
    out = set()
    for p in parent:
        seen = set()
        q = parent.get(p)
        while q is not None and q not in seen and q > 0:
            if q in roots:
                out.add(p)
                break
            seen.add(q)
            q = parent.get(q)
    return sorted(out - roots)


def is_descendant_of(pid, ancestors, proc_root="/proc"):
    seen = set()
    q = ppid_of(pid, proc_root)
    anc = set(ancestors)
    while q is not None and q > 0 and q not in seen:
        if q in anc:
            return True
        seen.add(q)
        q = ppid_of(q, proc_root)
    return False


def environ_has(pid, needle, proc_root="/proc"):
    """True if /proc/<pid>/environ contains the exact entry `needle` ("KEY=value").

    Unreadable environ (other users' processes) -> False.
    """
    try:
        raw = _read(os.path.join(proc_root, str(pid), "environ"), binary=True)
    except OSError:
        return False
    return needle.encode() in raw.split(b"\0")


def marked_pids(needle, proc_root="/proc", exclude=()):
    """Live, non-zombie pids carrying the marker env entry (leftover detection)."""
    ex = set(exclude)
    return [p for p in all_pids(proc_root)
            if p not in ex and not is_zombie(p, proc_root) and environ_has(p, needle, proc_root)]


def status_field(pid, name, tid=None, proc_root="/proc"):
    base = os.path.join(proc_root, str(pid)) if tid is None else \
        os.path.join(proc_root, str(pid), "task", str(tid))
    try:
        for line in _read(os.path.join(base, "status")).splitlines():
            if line.startswith(name + ":"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def cpus_allowed_list(pid, tid=None, proc_root="/proc"):
    return status_field(pid, "Cpus_allowed_list", tid=tid, proc_root=proc_root)


def expand_cpu_list(text):
    """'0-2,5' -> [0, 1, 2, 5] (kernel Cpus_allowed_list format)."""
    out = []
    if not text:
        return out
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def thread_ids(pid, proc_root="/proc"):
    try:
        return sorted(int(t) for t in os.listdir(os.path.join(proc_root, str(pid), "task"))
                      if t.isdigit())
    except OSError:
        return []


def thread_comm(pid, tid, proc_root="/proc"):
    try:
        return _read(os.path.join(proc_root, str(pid), "task", str(tid), "comm")).strip()
    except OSError:
        return None


def identify_thread(pid, tid, comm, main_comm, declared):
    """(class, identified_as, source) for one thread.

    declared: {tid(int): python thread name} the process declared about itself, or {}.
    """
    if tid in declared:
        return "python", declared[tid], "declared"
    if tid == pid:
        return "python", "main", "tid==pid"
    for rule, rx in _RULES_RE:
        if comm is not None and rx.search(comm):
            return rule["class"], rule["identified_as"], "comm"
    if comm is not None and comm == main_comm:
        return "python", None, "comm==main"
    return "other", None, "comm"


def thread_inventory(pid, declared=None, proc_root="/proc"):
    """[{tid, comm, py_name, class, identified_as, source, cpus_allowed}] for every thread."""
    declared = {int(k): v for k, v in (declared or {}).items()}
    main_comm = thread_comm(pid, pid, proc_root)
    out = []
    for tid in thread_ids(pid, proc_root):
        comm = thread_comm(pid, tid, proc_root)
        cls, ident, src = identify_thread(pid, tid, comm, main_comm, declared)
        out.append({"tid": tid, "comm": comm, "py_name": declared.get(tid), "class": cls,
                    "identified_as": ident, "source": src,
                    "cpus_allowed": cpus_allowed_list(pid, tid, proc_root)})
    return out


def nic_counters(sys_root="/sys"):
    """{iface: {rx_bytes, tx_bytes}} or {"unreadable": reason}."""
    base = os.path.join(sys_root, "class", "net")
    try:
        ifaces = sorted(os.listdir(base))
    except OSError as e:
        return {"unreadable": "%s: %s" % (base, e)}
    out = {}
    for i in ifaces:
        d = {}
        for k in ("rx_bytes", "tx_bytes"):
            try:
                d[k] = int(_read(os.path.join(base, i, "statistics", k)).strip())
            except (OSError, ValueError) as e:
                d[k] = {"unreadable": str(e)}
        out[i] = d
    return out


def declared_threads_from_ready(ready_doc):
    """{tid: name} from a ready file's info.threads when it is a mapping; else {}."""
    info = (ready_doc or {}).get("info") or {}
    th = info.get("threads")
    if isinstance(th, dict):
        try:
            return {int(k): str(v) for k, v in th.items()}
        except (TypeError, ValueError):
            return {}
    return {}


def dumps_small(obj):
    return json.dumps(obj, sort_keys=True)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pid -> role registry procs/roles.json (INTERFACES.md 5.3), read-only side.

Process identity comes ONLY from this registry (pid + starttime_ticks); nothing here looks at
command lines or process names.
"""

import os

from capcommon.exitcodes import ConfigError
from capcommon.jsonio import read_json_stamped

ROLES_SCHEMA = "capkit.roles/1"
_PROC_KEYS = ("pid", "starttime_ticks", "role", "worker")


class RolesRegistry:
    def __init__(self, generation, procs, unattributed, identity=None, path=None, mtime_ns=None):
        self.generation = generation
        self.procs = procs
        self.unattributed = unattributed
        self.identity = identity
        self.path = path
        self.mtime_ns = mtime_ns

    def by_pid(self):
        return {p["pid"]: p for p in self.procs}

    def roles(self):
        return sorted({p["role"] for p in self.procs})


def validate_doc(doc):
    gen = doc.get("generation")
    if not isinstance(gen, int) or isinstance(gen, bool) or gen < 0:
        raise ConfigError("roles.json generation must be an int >= 0")
    procs = doc.get("procs")
    if not isinstance(procs, list):
        raise ConfigError("roles.json procs must be a list")
    seen = set()
    for p in procs:
        if not isinstance(p, dict) or any(k not in p for k in _PROC_KEYS):
            raise ConfigError("roles.json proc entry needs %s: %r" % (_PROC_KEYS, p))
        if not isinstance(p["pid"], int) or p["pid"] <= 0:
            raise ConfigError("roles.json bad pid %r" % (p["pid"],))
        if not isinstance(p["role"], str) or not p["role"]:
            raise ConfigError("roles.json bad role %r" % (p["role"],))
        if p["pid"] in seen:
            raise ConfigError("roles.json lists pid %d twice" % p["pid"])
        seen.add(p["pid"])
    un = doc.get("unattributed", [])
    if not isinstance(un, list):
        raise ConfigError("roles.json unattributed must be a list")
    return gen, procs, un


def load(path):
    st = os.stat(path)
    ident, doc = read_json_stamped(path, ROLES_SCHEMA)
    gen, procs, un = validate_doc(doc)
    return RolesRegistry(gen, procs, un, identity=ident, path=path, mtime_ns=st.st_mtime_ns)


class RolesWatcher:
    """Re-reads roles.json when its mtime changes; keeps the last good registry on a bad read
    (a half-written file cannot happen with atomic replace, but a malformed one is recorded)."""

    def __init__(self, path):
        self.path = path
        self.current = None
        self.errors = []
        self.reloads = 0

    def poll(self):
        """Return (registry or None, changed: bool)."""
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            if self.current is None:
                return None, False
            self.errors.append("roles.json vanished")
            return self.current, False
        if self.current is not None and st.st_mtime_ns == self.current.mtime_ns:
            return self.current, False
        try:
            reg = load(self.path)
        except (ConfigError, ValueError, OSError) as e:
            self.errors.append(str(e)[:300])
            return self.current, False
        changed = self.current is None or reg.generation != self.current.generation
        if self.current is not None and reg.generation < self.current.generation:
            self.errors.append("roles.json generation went backwards %d -> %d"
                               % (self.current.generation, reg.generation))
        self.current = reg
        if changed:
            self.reloads += 1
        return reg, changed

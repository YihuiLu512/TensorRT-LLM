# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identity stamped into every capkit output file (INTERFACES.md section 2.1).

Identity is always passed explicitly by the caller; nothing here infers it from paths or names.
"""

import dataclasses
import os
import re

from .exitcodes import ConfigError

EXPERIMENT_RE = re.compile(r"^E\d{1,2}[A-Za-z0-9-]*$")
ARM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROLE_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
TIER_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")

# The first six fields must be identical across every file of one launch tree.
LAUNCH_KEY_FIELDS = ("experiment", "arm", "launch_seq", "job_id", "restart_count", "register_sha")
ALL_FIELDS = LAUNCH_KEY_FIELDS + ("role", "worker", "pid", "tier_seq", "tier_label")
IDENTITY_SCHEMA = "capkit.identity/1"


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


@dataclasses.dataclass(frozen=True)
class Identity:
    experiment: str
    arm: str
    launch_seq: int
    job_id: str
    restart_count: int
    register_sha: str
    role: object  # str, or None only inside probe snapshots of not-yet-attributed product processes
    worker: int = 0
    pid: object = None  # int or None
    tier_seq: object = None  # int or None
    tier_label: object = None  # str or None

    def __post_init__(self):
        validate_identity_fields(dataclasses.asdict(self), allow_null_role=True)

    def to_dict(self):
        return dataclasses.asdict(self)

    def with_tier(self, tier_seq, tier_label):
        return dataclasses.replace(self, tier_seq=tier_seq, tier_label=tier_label)

    def with_worker(self, worker):
        return dataclasses.replace(self, worker=worker)

    def with_pid(self, pid=None):
        return dataclasses.replace(self, pid=os.getpid() if pid is None else pid)

    def with_role(self, role):
        return dataclasses.replace(self, role=role)

    def launch_key(self):
        return tuple(getattr(self, f) for f in LAUNCH_KEY_FIELDS)

    @classmethod
    def from_dict(cls, d, allow_null_role=False):
        if not isinstance(d, dict):
            raise ConfigError("identity must be an object")
        missing = [f for f in ALL_FIELDS if f not in d]
        extra = [k for k in d if k not in ALL_FIELDS]
        if missing or extra:
            raise ConfigError("identity fields missing=%s extra=%s" % (missing, extra))
        validate_identity_fields(d, allow_null_role=allow_null_role)
        return cls(**{f: d[f] for f in ALL_FIELDS})


def validate_identity_fields(d, allow_null_role=False):
    def bad(field, why):
        raise ConfigError("identity.%s invalid: %r (%s)" % (field, d.get(field), why))

    if not isinstance(d.get("experiment"), str) or not EXPERIMENT_RE.match(d["experiment"]):
        bad("experiment", EXPERIMENT_RE.pattern)
    if not isinstance(d.get("arm"), str) or not ARM_RE.match(d["arm"]):
        bad("arm", ARM_RE.pattern)
    if not _is_int(d.get("launch_seq")) or d["launch_seq"] < 0:
        bad("launch_seq", "int >= 0")
    if not isinstance(d.get("job_id"), str) or not JOB_ID_RE.match(d["job_id"]):
        bad("job_id", JOB_ID_RE.pattern)
    if not _is_int(d.get("restart_count")) or d["restart_count"] < 0:
        bad("restart_count", "int >= 0")
    if not isinstance(d.get("register_sha"), str) or not SHA256_RE.match(d["register_sha"]):
        bad("register_sha", "64 lowercase hex")
    role = d.get("role")
    if role is None:
        if not allow_null_role:
            bad("role", "required")
    elif not isinstance(role, str) or not ROLE_RE.match(role):
        bad("role", ROLE_RE.pattern)
    if not _is_int(d.get("worker")) or d["worker"] < 0:
        bad("worker", "int >= 0")
    pid = d.get("pid")
    if pid is not None and (not _is_int(pid) or pid <= 0):
        bad("pid", "positive int or null")
    ts, tl = d.get("tier_seq"), d.get("tier_label")
    if (ts is None) != (tl is None):
        bad("tier_seq", "tier_seq and tier_label must both be set or both be null")
    if ts is not None:
        if not _is_int(ts) or ts < -2:
            bad("tier_seq", "int >= -2 (-1 = pre, -2 = post)")
        if not isinstance(tl, str) or not TIER_LABEL_RE.match(tl):
            bad("tier_label", TIER_LABEL_RE.pattern)


def check_same_launch(identities):
    """Raise ConfigError unless all identities share the six launch-key fields."""
    keys = set()
    for ident in identities:
        d = ident.to_dict() if isinstance(ident, Identity) else ident
        keys.add(tuple(d.get(f) for f in LAUNCH_KEY_FIELDS))
    if len(keys) > 1:
        raise ConfigError("identity launch keys differ: %s" % sorted(map(str, keys)))
    return keys.pop() if keys else None

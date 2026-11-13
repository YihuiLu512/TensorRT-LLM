# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch tree paths and STATE (INTERFACES.md sections 5.1, 5.7).

Paths are built from explicit values; nothing parses identity out of a path.
"""

import os

from .clock import wall_iso
from .exitcodes import ConfigError
from .identity import ARM_RE, JOB_ID_RE, ROLE_RE
from .tiers import POST_LABEL, POST_SEQ, PRE_LABEL, PRE_SEQ

STATE_VALUES = ("VALID", "VOID", "NOT_RUN", "ENV_INVALID")

# Launch-level conclusions from best to worst (INTERFACES.md 5.7). Every writer that records a
# conclusion on top of an earlier one combines them with worst_state: a later record can make a
# launch worse, never better.
STATE_SEVERITY = ("VALID", "NOT_RUN", "VOID", "ENV_INVALID")
assert sorted(STATE_SEVERITY) == sorted(STATE_VALUES)


def state_rank(status):
    """Position of `status` in STATE_SEVERITY (0 = VALID); ConfigError for anything else."""
    try:
        return STATE_SEVERITY.index(status)
    except ValueError:
        raise ConfigError("not a STATE value: %r" % (status,)) from None


def worst_state(statuses):
    """The worst of one or more STATE values; ConfigError on an empty input or a non-STATE value."""
    statuses = list(statuses)
    if not statuses:
        raise ConfigError("worst_state of nothing")
    return max(statuses, key=state_rank)


def tree_dir_name(job_id, restart_count, launch_seq, arm):
    """runs/J<job>-r<restart>-L<seq>_<arm>/ (DATA-AND-ANALYSIS 3). For humans only."""
    if not JOB_ID_RE.match(str(job_id)) or not ARM_RE.match(arm):
        raise ConfigError("bad job_id/arm for a tree name")
    return "J%s-r%d-L%d_%s" % (job_id, restart_count, launch_seq, arm)


def job_dir_name(job_id, restart_count):
    """runs/J<job>-r<restart>/ : job-level files (job.json, env/slurm_view.json). For humans."""
    if not JOB_ID_RE.match(str(job_id)) or not isinstance(restart_count, int) \
            or restart_count < 0:
        raise ConfigError("bad job_id/restart_count for a job dir name")
    return "J%s-r%d" % (job_id, restart_count)


def launch_shm_subdir(job_shm_root, restart_count, launch_seq):
    """Per-launch snapshot directory <job shm root>/r<restart>-L<seq> (INTERFACES.md 2.2), so a
    launch never sees snapshot files left behind by an earlier launch of the same job."""
    if not isinstance(restart_count, int) or not isinstance(launch_seq, int) \
            or restart_count < 0 or launch_seq < 0:
        raise ConfigError("restart_count and launch_seq must be ints >= 0")
    return os.path.join(job_shm_root, "r%d-L%d" % (restart_count, launch_seq))


def tier_dir_name(seq, label):
    if seq == PRE_SEQ:
        return PRE_LABEL
    if seq == POST_SEQ:
        return POST_LABEL
    if not isinstance(seq, int) or seq < 0:
        raise ConfigError("bad tier seq %r" % (seq,))
    return "%03d_%s" % (seq, label)


def worker_tag(role, worker):
    if not ROLE_RE.match(role):
        raise ConfigError("bad role %r" % (role,))
    return "%s.w%d" % (role, worker)


class RunTree:
    """Path helper for one launch tree <T>."""

    def __init__(self, root):
        self.root = root
        self.raw = os.path.join(root, "raw")

    # top-level
    @property
    def launch_json(self):
        return os.path.join(self.raw, "launch.json")

    @property
    def launch_end_json(self):
        return os.path.join(self.raw, "launch_end.json")

    @property
    def dims_json(self):
        return os.path.join(self.raw, "dims.json")

    @property
    def state_file(self):
        return os.path.join(self.root, "STATE")

    @property
    def control_dir(self):
        return os.path.join(self.root, "control")

    @property
    def stop_json(self):
        return os.path.join(self.control_dir, "stop.json")

    @property
    def judge_dir(self):
        return os.path.join(self.root, "judge")

    # raw subdirs
    def env(self, *parts):
        return os.path.join(self.raw, "env", *parts)

    def procs(self, *parts):
        return os.path.join(self.raw, "procs", *parts)

    @property
    def roles_json(self):
        return self.procs("roles.json")

    def port_file(self, role):
        return self.procs("ports", "%s.json" % role)

    def ready_file(self, role, worker=None):
        name = role if worker is None else worker_tag(role, worker)
        return self.procs("ready", "%s.json" % name)

    def threads_file(self, role, pid):
        return self.procs("threads", "%s.%d.json" % (role, pid))

    def tier_dir(self, seq, label):
        return os.path.join(self.raw, "tiers", tier_dir_name(seq, label))

    def tier_file(self, seq, label, role, worker, name, ext):
        """raw/tiers/<seq>_<label>/<role>.w<k>.<name>.<ext>"""
        return os.path.join(self.tier_dir(seq, label),
                            "%s.%s.%s" % (worker_tag(role, worker), name, ext))

    def sidecar(self, *parts):
        return os.path.join(self.raw, "sidecar", *parts)

    def canary(self, *parts):
        return os.path.join(self.raw, "canary", *parts)

    def log_file(self, role, worker, pid, stream):
        if stream not in ("out", "err"):
            raise ConfigError("stream must be out or err")
        return os.path.join(self.raw, "logs", "%s.%d.%s" % (worker_tag(role, worker), pid, stream))

    def make_skeleton(self):
        """Create the directory skeleton (idempotent)."""
        for d in (self.raw, self.env(), self.env("versions"), self.procs(), self.procs("ports"),
                  self.procs("ready"), self.procs("threads"), os.path.join(self.raw, "tiers"),
                  self.sidecar(), self.canary(), os.path.join(self.raw, "logs"),
                  self.control_dir):
            os.makedirs(d, exist_ok=True)
        return self


def append_state(tree_root, status, reason, by):
    """Append one STATE line: <ISO UTC>\t<status>\t<by>\t<reason>."""
    if status not in STATE_VALUES:
        raise ConfigError("STATE must be one of %s" % (STATE_VALUES,))
    for v in (reason, by):
        if "\t" in v or "\n" in v:
            raise ConfigError("STATE fields must not contain tabs/newlines")
    line = "%s\t%s\t%s\t%s\n" % (wall_iso(), status, by, reason)
    with open(os.path.join(tree_root, "STATE"), "a", encoding="utf-8") as f:
        f.write(line)
    return line


def read_state(tree_root):
    """All STATE lines as (iso, status, by, reason); [] if missing. Current = last."""
    p = os.path.join(tree_root, "STATE")
    if not os.path.exists(p):
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 3)
            if len(parts) != 4 or parts[1] not in STATE_VALUES:
                raise ConfigError("%s: malformed STATE line %r" % (p, line))
            out.append(tuple(parts))
    return out

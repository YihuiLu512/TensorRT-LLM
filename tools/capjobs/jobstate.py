# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job end criterion (plan RESOURCES section 4 "等待作业"):

ENDED  <=> `squeue -h -j J` exits 0 with empty output AND `sacct -j J` reports a terminal state
for the job allocation. Any command error, a missing sacct row, or a non-terminal state means the
answer is NOT_ENDED or UNKNOWN, never ENDED. Log strings are never used.
"""

import re

from capcommon.exitcodes import ConfigError

from .remote import run_remote

ENDED = "ENDED"
NOT_ENDED = "NOT_ENDED"
UNKNOWN = "UNKNOWN"

# sacct terminal states (State column; "CANCELLED by 123" is reduced to its first word).
TERMINAL = ("BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
            "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT")
NON_TERMINAL = ("PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING",
                "CONFIGURING", "REQUEUE_FED", "REQUEUE_HOLD", "SPECIAL_EXIT", "STOPPED",
                "SIGNALING", "STAGE_OUT", "RESV_DEL_HOLD", "REVOKED", "PREEMPTED_REQUEUE")
JOB_ID_RE = re.compile(r"^[0-9]+$")


class JobStateError(ConfigError):
    pass


def squeue_argv(job_id):
    return ["squeue", "-h", "-j", str(job_id), "-o", "%i %T"]


def sacct_argv(job_id):
    return ["sacct", "-j", str(job_id), "-X", "-n", "-P", "-o", "JobID,State,ExitCode"]


def decide(job_id, squeue_rc, squeue_out, sacct_rc, sacct_out):
    """Pure decision from the two raw results. Returns (state, reason, sacct_state)."""
    job_id = str(job_id)
    if squeue_rc is None or squeue_rc != 0:
        return UNKNOWN, "squeue rc=%r" % (squeue_rc,), None
    if squeue_out.strip():
        return NOT_ENDED, "squeue still lists the job", None
    if sacct_rc is None or sacct_rc != 0:
        return UNKNOWN, "sacct rc=%r" % (sacct_rc,), None
    rows = [ln.split("|") for ln in sacct_out.splitlines() if ln.strip()]
    mine = [r for r in rows if r and r[0] == job_id]
    if len(mine) != 1 or len(mine[0]) < 2:
        return UNKNOWN, "sacct has %d allocation rows for job %s" % (len(mine), job_id), None
    state = mine[0][1].split()[0] if mine[0][1].split() else ""
    if state in TERMINAL:
        return ENDED, "squeue empty and sacct state %s" % state, state
    if state in NON_TERMINAL:
        return NOT_ENDED, "sacct state %s is not terminal" % state, state
    return UNKNOWN, "unrecognised sacct state %r" % state, state


def query(runner, login_host, job_id):
    if not JOB_ID_RE.match(str(job_id)):
        raise JobStateError("job id must be numeric for a cluster job: %r" % (job_id,))
    sq = run_remote(runner, login_host, squeue_argv(job_id))
    sa = run_remote(runner, login_host, sacct_argv(job_id))
    state, reason, sacct_state = decide(job_id, sq["rc"], sq["stdout"], sa["rc"], sa["stdout"])
    return {"job_id": str(job_id), "state": state, "reason": reason, "sacct_state": sacct_state,
            "squeue": sq, "sacct": sa}

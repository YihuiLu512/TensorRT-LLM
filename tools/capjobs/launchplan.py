# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch plan (schema capkit.launch_plan/1): what one job runs, launch by launch.

{"schema": "capkit.launch_plan/1", "experiment": "E0", "register_sha": "<64hex>",
 "job_arm": "job", "margin_s": 120,
 "launches": [{"launch_seq": 0, "arm": "nullsse-a", "budget_s": 900,
               "argv": ["{PY}", "-m", "caplaunch", "--scenario", "{SNAP}/configs/...", ...]}]}

The identity arguments (--experiment --arm --launch-seq --job-id --restart-count --register-sha
--role --run-dir) are appended by capjobs.launches from the plan and the Slurm environment; a plan
argv that contains any of them is rejected, so identity has exactly one source. "job_arm" is the
arm label written into job-level files (env/*.json) which belong to no single launch.
"""

import json
import re

from capcommon.exitcodes import ConfigError
from capcommon.identity import ARM_RE, EXPERIMENT_RE, SHA256_RE

SCHEMA = "capkit.launch_plan/1"
PLACEHOLDERS = ("PY", "SNAP", "JOB_DIR", "SHM_ROOT", "LOCAL_ROOT", "JOB_ID", "RESTART_COUNT")
RESERVED_FLAGS = ("--experiment", "--arm", "--launch-seq", "--job-id", "--restart-count",
                  "--register-sha", "--role", "--run-dir")
_PH_RE = re.compile(r"\{([^{}]*)\}")


class PlanError(ConfigError):
    pass


def _int(v, lo):
    return isinstance(v, int) and not isinstance(v, bool) and v >= lo


def validate(doc):
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise PlanError("launch plan must have schema %s" % SCHEMA)
    extra = set(doc) - {"schema", "experiment", "register_sha", "job_arm", "margin_s", "launches"}
    if extra:
        raise PlanError("unknown launch plan keys %s" % sorted(extra))
    if not isinstance(doc.get("experiment"), str) or not EXPERIMENT_RE.match(doc["experiment"]):
        raise PlanError("bad experiment")
    if not isinstance(doc.get("register_sha"), str) or not SHA256_RE.match(doc["register_sha"]):
        raise PlanError("register_sha must be 64 lowercase hex")
    if not isinstance(doc.get("job_arm"), str) or not ARM_RE.match(doc["job_arm"]):
        raise PlanError("bad job_arm")
    if not _int(doc.get("margin_s"), 0):
        raise PlanError("margin_s must be an int >= 0")
    launches = doc.get("launches")
    if not isinstance(launches, list) or not launches:
        raise PlanError("launches must be a non-empty list")
    seqs = set()
    for i, ln in enumerate(launches):
        if not isinstance(ln, dict) or set(ln) != {"launch_seq", "arm", "budget_s", "argv"}:
            raise PlanError("launch %d must have exactly launch_seq, arm, budget_s, argv" % i)
        if not _int(ln["launch_seq"], 0) or ln["launch_seq"] in seqs:
            raise PlanError("launch %d: launch_seq must be a unique int >= 0" % i)
        seqs.add(ln["launch_seq"])
        if not isinstance(ln["arm"], str) or not ARM_RE.match(ln["arm"]):
            raise PlanError("launch %d: bad arm" % i)
        if not _int(ln["budget_s"], 1):
            raise PlanError("launch %d: budget_s must be an int > 0" % i)
        argv = ln["argv"]
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a
                                                              for a in argv):
            raise PlanError("launch %d: argv must be a non-empty list of strings" % i)
        for a in argv:
            flag = a.split("=", 1)[0]
            if flag in RESERVED_FLAGS:
                raise PlanError("launch %d: argv must not contain %s (appended by capjobs)"
                                % (i, flag))
            for name in _PH_RE.findall(a):
                if name not in PLACEHOLDERS:
                    raise PlanError("launch %d: unknown placeholder {%s}" % (i, name))
    return doc


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return validate(json.load(f))


def arms(doc):
    """Arms in first-appearance order."""
    out = []
    for ln in doc["launches"]:
        if ln["arm"] not in out:
            out.append(ln["arm"])
    return out


def substitute(arg, values):
    """Replace {NAME} tokens; every NAME must be known and present in values."""
    def rep(m):
        name = m.group(1)
        if name not in PLACEHOLDERS or name not in values:
            raise PlanError("no value for placeholder {%s}" % name)
        return str(values[name])
    return _PH_RE.sub(rep, arg)

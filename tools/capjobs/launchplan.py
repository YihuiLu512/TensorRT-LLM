# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch plan (schema capkit.launch_plan/1): what one job runs, launch by launch.

{"schema": "capkit.launch_plan/1", "experiment": "E0", "register_sha": "<64hex>",
 "job_arm": "job", "margin_s": 120, "kill_grace_s": 120,
 "launches": [{"launch_seq": 0, "arm": "nullsse-a", "budget_s": 900,
               "argv": ["{PY}", "-m", "caplaunch", "--scenario", "{SNAP}/configs/...", ...]}]}

The identity arguments (--experiment --arm --launch-seq --job-id --restart-count --register-sha
--role --run-dir), --overlay-manifest, --nvidia-smi and --job-env-dir are appended by
capjobs.launches from the plan, the job's snapshot, job.env, the job dir and the Slurm
environment; a plan argv that contains any of them is rejected, so each has exactly one source.
"job_arm" is the arm label written into job-level files (env/*.json) which belong to no single
launch. "kill_grace_s" is how long a launch that overran budget_s gets after
SIGTERM before its whole process tree is killed. check_kill_grace() refuses a plan whose
kill_grace_s is shorter than any launch's worst-case orchestrator stop (worst_stop_s), computed
from that launch's --scenario (a {SNAP}/ path) and --kill-after-s.
"""

import json
import os
import re

from capcommon.exitcodes import ConfigError
from capcommon.identity import ARM_RE, EXPERIMENT_RE, SHA256_RE

SCHEMA = "capkit.launch_plan/1"
PLACEHOLDERS = ("PY", "SNAP", "JOB_DIR", "SHM_ROOT", "LOCAL_ROOT", "JOB_ID", "RESTART_COUNT")
RESERVED_FLAGS = ("--experiment", "--arm", "--launch-seq", "--job-id", "--restart-count",
                  "--register-sha", "--role", "--run-dir", "--overlay-manifest", "--nvidia-smi",
                  "--job-env-dir")
_PH_RE = re.compile(r"\{([^{}]*)\}")


class PlanError(ConfigError):
    pass


def _int(v, lo):
    return isinstance(v, int) and not isinstance(v, bool) and v >= lo


def validate(doc):
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise PlanError("launch plan must have schema %s" % SCHEMA)
    keys = {"schema", "experiment", "register_sha", "job_arm", "margin_s", "kill_grace_s",
            "launches"}
    if set(doc) != keys:
        raise PlanError("launch plan keys must be exactly %s; unknown %s, missing %s"
                        % (sorted(keys), sorted(set(doc) - keys), sorted(keys - set(doc))))
    if not isinstance(doc.get("experiment"), str) or not EXPERIMENT_RE.match(doc["experiment"]):
        raise PlanError("bad experiment")
    if not isinstance(doc.get("register_sha"), str) or not SHA256_RE.match(doc["register_sha"]):
        raise PlanError("register_sha must be 64 lowercase hex")
    if not isinstance(doc.get("job_arm"), str) or not ARM_RE.match(doc["job_arm"]):
        raise PlanError("bad job_arm")
    if not _int(doc.get("margin_s"), 0):
        raise PlanError("margin_s must be an int >= 0")
    if not _int(doc.get("kill_grace_s"), 1):
        raise PlanError("kill_grace_s must be an int >= 1")
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


def _flag_value(argv, flag):
    """The value of `flag X` or `flag=X` in argv; None when absent; error when repeated."""
    vals = []
    for i, a in enumerate(argv):
        if a == flag:
            vals.append(argv[i + 1] if i + 1 < len(argv) else None)
        elif a.startswith(flag + "="):
            vals.append(a.split("=", 1)[1])
    if len(vals) > 1 or (vals and vals[0] is None):
        raise PlanError("%s must appear once with a value" % flag)
    return vals[0] if vals else None


def worst_stop_s(ln, snap_root):
    """(seconds, scenario relpath): the longest a signalled caplaunch takes to stop on its own:
    every role in stop_order gets stop_grace_s + kill_after_s, then the leftover check sends
    SIGTERM and SIGKILL with up to kill_after_s each."""
    argv = ln["argv"]
    scen, kas = _flag_value(argv, "--scenario"), _flag_value(argv, "--kill-after-s")
    if scen is None or kas is None:
        raise PlanError("launch %d: argv needs --scenario and --kill-after-s (kill_grace_s is "
                        "checked against them)" % ln["launch_seq"])
    if not scen.startswith("{SNAP}/") or len(_PH_RE.findall(scen)) != 1:
        raise PlanError("launch %d: --scenario must be a {SNAP}/ path" % ln["launch_seq"])
    rel = os.path.normpath(scen[len("{SNAP}/"):])
    if rel.startswith(".."):
        raise PlanError("launch %d: --scenario leaves {SNAP}" % ln["launch_seq"])
    try:
        kill_after = float(kas)
    except ValueError:
        raise PlanError("launch %d: bad --kill-after-s %r" % (ln["launch_seq"], kas))
    if not kill_after > 0:
        raise PlanError("launch %d: --kill-after-s must be > 0" % ln["launch_seq"])
    path = os.path.join(snap_root, rel)
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise PlanError("launch %d: scenario %s: %s" % (ln["launch_seq"], path, e))
    grace, order = (doc.get("stop_grace_s"), doc.get("stop_order")) if isinstance(doc, dict) \
        else (None, None)
    if not isinstance(grace, (int, float)) or isinstance(grace, bool) or grace < 0 \
            or not isinstance(order, list) or not order:
        raise PlanError("launch %d: scenario %s needs stop_grace_s >= 0 and a non-empty "
                        "stop_order" % (ln["launch_seq"], path))
    return len(order) * (grace + kill_after) + 2 * kill_after, rel


def check_kill_grace(doc, snap_root, snapshot_files=None):
    """Refuse the plan unless kill_grace_s >= worst_stop_s of every launch; snapshot_files (the
    relpaths the job's snapshot will hold) also requires each scenario to be among them.
    Returns {launch_seq: worst-case seconds}."""
    out = {}
    for ln in doc["launches"]:
        worst, rel = worst_stop_s(ln, snap_root)
        if snapshot_files is not None and rel not in snapshot_files:
            raise PlanError("launch %d: scenario %s is not in the job snapshot"
                            % (ln["launch_seq"], rel))
        if doc["kill_grace_s"] < worst:
            raise PlanError("kill_grace_s %d < launch %d's worst-case orchestrator stop %.1f s "
                            "(roles x (stop_grace_s + kill_after_s) + 2 x kill_after_s)"
                            % (doc["kill_grace_s"], ln["launch_seq"], worst))
        out[ln["launch_seq"]] = worst
    return out


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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch plan (schema capkit.launch_plan/1): what one job runs, launch by launch.

{"schema": "capkit.launch_plan/1", "experiment": "E0", "register_sha": "<64hex>",
 "job_arm": "job", "margin_s": 120, "kill_grace_s": 120,
 "launches": [{"launch_seq": 0, "arm": "nullsse-a", "budget_s": 900,
               "argv": ["{PY}", "-m", "caplaunch", "--workdir", "/abs/W",
                        "--scenario", "02_e0/E0/register/x.scenario.json", ...]}]}

A launch may carry "body": "script" (default "caplaunch"; capcommon.policy.SCRIPT_BODY_EXPERIMENTS
names the experiments allowed to use it, checked by capjobs.submit). A script launch runs
argv = [{PY} | bash | /bin/bash, {SNAP}/<file in the job's code snapshot>, args...] as is: no
scenario, no pin table, no job inputs and no appended flags; capjobs.launches passes the
identity, the launch tree (CAPKIT_RUN_DIR) and the job files in the environment instead
(launches.script_env). The script owns everything in between; the job keeps its gate, budget,
process-tree kill, copy-back, verification and STATE line. Plans do not mix the two bodies.
The identity arguments (--experiment --arm --launch-seq --job-id --restart-count --register-sha
--role --run-dir), --overlay-manifest, --nvidia-smi, --job-env-dir, --pin-table and --job-input
are appended by capjobs.launches from the plan, the job's snapshot, job.env, the job dir, the pin
table and job inputs it generates on the node and the Slurm environment; a plan argv that
contains any of them is rejected, so each has exactly one source.
"job_arm" is the arm label written into job-level files (env/*.json) which belong to no single
launch. "kill_grace_s" is how long a launch that overran budget_s gets after
SIGTERM before its whole process tree is killed. check_kill_grace() refuses a plan whose
kill_grace_s is shorter than any launch's worst-case orchestrator stop (worst_stop_s), computed
from that launch's --scenario and --kill-after-s.
A launch's --scenario is the registered scenario in the work dir (scenario_path): caplaunch and
the judge require it inside --workdir at the register's files.scenario relpath, so both flags are
literal paths (no placeholder) and the scenario is relative to --workdir or an absolute path
inside it.
"""

import json
import os
import re

from capcommon import jobinputs
from capcommon.exitcodes import ConfigError
from capcommon.identity import ARM_RE, EXPERIMENT_RE, SHA256_RE

SCHEMA = "capkit.launch_plan/1"
# Upper bound of one launch's pin-table generation in the job (capjobs.launches.pin_table): part
# of every in-allocation time check (capjobs.launches before each launch, capjobs.submit's budget).
PINS_TIMEOUT_S = 120
# Upper bound of one job-input generator run (capjobs.jobinputs; errata B2B-9), counted per kind
# the launch's scenario declares, in the same time checks.
JOB_INPUT_TIMEOUT_S = 600
PLACEHOLDERS = ("PY", "SNAP", "JOB_DIR", "SHM_ROOT", "LOCAL_ROOT", "JOB_ID", "RESTART_COUNT")
RESERVED_FLAGS = ("--experiment", "--arm", "--launch-seq", "--job-id", "--restart-count",
                  "--register-sha", "--role", "--run-dir", "--overlay-manifest", "--nvidia-smi",
                  "--job-env-dir", "--pin-table", "--job-input")
_PH_RE = re.compile(r"\{([^{}]*)\}")
BODIES = ("caplaunch", "script")
SCRIPT_INTERPRETERS = ("{PY}", "bash", "/bin/bash")
SCRIPT_PREFIX = "{SNAP}/"


def body_of(ln):
    return ln.get("body", "caplaunch")


def script_relpath(ln):
    """The snapshot-relative path of a script launch's file (argv[1] after {SNAP}/)."""
    return ln["argv"][1][len(SCRIPT_PREFIX):]


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
        if not isinstance(ln, dict) or not ({"launch_seq", "arm", "budget_s", "argv"} <= set(ln)
                                            <= {"launch_seq", "arm", "budget_s", "argv",
                                                "body"}):
            raise PlanError("launch %d must have exactly launch_seq, arm, budget_s, argv "
                            "(and optionally body)" % i)
        if body_of(ln) not in BODIES:
            raise PlanError("launch %d: body must be one of %s" % (i, BODIES))
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
        if body_of(ln) == "script":
            rel = argv[1][len(SCRIPT_PREFIX):] if len(argv) > 1 else ""
            if argv[0] not in SCRIPT_INTERPRETERS or len(argv) < 2 \
                    or not argv[1].startswith(SCRIPT_PREFIX) or not rel \
                    or _PH_RE.search(rel) or os.path.isabs(rel) \
                    or ".." in rel.split("/"):
                raise PlanError("launch %d: a script launch's argv must start with one of %s "
                                "then %s<relative path in the code snapshot>"
                                % (i, SCRIPT_INTERPRETERS, SCRIPT_PREFIX))
    if len({body_of(ln) for ln in launches}) > 1:
        raise PlanError("a plan must not mix caplaunch and script launches")
    return doc


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return validate(json.load(f))


def flag_value(argv, flag):
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


def scenario_path(ln):
    """(work dir, scenario path relative to it) of one launch, both from its argv: --workdir is
    an absolute literal path; --scenario is relative to it, or absolute and inside it (realpath,
    as caplaunch computes the relpath it compares with the register)."""
    argv, seq = ln["argv"], ln["launch_seq"]
    wd, scen = flag_value(argv, "--workdir"), flag_value(argv, "--scenario")
    if wd is None or scen is None:
        raise PlanError("launch %d: argv needs --workdir and --scenario" % seq)
    if _PH_RE.search(wd) or _PH_RE.search(scen):
        raise PlanError("launch %d: --workdir and --scenario must be literal paths: the scenario "
                        "is the registered file in the work dir" % seq)
    if not os.path.isabs(wd):
        raise PlanError("launch %d: --workdir must be an absolute path" % seq)
    root = os.path.realpath(wd)
    path = os.path.realpath(scen if os.path.isabs(scen) else os.path.join(wd, scen))
    if os.path.commonpath([path, root]) != root or path == root:
        raise PlanError("launch %d: --scenario %s is not inside --workdir %s" % (seq, scen, wd))
    return wd, os.path.relpath(path, root)


def worst_stop_s(ln):
    """(seconds, scenario relpath): the longest a signalled caplaunch takes to stop on its own:
    every role in stop_order gets stop_grace_s + kill_after_s, then the leftover check sends
    SIGTERM and SIGKILL with up to kill_after_s each."""
    wd, rel = scenario_path(ln)
    kas = flag_value(ln["argv"], "--kill-after-s")
    if kas is None:
        raise PlanError("launch %d: argv needs --kill-after-s (kill_grace_s is checked against "
                        "it)" % ln["launch_seq"])
    try:
        kill_after = float(kas)
    except ValueError:
        raise PlanError("launch %d: bad --kill-after-s %r" % (ln["launch_seq"], kas))
    if not kill_after > 0:
        raise PlanError("launch %d: --kill-after-s must be > 0" % ln["launch_seq"])
    doc = load_scenario(ln)
    grace, order = (doc.get("stop_grace_s"), doc.get("stop_order")) if isinstance(doc, dict) \
        else (None, None)
    if not isinstance(grace, (int, float)) or isinstance(grace, bool) or grace < 0 \
            or not isinstance(order, list) or not order:
        raise PlanError("launch %d: scenario %s needs stop_grace_s >= 0 and a non-empty "
                        "stop_order" % (ln["launch_seq"], rel))
    return len(order) * (grace + kill_after) + 2 * kill_after, rel


def load_scenario(ln):
    """The JSON document at a launch's scenario_path."""
    wd, rel = scenario_path(ln)
    path = os.path.join(os.path.realpath(wd), rel)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise PlanError("launch %d: scenario %s: %s" % (ln["launch_seq"], path, e))


def in_job_bound_s(ln):
    """Seconds one launch may spend in the job before its launcher starts: the pin-table bound
    plus JOB_INPUT_TIMEOUT_S per job-input kind its scenario declares; 0 for a script launch
    (neither is generated)."""
    if body_of(ln) == "script":
        return 0
    return PINS_TIMEOUT_S + JOB_INPUT_TIMEOUT_S * len(jobinputs.kinds_of(load_scenario(ln)))


def check_kill_grace(doc):
    """Refuse the plan unless kill_grace_s >= worst_stop_s of every launch.
    Returns {launch_seq: worst-case seconds}."""
    out = {}
    for ln in doc["launches"]:
        if body_of(ln) == "script":  # no orchestrator: the job's kill_grace_s is the bound
            continue
        worst, _rel = worst_stop_s(ln)
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

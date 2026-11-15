# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario config (`capkit.scenario/1`, INTERFACES.md section 11) and the tool argument table.

The orchestrator builds every command line only from this table (INTERFACES.md 11) plus the
scenario's verbatim extra args; it never guesses arguments from module or role names.
"""

import functools
import importlib.util
import os

from capcommon import jobinputs
from capcommon.exitcodes import ConfigError
from capcommon.expect import validate_reconcile
from capcommon.identity import EXPERIMENT_RE, ROLE_RE
from capcommon.runtree import RunTree
from capcommon.templates import PLACEHOLDER_RE, placeholders

SCHEMA = "capkit.scenario/1"

_CAPPROBE_SITES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "capprobe", "sites.py")


@functools.lru_cache(maxsize=None)
def probe_proc_kinds():
    """capprobe's process kinds (PROC_KINDS of tools/capprobe/sites.py). sites.py is loaded
    alone, by path: importing the capprobe package would parse this process's CAPPROBE_*."""
    try:
        spec = importlib.util.spec_from_file_location("_caplaunch_capprobe_sites",
                                                      _CAPPROBE_SITES)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return frozenset(mod.PROC_KINDS)
    except (OSError, AttributeError) as e:
        raise ConfigError("cannot read capprobe process kinds from %s: %s"
                          % (_CAPPROBE_SITES, e)) from e

# Which INTERFACES.md 11 argument groups each capkit tool takes. identity, --run-dir,
# --stop-grace-s and --cpus ("common") are passed to every capkit tool.
#   loop: --loop; tiers: --tier-table + --origin-ns; origin: --origin-ns only;
#   snapshot: --shm-dir + --snapshot-interval-ms; shm_only: --shm-dir; ready: --ready-file;
#   port: --port-file (capkit JSON port file); workers: --workers W;
#   sidecar / canary: orchestrator-filled observer arguments;
#   pin_table: --pin-table <launch tree>/raw/env/pin_table.json (the orchestrator's copy);
#   single: the tool runs one process only (scenario workers and expected_procs must be 1).
# Not an argument group: role_fixed {role key: value} that validate_role requires of every role
# running the tool (its measure_class must be one of the orchestrator's MONITOR_CLASSES when
# given; asserted in orchestrator.py, which imports this module).
TOOL_TABLE = {
    "capload": {"loop": True, "tiers": True, "snapshot": True, "ready": True, "port": False,
                "workers": True},
    "capload.nullsse": {"loop": True, "tiers": True, "snapshot": True, "ready": True,
                        "port": True, "workers": True},
    "capgrpc.load": {"loop": True, "tiers": True, "snapshot": True, "ready": True, "port": False,
                     "workers": True},
    "capgrpc.nullgrpc": {"loop": True, "tiers": True, "snapshot": True, "ready": True,
                         "port": True, "workers": True},
    "capselfcheck.relay": {"loop": True, "tiers": True, "snapshot": True, "ready": True,
                           "port": True, "workers": False},
    # S-PB (errata B2B-3): takes --workers, which must be 1 (one process, expected_procs 1).
    "capstubs.prebaked_server": {"loop": True, "tiers": True, "snapshot": True, "ready": True,
                                 "port": True, "workers": True, "single": True},
    # --ready-file: INTERFACES 2.2 ("all long-lived processes"); optional in B4's parsers.
    "capenv.sidecar": {"origin": True, "shm_only": True, "sidecar": True, "ready": True},
    "capenv.canary": {"origin": True, "canary": True, "ready": True},
    # C13 positive control / V1-c/d load (INTERFACES errata SB-B1-3): an aux observer role (its
    # spin processes stay unregistered monitor children); one started process, no snapshots.
    "capenv.spinners": {"origin": True, "ready": True, "pin_table": True,
                        "role_fixed": {"kind": "aux", "measure_class": "observer",
                                       "expected_procs": 1, "asyncio": False,
                                       "uses_grpc": False, "snapshots": "none"}},
    # E0(c) HTTP smoke client (this package); identity/--run-dir/--cpus only, no ready file.
    "caplaunch.httpprobe": {},
    # caplaunch.smoke as a role (errata B2B-12): same arguments as httpprobe, no ready file.
    "caplaunch.smoke": {},
}
# Flags the orchestrator owns; a scenario's extra args must not repeat them.
MANAGED_FLAGS = ("--experiment", "--arm", "--launch-seq", "--job-id", "--restart-count",
                 "--register-sha", "--role", "--worker", "--run-dir", "--stop-grace-s", "--cpus",
                 "--loop", "--tier-table", "--origin-ns", "--shm-dir", "--snapshot-interval-ms",
                 "--ready-file", "--port-file", "--workers", "--roles-file", "--out-dir",
                 "--logs-dir", "--cpus-watch", "--out", "--pin-table")

ROLE_KEYS_REQUIRED = ("role", "launcher", "kind", "start_after", "pin_role", "cpus_needed",
                      "numa", "workers", "expected_procs", "measure_class", "asyncio",
                      "uses_grpc", "loop", "probe_tier", "readiness", "backlog_slots",
                      "snapshots", "args", "env", "proc_kinds")
ROLE_KEYS_OPTIONAL = ("module", "command", "port", "keep_numa_affinity_env", "note")
KINDS = ("server", "client", "aux")
LAUNCHERS = ("capkit", "product")
MEASURE_CLASSES = ("load_source", "stub_out_of_domain", "stub_in_domain", "under_test",
                   "observer")
SNAPSHOT_SCOPES = ("none", "all", "workers")
PROBE_TIERS = ("P0", "P1", "P2")
# The capprobe switch that the lag group needs, and the scenario param it is set from.
LAG_PARAM = "lag_period_us"


def product_writes_snapshots(r):
    """Whether a role's processes write snapshots through capprobe: a product role with probe
    tier P2 (a probe group on). P0 / P1 turn no group on, and CAPPROBE_INJECT (the other switch
    that makes capprobe write) reaches no product process: the orchestrator refuses an outer one
    and validate_role refuses CAPPROBE_* in a role's env, so their processes never write a
    snapshot whatever the role's snapshots scope says."""
    return r["launcher"] == "product" and r["probe_tier"] == "P2"
LOOPS = ("asyncio", "uvloop")
READY_KINDS = ("capkit_ready", "plain_port_file", "report_addr", "http_get", "file_exists")
PORT_KINDS = ("plain_port_file", "report_addr", "none")
TOP_KEYS = ("schema", "experiment", "name", "tier_table", "schedule", "prompts", "probe_tier",
            "probe_groups", "stop_grace_s", "stop_order", "reconcile", "roles", "dims",
            "params_required", "note", "job_inputs")
TOP_KEYS_OPTIONAL = ("note", "job_inputs")
DIMS_KEYS = ("tokenizer", "stub", "response_timing_switches", "interpretation", "log_sink")

PH_SIMPLE = ("run_dir", "raw", "nvidia_smi")
PH_ROLE = ("port_file", "addr")  # reference another role (must be in start_after)
PH_OWN = ("own",)  # own:port_file | own:report_addr | own:role
OWN_KEYS = ("port_file", "report_addr", "role", "ready_file")
# {input:X}: absolute path of a registered input, or of a job input (capcommon.jobinputs) the
# scenario's job_inputs section declares (errata B2B-9)
INPUT_KEYS = jobinputs.REGISTERED_INPUTS + tuple(jobinputs.KINDS)


def _err(where, msg):
    raise ConfigError("scenario %s: %s" % (where, msg))


def _check_template(where, s, role_names, start_after, params, job_kinds=()):
    if not isinstance(s, str):
        _err(where, "template items must be strings, got %r" % (s,))
    for name, arg in placeholders(s):
        if name in PH_SIMPLE and arg is None:
            continue
        if name == "param":
            if arg not in params:
                _err(where, "{param:%s} is not listed in params_required" % arg)
            continue
        if name in PH_ROLE:
            if arg not in role_names:
                _err(where, "{%s:%s} names an unknown role" % (name, arg))
            if arg not in start_after:
                _err(where, "{%s:%s} needs %s in start_after" % (name, arg, arg))
            continue
        if name == "own" and arg in OWN_KEYS:
            continue
        if name == "input" and arg in INPUT_KEYS:
            if arg in jobinputs.KINDS and arg not in job_kinds:
                _err(where, "{input:%s} needs %s in the scenario's job_inputs" % (arg, arg))
            continue
        _err(where, "unknown placeholder {%s%s}" % (name, "" if arg is None else ":" + arg))


def _check_argv_items(where, items, role_names, start_after, params, job_kinds=()):
    if not isinstance(items, list):
        _err(where, "must be a list")
    for it in items:
        if isinstance(it, dict):
            if set(it) != {"splice_json", "key"} or not isinstance(it["key"], str):
                _err(where, "dict items must be exactly {splice_json, key}")
            _check_template(where, it["splice_json"], role_names, start_after, params, job_kinds)
        else:
            _check_template(where, it, role_names, start_after, params, job_kinds)


def _check_readiness(where, lst, launcher, tool):
    if not isinstance(lst, list) or not lst:
        _err(where, "readiness must be a non-empty list of criteria")
    kinds = []
    for c in lst:
        if not isinstance(c, dict) or c.get("kind") not in READY_KINDS:
            _err(where, "readiness kind must be one of %s" % (READY_KINDS,))
        k = c["kind"]
        kinds.append(k)
        allowed = {"kind"}
        if k == "http_get":
            allowed |= {"path", "expect_status", "timeout_s"}
            if not isinstance(c.get("path"), str) or not c["path"].startswith("/"):
                _err(where, "http_get needs path starting with /")
            if not isinstance(c.get("expect_status"), int):
                _err(where, "http_get needs an int expect_status")
            if not isinstance(c.get("timeout_s"), (int, float)) or c["timeout_s"] <= 0:
                _err(where, "http_get needs timeout_s > 0")
        elif k == "file_exists":
            allowed |= {"path"}
            if not isinstance(c.get("path"), str) or not c["path"]:
                _err(where, "file_exists needs a path template")
        if set(c) - allowed:
            _err(where, "readiness %s: unknown keys %s" % (k, sorted(set(c) - allowed)))
    if launcher == "capkit" and tool.get("ready") and "capkit_ready" not in kinds:
        _err(where, "a capkit tool with --ready-file must use readiness capkit_ready")
    if "capkit_ready" in kinds and not (launcher == "capkit" and tool.get("ready")):
        _err(where, "capkit_ready is only for capkit tools that take --ready-file")
    return kinds


def validate_role(r, role_names, params, job_kinds=()):
    if not isinstance(r, dict):
        _err("roles", "each role must be an object")
    where = "role %s" % r.get("role")
    missing = [k for k in ROLE_KEYS_REQUIRED if k not in r]
    extra = [k for k in r if k not in ROLE_KEYS_REQUIRED + ROLE_KEYS_OPTIONAL]
    if missing or extra:
        _err(where, "keys missing=%s extra=%s" % (missing, extra))
    if not isinstance(r["role"], str) or not ROLE_RE.match(r["role"]) or r["role"] == "launch":
        _err(where, "bad role name")
    if r["launcher"] not in LAUNCHERS or r["kind"] not in KINDS:
        _err(where, "launcher must be %s and kind %s" % (LAUNCHERS, KINDS))
    if r["measure_class"] not in MEASURE_CLASSES:
        _err(where, "measure_class must be one of %s" % (MEASURE_CLASSES,))
    for k in ("cpus_needed", "workers", "expected_procs"):
        if not isinstance(r[k], int) or isinstance(r[k], bool) or r[k] < 1:
            _err(where, "%s must be an int >= 1" % k)
    if r["expected_procs"] < r["workers"]:
        _err(where, "expected_procs < workers")
    if r["numa"] is not None and (not isinstance(r["numa"], int) or r["numa"] < 0):
        _err(where, "numa must be null or an int >= 0")
    if not isinstance(r["pin_role"], str) or not r["pin_role"]:
        _err(where, "pin_role must be a non-empty string (a key of pin_table.json roles)")
    for k in ("asyncio", "uses_grpc"):
        if not isinstance(r[k], bool):
            _err(where, "%s must be a bool" % k)
    if r["asyncio"] and r["loop"] not in LOOPS:
        _err(where, "an asyncio role needs loop in %s" % (LOOPS,))
    if not r["asyncio"] and r["loop"] is not None:
        _err(where, "a non-asyncio role has loop null")
    if r["snapshots"] not in SNAPSHOT_SCOPES:
        _err(where, "snapshots must be one of %s" % (SNAPSHOT_SCOPES,))
    bs = r["backlog_slots"]
    if not isinstance(bs, list) or any(not isinstance(s, str) or not s.startswith("backlog.")
                                       for s in bs):
        _err(where, "backlog_slots must be a list of backlog.* slot names")
    if bs and r["snapshots"] == "none":
        _err(where, "backlog_slots need snapshots != none")
    if not isinstance(r["start_after"], list) or any(s not in role_names or s == r["role"]
                                                     for s in r["start_after"]):
        _err(where, "start_after must list other existing roles")
    if not isinstance(r["env"], dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                             for k, v in r["env"].items()):
        _err(where, "env must map str -> str")
    # the capprobe switches (CAPPROBE_*) are set by the orchestrator alone (_role_env), from
    # registered scenario fields: a role env entry would bypass that (e.g. CAPPROBE_INJECT making a
    # P0/P1 product process write snapshots nobody expects)
    probe_keys = sorted(k for k in r["env"] if k.startswith("CAPPROBE_"))
    if probe_keys:
        _err(where, "env must not set capprobe switches %s (the orchestrator sets them)"
             % probe_keys)
    # proc_kinds: the capprobe process kinds (capprobe SITES proc_kind) this role runs; the judge
    # decides from them which overlay sites C8 expects, so an unknown name (a typo) would only
    # drop sites from C8. Only product roles run probed code.
    pk = r["proc_kinds"]
    if not isinstance(pk, list) or any(not isinstance(k, str) or not k for k in pk) \
            or len(set(pk)) != len(pk):
        _err(where, "proc_kinds must be a list of distinct non-empty strings")
    if pk and r["launcher"] != "product":
        _err(where, "only product roles run probed code; proc_kinds must be []")
    unknown = sorted(set(pk) - probe_proc_kinds()) if pk else []
    if unknown:
        _err(where, "unknown proc_kinds %s (capprobe PROC_KINDS: %s)"
             % (unknown, sorted(probe_proc_kinds())))
    tool = {}
    if r["launcher"] == "capkit":
        if r.get("module") not in TOOL_TABLE or "command" in r:
            _err(where, "capkit roles need module in %s and no command" % sorted(TOOL_TABLE))
        tool = TOOL_TABLE[r["module"]]
        for k, v in sorted(tool.get("role_fixed", {}).items()):
            if r[k] != v or type(r[k]) is not type(v):
                _err(where, "%s needs %s %r (INTERFACES errata SB-B1-3)" % (r["module"], k, v))
        if r["probe_tier"] is not None:
            _err(where, "capkit tools have probe_tier null")
        if "port" in r:
            _err(where, "capkit roles take their port file from the tool table, not 'port'")
        if r["workers"] > 1 and not tool.get("workers"):
            _err(where, "%s takes no --workers; workers must be 1" % r["module"])
        if tool.get("single") and (r["workers"] != 1 or r["expected_procs"] != 1):
            _err(where, "%s runs one process; workers and expected_procs must be 1"
                 % r["module"])
        if tool.get("snapshot") and r["snapshots"] == "none":
            _err(where, "%s writes snapshots; snapshots must not be none" % r["module"])
        flat = [a for a in r["args"] if isinstance(a, str)]
        clash = [a for a in flat if a.split("=", 1)[0] in MANAGED_FLAGS]
        if clash:
            _err(where, "args repeat orchestrator-managed flags %s" % clash)
    else:
        cmd = r.get("command")
        if "module" in r or not isinstance(cmd, list) or not cmd:
            _err(where, "product roles need a non-empty command list and no module")
        if r["probe_tier"] not in PROBE_TIERS:
            _err(where, "product roles need probe_tier in %s" % (PROBE_TIERS,))
        # A role that writes snapshots (P2) must expect them from every process, its started
        # process (the registered entry) included: with none or workers a probe that stayed off
        # in that process (e.g. its CAPPROBE_ENTRY rule did not match) would leave no missing
        # snapshot for RECORDS to see.
        if product_writes_snapshots(r) and r["snapshots"] != "all":
            _err(where, "probe_tier P2 writes snapshots; snapshots must be all")
        port = r.get("port")
        if not isinstance(port, dict) or port.get("kind") not in PORT_KINDS:
            _err(where, "product roles need port {kind: %s, host}" % (PORT_KINDS,))
        if port["kind"] != "none" and not isinstance(port.get("host"), str):
            _err(where, "port.host must be given explicitly")
        if set(port) - {"kind", "host"}:
            _err(where, "port has unknown keys")
        _check_argv_items(where + " command", cmd, role_names, r["start_after"], params,
                          job_kinds)
    if "keep_numa_affinity_env" in r and not isinstance(r["keep_numa_affinity_env"], bool):
        _err(where, "keep_numa_affinity_env must be a bool")
    _check_argv_items(where + " args", r["args"], role_names, r["start_after"], params,
                      job_kinds)
    kinds = _check_readiness(where, r["readiness"], r["launcher"], tool)
    if r["launcher"] == "product":
        pk = r["port"]["kind"]
        if pk in ("plain_port_file", "report_addr") and pk not in kinds:
            _err(where, "port kind %s needs the same readiness criterion" % pk)
    return r


def validate_scenario(doc):
    """Validate a scenario document (structure only; cross-checks need the tier table and pin
    table, see cross_check). Returns {role: role dict} in declaration order."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        _err("top", "schema must be %s" % SCHEMA)
    missing = [k for k in TOP_KEYS if k not in doc and k not in TOP_KEYS_OPTIONAL]
    extra = [k for k in doc if k not in TOP_KEYS]
    if missing or extra:
        _err("top", "keys missing=%s extra=%s" % (missing, extra))
    if not isinstance(doc["experiment"], str) or not EXPERIMENT_RE.match(doc["experiment"]):
        _err("top", "bad experiment")
    for k in ("tier_table", "schedule", "prompts"):
        v = doc[k]
        if not isinstance(v, str) or not v or v.startswith("/") or ".." in v.split("/"):
            _err("top", "%s must be a relpath inside <W>" % k)
    if doc["probe_tier"] not in PROBE_TIERS + (None,):
        _err("top", "probe_tier must be P0/P1/P2/null")
    pg = doc["probe_groups"]
    if not isinstance(pg, list) or any(not isinstance(g, str) for g in pg):
        _err("top", "probe_groups must be a list of strings")
    if doc["probe_tier"] == "P2" and not pg or doc["probe_tier"] in ("P0", "P1") and pg:
        _err("top", "probe_groups must be non-empty exactly for P2")
    if not isinstance(doc["stop_grace_s"], (int, float)) or doc["stop_grace_s"] < 0:
        _err("top", "stop_grace_s must be >= 0")
    params = doc["params_required"]
    if not isinstance(params, list) or any(not isinstance(p, str) for p in params):
        _err("top", "params_required must be a list of names")
    dims = doc["dims"]
    if not isinstance(dims, dict) or set(dims) != set(DIMS_KEYS):
        _err("top", "dims must have exactly %s" % (DIMS_KEYS,))
    roles = doc["roles"]
    if not isinstance(roles, list) or not roles:
        _err("top", "roles must be a non-empty list")
    names = [r.get("role") if isinstance(r, dict) else None for r in roles]
    if len(set(names)) != len(names):
        _err("top", "role names must be unique")
    job_kinds = ()
    if "job_inputs" in doc:
        jobinputs.validate_section(doc["job_inputs"], set(params))
        job_kinds = tuple(jobinputs.kinds_of(doc))
    out = {}
    for r in roles:
        out[r["role"]] = validate_role(r, set(names), set(params), job_kinds)
        if r["launcher"] == "product" and r["probe_tier"] != doc["probe_tier"]:
            _err("role %s" % r["role"], "probe_tier differs from the launch probe_tier")
    if sorted(doc["stop_order"]) != sorted(names):
        _err("top", "stop_order must list every role exactly once")
    if "lag" in pg and any(d["launcher"] == "product" for d in out.values()) \
            and LAG_PARAM not in params:
        _err("top", "probe group lag on product roles needs %s in params_required (the "
             "orchestrator sets CAPPROBE_LAG_PERIOD_US from it)" % LAG_PARAM)
    _toposort(out)
    validate_reconcile(doc["reconcile"], set(names))
    for r in out.values():
        if r.get("module") == "capenv.spinners":
            _check_target_role(r, out)
    return out


def _flag_values(args, flag):
    """Values of `flag V` / `flag=V` in a role's args (the item after a bare flag as it is, None
    when the flag is the last item)."""
    vals = []
    for i, a in enumerate(args):
        if not isinstance(a, str):
            continue
        if a == flag:
            vals.append(args[i + 1] if i + 1 < len(args) else None)
        elif a.startswith(flag + "="):
            vals.append(a[len(flag) + 1:])
    return vals


def _check_target_role(r, roles):
    """capenv.spinners --target-role X (INTERFACES errata SB-B1-3): X is a literal role name of
    this scenario whose pin_role is X (the tool looks X up among the pin table's roles, the
    judge's C9/C13 look up the role name), started before this role (in its start_after)."""
    where = "role %s" % r["role"]
    vals = _flag_values(r["args"], "--target-role")
    if len(vals) > 1:
        _err(where, "--target-role given %d times" % len(vals))
    if not vals:
        return
    x = vals[0]
    if not isinstance(x, str) or not x or "{" in x:
        _err(where, "--target-role needs a literal role name, got %r" % (x,))
    if x not in roles or x == r["role"]:
        _err(where, "--target-role %s is not another role of this scenario" % x)
    if roles[x]["pin_role"] != x:
        _err(where, "--target-role %s: role %s has pin_role %s; the target role's pin_role "
             "must equal its name" % (x, x, roles[x]["pin_role"]))
    if x not in r["start_after"]:
        _err(where, "--target-role %s needs %s in start_after" % (x, x))


def _toposort(roles):
    """Start order honouring start_after; raises on cycles."""
    order, done, busy = [], set(), set()

    def visit(n):
        if n in done:
            return
        if n in busy:
            _err("top", "start_after has a cycle through %s" % n)
        busy.add(n)
        for d in roles[n]["start_after"]:
            visit(d)
        busy.discard(n)
        done.add(n)
        order.append(n)

    for n in roles:
        visit(n)
    return order


start_order = _toposort


def cross_check(doc, roles, tier_table, pin_roles, snapshot_interval_ns):
    """Checks that need the tier table and the pin table (exit 3 on failure).

    pin_roles: the "roles" mapping of pin_table.json ({name: {"cpus", "numa_node"}}).
    Returns {role: [cpus]}.
    """
    cpus_of, owner = {}, {}
    for name, r in roles.items():
        pr = pin_roles.get(r["pin_role"])
        if pr is None:
            _err("role %s" % name, "pin_role %s is not in the pin table" % r["pin_role"])
        cpus = list(pr.get("cpus") or [])
        if len(cpus) != r["cpus_needed"]:
            _err("role %s" % name, "pin table gives %d cpus, cpus_needed is %d"
                 % (len(cpus), r["cpus_needed"]))
        if r["launcher"] == "capkit" and TOOL_TABLE[r["module"]].get("workers") \
                and len(cpus) < r["workers"]:
            _err("role %s" % name, "fewer cpus than workers")
        for c in cpus:
            if c in owner:
                _err("role %s" % name, "cpu %d is also pinned to role %s (D1)" % (c, owner[c]))
            owner[c] = name
        cpus_of[name] = cpus
    gates = []
    if tier_table.default_gate is not None:
        gates.append(("default_gate", tier_table.default_gate))
    for t in tier_table.tiers:
        if t.gate is not None:
            gates.append(("tier %d" % t.seq, t.gate))
    for where, g in gates:
        if g["poll_ns"] < 2 * snapshot_interval_ns:
            _err(where, "gate poll_ns < 2 x snapshot interval (INTERFACES 8.1)")
        if g["max_snapshot_age_ns"] < 2 * snapshot_interval_ns:
            _err(where, "gate max_snapshot_age_ns < 2 x snapshot interval (INTERFACES 8.1)")
        for rd in g["readings"]:
            for term in rd["terms"]:
                r = roles.get(term["role"])
                if r is None:
                    _err(where, "gate term names unknown role %s" % term["role"])
                if term["slot"] not in r["backlog_slots"]:
                    _err(where, "role %s does not declare backlog slot %s"
                         % (term["role"], term["slot"]))
    for spec in list(tier_table.launch_inject) + [s for t in tier_table.tiers for s in t.inject]:
        if spec.role not in roles:
            _err("tier table inject", "role %s is not in the scenario" % spec.role)
    if doc["experiment"] != tier_table.experiment:
        _err("top", "experiment %s != tier table experiment %s"
             % (doc["experiment"], tier_table.experiment))
    return cpus_of


class ArgvContext:
    """Everything build_argv needs; all values explicit (no defaults)."""

    def __init__(self, identity, run_dir, raw_dir, params, stop_grace_s, shm_dir,
                 snapshot_interval_ms, tier_table_path, origin_ns, own_files, port_file_of,
                 addr_of, sidecar=None, canary=None, inputs=None, nvidia_smi=None):
        self.identity = identity  # capcommon Identity of the orchestrator (role "launch")
        self.run_dir = run_dir
        self.raw_dir = raw_dir
        self.params = params
        self.stop_grace_s = stop_grace_s
        self.shm_dir = shm_dir
        self.snapshot_interval_ms = snapshot_interval_ms
        self.tier_table_path = tier_table_path
        self.origin_ns = origin_ns
        self.own_files = own_files  # role -> {"port_file", "report_addr", "ready_file"}
        self.port_file_of = port_file_of  # role -> capkit port file path
        self.addr_of = addr_of  # role -> "host:port" (resolved when the role is ready)
        self.sidecar = sidecar  # {"roles_file", "out_dir", "logs_dir", "cpus_watch"}
        self.canary = canary  # {"out"}
        # {"tier_table"|"schedule"|"prompts"|<job input kind>: absolute path}
        self.inputs = inputs or {}
        self.nvidia_smi = nvidia_smi  # orchestrator --nvidia-smi


def resolve(text, role, ctx, runtime=True):
    """Substitute placeholders. runtime=False leaves {addr:R} unresolved (for launch.json)."""
    def sub(m):
        name, arg = m.group(1), m.group(2)
        if name == "run_dir":
            return ctx.run_dir
        if name == "raw":
            return ctx.raw_dir
        if name == "nvidia_smi":
            return ctx.nvidia_smi
        if name == "param":
            return ctx.params[arg]
        if name == "port_file":
            return ctx.port_file_of(arg)
        if name == "addr":
            return ctx.addr_of(arg) if runtime else m.group(0)
        if name == "input":
            return ctx.inputs[arg]
        if name == "own":
            if arg == "role":
                return role
            return ctx.own_files[role][arg]
        raise ConfigError("unknown placeholder %s" % m.group(0))
    return PLACEHOLDER_RE.sub(sub, text)


def _splice(item, role, ctx, runtime):
    import json  # noqa: PLC0415
    path = resolve(item["splice_json"], role, ctx, runtime)
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise ConfigError("splice_json %s: %s" % (path, e))
    val = doc.get(item["key"]) if isinstance(doc, dict) else None
    if not isinstance(val, list) or any(not isinstance(v, str) for v in val):
        raise ConfigError("splice_json %s: key %s is not a list of strings" % (path, item["key"]))
    return val


def _items(items, role, ctx, runtime):
    out = []
    for it in items:
        if isinstance(it, dict):
            out.extend(_splice(it, role, ctx, runtime))
        else:
            out.append(resolve(it, role, ctx, runtime))
    return out


def build_argv(r, ctx, cpus, python_exe, runtime=True):
    """Full argv for one role (without the taskset prefix)."""
    from capcommon.cli import format_cpu_list  # noqa: PLC0415
    name = r["role"]
    if r["launcher"] == "product":
        return _items(r["command"], name, ctx, runtime) + _items(r["args"], name, ctx, runtime)
    tool = TOOL_TABLE[r["module"]]
    i = ctx.identity
    argv = [python_exe, "-m", r["module"],
            "--experiment", i.experiment, "--arm", i.arm, "--launch-seq", str(i.launch_seq),
            "--job-id", i.job_id, "--restart-count", str(i.restart_count),
            "--register-sha", i.register_sha, "--role", name,
            "--run-dir", ctx.run_dir, "--stop-grace-s", repr(float(ctx.stop_grace_s)),
            "--cpus", format_cpu_list(cpus)]
    if tool.get("loop"):
        argv += ["--loop", r["loop"]]
    if tool.get("tiers"):
        argv += ["--tier-table", ctx.tier_table_path, "--origin-ns", str(ctx.origin_ns)]
    elif tool.get("origin"):
        argv += ["--origin-ns", str(ctx.origin_ns)]
    if tool.get("snapshot"):
        argv += ["--shm-dir", ctx.shm_dir, "--snapshot-interval-ms",
                 str(ctx.snapshot_interval_ms)]
    elif tool.get("shm_only"):
        argv += ["--shm-dir", ctx.shm_dir]
    if tool.get("ready"):
        argv += ["--ready-file", ctx.own_files[name]["ready_file"]]
    if tool.get("port"):
        argv += ["--port-file", ctx.port_file_of(name)]
    if tool.get("workers"):
        argv += ["--workers", str(r["workers"])]
    if tool.get("sidecar"):
        s = ctx.sidecar
        argv += ["--roles-file", s["roles_file"], "--out-dir", s["out_dir"],
                 "--logs-dir", s["logs_dir"], "--cpus-watch", s["cpus_watch"]]
    if tool.get("canary"):
        argv += ["--out", ctx.canary["out"]]
    if tool.get("pin_table"):
        argv += ["--pin-table", RunTree(ctx.run_dir).env("pin_table.json")]
    return argv + _items(r["args"], name, ctx, runtime)

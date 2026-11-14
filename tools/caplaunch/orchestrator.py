# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capkit-launch: one launch = start roles -> warmup -> tiers -> stop -> collect (design 2.8).

INTERFACES.md sections 0, 2.3, 5.1-5.4, 5.7, 8 define what is written here. The origin is chosen
before the first process starts; every role must be ready before it (no separate ready timeout).
Between tiers, drain tiers carry the C11 drain gate; the next tier's start time is fixed by the
tier table, so the orchestrator never signals tools to switch tiers.
"""

import hashlib
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time

from capcommon import gate as gatemod
from capcommon import jobinputs
from capcommon import probe_entry
from capcommon.capbin import read_capbin
from capcommon.clock import now_ns, wall_iso
from capcommon.exitcodes import OK, RUNTIME, STOPPED_EARLY, ConfigError
from capcommon.hashing import is_git_sha, is_sha256_hex, sha256_bytes, sha256_file
from capcommon.identity import Identity
from capcommon.jsonio import atomic_write_bytes, dumps, read_json, write_json_stamped
from capcommon.lifecycle import STOP_SCHEMA
from capcommon.runtree import RunTree, append_state, launch_shm_subdir
from capcommon.templates import parse_params
from capcommon.snapshot import (SnapshotError, decode_snapshot, ensure_own_dir,
                               list_snapshot_files)
from capcommon.tiers import TierTable

from . import procfs
from . import scenario as sc
from .ready import (ReadyError, http_get_status, read_capkit_port, read_capkit_ready,
                    read_plain_port, read_report_addr)

MARK_ENV = "CAPKIT_LAUNCH_MARK"
INJECT_ENV = "CAPPROBE_INJECT"
LAG_ENV = "CAPPROBE_LAG_PERIOD_US"
PATCH_ENV = "CAPPROBE_PATCH_SHA"
NONE = "none"  # explicit "not applicable" value for the provenance arguments
# Roles whose own unregistered descendants (e.g. the sidecar's nvidia-smi queries) are not part of
# the process set under test: they are left out of the window pid comparison and of
# roles.json "unattributed", and listed in procs/monitor_children.json instead. Roles of every
# other class (under_test, stub_*) keep both checks.
MONITOR_CLASSES = ("load_source", "observer")
assert set(MONITOR_CLASSES) < set(sc.MEASURE_CLASSES)
# Helper processes a capkit tool's runtime starts on its own (category aux_helper): attributed to
# the tool's role in roles.json "aux_helpers" and tier.json scans, never unattributed and never
# part of the compared pid set. Matched on the command line `python [interpreter options] -c
# <code>` with <code> matching a rule exactly; anything else a tool role starts that is not
# registered stays unattributed.
AUX_HELPER_RULES = (
    ("mp_resource_tracker", re.compile(r"from multiprocessing\.resource_tracker import main;"
                                       r"main\(\d+\)")),
    ("mp_forkserver", re.compile(r"from multiprocessing\.forkserver import main; "
                                 r"main\(.*\)")),
)
# CPython interpreter options multiprocessing copies in front of `-c` from the parent's flags and
# environment (subprocess._args_from_interpreter_flags, e.g. `-B` under PYTHONDONTWRITEBYTECODE=1,
# `-OO`, `-I`, `-W <arg>`, `-X dev`), plus `-u`. Options without an argument may be grouped
# (`-BE`); `-W` and `-X` take their argument in the same token or the next one.
AUX_HELPER_NOARG_OPTS = frozenset("bBdEIOPqsSuv")
AUX_HELPER_ARG_OPTS = frozenset("WX")


def _interpreter_options_end(args):
    """Index of `-c` in args (argv after the executable) when everything before it is CPython
    interpreter options (AUX_HELPER_NOARG_OPTS / AUX_HELPER_ARG_OPTS), else None."""
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "-c":
            return i
        if len(tok) < 2 or tok[0] != "-" or tok[1] == "-":
            return None
        j = 1
        while j < len(tok):
            ch = tok[j]
            if ch in AUX_HELPER_NOARG_OPTS:
                j += 1
                continue
            if ch not in AUX_HELPER_ARG_OPTS:
                return None
            if j + 1 == len(tok):  # the argument is the next token
                i += 1
                if i >= len(args):
                    return None
            break
        i += 1
    return None


def aux_helper_rule(argv):
    """Name of the AUX_HELPER_RULES entry matching `python [interpreter options] -c <code>` argv
    (<code> last, nothing after it), else None."""
    if not argv or len(argv) < 3:
        return None
    k = _interpreter_options_end(argv[1:])
    if k is None or k + 3 != len(argv):
        return None
    for name, rx in AUX_HELPER_RULES:
        if rx.fullmatch(argv[-1]):
            return name
    return None


class LaunchAbort(Exception):
    """Internal: stop the launch now with (state, reason, exit code)."""

    def __init__(self, state, reason, code):
        super().__init__(reason)
        self.state, self.reason, self.code = state, reason, code


def _opt(v):
    return None if v == NONE else v


def _rel_inside(path, root, what):
    ap, ar = os.path.realpath(path), os.path.realpath(root)
    if os.path.commonpath([ap, ar]) != ar:
        raise ConfigError("%s %s is not inside --workdir %s" % (what, path, root))
    return os.path.relpath(ap, ar)


def _private_dir(path, what):
    """capcommon.snapshot.ensure_own_dir (its checks cover the last path component only); `what`
    names the option or directory in the error."""
    try:
        return ensure_own_dir(path)
    except SnapshotError as e:
        raise ConfigError("%s: %s" % (what, e)) from e


class _GateTimeout(Exception):
    """Internal: an enforce-mode drain gate did not pass before its deadline."""

    def __init__(self, seq):
        super().__init__("drain gate timeout in tier %d" % seq)
        self.seq = seq


class RoleRT:
    """Runtime state of one scenario role (its top-level process)."""

    def __init__(self, decl, cpus):
        self.decl = decl
        self.name = decl["role"]
        self.cpus = cpus
        self.popen = None
        self.pid = None
        self.starttime = None
        self.started_ns = None
        self.ready = False
        self.ready_ns = None
        self.ready_doc = None
        self.addr = None  # (host, port)
        self.returncode = None
        self.exited_ns = None
        self.argv = None
        self.logs = None
        self.forced = None
        self.probe_entry = None  # product roles: the CAPPROBE_ENTRY rule it was started with


class Launch:
    def __init__(self, args, proc_root="/proc", sys_root="/sys"):
        self.a = args
        self.proc_root, self.sys_root = proc_root, sys_root
        self.ident = Identity(experiment=args.experiment, arm=args.arm,
                              launch_seq=args.launch_seq, job_id=args.job_id,
                              restart_count=args.restart_count, register_sha=args.register_sha,
                              role=args.role, worker=0, pid=os.getpid())
        if args.role != "launch":
            raise ConfigError("--role must be launch for the orchestrator")
        self.launch_key = self.ident.launch_key()
        for k in ("snapshot_interval_ms", "origin_lead_ms", "housekeeping_ms"):
            if getattr(args, k) <= 0:
                raise ConfigError("--%s must be > 0" % k.replace("_", "-"))
        if args.kill_after_s <= 0:
            raise ConfigError("--kill-after-s must be > 0")
        self.snap_ns = args.snapshot_interval_ms * 1_000_000
        # A late gate poll waits until this long after the previous poll, so every live writer
        # has published a newer snapshot (seq advanced) by then; see _gate_next_abs.
        self.gate_min_gap_ns = self.snap_ns + self.snap_ns // 2
        self.hk_ns = args.housekeeping_ms * 1_000_000
        self.W = os.path.realpath(args.workdir)
        if not os.path.isdir(self.W):
            raise ConfigError("--workdir %s is not a directory" % args.workdir)
        self._check_provenance()
        self._check_nvidia_smi()
        self._load_inputs()
        self._load_overlay()
        self._check_outer_probe_env()
        self.taskset = shutil.which("taskset")
        if self.taskset is None:
            raise ConfigError("taskset not found on PATH (needed to pin roles)")
        self.tree = RunTree(args.run_dir)
        self.mark = "%s=%s" % (MARK_ENV, secrets.token_hex(12))
        self.procs = []  # roles.json "procs"
        self.unattributed = []
        self.generation = 0
        self.snap_meta = {}  # pid -> latest snapshot meta (versions, threads)
        self.events_log = []
        self.gate_outcomes = []
        self.tiers_completed = None
        self.stop_doc = None
        self.early_exit = None
        self.monitor_children = {}  # (pid, starttime) -> procs/monitor_children.json entry
        self.aux_helpers = []  # roles.json "aux_helpers"
        self.n_proc_scans = 0
        self.cleaning = False  # set by the first stop signal or when the timeline is over
        self.hurry = False  # a stop signal arrived during cleanup: SIGKILL without grace
        self.signals = []

    def _check_outer_probe_env(self):
        """Config errors before anything is written or started: an outer value that contradicts
        what this orchestrator sets (version-report variables for every role; capprobe switches
        for product roles), and an outer CAPPROBE_INJECT when the scenario has product roles (no
        scenario field carries a product-side injection, so it is never passed through)."""
        sw = list(self._version_env().items())
        if any(d["launcher"] == "product" for d in self.roles.values()):
            if os.environ.get(INJECT_ENV):
                raise ConfigError("environment %s=%r: product-side injection is not supported by "
                                  "this orchestrator" % (INJECT_ENV, os.environ[INJECT_ENV]))
            sw += self._probe_switches()
        for key, val in sw:
            if val is not None and key in os.environ and os.environ[key] != val:
                raise ConfigError("environment %s=%r conflicts with %r set by the orchestrator"
                                  % (key, os.environ[key], val))

    def _probe_switches(self):
        """capprobe switches a product role gets from the scenario: tier, groups, and with the
        lag group the lag period (scenario param lag_period_us, checked in _load_inputs)."""
        out = [("CAPPROBE_TIER", self.scen["probe_tier"]),
               ("CAPPROBE_GROUPS", ",".join(self.scen["probe_groups"]))]
        if "lag" in self.scen["probe_groups"]:
            out.append((LAG_ENV, self.params[sc.LAG_PARAM]))
        return out

    def _version_env(self):
        """Version-report variables every role gets (INTERFACES 2.5): CAPKIT_CODE_SHA =
        --capkit-sha; with launch tier P1/P2 also CAPPROBE_PATCH_SHA = the overlay manifest's
        target_commit (None when there is no manifest: the launch then stops as
        overlay_manifest_missing before any role starts). _role_env removes an outer
        CAPPROBE_PATCH_SHA first, so P0 / no-tier launches never carry one."""
        out = {"CAPKIT_CODE_SHA": self.a.capkit_sha}
        if self.scen["probe_tier"] in ("P1", "P2"):
            out[PATCH_ENV] = self.overlay_target
        return out

    def _check_provenance(self):
        a = self.a
        for k in ("capkit_sha", "baseline_sha", "probe_tree_sha"):
            v = getattr(a, k)
            if v != NONE and not is_git_sha(v):
                raise ConfigError("--%s must be a 40-hex git sha or none" % k.replace("_", "-"))
        for k in ("image_sha256", "hw_profile_sha"):
            v = getattr(a, k)
            if v != NONE and not is_sha256_hex(v):
                raise ConfigError("--%s must be 64-hex or none" % k.replace("_", "-"))
        if a.capkit_sha == NONE:
            raise ConfigError("--capkit-sha is required (code provenance, DATA 4.3)")

    def _check_nvidia_smi(self):
        p = self.a.nvidia_smi
        if not os.path.isabs(p) or not os.path.isfile(p) or not os.access(p, os.X_OK):
            raise ConfigError("--nvidia-smi %s must be an absolute path to an executable file" % p)
        self.nvsmi_real = os.path.realpath(p)
        self.nvsmi_record = {"path": p, "realpath": self.nvsmi_real, "sha256": sha256_file(p)}

    # ---- inputs ------------------------------------------------------------------------------
    def _load_inputs(self):
        a = self.a
        reg_path = os.path.join(self.W, a.register) if not os.path.isabs(a.register) \
            else a.register
        self.register_rel = _rel_inside(reg_path, self.W, "--register")
        got = sha256_file(reg_path)
        if got != a.register_sha:
            raise ConfigError("sha256(--register) %s != --register-sha %s" % (got, a.register_sha))
        reg = read_json(reg_path)
        if not isinstance(reg, dict) or reg.get("schema") != "capkit.register/1":
            raise ConfigError("--register is not a capkit.register/1 file")
        if reg.get("experiment") != a.experiment or a.arm not in (reg.get("arms") or []):
            raise ConfigError("register experiment/arms do not cover %s/%s"
                              % (a.experiment, a.arm))
        has_f, has_fa = "files" in reg, "files_by_arm" in reg
        if has_f == has_fa:
            raise ConfigError("register needs exactly one of files / files_by_arm")
        files = reg["files"] if has_f else (reg["files_by_arm"] or {}).get(a.arm)
        if not isinstance(files, dict):
            raise ConfigError("register has no files for arm %s" % a.arm)
        scen_path = a.scenario if os.path.isabs(a.scenario) else os.path.join(self.W, a.scenario)
        self.scenario_rel = _rel_inside(scen_path, self.W, "--scenario")
        with open(scen_path, "rb") as f:
            raw = f.read()
        import json  # noqa: PLC0415
        try:
            self.scen = json.loads(raw.decode("utf-8"))
        except ValueError as e:
            raise ConfigError("--scenario is not JSON: %s" % e)
        self.roles = sc.validate_scenario(self.scen)
        self.scen_sha = sha256_file(scen_path)
        self.inputs = {}
        for key, rel in (("tier_table", self.scen["tier_table"]),
                         ("schedule", self.scen["schedule"]),
                         ("prompts", self.scen["prompts"]),
                         ("scenario", self.scenario_rel)):
            path = os.path.join(self.W, rel)
            sha = sha256_file(path)
            regd = files.get(key)
            if not isinstance(regd, dict) or regd.get("relpath") != rel \
                    or regd.get("sha256") != sha:
                raise ConfigError("%s %s (sha %s) does not match register entry %r"
                                  % (key, rel, sha, regd))
            self.inputs[key] = {"relpath": rel, "sha256": sha, "path": path}
        self.tt = TierTable.load(self.inputs["tier_table"]["path"])
        hdr, _ = read_capbin(self.inputs["schedule"]["path"], columns=[])
        if (hdr.get("meta") or {}).get("tier_table_sha256") != self.tt.sha256:
            raise ConfigError("schedule header tier_table_sha256 != tier table sha")
        for seq in [t.seq for t in self.tt.tiers if t.kind == "drain"]:
            gg, _src = self.tt.gate_of(seq)
            if gg["rule"] != "abs_max":
                raise ConfigError("tier %d: gate rule %s needs a registered band file; this "
                                  "orchestrator supports abs_max only (INTERFACES 8.1)"
                                  % (seq, gg["rule"]))
        pin = read_json(a.pin_table)
        if not isinstance(pin, dict) or pin.get("schema") != "capkit.pin_table/1" \
                or not isinstance(pin.get("roles"), dict):
            raise ConfigError("--pin-table is not a capkit.pin_table/1 file with roles")
        self.pin = pin
        self.pin_sha = sha256_file(a.pin_table)
        self.cpus_of = sc.cross_check(self.scen, self.roles, self.tt, pin["roles"], self.snap_ns)
        allc = sorted({c for v in pin["roles"].values() for c in (v.get("cpus") or [])})
        self.cpus_watch = ",".join(str(c) for c in allc)
        self.params = parse_params(a.param)
        req = set(self.scen["params_required"])
        if set(self.params) != req:
            raise ConfigError("--param keys %s != scenario params_required %s"
                              % (sorted(self.params), sorted(req)))
        lag = self.params.get(sc.LAG_PARAM)
        # ASCII digits only: str.isdigit() also accepts e.g. '²', which int() rejects
        if sc.LAG_PARAM in req and not (lag.isascii() and lag.isdigit() and int(lag) > 0):
            raise ConfigError("--param %s must be an integer > 0, got %r" % (sc.LAG_PARAM, lag))
        self._load_job_inputs()
        self.rt = {n: RoleRT(d, self.cpus_of[n]) for n, d in self.roles.items()}
        self.order = sc.start_order(self.roles)

    def _load_job_inputs(self):
        """--job-input KIND=PATH (errata B2B-9): exactly the kinds the scenario's job_inputs
        declares; each file has its kind's schema and a record (PATH.record.json) saying it was
        generated by the kind's module from inputs with this launch's registered shas. The bytes
        read here are what _prepare_tree copies into raw/env/job_inputs/."""
        want = jobinputs.kinds_of(self.scen)
        given = {}
        for kv in self.a.job_input:
            k, sep, path = kv.partition("=")
            if not sep or k not in jobinputs.KINDS or k in given:
                raise ConfigError("--job-input must be KIND=PATH, KIND in %s, each once: %r"
                                  % (list(jobinputs.KINDS), kv))
            given[k] = path
        if sorted(given) != sorted(want):
            raise ConfigError("--job-input kinds %s != scenario job_inputs %s"
                              % (sorted(given), sorted(want)))
        reg = {k: self.inputs[k]["sha256"] for k in jobinputs.REGISTERED_INPUTS}
        self.job_inputs = {}
        for k in want:
            path = given[k]
            if not os.path.isabs(path):
                raise ConfigError("--job-input %s: %s is not an absolute path" % (k, path))
            try:
                with open(path, "rb") as f:
                    data = f.read()
                with open(path + ".record.json", "rb") as f:
                    rec_bytes = f.read()
            except OSError as e:
                raise ConfigError("--job-input %s: %s" % (k, e))
            try:
                sha = jobinputs.load_verified(k, data, rec_bytes, reg)
            except ConfigError as e:
                raise ConfigError("--job-input %s: %s" % (path, e))
            self.job_inputs[k] = {"source": path, "data": data, "record": rec_bytes,
                                  "sha256": sha, "size": len(data),
                                  "record_sha256": sha256_bytes(rec_bytes),
                                  "module": jobinputs.KINDS[k][0]}

    def _load_overlay(self):
        """--overlay-manifest: the job-level overlay manifest, copied verbatim into
        raw/env/overlay_manifest.json. A missing file is recorded; it stops the launch as NOT_RUN
        only when a role carries a probe tier (checked before any process starts)."""
        path = self.a.overlay_manifest
        self.overlay_needed_by = sorted(n for n, d in self.roles.items()
                                        if d["probe_tier"] is not None)
        self.overlay_bytes = None
        if os.path.lexists(path):
            if not os.path.isfile(path):
                raise ConfigError("--overlay-manifest %s is not a regular file" % path)
            with open(path, "rb") as f:
                self.overlay_bytes = f.read()
        if os.path.isdir(self.a.job_env_dir):
            dup = os.path.join(self.a.job_env_dir, "overlay_manifest.json")
            if os.path.isfile(dup):
                with open(dup, "rb") as f:
                    if f.read() != self.overlay_bytes:
                        raise ConfigError("%s differs from --overlay-manifest %s" % (dup, path))
        self.overlay_target = None
        if self.overlay_bytes is not None and self.scen["probe_tier"] in ("P1", "P2"):
            import json  # noqa: PLC0415
            try:
                tgt = json.loads(self.overlay_bytes.decode("utf-8")).get("target_commit")
            except (ValueError, AttributeError) as e:
                raise ConfigError("--overlay-manifest %s is not a JSON object: %s" % (path, e))
            if not is_git_sha(tgt):
                raise ConfigError("--overlay-manifest %s: target_commit %r is not a 40-hex sha "
                                  "(CAPPROBE_PATCH_SHA of a %s launch)"
                                  % (path, tgt, self.scen["probe_tier"]))
            self.overlay_target = tgt
        self.overlay_record = {
            "source": path, "present": self.overlay_bytes is not None,
            "sha256": None if self.overlay_bytes is None else
            hashlib.sha256(self.overlay_bytes).hexdigest(),
            "needed_by_roles": self.overlay_needed_by}

    # ---- tree, paths, launch.json -----------------------------------------------------------
    def _prepare_tree(self):
        T = self.tree
        _private_dir(T.root, "--run-dir")
        # the orchestrator is the first writer of a fresh tree: never after another STATE line
        for f in (T.launch_json, T.state_file):
            if os.path.exists(f):
                raise ConfigError("%s already has %s (one launch per tree)"
                                  % (T.root, os.path.relpath(f, T.root)))
        self.shm_dir = launch_shm_subdir(self.a.shm_dir, self.a.restart_count, self.a.launch_seq)
        _private_dir(self.a.shm_dir, "--shm-dir")
        _private_dir(self.shm_dir, "snapshot dir")
        if os.listdir(self.shm_dir):
            raise ConfigError("snapshot dir %s is not empty (stale launch?)" % self.shm_dir)
        T.make_skeleton()
        os.makedirs(T.procs("identity"), exist_ok=True)
        if os.path.isdir(self.a.job_env_dir):  # job-level gate results -> this tree (INTERFACES 5.1)
            src = self.a.job_env_dir
            for n in sorted(os.listdir(src)):
                p = os.path.join(src, n)
                if os.path.isfile(p):
                    shutil.copyfile(p, T.env(n))
        if self.overlay_bytes is not None:  # INTERFACES 6.6; read by the judge (C8, PROBE-TIER)
            atomic_write_bytes(T.env("overlay_manifest.json"), self.overlay_bytes)
        if not os.path.exists(T.env("pin_table.json")):
            shutil.copyfile(self.a.pin_table, T.env("pin_table.json"))
        for k, ji in self.job_inputs.items():  # errata B2B-9; read by the judge
            os.makedirs(T.env("job_inputs"), exist_ok=True)
            atomic_write_bytes(T.env("job_inputs", "%s.json" % k), ji["data"])
            atomic_write_bytes(T.env("job_inputs", "%s.record.json" % k), ji["record"])
        self.own = {}
        for n in self.roles:
            self.own[n] = {"ready_file": T.ready_file(n),
                           "port_file": T.procs("ports", "%s.port" % n),
                           "report_addr": T.procs("ports", "%s.addr" % n)}
            os.makedirs(self._role_cwd(n), exist_ok=True)

    def _ctx(self):
        return sc.ArgvContext(
            identity=self.ident, run_dir=self.tree.root, raw_dir=self.tree.raw,
            params=self.params, stop_grace_s=self.scen["stop_grace_s"], shm_dir=self.shm_dir,
            snapshot_interval_ms=self.a.snapshot_interval_ms,
            tier_table_path=self.inputs["tier_table"]["path"], origin_ns=self.origin_ns,
            own_files=self.own, port_file_of=self.tree.port_file, addr_of=self._addr_str,
            sidecar={"roles_file": self.tree.roles_json, "out_dir": self.tree.sidecar(),
                     "logs_dir": os.path.join(self.tree.raw, "logs"),
                     "cpus_watch": self.cpus_watch},
            canary={"out": self.tree.canary("canary.jsonl")},
            inputs=dict({k: os.path.realpath(self.inputs[k]["path"])
                         for k in ("tier_table", "schedule", "prompts")},
                        **{k: os.path.realpath(self.tree.env("job_inputs", "%s.json" % k))
                           for k in self.job_inputs}),
            nvidia_smi=self.a.nvidia_smi)

    def _addr_str(self, role):
        rt = self.rt[role]
        if rt.addr is None:
            raise ConfigError("role %s has no address yet" % role)
        h, p = rt.addr
        return "%s:%d" % (h, p)

    def _write_launch_json(self):
        a = self.a
        ctx = self._ctx()
        roles = []
        for n, d in self.roles.items():
            roles.append({
                "role": n, "launcher": d["launcher"], "module": d.get("module"),
                "kind": d["kind"], "args": sc.build_argv(d, ctx, self.cpus_of[n], sys.executable,
                                                         runtime=False),
                "cpus": self.cpus_of[n], "pin_role": d["pin_role"], "numa": d["numa"],
                "workers": d["workers"], "expected_procs": d["expected_procs"],
                "measure_class": d["measure_class"], "asyncio": d["asyncio"],
                "uses_grpc": d["uses_grpc"], "loop": d["loop"], "probe_tier": d["probe_tier"],
                "readiness": d["readiness"], "backlog_slots": d["backlog_slots"],
                "proc_kinds": d["proc_kinds"],
                "snapshots": d["snapshots"], "env": d["env"], "start_after": d["start_after"]})
        sub = None
        sp = os.path.join(self.W, "submissions.tsv")
        if os.path.exists(sp):
            from capcommon.ledger import submission_for_job  # noqa: PLC0415
            row = submission_for_job(sp, a.job_id)
            if row is not None:
                sub = {"job_id": a.job_id, "restart_count": a.restart_count,
                       "submitted_at": row.get("submitted_at") or None}
        payload = {
            "origin_ns": self.origin_ns, "origin_lead_ns": a.origin_lead_ms * 1_000_000,
            "shm_dir": self.shm_dir,
            "register": {"relpath": self.register_rel, "sha256": a.register_sha},
            "tier_table": {k: self.inputs["tier_table"][k] for k in ("relpath", "sha256")},
            "schedule": {k: self.inputs["schedule"][k] for k in ("relpath", "sha256")},
            "prompts": {k: self.inputs["prompts"][k] for k in ("relpath", "sha256")},
            "job_inputs": {k: {"relpath": os.path.relpath(self.tree.env("job_inputs",
                                                                        "%s.json" % k),
                                                          self.tree.root),
                               "sha256": ji["sha256"], "size": ji["size"],
                               "record_sha256": ji["record_sha256"], "module": ji["module"],
                               "source": ji["source"]}
                           for k, ji in self.job_inputs.items()},
            "scenario": {"relpath": self.scenario_rel, "sha256": self.scen_sha,
                         "doc": self.scen},
            "code": {"capkit_sha": a.capkit_sha, "probe_tree_sha": _opt(a.probe_tree_sha),
                     "baseline_sha": _opt(a.baseline_sha), "snapshot_dir": _opt(a.snapshot_dir)},
            "image": {"path": _opt(a.image_path), "sha256": _opt(a.image_sha256)},
            "probe_tier": {"tier": self.scen["probe_tier"], "groups": self.scen["probe_groups"]},
            "roles": roles, "reconcile": self.scen["reconcile"],
            "hw_profile_sha": _opt(a.hw_profile_sha), "submission": sub,
            "overlay_manifest": self.overlay_record, "nvidia_smi": self.nvsmi_record,
            "probe_entry": self._probe_entry_record(),
            "pin_table": {"path": a.pin_table, "sha256": self.pin_sha},
            "params": self.params, "launch_mark": self.mark,
            "argv_runtime_placeholders": "{addr:<role>} is resolved when the role starts; "
                                         "the resolved command line is in procs/roles.json",
            "orchestrator": {"python": sys.executable, "taskset": self.taskset,
                             "snapshot_interval_ms": a.snapshot_interval_ms,
                             "housekeeping_ms": a.housekeeping_ms,
                             "kill_after_s": a.kill_after_s, "argv": list(sys.argv),
                             "thread_rules_sha": procfs.RULES_SHA,
                             "monitor_classes": list(MONITOR_CLASSES),
                             "aux_helper_rules": [[n, rx.pattern] for n, rx in AUX_HELPER_RULES]},
        }
        write_json_stamped(self.tree.launch_json, "capkit.launch/1", self.ident, payload)
        for n, d in self.roles.items():
            if d["launcher"] == "product":
                ident = self.ident.with_role(n).to_dict()
                ident["pid"] = None
                atomic_write_bytes(self.tree.procs("identity", "%s.json" % n),
                                   (dumps({"schema": "capkit.identity/1",
                                           "identity": ident}) + "\n").encode())

    def _probe_entry_record(self):
        """launch.json `probe_entry`: which roles get a CAPPROBE_ENTRY rule and how it matches.
        The rule itself is built once, from the role's runtime argv, when the role starts; that
        object is what is injected and what roles.json procs[].probe_entry and the start event
        record."""
        return {"env": probe_entry.ENV_VAR, "schema": probe_entry.SCHEMA,
                "match": probe_entry.MATCH_RULE,
                "exact_rule_in": "procs/roles.json procs[].probe_entry",
                "roles": sorted(n for n, d in self.roles.items() if d["launcher"] == "product")}

    def _write_roles(self):
        self.generation += 1
        write_json_stamped(self.tree.roles_json, "capkit.roles/1", self.ident,
                           {"generation": self.generation, "procs": self.procs,
                            "unattributed": self.unattributed, "aux_helpers": self.aux_helpers})

    # ---- processes ---------------------------------------------------------------------------
    def _role_env(self, d):
        """Environment of one role, without an entry rule (see _entry_rule). Every role gets the
        version-report variables (_version_env), product roles also the capprobe switches; no
        role inherits an outer CAPPROBE_ENTRY or CAPPROBE_PATCH_SHA."""
        env = dict(os.environ)
        env.update(d["env"])
        env.pop(probe_entry.ENV_VAR, None)
        env.pop(PATCH_ENV, None)
        env.update({k: v for k, v in self._version_env().items() if v is not None})
        k, _, v = self.mark.partition("=")
        env[k] = v
        if d["launcher"] == "product":
            if not d.get("keep_numa_affinity_env", False):
                env["TLLM_NUMA_AWARE_WORKER_AFFINITY"] = "0"  # D2: worker must not unpin
            env.update(self._probe_switches())  # outer conflicts: _check_outer_probe_env
            env["CAPPROBE_SHM_DIR"] = self.shm_dir
            env["CAPPROBE_INTERVAL_MS"] = str(self.a.snapshot_interval_ms)
            env["CAPPROBE_IDENTITY_FILE"] = self.tree.procs("identity", "%s.json" % d["role"])
            env["CAPPROBE_ROLE_HINT"] = d["role"]
        return env

    def _entry_rule(self, d, argv, env):
        """CAPPROBE_ENTRY rule of a product role: names exactly the command line `argv` it is
        started with, resolved with that role's own PATH and cwd (only that process may switch
        probes on; INTERFACES erratum R1b). None for other roles."""
        if d["launcher"] != "product":
            return None
        try:
            return probe_entry.build_rule(argv, self._role_cwd(d["role"]),
                                          env.get("PATH", os.defpath))
        except probe_entry.EntryRuleError as e:
            raise LaunchAbort("NOT_RUN", "role %s: %s" % (d["role"], e), RUNTIME)

    def _role_cwd(self, name):
        return os.path.join(self.tree.control_dir, "cwd", name)

    def _register(self, pid, role, worker, parent, via, cpus, probe_entry_rule=None):
        st = procfs.starttime_ticks(pid, self.proc_root)
        for p in self.procs:
            if p["pid"] == pid and p["starttime_ticks"] == st:
                return p
        entry = {"pid": pid, "starttime_ticks": st, "role": role, "worker": worker,
                 "parent_pid": parent, "cmdline": procfs.cmdline(pid, self.proc_root),
                 "cpus": cpus, "cpus_readback": procfs.cpus_allowed_list(pid, None,
                                                                          self.proc_root),
                 "discovered_via": via, "registered_mono_ns": now_ns()}
        if probe_entry_rule is not None:
            entry["probe_entry"] = probe_entry_rule
        self.procs.append(entry)
        self.unattributed = [u for u in self.unattributed
                             if not (u["pid"] == pid and u["starttime_ticks"] == st)]
        self.monitor_children.pop((pid, st), None)  # seen before its role reported it
        self._write_roles()
        return entry

    def _start_role(self, name):
        rt = self.rt[name]
        d = rt.decl
        argv = sc.build_argv(d, self._ctx(), rt.cpus, sys.executable, runtime=True)
        full = [self.taskset, "-c", ",".join(str(c) for c in rt.cpus)] + argv
        logdir = os.path.join(self.tree.raw, "logs")
        tmp_out = os.path.join(logdir, "%s.w0.starting.out" % name)
        tmp_err = os.path.join(logdir, "%s.w0.starting.err" % name)
        env = self._role_env(d)
        rule = self._entry_rule(d, argv, env)
        if rule is not None:
            env[probe_entry.ENV_VAR] = probe_entry.encode(rule)
        with open(tmp_out, "wb") as fo, open(tmp_err, "wb") as fe:
            try:
                rt.popen = subprocess.Popen(full, env=env, stdout=fo,
                                            stderr=fe, stdin=subprocess.DEVNULL,
                                            start_new_session=True, cwd=self._role_cwd(name))
            except OSError as e:
                raise LaunchAbort("NOT_RUN", "cannot start role %s: %s" % (name, e), RUNTIME)
        rt.probe_entry = rule
        rt.pid = rt.popen.pid
        rt.started_ns = now_ns()
        rt.argv = full
        out, err = (self.tree.log_file(name, 0, rt.pid, s) for s in ("out", "err"))
        os.replace(tmp_out, out)
        os.replace(tmp_err, err)
        rt.logs = [out, err]
        rt.starttime = procfs.starttime_ticks(rt.pid, self.proc_root)
        self._register(rt.pid, name, 0, os.getpid(), "launched", rt.cpus, rule)
        self._log("start", role=name, pid=rt.pid, argv=full, probe_entry=rt.probe_entry)

    def _log(self, what, **kw):
        kw.update({"event": what, "t_mono_ns": now_ns(), "wall": wall_iso()})
        self.events_log.append(kw)

    def _check_ready(self, name):
        """Evaluate every readiness criterion of a started role; set rt.ready."""
        rt = self.rt[name]
        d = rt.decl
        ok_all = True
        for c in d["readiness"]:
            k = c["kind"]
            if k == "capkit_ready":
                got = read_capkit_ready(self.own[name]["ready_file"], self.launch_key, name)
                if got is None:
                    ok_all = False
                    continue
                rt.ready_doc = got[1]
                if sc.TOOL_TABLE[d["module"]].get("port") and rt.addr is None:
                    pa = read_capkit_port(self.tree.port_file(name), self.launch_key)
                    if pa is None:
                        ok_all = False
                        continue
                    rt.addr = pa
            elif k == "plain_port_file":
                port = read_plain_port(self.own[name]["port_file"])
                if port is None:
                    ok_all = False
                    continue
                rt.addr = (d["port"]["host"], port)
            elif k == "report_addr":
                ha = read_report_addr(self.own[name]["report_addr"])
                if ha is None:
                    ok_all = False
                    continue
                rt.addr = ha
            elif k == "http_get":
                if rt.addr is None:
                    ok_all = False
                    continue
                st = http_get_status(rt.addr[0], rt.addr[1], c["path"], c["timeout_s"])
                if st != c["expect_status"]:
                    ok_all = False
            elif k == "file_exists":
                if not os.path.exists(sc.resolve(c["path"], name, self._ctx())):
                    ok_all = False
        if ok_all and not rt.ready:
            rt.ready, rt.ready_ns = True, now_ns()
            for ch in (rt.ready_doc or {}).get("children") or []:
                self._register(ch["pid"], name, ch["worker"], rt.pid, "ready.children", rt.cpus)
            # taskset has exec'd by now: refresh what registration read right after fork
            for p in self.procs:
                if p["role"] == name:
                    p["cmdline"] = procfs.cmdline(p["pid"], self.proc_root) or p["cmdline"]
                    p["cpus_readback"] = procfs.cpus_allowed_list(p["pid"], None, self.proc_root)
            self._write_roles()
            self._log("ready", role=name, pid=rt.pid)
        return rt.ready

    # ---- discovery and liveness --------------------------------------------------------------
    def _proc_by_pid(self, pid):
        for p in self.procs:
            if p["pid"] == pid:
                return p
        return None

    def _top_roles(self):
        """{top-level pid: role} of every started role."""
        return {rt.pid: n for n, rt in self.rt.items() if rt.pid is not None}

    def _is_monitor(self, role):
        return self.roles[role]["measure_class"] in MONITOR_CLASSES

    def _proc_scan(self):
        """One pass over /proc: (descendant records with their top-level role, now)."""
        tops = self._top_roles()
        recs = procfs.descendant_records(list(tops), self.proc_root)
        for r in recs:
            r["role"] = tops[r["root"]]
        self.n_proc_scans += 1
        return recs, now_ns()

    def _aux_helper(self, rec, t_ns):
        """The roles.json aux_helpers entry for rec (adding it on first sight), or None when rec is
        not a whitelisted helper of a capkit tool role."""
        k = (rec["pid"], rec["starttime_ticks"])
        for e in self.aux_helpers:
            if (e["pid"], e["starttime_ticks"]) == k:
                return e
        if self.roles[rec["role"]]["launcher"] != "capkit":
            return None
        argv = procfs.cmdline(rec["pid"], self.proc_root)
        rule = aux_helper_rule(argv)
        if rule is None:
            return None
        e = {"pid": rec["pid"], "ppid": rec["ppid"], "starttime_ticks": rec["starttime_ticks"],
             "cmdline": argv, "role": rec["role"],
             "measure_class": self.roles[rec["role"]]["measure_class"],
             "category": "aux_helper", "rule": rule, "first_seen_mono_ns": t_ns}
        self.aux_helpers.append(e)
        self._write_roles()
        return e

    def _note_monitor_child(self, rec, t_ns, via):
        """Audit entry for an unregistered descendant of a MONITOR_CLASSES role."""
        k = (rec["pid"], rec["starttime_ticks"])
        e = self.monitor_children.get(k)
        if e is None:
            exe = procfs.exe_path(rec["pid"], self.proc_root)
            e = self.monitor_children[k] = {
                "exe": exe, "is_nvidia_smi": None if exe is None else exe == self.nvsmi_real,
                "pid": rec["pid"], "ppid": rec["ppid"], "comm": rec["comm"],
                "starttime_ticks": rec["starttime_ticks"], "role": rec["role"],
                "measure_class": self.roles[rec["role"]]["measure_class"],
                "first_seen_mono_ns": t_ns, "last_seen_mono_ns": t_ns, "n_seen": 0,
                "seen_via": []}
        e["last_seen_mono_ns"] = t_ns
        e["n_seen"] += 1
        if via not in e["seen_via"]:
            e["seen_via"].append(via)

    def _discover(self):
        """Register product children that self-report via probe snapshots; list unregistered
        descendants (reported only, never given a role: INTERFACES 5.3). A snapshot writer below
        a non-product role is never registered and always listed as unattributed. Other
        unregistered descendants of MONITOR_CLASSES roles go to the monitor-children audit list
        instead of unattributed."""
        files = list_snapshot_files(self.shm_dir)
        changed = False
        for pid, path in sorted(files.items()):
            try:
                with open(path, "rb") as f:
                    snap = decode_snapshot(f.read())
            except (OSError, SnapshotError):
                continue
            self.snap_meta[pid] = snap.meta
            if self._proc_by_pid(pid) is not None:
                continue
            reg_pids = {p["pid"]: p for p in self.procs}
            anc, q, seen = None, procfs.ppid_of(pid, self.proc_root), set()
            while q is not None and q > 0 and q not in seen:
                if q in reg_pids:
                    anc = reg_pids[q]
                    break
                seen.add(q)
                q = procfs.ppid_of(q, self.proc_root)
            st = snap.meta.get("starttime_ticks")
            if anc is not None and self.roles[anc["role"]]["launcher"] == "product":
                worker = 1 + max(p["worker"] for p in self.procs if p["role"] == anc["role"])
                self._register(pid, anc["role"], worker, snap.meta.get("ppid"),
                               "probe.snapshot", self.rt[anc["role"]].cpus)
            elif not any(u["pid"] == pid and u["starttime_ticks"] == st
                         for u in self.unattributed):
                self.unattributed.append({"pid": pid, "starttime_ticks": st,
                                          "seen_via": "probe.snapshot",
                                          "argv": snap.meta.get("argv"),
                                          "role_hint": snap.meta.get("role_hint"),
                                          "registered_ancestor": None if anc is None else
                                          {"pid": anc["pid"], "role": anc["role"]}})
                changed = True
        recs, t = self._proc_scan()
        known = {(p["pid"], p["starttime_ticks"]) for p in self.procs}
        for rec in recs:
            pid, st = rec["pid"], rec["starttime_ticks"]
            if (pid, st) in known:
                continue
            if any(u["pid"] == pid and u["starttime_ticks"] == st for u in self.unattributed):
                continue
            if self._aux_helper(rec, t) is not None:
                continue
            if self._is_monitor(rec["role"]):
                self._note_monitor_child(rec, t, "housekeeping")
                continue
            self.unattributed.append({"pid": pid, "starttime_ticks": st,
                                      "seen_via": "proc.children",
                                      "argv": procfs.cmdline(pid, self.proc_root)})
            changed = True
        if changed:
            self._write_roles()

    def _poll_exits(self):
        """Record exits of top-level processes; return [(role, rc)] that exited just now."""
        out = []
        for rt in self.rt.values():
            if rt.popen is not None and rt.returncode is None:
                rc = rt.popen.poll()
                if rc is not None:
                    rt.returncode, rt.exited_ns = rc, now_ns()
                    out.append((rt.name, rc))
                    self._log("exit", role=rt.name, pid=rt.pid, rc=rc)
        return out

    def _dead_registered(self):
        """Registered non-top-level processes that are gone (pid + starttime)."""
        tops = {rt.pid for rt in self.rt.values()}
        return [p for p in self.procs if p["pid"] not in tops
                and not procfs.alive(p["pid"], p["starttime_ticks"], self.proc_root)]

    def _check_early_exit(self, allow_clients_after_ns):
        """Raise LaunchAbort(VOID) when any process exits before it is allowed to."""
        now = now_ns()
        for name, rc in self._poll_exits():
            d = self.roles[name]
            if d["kind"] == "client" and allow_clients_after_ns is not None \
                    and now >= allow_clients_after_ns and rc == OK:
                continue
            self.early_exit = {"role": name, "pid": self.rt[name].pid, "rc": rc,
                               "t_mono_ns": now}
            raise LaunchAbort("VOID", "role %s pid %d exited early rc=%s"
                              % (name, self.rt[name].pid, rc), STOPPED_EARLY)
        dead = [p for p in self._dead_registered()
                if not (self.roles[p["role"]]["kind"] == "client"
                        and allow_clients_after_ns is not None and now >= allow_clients_after_ns)]
        if dead:
            p = dead[0]
            self.early_exit = {"role": p["role"], "pid": p["pid"], "rc": None, "t_mono_ns": now}
            raise LaunchAbort("VOID", "registered process %d (role %s) exited early"
                              % (p["pid"], p["role"]), STOPPED_EARLY)

    # ---- window boundaries (INTERFACES 5.4) ---------------------------------------------------
    def _snapshot_procs(self, roles=None):
        """Registered procs that are expected to write snapshots (role snapshots scope; a
        product role only when it writes them at all, scenario.product_writes_snapshots)."""
        out = []
        for p in self.procs:
            d = self.roles[p["role"]]
            if roles is not None and p["role"] not in roles:
                continue
            if d["launcher"] == "product" and not sc.product_writes_snapshots(d):
                continue
            if d["snapshots"] == "all" or (d["snapshots"] == "workers"
                                           and p["discovered_via"] != "launched"):
                out.append(p)
        return out

    def _declared_threads(self, p):
        decl = {}
        rt = self.rt[p["role"]]
        if p["pid"] == rt.pid:
            decl.update(procfs.declared_threads_from_ready(rt.ready_doc))
        th = (self.snap_meta.get(p["pid"]) or {}).get("threads")
        if isinstance(th, dict):
            try:
                decl.update({int(k): str(v) for k, v in th.items()})
            except (TypeError, ValueError):
                pass
        return decl

    def _scan(self, which):
        """pid set, threads of every registered live process, NIC counters. `descendants` (the
        set compared across the window) leaves out the subtrees of MONITOR_CLASSES roles; their
        unregistered descendants are listed in `monitor_descendants` and the audit list."""
        live = []
        for p in self.procs:
            if procfs.alive(p["pid"], p["starttime_ticks"], self.proc_root):
                live.append([p["pid"], p["starttime_ticks"]])
        recs, t = self._proc_scan()
        known = {(p["pid"], p["starttime_ticks"]) for p in self.procs}
        desc, mon, aux = [], [], []
        for rec in recs:
            if (rec["pid"], rec["starttime_ticks"]) not in known and \
                    self._aux_helper(rec, t) is not None:
                aux.append([rec["pid"], rec["starttime_ticks"], rec["role"]])
            elif not self._is_monitor(rec["role"]):
                desc.append([rec["pid"], rec["starttime_ticks"]])
            elif (rec["pid"], rec["starttime_ticks"]) not in known:
                mon.append([rec["pid"], rec["starttime_ticks"], rec["role"]])
                self._note_monitor_child(rec, t, which)
        threads = {}
        for p in self.procs:
            if [p["pid"], p["starttime_ticks"]] in live:
                inv = procfs.thread_inventory(p["pid"], self._declared_threads(p), self.proc_root)
                threads[str(p["pid"])] = [{k: t[k] for k in ("tid", "comm", "py_name", "class",
                                                              "identified_as", "cpus_allowed")}
                                          for t in inv]
        return {"t_mono_ns": now_ns(), "registered_live": live,
                "registered_total": len(self.procs), "descendants": sorted(desc),
                "monitor_descendants": sorted(mon), "aux_helpers": sorted(aux),
                "threads": threads, "nic_counters": procfs.nic_counters(self.sys_root)}

    def _boundary(self, seq, which, boundary_abs):
        """Start of a window boundary: scan now, queue snapshot copies (first >= boundary)."""
        st = self.tier_state.setdefault(seq, {"scans": {}, "copies": [], "pending": 0})
        st["scans"][which] = self._scan(which)
        t = self.tt.tiers[seq]
        d = os.path.join(self.tree.tier_dir(seq, t.label), "snapshots")
        os.makedirs(d, exist_ok=True)
        for p in self._snapshot_procs():
            job = {"seq": seq, "which": which, "pid": p["pid"], "role": p["role"],
                   "boundary_abs": boundary_abs, "deadline": boundary_abs + 2 * self.snap_ns,
                   "dest": os.path.join(d, "%s.%d.%s.bin" % (p["role"], p["pid"], which))}
            self.copy_jobs.append(job)
            st["pending"] += 1

    def _service_copies(self):
        now = now_ns()
        keep = []
        for j in self.copy_jobs:
            path = os.path.join(self.shm_dir, "%d.bin" % j["pid"])
            rec = None
            try:
                with open(path, "rb") as f:
                    buf = f.read()
                snap = decode_snapshot(buf)
                if snap.t_mono_ns >= j["boundary_abs"]:
                    atomic_write_bytes(j["dest"], buf)
                    rec = {"pid": j["pid"], "role": j["role"], "which": j["which"],
                           "seq": snap.seq, "t_mono_ns": snap.t_mono_ns,
                           "offset_from_boundary_ns": snap.t_mono_ns - j["boundary_abs"],
                           "file": os.path.basename(j["dest"])}
            except (OSError, SnapshotError):
                pass
            if rec is None and now > j["deadline"]:
                rec = {"pid": j["pid"], "role": j["role"], "which": j["which"],
                       "snapshot_missing": True,
                       "detail": "no snapshot with t_mono_ns >= boundary within 2 periods"}
            if rec is None:
                keep.append(j)
                continue
            st = self.tier_state[j["seq"]]
            st["copies"].append(rec)
            st["pending"] -= 1
            self._maybe_write_tier(j["seq"])
        self.copy_jobs = keep

    def _maybe_write_tier(self, seq, force_reason=None):
        st = self.tier_state.get(seq)
        if st is None or st.get("written"):
            return
        if force_reason is None and (st["pending"] > 0 or "wend" not in st["scans"]):
            return
        t = self.tt.tiers[seq]
        o = self.origin_ns
        ws, we = st["scans"].get("wstart"), st["scans"].get("wend")
        stable = None
        if ws is not None and we is not None:
            stable = (ws["registered_live"] == we["registered_live"]
                      and len(ws["registered_live"]) == ws["registered_total"]
                      and ws["descendants"] == we["descendants"])
        spec = dict(self.tt.doc["tiers"][seq])
        payload = {"tier": spec, "abs_start_ns": o + t.start_ns,
                   "abs_window_start_ns": o + t.window_start_ns,
                   "abs_window_end_ns": o + t.window_end_ns, "abs_end_ns": o + t.end_ns,
                   "scans": {k: {kk: v[kk] for kk in ("t_mono_ns", "registered_live",
                                                      "registered_total", "descendants",
                                                      "monitor_descendants", "aux_helpers")}
                             for k, v in st["scans"].items()},
                   "pids_stable": stable,
                   "threads": {k: v["threads"] for k, v in st["scans"].items()},
                   "nic_counters": {k: v["nic_counters"] for k, v in st["scans"].items()},
                   "snapshot_copies": sorted(st["copies"], key=lambda r: (r["which"], r["pid"])),
                   "thread_rules_sha": procfs.RULES_SHA, "incomplete": force_reason}
        ident = self.ident.with_tier(seq, t.label)
        write_json_stamped(os.path.join(self.tree.tier_dir(seq, t.label), "tier.json"),
                           "capkit.tier/1", ident, payload)
        st["written"] = True

    # ---- drain gate (INTERFACES 8.3) ---------------------------------------------------------
    def _gate_begin(self, seq):
        t = self.tt.tiers[seq]
        g, src = self.tt.gate_of(seq)
        deadline_rel = t.end_ns - g["deadline_guard_ns"]
        polls = gatemod.planned_polls(t.start_ns, deadline_rel, g["poll_ns"])
        self.gate_run = {"seq": seq, "gate": g, "source": src, "deadline_rel_ns": deadline_rel,
                         "planned": polls, "next": 0, "polls": [], "prev_seqs": None,
                         "last_poll_abs_ns": None,
                         "tracker": gatemod.GateTracker(g),
                         "roles": sorted({term["role"] for r in g["readings"]
                                          for term in r["terms"]})}
        if not polls:
            self._gate_end()

    def _gate_next_abs(self):
        """When the next poll may run: its planned time, but when the loop is late (several planned
        polls already due) not sooner than gate_min_gap_ns after the previous poll -- back-to-back
        polls would see an unchanged seq and reset the settle count. Never later than the gate
        deadline. Every planned poll is still made and recorded (the judge reconciles them)."""
        gr = self.gate_run
        t = self.origin_ns + gr["planned"][gr["next"]]
        last = gr["last_poll_abs_ns"]
        if last is not None:
            t = max(t, min(last + self.gate_min_gap_ns, self.origin_ns + gr["deadline_rel_ns"]))
        return t

    def _gate_poll(self):
        gr = self.gate_run
        g = gr["gate"]
        planned = gr["planned"][gr["next"]]
        gr["next"] += 1
        procs = self._snapshot_procs(set(gr["roles"]))
        by_role, errors, fields = gatemod.snapshots_by_role(
            self.shm_dir, procs, self.launch_key, g["max_snapshot_age_ns"],
            prev_seqs=gr["prev_seqs"])
        now = now_ns()
        gr["last_poll_abs_ns"] = now
        sat, values, details = gatemod.evaluate_gate(g, by_role)
        missing_roles = [r for r in gr["roles"] if not any(p["role"] == r for p in procs)]
        if errors or missing_roles:
            sat = False  # an unusable snapshot of any gate process is "not satisfied", never 0
        gr["prev_seqs"] = {pid: f["seq"] for pid, f in fields.items()}
        passed = gr["tracker"].update(sat, now - self.origin_ns)
        gr["polls"].append({"planned_rel_ns": planned, "actual_rel_ns": now - self.origin_ns,
                            "now_ns": now, "values": values, "details": details,
                            "satisfied": sat,
                            "snapshots": {str(k): v for k, v in fields.items()},
                            "errors": {str(k): v for k, v in errors.items()},
                            "roles_without_procs": missing_roles})
        if passed or gr["next"] >= len(gr["planned"]):
            self._gate_end()

    def _gate_end(self):
        gr = self.gate_run
        self.gate_run = None
        g, seq = gr["gate"], gr["seq"]
        tr = gr["tracker"]
        if tr.passed:
            action = "none"
        elif g["mode"] == "record_only":
            action = "recorded_only"
        else:
            action = "stopped"
        t = self.tt.tiers[seq]
        write_json_stamped(
            os.path.join(self.tree.tier_dir(seq, t.label), "drain_gate.json"),
            "capkit.drain_gate/1", self.ident.with_tier(seq, t.label),
            {"gate": g, "source": gr["source"], "polls": gr["polls"], "n_polls": len(gr["polls"]),
             "n_planned_polls": len(gr["planned"]), "passed": tr.passed,
             "t_pass_rel_ns": tr.t_pass_rel_ns, "start_rel_ns": t.start_ns,
             "deadline_rel_ns": gr["deadline_rel_ns"], "action": action})
        self.gate_outcomes.append({"seq": seq, "passed": tr.passed,
                                   "t_pass_rel_ns": tr.t_pass_rel_ns, "action": action})
        if action == "stopped":
            raise _GateTimeout(seq)

    # ---- thread inventory after warmup (INTERFACES 5.3) --------------------------------------
    def _thread_inventory(self):
        for p in self.procs:
            if not procfs.alive(p["pid"], p["starttime_ticks"], self.proc_root):
                continue
            inv = procfs.thread_inventory(p["pid"], self._declared_threads(p), self.proc_root)
            write_json_stamped(self.tree.threads_file(p["role"], p["pid"]), "capkit.threads/1",
                               self.ident, {"pid": p["pid"], "role": p["role"],
                                            "worker": p["worker"], "taken_mono_ns": now_ns(),
                                            "threads": inv, "rules_sha": procfs.RULES_SHA,
                                            "rules": list(procfs.THREAD_RULES)})

    # ---- main --------------------------------------------------------------------------------
    def run(self):
        """Run the launch; returns the process exit code. Always stops what it started."""
        self._prepare_tree()
        self.origin_ns = now_ns() + self.a.origin_lead_ms * 1_000_000
        self._write_launch_json()
        self._write_roles()
        self.tier_state, self.copy_jobs, self.gate_run = {}, [], None
        state, reason, code = "VALID", "completed all tiers", OK
        stop_reason, after = "completed", len(self.tt.tiers) - 1
        old = {}
        try:
            try:
                try:
                    for sg in (signal.SIGTERM, signal.SIGINT):
                        old[sg] = signal.signal(sg, self._on_signal)
                    self._startup()
                    self._timeline()
                finally:
                    self.cleaning = True  # from here on a stop signal never interrupts cleanup
            except _GateTimeout as e:
                state, code = "VALID", STOPPED_EARLY
                reason = "stopped: drain_gate_timeout in tier %d; later tiers not run" % e.seq
                stop_reason, after = "drain_gate_timeout", e.seq - 1
            except LaunchAbort as e:
                state, reason, code = e.state, e.reason, e.code
                stop_reason, after = getattr(e, "stop_reason", "aborted"), self.tiers_completed
            except BaseException as e:  # still stop every process we started
                self._stop_all("orchestrator_error: %r" % (e,), self.tiers_completed)
                self._verify_exit()
                raise
            self._stop_all(stop_reason, after)
            left = self._verify_exit()
            if left["after"]:
                state, code = "VOID", RUNTIME
                reason += "; leftover processes persisted after SIGKILL: %s" % left["after"]
            self._finalize(state, reason, left)
            return code
        finally:
            for s, h in old.items():
                signal.signal(s, h)

    def _on_signal(self, signum, _frame):
        """First stop signal: abort the launch (NOT_RUN) and go to cleanup. Any signal during
        cleanup (stop, verify, finalize) is recorded and only drops the remaining grace periods,
        so every started process is still stopped and the end files are still written."""
        self.signals.append({"signum": signum, "t_mono_ns": now_ns(),
                             "during_cleanup": self.cleaning})
        if self.cleaning:
            self.hurry = True
            return
        self.cleaning = True
        e = LaunchAbort("NOT_RUN", "orchestrator received signal %d" % signum, STOPPED_EARLY)
        e.stop_reason = "orchestrator_signalled"
        raise e

    def _abort(self, state, reason, code, stop_reason):
        e = LaunchAbort(state, reason, code)
        e.stop_reason = stop_reason
        return e

    def _startup(self):
        if not os.path.isdir(self.a.job_env_dir):
            raise self._abort("NOT_RUN", "--job-env-dir %s is not a directory (job-level env "
                              "results missing)" % self.a.job_env_dir, STOPPED_EARLY,
                              "job_env_dir_missing")
        if self.overlay_bytes is None and self.overlay_needed_by:
            raise self._abort("NOT_RUN", "overlay manifest %s is missing; roles with a probe tier: "
                              "%s" % (self.a.overlay_manifest, ", ".join(self.overlay_needed_by)),
                              STOPPED_EARLY, "overlay_manifest_missing")
        pending = list(self.order)
        while True:
            for n in list(pending):
                if all(self.rt[d].ready for d in self.roles[n]["start_after"]):
                    self._start_role(n)
                    pending.remove(n)
            for n, rt in self.rt.items():
                if rt.pid is not None and not rt.ready:
                    try:
                        self._check_ready(n)
                    except (ReadyError, ConfigError) as e:
                        raise self._abort("NOT_RUN", "role %s: %s" % (n, e), STOPPED_EARLY,
                                          "not_ready_before_origin")
            self._discover()
            try:
                self._check_early_exit(None)
            except LaunchAbort as e:
                if getattr(e, "stop_reason", None) == "orchestrator_signalled":
                    raise
                raise self._abort("NOT_RUN", "before origin: " + e.reason, STOPPED_EARLY,
                                  "not_ready_before_origin")
            if not pending and all(rt.ready for rt in self.rt.values()):
                break
            now = now_ns()
            if now >= self.origin_ns:
                late = sorted(n for n, rt in self.rt.items() if not rt.ready)
                raise self._abort("NOT_RUN", "not ready before origin: %s" % late,
                                  STOPPED_EARLY, "not_ready_before_origin")
            time.sleep(min(self.hk_ns, self.origin_ns - now) / 1e9)
        self._log("all_ready", lead_left_ns=self.origin_ns - now_ns())

    def _events(self):
        o, ev = self.origin_ns, []
        for t in self.tt.tiers:
            ev.append((o + t.start_ns, t.seq, 0, "tier_start"))
            ev.append((o + t.window_start_ns, t.seq, 1, "wstart"))
            ev.append((o + t.window_end_ns, t.seq, 2, "wend"))
            ev.append((o + t.end_ns, t.seq, 3, "tier_end"))
        ev.append((o + self.tt.tiers[0].end_ns, 0, 4, "thread_inventory"))
        return sorted(ev)

    def _timeline(self):
        ev = self._events()
        end_abs = self.origin_ns + self.tt.end_ns
        i, next_hk = 0, now_ns()
        while i < len(ev) or self.gate_run is not None or self.copy_jobs:
            now = now_ns()
            while i < len(ev) and ev[i][0] <= now:
                t_abs, seq, _p, kind = ev[i]
                i += 1
                if kind == "tier_start" and self.tt.tiers[seq].kind == "drain":
                    self._gate_begin(seq)
                elif kind in ("wstart", "wend"):
                    self._boundary(seq, kind, t_abs)
                elif kind == "tier_end":
                    self.tiers_completed = seq
                    self._log("tier_end", seq=seq)
                elif kind == "thread_inventory":
                    self._thread_inventory()
            if self.gate_run is not None and self._gate_next_abs() <= now_ns():
                self._gate_poll()
            self._service_copies()
            if now_ns() >= next_hk:
                self._discover()
                self._check_early_exit(end_abs)
                next_hk += self.hk_ns
            wake = [next_hk]
            if i < len(ev):
                wake.append(ev[i][0])
            if self.gate_run is not None:
                wake.append(self._gate_next_abs())
            if self.copy_jobs:
                wake.append(now_ns() + max(self.snap_ns // 8, 1_000_000))
            delay = min(wake) - now_ns()
            if delay > 0:
                time.sleep(delay / 1e9)
        self._wait_clients(end_abs)

    def _wait_clients(self, end_abs):
        """After the last tier, clients finish on their own within stop grace (INTERFACES 2.3)."""
        limit = now_ns() + int((self.scen["stop_grace_s"] + self.a.kill_after_s) * 1e9)
        while now_ns() < limit:
            self._check_early_exit(end_abs)
            if all(rt.returncode is not None for rt in self.rt.values()
                   if rt.decl["kind"] == "client"):
                return
            time.sleep(self.hk_ns / 1e9)
        self._log("clients_not_exited", roles=[n for n, rt in self.rt.items()
                                               if rt.decl["kind"] == "client"
                                               and rt.returncode is None])

    # ---- stop, verify, finalize --------------------------------------------------------------
    def _stop_all(self, reason, after_seq):
        if not os.path.exists(self.tree.stop_json):
            self.stop_doc = {"schema": STOP_SCHEMA, "reason": reason, "t_mono_ns": now_ns(),
                             "after_tier_seq": after_seq, "written_wall": wall_iso()}
            atomic_write_bytes(self.tree.stop_json, (dumps(self.stop_doc) + "\n").encode())
        grace = self.scen["stop_grace_s"] + self.a.kill_after_s
        for name in self.scen["stop_order"]:
            rt = self.rt[name]
            if rt.popen is None:
                continue
            if rt.popen.poll() is None:
                try:
                    rt.popen.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
                limit = time.monotonic() + grace
                while rt.popen.poll() is None and time.monotonic() < limit and not self.hurry:
                    time.sleep(0.02)
                if rt.popen.poll() is None:
                    rt.forced = ("SIGKILL to process group after a stop signal during cleanup"
                                 if self.hurry else
                                 "SIGKILL to process group after %.1fs" % grace)
                    try:  # popen.pid: a stop signal can land before rt.pid is assigned
                        os.killpg(rt.popen.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        rt.popen.kill()
                    rt.popen.wait()
            if rt.returncode is None:
                rt.returncode, rt.exited_ns = rt.popen.returncode, now_ns()
        self._log("stopped", reason=reason)

    def _live_ours(self):
        live = {}
        for p in self.procs:
            if procfs.alive(p["pid"], p["starttime_ticks"], self.proc_root):
                live[p["pid"]] = "registered:%s" % p["role"]
        for pid in procfs.marked_pids(self.mark, self.proc_root, exclude=[os.getpid()]):
            live.setdefault(pid, "launch_mark")
        return live

    def _verify_exit(self):
        """Every registered or marked process must be gone; kill leftovers and record them."""
        before = self._live_ours()
        if before:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if sig == signal.SIGTERM and self.hurry:
                    continue
                for pid in list(self._live_ours()):
                    try:
                        os.kill(pid, sig)
                    except (ProcessLookupError, PermissionError):
                        pass
                limit = time.monotonic() + self.a.kill_after_s
                while self._live_ours() and time.monotonic() < limit:
                    time.sleep(0.05)
        after = self._live_ours()
        res = {"before": [{"pid": k, "why": v} for k, v in sorted(before.items())],
               "after": [{"pid": k, "why": v} for k, v in sorted(after.items())],
               "marker_scan": "environ entry %s=<token in launch.json launch_mark>" % MARK_ENV}
        self._log("verify_exit", **res)
        return res

    def _versions(self):
        """Copy every process's version self-report into raw/env/versions/ (INTERFACES 5.5)."""
        files = list_snapshot_files(self.shm_dir)
        out = {}
        for p in self.procs:
            rec, src = None, None
            path = files.get(p["pid"])
            if path is not None:
                try:
                    with open(path, "rb") as f:
                        meta = decode_snapshot(f.read()).meta
                    if meta.get("starttime_ticks") == p["starttime_ticks"]:
                        rec, src = meta.get("versions"), "probe.snapshot"
                except (OSError, SnapshotError):
                    pass
            rt = self.rt[p["role"]]
            if rec is None and p["pid"] == rt.pid and rt.ready_doc is not None:
                rec, src = rt.ready_doc.get("versions"), "ready_file"
            out[(p["role"], p["worker"], p["pid"])] = rec
            if rec is None:
                continue
            ident = self.ident.with_role(p["role"]).with_worker(p["worker"]).with_pid(p["pid"])
            write_json_stamped(self.tree.env("versions", "%s.w%d.%d.json"
                                             % (p["role"], p["worker"], p["pid"])),
                               "capkit.version_record/1", ident,
                               {"versions": rec, "source": src})
        return out

    ENV_PREFIXES = ("TLLM_", "TRTLLM_", "CAPPROBE_", "CAPKIT_", "PROTOCOL_BUFFERS_", "GRPC_",
                    "NVIDIA_", "CUDA_", "OMP_", "PYTHON", "LD_LIBRARY_PATH", "UV_", "SLURM_JOB_ID",
                    "PMIX_", "PMI_")

    def _canary_baseline(self):
        seq = self.tt.baseline_seq()
        path = self.tree.canary("canary.jsonl")
        if seq is None or not os.path.exists(path):
            return None
        from capcommon.jsonio import read_jsonl  # noqa: PLC0415
        try:
            hdr, rows = read_jsonl(path)
            cols = hdr["columns"]
            ti, ii = cols.index("t_rel_ns"), cols.index("iters")
        except (ConfigError, ValueError, KeyError, OSError):
            return None
        ws, we = self.tt.window_of(seq)
        vals = sorted(r[ii] for r in rows if ws <= r[ti] < we and isinstance(r[ii], int))
        if not vals:
            return None
        return {"tier_seq": seq, "column": "iters", "n_rows": len(vals),
                "median_per_row": vals[len(vals) // 2], "min": vals[0], "max": vals[-1]}

    def _dims(self, versions):
        hw = None
        hp = self.tree.env("hw_profile.json")
        if os.path.exists(hp):
            try:
                hw = read_json(hp)
            except (OSError, ValueError):
                hw = None

        def hwk(k):
            v = (hw or {}).get(k)
            return v
        env = {k: v for k, v in sorted(os.environ.items()) if k.startswith(self.ENV_PREFIXES)}
        role_env = {n: d["env"] for n, d in self.roles.items()}
        gc = {"%s.w%d.%d" % k: (v or {}).get("gc_enabled") for k, v in versions.items()}
        loops = {"%s.w%d.%d" % k: (v or {}).get("loop_impl") for k, v in versions.items()}
        pbs = {"%s.w%d.%d" % k: (v or {}).get("protobuf_impl") for k, v in versions.items()}
        d = self.scen["dims"]
        axis = self.tt.axis
        return {
            "code_version": {"capkit_sha": self.a.capkit_sha,
                             "baseline_sha": _opt(self.a.baseline_sha),
                             "probe_tree_sha": _opt(self.a.probe_tree_sha)},
            "image_sha": _opt(self.a.image_sha256), "node": os.uname().nodename,
            "machine_model": hwk("chassis_model"), "hw_profile_sha": _opt(self.a.hw_profile_sha),
            "cores_numa": None if hw is None else {k: hwk(k) for k in (
                "sockets", "cores_per_socket", "threads_per_core", "numa_layout")},
            "canary_rate_baseline": self._canary_baseline(), "tokenizer": d["tokenizer"],
            "pin_table_sha": self.pin_sha,
            "env_vars": {"orchestrator": env, "roles": role_env,
                         "injected_for_product_roles": ["TLLM_NUMA_AWARE_WORKER_AFFINITY=0",
                                                        "CAPPROBE_TIER", "CAPPROBE_GROUPS",
                                                        "CAPPROBE_SHM_DIR", "CAPPROBE_INTERVAL_MS",
                                                        "CAPPROBE_IDENTITY_FILE",
                                                        "CAPPROBE_ROLE_HINT",
                                                        probe_entry.ENV_VAR]},
            "stub": d["stub"],
            "load_shape": {"tier_table": self.tt.name, "axis": axis, "dist": self.tt.dist,
                           "seed": self.tt.seed,
                           "load_values": [[t.seq, t.kind, t.label, t.load_value]
                                           for t in self.tt.tiers]},
            "probe_tier": {"tier": self.scen["probe_tier"], "groups": self.scen["probe_groups"]},
            "log_sink": d["log_sink"], "gc": gc,
            "response_timing_switches": d["response_timing_switches"],
            "loop_impls": loops, "protobuf_impls": pbs, "interpretation": d["interpretation"]}

    def _finalize(self, state, reason, left):
        versions = self._versions()
        o = self.origin_ns
        payload = {
            "stop": self.stop_doc, "state": state, "reason": reason,
            "ready_at_rel_ns": {n: (None if rt.ready_ns is None else rt.ready_ns - o)
                                for n, rt in self.rt.items()},
            "tiers_completed": self.tiers_completed, "gate_outcomes": self.gate_outcomes,
            "exit_codes": {n: rt.returncode for n, rt in self.rt.items()},
            "forced_kills": {n: rt.forced for n, rt in self.rt.items() if rt.forced},
            "early_exit": self.early_exit, "leftovers": left, "signals": self.signals,
            "argv": {n: rt.argv for n, rt in self.rt.items()},
            "versions_missing": ["%s.w%d.%d" % k for k, v in versions.items() if v is None],
            "events": self.events_log, "ended_mono_ns": now_ns()}
        for seq in sorted(self.tier_state):
            self._maybe_write_tier(seq, force_reason="launch stopped before the boundary "
                                                     "work of this tier completed")
        write_json_stamped(
            self.tree.procs("monitor_children.json"), "capkit.monitor_children/1", self.ident,
            {"monitor_classes": list(MONITOR_CLASSES), "n_proc_scans": self.n_proc_scans,
             "coverage": "processes caught by a /proc scan (housekeeping or window boundary); "
                         "a child that lived only between two scans is not listed",
             "children": sorted(self.monitor_children.values(),
                                key=lambda e: (e["first_seen_mono_ns"], e["pid"]))})
        write_json_stamped(self.tree.launch_end_json, "capkit.launch_end/1", self.ident, payload)
        write_json_stamped(self.tree.dims_json, "capkit.dims/1", self.ident,
                           self._dims(versions))
        append_state(self.tree.root, state, reason.replace("\t", " ").replace("\n", " "),
                     "caplaunch")

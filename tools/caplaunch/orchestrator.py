# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capkit-launch: one launch = start roles -> warmup -> tiers -> stop -> collect (design 2.8).

INTERFACES.md sections 0, 2.3, 5.1-5.4, 5.7, 8 define what is written here. The origin is chosen
before the first process starts; every role must be ready before it (no separate ready timeout).
Between tiers, drain tiers carry the C11 drain gate; the next tier's start time is fixed by the
tier table, so the orchestrator never signals tools to switch tiers.
"""

import os
import secrets
import shutil
import signal
import subprocess
import sys
import time

from capcommon import gate as gatemod
from capcommon.capbin import read_capbin
from capcommon.clock import now_ns, wall_iso
from capcommon.exitcodes import OK, RUNTIME, STOPPED_EARLY, ConfigError
from capcommon.hashing import is_git_sha, is_sha256_hex, sha256_file
from capcommon.identity import Identity
from capcommon.jsonio import atomic_write_bytes, dumps, read_json, write_json_stamped
from capcommon.lifecycle import STOP_SCHEMA
from capcommon.runtree import RunTree, append_state, launch_shm_subdir
from capcommon.snapshot import SnapshotError, decode_snapshot, list_snapshot_files
from capcommon.tiers import TierTable

from . import procfs
from . import scenario as sc
from .ready import (ReadyError, http_get_status, read_capkit_port, read_capkit_ready,
                    read_plain_port, read_report_addr)

MARK_ENV = "CAPKIT_LAUNCH_MARK"
NONE = "none"  # explicit "not applicable" value for the provenance arguments


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
        self.hk_ns = args.housekeeping_ms * 1_000_000
        self.W = os.path.realpath(args.workdir)
        if not os.path.isdir(self.W):
            raise ConfigError("--workdir %s is not a directory" % args.workdir)
        self._check_provenance()
        self._load_inputs()
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
        self.params = {}
        for kv in a.param:
            k, sep, v = kv.partition("=")
            if not sep or not k:
                raise ConfigError("--param must be KEY=VALUE, got %r" % kv)
            if k in self.params:
                raise ConfigError("--param %s given twice" % k)
            self.params[k] = v
        req = set(self.scen["params_required"])
        if set(self.params) != req:
            raise ConfigError("--param keys %s != scenario params_required %s"
                              % (sorted(self.params), sorted(req)))
        self.rt = {n: RoleRT(d, self.cpus_of[n]) for n, d in self.roles.items()}
        self.order = sc.start_order(self.roles)

    # ---- tree, paths, launch.json -----------------------------------------------------------
    def _prepare_tree(self):
        T = self.tree
        if os.path.exists(T.launch_json):
            raise ConfigError("%s already has raw/launch.json (one launch per tree)" % T.root)
        T.make_skeleton()
        os.makedirs(T.procs("identity"), exist_ok=True)
        self.shm_dir = launch_shm_subdir(self.a.shm_dir, self.a.restart_count, self.a.launch_seq)
        if os.path.isdir(self.shm_dir) and os.listdir(self.shm_dir):
            raise ConfigError("snapshot dir %s is not empty (stale launch?)" % self.shm_dir)
        os.makedirs(self.shm_dir, exist_ok=True)
        if self.a.job_env_dir is not None:  # job-level gate results -> this tree (INTERFACES 5.1)
            src = self.a.job_env_dir
            for n in sorted(os.listdir(src)):
                p = os.path.join(src, n)
                if os.path.isfile(p):
                    shutil.copyfile(p, T.env(n))
        if not os.path.exists(T.env("pin_table.json")):
            shutil.copyfile(self.a.pin_table, T.env("pin_table.json"))
        self.own = {}
        for n in self.roles:
            self.own[n] = {"ready_file": T.ready_file(n),
                           "port_file": T.procs("ports", "%s.port" % n),
                           "report_addr": T.procs("ports", "%s.addr" % n)}
            os.makedirs(os.path.join(T.control_dir, "cwd", n), exist_ok=True)

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
            inputs={k: os.path.realpath(self.inputs[k]["path"])
                    for k in ("tier_table", "schedule", "prompts")})

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
            "scenario": {"relpath": self.scenario_rel, "sha256": self.scen_sha,
                         "doc": self.scen},
            "code": {"capkit_sha": a.capkit_sha, "probe_tree_sha": _opt(a.probe_tree_sha),
                     "baseline_sha": _opt(a.baseline_sha), "snapshot_dir": _opt(a.snapshot_dir)},
            "image": {"path": _opt(a.image_path), "sha256": _opt(a.image_sha256)},
            "probe_tier": {"tier": self.scen["probe_tier"], "groups": self.scen["probe_groups"]},
            "roles": roles, "reconcile": self.scen["reconcile"],
            "hw_profile_sha": _opt(a.hw_profile_sha), "submission": sub,
            "pin_table": {"path": a.pin_table, "sha256": self.pin_sha},
            "params": self.params, "launch_mark": self.mark,
            "argv_runtime_placeholders": "{addr:<role>} is resolved when the role starts; "
                                         "the resolved command line is in procs/roles.json",
            "orchestrator": {"python": sys.executable, "taskset": self.taskset,
                             "snapshot_interval_ms": a.snapshot_interval_ms,
                             "housekeeping_ms": a.housekeeping_ms,
                             "kill_after_s": a.kill_after_s, "argv": list(sys.argv),
                             "thread_rules_sha": procfs.RULES_SHA},
        }
        write_json_stamped(self.tree.launch_json, "capkit.launch/1", self.ident, payload)
        for n, d in self.roles.items():
            if d["launcher"] == "product":
                ident = self.ident.with_role(n).to_dict()
                ident["pid"] = None
                atomic_write_bytes(self.tree.procs("identity", "%s.json" % n),
                                   (dumps({"schema": "capkit.identity/1",
                                           "identity": ident}) + "\n").encode())

    def _write_roles(self):
        self.generation += 1
        write_json_stamped(self.tree.roles_json, "capkit.roles/1", self.ident,
                           {"generation": self.generation, "procs": self.procs,
                            "unattributed": self.unattributed})

    # ---- processes ---------------------------------------------------------------------------
    def _role_env(self, d):
        env = dict(os.environ)
        env.update(d["env"])
        k, _, v = self.mark.partition("=")
        env[k] = v
        if d["launcher"] == "product":
            if not d.get("keep_numa_affinity_env", False):
                env["TLLM_NUMA_AWARE_WORKER_AFFINITY"] = "0"  # D2: worker must not unpin
            for key, val in (("CAPPROBE_TIER", self.scen["probe_tier"]),
                             ("CAPPROBE_GROUPS", ",".join(self.scen["probe_groups"]))):
                if key in os.environ and os.environ[key] != val:
                    raise ConfigError("environment %s=%r conflicts with scenario %r"
                                      % (key, os.environ[key], val))
                env[key] = val
            env["CAPPROBE_SHM_DIR"] = self.shm_dir
            env["CAPPROBE_INTERVAL_MS"] = str(self.a.snapshot_interval_ms)
            env["CAPPROBE_IDENTITY_FILE"] = self.tree.procs("identity", "%s.json" % d["role"])
            env["CAPPROBE_ROLE_HINT"] = d["role"]
        return env

    def _register(self, pid, role, worker, parent, via, cpus):
        st = procfs.starttime_ticks(pid, self.proc_root)
        for p in self.procs:
            if p["pid"] == pid and p["starttime_ticks"] == st:
                return p
        entry = {"pid": pid, "starttime_ticks": st, "role": role, "worker": worker,
                 "parent_pid": parent, "cmdline": procfs.cmdline(pid, self.proc_root),
                 "cpus": cpus, "cpus_readback": procfs.cpus_allowed_list(pid, None,
                                                                          self.proc_root),
                 "discovered_via": via, "registered_mono_ns": now_ns()}
        self.procs.append(entry)
        self.unattributed = [u for u in self.unattributed
                             if not (u["pid"] == pid and u["starttime_ticks"] == st)]
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
        with open(tmp_out, "wb") as fo, open(tmp_err, "wb") as fe:
            try:
                rt.popen = subprocess.Popen(full, env=self._role_env(d), stdout=fo, stderr=fe,
                                            stdin=subprocess.DEVNULL, start_new_session=True,
                                            cwd=os.path.join(self.tree.control_dir, "cwd",
                                                             name))
            except OSError as e:
                raise LaunchAbort("NOT_RUN", "cannot start role %s: %s" % (name, e), RUNTIME)
        rt.pid = rt.popen.pid
        rt.started_ns = now_ns()
        rt.argv = full
        out, err = (self.tree.log_file(name, 0, rt.pid, s) for s in ("out", "err"))
        os.replace(tmp_out, out)
        os.replace(tmp_err, err)
        rt.logs = [out, err]
        rt.starttime = procfs.starttime_ticks(rt.pid, self.proc_root)
        self._register(rt.pid, name, 0, os.getpid(), "launched", rt.cpus)
        self._log("start", role=name, pid=rt.pid, argv=full)

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

    def _discover(self):
        """Register product children that self-report via probe snapshots; list unregistered
        descendants (reported only, never given a role: INTERFACES 5.3)."""
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
            elif anc is None and not any(u["pid"] == pid and u["starttime_ticks"] == st
                                         for u in self.unattributed):
                self.unattributed.append({"pid": pid, "starttime_ticks": st,
                                          "seen_via": "probe.snapshot",
                                          "argv": snap.meta.get("argv"),
                                          "role_hint": snap.meta.get("role_hint")})
                changed = True
        tops = [rt.pid for rt in self.rt.values() if rt.pid is not None]
        known = {(p["pid"], p["starttime_ticks"]) for p in self.procs}
        for pid in procfs.descendants(tops, self.proc_root):
            st = procfs.starttime_ticks(pid, self.proc_root)
            if st is None or (pid, st) in known:
                continue
            if any(u["pid"] == pid and u["starttime_ticks"] == st for u in self.unattributed):
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
        """Registered procs that are expected to write snapshots (role snapshots scope)."""
        out = []
        for p in self.procs:
            d = self.roles[p["role"]]
            if roles is not None and p["role"] not in roles:
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

    def _scan(self):
        """pid set, threads of every registered live process, NIC counters."""
        tops = [rt.pid for rt in self.rt.values() if rt.pid is not None]
        live = []
        for p in self.procs:
            if procfs.alive(p["pid"], p["starttime_ticks"], self.proc_root):
                live.append([p["pid"], p["starttime_ticks"]])
        desc = []
        for pid in procfs.descendants(tops, self.proc_root):
            desc.append([pid, procfs.starttime_ticks(pid, self.proc_root)])
        threads = {}
        for p in self.procs:
            if [p["pid"], p["starttime_ticks"]] in live:
                inv = procfs.thread_inventory(p["pid"], self._declared_threads(p), self.proc_root)
                threads[str(p["pid"])] = [{k: t[k] for k in ("tid", "comm", "py_name", "class",
                                                              "identified_as", "cpus_allowed")}
                                          for t in inv]
        return {"t_mono_ns": now_ns(), "registered_live": live,
                "registered_total": len(self.procs), "descendants": sorted(desc),
                "threads": threads, "nic_counters": procfs.nic_counters(self.sys_root)}

    def _boundary(self, seq, which, boundary_abs):
        """Start of a window boundary: scan now, queue snapshot copies (first >= boundary)."""
        st = self.tier_state.setdefault(seq, {"scans": {}, "copies": [], "pending": 0})
        st["scans"][which] = self._scan()
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
                                                      "registered_total", "descendants")}
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
                         "tracker": gatemod.GateTracker(g),
                         "roles": sorted({term["role"] for r in g["readings"]
                                          for term in r["terms"]})}
        if not polls:
            self._gate_end()

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
        old = {s: signal.signal(s, self._on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            try:
                self._startup()
                self._timeline()
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
        e = LaunchAbort("NOT_RUN", "orchestrator received signal %d" % signum, STOPPED_EARLY)
        e.stop_reason = "orchestrator_signalled"
        raise e

    def _abort(self, state, reason, code, stop_reason):
        e = LaunchAbort(state, reason, code)
        e.stop_reason = stop_reason
        return e

    def _startup(self):
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
            gr = self.gate_run
            if gr is not None and self.origin_ns + gr["planned"][gr["next"]] <= now_ns():
                self._gate_poll()
            self._service_copies()
            if now_ns() >= next_hk:
                self._discover()
                self._check_early_exit(end_abs)
                next_hk += self.hk_ns
            wake = [next_hk]
            if i < len(ev):
                wake.append(ev[i][0])
            gr = self.gate_run
            if gr is not None:
                wake.append(self.origin_ns + gr["planned"][gr["next"]])
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
                while rt.popen.poll() is None and time.monotonic() < limit:
                    time.sleep(0.02)
                if rt.popen.poll() is None:
                    rt.forced = "SIGKILL to process group after %.1fs" % grace
                    try:
                        os.killpg(rt.pid, signal.SIGKILL)
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
                                                        "CAPPROBE_ROLE_HINT"]},
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
            "early_exit": self.early_exit, "leftovers": left,
            "argv": {n: rt.argv for n, rt in self.rt.items()},
            "versions_missing": ["%s.w%d.%d" % k for k, v in versions.items() if v is None],
            "events": self.events_log, "ended_mono_ns": now_ns()}
        for seq in sorted(self.tier_state):
            self._maybe_write_tier(seq, force_reason="launch stopped before the boundary "
                                                     "work of this tier completed")
        write_json_stamped(self.tree.launch_end_json, "capkit.launch_end/1", self.ident, payload)
        write_json_stamped(self.tree.dims_json, "capkit.dims/1", self.ident,
                           self._dims(versions))
        append_state(self.tree.root, state, reason.replace("\t", " ").replace("\n", " "),
                     "caplaunch")

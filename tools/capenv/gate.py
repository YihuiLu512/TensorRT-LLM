# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capenv.gate: job-level environment gate (RESOURCES.md 3 and 5; INTERFACES.md 5.5, 11).

Run inside the same job step and container as the processes under test. Writes into
<run-dir>/env/ (the job directory) and prints ONE JSON verdict line:
    {"verdict": "PASS" | "ENV_INVALID", "reasons": [...], ...}
Exit 0 on PASS, 6 on ENV_INVALID, 3 on bad configuration.

Items (RESOURCES.md 3, "in-job exclusivity"), each with the command, rc, stdout sha and source:
  1 scontrol show job $JOB    rc 0; OverSubscribe=NO; requested GPUs (ReqTRES; without it the
    allocation, TRES / AllocTRES) <= --max-alloc-gpus. On a whole-node allocation Slurm puts
    every GPU of the node into AllocTRES although one was requested; that number is recorded
    (alloc_gpus), and what the job can use is checked by item 5 (visible GPUs == requested)
  2 scontrol show node $NODE  rc 0; CPUAlloc == CPUTot
  3 squeue -h -w $NODE -o '%i %u %T'   rc 0; own job listed (positive control); only that line
  4 /proc/self/status Cpus_allowed_list covers every online cpu
  5 nvidia-smi (--nvidia-smi): visible GPUs == --max-alloc-gpus (the number requested) and no
    compute process, by capenv.gpu.check_visible: requesting >= 1 needs a listed GPU first
    (positive control, else NOT_RUN); requesting 0 needs nvidia-smi to report no GPU (the
    positive control and the compute-process check are then NOT_RUN, "needs a listed GPU")
  6 idle window: per-core /proc/stat busy fraction <= --idle-busy-max-frac on every online
    core over --idle-window-s (pre phase only; the sidecar keeps watching during the run); each
    core's irq and softirq ticks are recorded too (the idle reading of D14, capjudge.pilot)
Items 1-3 fall back to <job>/env/slurm_view.json (written from the login node by capjobs)
when the command is missing or fails in the container; the gate waits for that file at most
--slurm-view-wait-s seconds. Any item that is not PASS (RED or NOT_RUN) makes the gate
ENV_INVALID: an item that could not be checked never counts as passed.

Pre phase also writes hw_gate.json (node profile sha vs the registered main-model sha; a
mismatch passes only if every differing identity field is registered comparable in
--comparable-file). --registered-hw-sha UNREGISTERED is the first job on a model that has no
registered profile yet: the gate still reads the profile (hw_gate.json gets node_hw_sha, status
NOT_RUN "profile not registered") and runs every item, and the verdict is always ENV_INVALID
(exit 6), so the job stops after the gate and the profile can be registered. The pre phase also
writes memguard.json (memory limit read-back for this process and for every
pid in --roles-file when given). Post phase repeats items 1-5 into exclusivity.post.json.
"""

import argparse
import json
import os
import sys
import time

from capcommon import exitcodes
from capcommon.cli import add_identity_args
from capcommon.exitcodes import ConfigError
from capcommon.hashing import is_sha256_hex, sha256_bytes, sha256_file
from capcommon.hwprofile import diff_identity, profile_sha256, validate_profile
from capcommon.identity import Identity
from capcommon.jsonio import read_json, write_json_stamped

from . import gpu as gpu_mod, memguard, procfs, roles as roles_mod, runcmd
from .hwprofile import load_profile, parse_kv_tokens

EXCL_SCHEMA = "capkit.exclusivity/1"
HWGATE_SCHEMA = "capkit.hw_gate/1"
UNREGISTERED = "UNREGISTERED"  # --registered-hw-sha value: collect the profile, never pass
GATE_SCHEMA = "capkit.env_gate/1"
COMPARABLE_SCHEMA = "capkit.hw_comparable/1"
SLURM_VIEW_SCHEMA = "capkit.slurm_view/1"


def _item(i, cmd, rc, stdout, source, passed, detail, status=None):
    return {"id": i, "cmd": cmd, "rc": rc,
            "stdout_sha": sha256_bytes(stdout.encode()) if isinstance(stdout, str) else None,
            "passed": bool(passed), "status": status or ("PASS" if passed else "RED"),
            "detail": detail, "source": source}


def _gpus_in_tres(tres):
    """GPUs in a TRES string ('cpu=224,mem=1T,node=1,gres/gpu=1'): the gres/gpu total if
    present, else the sum of typed gres/gpu:<type> entries, else 0."""
    total, typed = None, 0
    for part in (tres or "").split(","):
        k, _, v = part.partition("=")
        if not v.isdigit():
            continue
        if k == "gres/gpu":
            total = int(v)
        elif k.startswith("gres/gpu:"):
            typed += int(v)
    return total if total is not None else typed


class SlurmSource:
    """Runs scontrol/squeue in the container, or reads slurm_view.json when they fail."""

    def __init__(self, view_file, wait_s, runner, timeout_s, sleep=time.sleep):
        self.view_file, self.wait_s = view_file, wait_s
        self.runner, self.timeout_s, self.sleep = runner, timeout_s, sleep
        self._view = None
        self.view_error = None

    def view(self):
        if self._view is not None or self.view_error is not None:
            return self._view
        end = time.monotonic() + self.wait_s
        while not os.path.exists(self.view_file):
            if time.monotonic() >= end:
                self.view_error = "slurm_view file did not appear within %.1fs" % self.wait_s
                return None
            self.sleep(min(1.0, max(0.05, self.wait_s / 50)))
        try:
            doc = read_json(self.view_file)
            if doc.get("schema") != SLURM_VIEW_SCHEMA:
                raise ConfigError("schema %r" % doc.get("schema"))
            self._view = {it["id"]: it for it in doc.get("items", [])}
        except (ConfigError, ValueError, KeyError, OSError, AttributeError) as e:
            self.view_error = "slurm_view unreadable: %s" % e
        return self._view

    def get(self, item_id, argv):
        """(stdout, rc, cmd, source, why_not)."""
        rec = self.runner(argv, self.timeout_s)
        if runcmd.ok(rec):
            return rec["stdout"], 0, argv, "container", None
        local_why = runcmd.why(rec)
        v = self.view()
        if v is None:
            return None, rec.get("rc"), argv, "container", "%s; %s" % (local_why,
                                                                        self.view_error)
        it = v.get(item_id)
        if it is None:
            return None, None, argv, "slurm_view", "%s; slurm_view has no item %d" % (
                local_why, item_id)
        return it.get("stdout"), it.get("rc"), it.get("cmd"), "slurm_view", \
            (None if it.get("rc") == 0 else "slurm_view item rc=%s" % it.get("rc"))


def item1(src, job_id, max_gpus):
    out, rc, cmd, source, why = src.get(1, ["scontrol", "show", "job", job_id])
    if why or rc != 0:
        return _item(1, cmd, rc, out, source, False, {"why": why or "rc %s" % rc}), {}
    kv = parse_kv_tokens(out)
    rec = {k: kv.get(k) for k in ("JobId", "OverSubscribe", "NumCPUs", "NumNodes", "NodeList",
                                  "TRES", "ReqTRES", "AllocTRES", "TresPerNode", "Partition")}
    alloc = kv.get("TRES") or kv.get("AllocTRES")
    rec["alloc_gpus"] = _gpus_in_tres(alloc) if alloc else None
    # the requested count; an output without ReqTRES is checked on the allocation (never less)
    rec["gpus_checked_from"] = "ReqTRES" if kv.get("ReqTRES") else (
        "TRES" if kv.get("TRES") else "AllocTRES")
    tres = kv.get(rec["gpus_checked_from"])
    gpus = _gpus_in_tres(tres)
    rec["req_gpus"] = gpus if tres else None
    bad = []
    if not tres:  # no TRES at all is an unreadable allocation, not "0 GPUs"
        bad.append("no ReqTRES / TRES / AllocTRES in the scontrol output: GPU request "
                   "unreadable")
    if kv.get("JobId") != str(job_id):
        bad.append("JobId %r != %r (wrong job queried)" % (kv.get("JobId"), job_id))
    if kv.get("OverSubscribe") != "NO":
        bad.append("OverSubscribe=%r, need NO (exclusive)" % kv.get("OverSubscribe"))
    if tres and gpus > max_gpus:
        bad.append("%s GPUs %d > --max-alloc-gpus %d" % (rec["gpus_checked_from"], gpus,
                                                         max_gpus))
    rec["problems"] = bad
    return _item(1, cmd, rc, out, source, not bad, rec), kv


def item2(src, node):
    out, rc, cmd, source, why = src.get(2, ["scontrol", "show", "node", node])
    if why or rc != 0:
        return _item(2, cmd, rc, out, source, False, {"why": why or "rc %s" % rc})
    kv = parse_kv_tokens(out)
    rec = {k: kv.get(k) for k in ("NodeName", "CPUAlloc", "CPUTot", "Sockets", "ThreadsPerCore",
                                  "RealMemory", "Gres", "State", "Partitions")}
    bad = []
    if kv.get("NodeName") != node:
        bad.append("NodeName %r != %r" % (kv.get("NodeName"), node))
    if kv.get("CPUAlloc") is None or kv.get("CPUAlloc") != kv.get("CPUTot"):
        bad.append("CPUAlloc %r != CPUTot %r" % (kv.get("CPUAlloc"), kv.get("CPUTot")))
    rec["problems"] = bad
    return _item(2, cmd, rc, out, source, not bad, rec)


def item3(src, job_id, node):
    out, rc, cmd, source, why = src.get(3, ["squeue", "-h", "-w", node, "-o", "%i %u %T"])
    if why or rc != 0:
        it = _item(3, cmd, rc, out, source, False, {"why": why or "rc %s" % rc})
        it["positive_control_ok"] = False
        return it
    lines = [ln.split() for ln in out.splitlines() if ln.strip()]
    own = [ln for ln in lines if ln and ln[0] == str(job_id)]
    others = [ln for ln in lines if not ln or ln[0] != str(job_id)]
    pc = len(own) >= 1
    bad = []
    if not pc:
        bad.append("own job %s not listed: the query itself is not trustworthy" % job_id)
    if others:
        bad.append("other jobs on the node: %s" % [" ".join(x) for x in others])
    it = _item(3, cmd, rc, out, source, pc and not others,
               {"lines": len(lines), "own": len(own), "others": others, "problems": bad},
               status=None if pc else "NOT_RUN")
    it["positive_control_ok"] = pc
    return it


def item4(proc_root=procfs.PROC, sys_root=procfs.SYS):
    st, err = procfs.try_read_text(os.path.join(proc_root, "self/status"))
    on, err2 = procfs.try_read_text(os.path.join(sys_root, "devices/system/cpu/online"))
    if st is None or on is None:
        return _item(4, ["read", "/proc/self/status", "/sys/devices/system/cpu/online"], None,
                     None, "container", False, {"why": err or err2}, status="NOT_RUN")
    allowed_txt = procfs.parse_status(st).get("Cpus_allowed_list")
    allowed = set(procfs.parse_cpu_list_text(allowed_txt))
    online = procfs.parse_cpu_list_text(on)
    missing = sorted(set(online) - allowed)
    return _item(4, ["read", "/proc/self/status", "/sys/devices/system/cpu/online"], 0,
                 st + on, "container", bool(online) and not missing,
                 {"cpus_allowed_list": allowed_txt, "online": on.strip(),
                  "online_not_allowed": missing,
                  "rule": "every online cpu is in Cpus_allowed_list (offline possible cpus in "
                          "the mask are ignored)"})


def item5(runner, timeout_s, expect_gpus, nvsmi):
    v = gpu_mod.check_visible(expect_gpus, timeout_s, runner, nvsmi=nvsmi)
    qs = v["queries"]
    out = "".join(q.get("stdout") or "" for q in qs) if any(
        q.get("stdout") is not None for q in qs) else None
    rcs = [q.get("rc") for q in qs]
    detail = {k: v[k] for k in ("expected_gpus", "visible_gpus", "nvidia_smi", "compute_apps",
                                "count_check", "positive_control", "apps_check", "reasons")}
    detail["gpu_count"] = v["visible_gpus"]
    detail["queries"] = [{k: q.get(k) for k in ("cmd", "rc", "stdout_sha256", "reason")}
                         for q in qs]
    it = _item(5, [q.get("cmd") for q in qs], rcs[-1] if len(set(rcs)) == 1 else rcs, out,
               "container", v["status"] == "PASS", detail, status=v["status"])
    it["positive_control_ok"] = v["positive_control_ok"]
    return it


def item6(window_s, max_frac, proc_root=procfs.PROC, sys_root=procfs.SYS, sleep=time.sleep):
    if window_s is None or max_frac is None:
        return _item(6, ["read", "/proc/stat"], None, None, "container", False,
                     {"why": "--idle-window-s / --idle-busy-max-frac not given"},
                     status="NOT_RUN")
    online = procfs.online_cpus(sys_root)
    a = procfs.read_proc_stat_cpus(proc_root)
    t0 = time.monotonic_ns()
    sleep(window_s)
    b = procfs.read_proc_stat_cpus(proc_root)
    t1 = time.monotonic_ns()
    per, busy_cores, missing = {}, [], []
    for c in online:
        if c not in a or c not in b:
            missing.append(c)
            continue
        d = {k: b[c][k] - a[c][k] for k in procfs.STAT_CPU_COLS}
        tot = sum(d.values())
        busy = sum(d[k] for k in procfs.BUSY_COLS)
        frac = busy / tot if tot > 0 else None
        # irq / softirq per core: the idle-window reading of the D14 rule (capjudge.pilot)
        per[str(c)] = {"busy_ticks": busy, "total_ticks": tot, "busy_frac": frac,
                       "irq_ticks": d["irq"], "softirq_ticks": d["softirq"]}
        if frac is None or frac > max_frac:
            busy_cores.append(c)
    checked = len(per)
    status = "NOT_RUN" if checked == 0 else None
    return _item(6, ["read", "/proc/stat", "x2"], 0 if checked else None, None, "container",
                 checked > 0 and not busy_cores and not missing,
                 {"window_ns": t1 - t0, "max_frac": max_frac, "expected_cores": len(online),
                  "checked_cores": checked, "busy_cores": busy_cores, "missing": missing,
                  "per_core": per}, status=status)


def hw_gate(profile_path, registered_sha, comparable_file):
    """(payload, passed). comparable_file None == --no-comparable. registered_sha UNREGISTERED:
    status NOT_RUN and never passed, whatever the profile says."""
    out = {"registered_hw_sha": registered_sha, "node_hw_sha": None, "diff_fields": None,
           "comparable_fields": None, "comparable_file_sha256": None, "passed": False,
           "status": "RED", "why": None, "profile_file": profile_path}
    unregistered = registered_sha == UNREGISTERED
    try:
        prof, sha = load_profile(profile_path)
    except (ConfigError, OSError, ValueError, KeyError) as e:
        out["why"] = "hw_profile unreadable: %s" % e
        out["status"] = "NOT_RUN"
        if unregistered:
            out["why"] = "profile not registered (--registered-hw-sha UNREGISTERED); " + out["why"]
        return out, False
    out["node_hw_sha"] = sha
    out["node_name"] = prof.get("node_name")
    out["unreadable_identity_fields"] = sorted(
        k for k, v in prof.items() if isinstance(v, dict) and "unreadable" in v)
    if unregistered:
        out["status"] = "NOT_RUN"
        out["why"] = ("profile not registered (--registered-hw-sha UNREGISTERED): node_hw_sha "
                      "recorded for registration; the gate never passes in this mode")
        if comparable_file is not None:
            out["why"] += "; --comparable-file ignored"
        return out, False
    if sha == registered_sha:
        out["passed"], out["status"], out["why"] = True, "PASS", "identical"
        return out, True
    if comparable_file is None:
        out["why"] = "profile sha differs and --no-comparable"
        return out, False
    try:
        doc = read_json(comparable_file)
        if doc.get("schema") != COMPARABLE_SCHEMA:
            raise ConfigError("schema %r != %s" % (doc.get("schema"), COMPARABLE_SCHEMA))
        reg_prof = validate_profile(doc["registered_profile"])
        fields = doc["comparable_fields"]
        if not isinstance(fields, list):
            raise ConfigError("comparable_fields must be a list")
    except (ConfigError, OSError, ValueError, KeyError, AttributeError) as e:
        out["why"] = "comparable file unusable: %s" % e
        return out, False
    out["comparable_file_sha256"] = sha256_file(comparable_file)
    if profile_sha256(reg_prof) != registered_sha:
        out["why"] = "comparable file's registered_profile does not hash to --registered-hw-sha"
        return out, False
    diff = diff_identity(prof, reg_prof)
    out["diff_fields"], out["comparable_fields"] = diff, sorted(fields)
    not_cov = sorted(set(diff) - set(fields))
    out["passed"] = not not_cov
    out["status"] = "PASS" if out["passed"] else "RED"
    out["why"] = "all differing fields registered comparable" if not not_cov else \
        "fields differ without comparability registration: %s" % not_cov
    return out, out["passed"]


def run_gate(args, ident, runner=runcmd.run, sleep=time.sleep):
    env_dir = os.path.join(args.run_dir, "env")
    if not os.path.isdir(env_dir):
        raise ConfigError("%s does not exist (job directory env/)" % env_dir)
    reasons, files = [], {}
    src = SlurmSource(args.slurm_view_file, args.slurm_view_wait_s, runner, args.timeout_s,
                      sleep=sleep)
    if args.max_alloc_gpus is None:
        raise ConfigError("--max-alloc-gpus is required (the GPU count requested from Slurm)")
    if not args.nvidia_smi:
        raise ConfigError("--nvidia-smi is required (the nvidia-smi executable to run)")
    it1, jobkv = item1(src, args.job_id, args.max_alloc_gpus)
    node = os.environ.get("SLURMD_NODENAME") or jobkv.get("NodeList")
    if not node:
        it2 = _item(2, None, None, None, "container", False,
                    {"why": "no SLURMD_NODENAME and no NodeList"}, status="NOT_RUN")
        it3 = _item(3, None, None, None, "container", False, {"why": "no node name"},
                    status="NOT_RUN")
        it3["positive_control_ok"] = False
    else:
        it2 = item2(src, node)
        it3 = item3(src, args.job_id, node)
    items = [it1, it2, it3, item4(),
             item5(runner, args.timeout_s, args.max_alloc_gpus, args.nvidia_smi)]
    if args.phase == "pre":
        items.append(item6(args.idle_window_s, args.idle_busy_max_frac, sleep=sleep))
    for it in items:
        if not it["passed"]:
            reasons.append("item %d %s: %s" % (it["id"], it["status"],
                                               json.dumps(it["detail"], sort_keys=True)[:400]))
    excl = {"phase": args.phase, "items": items, "expected_items": len(items),
            "positive_control_ok": it3.get("positive_control_ok", False),
            "slurm_view_file": args.slurm_view_file, "slurm_view_error": src.view_error}
    name = "exclusivity.json" if args.phase == "pre" else "exclusivity.post.json"
    p = os.path.join(env_dir, name)
    write_json_stamped(p, EXCL_SCHEMA, ident, excl)
    files[name] = sha256_file(p)
    if args.phase == "pre":
        prof_path = args.hw_profile or os.path.join(env_dir, "hw_profile.json")
        hg, ok = hw_gate(prof_path, args.registered_hw_sha, args.comparable_file)
        p = os.path.join(env_dir, "hw_gate.json")
        write_json_stamped(p, HWGATE_SCHEMA, ident, hg)
        files["hw_gate.json"] = sha256_file(p)
        if not ok:
            reasons.append("hw_gate %s: %s" % (hg["status"], hg["why"]))
        pids = [os.getpid()]
        source = "gate process itself (same job step / container as the tested processes)"
        if args.roles_file:
            pids += [x["pid"] for x in roles_mod.load(args.roles_file).procs]
            source += " + roles.json"
        mg = memguard.check(pids)
        mg["source_of_expected"] = source
        p = os.path.join(env_dir, "memguard.json")
        write_json_stamped(p, memguard.SCHEMA, ident, mg)
        files["memguard.json"] = sha256_file(p)
        if mg["verdict"] != "PASS":
            reasons.append("memguard %s: no finite memory limit for pids %s"
                           % (mg["verdict"], mg["red_pids"]))
    if args.registered_hw_sha == UNREGISTERED:
        reasons.append("registered hw sha is UNREGISTERED: profile collection only, never PASS")
    verdict = "PASS" if not reasons else "ENV_INVALID"
    out = {"verdict": verdict, "phase": args.phase, "reasons": reasons, "files": files,
           "items": {str(it["id"]): it["status"] for it in items},
           "registered_hw_sha": args.registered_hw_sha}
    if args.phase == "pre":
        out["hw_gate"] = {"status": hg["status"], "node_hw_sha": hg["node_hw_sha"]}
    p = os.path.join(env_dir, "gate.%s.json" % args.phase)
    write_json_stamped(p, GATE_SCHEMA, ident, out)
    return out


def build_parser():
    ap = argparse.ArgumentParser(prog="capenv.gate", description=__doc__.split("\n\n")[0])
    add_identity_args(ap)
    ap.add_argument("--run-dir", required=True, help="the job directory (writes its env/)")
    ap.add_argument("--phase", required=True, choices=("pre", "post"))
    ap.add_argument("--registered-hw-sha", required=True,
                    help="64 lowercase hex, or UNREGISTERED (collect the profile; exit 6)")
    cg = ap.add_mutually_exclusive_group(required=True)
    cg.add_argument("--comparable-file")
    cg.add_argument("--no-comparable", action="store_true")
    ap.add_argument("--slurm-view-file", required=True)
    ap.add_argument("--slurm-view-wait-s", type=float, required=True)
    ap.add_argument("--max-alloc-gpus", type=int, required=True,
                    help="GPUs requested from Slurm: item 1 bound, item 5 visible count")
    ap.add_argument("--nvidia-smi", required=True, help="nvidia-smi executable (item 5)")
    ap.add_argument("--idle-window-s", type=float, default=None,
                    help="item 6 idle window; required with --phase pre")
    ap.add_argument("--idle-busy-max-frac", type=float, default=None,
                    help="item 6 per-core busy bound in [0, 1]; required with --phase pre")
    o = ap.add_argument_group("optional")
    o.add_argument("--hw-profile", help="default <run-dir>/env/hw_profile.json")
    o.add_argument("--roles-file", help="also read back memory limits of these pids")
    o.add_argument("--timeout-s", type=float, default=60.0, help="per external command")
    return ap


def main(argv=None, runner=runcmd.run):
    args = build_parser().parse_args(argv)
    try:
        if args.registered_hw_sha != UNREGISTERED and not is_sha256_hex(args.registered_hw_sha):
            raise ConfigError("--registered-hw-sha must be 64 lowercase hex or %s" % UNREGISTERED)
        if args.max_alloc_gpus < 0:
            raise ConfigError("--max-alloc-gpus must be >= 0")
        if args.phase == "pre":
            if args.idle_window_s is None or args.idle_busy_max_frac is None:
                raise ConfigError("--phase pre needs --idle-window-s and --idle-busy-max-frac")
            if not args.idle_window_s > 0 or not 0 <= args.idle_busy_max_frac <= 1:
                raise ConfigError("--idle-window-s must be > 0 and --idle-busy-max-frac in "
                                  "[0, 1]")
        if args.no_comparable:
            args.comparable_file = None
        ident = Identity(experiment=args.experiment, arm=args.arm, launch_seq=args.launch_seq,
                         job_id=args.job_id, restart_count=args.restart_count,
                         register_sha=args.register_sha, role=args.role, worker=args.worker,
                         pid=os.getpid())
        out = run_gate(args, ident, runner=runner)
    except ConfigError as e:
        print("capenv.gate: %s" % e, file=sys.stderr)
        return exitcodes.CONFIG
    print(json.dumps(out, sort_keys=True))
    if args.registered_hw_sha == UNREGISTERED:
        return exitcodes.ENV_INVALID  # profile collection only: never a pass, whatever happened
    return exitcodes.OK if out["verdict"] == "PASS" else exitcodes.ENV_INVALID


if __name__ == "__main__":
    sys.exit(main())

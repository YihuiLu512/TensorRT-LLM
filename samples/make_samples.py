# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the judge self-test samples (INTERFACES 10.4): each sample is a minimal work dir W with
one launch tree, its own REGISTRY.tsv / submissions.tsv (local-* job) / empty JUDGED.tsv, the
registered files, and expect.json. Every launch.json is marked "synthetic": true (plan P8: real
samples come from the first E0 job and replace these as the良品 reference).

    python3.11 make_samples.py --out <new directory>

Topology simulated: capload (load_source, 2 workers) -> nullsse (under_test, P2 probe site F2).
Tiers: warmup, drain, up r20 (baseline), drain, up r40, drain. Times are synthetic; all
values come from one deterministic simulation (random.Random seeds fixed).
"""

import argparse
import array
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from capcommon import expect, hist, ledger  # noqa: E402
from capcommon.clock import wall_iso  # noqa: E402
from capcommon.capbin import write_capbin  # noqa: E402
from capcommon.hashing import sha256_bytes, sha256_file  # noqa: E402
from capcommon.identity import Identity  # noqa: E402
from capcommon.jsonio import JsonlWriter, atomic_write_bytes, dumps, write_json_stamped  # noqa
from capcommon.runtree import RunTree, tree_dir_name  # noqa: E402
from capcommon.schedule import BeatSchedule, write_schedule  # noqa: E402
from capcommon.snapshot import SCHEMA as SNAP_SCHEMA  # noqa: E402
from capcommon.snapshot import SlotRegistry, encode_snapshot  # noqa: E402
from capcommon.tiers import TierTable  # noqa: E402

S = 1_000_000_000
MS = 1_000_000
US = 1_000
ORIGIN = 10 ** 12
CODE_SHA = "c" * 40
HW_SHA = "d" * 64
IMAGE_SHA = "e" * 64
REG_TS0 = "2026-11-13T00:00:00.%03dZ"
SUBMITTED_AT = "2026-11-13T00:00:10.000Z"
CAPLOAD_PIDS = (1001, 1002)
NULLSSE_PID = 2001
SPIN_TID = 2003
CLK = 100
GPU_EVERY_NS = 2 * S  # sidecar --gpu-every-s 2: GPU-IDLE must read the period from meta.json
STATUS = {"timeout_registered": 1, "conn_refused": 2}
GATE = {"mode": "enforce", "rule": "abs_max",
        "readings": [
            {"name": "loadgen_inflight", "terms": [{"role": "capload", "slot": "backlog.inflight",
                                                    "sign": 1, "agg": "sum"}], "abs_max": 0},
            {"name": "sink_streams", "terms": [{"role": "nullsse",
                                                "slot": "backlog.active_streams", "sign": 1,
                                                "agg": "sum"}], "abs_max": 0}],
        "poll_ns": 200 * MS, "settle_polls": 2, "deadline_guard_ns": 200 * MS,
        "max_snapshot_age_ns": 400 * MS}
BANDS = {
    "T1:req_first_lag_p50_ns:capload": (-50 * US, 50 * US),
    "T1:msg_lag_p50_ns:capload": (-50 * US, 50 * US),
    "T2:delivery_ratio_norm:capload": (0.9, 1.1),
    "T1:inflight_derived:capload": (-3, 3),
    "T1:backlog.inflight:capload": (-3, 3),
    "T1:backlog.active_streams:nullsse": (-3, 3),
    "T1:send_lag_p50_ns:capload": (-20 * US, 20 * US),
    "T3:send_lag_p99_ns:capload": (0, 100 * US),
    "T3:fionread_nonzero_frac:capload": (0.0, 0.05),
    "T3:loop_lag_p99_ns:capload": (0, 2 * MS),
    "T3:foreign_busy_frac:nullsse": (-0.05, 0.05),
}


def tier(seq, label, kind, start, ramp, window, tail, rate, baseline=False):
    load = kind != "drain"
    return {"seq": seq, "label": label, "kind": kind, "start_ns": start, "ramp_ns": ramp,
            "window_ns": window, "tail_ns": tail, "rate_rps": float(rate), "streams_c": None,
            "max_tokens": 3 if load else 0, "tokens_per_msg": 1, "stream_interval": 1,
            "beat_ns": 20 * MS, "prompt_set": "tok8" if load else None, "inject": [],
            "gate": None, "baseline": baseline, "note": "synthetic",
            "load_value": float(rate) if kind in ("up", "down") else None}


def tier_table_doc():
    tiers = [tier(0, "warmup", "warmup", 0, 0, 2 * S, 0, 10),
             tier(1, "drain-01", "drain", 2 * S, 0, 2 * S, 0, 0),
             tier(2, "up-r20", "up", 4 * S, S, 4 * S, S, 20, baseline=True),
             tier(3, "drain-02", "drain", 10 * S, 0, 2 * S, 0, 0),
             tier(4, "up-r40", "up", 12 * S, S, 4 * S, S, 40),
             tier(5, "drain-03", "drain", 18 * S, 0, 2 * S, 0, 0)]
    return {"schema": "capkit.tiers/1", "experiment": "E0", "name": "synthetic-judge-sample",
            "arrival": {"dist": "fixed", "seed": 20261113},
            "axis": {"name": "R", "load_key": "rate_rps", "unit": "req/s", "order": "asc"},
            "request_timeout_ns": S, "beat_origin_rel_ns": 0, "default_gate": GATE,
            "launch_inject": [], "tiers": tiers}


class Sim:
    """One deterministic simulation of every request of the schedule."""

    def __init__(self, tt, planned, tseq, opts):
        self.tt, self.opts = tt, opts
        beat = BeatSchedule(tt)
        rng = random.Random(1234)
        by_tier = {}
        for rid, ts in enumerate(tseq):
            by_tier.setdefault(ts, []).append(rid)
        timeouts = set(by_tier[4][-5:]) if opts.get("overload") else set()
        refused = set(by_tier[4][:2]) if opts.get("c3") else set()
        grow = opts.get("overload") or opts.get("spin")
        self.reqs = []
        for rid, (p, ts) in enumerate(zip(planned, tseq)):
            t = tt.tiers[ts]
            w = rid % 2
            sl = 30 * US + rng.randrange(20 * US)
            if opts.get("c1") and ts == 4 and w == 0 and rng.random() < 0.3:
                sl += 500 * US
            sent = p + sl
            r = {"rid": rid, "tier": ts, "worker": w, "planned": p, "sent": sent,
                 "status": 0, "fail": None, "msgs": [], "chunks": []}
            self.reqs.append(r)
            if rid in refused:
                r.update(status=STATUS["conn_refused"], fail="conn_refused", first=-1,
                         done=sent + 100 * US, stream=None)
                continue
            acc = sent + 80 * US
            fs = beat.step_at_or_after(acc)
            offs = expect.msg_step_offsets(t.max_tokens, t.tokens_per_msg, t.stream_interval,
                                           "product")
            send_times = [beat.planned_rel_ns(fs + o) + 10 * US for o in offs]
            r["stream"] = {"accept": acc, "first_step": fs, "send": send_times}
            if rid in timeouts:
                r.update(status=STATUS["timeout_registered"], fail="timeout_registered",
                         first=-1, done=sent + tt.request_timeout_ns)
                continue
            recv = []
            for o in offs:
                pl = beat.planned_rel_ns(fs + o)
                extra = (pl - t.start_ns) // 20 if (grow and ts == 4) else 0  # 50 ms per s
                recv.append(pl + 150 * US + rng.randrange(30 * US) + extra)
            recv.append(recv[-1] + 20 * US)  # usage frame
            r["chunks"] = list(enumerate(recv))
            r.update(first=recv[0], done=recv[-1] + 10 * US)

    def inflight(self, worker, t):
        return sum(1 for r in self.reqs if r["worker"] == worker and r["sent"] <= t < r["done"])

    def active_streams(self, t):
        return sum(1 for r in self.reqs if r.get("stream")
                   and r["stream"]["accept"] <= t < r["stream"]["send"][-1])

    def msgs_sent(self, t):
        n = sum(1 for r in self.reqs if r.get("stream") for x in r["stream"]["send"] if x <= t)
        if self.opts.get("c6") and t >= 15 * S:
            n -= 3
        return n


def write_plain(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic_write_bytes(path, (dumps(doc) + "\n").encode("utf-8"))


ROLES = [
    {"role": "capload", "module": "capload", "args": [], "cpus": [2, 3], "workers": 2,
     "expected_procs": 2, "measure_class": "load_source", "asyncio": True, "uses_grpc": False,
     "loop": "asyncio", "probe_tier": None, "kind": "client"},
    {"role": "nullsse", "module": "capload.nullsse", "args": [], "cpus": [4], "workers": 1,
     "expected_procs": 1, "measure_class": "under_test", "asyncio": True, "uses_grpc": False,
     "loop": "asyncio", "probe_tier": "P2", "kind": "server", "proc_kinds": ["frontend"]},
]
# capload's multiprocessing.resource_tracker: caplaunch attributes it to capload as an aux helper
# (roles.json aux_helpers, tier.json scans), never as a process of the role nor unattributed.
AUX_HELPER = {"pid": 1003, "ppid": 1001, "starttime_ticks": 6003,
              "cmdline": ["python", "-c", "from multiprocessing.resource_tracker import main;"
                          "main(9)"], "role": "capload", "measure_class": "load_source",
              "category": "aux_helper", "rule": "mp_resource_tracker",
              "first_seen_mono_ns": ORIGIN - 3 * S}
RECONCILE = {"schema": "capkit.reconcile/1", "chains": [
    {"name": "requests", "kind": "requests", "gate": None, "include_usage": None,
     "hops": [{"role": "capload", "source": "summary.counts.ok", "scope": "tier"},
              {"role": "nullsse", "source": "summary.counts.requests", "scope": "tier"}],
     "corrections": []},
    {"name": "sse", "kind": "chunks", "gate": "product", "include_usage": True,
     "hops": [{"role": "capload", "source": "summary.frames.frames", "scope": "tier"},
              {"role": "nullsse", "source": "summary.counts.frames_sent", "scope": "tier"}],
     "corrections": []},
    {"name": "probe-msgs", "kind": "messages", "gate": "product", "include_usage": None,
     "hops": [{"role": "nullsse", "source": "slot:count.F2.msgs", "scope": "tier"}],
     "corrections": []}]}
CHECKS = ["C1", "C3", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C13", "GPU-IDLE"]


class Builder:
    def __init__(self, W, name, opts):
        self.W, self.name, self.opts = W, name, opts
        os.makedirs(os.path.join(W, "register"))
        self.reg_n = 0
        self.job_id = "local-sample-%s" % name.replace("_", "-")

    def rel(self, *p):
        return os.path.join(*p)

    def register_file(self, relpath, kind, registered_at=None):
        sha = sha256_file(os.path.join(self.W, relpath))
        if registered_at is None:
            registered_at = REG_TS0 % self.reg_n
            self.reg_n += 1
        ledger.registry_append(os.path.join(self.W, "REGISTRY.tsv"), "E0", relpath, sha,
                               "kind=%s;synthetic sample" % kind, registered_at=registered_at)
        return {"relpath": relpath, "sha256": sha}

    def build_registered(self):
        W = self.W
        tt_rel = "register/tiers.json"
        write_plain(os.path.join(W, tt_rel), tier_table_doc())
        self.tt = TierTable.load(os.path.join(W, tt_rel))
        sc_rel = "register/schedule.capbin"
        write_schedule(os.path.join(W, sc_rel), self.tt, 100000)
        pr_rel = "register/prompts.json"
        write_plain(os.path.join(W, pr_rel), {
            "schema": "capkit.prompts/1", "source": {"tokenizer": "none", "note": "synthetic"},
            "sets": {"tok8": {"kind": "token_ids", "items": [[1, 2, 3, 4, 5, 6, 7, 8]]}}})
        sn_rel = "register/scenario.json"
        self.scenario = {"schema": "capkit.scenario/1", "synthetic": True,
                         "roles": [{k: v for k, v in r.items() if k != "cpus"} for r in ROLES],
                         "tier_table": tt_rel, "schedule": sc_rel, "prompts": pr_rel,
                         "probe_tier": "P2", "stop_order": ["capload", "nullsse"],
                         "reconcile": RECONCILE}
        write_plain(os.path.join(W, sn_rel), self.scenario)
        pi_rel = "register/pilot.json"
        write_plain(os.path.join(W, pi_rel), {
            "schema": "capkit.pilot/1", "experiment": "E0", "synthetic": True,
            "environment": {"hw_profile_sha": HW_SHA, "image_sha": IMAGE_SHA,
                            "capkit_sha": CODE_SHA, "baseline_sha": "2f22a98d6e"},
            "inputs": [], "noise_bands": {k: {"regime": "subsat", "min": lo, "max": hi, "n": 6,
                                              "unit": "synthetic", "source_tiers": []}
                                          for k, (lo, hi) in BANDS.items()},
            "window_ns": None, "warmup_ns": None, "tier_resolution": None,
            "launches_per_arm": None, "min_detectable_diff": None,
            "i2_calibration": {"enters_criteria": False, "per_loop_impl": {}},
            "d13_switchinterval_s": None,
            "d14_softirq": {"irq_softirq_is_experiment": True, "note": "synthetic"},
            "core_budget": None, "drain_max_wait_ns": None, "canary_band": None})
        self.files = {"tier_table": self.register_file(tt_rel, "tier_table"),
                      "schedule": self.register_file(sc_rel, "schedule"),
                      "prompts": self.register_file(pr_rel, "prompts"),
                      "scenario": self.register_file(sn_rel, "scenario")}
        pilot = self.register_file(pi_rel, "pilot")
        rg_rel = "register/register.json"
        write_plain(os.path.join(W, rg_rel), {
            "schema": "capkit.register/1", "experiment": "E0", "synthetic": True,
            "arms": ["synthetic-a"], "files": self.files,
            "calibration": [{"kind": "pilot", "relpath": pilot["relpath"],
                             "sha256": pilot["sha256"]}],
            "expected": {"launches_per_arm": 1, "job_budget": 0},
            "judge": {"mode": "P+X", "checks": CHECKS}})
        late = "2026-11-13T00:00:20.000Z" if self.opts.get("register_late") else None
        self.register = self.register_file(rg_rel, "register", registered_at=late)

    def build_ledgers(self):
        sub = os.path.join(self.W, "submissions.tsv")
        base = {c: "" for c in ledger.SUBMISSION_COLUMNS}
        base.update({"plan_id": "plan-" + self.name, "experiment": "E0",
                     "arms": "synthetic-a", "launches": "1", "gpus": "0", "exclusive": "0",
                     "basis": "synthetic judge self-test sample (local, no cluster)",
                     "mounts_json": "[]", "tokenizer_dir_sha256": "none",
                     "assertions_json": json.dumps({k: True for k in
                                                    ("experiment_no_model", "gpus_le_1",
                                                     "no_model_mounts", "tokenizer_no_weights",
                                                     "basis_present")})})
        ledger.submissions_append(sub, dict(base, event="PLANNED",
                                            event_at="2026-11-13T00:00:09.000Z"))
        ledger.submissions_append(sub, dict(base, event="SUBMITTED", job_id=self.job_id,
                                            submitted_at=SUBMITTED_AT,
                                            event_at=SUBMITTED_AT))
        open(os.path.join(self.W, "JUDGED.tsv"), "w").close()

    # ---- the launch tree ---------------------------------------------------------------------
    def ident(self, role, worker=0, pid=None, tier=None):
        i = Identity(experiment="E0", arm="synthetic-a", launch_seq=0, job_id=self.job_id,
                     restart_count=0, register_sha=self.register["sha256"], role=role,
                     worker=worker, pid=pid)
        return i.with_tier(tier.seq, tier.label) if tier is not None else i

    def threads_of(self):
        """{pid: [(tid, comm, identified_as, ticks_per_s)]}."""
        th = {pid: [(pid, "python", "event_loop", 30)] for pid in CAPLOAD_PIDS}
        main = [(NULLSSE_PID, "python", "event_loop", 20),
                (NULLSSE_PID + 1, "capkit-snap", "snapshot_writer", 1)]
        if self.opts.get("spin"):
            main.append((SPIN_TID, "python", "spin_thread", 70))
        th[NULLSSE_PID] = main
        return th

    def pid_role(self):
        d = {pid: ("capload", k) for k, pid in enumerate(CAPLOAD_PIDS)}
        d[NULLSSE_PID] = ("nullsse", 0)
        return d

    def cpu_of(self, pid):
        return {1001: 2, 1002: 3, NULLSSE_PID: 4}[pid]

    def build_tree(self):
        self.tree_rel = os.path.join("runs", tree_dir_name(self.job_id, 0, 0, "synthetic-a"))
        self.T = RunTree(os.path.join(self.W, self.tree_rel)).make_skeleton()
        sc = os.path.join(self.W, self.files["schedule"]["relpath"])
        from capcommon.capbin import read_capbin
        _h, cols = read_capbin(sc)
        self.sim = Sim(self.tt, list(cols["planned_rel_ns"]), list(cols["tier_seq"]), self.opts)
        roles = [dict(r, args=["--synthetic"]) for r in ROLES]
        L = self.ident("launch")
        write_json_stamped(self.T.launch_json, "capkit.launch/1", L, {
            "synthetic": True, "origin_ns": ORIGIN, "origin_lead_ns": 5 * S,
            "shm_dir": "/dev/shm/synthetic", "register": dict(self.register),
            "tier_table": self.files["tier_table"], "schedule": self.files["schedule"],
            "prompts": self.files["prompts"],
            "scenario": dict(self.files["scenario"], doc=self.scenario),
            "code": {"capkit_sha": CODE_SHA, "probe_tree_sha": None,
                     "baseline_sha": "2f22a98d6e", "snapshot_dir": None},
            "image": {"path": "synthetic.sqsh", "sha256": IMAGE_SHA},
            "probe_tier": {"tier": "P2", "groups": ["count"]}, "roles": roles,
            "reconcile": RECONCILE, "hw_profile_sha": HW_SHA,
            "submission": {"job_id": self.job_id, "restart_count": 0,
                           "submitted_at": SUBMITTED_AT}})
        procs = [{"pid": pid, "starttime_ticks": 5000 + pid, "role": role, "worker": k,
                  "parent_pid": 900, "cmdline": ["python", "-m", role], "cpus":
                  [self.cpu_of(pid)], "discovered_via": "ready.children",
                  "registered_mono_ns": ORIGIN - 4 * S}
                 for pid, (role, k) in sorted(self.pid_role().items())]
        self.procs = procs
        write_json_stamped(self.T.roles_json, "capkit.roles/1", L,
                           {"generation": 1, "procs": procs, "unattributed": [],
                            "aux_helpers": [AUX_HELPER]})
        self.build_env(L)
        self.build_inventory()
        write_json_stamped(self.T.launch_end_json, "capkit.launch_end/1", L, {
            "stop": None, "ready_at_rel_ns": {"capload": -S, "nullsse": -2 * S},
            "tiers_completed": 5, "gate_outcomes": [], "exit_codes": {"capload": 0,
                                                                      "nullsse": 0},
            "ended_mono_ns": ORIGIN + 21 * S})

    def build_env(self, L):
        write_plain(self.T.env("pin_table.json"), {
            "schema": "capkit.pin_table/1", "source_sha": HW_SHA, "rules": "synthetic",
            "siblings_left_idle": [], "roles": {"capload": {"cpus": [2, 3], "numa_node": 0},
                                                "nullsse": {"cpus": [4], "numa_node": 0}},
            "unassigned": []})
        # the key set capjobs.envfiles gpu-idle writes (capenv.gpu.check_visible + gpu_count,
        # passed, outcome); the judge reads positive_control_ok
        smi = "/usr/bin/nvidia-smi"
        q1 = [smi, "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,"
              "utilization.gpu,driver_version", "--format=csv,noheader,nounits"]
        q2 = [smi, "--query-compute-apps=gpu_bus_id,pid,used_memory",
              "--format=csv,noheader,nounits"]
        out1 = "0, GPU-synthetic, 00000000:1B:00.0, synthetic, 183359, 0, 0, 0\n"
        write_json_stamped(self.T.env("gpu_idle.json"), "capkit.gpu_idle/1",
                           self.ident("jobs"), {
            "status": "PASS", "reasons": [], "expected_gpus": 1, "nvidia_smi": smi,
            "visible_gpus": 1,
            "gpus": [{"index": 0, "uuid": "GPU-synthetic", "bus_id": "0000:1B:00.0",
                      "name": "synthetic", "mem_total_mib": 183359, "mem_used_mib": 0,
                      "util_pct": 0, "driver": "0"}],
            "compute_apps": [],
            "queries": [{"cmd": q1, "rc": 0, "reason": None, "stderr": "", "stdout": out1,
                         "stdout_sha256": sha256_bytes(out1.encode())},
                        {"cmd": q2, "rc": 0, "reason": None, "stderr": "", "stdout": "",
                         "stdout_sha256": sha256_bytes(b"")}],
            "count_check": {"status": "PASS", "why": "visible 1, requested 1"},
            "positive_control": {"status": "PASS", "why": "listed 1 GPU(s)"},
            "apps_check": {"status": "PASS", "why": "0 compute process(es)"},
            "positive_control_ok": True, "gpu_count": 1, "passed": True, "outcome": "PASS"})
        write_plain(self.T.env("overlay_manifest.json"), {
            "schema": "capkit.overlay/1", "tier": "P1" if self.opts.get("probe_mismatch")
            else "P2", "image_commit": "synthetic", "base_commit": "2f22a98d6e",
            "target_commit": "synthetic", "image_is_ancestor_of_target": True,
            "groups": ["count"], "files": [], "capkit_packages": [], "capkit_code_sha": CODE_SHA,
            "non_python_changed": [], "unshipped_allowed": [],
            "sites": {"F2": {"group": "count", "proc_kind": "frontend", "file": "synthetic",
                             "lines": [1], "slots": ["count.F2.msgs"]},
                      # no worker process in this topology: C8 lists it as not applicable
                      "W1": {"group": "count", "proc_kind": "worker", "file": "synthetic",
                             "lines": [1], "slots": ["count.W1.synthetic"]}},
            "sites_selection": "all_sites_of_open_groups", "patchcheck": None,
            "created_wall": "synthetic"})
        for pid, (role, k) in self.pid_role().items():
            rec = {"python": "3.12.3 synthetic", "implementation": "cpython",
                   "executable": "/usr/bin/python3", "pid": pid, "grpc": None,
                   "protobuf": None, "protobuf_impl": None,
                   "loop_impl": "asyncio.unix_events._UnixSelectorEventLoop",
                   "gc_enabled": True, "switchinterval_s": 0.005, "capkit_code_sha": CODE_SHA,
                   "probe_tier": None, "probe_patch_sha": None, "probe_groups": [], "env": {}}
            if role == "nullsse":
                rec.update(probe_tier="P2", probe_patch_sha="f" * 40, probe_groups=["count"])
                if self.opts.get("c10"):
                    rec["capkit_code_sha"] = None
            write_json_stamped(self.T.env("versions", "%s.w%d.%d.json" % (role, k, pid)),
                               "capkit.version_record/1", self.ident(role, k, pid),
                               {"versions": rec, "source": "ready_file"})

    def build_inventory(self):
        for pid, ths in self.threads_of().items():
            role, _k = self.pid_role()[pid]
            if role != "nullsse":
                continue
            lst = [{"tid": tid, "comm": comm, "py_name": None, "class": "python",
                    "identified_as": ident, "source": "synthetic"}
                   for tid, comm, ident, _r in ths]
            if self.opts.get("c13_threads"):
                lst.append({"tid": 2999, "comm": "mystery", "py_name": None, "class": "other",
                            "identified_as": None, "source": "comm"})
            write_json_stamped(self.T.threads_file(role, pid), "capkit.threads/1",
                               self.ident("launch"), {"pid": pid, "taken_mono_ns": ORIGIN + S,
                                                      "threads": lst, "rules_sha": "0" * 64})

    # ---- tiers ---------------------------------------------------------------------------------
    def snapshot_bytes(self, pid, t_rel, seq, count_at=None):
        role, k = self.pid_role()[pid]
        reg = SlotRegistry()
        if role == "capload":
            reg.values[reg.gauge("backlog.inflight")] = self.sim.inflight(k, t_rel)
            reg.gauge("backlog.unread_bytes")
        else:
            reg.values[reg.gauge("backlog.active_streams")] = self.sim.active_streams(t_rel)
            reg.values[reg.counter("count.F2.msgs")] = self.sim.msgs_sent(
                t_rel if count_at is None else count_at)
        meta = {"schema": SNAP_SCHEMA, "pid": pid, "ppid": 900, "starttime_ticks": 5000 + pid,
                "argv": ["synthetic"], "interval_ns": 100 * MS, "slots": reg.slots,
                "identity": self.ident(role, k, pid).to_dict(), "role_hint": role,
                "versions": None}
        mb = json.dumps(meta, sort_keys=True).encode()
        mb += b" " * ((-len(mb)) % 8)
        return encode_snapshot(reg, seq, ORIGIN + t_rel, mb)

    def threads_at(self):
        return {str(pid): [{"tid": tid, "comm": comm, "py_name": None, "class": "python",
                            "identified_as": ia, "cpus_allowed": str(self.cpu_of(pid))}
                           for tid, comm, ia, _r in ths]
                for pid, ths in self.threads_of().items()}

    def scan(self, at):
        """caplaunch's window-boundary scan (tier.json `scans`): registered live processes and
        their descendants as [pid, starttime_ticks]: none under test; capload (a monitor-class
        role) has one aux helper, listed apart."""
        live = [[pid, 5000 + pid] for pid in sorted(self.pid_role())]
        return {"t_mono_ns": ORIGIN + at, "registered_live": live,
                "registered_total": len(live), "descendants": [], "monitor_descendants": [],
                "aux_helpers": [[AUX_HELPER["pid"], AUX_HELPER["starttime_ticks"],
                                 AUX_HELPER["role"]]]}

    def build_tiers(self):
        for t in self.tt.tiers:
            d = self.T.tier_dir(t.seq, t.label)
            os.makedirs(d, exist_ok=True)
            L = self.ident("launch").with_tier(t.seq, t.label)
            copies = []
            if t.kind != "drain":
                self.write_records(t)
                for edge, at in (("wstart", t.window_start_ns), ("wend", t.window_end_ns)):
                    for pid, (role, _k) in self.pid_role().items():
                        ts = at + 30 * MS
                        stuck = (self.opts.get("c8") and t.seq == 4 and edge == "wend")
                        b = self.snapshot_bytes(pid, ts, ts // (100 * MS) + 1,
                                                count_at=t.window_start_ns + 30 * MS
                                                if stuck else None)
                        p = os.path.join(d, "snapshots", "%s.%d.%s.bin" % (role, pid, edge))
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        atomic_write_bytes(p, b)
                        copies.append({"pid": pid, "role": role, "which": edge,
                                       "seq": ts // (100 * MS) + 1, "t_mono_ns": ORIGIN + ts,
                                       "offset_from_boundary_ns": 30 * MS,
                                       "file": os.path.basename(p)})
            else:
                write_json_stamped(os.path.join(d, "drain_gate.json"), "capkit.drain_gate/1",
                                   L, self.gate_record(t))
            nic = {"lo": {"rx_bytes": 0, "tx_bytes": 0}}
            write_json_stamped(os.path.join(d, "tier.json"), "capkit.tier/1", L, {
                "tier": self.tt.doc["tiers"][t.seq], "abs_start_ns": ORIGIN + t.start_ns,
                "abs_window_start_ns": ORIGIN + t.window_start_ns,
                "abs_window_end_ns": ORIGIN + t.window_end_ns, "abs_end_ns": ORIGIN + t.end_ns,
                "scans": {"wstart": self.scan(t.window_start_ns),
                          "wend": self.scan(t.window_end_ns)},
                "pids_stable": True,
                "threads": {"wstart": self.threads_at(), "wend": self.threads_at()},
                "nic_counters": {"wstart": nic, "wend": nic},
                "snapshot_copies": sorted(copies, key=lambda c: (c["which"], c["pid"])),
                "thread_rules_sha": "0" * 64, "incomplete": None})

    def write_records(self, t):
        reqs = [r for r in self.sim.reqs if r["tier"] == t.seq]
        d = self.T.tier_dir(t.seq, t.label)
        for w, pid in enumerate(CAPLOAD_PIDS):
            mine = [r for r in reqs if r["worker"] == w]
            ok = [r for r in mine if r["status"] == 0]
            fails = {}
            for r in mine:
                if r["fail"]:
                    fails[r["fail"]] = fails.get(r["fail"], 0) + 1
            hint = {"timeout_registered": "overload", "conn_refused": "non_overload"}
            I = self.ident("capload", w, pid, t)
            cols = {k: array.array(tc) for k, tc in (
                ("req_id", "q"), ("planned_rel_ns", "q"), ("sent_rel_ns", "q"),
                ("first_rel_ns", "q"), ("done_rel_ns", "q"), ("frames", "i"), ("bytes", "q"),
                ("tokens", "i"), ("status", "h"), ("conn_id", "i"), ("new_conn", "b"),
                ("tier_seq", "h"))}
            ch = {k: array.array(tc) for k, tc in (("req_id", "q"), ("idx", "i"),
                                                   ("recv_rel_ns", "q"), ("nbytes", "i"),
                                                   ("tier_seq", "h"))}
            sl = hist.new_counts()
            frames = 0
            for r in mine:
                chunks = r["chunks"]
                if self.opts.get("c7") and t.seq == 4 and r["rid"] % 10 == 0:
                    chunks = [c for c in chunks if c[0] != 1]
                frames += len(chunks)
                for col, v in (("req_id", r["rid"]), ("planned_rel_ns", r["planned"]),
                               ("sent_rel_ns", r["sent"]), ("first_rel_ns", r["first"]),
                               ("done_rel_ns", r["done"]), ("frames", len(chunks)),
                               ("bytes", 100 * len(chunks)), ("tokens", 3 if chunks else 0),
                               ("status", r["status"]), ("conn_id", w), ("new_conn", 0),
                               ("tier_seq", t.seq)):
                    cols[col].append(v)
                hist.add(sl, r["sent"] - r["planned"])
                for idx, recv in chunks:
                    for col, v in (("req_id", r["rid"]), ("idx", idx), ("recv_rel_ns", recv),
                                   ("nbytes", 100), ("tier_seq", t.seq)):
                        ch[col].append(v)
            meta = {"status_codes": {str(v): k for k, v in STATUS.items()}, "overflow": 0}
            write_capbin(self.T.tier_file(t.seq, t.label, "capload", w, "requests", "capbin"),
                         "loadgen.requests", I, list(cols.items()), meta)
            cmeta = {"chunk_sample_mod": 1,
                     "overflow": 3 if (self.opts.get("overflow") and t.seq == 4) else 0}
            write_capbin(self.T.tier_file(t.seq, t.label, "capload", w, "chunks", "capbin"),
                         "loadgen.chunks", I, list(ch.items()), cmeta)
            lag = hist.new_counts()
            lag[hist.bucket_index(50 * US)] = 100
            n_ok = len(ok)
            sent = len(mine) - (1 if (self.opts.get("p_sent") and t.seq == 4 and w == 1)
                                else 0)
            write_json_stamped(self.T.tier_file(t.seq, t.label, "capload", w, "summary",
                                                "json"), "capkit.tier_summary/1", I, {
                "tool": "capload", "n_workers": 2, "schedule_sha256":
                    self.files["schedule"]["sha256"], "tier_table_sha256": self.tt.sha256,
                "request_timeout_ns": self.tt.request_timeout_ns,
                "counts": {"planned": len(mine), "sent": sent, "ok": n_ok,
                           "failed": len(mine) - len(ok), "not_sent": len(mine) - sent},
                "failures": [{"reason": k, "count": v, "class_hint": hint[k]}
                             for k, v in sorted(fails.items())],
                "frames": {"frames": frames, "bytes": 100 * frames,
                           "completion_tokens": 3 * len(ok)},
                "send_lag_ns": {"hist": {"bins": "loghist/1", "counts": sl}, "max": 0,
                                "n": len(mine)},
                "fionread": {"samples": 100, "nonzero": 0, "max_bytes": 0},
                "conn_pool": {"size_max": 1, "new_conns": 1, "reconnects": 0,
                              "peer_idle_closed": 0, "conn_fail": 0},
                "inject": [], "loop_lag_ns": {"hist": {"bins": "loghist/1", "counts": lag},
                                              "n": 100},
                "overflow": {"requests": 0, "chunks": 0, "fionread": 0}, "stop": None})
        self.write_server(t, reqs)

    def write_server(self, t, reqs):
        served = [r for r in reqs if r.get("stream")]
        I = self.ident("nullsse", 0, NULLSSE_PID, t)
        cols = {k: array.array(tc) for k, tc in (("req_id", "q"), ("accept_rel_ns", "q"),
                                                 ("first_step", "q"), ("n_msgs", "i"),
                                                 ("tokens", "i"), ("tier_seq", "h"))}
        for r in served:
            st = r["stream"]
            for col, v in (("req_id", r["rid"]), ("accept_rel_ns", st["accept"]),
                           ("first_step", st["first_step"]), ("n_msgs", len(st["send"])),
                           ("tokens", 3), ("tier_seq", t.seq)):
                cols[col].append(v)
        write_capbin(self.T.tier_file(t.seq, t.label, "nullsse", 0, "streams", "capbin"),
                     "server.streams", I, list(cols.items()), {"overflow": 0})
        n = len(served) - (1 if (self.opts.get("c5") and t.seq == 4) else 0)
        frames = sum(len(r["stream"]["send"]) + 1 for r in served)
        write_json_stamped(self.T.tier_file(t.seq, t.label, "nullsse", 0, "summary", "json"),
                           "capkit.tier_summary/1", I, {
            "tool": "nullsse", "n_workers": 1, "schedule_sha256": None,
            "tier_table_sha256": self.tt.sha256,
            "counts": {"requests": n, "frames_sent": frames, "bytes_sent": 100 * frames,
                       "accepts": len(served), "idle_closed": 0},
            "inject": [], "overflow": {"streams": 0}, "stop": None})

    def gate_record(self, t):
        g = GATE
        key = list(self.ident("launch").launch_key())
        deadline = t.end_ns - g["deadline_guard_ns"]
        polls, consec, passed, t_pass = [], 0, False, None
        pt = t.start_ns
        while pt <= deadline and not passed:
            actual = pt + MS
            fields, vals, details = {}, {}, {}
            for pid in sorted(self.pid_role()):
                snap_t = actual - 20 * MS
                fields[str(pid)] = {"pid": pid, "seq": snap_t // (100 * MS) + 1,
                                    "t_mono_ns": ORIGIN + snap_t,
                                    "starttime_ticks": 5000 + pid, "launch_key": key}
            infl = [self.sim.inflight(k, actual - 20 * MS) for k in range(2)]
            act = self.sim.active_streams(actual - 20 * MS)
            for name, role, slot, vlist in (("loadgen_inflight", "capload", "backlog.inflight",
                                             infl),
                                            ("sink_streams", "nullsse",
                                             "backlog.active_streams", [act])):
                v = sum(vlist)
                vals[name] = v
                details[name] = [{"role": role, "slot": slot, "agg": "sum", "sign": 1,
                                  "value": v, "n_procs": len(vlist), "seqs": []}]
            sat = all(v <= 0 for v in vals.values())
            consec = consec + 1 if sat else 0
            polls.append({"planned_rel_ns": pt, "actual_rel_ns": actual,
                          "now_ns": ORIGIN + actual, "values": vals, "details": details,
                          "snapshots": fields, "errors": {}})
            if consec >= g["settle_polls"]:
                passed, t_pass = True, pt
            pt += g["poll_ns"]
        if self.opts.get("c11") and t.seq == 3:  # stale snapshot recorded as usable
            polls[-1]["snapshots"][str(NULLSSE_PID)]["t_mono_ns"] -= S
        return {"gate": g, "source": "default", "polls": polls, "n_polls": len(polls),
                "passed": passed, "t_pass_rel_ns": t_pass, "deadline_rel_ns": deadline,
                "action": "none" if passed else "stopped"}

    # ---- sidecar ---------------------------------------------------------------------------
    def build_sidecar(self):
        times = [k * S for k in range(-1, 22)]
        gpu_times = [k * GPU_EVERY_NS for k in range(-1, 22 * S // GPU_EVERY_NS)]
        sc = self.T.sidecar
        pc = {"ok": True, "gpu_count": 1, "rc": 0, "reason": None,
              "cmd": ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]}
        rows = {"threads": len(times) * sum(len(v) for v in self.threads_of().values()),
                "cpus": len(times) * 3, "procs": 0, "snapshots": len(times) * 3,
                "gpu": len(gpu_times), "logs": 0, "tids": 0}
        write_json_stamped(sc("meta.json"), "capkit.sidecar.meta/1", self.ident("sidecar"), {
            "clk_tck": CLK, "page_kb": 4, "interval_ns": 1000 * MS, "origin_ns": ORIGIN,
            "cpus_watch": [2, 3, 4], "gpu_every_ns": GPU_EVERY_NS, "numa_every_ns": 0,
            "gpu_positive_control": pc, "logs_dir": "synthetic", "shm_dir": "synthetic",
            "roles_file": "synthetic", "freq_source": "synthetic", "unreadable": {},
            "stats": {}, "pids_gone": [], "unregistered_snapshot_pids": [],
            "roles_generations": [], "roles_errors": [], "rows": rows, "own_cpu_ns": 0,
            "started_wall": "synthetic", "final": True, "stop": None})
        th_cols = ["t_rel_ns", "pid", "tid", "comm", "utime_ticks", "stime_ticks",
                   "run_delay_ns", "voluntary_ctxt", "nonvoluntary_ctxt", "last_cpu",
                   "cpus_allowed"]
        with JsonlWriter(sc("threads.jsonl"), "capkit.sidecar.threads/1",
                         self.ident("sidecar"), th_cols) as w:
            for t in times:
                for pid, ths in sorted(self.threads_of().items()):
                    cpu = self.cpu_of(pid)
                    for tid, comm, _ia, rate in ths:
                        allowed = str(cpu)
                        if self.opts.get("c9") and pid == 1001 and t == 14 * S:
                            allowed = "2-4"
                        ticks = (t + S) // S * rate
                        w.write_row([t, pid, tid, comm, ticks, 0, 0, 0, 0, cpu, allowed])
        with JsonlWriter(sc("cpus.jsonl"), "capkit.sidecar.cpus/1", self.ident("sidecar"),
                         ["t_rel_ns", "cpu", "user", "nice", "system", "idle", "iowait", "irq",
                          "softirq", "steal", "freq_khz"]) as w:
            for t in times:
                n = (t + S) // S
                for cpu in (2, 3, 4):
                    user = sum(rate for pid, ths in self.threads_of().items()
                               if self.cpu_of(pid) == cpu for _t, _c, _i, rate in ths) * n
                    if self.opts.get("c13_foreign") and cpu == 4:
                        user += 30 * n
                    w.write_row([t, cpu, user, 0, 0, max(0, 100 * n - user - n), 0, 0, n, 0,
                                 2000000])
        with JsonlWriter(sc("gpu.jsonl"), "capkit.sidecar.gpu/1", self.ident("sidecar"),
                         ["t_rel_ns", "gpu_index", "util_pct", "mem_used_mib", "compute_pids",
                          "query_rc"]) as w:
            # rewrite the header with capenv.sidecar's extra keys (INTERFACES 5.6)
            w._f.seek(0)
            w._f.truncate()
            w._f.write(json.dumps({"_header": True, "schema": "capkit.sidecar.gpu/1",
                                   "identity": self.ident("sidecar").to_dict(),
                                   "columns": w.columns, "gpu_count": 1,
                                   "positive_control_ok": True,
                                   "written_wall": wall_iso()}, sort_keys=True) + "\n")
            for t in gpu_times:
                util = 5 if (self.opts.get("gpu_busy") and t == 6 * S) else 0
                if self.opts.get("gpu_unreadable") and t == 14 * S:
                    util = None  # nvidia-smi printed [N/A]
                w.write_row([t, 0, util, 0, [], 0])
        with JsonlWriter(sc("snapshots.jsonl"), "capkit.sidecar.snapshots/1",
                         self.ident("sidecar"), ["t_rel_ns", "pid", "seq", "snap_t_rel_ns",
                                                 "counters", "hists"]) as w:
            for t in times:
                snap_t = t - 30 * MS
                for pid, (role, k) in sorted(self.pid_role().items()):
                    if role == "capload":
                        v = self.sim.inflight(k, snap_t)
                        if self.opts.get("x_contradiction") and 13 * S <= snap_t < 17 * S:
                            v += 10 * ((snap_t - 13 * S) // S + 1)
                        c = {"backlog.inflight": v, "backlog.unread_bytes": 0}
                    else:
                        c = {"backlog.active_streams": self.sim.active_streams(snap_t),
                             "count.F2.msgs": self.sim.msgs_sent(snap_t)}
                    w.write_row([t, pid, snap_t // (100 * MS) + 1, snap_t, c, {}])


# ---- sample table ------------------------------------------------------------------------------
# name -> (simulation / file options, expect). expect: exit_code, tier_verdicts (only listed tiers
# are compared), red_checks (must all be RED somewhere in checks.tsv), not_run_checks likewise.
GOOD_V = {"2": "STABLE", "4": "STABLE"}
SAMPLES = {
    "good": ({}, {"exit_code": 0, "tier_verdicts": GOOD_V, "red_checks": []}),
    "c1_send_lag": ({"c1": True}, {"exit_code": 10, "tier_verdicts": {"2": "STABLE",
                                                                      "4": "VOID"},
                                   "red_checks": ["C1"]}),
    "c3_non_overload": ({"c3": True}, {"exit_code": 10, "tier_verdicts": {"4": "VOID"},
                                       "red_checks": ["C3"]}),
    "c5_requests": ({"c5": True}, {"exit_code": 10, "tier_verdicts": {"4": "VOID"},
                                   "red_checks": ["C5"]}),
    "c6_probe_miscount": ({"c6": True}, {"exit_code": 10, "tier_verdicts": {"4": "VOID"},
                                         "red_checks": ["C6"]}),
    "c7_dropped_chunks": ({"c7": True}, {"exit_code": 10, "tier_verdicts": {"4": "VOID"},
                                         "red_checks": ["C7"]}),
    "c8_probe_unwired": ({"c8": True}, {"exit_code": 10, "tier_verdicts": {"4": "VOID"},
                                        "red_checks": ["C8"]}),
    "c9_pinning": ({"c9": True}, {"exit_code": 10, "tier_verdicts": {"4": "VOID"},
                                  "red_checks": ["C9"]}),
    "c10_versions": ({"c10": True}, {"exit_code": 11, "tier_verdicts": {"2": "NOT_RUN",
                                                                        "4": "NOT_RUN"},
                                     "red_checks": [], "not_run_checks": ["C10"]}),
    "c11_stale_gate": ({"c11": True}, {"exit_code": 10, "tier_verdicts": {"4": "VOID"},
                                       "red_checks": ["C11"]}),
    "c13_unknown_thread": ({"c13_threads": True}, {"exit_code": 10,
                                                   "tier_verdicts": {"2": "VOID", "4": "VOID"},
                                                   "red_checks": ["C13"]}),
    "c13_foreign_busy": ({"c13_foreign": True}, {"exit_code": 10, "tier_verdicts": {
        "2": "VOID", "4": "VOID"}, "red_checks": ["C13"]}),
    "gpu_busy": ({"gpu_busy": True}, {"exit_code": 10, "tier_verdicts": {"2": "VOID",
                                                                         "4": "VOID"},
                                      "red_checks": ["GPU-IDLE"]}),
    "gpu_util_unreadable": ({"gpu_unreadable": True}, {
        "exit_code": 11, "tier_verdicts": {"2": "NOT_RUN", "4": "NOT_RUN"}, "red_checks": [],
        "not_run_checks": ["GPU-IDLE"]}),
    "p_sent_short": ({"p_sent": True}, {"exit_code": 11, "tier_verdicts": {"2": "STABLE",
                                                                           "4": "NOT_RUN"},
                                        "red_checks": [], "not_run_checks": ["P-SENT"]}),
    "records_overflow": ({"overflow": True}, {"exit_code": 10, "tier_verdicts": {"4": "VOID"},
                                              "red_checks": ["RECORDS"]}),
    "probe_tier_mismatch": ({"probe_mismatch": True}, {"exit_code": 10, "tier_verdicts": {
        "2": "VOID", "4": "VOID"}, "red_checks": ["PROBE-TIER"]}),
    "x_contradiction": ({"x_contradiction": True}, {"exit_code": 10, "tier_verdicts": {
        "2": "STABLE", "4": "VOID"}, "red_checks": []}),
    "sat_overloaded_tier": ({"overload": True}, {"exit_code": 0, "tier_verdicts": {
        "2": "STABLE", "4": "UNSTABLE"}, "red_checks": []}),
    "sat_spin_thread_same_process": ({"spin": True}, {"exit_code": 0, "tier_verdicts": {
        "2": "STABLE", "4": "UNSTABLE"}, "red_checks": []}),
    "input_registered_late": ({"register_late": True}, {"exit_code": 13}),
    "input_identity_mismatch": ({"identity_mismatch": True}, {"exit_code": 13}),
}


def build(out, name, opts, exp):
    W = os.path.join(out, name)
    os.makedirs(W)
    b = Builder(W, name, opts)
    b.build_registered()
    b.build_ledgers()
    b.build_tree()
    b.build_tiers()
    b.build_sidecar()
    if opts.get("identity_mismatch"):
        t = b.tt.tiers[4]
        p = b.T.tier_file(t.seq, t.label, "nullsse", 0, "summary", "json")
        with open(p) as f:
            doc = json.load(f)
        doc["identity"]["arm"] = "synthetic-b"
        write_plain(p, doc)
    exp = dict(exp, tree=b.tree_rel, synthetic=True)
    if name == "good":
        t = b.tt.tiers[4]
        target = os.path.relpath(b.T.tier_file(t.seq, t.label, "nullsse", 0, "summary",
                                               "json"), W)
        write_plain(os.path.join(W, "sabotage.json"), {
            "file": target, "json_path": ["counts", "requests"], "add": -1,
            "expect": {"exit_code": 10, "tier_verdicts": {"4": "VOID"}, "red_checks": ["C5"]}})
    write_plain(os.path.join(W, "expect.json"), exp)
    return W


def build_empty(out):
    W = os.path.join(out, "empty_tree")
    os.makedirs(W)
    open(os.path.join(W, "JUDGED.tsv"), "w").close()
    # git keeps no empty directories: the runner creates runs/empty (and asserts it is empty)
    write_plain(os.path.join(W, "expect.json"), {"exit_code": 11, "tree": "runs/empty",
                                                 "create_empty_tree": True,
                                                 "synthetic": True})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True,
                    help="directory to build into; no sample subdirectory may exist yet")
    ap.add_argument("--only", action="append", help="build only these samples")
    a = ap.parse_args(argv)
    names = [n for n in list(SAMPLES) + ["empty_tree"] if not a.only or n in a.only]
    clash = [n for n in names if os.path.exists(os.path.join(a.out, n))]
    if clash:
        sys.stderr.write("refusing: %s already exist under %s\n" % (clash, a.out))
        return 2
    os.makedirs(a.out, exist_ok=True)
    for name, (opts, exp) in SAMPLES.items():
        if a.only and name not in a.only:
            continue
        build(a.out, name, opts, exp)
        print("built", name)
    if not a.only or "empty_tree" in a.only:
        build_empty(a.out)
        print("built empty_tree")
    return 0


if __name__ == "__main__":
    sys.exit(main())

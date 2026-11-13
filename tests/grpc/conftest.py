# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for the gRPC tool tests: tier tables, schedules, prompts, and a launch harness that
starts the tools as real processes on 127.0.0.1 (port 0 + port files) and always kills them."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

SHA = "a" * 64
MS = 1_000_000
GATE = {"mode": "enforce", "rule": "abs_max",
        "readings": [{"name": "inflight",
                      "terms": [{"role": "drain", "slot": "backlog.inflight", "sign": 1,
                                 "agg": "sum"}], "abs_max": 0}],
        "poll_ns": 100 * MS, "settle_polls": 2, "deadline_guard_ns": 50 * MS,
        "max_snapshot_age_ns": 200 * MS}


def tier(seq, label, kind, start, window, rate, n=8, k=1, s=1, beat=10 * MS, inject=None,
         prompt_set="p"):
    load = kind != "drain"
    return {"seq": seq, "label": label, "kind": kind, "start_ns": start, "ramp_ns": 0,
            "window_ns": window, "tail_ns": 0, "rate_rps": rate if load else 0.0,
            "streams_c": None, "max_tokens": n if load else 0, "tokens_per_msg": k,
            "stream_interval": s, "beat_ns": beat, "prompt_set": prompt_set if load else None,
            "inject": inject or [], "gate": None, "baseline": False, "note": "",
            "load_value": rate if load else None}


def table_doc(spec, launch_inject=None, timeout_ns=10_000_000_000):
    """spec: list of (kind, window_ms, rate, extra-kwargs); start times are chained."""
    tiers, start = [], 0
    for i, (kind, win_ms, rate, kw) in enumerate(spec):
        label = "%s-%d" % (kind, i)
        tiers.append(tier(i, label, kind, start, win_ms * MS, rate, **kw))
        start += win_ms * MS
    return {"schema": "capkit.tiers/1", "experiment": "E0", "name": "grpc-test",
            "arrival": {"dist": "fixed", "seed": 1},
            "axis": {"name": "R", "load_key": "rate_rps", "unit": "req/s", "order": "asc"},
            "request_timeout_ns": timeout_ns, "beat_origin_rel_ns": 0, "default_gate": GATE,
            "launch_inject": launch_inject or [], "tiers": tiers}


def prompts_doc():
    return {"schema": "capkit.prompts/1", "source": {"tokenizer": None, "note": "test"},
            "sets": {"p": {"kind": "token_ids", "items": [[1, 2, 3], [4, 5, 6, 7]]},
                     "t": {"kind": "text", "items": ["hello", "world"]}}}


class Launch:
    """One local launch: run tree, shm dir, files, processes. kill_all() at teardown."""

    def __init__(self, tmp_path, python):
        from capcommon.runtree import RunTree
        self.tmp = tmp_path
        self.py = python
        self.run_dir = str(tmp_path / "T")
        self.tree = RunTree(self.run_dir)
        self.tree.make_skeleton()
        self.shm = str(tmp_path / "shm")
        os.makedirs(self.shm, exist_ok=True)
        self.procs = {}
        self.origin = None
        self.tt_path = None

    def write_table(self, doc, max_rows=100000):
        from capcommon.schedule import write_schedule
        from capcommon.tiers import TierTable
        self.tt_path = str(self.tmp / "tiers.json")
        with open(self.tt_path, "w") as f:
            json.dump(doc, f)
        self.tt = TierTable.load(self.tt_path)
        self.sched_path = str(self.tmp / "schedule.capbin")
        self.sched_meta = write_schedule(self.sched_path, self.tt, max_rows)
        self.prompts_path = str(self.tmp / "prompts.json")
        with open(self.prompts_path, "w") as f:
            json.dump(prompts_doc(), f)
        return self.tt

    def set_origin(self, lead_s):
        self.origin = time.monotonic_ns() + int(lead_s * 1e9)
        return self.origin

    def common(self, role, cpus="0", loop="asyncio"):
        return ["--experiment", "E0", "--arm", "unit-a", "--launch-seq", "1",
                "--job-id", "local-grpctest", "--restart-count", "0", "--register-sha", SHA,
                "--role", role, "--run-dir", self.run_dir, "--loop", loop,
                "--tier-table", self.tt_path, "--origin-ns", str(self.origin),
                "--shm-dir", self.shm, "--snapshot-interval-ms", "50",
                "--ready-file", self.tree.ready_file(role), "--cpus", cpus,
                "--stop-grace-s", "1.0", "--lag-period-us", "1000"]

    def start(self, role, module, args):
        env = dict(os.environ)
        env["PYTHONPATH"] = TOOLS
        env.pop("CAPKIT_CODE_SHA", None)
        out = open(str(self.tmp / ("%s.out" % role)), "wb")
        err = open(str(self.tmp / ("%s.err" % role)), "wb")
        p = subprocess.Popen([self.py, "-m", module] + args, env=env, stdout=out, stderr=err,
                             cwd=str(self.tmp))
        self.procs[role] = p
        return p

    def server(self, role, module, extra, cpus="0", loop="asyncio"):
        args = self.common(role, cpus, loop) + ["--port-file", self.tree.port_file(role),
                                                "--host", "127.0.0.1", "--port", "0"] + extra
        return self.start(role, module, args)

    def wait_file(self, path, timeout_s, role=None):
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if os.path.exists(path):
                return True
            if role and self.procs[role].poll() is not None:
                raise AssertionError("%s exited %s before %s:\n%s" % (
                    role, self.procs[role].returncode, path, self.stderr(role)))
            time.sleep(0.02)
        raise AssertionError("timeout waiting for %s\n%s" % (path, self.stderr(role) if role
                                                               else ""))

    def stderr(self, role):
        p = self.tmp / ("%s.err" % role)
        return p.read_text(errors="replace") if p.exists() else ""

    def wait_exit(self, role, timeout_s):
        try:
            return self.procs[role].wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            raise AssertionError("%s did not exit in %ss\n%s" % (role, timeout_s,
                                                                  self.stderr(role)))

    def stop(self, role, timeout_s=10):
        p = self.procs[role]
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
        return self.wait_exit(role, timeout_s)

    def kill_all(self):
        for p in self.procs.values():
            if p.poll() is None:
                p.kill()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

    def summaries(self, role):
        """{tier_seq: [summary docs of every worker]} read from the run tree."""
        from capcommon.jsonio import read_json_stamped
        out = {}
        base = os.path.join(self.run_dir, "raw", "tiers")
        for d in sorted(os.listdir(base)):
            for f in sorted(os.listdir(os.path.join(base, d))):
                if f.startswith(role + ".w") and f.endswith(".summary.json"):
                    ident, doc = read_json_stamped(os.path.join(base, d, f),
                                                   "capkit.tier_summary/1")
                    out.setdefault(ident.tier_seq, []).append(doc)
        return out

    def records(self, role, name):
        """All rows of <role>.w*.<name>.capbin across tier dirs: {col: list}, plus headers."""
        from capcommon.capbin import read_capbin
        cols, headers = {}, []
        base = os.path.join(self.run_dir, "raw", "tiers")
        for d in sorted(os.listdir(base)):
            for f in sorted(os.listdir(os.path.join(base, d))):
                if f.startswith(role + ".w") and f.endswith(".%s.capbin" % name):
                    h, c = read_capbin(os.path.join(base, d, f))
                    headers.append(h)
                    for k, v in c.items():
                        cols.setdefault(k, []).extend(v)
        return cols, headers


@pytest.fixture
def launch(tmp_path):
    L = Launch(tmp_path, sys.executable)
    try:
        yield L
    finally:
        L.kill_all()


def total(summ, path):
    """Sum of summary field `path` (dotted) over all tiers and workers."""
    acc = 0
    for docs in summ.values():
        for d in docs:
            x = d
            for k in path.split("."):
                x = x[k]
            acc += x
    return acc


def per_tier(summ, path):
    out = {}
    for seq, docs in summ.items():
        acc = 0
        for d in docs:
            x = d
            for k in path.split("."):
                x = x[k]
            acc += x
        out[seq] = acc
    return out

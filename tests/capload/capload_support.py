# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test support: a local launch of capload -> capload.nullsse on 127.0.0.1 (functional only).

Not named conftest.py on purpose: test modules import helpers from here by a unique module
name, so several test directories can be collected in one pytest session.
"""

import copy
import glob
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

from capcommon.capbin import read_capbin  # noqa: E402
from capcommon.jsonio import read_json_stamped  # noqa: E402
from capcommon.runtree import RunTree  # noqa: E402
from capcommon.schedule import write_schedule  # noqa: E402
from capcommon.tiers import TierTable  # noqa: E402

SHA = "b" * 64
MS = 1_000_000
GATE = {"mode": "enforce", "rule": "abs_max",
        "readings": [{"name": "inflight",
                      "terms": [{"role": "capload", "slot": "backlog.inflight", "sign": 1,
                                 "agg": "sum"}], "abs_max": 0}],
        "poll_ns": 100 * MS, "settle_polls": 3, "deadline_guard_ns": 50 * MS,
        "max_snapshot_age_ns": 200 * MS}


def tier(seq, label, kind, start, dur, rate, n=5, k=1, s=1, beat=10 * MS, inject=None,
         baseline=False, prompt_set="p"):
    load = kind != "drain"
    return {"seq": seq, "label": label, "kind": kind, "start_ns": start, "ramp_ns": 0,
            "window_ns": dur, "tail_ns": 0, "rate_rps": rate if load else 0.0,
            "streams_c": None, "max_tokens": n if load else 0, "tokens_per_msg": k,
            "stream_interval": s, "beat_ns": beat, "prompt_set": prompt_set if load else None,
            "inject": inject or [], "gate": None, "baseline": baseline, "note": "",
            "load_value": (rate if load else None)}


def table_doc(spec, timeout_ns=30_000 * MS):
    """spec: list of (kind, dur_ms, rate, extra kwargs) -> tier table doc with contiguous tiers."""
    tiers = []
    start = 0
    for i, (kind, dur_ms, rate, kw) in enumerate(spec):
        label = "%s-%d" % (kind, i)
        tiers.append(tier(i, label, kind, start, dur_ms * MS, rate, **kw))
        start += dur_ms * MS
    return {"schema": "capkit.tiers/1", "experiment": "E1", "name": "unit",
            "arrival": {"dist": "fixed", "seed": 1},
            "axis": {"name": "R", "load_key": "rate_rps", "unit": "req/s", "order": "asc"},
            "request_timeout_ns": timeout_ns, "beat_origin_rel_ns": 0,
            "default_gate": copy.deepcopy(GATE), "launch_inject": [], "tiers": tiers}


def basic_spec(rate_w=40.0, rate_up=80.0, n=5, up_kw=None, drain_ms=300):
    up = {"n": n, "baseline": True}
    up.update(up_kw or {})
    return [("warmup", 500, rate_w, {"n": n}), ("drain", drain_ms, 0.0, {}),
            ("up", 600, rate_up, up), ("drain", 500, 0.0, {})]


class Launch:
    def __init__(self, root, doc, prompts=None, lead_s=2.5):
        self.root = str(root)
        os.makedirs(self.root, exist_ok=True)
        self.run_dir = os.path.join(self.root, "T")
        RunTree(self.run_dir).make_skeleton()
        self.shm = os.path.join(self.root, "shm")
        os.makedirs(self.shm, exist_ok=True)
        self.table_path = os.path.join(self.root, "tiers.json")
        with open(self.table_path, "w") as f:
            json.dump(doc, f)
        self.table = TierTable.load(self.table_path)
        self.sched_path = os.path.join(self.root, "schedule.capbin")
        self.sched_meta = write_schedule(self.sched_path, self.table, 100000)
        self.prompts_path = os.path.join(self.root, "prompts.json")
        with open(self.prompts_path, "w") as f:
            json.dump(prompts or {"schema": "capkit.prompts/1", "sets": {
                "p": {"kind": "token_ids", "items": [[1, 2, 3], [4, 5, 6, 7], [8]]}}}, f)
        self.cpus = sorted(os.sched_getaffinity(0))
        self.origin = time.monotonic_ns() + int(lead_s * 1e9)
        self.port_file = os.path.join(self.run_dir, "raw", "procs", "ports", "nullsse.json")
        self.procs = []

    def common(self, role):
        return ["--experiment", "E1", "--arm", "unit-a", "--launch-seq", "1",
                "--job-id", "local-test", "--restart-count", "0", "--register-sha", SHA,
                "--role", role, "--run-dir", self.run_dir, "--loop", "asyncio",
                "--tier-table", self.table_path, "--origin-ns", str(self.origin),
                "--shm-dir", self.shm, "--snapshot-interval-ms", "50",
                "--ready-file", os.path.join(self.run_dir, "raw", "procs", "ready",
                                             "%s.json" % role),
                "--stop-grace-s", "1.0"]

    def _spawn(self, argv, name):
        env = dict(os.environ, PYTHONPATH=TOOLS)
        out = open(os.path.join(self.root, "%s.out" % name), "w")
        err = open(os.path.join(self.root, "%s.err" % name), "w")
        p = subprocess.Popen([sys.executable] + argv, env=env, stdout=out, stderr=err)
        p.logname = os.path.join(self.root, "%s.err" % name)
        self.procs.append(p)
        return p

    def start_sink(self, workers=2, pacing="asap", chunk_bytes=200, idle_close_s=5.0, extra=()):
        argv = (["-m", "capload.nullsse"] + self.common("nullsse")
                + ["--cpus", ",".join(str(c) for c in self.cpus[:workers]),
                   "--port-file", self.port_file, "--host", "127.0.0.1", "--port", "0",
                   "--workers", str(workers), "--chunk-bytes", str(chunk_bytes),
                   "--pacing", pacing, "--idle-close-s", str(idle_close_s),
                   "--lag-period-us", "2000"] + list(extra))
        return self._spawn(argv, "nullsse")

    def start_load(self, workers=2, conn_policy="keepalive", recv_mode="count",
                   pool_reset="per-tier", pool_pick="roundrobin", sample_mod=1, extra=(),
                   endpoint="/v1/completions"):
        argv = (["-m", "capload"] + self.common("capload")
                + ["--cpus", ",".join(str(c) for c in self.cpus[-workers:]),
                   "--target-port-file", self.port_file, "--endpoint", endpoint,
                   "--model", "m", "--workers", str(workers), "--schedule", self.sched_path,
                   "--prompts", self.prompts_path, "--conn-policy", conn_policy,
                   "--pool-pick", pool_pick, "--pool-seed", "3", "--pool-reset", pool_reset,
                   "--recv-mode", recv_mode, "--chunk-sample-mod", str(sample_mod),
                   "--fionread-every-ms", "20", "--lag-period-us", "2000",
                   "--max-records-per-worker", "200000"] + list(extra))
        return self._spawn(argv, "capload")

    def wait(self, p, timeout):
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            raise AssertionError("process did not exit in %.1fs; stderr:\n%s"
                                 % (timeout, open(p.logname).read()[-3000:]))

    def stop(self, p, timeout=10):
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
        return self.wait(p, timeout)

    def run_pair(self, sink_kw=None, load_kw=None, load_timeout=None):
        """Start sink then load; wait for load; SIGTERM the sink. -> (load rc, sink rc)."""
        sink = self.start_sink(**(sink_kw or {}))
        load = self.start_load(**(load_kw or {}))
        if load_timeout is None:
            load_timeout = (self.origin - time.monotonic_ns()) / 1e9 + self.table.end_ns / 1e9 + 15
        rc_l = self.wait(load, load_timeout)
        rc_s = self.stop(sink)
        return rc_l, rc_s

    def errtext(self, name):
        with open(os.path.join(self.root, "%s.err" % name)) as f:
            return f.read()

    # ---- readers ---------------------------------------------------------------------------
    def summaries(self, role, seq):
        t = self.table.tiers[seq]
        d = RunTree(self.run_dir).tier_dir(seq, t.label)
        out = []
        for p in sorted(glob.glob(os.path.join(d, "%s.w*.summary.json" % role))):
            out.append(read_json_stamped(p, "capkit.tier_summary/1")[1])
        return out

    def capbins(self, role, name, seq):
        t = self.table.tiers[seq]
        d = RunTree(self.run_dir).tier_dir(seq, t.label)
        out = []
        for p in sorted(glob.glob(os.path.join(d, "%s.w*.%s.capbin" % (role, name)))):
            out.append(read_capbin(p))
        return out

    def sum_count(self, role, seq, path):
        tot = 0
        for s in self.summaries(role, seq):
            v = s
            for part in path.split("."):
                v = v[part]
            tot += v
        return tot

    def kill_all(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.wait()


@pytest.fixture
def launch(tmp_path):
    made = []

    def _make(doc=None, **kw):
        lz = Launch(tmp_path / ("L%d" % len(made)), doc or table_doc(basic_spec()), **kw)
        made.append(lz)
        return lz

    yield _make
    for lz in made:
        lz.kill_all()

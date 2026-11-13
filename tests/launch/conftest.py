# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tests/launch: put capcommon, caplaunch, capstubs on sys.path; build minimal workdirs."""

import json
import os
import subprocess
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(C, "tools")
PATHS = [os.path.join(HERE, "fakepkgs"), TOOLS]
for _p in reversed(PATHS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from capcommon.hashing import sha256_file  # noqa: E402
from capcommon.schedule import write_schedule  # noqa: E402
from capcommon.tiers import TierTable  # noqa: E402

DUMMY = os.path.join(HERE, "fakes", "dummy_product.py")
MS = 1_000_000
ZERO_SHA = "0" * 40


def gate(roles_slots, poll_ms=100, settle=2, max_age_ms=150, guard_ms=100):
    return {"mode": "enforce", "rule": "abs_max",
            "readings": [{"name": "%s_%s" % (r, s.split(".")[-1]),
                          "terms": [{"role": r, "slot": s, "sign": 1, "agg": "sum"}],
                          "abs_max": 0} for r, s in roles_slots],
            "poll_ns": poll_ms * MS, "settle_polls": settle, "deadline_guard_ns": guard_ms * MS,
            "max_snapshot_age_ns": max_age_ms * MS}


def tier_doc(gate_cfg, load_ms=600, drain_ms=800, rates=(20.0, 40.0, 20.0)):
    tiers, t = [], 0
    kinds = ["warmup", "drain", "up", "drain", "down", "drain"]
    li = 0
    for seq, kind in enumerate(kinds):
        dur = drain_ms if kind == "drain" else load_ms
        load = kind != "drain"
        rate = rates[li] if load else 0.0
        tiers.append({"seq": seq, "label": "%s-%d" % (kind, seq), "kind": kind, "start_ns": t,
                      "ramp_ns": 0, "window_ns": dur * MS, "tail_ns": 0,
                      "rate_rps": rate, "streams_c": None, "max_tokens": 4 if load else 0,
                      "tokens_per_msg": 1, "stream_interval": 1, "beat_ns": 0,
                      "prompt_set": "p" if load else None, "inject": [], "gate": None,
                      "baseline": kind == "up", "note": "",
                      "load_value": (rate if kind != "warmup" else None) if load else None})
        if load:
            li += 1
        t += dur * MS
    return {"schema": "capkit.tiers/1", "experiment": "E0", "name": "test", "arrival":
            {"dist": "fixed", "seed": 1}, "axis": {"name": "R", "load_key": "rate_rps",
                                                   "unit": "req/s", "order": "asc"},
            "request_timeout_ns": None, "beat_origin_rel_ns": 0, "default_gate": gate_cfg,
            "launch_inject": [], "tiers": tiers}


def role(name, launcher, kind, cpus_needed=1, workers=1, expected=1, mclass="stub_out_of_domain",
         asyncio=True, loop="asyncio", probe_tier=None, readiness=None, backlog=(),
         snapshots="all", args=(), env=None, start_after=(), **extra):
    r = {"role": name, "launcher": launcher, "kind": kind, "start_after": list(start_after),
         "pin_role": name, "cpus_needed": cpus_needed, "numa": None, "workers": workers,
         "expected_procs": expected, "measure_class": mclass, "asyncio": asyncio,
         "uses_grpc": False, "loop": loop if asyncio else None, "probe_tier": probe_tier,
         "readiness": readiness or [{"kind": "capkit_ready"}], "backlog_slots": list(backlog),
         "snapshots": snapshots, "args": list(args), "env": env or {}}
    r.update(extra)
    return r


def default_roles(server_args=(), client_args=(), product_args=(), with_observers=True):
    rs = [
        role("nullsse", "capkit", "server", cpus_needed=2, workers=2, expected=2,
             module="capload.nullsse", backlog=["backlog.active_streams"],
             args=["--slot", "backlog.active_streams"] + list(server_args)),
        role("product", "product", "server", asyncio=False, probe_tier="P0", snapshots="none",
             command=[sys.executable, DUMMY], port={"kind": "plain_port_file",
                                                    "host": "127.0.0.1"},
             readiness=[{"kind": "plain_port_file"},
                        {"kind": "http_get", "path": "/health", "expect_status": 200,
                         "timeout_s": 1.0}],
             args=["--host", "127.0.0.1", "--port-file", "{own:port_file}"] + list(product_args)),
        role("capload", "capkit", "client", cpus_needed=2, workers=1, mclass="load_source",
             module="capload", backlog=["backlog.inflight"], start_after=["nullsse", "product"],
             args=["--slot", "backlog.inflight", "--exit-after-end",
                   "--target-port-file", "{port_file:nullsse}", "--peer", "{addr:product}"]
             + list(client_args)),
    ]
    if with_observers:
        rs.append(role("sidecar", "capkit", "aux", asyncio=False, mclass="observer",
                       module="capenv.sidecar", snapshots="none", start_after=["capload"],
                       args=["--interval-ms", "50", "--gpu-every-s", "1", "--numa-every-s",
                             "1"]))
        rs.append(role("canary", "capkit", "aux", asyncio=False, mclass="observer",
                       module="capenv.canary", snapshots="none",
                       args=["--interval-ms", "50"]))
    return rs


def pick_cpus(roles):
    """Disjoint cpus from this process's affinity (probed at run time, never hardcoded)."""
    avail = sorted(os.sched_getaffinity(0))
    need = sum(r["cpus_needed"] for r in roles)
    if len(avail) < need:
        import pytest  # noqa: PLC0415
        pytest.skip("needs %d cpus, have %d" % (need, len(avail)))
    out, i = {}, 0
    for r in roles:
        out[r["pin_role"]] = {"cpus": avail[i:i + r["cpus_needed"]], "numa_node": 0}
        i += r["cpus_needed"]
    return out


def make_workdir(tmp, tdoc, roles, stop_order=None, params_required=(), reconcile=None):
    """A minimal <W>: tier table, schedule, prompts, scenario, register.json, pin table."""
    W = os.path.join(str(tmp), "W")
    os.makedirs(os.path.join(W, "reg"), exist_ok=True)
    tt_path = os.path.join(W, "reg", "tiers.json")
    with open(tt_path, "w") as f:
        json.dump(tdoc, f)
    tt = TierTable.load(tt_path)
    write_schedule(os.path.join(W, "reg", "schedule.capbin"), tt, 100000)
    with open(os.path.join(W, "reg", "prompts.json"), "w") as f:
        json.dump({"schema": "capkit.prompts/1", "source": {"tokenizer": None, "note": "t"},
                   "sets": {"p": {"kind": "token_ids", "items": [[1, 2, 3]]}}}, f)
    names = [r["role"] for r in roles]
    scen = {"schema": "capkit.scenario/1", "experiment": "E0", "name": "t",
            "tier_table": "reg/tiers.json", "schedule": "reg/schedule.capbin",
            "prompts": "reg/prompts.json", "probe_tier": "P0", "probe_groups": [],
            "stop_grace_s": 1.0, "stop_order": stop_order or names,
            "reconcile": reconcile or {"schema": "capkit.reconcile/1", "chains": [
                {"name": "requests", "kind": "requests", "gate": None, "include_usage": None,
                 "hops": [{"role": names[0], "source": "summary.counts.ok", "scope": "tier"}],
                 "corrections": []}]},
            "roles": roles, "params_required": list(params_required),
            "dims": {"tokenizer": None, "stub": "test", "response_timing_switches": None,
                     "interpretation": "test", "log_sink": "file"}}
    with open(os.path.join(W, "reg", "scenario.json"), "w") as f:
        json.dump(scen, f)
    files = {k: {"relpath": "reg/%s" % n, "sha256": sha256_file(os.path.join(W, "reg", n))}
             for k, n in (("tier_table", "tiers.json"), ("schedule", "schedule.capbin"),
                          ("prompts", "prompts.json"), ("scenario", "scenario.json"))}
    reg = {"schema": "capkit.register/1", "experiment": "E0", "arms": ["arm-a"],
           "files": files, "calibration": [], "inline_thresholds": {},
           "expected": {"launches_per_arm": 1, "job_budget": 1}, "judge": {}}
    with open(os.path.join(W, "reg", "register.json"), "w") as f:
        json.dump(reg, f)
    pin = {"schema": "capkit.pin_table/1", "source_sha": None, "rules": "test",
           "siblings_left_idle": [], "roles": pick_cpus(roles), "unassigned": []}
    with open(os.path.join(W, "pin_table.json"), "w") as f:
        json.dump(pin, f)
    return {"W": W, "register": os.path.join(W, "reg", "register.json"),
            "register_sha": sha256_file(os.path.join(W, "reg", "register.json")),
            "scenario": os.path.join(W, "reg", "scenario.json"),
            "pin": os.path.join(W, "pin_table.json")}


def launch_argv(wd, run_dir, shm_root, lead_ms=4000, params=(), launch_seq=0):
    a = [sys.executable, "-m", "caplaunch",
         "--experiment", "E0", "--arm", "arm-a", "--launch-seq", str(launch_seq),
         "--job-id", "local-t", "--restart-count", "0", "--register-sha", wd["register_sha"],
         "--role", "launch", "--run-dir", run_dir, "--workdir", wd["W"],
         "--register", wd["register"], "--scenario", wd["scenario"], "--pin-table", wd["pin"],
         "--shm-dir", shm_root, "--snapshot-interval-ms", "40", "--origin-lead-ms",
         str(lead_ms), "--housekeeping-ms", "40", "--kill-after-s", "2",
         "--capkit-sha", ZERO_SHA, "--probe-tree-sha", "none", "--baseline-sha", ZERO_SHA,
         "--snapshot-dir", "none", "--image-path", "none", "--image-sha256", "none",
         "--hw-profile-sha", "none"]
    for p in params:
        a += ["--param", p]
    return a


def run_launch(argv, timeout=120):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(PATHS)
    env.pop("CAPPROBE_TIER", None)
    env.pop("CAPPROBE_GROUPS", None)
    return subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)

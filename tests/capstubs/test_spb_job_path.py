# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The job path of S-PB with job inputs (errata B2B-9, RS21): python -m capjobs.jobinputs runs
the REAL generators (capstubs.prebaked, capgrpc.reqbytes openengine; the pb2 module is the test
double oe_double) for a scenario that declares them, then python -m caplaunch starts S-PB and
capgrpc.load --request-format openengine with --job-input, the way capjobs.launches does in a
job. The judge's job-input check and C10's stub-module item then read the finished tree, in
Generate-only mode and in frontend mode (--sfe-module = a stand-in S-FE module listed in the
overlay manifest's stub_modules). The child processes get PYTHONPATH here; in a job the overlay
is installed into site-packages instead (errata B2B-14)."""

import hashlib

import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import HERE, METHOD, STANDIN, TOOLS, g, per_tier, total
from capcommon import expect
from capcommon.hashing import sha256_file
from capcommon.runtree import read_state
from capcommon.schedule import write_schedule
from capcommon.tiers import TierTable

ZERO = "0" * 40


def _role(name, module, args, backlog, start_after=()):
    return {"role": name, "launcher": "capkit", "kind": "server" if name == "spb" else "client",
            "start_after": list(start_after), "pin_role": name, "cpus_needed": 1, "numa": None,
            "workers": 1, "expected_procs": 1,
            "measure_class": "stub_out_of_domain" if name == "spb" else "load_source",
            "asyncio": True, "uses_grpc": True, "loop": "asyncio", "probe_tier": None,
            "readiness": [{"kind": "capkit_ready"}], "backlog_slots": backlog,
            "snapshots": "all", "args": args, "env": {}, "proc_kinds": [], "module": module}


def _workdir(tmp, frontend=False):
    W = os.path.join(tmp, "W")
    os.makedirs(os.path.join(W, "reg"))
    tt_doc = g.table_doc([("warmup", 400, 25.0, {"n": 5}), ("drain", 300, 0.0, {}),
                          ("up", 500, 40.0, {"n": 6}), ("drain", 400, 0.0, {})])
    with open(os.path.join(W, "reg", "tiers.json"), "w") as f:
        json.dump(tt_doc, f)
    write_schedule(os.path.join(W, "reg", "schedule.capbin"),
                   TierTable.load(os.path.join(W, "reg", "tiers.json")), 100000)
    with open(os.path.join(W, "reg", "prompts.json"), "w") as f:
        json.dump(g.prompts_doc(), f)
    sfe = ["--sfe-module", "{param:sfe_module}", "--vocab-size", "151"] if frontend else []
    roles = [
        _role("spb", "capstubs.prebaked_server",
              ["--host", "127.0.0.1", "--port", "0", "--prebaked", "{input:prebaked}",
               "--schedule", "{input:schedule}", "--lag-period-us", "1000"] + sfe,
              ["backlog.pending_msgs", "backlog.active_streams", "backlog.yielded"]),
        _role("drain", "capgrpc.load",
              ["--target-port-file", "{port_file:spb}", "--method", METHOD,
               "--channels-per-worker", "1", "--in-process", "--schedule", "{input:schedule}",
               "--request-format", "openengine", "--request-bytes", "{input:request_bytes}",
               "--msg-sample-mod", "1", "--max-records-per-worker", "1000",
               "--lag-period-us", "1000"],
              ["backlog.inflight"], start_after=["spb"])]
    scen = {"schema": "capkit.scenario/1", "experiment": "E0", "name": "spb-job-path",
            "tier_table": "reg/tiers.json", "schedule": "reg/schedule.capbin",
            "prompts": "reg/prompts.json", "probe_tier": None, "probe_groups": [],
            "stop_grace_s": 1.0, "stop_order": ["drain", "spb"],
            "reconcile": {"schema": "capkit.reconcile/1", "chains": [
                {"name": "requests", "kind": "requests", "gate": None, "include_usage": None,
                 "hops": [{"role": "drain", "source": "summary.counts.ok", "scope": "tier"},
                          {"role": "spb", "source": "summary.counts.streams_accepted",
                           "scope": "tier"}], "corrections": []}]},
            "roles": roles,
            "params_required": ["token_ids"] + (["sfe_module"] if frontend else []),
            "dims": {"tokenizer": None, "stub": "S-PB", "response_timing_switches": None,
                     "interpretation": "test", "log_sink": "file"},
            "job_inputs": {
                "prebaked": {"args": ["--pb2-module", "oe_double", "--tier-table",
                                      "{input:tier_table}", "--token-ids", "{param:token_ids}",
                                      "--prompt-tokens", "4"]},
                "request_bytes": {"args": ["openengine", "--pb2-module", "oe_double",
                                           "--prompts", "{input:prompts}", "--tier-table",
                                           "{input:tier_table}", "--model", "m0"]}}}
    with open(os.path.join(W, "reg", "scenario.json"), "w") as f:
        json.dump(scen, f)
    files = {k: {"relpath": "reg/%s" % n, "sha256": sha256_file(os.path.join(W, "reg", n))}
             for k, n in (("tier_table", "tiers.json"), ("schedule", "schedule.capbin"),
                          ("prompts", "prompts.json"), ("scenario", "scenario.json"))}
    with open(os.path.join(W, "reg", "register.json"), "w") as f:
        json.dump({"schema": "capkit.register/1", "experiment": "E0", "arms": ["arm-a"],
                   "files": files, "calibration": [], "inline_thresholds": {},
                   "expected": {"launches_per_arm": 1, "job_budget": 1}, "judge": {}}, f)
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) >= 2, "needs 2 cpus"
    with open(os.path.join(W, "pin_table.json"), "w") as f:
        json.dump({"schema": "capkit.pin_table/1", "source_sha": None, "rules": "test",
                   "siblings_left_idle": [], "roles": {
                       "spb": {"cpus": [cpus[0]], "numa_node": 0},
                       "drain": {"cpus": [cpus[1]], "numa_node": 0}}, "unassigned": []}, f)
    shutil.copy2(shutil.which("sleep"), os.path.join(W, "nvsmi"))
    stubs = []
    if frontend:  # the snapshot's copy of the S-FE module, as capprobe.overlay lists it
        data = STANDIN.encode()
        with open(os.path.join(W, "fake_openengine.py"), "wb") as f:
            f.write(data)
        stubs = [{"key": "sfe", "repo_path": "tests/fake_openengine.py", "commit": ZERO,
                  "src": "exp/tests/fake_openengine.py",
                  "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}]
    with open(os.path.join(W, "overlay_manifest.json"), "w") as f:
        json.dump({"schema": "capkit.overlay/1", "tier": "P0", "groups": [], "files": [],
                   "capkit_packages": [], "capkit_code_sha": ZERO, "sites": {},
                   "stub_modules": stubs}, f)
    os.makedirs(os.path.join(W, "jobenv"))
    return W, files


def _env(*paths):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    for k in ("CAPPROBE_TIER", "CAPPROBE_GROUPS", "CAPPROBE_INJECT", "CAPPROBE_PATCH_SHA",
              "CAPPROBE_LAG_PERIOD_US", "CAPKIT_CODE_SHA"):
        env.pop(k, None)
    return env


def _summaries(T, role):
    from capcommon.jsonio import read_json_stamped
    out = {}
    base = os.path.join(T, "raw", "tiers")
    for d in sorted(os.listdir(base)):
        for f in sorted(os.listdir(os.path.join(base, d))):
            if f.startswith(role + ".w") and f.endswith(".summary.json"):
                ident, doc = read_json_stamped(os.path.join(base, d, f), "capkit.tier_summary/1")
                out.setdefault(ident.tier_seq, []).append(doc)
    return out


@pytest.mark.parametrize("frontend", [False, True], ids=["generate_only", "frontend"])
def test_spb_job_path_with_generated_job_inputs(tmp_path, frontend):
    W, files = _workdir(str(tmp_path), frontend)
    params = ["--param", "token_ids=11,12,13"]
    if frontend:
        params += ["--param", "sfe_module=%s" % os.path.join(W, "fake_openengine.py")]
    gen = os.path.join(str(tmp_path), "job", "job_inputs", "L0")
    r = subprocess.run([sys.executable, "-m", "capjobs.jobinputs", "--workdir", W, "--scenario",
                        "reg/scenario.json", "--python", sys.executable, "--out-dir", gen]
                       + params, env=_env(TOOLS, HERE),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    job_args = r.stdout.split()
    assert job_args == ["--job-input", "request_bytes=%s" % os.path.join(gen,
                                                                         "request_bytes.json"),
                        "--job-input", "prebaked=%s" % os.path.join(gen, "prebaked.json")]
    T = os.path.join(str(tmp_path), "T")
    reg = os.path.join(W, "reg", "register.json")
    argv = [sys.executable, "-m", "caplaunch", "--experiment", "E0", "--arm", "arm-a",
            "--launch-seq", "0", "--job-id", "local-spb", "--restart-count", "0",
            "--register-sha", sha256_file(reg), "--role", "launch", "--run-dir", T,
            "--workdir", W, "--register", reg, "--scenario", os.path.join(W, "reg",
                                                                          "scenario.json"),
            "--pin-table", os.path.join(W, "pin_table.json"),
            "--shm-dir", os.path.join(str(tmp_path), "shm"), "--snapshot-interval-ms", "40",
            "--origin-lead-ms", "4000", "--housekeeping-ms", "40", "--kill-after-s", "2",
            "--capkit-sha", ZERO, "--probe-tree-sha", "none", "--baseline-sha", ZERO,
            "--snapshot-dir", "none", "--image-path", "none", "--image-sha256", "none",
            "--hw-profile-sha", "none", "--overlay-manifest",
            os.path.join(W, "overlay_manifest.json"), "--nvidia-smi", os.path.join(W, "nvsmi"),
            "--job-env-dir", os.path.join(W, "jobenv")] + params + job_args
    r = subprocess.run(argv, env=_env(TOOLS), capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    assert [s[1] for s in read_state(T)] == ["VALID"], read_state(T)
    with open(os.path.join(T, "raw", "launch.json")) as f:
        L = json.load(f)
    assert sorted(L["job_inputs"]) == ["prebaked", "request_bytes"]
    pb_copy = os.path.join(T, "raw", "env", "job_inputs", "prebaked.json")
    roles = {x["role"]: x for x in L["roles"]}
    assert pb_copy in roles["spb"]["args"]  # {input:prebaked} = the copy in the tree
    assert os.path.join(T, "raw", "env", "job_inputs", "request_bytes.json") in \
        roles["drain"]["args"]
    # the data went through: every planned request answered with the prebaked stream
    ld, sp = _summaries(T, "drain"), _summaries(T, "spb")
    planned = per_tier(ld, "counts.planned")
    assert planned[0] > 0 and planned[2] > 0
    assert per_tier(ld, "counts.ok") == planned and total(ld, "counts.failed") == 0
    for seq, n in ((0, 5), (2, 6)):
        m = expect.msgs_per_request(n, 1, 1, "product") + 1
        assert sp[seq][0]["counts"]["msgs_sent"] == planned[seq] * m == ld[seq][0]["msgs"]["msgs"]
        assert sp[seq][0]["prebaked"]["sha256"] == sha256_file(pb_copy)
    # the judge's job-input check and C10's stub-module item on the finished tree
    from capjudge import checks as JC
    from capjudge import inputs as JIN
    from capjudge.treedata import LaunchData
    ctx = JIN.Ctx(W, T)
    ctx.launch, ctx.register_files = L, files
    JIN._load_job_inputs(ctx)
    ctx.roles = roles
    ld = LaunchData(ctx)
    out = JC.stub_module_problems(ctx, ld)
    assert out == {"expected": 1, "checked": 1, "problems": []}
    info, why = ld.ready_tool_info("spb")
    assert why is None
    if frontend:  # the sha S-PB reported equals the manifest's stub module sfe
        assert info["control"]["sha256"] == ld.overlay["stub_modules"][0]["sha256"]
    else:
        assert info["control"] is None

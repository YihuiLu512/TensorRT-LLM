# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generator of configs/glm: scenario TEMPLATES for E10-S, the substitute arms of design card E10
(glm engine Python side; the real-engine arm G2 is out by Q6). The JSON files are the
deliverable; rerun into a NEW dir and diff.

Topology (as E8 with the host swapped): capgrpc.load (K = 4 connections, --planned-msgs) ->
capstubs.serve_glm host = `glm5-serve <tok> --grpc --grpc-protocol openengine` run by the glm
venv python ({param:glm_python}) with
  G0: glm's own MockEngine (--engine mock; mock_step_ms = the tier beat), or
  G1: capstubs.glm_g1.G1Engine (--engine g1; absolute step grid, one enqueue per step).
The substitute steps on a fixed period ({param:glm_step_ms}); serve_glm refuses a tier table
whose load beats differ from it. Shapes: mc (M axis, fixed beat, C doubling) and r (R axis,
max_tokens 1). The host gets 4 cores (event loop + dispatch thread + substitute thread + gRPC
core threads) plus 1 for the process-tree sampler (capstubs.procmon).

Every arm is one scenario + one tier-table template + one values checklist, as configs/snx.
Arms e10-<engine>-<shape>-vload are the same scenario with a launch-level load-source injection
in the tier table (V control, design card E10: busy delta_ns at capgrpc.load's recv.msg, once per
received message in every reader worker; the boundary should not move). Arms
e10-<engine>-<shape>-k8 double the load source (capgrpc.load with K = 8 connections / workers on
8 cores; design card E10 control "load source doubled"; the boundary should not move). Arms
e10-<engine>-mc-tps4 are the mc arm with tokens_per_msg 4 in the tier table (the substitute is
run with {param:glm_tokens_per_step} = 4, so every step message carries 4 tokens; design card E10
dimension check "tokens_per_step = 4"); scenario as the base arm.
Usage: python configs/gen_glm_configs.py <new output dir>   (writes <dir>/glm)
"""
import json
import os
import sys

OUT = sys.argv[1]
SUB = "glm"
PB2 = "tensorrt_llm.grpc.openengine._generated.generation_pb2"
GENERATE = "/openengine.v1.Inference/Generate"
LADDER = 8
K = 4
HOST_CPUS, PROCMON_CPUS = 4, 1
REG = "04_frontend/E10/register"
TEMPLATE_NOTE = ("TEMPLATE: every TBD value is filled at registration (REPLAN-20261116 R2: E10 "
                 "coverage round, exploratory)")
ENGINE_NOTE = {"g0": "G0 glm MockEngine (sleep(step) then work: period = step + work)",
               "g1": "G1 capstubs.glm_g1 (absolute step grid, one enqueue per step)"}


def w(name, doc):
    d = os.path.join(OUT, SUB)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    assert not os.path.exists(p), p
    with open(p, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=False)
        f.write("\n")


def role(name, launcher, kind, mclass, asyncio, loop, uses_grpc, probe_tier, readiness,
         backlog, snapshots, args, start_after=(), cpus_needed=1, workers=1, expected=1,
         note="", **kw):
    r = {"role": name, "launcher": launcher, "kind": kind, "start_after": list(start_after),
         "pin_role": name, "cpus_needed": cpus_needed, "numa": None, "workers": workers,
         "expected_procs": expected, "measure_class": mclass, "asyncio": asyncio,
         "uses_grpc": uses_grpc, "loop": loop, "probe_tier": probe_tier,
         "readiness": readiness, "backlog_slots": list(backlog), "snapshots": snapshots,
         "args": list(args), "env": {}, "proc_kinds": []}
    r.update(kw)
    if note:
        r["note"] = note
    return r


CAPKIT_READY = [{"kind": "capkit_ready"}]
READER_SLOTS = ["backlog.inflight", "backlog.active_streams", "backlog.read"]


def observers(after):
    return [
        role("sidecar", "capkit", "aux", "observer", False, None, False, None, CAPKIT_READY, [],
             "none", ["--interval-ms", "{param:sidecar_interval_ms}", "--gpu-every-s",
                      "{param:sidecar_gpu_every_s}", "--numa-every-s",
                      "{param:sidecar_numa_every_s}", "--nvidia-smi", "{nvidia_smi}"],
             start_after=[after], module="capenv.sidecar"),
        role("canary", "capkit", "aux", "observer", False, None, False, None, CAPKIT_READY, [],
             "none", ["--interval-ms", "{param:canary_interval_ms}"], module="capenv.canary")]


def host(engine):
    args = ["--engine", {"g0": "mock", "g1": "g1"}[engine], "--step-ms", "{param:glm_step_ms}",
            "--tokens-per-step", "{param:glm_tokens_per_step}", "--tier-table",
            "{input:tier_table}", "--out-dir", "{raw}/snx", "--record",
            "{raw}/snx/host.serve_glm.json", "--inject", "{param:glm_inject}", "--diag",
            "{param:glm_diag}", "--pick-port", "--report-host", "127.0.0.1", "--report-addr",
            "{own:report_addr}", "--procmon-out", "{raw}/snx/procmon.host.jsonl",
            "--procmon-interval-ms", "1000", "--procmon-cores", str(PROCMON_CPUS),
            "--", "{param:tokenizer_dir}", "--host", "127.0.0.1", "--max_seq_len",
            "{param:nx_max_seq_len}", "--grpc", "--grpc-protocol", "openengine"]
    return role("host", "product", "server", "under_test", True, "asyncio", True, "P0",
                [{"kind": "report_addr"}], [], "none", args,
                command=["{param:glm_python}", "-m", "capstubs.serve_glm"],
                cpus_needed=HOST_CPUS + PROCMON_CPUS,
                port={"kind": "report_addr", "host": "127.0.0.1"},
                note="serve_glm host (glm5-serve --grpc --grpc-protocol openengine, %s): event "
                     "loop + glm5-engine-dispatch + substitute thread + gRPC core threads on %d "
                     "cores, procmon on %d" % (ENGINE_NOTE[engine], HOST_CPUS, PROCMON_CPUS))


def req_bytes():
    """OpenEngine request bytes; model = the tokenizer dir (the host's served model name)."""
    return {"args": ["openengine", "--pb2-module", PB2, "--prompts", "{input:prompts}",
                     "--tier-table", "{input:tier_table}", "--model", "{param:tokenizer_dir}"]}


def grpc_load(target, procs):
    args = ["--target", "{addr:%s}" % target, "--method", GENERATE,
            "--channels-per-worker", "1", "--request-format", "openengine",
            "--request-bytes", "{input:request_bytes}", "--planned-msgs",
            "--msg-sample-mod", "{param:msg_sample_mod}", "--schedule", "{input:schedule}",
            "--lag-period-us", "{param:lag_period_us}",
            "--max-records-per-worker", "{param:reader_max_records_per_worker}",
            "--max-msg-records-per-worker", "{param:reader_max_msg_records_per_worker}"]
    return role("reader", "capkit", "client", "load_source", True, "asyncio", True, None,
                CAPKIT_READY, READER_SLOTS, "all", args, start_after=[target],
                cpus_needed=procs, workers=procs, expected=procs + 1, module="capgrpc.load",
                note="capgrpc.load, %d connection(s) (one channel per worker), --planned-msgs "
                     "(message lag against the schedule's beat grid)" % procs)


def _params(obj, out):
    if isinstance(obj, str):
        rest = obj
        while "{param:" in rest:
            rest = rest[rest.index("{param:") + 7:]
            out.add(rest[:rest.index("}")])
    elif isinstance(obj, dict):
        [_params(v, out) for v in obj.values()]
    elif isinstance(obj, list):
        [_params(v, out) for v in obj]
    return out


def scen(name, roles, note, job_inputs):
    return {"schema": "capkit.scenario/1", "experiment": "E10-S", "name": name,
            "tier_table": "%s/%s.tiers.json" % (REG, name),
            "schedule": "%s/%s.schedule.capbin" % (REG, name),
            "prompts": "%s/%s.prompts.json" % (REG, name),
            "probe_tier": "P0", "probe_groups": [], "stop_grace_s": 10.0,
            "stop_order": ["reader", "host", "sidecar", "canary"],
            "reconcile": {"schema": "capkit.reconcile/1", "chains": [
                {"name": "requests", "kind": "requests", "gate": None, "include_usage": None,
                 "hops": [{"role": "reader", "source": "summary.counts.ok", "scope": "tier"}],
                 "corrections": []}]},
            "roles": roles,
            "params_required": sorted(_params([roles, job_inputs], set())),
            "dims": {"tokenizer": "TBD-by-registration (configs/tokenizers/GLM-5.3-Flash)",
                     "stub": "E10 substitute in the glm5-serve process (capstubs.serve_glm "
                             "--engine mock|g1); no glm file changed",
                     "response_timing_switches": "glm_step_ms = the tier beat; "
                                                 "glm_tokens_per_step",
                     "interpretation": "REPLAN-20261116 R2 E10 coverage round: exploratory",
                     "log_sink": "raw/logs on the node-local disk (orchestrator redirect)"},
            "note": note + ". " + TEMPLATE_NOTE, "job_inputs": job_inputs}


def gate_t(readings):
    return {"mode": "enforce", "rule": "abs_max", "readings": readings,
            "poll_ns": "TBD:gate_poll_ns", "settle_polls": "TBD:gate_settle_polls",
            "deadline_guard_ns": "TBD:gate_deadline_guard_ns",
            "max_snapshot_age_ns": "TBD:gate_max_snapshot_age_ns"}


def rd(name, role_, slot):
    return {"name": name, "terms": [{"role": role_, "slot": slot, "sign": 1, "agg": "sum"}],
            "abs_max": 0}


READER_GATE = gate_t([rd("reader_inflight", "reader", "backlog.inflight"),
                      rd("reader_active", "reader", "backlog.active_streams")])
M_AXIS = {"name": "M", "load_key": "nominal_msgs_per_s", "unit": "msg/s", "order": "asc"}
R_AXIS = {"name": "R", "load_key": "rate_rps", "unit": "req/s", "order": "asc"}
SHAPES = {
    "mc": (M_AXIS, {"max_tokens": "TBD:n_tokens", "beat_ns": "TBD:beat_ns"},
           {"streams_c": "TBD:c_warmup", "rate_rps": "from_streams_c"},
           lambda i: {"streams_c": "TBD:c_%d" % i, "rate_rps": "from_streams_c"}),
    "r": (R_AXIS, {"max_tokens": 1, "beat_ns": "TBD:beat_ns"},
          {"rate_rps": "TBD:rate_warmup"},
          lambda i: {"rate_rps": "TBD:rate_%d" % i}),
}
SHAPE_NOTES = {
    "mc": "M axis, fixed beat (the substitute's step period), C doubling: active requests per "
          "step grow",
    "r": "R axis: shortest output (max_tokens 1), rate doubling",
}


VLOAD = [{"role": "reader", "site": "recv.msg", "mode": "busy", "delta_ns": "TBD:delta_ns"}]


def tpl(name, shape, launch_inject=(), tokens_per_msg=1):
    axis, common, warm, up = SHAPES[shape]
    common = dict(common, tokens_per_msg=tokens_per_msg, stream_interval="TBD:stream_interval",
                  prompt_set="TBD:prompt_set")
    tiers = [dict({"label": "warmup", "kind": "warmup", "ramp_ns": 0,
                   "window_ns": "TBD:warmup_ns", "tail_ns": 0}, **warm),
             {"label": "drain-01", "kind": "drain", "window_ns": "TBD:drain_max_wait_ns"}]
    for i in range(LADDER):
        tiers.append(dict({"label": "up-%02d" % (i + 1), "kind": "up", "ramp_ns": "TBD:ramp_ns",
                           "window_ns": "TBD:window_ns", "tail_ns": "TBD:tail_ns",
                           "baseline": i == 0}, **up(i)))
        tiers.append({"label": "drain-%02d" % (i + 2), "kind": "drain",
                      "window_ns": "TBD:drain_max_wait_ns"})
    return {"schema": "capkit.tiers_template/1", "experiment": "E10-S", "name": name,
            "arrival": {"dist": "fixed", "seed": "TBD:arrival_seed"}, "axis": axis,
            "request_timeout_ns": "TBD:request_timeout_ns", "default_gate": READER_GATE,
            "launch_inject": list(launch_inject), "common": common, "tiers": tiers,
            "note": "; ".join((SHAPE_NOTES[shape],) + (("V: busy delta at the reader's recv.msg "
                                                       "(every message, every worker)",)
                                                      if launch_inject else ()) + (TEMPLATE_NOTE,))}


def e10(engine, shape, vload=False, k=K, tps=1):
    name = "e10-%s-%s%s%s%s" % (engine, shape, "-vload" if vload else "",
                                "" if k == K else "-k%d" % k, "" if tps == 1 else "-tps%d" % tps)
    roles = [host(engine), grpc_load("host", k)] + observers("reader")
    w(name + ".scenario.json",
      scen(name, roles, "E10-S %s: capgrpc.load (%d connections) -> serve_glm host (glm5-serve "
                        "--grpc --grpc-protocol openengine, %s)" % (name, k, ENGINE_NOTE[engine]),
           {"request_bytes": req_bytes()}))
    w(name + ".tiers.template.json", tpl(name, shape, VLOAD if vload else (), tps))


for _engine in ("g0", "g1"):
    for _shape in ("mc", "r"):
        for _vload in (False, True):
            e10(_engine, _shape, _vload)
        e10(_engine, _shape, k=2 * K)
    e10(_engine, "mc", tps=4)

_d = os.path.join(OUT, SUB)
for n in sorted(os.listdir(_d)):
    if n.endswith(".tiers.template.json"):
        with open(os.path.join(_d, n)) as f:
            t = json.load(f)
        names = set()

        def walk(o):
            if isinstance(o, str) and o.startswith("TBD:"):
                names.add(o[4:])
            elif isinstance(o, dict):
                [walk(v) for v in o.values()]
            elif isinstance(o, list):
                [walk(v) for v in o]
        walk(t)
        w(n.replace(".tiers.template.json", ".values.todo.json"),
          {"schema": "capkit.tier_values/1", "values": {k: None for k in sorted(names)}})

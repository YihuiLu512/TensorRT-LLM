# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generator of configs/e6: scenario TEMPLATES for card E6 arm (iii) -- n independent
single-frontend trtllm-serve processes, each with its own S-PB stub process and its own capload
role (design card E6: the "pure process copy" reference; no shared stub, no multi-frontend
mechanism, so there is no connection-landing problem: capload<i> targets frontend<i> only, and
frontend<i> reaches spb<i> only).

Each role triple <i> is the E4-4b main tier (configs/e4 e4b-*): S-PB frontend mode on one core,
the new-arch frontend with 1 + 8 + 2 cores (postproc pool off), capload. The n capload roles read
the SAME registered schedule and prompts, so every tier's load is PER FRONTEND (group load =
n x the tier value) and the n arrival streams are identical in time (one arrival seed).
capload workers per role: W_PER_FRONTEND[n] (core budget: every role on one NUMA node of the
target, 56 physical cores; n = 4 needs 4 x (11 + 1 + 1) + 2 observers = 54).

Every arm is one scenario + one tier-table template + one values checklist; tier values are
"TBD:<name>", filled at registration (same names and shapes as configs/e4).

Usage: python configs/gen_e6_configs.py <new output dir>   (writes <dir>/e6)
"""
import json
import os
import sys

OUT = sys.argv[1]
SUB, EXP, REG = "e6", "E6", "04_frontend/E6/register"
PB2 = "tensorrt_llm.grpc.openengine._generated.generation_pb2"
LADDER = 8
NS = (2, 4)
W_PER_FRONTEND = {2: 2, 4: 1}
FE_INPUT_WORKERS = 8
FE_GRPC_THREADS = 2
FE_CPUS = 1 + FE_INPUT_WORKERS + FE_GRPC_THREADS
PROMPTS = "tid128"
TEMPLATE_NOTE = ("TEMPLATE: every TBD value is filled at registration (coverage round, "
                 "exploratory)")
CAPKIT_READY = [{"kind": "capkit_ready"}]
STUB_SLOTS = ["backlog.pending_msgs", "backlog.active_streams", "backlog.yielded"]


def w(name, doc):
    d = os.path.join(OUT, SUB)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    assert not os.path.exists(p), p
    with open(p, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=False)
        f.write("\n")


def role(name, launcher, kind, mclass, asyncio, loop, uses_grpc, probe_tier, readiness,
         backlog, snapshots, args, start_after=(), cpus_needed=1, workers=1,
         expected=1, note="", proc_kinds=(), **kw):
    r = {"role": name, "launcher": launcher, "kind": kind, "start_after": list(start_after),
         "pin_role": name, "cpus_needed": cpus_needed, "numa": None, "workers": workers,
         "expected_procs": expected, "measure_class": mclass, "asyncio": asyncio,
         "uses_grpc": uses_grpc, "loop": loop, "probe_tier": probe_tier,
         "readiness": readiness, "backlog_slots": list(backlog), "snapshots": snapshots,
         "args": list(args), "env": {}, "proc_kinds": list(proc_kinds)}
    r.update(kw)
    if note:
        r["note"] = note
    return r


def observers(after):
    return [
        role("sidecar", "capkit", "aux", "observer", False, None, False, None, CAPKIT_READY, [],
             "none", ["--interval-ms", "{param:sidecar_interval_ms}",
                      "--gpu-every-s", "{param:sidecar_gpu_every_s}",
                      "--numa-every-s", "{param:sidecar_numa_every_s}",
                      "--nvidia-smi", "{nvidia_smi}"],
             start_after=after, module="capenv.sidecar"),
        role("canary", "capkit", "aux", "observer", False, None, False, None, CAPKIT_READY, [],
             "none", ["--interval-ms", "{param:canary_interval_ms}"], module="capenv.canary"),
    ]


PREBAKED = {"args": ["--pb2-module", PB2, "--tier-table", "{input:tier_table}",
                     "--token-ids", "{param:spb_token_ids}",
                     "--prompt-tokens", "{param:spb_prompt_tokens}"]}
FINGERPRINT = {"args": ["--tokenizer", "{param:tokenizer_dir}", "--trust-remote-code", "false",
                        "--tokenizer-mode", "auto"]}


def spb(i):
    """S-PB frontend mode (as configs/e4 e4b-*), one process on one core."""
    args = ["--host", "127.0.0.1", "--port", "0", "--prebaked", "{input:prebaked}",
            "--lag-period-us", "{param:lag_period_us}",
            "--sfe-module", "{param:sfe_module}", "--sfe-profile", "trtllm-like",
            "--model", "{param:tokenizer_dir}",
            "--max-context-length", "{param:sfe_max_context_length}",
            {"splice_json": "{input:fingerprint}", "key": "sfe_args"}]
    return role("spb%d" % i, "capkit", "server", "stub_out_of_domain", True, "asyncio", True,
                None, CAPKIT_READY, STUB_SLOTS, "all", args, module="capstubs.prebaked_server",
                note="S-PB frontend mode for frontend%d only (E6 arm iii: one stub per "
                     "frontend); one process on one core" % i)


def frontend(i):
    """The E4-4b main-tier frontend (configs/e4), connected to spb<i> only."""
    stub = "spb%d" % i
    return role("frontend%d" % i, "product", "server", "under_test", True, "uvloop", True, "P0",
                [{"kind": "report_addr"},
                 {"kind": "http_get", "path": "/health", "expect_status": 200,
                  "timeout_s": 2.0}], [], "none",
                ["{param:tokenizer_dir}", "--host", "127.0.0.1", "--port", "0",
                 "--report_addr", "{own:report_addr}", "--openengine-endpoint",
                 "{addr:%s}" % stub, "--num_serve_frontends", "1",
                 "--num_postprocess_workers", "0",
                 "--num_input_processor_workers", str(FE_INPUT_WORKERS)],
                command=["{param:trtllm_serve}"], start_after=[stub], cpus_needed=FE_CPUS,
                proc_kinds=["frontend", "frontend_openengine", "frontend_single"],
                port={"kind": "report_addr", "host": "127.0.0.1"},
                note="E6 arm iii frontend %d: an independent single-frontend trtllm-serve "
                     "(card E4 main tier, %d cores = 1 event loop + %d input-processor threads "
                     "+ %d for grpcio's threads, unverified)" % (i, FE_CPUS, FE_INPUT_WORKERS,
                                                                 FE_GRPC_THREADS))


CAPLOAD_ARGS = ["--endpoint", "/v1/completions", "--schedule", "{input:schedule}",
                "--prompts", "{input:prompts}", "--conn-policy", "keepalive",
                "--pool-pick", "{param:capload_pool_pick}", "--pool-seed", "{param:capload_pool_seed}",
                "--pool-reset", "{param:capload_pool_reset}", "--recv-mode", "count",
                "--chunk-sample-mod", "{param:capload_chunk_sample_mod}",
                "--fionread-every-ms", "{param:capload_fionread_every_ms}",
                "--lag-period-us", "{param:lag_period_us}",
                "--max-records-per-worker", "{param:capload_max_records_per_worker}"]


def capload(i, workers):
    fe = "frontend%d" % i
    return role("capload%d" % i, "capkit", "client", "load_source", True, "asyncio", False, None,
                CAPKIT_READY, ["backlog.inflight"], "all",
                ["--target", "{addr:%s}" % fe, "--model", "{param:tokenizer_dir}"]
                + CAPLOAD_ARGS, start_after=[fe], cpus_needed=workers,
                workers=workers, expected=workers + 1, module="capload",
                note="capload for frontend%d only, %d workers (parent + %d); the full "
                     "registered schedule (tier load = load per frontend)" % (i, workers, workers))


# ---------------- scenario ----------------
def _params(obj, out):
    """Every {param:NAME} in the roles and job inputs: params_required is exactly this set."""
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


def req_chain(i):
    """Per-frontend request chain: capload<i> ok == spb<i> accepted streams (the pairs share
    nothing, so a request counted in one pair can never appear in another)."""
    return {"name": "requests%d" % i, "kind": "requests", "gate": None, "include_usage": None,
            "hops": [{"role": "capload%d" % i, "source": "summary.counts.ok", "scope": "tier"},
                     {"role": "spb%d" % i, "source": "summary.counts.streams_accepted",
                      "scope": "tier"}],
            "corrections": []}


def scen(name, n, roles, note):
    stop = (["capload%d" % i for i in range(n)] + ["frontend%d" % i for i in range(n)]
            + ["spb%d" % i for i in range(n)] + ["sidecar", "canary"])
    ji = {"fingerprint": FINGERPRINT, "prebaked": PREBAKED}
    return {"schema": "capkit.scenario/1", "experiment": EXP, "name": name,
            "tier_table": "%s/%s.tiers.json" % (REG, name),
            "schedule": "%s/%s.schedule.capbin" % (REG, name),
            "prompts": "%s/%s.prompts.json" % (REG, name),
            "probe_tier": "P0", "probe_groups": [], "stop_grace_s": 10.0, "stop_order": stop,
            "reconcile": {"schema": "capkit.reconcile/1",
                          "chains": [req_chain(i) for i in range(n)]},
            "roles": roles,
            "params_required": sorted(_params([roles, ji], set())),
            "dims": {"tokenizer": "TBD-by-registration (configs/tokenizers/GLM-5.3-Flash)",
                     "stub": "S-PB (capstubs.prebaked_server, frontend mode), one per frontend",
                     "response_timing_switches": "defaults (not set)",
                     "interpretation": "coverage round (REPLAN-20261116): exploratory, offline "
                                       "readout; rejudge with the formal rules later",
                     "log_sink": "raw/logs on the node-local disk (orchestrator redirect)"},
            "note": note + ". " + TEMPLATE_NOTE, "job_inputs": ji}


# ---------------- tier-table templates (same shapes and TBD names as configs/e4) -----------
def rd(name, terms):
    return {"name": name, "terms": [{"role": r, "slot": s, "sign": 1, "agg": "sum"}
                                    for r, s in terms], "abs_max": 0}


def gate(n):
    return {"mode": "enforce", "rule": "abs_max",
            "readings": [rd("loadgen_inflight", [("capload%d" % i, "backlog.inflight")
                                                 for i in range(n)]),
                         rd("spb_pending", [("spb%d" % i, "backlog.pending_msgs")
                                            for i in range(n)]),
                         rd("spb_active", [("spb%d" % i, "backlog.active_streams")
                                           for i in range(n)])],
            "poll_ns": "TBD:gate_poll_ns", "settle_polls": "TBD:gate_settle_polls",
            "deadline_guard_ns": "TBD:gate_deadline_guard_ns",
            "max_snapshot_age_ns": "TBD:gate_max_snapshot_age_ns"}


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
    "mc": "M axis, fixed C path (as configs/e4 e4b-mc): k = 1, finite streams of n_tokens, beat "
          "fixed, C doubling; C is PER FRONTEND (every capload role runs the whole schedule)",
    "r": "R axis (as configs/e4 e4b-r): max_tokens 1, rate doubling; the rate is PER FRONTEND "
         "(every capload role runs the whole schedule)",
}


def tpl(name, n, shape):
    axis, common, warm, up = SHAPES[shape]
    common = dict(common, tokens_per_msg=1, stream_interval=1, prompt_set=PROMPTS)
    tiers = [dict({"label": "warmup", "kind": "warmup", "ramp_ns": 0,
                   "window_ns": "TBD:warmup_ns", "tail_ns": 0}, **warm),
             {"label": "drain-01", "kind": "drain", "window_ns": "TBD:drain_max_wait_ns"}]
    for i in range(LADDER):
        tiers.append(dict({"label": "up-%02d" % (i + 1), "kind": "up", "ramp_ns": "TBD:ramp_ns",
                           "window_ns": "TBD:window_ns", "tail_ns": "TBD:tail_ns",
                           "baseline": i == 0}, **up(i)))
        tiers.append({"label": "drain-%02d" % (i + 2), "kind": "drain",
                      "window_ns": "TBD:drain_max_wait_ns"})
    return {"schema": "capkit.tiers_template/1", "experiment": EXP, "name": name,
            "arrival": {"dist": "fixed", "seed": "TBD:arrival_seed"}, "axis": axis,
            "request_timeout_ns": "TBD:request_timeout_ns", "default_gate": gate(n),
            "launch_inject": [], "common": common, "tiers": tiers,
            "note": "; ".join((SHAPE_NOTES[shape], "group load = %d x the tier value" % n,
                               TEMPLATE_NOTE))}


# ---------------- arms ----------------
def e6(n, shape):
    name = "e6-iii-n%d-%s" % (n, shape)
    wk = W_PER_FRONTEND[n]
    roles = []
    for i in range(n):
        roles += [spb(i), frontend(i), capload(i, wk)]
    roles += observers(["capload%d" % i for i in range(n)])
    note = ("E6 arm (iii) %s: %d independent copies of the E4-4b chain, capload<i> (%d "
            "workers) -> trtllm-serve new-arch frontend<i> (1 frontend, postproc pool off; under "
            "test, %d cores) -> spb<i> (S-PB frontend mode, one core); no shared stub, no "
            "multi-frontend mechanism, explicit per-frontend targets (no connection landing); "
            "%d cores in all" % (name, n, wk, FE_CPUS, n * (FE_CPUS + 1 + wk) + 2))
    w(name + ".scenario.json", scen(name, n, roles, note))
    t = tpl(name, n, shape)
    w(name + ".tiers.template.json", t)
    names = set()

    def walk(o):
        if isinstance(o, str) and o.startswith("TBD:"):
            names.add(o[4:])
        elif isinstance(o, dict):
            [walk(v) for v in o.values()]
        elif isinstance(o, list):
            [walk(v) for v in o]
    walk(t)
    w(name + ".values.todo.json",
      {"schema": "capkit.tier_values/1", "values": {k: None for k in sorted(names)}})


for _n in NS:
    for _shape in ("mc", "r"):
        e6(_n, _shape)

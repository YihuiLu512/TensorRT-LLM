# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generator of configs/snx: scenario TEMPLATES for the S-NX experiments E2b, E7, E8 and E9
(REPLAN-20261116 section 2; design cards E7, E8, E9, section 7 and 12.5(a)). The JSON files are the
deliverable; rerun into a NEW dir and diff.

Every arm is one scenario + one tier-table template + one values checklist, as configs/e2a..e4.
Tier values (rates, stream counts, beat periods, output lengths, gates) are "TBD:<name>", filled
at registration; scenario-level constants (process counts, core counts, K, n, w) are fixed here.

Topologies (S-NX = the product's rank 0 worker with its PyExecutor replaced by a NullExecutor):
  E2b  nx_role standalone: one NullExecutor, arrivals and self-drain on its step thread (no IPC)
       nx_role proxy: product proxy <-> ZMQ <-> NullWorker, raw drain (E2b part 2)
  E7   nx_role proxy: product proxy <-> ZMQ <-> NullWorker; drain raw | deserialize; K lanes
  E8   capgrpc.load (K connections) -> serve_nx host (trtllm-serve --grpc --grpc-protocol
       openengine) <-> ZMQ <-> NullWorker
  E9   O-NX: capload -> serve_nx original architecture (n frontends, postproc pool w) <-> ZMQ <->
       NullWorker; N-NX: capload -> new-architecture frontend (n frontends, pool w) -> gRPC ->
       serve_nx host <-> ZMQ <-> NullWorker
The S-NX roles take the tier table and the launch origin themselves (serve_nx --tier-table
{input:tier_table} --origin-ns {origin_ns}); the NullExecutor's step grid is the tier table's beat
schedule, its tokens per step and stream interval come from the tier (tier_shape).

Usage: python configs/gen_snx_configs.py <new output dir>   (writes <dir>/snx)
"""
import json, os, sys
OUT = sys.argv[1]
SUB = "snx"
PB2 = "tensorrt_llm.grpc.openengine._generated.generation_pb2"
GENERATE = "/openengine.v1.Inference/Generate"
LADDER = 8         # up tiers of a doubling ladder (coverage round: one launch per point)
K_E7 = 2           # E7 active drain lanes (card E7: K is a factor, start at K = 2; K = 1 apart)
K_E8 = 4           # E8 capgrpc.load connections (K frontends' worth of connections)
W_HTTP = 4         # capload workers (E4 value); n = 8 arms double it
FE_INPUT_WORKERS = 8
TEMPLATE_NOTE = ("TEMPLATE: every TBD value is filled at registration (REPLAN-20261116 R1 "
                 "coverage round: one launch per point, exploratory)")
REG = {"E2b": "05_engine/E2b/register", "E7": "05_engine/E7/register",
       "E8": "05_engine/E8/register", "E9": "05_engine/E9/register"}


def w(name, doc):
    d = os.path.join(OUT, SUB)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    assert not os.path.exists(p), p
    with open(p, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=False); f.write("\n")


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


CAPKIT_READY = [{"kind": "capkit_ready"}]


def observers(after):
    return [
        role("sidecar", "capkit", "aux", "observer", False, None, False, None, CAPKIT_READY, [],
             "none", ["--interval-ms", "{param:sidecar_interval_ms}", "--gpu-every-s",
                      "{param:sidecar_gpu_every_s}", "--numa-every-s",
                      "{param:sidecar_numa_every_s}", "--nvidia-smi", "{nvidia_smi}"],
             start_after=[after], module="capenv.sidecar"),
        role("canary", "capkit", "aux", "observer", False, None, False, None, CAPKIT_READY, [],
             "none", ["--interval-ms", "{param:canary_interval_ms}"], module="capenv.canary")]


# S-NX knobs shared by every arm (SnxConfig keys; the rest is derived from the tier table)
SNX_SET = ["--set", "busy_ns_per_step={param:snx_busy_ns_per_step}",
           "--set", "first_response_delay_steps={param:snx_first_response_delay_steps}",
           "--set", "idle_restart=\"grid\""]
NX_SLOTS = ["backlog.inflight", "backlog.engine_open", "backlog.pending_keys"]


# ---------------- S-NX roles ----------------
def nx(mode, consumer="raw", lanes=1):
    """capstubs.nx_role (capkit tool, kind client: it sends the schedule and exits after the
    table end). standalone: 2 cores (main thread with the loop; the step thread). proxy: the
    process plus the product's MPI worker (main thread, await_response_thread, step thread) and
    K drain subprocesses, all inheriting the role's core set: 4 + K cores (finer per-thread pinning
    is a known gap; the sidecar reports per-thread CPU)."""
    args = ["--mode", mode, "--schedule", "{input:schedule}", "--nxd-dir", "{raw}/nxd",
            "--snx-out-dir", "{raw}/snx", "--lag-period-us", "{param:lag_period_us}",
            "--prompt-len", "{param:nx_prompt_len}", "--tail-s", "{param:nx_tail_s}"] + SNX_SET
    if mode == "proxy":
        args += ["--model", "{param:tokenizer_dir}", "--max-seq-len", "{param:nx_max_seq_len}",
                 "--consumer", consumer, "--lanes", str(lanes)]
        cpus = 4 + (lanes if lanes >= 2 else 0)
        note = ("nx_role proxy: product proxy <-> ZMQ <-> NullWorker (rank 0), %s drain, %d "
                "active lane(s)%s; %d cores" % (consumer, lanes, " (launcher lane 0 only "
                "starts the worker; ipc PUSH/PULL)" if lanes >= 2 else " (tcp PAIR)", cpus))
    else:
        cpus = 2
        note = "nx_role standalone: NullExecutor in process, self-drained, no IPC; 2 cores"
    return role("nx", "capkit", "client", "under_test", True, "asyncio", False, None,
                CAPKIT_READY, NX_SLOTS, "all", args, cpus_needed=cpus, module="capstubs.nx_role",
                note=note)


def serve_nx(name, serve_args, start_after=(), cpus=4, readiness=None, proc_kinds=(),
             note=""):
    """python -m capstubs.serve_nx (product role): trtllm-serve with the rank 0 worker's
    PyExecutor replaced by a NullExecutor; it picks its own port and reports host:port once the
    port accepts (trtllm-serve's --report_addr refuses multi-frontend and gRPC serving).
    --gpus_per_node 1: trtllm-serve otherwise counts the node's GPUs (and refuses none)."""
    args = ["--tier-table", "{input:tier_table}", "--origin-ns", "{origin_ns}",
            "--out-dir", "{raw}/snx", "--record", "{raw}/snx/%s.serve_nx.json" % name,
            "--stream-interval", "{param:nx_stream_interval}",
            "--pick-port", "--report-host", "127.0.0.1", "--report-addr", "{own:report_addr}"] \
        + SNX_SET + ["--", "{param:tokenizer_dir}", "--host", "127.0.0.1",
                     "--max_seq_len", "{param:nx_max_seq_len}", "--gpus_per_node", "1"] \
        + list(serve_args)
    return role(name, "product", "server", "under_test", True, "uvloop", True, "P0",
                readiness or [{"kind": "report_addr"}], [], "none", args,
                command=["{param:python}", "-m", "capstubs.serve_nx"],
                start_after=start_after, cpus_needed=cpus, proc_kinds=proc_kinds,
                port={"kind": "report_addr", "host": "127.0.0.1"}, note=note)


HTTP_READY = [{"kind": "report_addr"},
              {"kind": "http_get", "path": "/health", "expect_status": 200, "timeout_s": 2.0}]


def host(start_after=()):
    """E8 / N-NX engine host: trtllm-serve <tok> --grpc --grpc-protocol openengine (one gRPC
    process, the product default) with S-NX."""
    return serve_nx("host", ["--grpc", "--grpc-protocol", "openengine"], start_after, cpus=4,
                    note="serve_nx host (--grpc --grpc-protocol openengine; 1 gRPC process): "
                         "event loop + dispatch thread; the MPI worker (rank 0, S-NX) inherits "
                         "the core set; 4 cores")


def fe_cpus(n, pool):
    return n * (1 + FE_INPUT_WORKERS + 2) + pool


def onx(n, pool):
    """E9 O-NX: the original architecture trtllm-serve <tok> with n frontends and a postproc pool
    of w, S-NX in its worker."""
    kinds = ["frontend"]
    return serve_nx("onx", ["--num_serve_frontends", str(n), "--num_postprocess_workers",
                            str(pool), "--num_input_processor_workers",
                            str(FE_INPUT_WORKERS)], cpus=fe_cpus(n, pool) + 3, readiness=HTTP_READY,
                    proc_kinds=kinds,
                    note="O-NX: original-architecture trtllm-serve, %d frontend(s), postproc "
                         "pool %d, S-NX worker; cores: per frontend 1 loop + %d input-processor "
                         "threads + 2, + pool, + 3 for the worker" % (n, pool, FE_INPUT_WORKERS))


def nnx_frontend(n, pool):
    """E9 N-NX: the new-architecture frontend in front of the host, started through the same
    entry script as O-NX (serve_nx: its port pick and address report work for n > 1, where
    trtllm-serve refuses --port 0 / --report_addr; the frontend runs no executor, so the S-NX
    patch has nothing to act on)."""
    kinds = ["frontend", "frontend_openengine"] + (["frontend_single"] if n == 1 else
                                                   ["frontend_launcher", "frontend_attached"])
    return serve_nx("frontend", ["--openengine-endpoint", "{addr:host}",
                                 "--num_serve_frontends", str(n),
                                 "--num_postprocess_workers", str(pool),
                                 "--num_input_processor_workers", str(FE_INPUT_WORKERS)],
                    start_after=["host"], cpus=fe_cpus(n, pool), readiness=HTTP_READY,
                    proc_kinds=kinds,
                    note="N-NX frontend: %d frontend(s), postproc pool %d per frontend; cores: "
                         "per frontend 1 loop + %d input-processor threads + 2, + pool"
                         % (n, pool, FE_INPUT_WORKERS))


# ---------------- load sources ----------------
READER_SLOTS = ["backlog.inflight", "backlog.active_streams", "backlog.read"]


def req_bytes():
    """OpenEngine request bytes for the host; model = the tokenizer dir (the host's served
    model name: trtllm-serve's default is the model path it was given)."""
    return {"args": ["openengine", "--pb2-module", PB2, "--prompts", "{input:prompts}",
                     "--tier-table", "{input:tier_table}", "--model", "{param:tokenizer_dir}"]}


def grpc_load(target, procs):
    """capgrpc.load (E8): K worker processes, one channel each (K connections = K frontends)."""
    args = ["--target", "{addr:%s}" % target, "--method", GENERATE,
            "--channels-per-worker", "1", "--request-format", "openengine",
            "--request-bytes", "{input:request_bytes}"] \
        + (["--in-process"] if procs == 1 else []) \
        + ["--msg-sample-mod", "{param:msg_sample_mod}", "--schedule", "{input:schedule}",
           "--lag-period-us", "{param:lag_period_us}",
           "--max-records-per-worker", "{param:reader_max_records_per_worker}",
           "--max-msg-records-per-worker", "{param:reader_max_msg_records_per_worker}"]
    return role("reader", "capkit", "client", "load_source", True, "asyncio", True, None,
                CAPKIT_READY, READER_SLOTS, "all", args, start_after=[target],
                cpus_needed=procs, workers=procs, expected=1 if procs == 1 else procs + 1,
                module="capgrpc.load",
                note="capgrpc.load, %d connection(s) (one channel per worker)" % procs)


CAPLOAD_ARGS = ["--endpoint", "/v1/completions", "--schedule", "{input:schedule}",
                "--prompts", "{input:prompts}", "--conn-policy", "keepalive",
                "--pool-pick", "{param:capload_pool_pick}", "--pool-seed",
                "{param:capload_pool_seed}", "--pool-reset", "{param:capload_pool_reset}",
                "--recv-mode", "count", "--chunk-sample-mod", "{param:capload_chunk_sample_mod}",
                "--fionread-every-ms", "{param:capload_fionread_every_ms}",
                "--lag-period-us", "{param:lag_period_us}",
                "--max-records-per-worker", "{param:capload_max_records_per_worker}"]


def capload(target, workers):
    return role("capload", "capkit", "client", "load_source", True, "asyncio", False, None,
                CAPKIT_READY, ["backlog.inflight"], "all",
                ["--target", "{addr:%s}" % target, "--model", "{param:tokenizer_dir}"]
                + CAPLOAD_ARGS, start_after=[target], cpus_needed=workers, workers=workers,
                expected=workers + 1, module="capload",
                note="capload, %d workers (parent + %d); --model = the served model name (the "
                     "tokenizer dir)" % (workers, workers))


# ---------------- scenario ----------------
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


def req_chain(*hops):
    """hops: (role, source, scope); scope launch = recorded, not checked per tier (the S-NX
    self-report is read offline: capstubs.null_executor summarize)."""
    return {"name": "requests", "kind": "requests", "gate": None, "include_usage": None,
            "hops": [{"role": r, "source": s, "scope": sc} for r, s, sc in hops],
            "corrections": []}


def scen(exp, name, roles, stop_order, chains, note, job_inputs=None):
    reg = REG[exp]
    tiers = {r["probe_tier"] for r in roles if r["launcher"] == "product"}
    assert len(tiers) <= 1, tiers
    d = {"schema": "capkit.scenario/1", "experiment": exp, "name": name,
         "tier_table": "%s/%s.tiers.json" % (reg, name),
         "schedule": "%s/%s.schedule.capbin" % (reg, name),
         "prompts": "%s/%s.prompts.json" % (reg, name),
         "probe_tier": tiers.pop() if tiers else None,
         "probe_groups": [], "stop_grace_s": 10.0, "stop_order": stop_order,
         "reconcile": {"schema": "capkit.reconcile/1", "chains": chains}, "roles": roles,
         "params_required": sorted(_params([roles, job_inputs or {}], set())),
         "dims": {"tokenizer": "TBD-by-registration (configs/tokenizers/GLM-5.3-Flash)",
                  "stub": "S-NX (capstubs NullExecutor in the product rank 0 worker; "
                          "capkit.snx_config/1 from the tier table + SNX_SET params)",
                  "response_timing_switches": "S-NX: first_response_delay_steps / "
                                              "busy_ns_per_step params; the rest defaults",
                  "interpretation": "REPLAN-20261116 R1 coverage round: exploratory, one "
                                    "launch per point",
                  "log_sink": "raw/logs on the node-local disk (orchestrator redirect)"},
         "note": note + ". " + TEMPLATE_NOTE}
    if job_inputs:
        d["job_inputs"] = job_inputs
    return d


def gate_t(readings):
    return {"mode": "enforce", "rule": "abs_max", "readings": readings,
            "poll_ns": "TBD:gate_poll_ns", "settle_polls": "TBD:gate_settle_polls",
            "deadline_guard_ns": "TBD:gate_deadline_guard_ns",
            "max_snapshot_age_ns": "TBD:gate_max_snapshot_age_ns"}


def rd(name, *terms):
    return {"name": name, "terms": [{"role": r, "slot": s, "sign": g, "agg": "sum"}
                                    for r, s, g in terms], "abs_max": 0}


NX_GATE = gate_t([rd("nx_inflight", ("nx", "backlog.inflight", 1)),
                  rd("nx_engine_open", ("nx", "backlog.engine_open", 1)),
                  rd("nx_pending_keys", ("nx", "backlog.pending_keys", 1))])
READER_GATE = gate_t([rd("reader_inflight", ("reader", "backlog.inflight", 1)),
                      rd("reader_active", ("reader", "backlog.active_streams", 1))])
CAPLOAD_GATE = gate_t([rd("loadgen_inflight", ("capload", "backlog.inflight", 1))])

M_AXIS = {"name": "M", "load_key": "nominal_msgs_per_s", "unit": "msg/s", "order": "asc"}
R_AXIS = {"name": "R", "load_key": "rate_rps", "unit": "req/s", "order": "asc"}
SHAPES = {
    "mc": (M_AXIS, {"max_tokens": "TBD:n_tokens", "beat_ns": "TBD:beat_ns"},
           {"streams_c": "TBD:c_warmup", "rate_rps": "from_streams_c"},
           lambda i: {"streams_c": "TBD:c_%d" % i, "rate_rps": "from_streams_c"}),
    "md": (M_AXIS, {"max_tokens": "TBD:n_tokens", "streams_c": "TBD:c_fixed"},
           {"beat_ns": "TBD:beat_ns_warmup", "rate_rps": "from_streams_c"},
           lambda i: {"beat_ns": "TBD:beat_ns_%d" % i, "rate_rps": "from_streams_c"}),
    "r": (R_AXIS, {"max_tokens": 1, "beat_ns": "TBD:beat_ns"},
          {"rate_rps": "TBD:rate_warmup"},
          lambda i: {"rate_rps": "TBD:rate_%d" % i}),
}
SHAPE_NOTES = {
    "mc": "M axis, fixed beat (the S-NX step period), C doubling: active requests per step "
          "grow (card E7 'fixed delta, scan active requests')",
    "md": "M axis, fixed C, beat halving: steps get denser (card E7 'fixed active, scan delta'); "
          "every tier's duration, the drain after it included, must be a multiple of its beat",
    "r": "R axis: shortest output (max_tokens 1), rate doubling",
}


def tpl(exp, name, shape, gate, s="TBD:stream_interval", prompt_set="TBD:prompt_set",
        n=LADDER, note=""):
    axis, common, warm, up = SHAPES[shape]
    common = dict(common, tokens_per_msg=1, stream_interval=s, prompt_set=prompt_set)
    tiers = [dict({"label": "warmup", "kind": "warmup", "ramp_ns": 0,
                   "window_ns": "TBD:warmup_ns", "tail_ns": 0}, **warm),
             {"label": "drain-01", "kind": "drain", "window_ns": "TBD:drain_max_wait_ns"}]
    for i in range(n):
        tiers.append(dict({"label": "up-%02d" % (i + 1), "kind": "up", "ramp_ns": "TBD:ramp_ns",
                           "window_ns": "TBD:window_ns", "tail_ns": "TBD:tail_ns",
                           "baseline": i == 0}, **up(i)))
        tiers.append({"label": "drain-%02d" % (i + 2), "kind": "drain",
                      "window_ns": "TBD:drain_max_wait_ns"})
    return {"schema": "capkit.tiers_template/1", "experiment": exp, "name": name,
            "arrival": {"dist": "fixed", "seed": "TBD:arrival_seed"}, "axis": axis,
            "request_timeout_ns": "TBD:request_timeout_ns", "default_gate": gate,
            "launch_inject": [], "common": common, "tiers": tiers,
            "note": "; ".join(x for x in (SHAPE_NOTES[shape], note, TEMPLATE_NOTE) if x)}


def arm(exp, name, scenario, template):
    w(name + ".scenario.json", scenario)
    w(name + ".tiers.template.json", template)


# ================= E2b (S-NX calibration; card E2, part E2b) =================
def e2b(mode, shape):
    name = "e2b-%s-%s" % ("sa" if mode == "standalone" else "px", shape)
    roles = [nx(mode)] + observers("nx")
    what = ("standalone: no IPC, no consumer, no other thread on the lock" if mode ==
            "standalone" else "behind the product proxy (tcp PAIR, raw drain)")
    arm("E2b", name,
        scen("E2b", name, roles, ["nx", "sidecar", "canary"],
             [req_chain(("nx", "summary.counts.ok", "tier"))],
             "E2b %s: S-NX calibration, %s; the per-step thread CPU table comes from the "
             "NullExecutor self-report (snx.*.steps.bin, summarize)" % (name, what)),
        tpl("E2b", name, shape, NX_GATE))


# ================= E7 (rank 0 worker non-GPU path) =================
def e7(consumer, lanes, shape):
    name = "e7-%s-k%d-%s" % ({"raw": "raw", "deserialize": "deser"}[consumer], lanes, shape)
    roles = [nx("proxy", consumer, lanes)] + observers("nx")
    arm("E7", name,
        scen("E7", name, roles, ["nx", "sidecar", "canary"],
             [req_chain(("nx", "summary.counts.ok", "tier"))],
             "E7 %s: nx_role proxy -> ZMQ -> rank 0 NullWorker, %s drain, K = %d%s"
             % (name, consumer, lanes, "; isolation check (b): the deserializing drain"
                if consumer == "deserialize" else "")),
        tpl("E7", name, shape, NX_GATE))


# ================= E8 (engine host) =================
def e8(procs, shape):
    name = "e8-k%d-%s" % (procs, shape)
    roles = [host(), grpc_load("host", procs)] + observers("reader")
    arm("E8", name,
        scen("E8", name, roles, ["reader", "host", "sidecar", "canary"],
             [req_chain(("reader", "summary.counts.ok", "tier"))],
             "E8 %s: capgrpc.load (%d connection(s)) -> serve_nx host (trtllm-serve --grpc "
             "--grpc-protocol openengine, S-NX worker)" % (name, procs),
             job_inputs={"request_bytes": req_bytes()}),
        tpl("E8", name, shape, READER_GATE))


# ================= E9 (old / new whole chain, same NullExecutor) =================
def e9(arch, n, pool, shape):
    name = "e9-%s%df%s-%s" % (arch.lower(), n, "P" if pool else "0", shape)
    workers = W_HTTP * (2 if n >= 8 else 1)
    if arch == "O":
        roles = [onx(n, pool), capload("onx", workers)] + observers("capload")
        stop = ["capload", "onx", "sidecar", "canary"]
        what = "O-NX: capload -> original-architecture trtllm-serve (S-NX worker)"
    else:
        roles = [host(), nnx_frontend(n, pool), capload("frontend", workers)] + \
            observers("capload")
        stop = ["capload", "frontend", "host", "sidecar", "canary"]
        what = "N-NX: capload -> new-architecture frontend -> gRPC -> serve_nx host (S-NX)"
    arm("E9", name,
        scen("E9", name, roles, stop, [req_chain(("capload", "summary.counts.ok", "tier"))],
             "E9 %s: %s; n = %d frontend(s), postproc pool %d; capload %d workers"
             % (name, what, n, pool, workers)),
        tpl("E9", name, shape, CAPLOAD_GATE,
            note="same NullExecutor parameters and arrival schedule in both arms; stream "
                 "interval s is a registered value (nx_stream_interval must equal it)"))


for _shape in ("mc", "md", "r"):
    e2b("standalone", _shape)
e2b("proxy", "mc")
for _shape in ("mc", "md", "r"):
    e7("raw", K_E7, _shape)
e7("raw", 1, "mc")
e7("deserialize", K_E7, "mc")
for _shape in ("mc", "md", "r"):
    e8(K_E8, _shape)
e8(1, "mc")
for _arch in ("O", "N"):
    for _n in (1, 8):
        for _shape in ("mc", "r"):
            e9(_arch, _n, 0, _shape)
    e9(_arch, 8, 4, "mc")


# ---------------- values checklists (fill, then python -m caplaunch.tiergen) ----------------
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

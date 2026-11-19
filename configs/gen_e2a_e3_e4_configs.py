# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generator of configs/e2a, configs/e3 and configs/e4: scenario TEMPLATES for the fast-track
order of E2a (S-PB, S-NL), E3-3a / 3b and E4-4b / 4b' (FASTTRACK-1117-PLAN v2 section 1, design
cards E2, E3, E4). The JSON files are the deliverable; rerun into a NEW dir and diff.

Every arm is one scenario + one tier-table template + one values checklist. Tier values (rates,
stream counts, beat periods, output lengths, deltas, windows, gates) are "TBD:<name>" and are
filled at registration time from the E1 / E2a results; the scenario-level constants below
(process counts, core counts) are fixed here and change only by regenerating.

Usage: python configs/gen_e2a_e3_e4_configs.py <new output dir>   (writes <dir>/{e2a,e3,e4})
"""
import json, os, sys
OUT = sys.argv[1]
PB2 = "tensorrt_llm.grpc.openengine._generated.generation_pb2"
GENERATE = "/openengine.v1.Inference/Generate"  # the product InferenceStub's Generate path
K_CONN = 4         # capgrpc.load processes of the multi-connection shape (one channel each)
W_HTTP = 4         # capload worker processes in E4
LADDER = 8         # up tiers of an open-loop ladder (doubling; bisection is a later launch)
# E4 main tier (card E4 "pinning"): the frontend gets no fewer cores than its runnable threads:
# its event loop, the input-processor pool (--num_input_processor_workers, written explicitly at
# the product default) and grpcio's own threads (count unverified; FT1 errata).
FE_INPUT_WORKERS = 8
FE_GRPC_THREADS = 2
FE_CPUS = 1 + FE_INPUT_WORKERS + FE_GRPC_THREADS
TEMPLATE_NOTE = ("TEMPLATE: every TBD value is filled at registration from the E1 / E2a "
                 "results (fast track, exploratory)")


def w(sub, name, doc):
    d = os.path.join(OUT, sub)
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


OBS_PARAMS = ["sidecar_interval_ms", "sidecar_gpu_every_s", "sidecar_numa_every_s",
              "canary_interval_ms"]
STUB_SLOTS = ["backlog.pending_msgs", "backlog.active_streams", "backlog.yielded"]
READER_SLOTS = ["backlog.inflight", "backlog.active_streams", "backlog.read"]

# ---------------- job inputs (capcommon.jobinputs, built in the job after the gate) ----------
def req_bytes(model):
    """OpenEngine request bytes (capgrpc.reqbytes openengine); model: the {param:...} the
    stub's --model uses too (any non-empty name is served; one name keeps them equal)."""
    return {"args": ["openengine", "--pb2-module", PB2, "--prompts", "{input:prompts}",
                     "--tier-table", "{input:tier_table}", "--model", model]}
PREBAKED = {"args": ["--pb2-module", PB2, "--tier-table", "{input:tier_table}",
                     "--token-ids", "{param:spb_token_ids}",
                     "--prompt-tokens", "{param:spb_prompt_tokens}"]}
# The frontend arms' tokenizer settings are fixed to what the frontend role's argv implies (no
# --trust_remote_code: false; no tokenizer_mode in a --config: auto), as in configs/e0 FP_INPUT;
# the S-FE fingerprint (S-PB frontend mode) and S-NL's own tokenizer load use the same three.
TOKENIZER_ARGS = ["--tokenizer", "{param:tokenizer_dir}", "--trust-remote-code", "false",
                  "--tokenizer-mode", "auto"]
FINGERPRINT = {"args": ["--tokenizer", "{param:tokenizer_dir}", "--trust-remote-code", "false",
                        "--tokenizer-mode", "auto"]}


# ---------------- stubs ----------------
def spb(mclass, frontend=False):
    """S-PB (capstubs.prebaked_server), one process on one core. Direct (a gRPC reader): with
    --schedule, so request_id must be the schedule's req_id and streams are filed by planned
    tier. Frontend mode (E4): Control + Health from the S-FE module, no --schedule (the
    frontend's request ids are not schedule ids; streams are filed by arrival time)."""
    args = ["--host", "127.0.0.1", "--port", "0", "--prebaked", "{input:prebaked}",
            "--lag-period-us", "{param:lag_period_us}"]
    if frontend:
        args += ["--sfe-module", "{param:sfe_module}", "--sfe-profile", "trtllm-like",
                 "--model", "{param:tokenizer_dir}",
                 "--max-context-length", "{param:sfe_max_context_length}",
                 {"splice_json": "{input:fingerprint}", "key": "sfe_args"}]
        note = ("S-PB frontend mode (as configs/e0s e0s-spb-frontend, --model = the tokenizer "
                "dir = the frontend's served model name); one process on one core")
    else:
        args += ["--schedule", "{input:schedule}"]
        note = "S-PB direct mode with --schedule; one process on one core"
    return role("spb", "capkit", "server", mclass, True, "asyncio", True, None, CAPKIT_READY,
                STUB_SLOTS, "all", args, module="capstubs.prebaked_server", note=note)


def snl(mclass, mode="full", frontend=False):
    """S-NL (capstubs.null_llm_server), one process on one core; the product server module is the
    default (tensorrt_llm.grpc.openengine.server). Direct: --schedule (request_id = req_id),
    --model = the request bytes' model. Frontend (E4 4b'): no --schedule, the product loads the
    tokenizer with the frontend's settings, --model = the tokenizer dir."""
    args = ["--host", "127.0.0.1", "--port", "0", "--mode", mode]
    if frontend:
        args += ["--model", "{param:tokenizer_dir}"]
    else:
        args += ["--model", "{param:model_name}"]
    args += ["--token-source", "fixed", "--token-ids", "{param:snl_token_ids}",
             "--lag-period-us", "{param:lag_period_us}"]
    if frontend:
        args += TOKENIZER_ARGS
        note = "S-NL full behind the frontend (no --schedule); one process on one core"
    else:
        args += ["--schedule", "{input:schedule}"]
        note = ("S-NL %s with --schedule; one process on one core%s"
                % (mode, "; no servicer, no gRPC traffic (the E3-3a metronome-only dry run)"
                   if mode == "metronome-only" else ""))
    return role("snl", "capkit", "server", mclass, True, "asyncio", True, None, CAPKIT_READY,
                STUB_SLOTS, "all", args, module="capstubs.null_llm_server", note=note)


STUB_MSGS = {"spb": "summary.counts.msgs_sent", "snl": "summary.counts.msgs_yielded"}

# ---------------- gRPC readers (role name "reader" in every arm) ----------------
def reader_args(target, procs):
    """The reader argv capgrpc.load and capgrpc.oe_sink share (capgrpc.loader.add_reader_args):
    one channel per process, OpenEngine request bytes, --planned-msgs (S-PB / S-NL messages
    carry no step stamp, errata FT1-9). One process runs in-process (no supervisor)."""
    return (["--target-port-file", "{port_file:%s}" % target, "--method", GENERATE,
             "--channels-per-worker", "1", "--request-format", "openengine",
             "--request-bytes", "{input:request_bytes}"]
            + (["--in-process"] if procs == 1 else [])
            + ["--msg-sample-mod", "{param:msg_sample_mod}", "--schedule", "{input:schedule}",
               "--lag-period-us", "{param:lag_period_us}",
               "--max-records-per-worker", "{param:reader_max_records_per_worker}",
               "--planned-msgs",
               "--max-msg-records-per-worker", "{param:reader_max_msg_records_per_worker}"])


def grpc_load(target, procs, mclass="load_source"):
    """capgrpc.load: procs = 1 -> the single-connection shape (one process, one channel);
    procs = K -> the multi-connection shape (K worker processes under a parent, one channel
    each, K connections)."""
    single = procs == 1
    return role("reader", "capkit", "client", mclass, True, "asyncio", True, None, CAPKIT_READY,
                READER_SLOTS, "all", reader_args(target, procs), start_after=[target],
                cpus_needed=procs, workers=procs, expected=1 if single else procs + 1,
                module="capgrpc.load",
                note=("capgrpc.load single-connection shape: one process, one channel"
                      if single else "capgrpc.load multi-connection shape: %d processes "
                      "(parent + %d workers), one channel each" % (procs, procs)))


def oe_sink(target, tier):
    """capgrpc.oe_sink (E3-3b): the product client classes, one process, one channel; --connect
    channel (S-PB direct mode answers no Control). Same reader argv as the weak reader
    (capgrpc.load single-connection shape) plus --tier / --connect."""
    return role("reader", "capkit", "client", "under_test", True, "asyncio", True, None,
                CAPKIT_READY, READER_SLOTS, "all",
                reader_args(target, 1) + ["--tier", tier, "--connect", "channel"],
                start_after=[target], module="capgrpc.oe_sink",
                note="capgrpc.oe_sink --tier %s --connect channel: the product client's read "
                     "path is under test" % tier)


# ---------------- HTTP side (E4) ----------------
CAPLOAD_ARGS = ["--endpoint", "/v1/completions", "--schedule", "{input:schedule}",
                "--prompts", "{input:prompts}", "--conn-policy", "keepalive",
                "--pool-pick", "{param:capload_pool_pick}", "--pool-seed", "{param:capload_pool_seed}",
                "--pool-reset", "{param:capload_pool_reset}", "--recv-mode", "count",
                "--chunk-sample-mod", "{param:capload_chunk_sample_mod}",
                "--fionread-every-ms", "{param:capload_fionread_every_ms}",
                "--lag-period-us", "{param:lag_period_us}",
                "--max-records-per-worker", "{param:capload_max_records_per_worker}"]
CAPLOAD_PARAMS = ["capload_pool_pick", "capload_pool_seed", "capload_pool_reset",
                  "capload_chunk_sample_mod", "capload_fionread_every_ms", "lag_period_us",
                  "capload_max_records_per_worker"]


def capload(workers):
    return role("capload", "capkit", "client", "load_source", True, "asyncio", False, None,
                CAPKIT_READY, ["backlog.inflight"], "all",
                ["--target", "{addr:frontend}", "--model", "{param:tokenizer_dir}"]
                + CAPLOAD_ARGS, start_after=["frontend"], cpus_needed=workers,
                workers=workers, expected=workers + 1, module="capload",
                note="capload, %d workers (parent + %d); --model = the frontend's served model "
                     "name (the tokenizer dir it was given)" % (workers, workers))


def frontend(stub, probe_tier="P0"):
    """The real new-arch trtllm-serve frontend as E0 launched it (configs/e0 e0a-frontend-sfe):
    one frontend, postproc pool off; plus --num_input_processor_workers at its default, written
    so the pool size is an explicit condition (card E4). probe_tier P1 (the probe patch, no
    group) only for the product-side V arms; capprobe then writes snapshots (errata FT1-18)."""
    return role("frontend", "product", "server", "under_test", True, "uvloop", True, probe_tier,
                [{"kind": "report_addr"},
                 {"kind": "http_get", "path": "/health", "expect_status": 200,
                  "timeout_s": 2.0}], [], "none" if probe_tier == "P0" else "all",
                ["{param:tokenizer_dir}", "--host", "127.0.0.1", "--port", "0",
                 "--report_addr", "{own:report_addr}", "--openengine-endpoint",
                 "{addr:%s}" % stub, "--num_serve_frontends", "1",
                 "--num_postprocess_workers", "0",
                 "--num_input_processor_workers", str(FE_INPUT_WORKERS)],
                command=["{param:trtllm_serve}"], start_after=[stub], cpus_needed=FE_CPUS,
                proc_kinds=["frontend", "frontend_openengine", "frontend_single"],
                port={"kind": "report_addr", "host": "127.0.0.1"},
                note="card E4 main tier: %d cores = 1 event loop + %d input-processor threads + "
                     "%d for grpcio's threads (unverified)" % (FE_CPUS, FE_INPUT_WORKERS,
                                                               FE_GRPC_THREADS))

# ---------------- scenario ----------------
REG = {"E2a": "03_calib/E2a/register", "E3": "04_frontend/E3/register",
       "E4": "04_frontend/E4/register"}  # planned registration dirs (DATA-AND-ANALYSIS 1)
SUB = {"E2a": "e2a", "E3": "e3", "E4": "e4"}


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


def req_chain(*hops):
    return {"name": "requests", "kind": "requests", "gate": None, "include_usage": None,
            "hops": [{"role": r, "source": s, "scope": "tier"} for r, s in hops],
            "corrections": []}


def msg_chain(stub):
    """Stub messages (incl. finished) vs the reader's msgs.msgs (incl. finished): the same chain
    as the FT1-sink job-path e2e (gate product + one finished message per request per hop)."""
    return {"name": "messages", "kind": "messages", "gate": "product", "include_usage": None,
            "hops": [{"role": stub, "source": STUB_MSGS[stub], "scope": "tier"},
                     {"role": "reader", "source": "summary.msgs.msgs", "scope": "tier"}],
            "corrections": [{"name": "finished_%s" % stub, "per_request": 1, "hop": stub},
                            {"name": "finished_reader", "per_request": 1, "hop": "reader"}]}


def scen(exp, name, roles, stop_order, chains, stub_desc, note, job_inputs=None, tok=None):
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
         "dims": {"tokenizer": tok, "stub": stub_desc,
                  "response_timing_switches": "defaults (not set)",
                  "interpretation": "fast track (FASTTRACK-1117-PLAN v2): exploratory, judged "
                                    "with the E0 round-1 thresholds; rejudge or rerun after E0 v2",
                  "log_sink": "raw/logs on the node-local disk (orchestrator redirect)"},
         "note": note + ". " + TEMPLATE_NOTE}
    if job_inputs:
        d["job_inputs"] = job_inputs
    return d


# ---------------- tier-table templates ----------------
def gate_t(readings):
    return {"mode": "enforce", "rule": "abs_max", "readings": readings,
            "poll_ns": "TBD:gate_poll_ns", "settle_polls": "TBD:gate_settle_polls",
            "deadline_guard_ns": "TBD:gate_deadline_guard_ns",
            "max_snapshot_age_ns": "TBD:gate_max_snapshot_age_ns"}


def rd(name, *terms):
    return {"name": name, "terms": [{"role": r, "slot": s, "sign": g, "agg": "sum"}
                                    for r, s, g in terms], "abs_max": 0}


def reader_gate(stub):
    """gRPC in flight (stub yielded - reader read, design 2.6), the stub's queue upstream of its
    yield, its open streams, and the reader's open streams and requests."""
    return gate_t([rd("grpc_in_flight", (stub, "backlog.yielded", 1),
                      ("reader", "backlog.read", -1)),
                   rd("%s_pending" % stub, (stub, "backlog.pending_msgs", 1)),
                   rd("%s_active" % stub, (stub, "backlog.active_streams", 1)),
                   rd("reader_active", ("reader", "backlog.active_streams", 1)),
                   rd("reader_inflight", ("reader", "backlog.inflight", 1))])


def stub_gate(stub, *extra):
    return gate_t(list(extra) + [rd("%s_pending" % stub, (stub, "backlog.pending_msgs", 1)),
                                 rd("%s_active" % stub, (stub, "backlog.active_streams", 1))])


M_AXIS = {"name": "M", "load_key": "nominal_msgs_per_s", "unit": "msg/s", "order": "asc"}
R_AXIS = {"name": "R", "load_key": "rate_rps", "unit": "req/s", "order": "asc"}
# axis shape -> (axis, common, warmup load keys, per-up-tier load keys(i))
#   mc: M axis, fixed beat, C doubles; md: M axis, fixed C, beat shrinks (values decreasing);
#   r: R axis, shortest output (max_tokens 1), rate doubles at a fixed beat.
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
    "mc": "M axis, fixed C path: k = 1, finite streams of n_tokens renewed at C / stream "
          "duration, beat fixed, C_0 < C_1 < ... (doubling); register once per output length "
          "(two values, design 2.1)",
    "md": "M axis, fixed beat path: k = 1, C fixed (c_fixed), beat_ns_0 > beat_ns_1 > ... "
          "(halving); every tier's duration, the drain after it included, must be a multiple of "
          "its beat; register once per "
          "output length",
    "r": "R axis: shortest output (max_tokens 1), rate_0 < rate_1 < ... (doubling); register "
         "once per beat period (at least two values, design 2.1 'R axis delta')",
}


def tpl(exp, name, shape, prompt_set, gate, inject=(), n=LADDER, note=""):
    axis, common, warm, up = SHAPES[shape]
    common = dict(common, tokens_per_msg=1, stream_interval=1, prompt_set=prompt_set)
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
            "launch_inject": list(inject), "common": common, "tiers": tiers,
            "note": "; ".join(x for x in (SHAPE_NOTES[shape], note, TEMPLATE_NOTE) if x)}


def inj(role_, site):
    return {"role": role_, "site": site, "mode": "busy", "delta_ns": "TBD:delta_ns"}


def arm(exp, name, scenario, template):
    w(SUB[exp], name + ".scenario.json", scenario)
    w(SUB[exp], name + ".tiers.template.json", template)


SHAPE_TAG = {1: "1c", K_CONN: "kc", 2 * K_CONN: "k2c"}
STUB_DESC = {"spb": "S-PB (capstubs.prebaked_server, direct mode)",
             "snl": "S-NL (capstubs.null_llm_server --mode full)"}
V_NOTE = {"vrdr": "V: busy delta at the reader's recv.msg (every message)",
          "vstub": "V: busy delta at the stub's send.msg (every message)",
          "vsrv": "V: busy delta at S-NL's send.msg, on the servicer's event loop (every "
                  "message)",
          "vload": "V: busy delta at capload's recv.chunk (every chunk)",
          "vf2": "V: busy delta at the frontend's F2 (GenerateStream._read, every gRPC message "
                 "read; capprobe CAPPROBE_INJECT); P1 overlay",
          "vf4": "V: busy delta at the frontend's F4 (per-request queue dequeue, every event "
                 "item; capprobe CAPPROBE_INJECT); P1 overlay"}


def direct_inputs(stub):
    ji = {"request_bytes": req_bytes("{param:model_name}")}
    if stub == "spb":
        ji = {"prebaked": PREBAKED, "request_bytes": ji["request_bytes"]}
    return ji


def direct_stub(stub, mclass):
    return spb(mclass) if stub == "spb" else snl(mclass)


def direct_arm(exp, name, stub_role, reader, shape, v, note, n=LADDER):
    """stub (direct mode) <- one gRPC reader role, plus the observers."""
    stub = stub_role["role"]
    roles = [stub_role, reader] + observers("reader")
    inject = {None: [], "vrdr": [inj("reader", "recv.msg")],
              "vstub": [inj(stub, "send.msg")], "vsrv": [inj(stub, "send.msg")]}[v]
    arm(exp, name,
        scen(exp, name, roles, ["reader", stub, "sidecar", "canary"],
             [req_chain(("reader", "summary.counts.ok"),
                        (stub, "summary.counts.streams_accepted")), msg_chain(stub)],
             STUB_DESC[stub], note, job_inputs=direct_inputs(stub)),
        tpl(exp, name, shape, "short", reader_gate(stub), inject, n,
            V_NOTE.get(v, "")))


# ================= E2a (card E2, part E2a; RUNBOOK-stage2 1.5) =================
def e2a(stub, procs, shape, v=None):
    name = "e2a-%s-%s-%s%s" % (stub, SHAPE_TAG[procs], shape, "-" + v if v else "")
    what = ("single-connection shape (the shape E3-3b / E4 readers see; the weak-reader "
            "reference)" if procs == 1 else "multi-connection shape, %d connections%s"
            % (procs, " (loader doubled: C12)" if procs == 2 * K_CONN else ""))
    direct_arm("E2a", name, direct_stub(stub, "under_test"), grpc_load(stub, procs), shape, v,
               "E2a %s: capgrpc.load -> one %s, %s" % (name, STUB_DESC[stub], what))


for _stub in ("spb", "snl"):
    for _procs in (1, K_CONN):
        for _shape in ("mc", "md", "r"):
            e2a(_stub, _procs, _shape)
    e2a(_stub, 2 * K_CONN, "mc")
    e2a(_stub, 1, "mc", "vrdr")
    e2a(_stub, 1, "mc", "vstub")
    e2a(_stub, K_CONN, "mc", "vstub")


# ================= E3 (card E3: 3a server side, 3b client side) =================
def e3a(procs, shape, v=None):
    name = "e3a-snl-%s-%s%s" % (SHAPE_TAG[procs], shape, "-" + v if v else "")
    direct_arm("E3", name, snl("under_test"), grpc_load("snl", procs), shape, v,
               "E3-3a %s: capgrpc.load, %d connections -> one S-NL full; under test: the real "
               "servicer + _format_result with the in-domain metronome on the same loop "
               "(compare e3a-mono-%s at the same values)%s"
               % (name, procs, shape, " (loader doubled: C12)" if procs == 2 * K_CONN else ""))


def e3a_mono(shape):
    name = "e3a-mono-%s" % shape
    roles = [snl("under_test", mode="metronome-only")] + observers("snl")
    arm("E3", name,
        scen("E3", name, roles, ["snl", "sidecar", "canary"],
             [req_chain(("snl", "summary.counts.streams_accepted"))],
             "S-NL (capstubs.null_llm_server --mode metronome-only)",
             "E3-3a %s: S-NL metronome-only, the same-process dry run (metronome, NullLlm and "
             "queues, no servicer, no gRPC traffic): register with the values of e3a-snl-kc-%s "
             "to give the 'with / without metronome' pair; no load role, so C3 and P-SENT are "
             "NOT_RUN (errata FT1-6)" % (name, shape)),
        tpl("E3", name, shape, "short", stub_gate("snl")))


def e3b(reader_kind, shape, v=None):
    name = "e3b-%s-%s%s" % (reader_kind, shape, "-" + v if v else "")
    tier = {"gs": "generate-stream", "ees": "engine-event-stream"}.get(reader_kind)
    reader = oe_sink("spb", tier) if tier else grpc_load("spb", 1)
    what = ("oe_sink --tier %s (under test)" % tier if tier else
            "the weak reader: capgrpc.load single-connection shape, the same reader argv "
            "without --tier / --connect (design 2.3 'weak-reader check')")
    direct_arm("E3", name, spb("stub_out_of_domain"), reader, shape, v,
               "E3-3b %s: one S-PB on one core -> %s" % (name, what))


for _shape in ("mc", "md", "r"):
    e3a(K_CONN, _shape)
for _shape in ("mc", "md", "r"):
    e3a_mono(_shape)
e3a(2 * K_CONN, "mc")
e3a(K_CONN, "mc", "vsrv")
e3a(K_CONN, "mc", "vrdr")
for _kind in ("gs", "ees", "weak"):
    for _shape in ("mc", "md", "r"):
        e3b(_kind, _shape)
e3b("ees", "mc", "vrdr")
e3b("ees", "mc", "vstub")


# ================= E4 (card E4: 4b main arm -> S-PB, 4b' -> S-NL) =================
E4_PROMPTS = "tid128"  # token-id prompts of length 128 (card E4 R axis: token-id, L = 128)
FE_STUB = {"spb": "S-PB (capstubs.prebaked_server, frontend mode)",
           "snl": "S-NL (capstubs.null_llm_server --mode full, product tokenizer)"}


def fe_stub(stub):
    return spb("stub_out_of_domain", frontend=True) if stub == "spb" else \
        snl("stub_out_of_domain", frontend=True)


def fe_inputs(stub, reader=False):
    ji = {"fingerprint": FINGERPRINT, "prebaked": PREBAKED} if stub == "spb" else {}
    if reader:
        ji["request_bytes"] = req_bytes("{param:tokenizer_dir}")
    return ji or None


def e4(stub, shape, tag="", v=None, workers=W_HTTP, n=LADDER, extra=""):
    """capload -> the real frontend (main tier) -> one stub."""
    name = "e4%s%s-%s%s" % ("b" if stub == "spb" else "bp", tag, shape, "-" + v if v else "")
    product_v = v in ("vf2", "vf4")
    roles = [fe_stub(stub), frontend(stub, "P1" if product_v else "P0"),
             capload(workers)] + observers("capload")
    inject = {None: [], "vstub": [inj(stub, "send.msg")],
              "vload": [inj("capload", "recv.chunk")],
              "vf2": [inj("frontend", "F2")], "vf4": [inj("frontend", "F4")]}[v]
    if product_v:
        extra += ("; product-side V: the frontend runs the probe patch (overlay P1, no probe "
                  "group) and capprobe busy-waits at %s; the job's overlay manifest must be P1, "
                  "so these arms never share a job with the P0 arms" % v[1:].upper())
    arm("E4", name,
        scen("E4", name, roles, ["capload", "frontend", stub, "sidecar", "canary"],
             [req_chain(("capload", "summary.counts.ok"),
                        (stub, "summary.counts.streams_accepted"))],
             FE_STUB[stub],
             "E4-%s %s: capload (%d workers) -> trtllm-serve new-arch frontend (1 frontend, "
             "postproc pool off; under test, %d cores) -> one %s on one core%s"
             % ("4b" if stub == "spb" else "4b'", name, workers, FE_CPUS, FE_STUB[stub], extra),
             job_inputs=fe_inputs(stub),
             tok="TBD-by-registration (configs/tokenizers/GLM-5.3-Flash)"),
        tpl("E4", name, shape, E4_PROMPTS,
            stub_gate(stub, rd("loadgen_inflight", ("capload", "backlog.inflight", 1))),
            inject, n, V_NOTE.get(v, "")))


def e4_weak(stub, shape):
    """The weak-reader check of card E4 / design 2.3: the same stub (same argv, frontend mode)
    read by capgrpc.load in the single-connection shape instead of the frontend."""
    name = "e4%s-weak-%s" % ("b" if stub == "spb" else "bp", shape)
    reader = grpc_load(stub, 1)
    roles = [fe_stub(stub), reader] + observers("reader")
    arm("E4", name,
        scen("E4", name, roles, ["reader", stub, "sidecar", "canary"],
             [req_chain(("reader", "summary.counts.ok"),
                        (stub, "summary.counts.streams_accepted")), msg_chain(stub)],
             FE_STUB[stub],
             "E4 weak-reader check %s: the 4b%s stub with the same argv, read by capgrpc.load "
             "(single connection) instead of the frontend; register with the 4b%s values of "
             "the frontend's boundary tier" % (name, "" if stub == "spb" else "'",
                                                "" if stub == "spb" else "'"),
             job_inputs=fe_inputs(stub, reader=True),
             tok="TBD-by-registration (configs/tokenizers/GLM-5.3-Flash)"),
        tpl("E4", name, shape, E4_PROMPTS, reader_gate(stub)))


for _stub in ("spb", "snl"):
    for _shape in ("mc", "md", "r"):
        e4(_stub, _shape)
        e4_weak(_stub, _shape)
for _shape in ("mc", "r"):
    e4("spb", _shape, tag="-nb", n=1,
       extra="; card E4 first step: the real frontend's near-boundary noise (one load tier, "
             "repeated launches at the nearest stable tier below the plateau)")
e4("spb", "mc", tag="-w2", workers=2 * W_HTTP, extra="; load source doubled")
e4("spb", "mc", v="vstub")
e4("spb", "mc", v="vload")
e4("spb", "mc", v="vf2")
e4("spb", "mc", v="vf4")


# ================= S-FE arms (cards E2 E2a S-FE, E3-3c, E4-4c; PLAN-R2R5 8) =================
SFE_NOTE = ("S-FE = tests/unittest/grpc/openengine_client/fake_openengine.py, zero changes, as "
            "configs/e0 e0a-frontend-sfe ({param:python} {param:sfe_script}; plain port file); "
            "one process on one core. Every stream sleeps --step-delay after each of its steps "
            "(no shared beat): register sfe_step_delay_s = the tier beat and "
            "sfe_tokens_per_step = tokens_per_msg. No capkit self-report (no backlog slots, no "
            "summary): its rate is read at the reader (design E2 'S-FE')")


def sfe(mclass, frontend=False):
    """S-FE, one process on one core. Direct (E2a, E3-3c): --model = the request bytes' model,
    no fingerprint (synthetic token ids; capgrpc.reqbytes sets ignore_eos). Frontend (E4-4c):
    --model = the tokenizer dir and the fingerprint job input, as S-PB frontend mode."""
    args = ["--profile", "trtllm-like", "--host", "127.0.0.1", "--port", "0",
            "--port-file", "{own:port_file}",
            "--model", "{param:tokenizer_dir}" if frontend else "{param:model_name}",
            "--max-context-length", "{param:sfe_max_context_length}",
            "--tokens-per-step", "{param:sfe_tokens_per_step}",
            "--step-delay", "{param:sfe_step_delay_s}"]
    if frontend:
        args.append({"splice_json": "{input:fingerprint}", "key": "sfe_args"})
    return role("sfe", "product", "server", mclass, True, "asyncio", True, "P0",
                [{"kind": "plain_port_file"}], [], "none", args,
                command=["{param:python}", "{param:sfe_script}"],
                port={"kind": "plain_port_file", "host": "127.0.0.1"},
                note=SFE_NOTE + ("; frontend mode: fingerprint args from the job input"
                                 if frontend else "; direct mode, no fingerprint"))


def sfe_reader(procs, mclass="load_source"):
    """capgrpc.load against S-FE: the reader argv of every direct arm with --target {addr:sfe}
    (S-FE writes a plain port file, not a capkit port file)."""
    r = grpc_load("sfe", procs, mclass)
    a = r["args"]
    i = a.index("--target-port-file")
    a[i:i + 2] = ["--target", "{addr:sfe}"]
    return r


def sfe_direct(exp, name, procs, shape, note):
    """capgrpc.load (procs connections) -> one S-FE (under test). Gate and chain read the reader
    only (S-FE reports nothing)."""
    roles = [sfe("under_test"), sfe_reader(procs)] + observers("reader")
    arm(exp, name,
        scen(exp, name, roles, ["reader", "sfe", "sidecar", "canary"],
             [req_chain(("reader", "summary.counts.ok"))],
             "S-FE (fake_openengine.py, zero changes, direct mode)", note,
             job_inputs={"request_bytes": req_bytes("{param:model_name}")}),
        tpl(exp, name, shape, "short",
            gate_t([rd("reader_active", ("reader", "backlog.active_streams", 1)),
                    rd("reader_inflight", ("reader", "backlog.inflight", 1))])))


# E2a S-FE: the single-connection shape (the shape E4-4c's frontend sees; the S-FE multi-connection
# row is e3c-sfe-kc-*, as the S-NL multi-connection rows are e3a-snl-kc-*). No md arm: S-FE's
# step delay is one value per launch, so the beat cannot change from tier to tier.
for _shape in ("mc", "r"):
    sfe_direct("E2a", "e2a-sfe-1c-%s" % _shape, 1, _shape,
               "E2a e2a-sfe-1c-%s: capgrpc.load single-connection shape -> one S-FE (design 2.3 "
               "'S-FE'; the E4-4c weak-reader reference)" % _shape)
# E3-3c (card E3 '3c'): 3a with S-FE in place of S-NL (S-FE's servicer against the real servicer:
# register with the values of e3a-snl-kc-<shape>), and 3b's oe_sink against S-NL in place of
# S-PB (the stub swap of design 2.3: register with the values of e3b-ees-<shape>).
for _shape in ("mc", "r"):
    sfe_direct("E3", "e3c-sfe-kc-%s" % _shape, K_CONN, _shape,
               "E3-3c e3c-sfe-kc-%s: capgrpc.load, %d connections -> one S-FE; pair of "
               "e3a-snl-kc-%s at the same values (S-FE's servicer against the real one)"
               % (_shape, K_CONN, _shape))
    direct_arm("E3", "e3c-ees-snl-%s" % _shape, snl("stub_out_of_domain"),
               oe_sink("snl", "engine-event-stream"), _shape, None,
               "E3-3c e3c-ees-snl-%s: one S-NL full on one core -> oe_sink --tier "
               "engine-event-stream (under test); pair of e3b-ees-%s at the same values (the "
               "stub swap S-PB -> S-NL)" % (_shape, _shape))


def e4c(shape):
    """E4-4c (card E4): capload -> the real frontend (main tier) -> one S-FE, every stream on
    its own cadence; register with the values of e4b-<shape> (4b: every step together)."""
    name = "e4c-%s" % shape
    roles = [sfe("stub_out_of_domain", frontend=True), frontend("sfe"),
             capload(W_HTTP)] + observers("capload")
    arm("E4", name,
        scen("E4", name, roles, ["capload", "frontend", "sfe", "sidecar", "canary"],
             [req_chain(("capload", "summary.counts.ok"))],
             "S-FE (fake_openengine.py, zero changes, frontend mode)",
             "E4-4c %s: capload (%d workers) -> trtllm-serve new-arch frontend (1 frontend, "
             "postproc pool off; under test, %d cores) -> one S-FE on one core (every stream on "
             "its own step delay; pair of e4b-%s at the same values)"
             % (name, W_HTTP, FE_CPUS, shape),
             job_inputs={"fingerprint": FINGERPRINT},
             tok="TBD-by-registration (configs/tokenizers/GLM-5.3-Flash)"),
        tpl("E4", name, shape, E4_PROMPTS,
            gate_t([rd("loadgen_inflight", ("capload", "backlog.inflight", 1))])))


e4c("mc")


# ---------------- values checklists (fill, then python -m caplaunch.tiergen) ----------------
for _sub in sorted(set(SUB.values())):
    _d = os.path.join(OUT, _sub)
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
            w(_sub, n.replace(".tiers.template.json", ".values.todo.json"),
              {"schema": "capkit.tier_values/1", "values": {k: None for k in sorted(names)}})

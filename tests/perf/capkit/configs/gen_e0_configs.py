# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generator of configs/e0/*.json (the JSON files are the deliverable; rerun into a NEW dir and diff).

Usage: python configs/gen_e0_configs.py <new output dir>
"""
import json, os, sys
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
REG = "02_e0/E0/register"   # planned registration dir under <W> (DATA-AND-ANALYSIS 2)

def w(name, doc):
    p = os.path.join(OUT, name)
    assert not os.path.exists(p), p
    with open(p, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=False); f.write("\n")

def role(name, launcher, kind, mclass, asyncio, loop, uses_grpc, probe_tier, readiness,
         backlog, snapshots, args, start_after=(), cpus_needed=1, workers=1,
         expected=1, env=None, note="", proc_kinds=(), **kw):
    r = {"role": name, "launcher": launcher, "kind": kind, "start_after": list(start_after),
         "pin_role": name, "cpus_needed": cpus_needed, "numa": None, "workers": workers,
         "expected_procs": expected, "measure_class": mclass, "asyncio": asyncio,
         "uses_grpc": uses_grpc, "loop": loop, "probe_tier": probe_tier,
         "readiness": readiness, "backlog_slots": list(backlog), "snapshots": snapshots,
         "args": list(args), "env": env or {}, "proc_kinds": list(proc_kinds)}
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
             start_after=after, cpus_needed=1,
             module="capenv.sidecar"),
        role("canary", "capkit", "aux", "observer", False, None, False, None, CAPKIT_READY, [],
             "none", ["--interval-ms", "{param:canary_interval_ms}"], cpus_needed=1,
             module="capenv.canary")]
OBS_PARAMS = ["sidecar_interval_ms", "sidecar_gpu_every_s", "sidecar_numa_every_s",
              "canary_interval_ms"]
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
DIMS = lambda stub, tok: {"tokenizer": tok, "stub": stub, "response_timing_switches": "defaults (not set)",
                          "interpretation": "E0 round 1: method self-check / pilot; not a capacity result",
                          "log_sink": "raw/logs on the node-local disk (orchestrator redirect)"}

def scen(name, roles, stop_order, params, reconcile, stub, tok=None, note="", job_inputs=None):
    d = {"schema": "capkit.scenario/1", "experiment": "E0", "name": name,
            "tier_table": "%s/%s.tiers.json" % (REG, name),
            "schedule": "%s/%s.schedule.capbin" % (REG, name),
            "prompts": "%s/%s.prompts.json" % (REG, name),
            "probe_tier": "P0" if any(r["launcher"] == "product" for r in roles) else None,
            "probe_groups": [], "stop_grace_s": 10.0,
            "stop_order": stop_order, "reconcile": reconcile, "roles": roles,
            "params_required": sorted(set(params)), "dims": DIMS(stub, tok), "note": note}
    if job_inputs:
        d["job_inputs"] = job_inputs
    return d

# ---------------- (a) capload -> nullsse ----------------
REQ_CHAIN = lambda a, b, bsrc: {"name": "requests", "kind": "requests", "gate": None,
    "include_usage": None, "hops": [{"role": a, "source": "summary.counts.ok", "scope": "tier"},
                                    {"role": b, "source": bsrc, "scope": "tier"}], "corrections": []}
recon_a1 = {"schema": "capkit.reconcile/1", "chains": [
    REQ_CHAIN("capload", "nullsse", "summary.counts.requests"),
    {"name": "sse", "kind": "chunks", "gate": "product", "include_usage": True,
     "hops": [{"role": "capload", "source": "summary.frames.frames", "scope": "tier"},
              {"role": "nullsse", "source": "summary.counts.frames_sent", "scope": "tier"}],
     "corrections": []}]}
a1 = [
    role("nullsse", "capkit", "server", "stub_out_of_domain", True, "asyncio", False, None,
         CAPKIT_READY, ["backlog.active_streams"], "all",
         ["--host", "127.0.0.1", "--port", "0", "--chunk-bytes", "{param:nullsse_chunk_bytes}",
          "--pacing", "beat", "--idle-close-s", "{param:nullsse_idle_close_s}",
          "--lag-period-us", "{param:lag_period_us}"], cpus_needed=2, workers=2, expected=3,
         module="capload.nullsse",
         note="expected_procs = parent + 2 SO_REUSEPORT workers (B1 forks workers as children); "
              "set to what capload.nullsse --workers 2 actually runs once B1 is final"),
    role("capload", "capkit", "client", "load_source", True, "asyncio", False, None,
         CAPKIT_READY, ["backlog.inflight"], "all",
         ["--target-port-file", "{port_file:nullsse}", "--model", "nullsse"] + CAPLOAD_ARGS,
         start_after=["nullsse"], cpus_needed=2, workers=2, expected=3, module="capload",
         note="expected_procs = parent + 2 workers (same caveat as nullsse)"),
] + observers(["capload"])
w("e0a-nullsse.scenario.json", scen("e0a-nullsse", a1,
  ["capload", "nullsse", "sidecar", "canary"],
  ["nullsse_chunk_bytes", "nullsse_idle_close_s"] + CAPLOAD_PARAMS + OBS_PARAMS, recon_a1,
  "nullsse (capkit null SSE sink)",
  note="E0 (a) second line: capload -> nullsse, sub-saturation tiers only"))

# ---------------- (a) capload -> real new-arch frontend -> S-FE ----------------
# The S-FE fingerprint arguments: a job input (capcommon.jobinputs) built in the job after the
# gate by python -m capstubs.fingerprint (SEAMS SM-6). Its tokenizer settings are fixed to what
# the frontend roles' argv implies: no --trust_remote_code (trtllm-serve's default false) and no
# --config setting tokenizer_mode (auto). Change both together: a frontend role that gains
# --trust_remote_code or a tokenizer_mode needs the same values here.
FP_INPUT = {"fingerprint": {"args": ["--tokenizer", "{param:tokenizer_dir}",
                                     "--trust-remote-code", "false",
                                     "--tokenizer-mode", "auto"]}}
recon_a2 = {"schema": "capkit.reconcile/1", "chains": [
    {"name": "requests", "kind": "requests", "gate": None, "include_usage": None,
     "hops": [{"role": "capload", "source": "summary.counts.ok", "scope": "tier"}],
     "corrections": []}]}
a2 = [
    role("sfe", "product", "server", "stub_out_of_domain", True, "asyncio", True, "P0",
         [{"kind": "plain_port_file"}], [], "none",
         ["--profile", "trtllm-like", "--host", "127.0.0.1", "--port", "0",
          "--port-file", "{own:port_file}", "--model", "{param:sfe_model}",
          "--max-context-length", "{param:sfe_max_context_length}",
          "--tokens-per-step", "{param:sfe_tokens_per_step}",
          "--step-delay", "{param:sfe_step_delay_s}",
          {"splice_json": "{input:fingerprint}", "key": "sfe_args"}],
         command=["{param:python}", "{param:sfe_script}"],
         port={"kind": "plain_port_file", "host": "127.0.0.1"},
         note="S-FE = tests/unittest/grpc/openengine_client/fake_openengine.py (CLI :1966-1984 "
              "at 2f22a98d6e; --port-file writes a plain port line, :2030-2034); fingerprint args "
              "from the job input fingerprint (python -m capstubs.fingerprint in the job after "
              "the gate), never hand-copied"),
    role("frontend", "product", "server", "under_test", True, "uvloop", True, "P0",
         [{"kind": "report_addr"},
          {"kind": "http_get", "path": "/health", "expect_status": 200, "timeout_s": 2.0}],
         [], "none",
         ["{param:tokenizer_dir}", "--host", "127.0.0.1", "--port", "0",
          "--report_addr", "{own:report_addr}", "--openengine-endpoint", "{addr:sfe}",
          "--num_serve_frontends", "1", "--num_postprocess_workers", "0"],
         command=["{param:trtllm_serve}"], start_after=["sfe"], cpus_needed=2,
         proc_kinds=["frontend", "frontend_openengine", "frontend_single"],
         port={"kind": "report_addr", "host": "127.0.0.1"},
         note="options at 2f22a98d6e: --openengine-endpoint serve.py:1343, --num_serve_frontends "
              ":1187, --num_postprocess_workers :1180, --report_addr :1442; uvloop per "
              "openengine_poc/launch.py:142. expected_procs=1 is unverified: E0(c) smoke lists the "
              "real child processes (P0 children cannot self-report, INTERFACES 12 U-5)"),
    role("capload", "capkit", "client", "load_source", True, "asyncio", False, None,
         CAPKIT_READY, ["backlog.inflight"], "all",
         ["--target", "{addr:frontend}", "--model", "{param:tokenizer_dir}"] + CAPLOAD_ARGS,
         start_after=["frontend"], cpus_needed=2, workers=2, expected=3, module="capload",
         note="--model = the frontend's served model name (default: the model dir it was given)"),
] + observers(["capload"])
w("e0a-frontend-sfe.scenario.json", scen("e0a-frontend-sfe", a2,
  ["capload", "frontend", "sfe", "sidecar", "canary"],
  ["python", "sfe_script", "sfe_model", "sfe_max_context_length", "sfe_tokens_per_step",
   "sfe_step_delay_s", "trtllm_serve", "tokenizer_dir"]
  + CAPLOAD_PARAMS + OBS_PARAMS, recon_a2,
  "S-FE (fake_openengine.py, zero changes)", tok="TBD-by-registration (configs/tokenizers/<name>)",
  job_inputs=FP_INPUT,
  note="E0 (a): capload -> trtllm-serve --openengine-endpoint -> S-FE; sub-saturation tiers "
       "only. Reconcile has only the capload hop at P0 (no probe counters); block-count "
       "corrections are U-6 (E0 smoke)."))

# ---------------- (b) beat source -> known-bottleneck relay(s) -> drain ----------------
def relay(name, upstream):
    return role(name, "capkit", "server", "under_test", True, "asyncio", True, None,
                CAPKIT_READY, ["backlog.queue_msgs", "backlog.active_streams"], "all",
                ["--host", "127.0.0.1", "--port", "0", "--upstream-port-file",
                 "{port_file:%s}" % upstream, "--lag-period-us", "{param:lag_period_us}"],
                start_after=[upstream], cpus_needed=1, module="capselfcheck.relay",
                note="the known answer; injected delta comes from the tier table (role %s)" % name)

def relay_scenario(name, n_relays, note):
    beat = role("beat", "capkit", "server", "stub_out_of_domain", True, "asyncio", True, None,
                CAPKIT_READY, ["backlog.pending_msgs", "backlog.active_streams",
                               "backlog.yielded"], "all",
                ["--host", "127.0.0.1", "--port", "0", "--msg-bytes", "{param:beat_msg_bytes}",
                 "--pacing", "beat", "--stamp", "--in-process", "--lag-period-us",
                 "{param:lag_period_us}"],
                cpus_needed=1, module="capgrpc.nullgrpc",
                note="--pacing beat requires --workers 1 (INTERFACES 11); --in-process keeps it "
                     "one process (expected_procs 1) instead of supervisor + 1 worker")
    relays = [relay("relay1", "beat")]
    if n_relays == 2:
        relays.append(relay("relay2", "relay1"))
    last = relays[-1]["role"]
    drain = role("drain", "capkit", "client", "stub_out_of_domain", True, "asyncio", True, None,
                 CAPKIT_READY, ["backlog.inflight", "backlog.active_streams", "backlog.read"],
                 "all",
                 ["--target-port-file", "{port_file:%s}" % last, "--method",
                  "{param:grpc_method}", "--channels-per-worker", "{param:channels_per_worker}",
                  "--request-format", "capkit", "--stamped", "--in-process", "--msg-sample-mod",
                  "{param:msg_sample_mod}", "--schedule", "{input:schedule}", "--prompts",
                  "{input:prompts}", "--lag-period-us", "{param:lag_period_us}",
                  "--max-records-per-worker", "{param:drain_max_records_per_worker}"],
                 start_after=[last], cpus_needed=1, module="capgrpc.load",
                 note="the drain end of design E0 (b); it also places the requests; "
                      "--in-process = one process (expected_procs 1)")
    roles = [beat] + relays + [drain] + observers(["drain"])
    # summary keys as written by capgrpc.load / capgrpc.nullgrpc / capselfcheck.relay; every
    # count is attributed to the request's planned tier, so every hop is comparable per tier.
    hops = [{"role": "beat", "source": "summary.counts.msgs_sent", "scope": "tier"}]
    for r in relays:
        hops += [{"role": r["role"], "source": "summary.counts.msgs_received", "scope": "tier"},
                 {"role": r["role"], "source": "summary.counts.msgs_forwarded", "scope": "tier"}]
    hops.append({"role": "drain", "source": "summary.msgs.msgs", "scope": "tier"})
    req_hops = [{"role": "drain", "source": "summary.counts.ok", "scope": "tier"}]
    req_hops += [{"role": r["role"], "source": "summary.counts.streams_accepted",
                  "scope": "tier"} for r in relays]
    req_hops.append({"role": "beat", "source": "summary.counts.streams_accepted",
                     "scope": "tier"})
    recon = {"schema": "capkit.reconcile/1", "chains": [
        {"name": "requests", "kind": "requests", "gate": None, "include_usage": None,
         "hops": req_hops, "corrections": []},
        {"name": "messages", "kind": "messages", "gate": "product", "include_usage": None,
         "hops": hops, "corrections": []}]}
    stop = ["drain"] + [r["role"] for r in reversed(relays)] + ["beat", "sidecar", "canary"]
    params = ["beat_msg_bytes", "lag_period_us", "grpc_method", "channels_per_worker",
              "msg_sample_mod", "drain_max_records_per_worker"] + OBS_PARAMS
    return scen(name, roles, stop, params, recon, "capgrpc.nullgrpc --pacing beat (beat source)",
                note=note)

w("e0b-s1-single.scenario.json", relay_scenario("e0b-s1-single", 1,
  "E0 (b) scenario 1: relay1 busy-waits delta on its event loop per message; tiers "
  "sub-saturation / near / over the plateau; >= 3 deltas = >= 3 tier tables"))
w("e0b-s2-two-stage.scenario.json", relay_scenario("e0b-s2-two-stage", 2,
  "E0 (b) scenario 2: relay1 -> relay2 with different deltas; the swap is a second tier "
  "table with the two specs' roles exchanged (same scenario file, files_by_arm)"))
w("e0b-s3-aux-thread.scenario.json", relay_scenario("e0b-s3-aux-thread", 1,
  "E0 (b) scenario 3: the busy-wait runs on an aux Python thread in the relay process "
  "(inject where=aux_thread)"))
w("e0b-s4-sleep.scenario.json", relay_scenario("e0b-s4-sleep", 1,
  "E0 (b) scenario 4: blocking without CPU (inject mode=sleep)"))

# ---------------- tier-table TEMPLATES (numbers = TBD-by-registration) ----------------
def gate_t(readings):
    return {"mode": "enforce", "rule": "abs_max", "readings": readings,
            "poll_ns": "TBD:gate_poll_ns", "settle_polls": "TBD:gate_settle_polls",
            "deadline_guard_ns": "TBD:gate_deadline_guard_ns",
            "max_snapshot_age_ns": "TBD:gate_max_snapshot_age_ns"}

def rd(name, *terms):
    return {"name": name, "terms": [{"role": r, "slot": s, "sign": g, "agg": "sum"}
                                    for r, s, g in terms], "abs_max": 0}

DRAIN = {"label": None, "kind": "drain", "window_ns": "TBD:drain_max_wait_ns"}

def load_tier(label, kind, value_key, baseline=False):
    t = {"label": label, "kind": kind, "ramp_ns": "TBD:ramp_ns", "window_ns": "TBD:window_ns",
         "tail_ns": "TBD:tail_ns", "baseline": baseline}
    t.update(value_key)
    return t

def with_drains(load_tiers, warmup):
    out = [warmup]
    for i, t in enumerate(load_tiers):
        out.append(dict(DRAIN, label="drain-%02d" % (i + 1)))
        out.append(t)
    out.append(dict(DRAIN, label="drain-%02d" % (len(load_tiers) + 1)))
    return out

def tpl(name, axis, common, tiers, gate, inject, note):
    return {"schema": "capkit.tiers_template/1", "experiment": "E0", "name": name,
            "arrival": {"dist": "fixed", "seed": "TBD:arrival_seed"}, "axis": axis,
            "request_timeout_ns": "TBD:request_timeout_ns", "default_gate": gate,
            "launch_inject": inject, "common": common, "tiers": tiers, "note": note}

R_AXIS = {"name": "R", "load_key": "rate_rps", "unit": "req/s", "order": "asc"}
M_AXIS = {"name": "M", "load_key": "nominal_msgs_per_s", "unit": "msg/s", "order": "asc"}
r = lambda k: {"rate_rps": "TBD:%s" % k}
a_tiers = with_drains([load_tier("up-sub-lo", "up", r("rate_sub_lo"), baseline=True),
                       load_tier("up-sub-hi", "up", r("rate_sub_hi")),
                       load_tier("down-sub-lo", "down", r("rate_sub_lo"))],
                      {"label": "warmup", "kind": "warmup", "ramp_ns": 0,
                       "window_ns": "TBD:warmup_ns", "tail_ns": 0, "rate_rps": "TBD:rate_warmup"})
common_a = {"max_tokens": "TBD:max_tokens", "tokens_per_msg": 1, "stream_interval": 1,
            "prompt_set": "short"}
w("e0a-nullsse.tiers.template.json", tpl(
    "e0a-nullsse", R_AXIS, dict(common_a, beat_ns="TBD:beat_ns"), a_tiers,
    gate_t([rd("loadgen_inflight", ("capload", "backlog.inflight", 1)),
            rd("sink_active", ("nullsse", "backlog.active_streams", 1))]), [],
    "two clearly sub-saturated rates (design E0 'load'); every tier duration must be a "
    "multiple of beat_ns"))
w("e0a-frontend-sfe.tiers.template.json", tpl(
    "e0a-frontend-sfe", R_AXIS, dict(common_a, beat_ns=0), a_tiers,
    gate_t([rd("loadgen_inflight", ("capload", "backlog.inflight", 1))]), [],
    "P0: the product processes publish no backlog slots, so the gate reads only capload"))

c = lambda k: {"streams_c": "TBD:%s" % k, "rate_rps": "from_streams_c"}
b_tiers = with_drains([load_tier("up-sub", "up", c("c_sub"), baseline=True),
                       load_tier("up-near", "up", c("c_near")),
                       load_tier("up-over", "up", c("c_over")),
                       load_tier("down-near", "down", c("c_near")),
                       load_tier("down-sub", "down", c("c_sub"))],
                      {"label": "warmup", "kind": "warmup", "ramp_ns": 0,
                       "window_ns": "TBD:warmup_ns", "tail_ns": 0,
                       "streams_c": "TBD:c_warmup", "rate_rps": "from_streams_c"})
common_b = {"max_tokens": "TBD:n_tokens", "tokens_per_msg": 1, "stream_interval": 1,
            "beat_ns": "TBD:beat_ns", "prompt_set": "short"}

def b_gate(relays):
    rs = [rd("grpc_in_flight", ("beat", "backlog.yielded", 1), ("drain", "backlog.read", -1)),
          rd("beat_pending", ("beat", "backlog.pending_msgs", 1)),
          rd("drain_active", ("drain", "backlog.active_streams", 1)),
          rd("drain_inflight", ("drain", "backlog.inflight", 1))]
    rs += [rd("%s_queue" % x, (x, "backlog.queue_msgs", 1)) for x in relays]
    return gate_t(rs)

def inj(role_, mode, delta, where=None):
    d = {"role": role_, "site": "relay.msg", "mode": mode, "delta_ns": "TBD:%s" % delta}
    if where:
        d["where"] = where
    return d

for name, relays, inject, note in (
        ("e0b-s1-single", ["relay1"], [inj("relay1", "busy", "delta_ns")],
         "one tier table per delta (>= 3 deltas)"),
        ("e0b-s2-two-stage", ["relay1", "relay2"],
         [inj("relay1", "busy", "delta_relay1"), inj("relay2", "busy", "delta_relay2")],
         "the swap = a second values file with delta_relay1 and delta_relay2 exchanged"),
        ("e0b-s3-aux-thread", ["relay1"], [inj("relay1", "busy", "delta_ns", "aux_thread")],
         "busy-wait on an aux Python thread of the relay process"),
        ("e0b-s4-sleep", ["relay1"], [inj("relay1", "sleep", "delta_ns")],
         "time.sleep(delta) on the relay event loop: blocks without CPU")):
    w("%s.tiers.template.json" % name,
      tpl(name, M_AXIS, common_b, b_tiers, b_gate(relays), inject,
          "sub-saturation / near / over the plateau (design E0 'load' (b)); " + note))

# ---------------- (c) frontend + S-FE + HTTP smoke client (no load tiers) ----------------
sfe_c, fe_c, cl_c = [dict(x) for x in a2[:3]]
probe = role("httpprobe", "capkit", "aux", "observer", False, None, False, None,
             [{"kind": "file_exists", "path": "{run_dir}/raw/httpprobe.json"}], [], "none",
             ["--addr", "{addr:frontend}", "--requests", "{param:httpprobe_requests_json}",
              "--timeout-s", "{param:httpprobe_timeout_s}",
              "--result", "{run_dir}/raw/httpprobe.json", "--hold"],
             start_after=["frontend"], module="caplaunch.httpprobe",
             note="sends each request once (completions + chat, streaming) and records per-frame "
                  "text deltas; caplaunch.smoke --tree reads raw/httpprobe.json")
recon_c = recon_a2  # the verdicts of e0c come from caplaunch.smoke --tree
w("e0c-frontend-http.scenario.json", scen("e0c-frontend-http", [sfe_c, fe_c, probe, cl_c],
  ["capload", "httpprobe", "frontend", "sfe"],
  ["python", "sfe_script", "sfe_model", "sfe_max_context_length", "sfe_tokens_per_step",
   "sfe_step_delay_s", "trtllm_serve", "tokenizer_dir",
   "httpprobe_requests_json", "httpprobe_timeout_s"] + CAPLOAD_PARAMS, recon_c,
  "S-FE (fake_openengine.py, zero changes)", tok="TBD-by-registration (configs/tokenizers/<name>)",
  job_inputs=FP_INPUT,
  note="E0 (c): real tokenizer + frontend + S-FE answer /v1/completions and "
       "/v1/chat/completions; empty-delta frames recorded (X2 premise); access-log lines "
       "counted by caplaunch.smoke --tree. httpprobe is ready (its result written) before the "
       "origin; capload then places one low warmup rate and a drain, because the tier-table "
       "schema needs a drain tier with a gate and only capload publishes a backlog slot here."))
w("e0c-frontend-http.tiers.template.json", tpl(
    "e0c-frontend-http", R_AXIS, dict(common_a, beat_ns=0),
    [{"label": "warmup", "kind": "warmup", "ramp_ns": 0, "window_ns": "TBD:hold_ns",
      "tail_ns": 0, "rate_rps": "TBD:rate_warmup"}, dict(DRAIN, label="drain-01")],
    gate_t([rd("loadgen_inflight", ("capload", "backlog.inflight", 1))]), [],
    "first light only: one low capload rate through the frontend, then a gated drain"))
w("e0c-httpprobe.requests.json", [
    {"name": "completions-stream", "endpoint": "/v1/completions",
     "body": {"model": "TBD:served-model-name", "prompt": "Hello", "max_tokens": 16,
              "stream": True, "stream_options": {"include_usage": True}}},
    {"name": "chat-stream", "endpoint": "/v1/chat/completions",
     "body": {"model": "TBD:served-model-name",
              "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16,
              "stream": True, "stream_options": {"include_usage": True}}}])

# ---------------- (c) smoke items config ----------------
cores = lambda *names: {n: {"param": "cores_%s" % n} for n in names}
w("e0c-smoke.json", {
    "schema": "capkit.smoke_config/1",
    "import_probe": {"code": "import tensorrt_llm, sys; print(tensorrt_llm.__version__)",
                     "timeout_s": 600},
    "interpreters": ["python", "python_capkit"],
    "loopback": {"duration_s": 10, "msg_bytes": 256},
    "httpprobe_out": "raw/httpprobe.json", "httpprobe_requests": 2,
    "topologies": [
        {"name": "E4-old-arch", "roles": cores("loadgen", "server", "postproc", "observers")},
        {"name": "E5-new-arch-frontend", "roles": cores("loadgen", "frontend", "sfe",
                                                         "observers")},
        {"name": "E6-E9-max", "roles": cores("loadgen", "frontend", "engine_side", "stubs",
                                              "observers")}],
    "note": "core counts per role come from the registered pin plan (--param cores_<role>=N); "
            "a topology with any missing count is NOT_RUN, never guessed"})

# ---------------- values checklists (fill, then python -m caplaunch.tiergen) ----------------
for n in sorted(os.listdir(OUT)):
    if n.endswith(".tiers.template.json"):
        with open(os.path.join(OUT, n)) as f:
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

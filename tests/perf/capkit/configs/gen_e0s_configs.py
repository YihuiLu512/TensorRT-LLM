# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generator of configs/e0s/*.json: the E0 supplementary smoke (RUNBOOK-stage2 1.4) items
(1) import probe, (2) S-PB + real frontend and (3) S-NL minimal + real OpenEngineServer + real
frontend. The JSON files are the deliverable; rerun into a NEW dir and diff.

Usage: python configs/gen_e0s_configs.py <new output dir>
"""
import json, os, sys
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
REG = "02_e0/E0s/register"   # planned registration dir under <W>
PB2 = "tensorrt_llm.grpc.openengine._generated.generation_pb2"

def w(name, doc):
    p = os.path.join(OUT, name)
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
                          "interpretation": "E0 supplementary smoke (RUNBOOK-stage2 1.4): feasibility, not a capacity result",
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

# Item (1) of every e0s launch: the import probe, as a caplaunch.smoke role that runs only the
# env items (env.import_probe with its compat-path control) and holds until stopped.
def importprobe():
    return role("importprobe", "capkit", "aux", "observer", False, None, False, None,
                [{"kind": "file_exists", "path": "{run_dir}/raw/smoke/smoke.json"}], [], "none",
                ["--config", "{param:smoke_config}", "--result-dir", "{raw}/smoke", "--items", "env",
                 "--param", "python={param:python}",
                 "--param", "cuda_compat_dir={param:cuda_compat_dir}", "--hold"],
                module="caplaunch.smoke",
                note="RUNBOOK-stage2 1.4 item (1): env.import_probe (tensorrt_llm, uvloop, "
                     "grpc_health; config e0s-smoke.json) with and without the compat path; "
                     "smoke.json under raw/smoke/ is read back by hand, not by the judge")
IMPORT_PARAMS = ["smoke_config", "python", "cuda_compat_dir"]

REQ_ONLY = lambda: {"schema": "capkit.reconcile/1", "chains": [
    {"name": "requests", "kind": "requests", "gate": None, "include_usage": None,
     "hops": [{"role": "capload", "source": "summary.counts.ok", "scope": "tier"}],
     "corrections": []}]}

# ---------------- (1) import probe (+ capload -> nullsse for the tier-table drain) ------------
ns = role("nullsse", "capkit", "server", "stub_out_of_domain", True, "asyncio", False, None,
          CAPKIT_READY, ["backlog.active_streams"], "all",
          ["--host", "127.0.0.1", "--port", "0", "--chunk-bytes", "{param:nullsse_chunk_bytes}",
           "--pacing", "beat", "--idle-close-s", "{param:nullsse_idle_close_s}",
           "--lag-period-us", "{param:lag_period_us}"], cpus_needed=1, workers=1, expected=2,
          module="capload.nullsse", note="as e0a-nullsse with one worker (parent + 1 worker)")
cl1 = role("capload", "capkit", "client", "load_source", True, "asyncio", False, None,
           CAPKIT_READY, ["backlog.inflight"], "all",
           ["--target-port-file", "{port_file:nullsse}", "--model", "nullsse"] + CAPLOAD_ARGS,
           start_after=["nullsse", "importprobe"], cpus_needed=1, workers=1, expected=2,
           module="capload", note="one low warmup rate and a gated drain: the tier-table schema "
                                  "needs a drain tier whose gate reads a backlog slot")
w("e0s-import-probe.scenario.json", scen(
  "e0s-import-probe", [importprobe(), ns, cl1], ["capload", "nullsse", "importprobe"],
  IMPORT_PARAMS + ["nullsse_chunk_bytes", "nullsse_idle_close_s"] + CAPLOAD_PARAMS, REQ_ONLY(),
  "nullsse (capkit null SSE sink)",
  note="E0 supplementary smoke item (1): import probe in the job's container; capload -> "
       "nullsse only places a warmup and a gated drain"))

# ---------------- (2) S-PB + real frontend + HTTP smoke client ----------------
spb = role("spb", "capkit", "server", "stub_out_of_domain", True, "asyncio", True, None,
           CAPKIT_READY, ["backlog.pending_msgs", "backlog.active_streams", "backlog.yielded"],
           "all",
           ["--host", "127.0.0.1", "--port", "0", "--prebaked", "{input:prebaked}",
            "--lag-period-us", "{param:lag_period_us}",
            "--max-stream-records", "{param:spb_max_stream_records}",
            "--sfe-module", "{param:sfe_module}", "--sfe-profile", "trtllm-like",
            "--model", "{param:sfe_model}",
            "--max-context-length", "{param:sfe_max_context_length}",
            {"splice_json": "{input:fingerprint}", "key": "sfe_args"}],
           start_after=["importprobe"], module="capstubs.prebaked_server",
           note="S-PB in frontend mode (D16): --sfe-module = the overlay manifest's stub module "
                "sfe in the snapshot ({SNAP}/<its src>); S-PB only reports the sha it read, the "
                "judge's C10 compares it (C10 is NOT_RUN here anyway: look for a 'role spb:' "
                "item among its reasons); "
                "fingerprint args from the job input (D15: as for S-FE); no --schedule, so "
                "streams are filed by acceptance time; --max-stream-records >= the schedule's "
                "rows + the httpprobe requests (they are not in the schedule)")
fe = role("frontend", "product", "server", "under_test", True, "uvloop", True, "P0",
          [{"kind": "report_addr"},
           {"kind": "http_get", "path": "/health", "expect_status": 200, "timeout_s": 2.0}],
          [], "none",
          ["{param:tokenizer_dir}", "--host", "127.0.0.1", "--port", "0",
           "--report_addr", "{own:report_addr}", "--openengine-endpoint", "{addr:spb}",
           "--num_serve_frontends", "1", "--num_postprocess_workers", "0"],
          command=["{param:trtllm_serve}"], start_after=["spb"], cpus_needed=2,
          proc_kinds=["frontend", "frontend_openengine", "frontend_single"],
          port={"kind": "report_addr", "host": "127.0.0.1"},
          note="as e0c-frontend-http, connected to S-PB instead of S-FE")
probe = role("httpprobe", "capkit", "aux", "observer", False, None, False, None,
             [{"kind": "file_exists", "path": "{run_dir}/raw/httpprobe.started.json"}], [],
             "none",
             ["--addr", "{addr:frontend}", "--requests", "{param:httpprobe_requests_json}",
              "--timeout-s", "{param:httpprobe_timeout_s}",
              "--started-file", "{run_dir}/raw/httpprobe.started.json",
              "--result", "{run_dir}/raw/httpprobe.json", "--hold"],
             start_after=["frontend"], module="caplaunch.httpprobe",
             note="one /v1/completions and one /v1/chat/completions request "
                  "(configs/e0/e0c-httpprobe.requests.json; their max_tokens must have a "
                  "prebaked stream, i.e. equal the tier table's max_tokens). Ready once the "
                  "requests are sent (--started-file): S-PB answers only from the origin on, "
                  "so --result is written after the origin; httpprobe_timeout_s must cover the "
                  "wait from the first request to the origin plus the first beat")
cl2 = role("capload", "capkit", "client", "load_source", True, "asyncio", False, None,
           CAPKIT_READY, ["backlog.inflight"], "all",
           ["--target", "{addr:frontend}", "--model", "{param:tokenizer_dir}"] + CAPLOAD_ARGS,
           start_after=["frontend", "httpprobe"], cpus_needed=1, workers=1, expected=2,
           module="capload",
           note="one low warmup rate and a gated drain through the frontend (as e0c)")
# fingerprint: the same fixed tokenizer settings as configs/gen_e0_configs.py FP_INPUT (what the
# frontend role's argv implies: trust_remote_code false, tokenizer_mode auto); change together.
spb_inputs = {
    "fingerprint": {"args": ["--tokenizer", "{param:tokenizer_dir}",
                             "--trust-remote-code", "false", "--tokenizer-mode", "auto"]},
    "prebaked": {"args": ["--pb2-module", PB2, "--tier-table", "{input:tier_table}",
                          "--token-ids", "{param:spb_token_ids}",
                          "--prompt-tokens", "{param:spb_prompt_tokens}"]}}
w("e0s-spb-frontend.scenario.json", scen(
  "e0s-spb-frontend", [importprobe(), spb, fe, probe, cl2],
  ["capload", "httpprobe", "frontend", "spb", "importprobe"],
  IMPORT_PARAMS + ["sfe_module", "sfe_model", "sfe_max_context_length", "tokenizer_dir",
                   "spb_token_ids", "spb_prompt_tokens", "spb_max_stream_records",
                   "trtllm_serve", "httpprobe_requests_json", "httpprobe_timeout_s"]
  + CAPLOAD_PARAMS, REQ_ONLY(), "S-PB (capstubs.prebaked_server, frontend mode)",
  tok="TBD-by-registration (configs/tokenizers/<name>)",
  job_inputs=spb_inputs,
  note="E0 supplementary smoke item (2): real tokenizer + frontend + S-PB answer "
       "/v1/completions and /v1/chat/completions (httpprobe: sent before the origin, answered "
       "from the origin on at S-PB's beat); capload places one low warmup rate and a gated "
       "drain. Fingerprint and prebaked bytes are job inputs built in the job after the gate "
       "(capjobs.launches). The origin lead must cover the serial start-up: importprobe's two "
       "cold imports, S-PB importing the S-FE module, the frontend start (configs/e0s/README.md)."))

# ---------------- (3) S-NL minimal + real OpenEngineServer + real frontend ----------------
# S-NL loads the tokenizer with the product's loader, with the same fixed settings as the
# fingerprint above (what the frontend role's argv implies); change together.
SNL_TOKENIZER = ["--tokenizer", "{param:tokenizer_dir}", "--trust-remote-code", "false",
                 "--tokenizer-mode", "auto"]
snl = role("snl", "capkit", "server", "stub_out_of_domain", True, "asyncio", True, None,
           CAPKIT_READY, ["backlog.pending_msgs", "backlog.active_streams", "backlog.yielded"],
           "all",
           ["--host", "127.0.0.1", "--port", "0", "--mode", "minimal",
            "--model", "{param:tokenizer_dir}", "--token-source", "fixed",
            "--token-ids", "{param:snl_token_ids}", "--lag-period-us", "{param:lag_period_us}",
            "--max-stream-records", "{param:snl_max_stream_records}"] + SNL_TOKENIZER,
           start_after=["importprobe"], module="capstubs.null_llm_server",
           note="S-NL minimal (errata FT1-1): the real OpenEngineServer and servicer on NullLlm, "
                "every step queued on arrival (no beat); the product server module is the "
                "default (tensorrt_llm.grpc.openengine.server); --model = the tokenizer dir "
                "(the frontend's served model name, as a real engine serves its model dir). "
                "No --schedule: the frontend's "
                "request ids are not schedule ids, so streams are filed by arrival time; "
                "--max-stream-records >= the schedule's rows + the httpprobe request")
fe3 = dict(fe, start_after=["snl"],
           args=[a if a != "{addr:spb}" else "{addr:snl}" for a in fe["args"]],
           note="as e0c-frontend-http, connected to S-NL minimal")
probe3 = role("httpprobe", "capkit", "aux", "observer", False, None, False, None,
              [{"kind": "file_exists", "path": "{run_dir}/raw/httpprobe.json"}], [], "none",
              ["--addr", "{addr:frontend}", "--requests", "{param:httpprobe_requests_json}",
               "--timeout-s", "{param:httpprobe_timeout_s}",
               "--result", "{run_dir}/raw/httpprobe.json", "--hold"],
              start_after=["frontend"], module="caplaunch.httpprobe",
              note="one /v1/completions request (e0s-snl-httpprobe.requests.json). S-NL minimal "
                   "answers on arrival (no beat), so the result is written before the origin "
                   "(as e0c with S-FE)")
w("e0s-snl-frontend.scenario.json", scen(
  "e0s-snl-frontend", [importprobe(), snl, fe3, probe3, dict(cl2)],
  ["capload", "httpprobe", "frontend", "snl", "importprobe"],
  IMPORT_PARAMS + ["snl_token_ids", "snl_max_stream_records", "tokenizer_dir",
                   "trtllm_serve", "httpprobe_requests_json", "httpprobe_timeout_s"]
  + CAPLOAD_PARAMS, REQ_ONLY(), "S-NL minimal (capstubs.null_llm_server --mode minimal)",
  tok="TBD-by-registration (configs/tokenizers/<name>)",
  note="E0 supplementary smoke item (3) (design 5.2 stage 0, D:203): real tokenizer + frontend "
       "+ the real OpenEngineServer on NullLlm answer one /v1/completions (httpprobe, before the "
       "origin); capload places one low warmup rate and a gated drain. If this item fails, "
       "E3-3a and the E8 ladder base use S-FE and E2a drops its S-NL arms (RUNBOOK-stage2 1.4). "
       "The origin lead must cover importprobe's two cold imports, S-NL importing the product "
       "and loading the tokenizer, and the frontend start (configs/e0s/README.md)."))

# ---------------- tier-table templates ----------------
def gate_t(readings):
    return {"mode": "enforce", "rule": "abs_max", "readings": readings,
            "poll_ns": "TBD:gate_poll_ns", "settle_polls": "TBD:gate_settle_polls",
            "deadline_guard_ns": "TBD:gate_deadline_guard_ns",
            "max_snapshot_age_ns": "TBD:gate_max_snapshot_age_ns"}

def rd(name, *terms):
    return {"name": name, "terms": [{"role": r, "slot": s, "sign": g, "agg": "sum"}
                                    for r, s, g in terms], "abs_max": 0}

def tpl(name, common, gate, note):
    return {"schema": "capkit.tiers_template/1", "experiment": "E0", "name": name,
            "arrival": {"dist": "fixed", "seed": "TBD:arrival_seed"},
            "axis": {"name": "R", "load_key": "rate_rps", "unit": "req/s", "order": "asc"},
            "request_timeout_ns": "TBD:request_timeout_ns", "default_gate": gate,
            "launch_inject": [], "common": common,
            "tiers": [{"label": "warmup", "kind": "warmup", "ramp_ns": 0,
                       "window_ns": "TBD:hold_ns", "tail_ns": 0, "rate_rps": "TBD:rate_warmup"},
                      {"label": "drain-01", "kind": "drain", "window_ns": "TBD:drain_max_wait_ns"}],
            "note": note}

w("e0s-import-probe.tiers.template.json", tpl(
    "e0s-import-probe", {"max_tokens": "TBD:max_tokens", "tokens_per_msg": 1,
                         "stream_interval": 1, "prompt_set": "short", "beat_ns": "TBD:beat_ns"},
    gate_t([rd("loadgen_inflight", ("capload", "backlog.inflight", 1)),
            rd("sink_active", ("nullsse", "backlog.active_streams", 1))]),
    "one low capload rate into nullsse, then a gated drain; the import probe runs before the "
    "origin"))
w("e0s-spb-frontend.tiers.template.json", tpl(
    "e0s-spb-frontend", {"max_tokens": 16, "tokens_per_msg": 1, "stream_interval": 1,
                         "prompt_set": "short", "beat_ns": "TBD:beat_ns"},
    gate_t([rd("loadgen_inflight", ("capload", "backlog.inflight", 1)),
            rd("spb_active", ("spb", "backlog.active_streams", 1)),
            rd("spb_pending", ("spb", "backlog.pending_msgs", 1))]),
    "max_tokens 16 = the httpprobe requests' max_tokens (S-PB rejects a max_tokens without a "
    "prebaked stream); beat_ns > 0 (S-PB has no asap pacing)"))
w("e0s-snl-frontend.tiers.template.json", tpl(
    "e0s-snl-frontend", {"max_tokens": "TBD:max_tokens", "tokens_per_msg": 1,
                         "stream_interval": 1, "prompt_set": "short", "beat_ns": 0},
    gate_t([rd("loadgen_inflight", ("capload", "backlog.inflight", 1)),
            rd("snl_active", ("snl", "backlog.active_streams", 1)),
            rd("snl_pending", ("snl", "backlog.pending_msgs", 1))]),
    "S-NL minimal has no beat (beat_ns 0) and serves any max_tokens"))
w("e0s-snl-httpprobe.requests.json", [
    {"name": "completions-stream", "endpoint": "/v1/completions",
     "body": {"model": "TBD:served-model-name", "prompt": "Hello", "max_tokens": 16,
              "stream": True, "stream_options": {"include_usage": True}}}])

# ---------------- import probe config (caplaunch.smoke, env items) ----------------
w("e0s-smoke.json", {
    "schema": "capkit.smoke_config/1",
    "import_probe": {"code": "import tensorrt_llm, uvloop, grpc_health.v1.health_pb2, sys; "
                             "print(tensorrt_llm.__version__)",
                     "timeout_s": 600},
    "note": "RUNBOOK-stage2 1.4 item (1): only the env items run (--items env)"})

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

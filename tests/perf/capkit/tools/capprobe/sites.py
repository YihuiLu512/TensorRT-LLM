# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Machine-readable site and slot tables of capprobe (INTERFACES.md sections 6.2, 6.5).

Pure data, no side effects: imported by capprobe itself (in product processes), by
capprobe.overlay (written into the overlay manifest) and by capprobe.patchcheck (which verifies
that every listed file:line in the probe tree really holds the guarded call for that site), and
loaded alone by path by caplaunch.scenario (PROC_KINDS validates each role's proc_kinds without
importing the capprobe package). Keep it free of imports and side effects.

`lines` lists, in ascending order, the line of the guarding `if` at every code location of the
site in the PATCHED probe tree (branch capprobe-J17871). A site has one slot and one probe
callable; when the event it counts is handled on more than one code path, each path carries the
same guarded call and the site lists one line per path.

`proc_kind` names the kind of process that passes the site (PROC_KINDS). A product role declares
in its scenario `proc_kinds` every kind its process tree contains; the judge (C8) expects a
window delta exactly at the sites whose kind is declared.
"""

# Every group named in the design. Only IMPLEMENTED_GROUPS may be switched on in this build;
# naming any other known group is an error at import (it would silently measure nothing).
ALL_GROUPS = ("count", "bytes", "lag", "queue", "ts", "pin", "stack")
IMPLEMENTED_GROUPS = ("count", "lag", "stack")
RESERVED_GROUPS = tuple(g for g in ALL_GROUPS if g not in IMPLEMENTED_GROUPS)

# Process kinds: each kind is the set of sites that every process of that kind passes while it
# serves the study's load (streaming /v1/completions). Roles declare kinds, never sites:
#   new-architecture single frontend:  frontend, frontend_openengine, frontend_single
#   new-architecture multi-frontend:   frontend, frontend_openengine, frontend_launcher,
#                                      frontend_attached (one role: launcher + attached frontends)
#   old-architecture trtllm-serve:     frontend
PROC_KINDS = {
    "frontend": "a process serving the OpenAI HTTP API through OpenAIServer: the new-architecture "
                "single frontend, the multi-frontend launcher and its attached frontends, and "
                "the old-architecture trtllm-serve frontend",
    "frontend_openengine": "a new-architecture frontend (single, launcher or attached) reaching "
                           "the engine through the OpenEngine client and facade",
    "frontend_single": "the new-architecture single frontend (--num_serve_frontends 1, "
                       "openengine_poc/launch.py)",
    "frontend_launcher": "the new-architecture multi-frontend launcher process",
    "frontend_attached": "a new-architecture multi-frontend attached frontend process",
}

# Count sites: one guarded call per pass on each code path; also the injection sites.
COUNT_SITES = {
    "H1": {"slot": "count.H1.arrivals", "per": "request"},
    "H3": {"slot": "count.H3.chunks", "per": "message"},
    "F2": {"slot": "count.F2.msgs", "per": "message"},
    "F4": {"slot": "count.F4.dequeued", "per": "message"},
}
# Event-loop lag install points (one per process kind); slot lag.<site> is a loghist/1 hist.
LAG_SITES = ("Lfe", "Lmfl", "Lmfa")
# Injection modes that make sense inside product code (drop/short belong to capkit tools).
PROBE_INJECT_MODES = ("busy", "sleep", "miscount")
INJECT_SITES = tuple(COUNT_SITES)

SITES = {
    "H1": {"group": "count", "proc_kind": "frontend",
           "file": "tensorrt_llm/serve/responses_utils.py",
           "lines": [3539], "slots": ["count.H1.arrivals"]},
    "H3": {"group": "count", "proc_kind": "frontend",
           "file": "tensorrt_llm/serve/openai_server.py",
           "lines": [3064], "slots": ["count.H3.chunks"]},
    "F2": {"group": "count", "proc_kind": "frontend_openengine",
           "file": "tensorrt_llm/grpc/openengine_client/stream.py",
           "lines": [267], "slots": ["count.F2.msgs"]},
    "F4": {"group": "count", "proc_kind": "frontend_openengine",
           "file": "tensorrt_llm/serve/openengine_poc/facade.py",
           "lines": [1424, 1630], "slots": ["count.F4.dequeued"]},
    "Lfe": {"group": "lag", "proc_kind": "frontend_single",
           "file": "tensorrt_llm/serve/openengine_poc/launch.py",
           "lines": [137], "slots": ["lag.Lfe"]},
    "Lmfl": {"group": "lag", "proc_kind": "frontend_launcher",
           "file": "tensorrt_llm/serve/openengine_poc/multi_frontend.py",
           "lines": [509], "slots": ["lag.Lmfl"]},
    "Lmfa": {"group": "lag", "proc_kind": "frontend_attached",
           "file": "tensorrt_llm/serve/openengine_poc/multi_frontend.py",
           "lines": [654], "slots": ["lag.Lmfa"]},
}

# Group `stack` (capprobe.stack): a stack sampler thread in every process where the group is on;
# it has no code location in the probe tree (no SITES entry) and only these slots. Categories of
# the main thread's samples and the run totals (meanings: capprobe.stack).
STACK_CATEGORIES = ("idle", "loop", "grpc", "pool_ipc", "tok", "assemble", "facade",
                    "openai_req", "openai_stream", "encode", "http_in", "http_out", "probe",
                    "trtllm_other", "other")
STACK_TOTALS = ("samples", "wall_ns", "main_cpu_ns", "proc_cpu_ns", "sampler_cpu_ns",
                "children_cpu_ns", "errors")

# What one pass means (cross-reference only, not part of the INTERFACES.md SITES contract).
# The lag sites share LAG_NOTE.
SITE_NOTES = {
    "H1": {"what": "ServerArrivalTimeMiddleware.__call__: every ASGI scope of type http whose "
                   "path starts with /v1/, whatever the route and status: generation requests, "
                   "but also GET /v1/models, other /v1/ routes and 404s under /v1/ (capkit's "
                   "load generators send only /v1/completions and /v1/chat/completions; /health "
                   "and /metrics are outside /v1/). Counted again per process a request passes: "
                   "an attached frontend forwards /v1/messages/batches routes to the launcher, "
                   "which counts them a second time, and the disagg router installs the same "
                   "middleware, so a request is counted at each probed hop"},
    "H3": {"what": "completion_generator: every pp_res before it is yielded "
                   "(/v1/completions streaming only; the chat stream generator is not "
                   "instrumented)"},
    "F2": {"what": "GenerateStream._read: every message read from the Generate call (not "
                   "EOF, not local cancellation; includes reads while draining after the "
                   "terminal event)"},
    "F4": {"what": "OpenEngineRequestOutput: every event item taken from the per-request queue "
                   "while the request is live (end markers and error items are not counted; "
                   "an item the engine cancelled is). Without a postprocessing pool, __anext__ "
                   "takes items only while the response handler waits for its next output and "
                   "stops once the request has finished. With a pool, _feed takes every item as "
                   "soon as it arrives and counts it while the postprocessing session is open, "
                   "so the backlog sits after F4 (in the pool and the result queue). F2 - F4 is "
                   "therefore the queue backlog only without a pool and only while every request "
                   "finishes normally: F2 also counts error, unknown and overflow events, and "
                   "events read after a request was aborted, failed or hit a stop string (with a "
                   "pool: after its session closed) are read by F2 but never taken by F4; filter "
                   "tiers by their failure counts before reading F2 - F4. The two paths also "
                   "differ at the end of a request: _feed counts items it takes between handing "
                   "the final event to the pool and the session closing, which __anext__ never "
                   "takes"},
    "Lfe": {"what": "single new-architecture frontend: run() on its uvloop, once "
                    "facade.start() has returned, before the HTTP server starts"},
    "Lmfl": {"what": "multi-frontend launcher: _run_launcher on its loop, once every attached "
                     "frontend is connected, before serving"},
    "Lmfa": {"what": "multi-frontend attached frontend: _run_attached on its loop, once the "
                     "launcher listens, before serving"},
}

# What every lag.<site> histogram contains beyond the serving period.
LAG_NOTE = ("The probe runs from installation until the process exits and is never stopped: "
            "the HTTP server's own start-up (uvicorn start, the application lifespan start-up) "
            "and the shutdown (draining, closing the facade, stopping attached frontends) are "
            "in the histogram, in every snapshot after them and in the exit snapshot. The "
            "judge reads only the change between the two snapshots of a tier window. On "
            "uvloop, call_at has millisecond resolution, so every sample may be up to about "
            "1 ms late on top of any real stall: lag below 1 ms is not resolved.")


def _build_slots():
    slots = {}
    for site, d in COUNT_SITES.items():
        slots[d["slot"]] = {"kind": "counter", "site": site, "group": "count", "per": d["per"]}
    for site in LAG_SITES:
        slots["lag.%s" % site] = {"kind": "hist", "site": site, "group": "lag", "per": "sample"}
        # How many times this install point was executed in the process (1 expected).
        slots["lag.%s.installs" % site] = {"kind": "counter", "site": site, "group": "lag",
                                           "per": "event"}
    for t in STACK_TOTALS:
        slots["stack.%s" % t] = {"kind": "counter", "site": None, "group": "stack",
                                 "per": "sample" if t == "samples" else "none"}
    slots["stack.children"] = {"kind": "gauge", "site": None, "group": "stack", "per": "none"}
    for c in STACK_CATEGORIES:
        slots["stack.main.n.%s" % c] = {"kind": "counter", "site": None, "group": "stack",
                                        "per": "sample"}
        for q in ("main.wall_ns", "main.cpu_ns", "thr.wall_ns"):
            slots["stack.%s.%s" % (q, c)] = {"kind": "counter", "site": None, "group": "stack",
                                             "per": "none"}
    for site in INJECT_SITES:
        for q in ("hits", "applied", "applied_ns"):
            slots["inject.%s.%s" % (site, q)] = {"kind": "counter", "site": site,
                                                 "group": "inject", "per": "event"}
    return slots


# {slot name: {kind, site, group, per}} (capcommon.snapshot.validate_slot_decls)
SLOTS = _build_slots()


def sites_for_groups(groups):
    """SITES entries whose group is in `groups` (the overlay manifest's `sites`); deep copies."""
    return {k: dict(v, lines=list(v["lines"]), slots=list(v["slots"]))
            for k, v in SITES.items() if v["group"] in groups}

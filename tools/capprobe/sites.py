# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Machine-readable site and slot tables of capprobe (INTERFACES.md sections 6.2, 6.5).

Pure data, no side effects: imported by capprobe itself (in product processes), by
capprobe.overlay (written into the overlay manifest) and by capprobe.patchcheck (which verifies
that every listed file:line in the probe tree really holds the guarded call for that site).

`lines` lists, in ascending order, the line of the guarding `if` at every code location of the
site in the PATCHED probe tree (branch capprobe-J17871). A site has one slot and one probe
callable; when the event it counts is handled on more than one code path, each path carries the
same guarded call and the site lists one line per path.
"""

# Every group named in the design. Only IMPLEMENTED_GROUPS may be switched on in this build;
# naming any other known group is an error at import (it would silently measure nothing).
ALL_GROUPS = ("count", "bytes", "lag", "queue", "ts", "pin")
IMPLEMENTED_GROUPS = ("count", "lag")
RESERVED_GROUPS = tuple(g for g in ALL_GROUPS if g not in IMPLEMENTED_GROUPS)

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
    "F2": {"group": "count", "proc_kind": "frontend",
           "file": "tensorrt_llm/grpc/openengine_client/stream.py",
           "lines": [267], "slots": ["count.F2.msgs"]},
    "F4": {"group": "count", "proc_kind": "frontend",
           "file": "tensorrt_llm/serve/openengine_poc/facade.py",
           "lines": [1424, 1630], "slots": ["count.F4.dequeued"]},
    "Lfe": {"group": "lag", "proc_kind": "frontend",
           "file": "tensorrt_llm/serve/openengine_poc/launch.py",
           "lines": [137], "slots": ["lag.Lfe"]},
    "Lmfl": {"group": "lag", "proc_kind": "frontend_launcher",
           "file": "tensorrt_llm/serve/openengine_poc/multi_frontend.py",
           "lines": [509], "slots": ["lag.Lmfl"]},
    "Lmfa": {"group": "lag", "proc_kind": "frontend_attached",
           "file": "tensorrt_llm/serve/openengine_poc/multi_frontend.py",
           "lines": [654], "slots": ["lag.Lmfa"]},
}

# What one pass means (cross-reference only, not part of the INTERFACES.md SITES contract).
SITE_NOTES = {
    "H1": {"what": "ServerArrivalTimeMiddleware.__call__: every ASGI scope of type http whose "
                   "path starts with /v1/ (health checks, /metrics and other routes are not "
                   "counted)"},
    "H3": {"what": "completion_generator: every pp_res before it is yielded "
                   "(/v1/completions streaming only; the chat stream generator is not "
                   "instrumented)"},
    "F2": {"what": "GenerateStream._read: every message read from the Generate call (not "
                   "EOF, not local cancellation; includes reads while draining after the "
                   "terminal event)"},
    "F4": {"what": "OpenEngineRequestOutput: every event item dequeued from the per-request "
                   "queue, on both paths that drain it: __anext__ without a postprocessing "
                   "pool and _feed with one (end markers and error items are not counted)"},
    "Lfe": {"what": "single new-architecture frontend: run() on its uvloop, once "
                    "facade.start() has returned, before the HTTP server starts"},
    "Lmfl": {"what": "multi-frontend launcher: _run_launcher on its loop, once every attached "
                     "frontend is connected, before serving"},
    "Lmfa": {"what": "multi-frontend attached frontend: _run_attached on its loop, once the "
                     "launcher listens, before serving"},
}


def _build_slots():
    slots = {}
    for site, d in COUNT_SITES.items():
        slots[d["slot"]] = {"kind": "counter", "site": site, "group": "count", "per": d["per"]}
    for site in LAG_SITES:
        slots["lag.%s" % site] = {"kind": "hist", "site": site, "group": "lag", "per": "sample"}
        # How many times this install point was executed in the process (1 expected).
        slots["lag.%s.installs" % site] = {"kind": "counter", "site": site, "group": "lag",
                                           "per": "event"}
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

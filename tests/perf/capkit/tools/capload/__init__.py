# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capload: open-loop multi-process HTTP SSE load generator (``python -m capload``) and the
multi-process SSE null sink (``python -m capload.nullsse``). INTERFACES.md sections 4.4, 4.5,
5.4, 11; design section 2.2.
"""


def _decl(kind, site, per):
    return {"kind": kind, "site": site, "per": per}


def _with_group(decls):
    return {name: dict(d, group=name.split(".", 1)[0]) for name, d in decls.items()}


# Injection sites and the modes each site supports (anything else -> exit 3 at start-up).
LOADGEN_SITES = {"send.req": ("busy", "sleep"),
                 "recv.chunk": ("busy", "sleep", "drop", "miscount")}
SINK_SITES = {"send.chunk": ("busy", "sleep", "drop", "miscount"),
              "short.req": ("short",)}


def _inject_decls(sites):
    out = {}
    for site in sites:
        for q in ("hits", "applied", "applied_ns"):
            out["inject.%s.%s" % (site, q)] = _decl("counter", site, "event")
    return out


# Machine-readable slot list of the load generator (INTERFACES.md 6.2). Registered eagerly at
# start-up in every capload process (parent included, so the drain gate finds the slot in every
# registered pid; the parent's values stay 0).
LOADGEN_SLOTS = _with_group(dict({
    "backlog.inflight": _decl("gauge", None, "none"),
    "backlog.unread_bytes": _decl("gauge", None, "none"),
    "self.dispatched": _decl("counter", "send.req", "request"),
    "self.written": _decl("counter", "send.req", "request"),
    "self.ok": _decl("counter", None, "request"),
    "self.failed": _decl("counter", None, "request"),
    "self.frames": _decl("counter", "recv.chunk", "message"),
    "self.bytes": _decl("counter", None, "event"),
    "self.conn_new": _decl("counter", None, "event"),
    "self.conn_reconnect": _decl("counter", None, "event"),
    "self.conn_peer_closed_idle": _decl("counter", None, "event"),
    "self.conn_fail": _decl("counter", None, "event"),
    "self.pool_size": _decl("gauge", None, "none"),
    "self.pool_idle": _decl("gauge", None, "none"),
    "self.send_lag": _decl("hist", "send.req", "request"),
    "lag.loop": _decl("hist", None, "sample"),
}, **_inject_decls(LOADGEN_SITES)))

SINK_SLOTS = _with_group(dict({
    "backlog.active_streams": _decl("gauge", None, "none"),
    "self.requests": _decl("counter", "short.req", "request"),
    "self.frames_sent": _decl("counter", "send.chunk", "message"),
    "self.bytes_sent": _decl("counter", None, "event"),
    "self.accepts": _decl("counter", None, "event"),
    "self.idle_closed": _decl("counter", None, "event"),
    "self.conns_open": _decl("gauge", None, "none"),
    "self.pending_msgs": _decl("gauge", None, "none"),
    "lag.loop": _decl("hist", None, "sample"),
}, **_inject_decls(SINK_SITES)))

# INTERFACES.md 6.2 asks a publishing package to export SLOTS; this package publishes two tools.
SLOTS = LOADGEN_SLOTS

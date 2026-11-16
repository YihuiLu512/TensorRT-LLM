# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capgrpc: multi-process grpc.aio open-loop load generator / stamped drain (B2).

Entry points: `python -m capgrpc.load` (the load generator / stamped drain, implemented in
capgrpc.loader), `python -m capgrpc.nullgrpc` (null sink / beat source) and
`python -m capgrpc.reqbytes` (request-bytes files). The shared runtime in `capgrpc.common` is also
used by `capselfcheck.relay`.
"""

TOOL = "capgrpc.load"

# Injection sites (INTERFACES.md 7.1). send.req: once per request just before the call is issued
# (busy/sleep); recv.msg: once per received message (busy/sleep); drop.msg: once per received
# message, mode=drop discards the message from every count (C7 failure sample).
SITES = ("send.req", "recv.msg", "drop.msg")


def inject_slot_decls(sites, per):
    out = {}
    for site in sites:
        for q in ("hits", "applied", "applied_ns"):
            out["inject.%s.%s" % (site, q)] = {"kind": "counter", "site": site, "group": "inject",
                                               "per": per[site]}
    return out


# Machine-readable slot list (INTERFACES.md 6.2); every process of this tool registers all of
# them at start-up (the supervisor process too, where they stay 0).
SLOTS = {
    "backlog.inflight": {"kind": "gauge", "site": None, "group": "backlog", "per": "request"},
    "backlog.active_streams": {"kind": "gauge", "site": None, "group": "backlog",
                               "per": "request"},
    "backlog.read": {"kind": "counter", "site": "recv.msg", "group": "backlog",
                     "per": "message"},
    "self.sent": {"kind": "counter", "site": "send.req", "group": "self", "per": "request"},
    "self.ok": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.failed": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.children_alive": {"kind": "gauge", "site": None, "group": "self", "per": "none"},
    "bytes.recv": {"kind": "counter", "site": "recv.msg", "group": "bytes", "per": "message"},
    "lag.send": {"kind": "hist", "site": "send.req", "group": "lag", "per": "request"},
    "lag.msg": {"kind": "hist", "site": "recv.msg", "group": "lag", "per": "message"},
    "lag.loop": {"kind": "hist", "site": None, "group": "lag", "per": "sample"},
}
SLOTS.update(inject_slot_decls(SITES, {"send.req": "request", "recv.msg": "message",
                                   "drop.msg": "message"}))

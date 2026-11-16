# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capgrpc.nullgrpc: grpc.aio null sink returning prebaked bytes, and absolute-beat source (B2).

Entry point: `python -m capgrpc.nullgrpc` (INTERFACES.md section 11).
"""

from capgrpc import inject_slot_decls

TOOL = "capgrpc.nullgrpc"

# send.msg: once per message just before it is handed to gRPC (busy/sleep);
# beat.step: once per beat step in the beat driver (busy/sleep).
SITES = ("send.msg", "beat.step")

SLOTS = {
    "backlog.active_streams": {"kind": "gauge", "site": None, "group": "backlog",
                               "per": "request"},
    "backlog.pending_msgs": {"kind": "gauge", "site": None, "group": "backlog",
                             "per": "message"},
    "backlog.yielded": {"kind": "counter", "site": "send.msg", "group": "backlog",
                        "per": "message"},
    "self.accepted": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.completed": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.cancelled": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.children_alive": {"kind": "gauge", "site": None, "group": "self", "per": "none"},
    "bytes.sent": {"kind": "counter", "site": "send.msg", "group": "bytes", "per": "message"},
    "count.beat_steps": {"kind": "counter", "site": "beat.step", "group": "count",
                         "per": "step"},
    "count.beat_late_steps": {"kind": "counter", "site": "beat.step", "group": "count",
                              "per": "step"},
    "count.beat_enqueued": {"kind": "counter", "site": "beat.step", "group": "count",
                            "per": "message"},
    "lag.beat_wake": {"kind": "hist", "site": "beat.step", "group": "lag", "per": "step"},
    "lag.loop": {"kind": "hist", "site": None, "group": "lag", "per": "sample"},
}
SLOTS.update(inject_slot_decls(SITES, {"send.msg": "message", "beat.step": "step"}))

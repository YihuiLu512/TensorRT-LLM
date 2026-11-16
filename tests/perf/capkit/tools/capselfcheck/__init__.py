# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capselfcheck: the known-bottleneck gRPC stream relay for E0 (b) (B2).

Entry point: `python -m capselfcheck.relay` (INTERFACES.md section 11).
"""

from capgrpc import inject_slot_decls

TOOL = "capselfcheck.relay"

# relay.msg: once per message received from upstream, on the event loop, before the message is
# queued for downstream. mode=busy (inline: spins on the loop; where=aux_thread: hands the spin to
# a plain Python thread of this process) or mode=sleep (time.sleep on the loop, no CPU).
SITES = ("relay.msg",)

SLOTS = {
    "backlog.queue_msgs": {"kind": "gauge", "site": None, "group": "backlog", "per": "message"},
    "backlog.active_streams": {"kind": "gauge", "site": None, "group": "backlog",
                               "per": "request"},
    "backlog.read": {"kind": "counter", "site": "relay.msg", "group": "backlog",
                     "per": "message"},
    "backlog.yielded": {"kind": "counter", "site": None, "group": "backlog", "per": "message"},
    "self.accepted": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.completed": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.failed": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.cancelled": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "bytes.forwarded": {"kind": "counter", "site": None, "group": "bytes", "per": "message"},
    "lag.relay_service": {"kind": "hist", "site": "relay.msg", "group": "lag",
                          "per": "message"},
    "lag.relay_hold": {"kind": "hist", "site": None, "group": "lag", "per": "message"},
    "lag.loop": {"kind": "hist", "site": None, "group": "lag", "per": "sample"},
}
SLOTS.update(inject_slot_decls(SITES, {"relay.msg": "message"}))

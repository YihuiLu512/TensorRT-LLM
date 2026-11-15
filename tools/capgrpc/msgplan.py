# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Planned time of each response message, derived from the schedule (readers of unstamped streams).

The OpenEngine stubs (S-PB, S-NL) send product messages without the step stamp of INTERFACES 7.4,
so a reader cannot read a message's planned time from the message. `--planned-msgs` (capgrpc.load
and capgrpc.oe_sink) derives it from what the reader already holds, the schedule and the tier
table:

    planned time of message j of a request planned at p in tier i
        = BeatSchedule.planned_rel_ns(step_at_or_after(p) + offsets_i[j])

with offsets_i = expect.msg_step_offsets(max_tokens_i, tokens_per_msg_i, stream_interval_i,
"product") and the last offset repeated once for the finished message (the stubs send it on the
step of the last token message). This is the open-loop plan as if the stub had accepted the
request at its planned time: the stub's own first step is step_at_or_after(accept time) (plus
--prefill-delay-us on S-NL), so the derived lag also contains the send lag, the request's transit
and up to one beat of step rounding. A message with no planned step (index past the expected
count, a tier without beat, or a step past the schedule end) is counted as unplanned and not put
into the lag.
"""

from capcommon import expect
from capcommon.exitcodes import ConfigError
from capcommon.schedule import BeatSchedule

GATE = "product"


class MsgPlan:
    """Per-tier step offsets of the product gate and the launch's beat schedule."""

    def __init__(self, tt):
        self.bs = BeatSchedule(tt)
        if self.bs.n_steps == 0:
            raise ConfigError("--planned-msgs needs a tier table with beat_ns > 0 tiers")
        self.offsets = [None] * len(tt)
        for t in tt.tiers:
            if not t.is_load:
                continue
            try:
                offs = expect.msg_step_offsets(t.max_tokens, t.tokens_per_msg, t.stream_interval,
                                               GATE)
            except expect.ExpectError as e:
                raise ConfigError("--planned-msgs: tier %d: %s" % (t.seq, e))
            self.offsets[t.seq] = tuple(offs) + (offs[-1],)  # + the finished message

    def stream(self, planned_rel_ns, seq):
        """The plan of one request, or None when its messages have no planned steps."""
        offs = self.offsets[seq]
        if offs is None:
            return None
        first = self.bs.step_at_or_after(planned_rel_ns)
        if first is None:
            return None
        return first, offs

    def planned(self, st, j):
        """(step, planned_rel_ns) of message j of a request planned by stream(), or None."""
        first, offs = st
        if j >= len(offs):
            return None
        step = first + offs[j]
        if step >= self.bs.n_steps:
            return None
        return step, self.bs.planned_rel_ns(step)

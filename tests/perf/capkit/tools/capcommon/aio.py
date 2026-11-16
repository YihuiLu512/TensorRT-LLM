# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""asyncio helpers: explicit loop selection, ns-precision scheduling, event-loop lag probe
(INTERFACES.md sections 2.2, 6.4). uvloop is optional and only imported when asked for.
"""

import asyncio
import time

from .exitcodes import ConfigError
from .versions import loop_impl_name

LOOP_CHOICES = ("asyncio", "uvloop")


def new_event_loop(loop_impl):
    """Create (not install) an event loop of the requested implementation."""
    if loop_impl == "asyncio":
        return asyncio.new_event_loop()
    if loop_impl == "uvloop":
        try:
            import uvloop  # noqa: PLC0415
        except ImportError as e:
            raise ConfigError("--loop uvloop requested but uvloop cannot be imported: %s" % e)
        return uvloop.new_event_loop()
    raise ConfigError("--loop must be one of %s, got %r" % (LOOP_CHOICES, loop_impl))


def run(main, loop_impl):
    """Run coroutine function main(loop) on a fresh loop of the requested implementation.

    Returns main's result. The loop's actual class is available inside main via
    capcommon.versions.loop_impl_name(loop).
    """
    loop = new_event_loop(loop_impl)
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(main(loop))
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def call_at_ns(loop, planned_mono_ns, callback, *args):
    """Schedule callback at a monotonic_ns instant (converted to the loop's clock now).

    Returns the TimerHandle. With uvloop the loop clock has millisecond resolution, so the
    callback may run up to ~1 ms late; callers must record actual times (lateness is data).
    """
    delay_s = (planned_mono_ns - time.monotonic_ns()) / 1e9
    return loop.call_at(loop.time() + delay_s, callback, *args)


async def sleep_until_ns(planned_mono_ns):
    delay = (planned_mono_ns - time.monotonic_ns()) / 1e9
    if delay > 0:
        await asyncio.sleep(delay)


class LoopLagProbe:
    """Periodic callback recording late = fire_time - planned_time (re-armed relative to the
    actual fire time; no catch-up bursts). Writes the loghist/1 histogram `lag.<name>` into a
    SlotRegistry and optionally per-sample rows into a RecordArray with fields
    (planned_rel_ns q, late_ns q, tier_seq h).
    """

    def __init__(self, loop, registry, name, period_ns, records=None, origin_ns=None,
                 tier_seq_at=None):
        if not isinstance(period_ns, int) or period_ns <= 0:
            raise ConfigError("lag probe period_ns must be an int > 0")
        if records is not None and (origin_ns is None or tier_seq_at is None):
            raise ConfigError("per-sample lag records need origin_ns and tier_seq_at")
        self.loop = loop
        self.period_ns = period_ns
        self.hist = registry.hist("lag.%s" % name)
        self.records = records
        self.origin_ns = origin_ns
        self.tier_seq_at = tier_seq_at
        self._planned = None
        self._handle = None
        self.running = False
        self.loop_impl = loop_impl_name(loop)

    def start(self):
        self.running = True
        self._arm(time.monotonic_ns())
        return self

    def _arm(self, now_ns):
        self._planned = now_ns + self.period_ns
        self._handle = call_at_ns(self.loop, self._planned, self._fire)

    def _fire(self):
        now = time.monotonic_ns()
        late = now - self._planned
        self.hist.add(late)
        if self.records is not None:
            rel = self._planned - self.origin_ns
            self.records.append(rel, late, self.tier_seq_at(rel))
        if self.running:
            self._arm(now)

    def stop(self):
        self.running = False
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

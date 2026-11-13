# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import signal
import time

import pytest

from capcommon import aio, lifecycle, versions
from capcommon.capbin import RecordArray
from capcommon.exitcodes import ConfigError
from capcommon.snapshot import SlotRegistry


def test_run_asyncio_and_uvloop_optional():
    async def main(loop):
        return versions.loop_impl_name(loop)

    name = aio.run(main, "asyncio")
    assert "asyncio" in name
    try:
        import uvloop  # noqa: F401
        have_uvloop = True
    except ImportError:
        have_uvloop = False
    if have_uvloop:
        assert "uvloop" in aio.run(main, "uvloop")
    else:
        with pytest.raises(ConfigError):
            aio.run(main, "uvloop")
    with pytest.raises(ConfigError):
        aio.run(main, "trio")


def test_call_at_ns_not_early():
    async def main(loop):
        fut = loop.create_future()
        planned = time.monotonic_ns() + 20_000_000
        aio.call_at_ns(loop, planned, lambda: fut.set_result(time.monotonic_ns()))
        fired = await fut
        return fired - planned

    late = aio.run(main, "asyncio")
    assert late >= -2_000_000  # loop clock rounding tolerance; never meaningfully early


def test_lag_probe_detects_blocking():
    """Positive control: a 50 ms blocking callback must show up as >= ~45 ms lag."""
    reg = SlotRegistry()
    recs = RecordArray([("planned_rel_ns", "q"), ("late_ns", "q"), ("tier_seq", "h")], 10_000)
    origin = time.monotonic_ns()

    async def main(loop):
        probe = aio.LoopLagProbe(loop, reg, "loop", 1_000_000, records=recs, origin_ns=origin,
                                 tier_seq_at=lambda rel: 0).start()
        await asyncio.sleep(0.05)
        loop.call_soon(lambda: time.sleep(0.05))
        await asyncio.sleep(0.1)
        probe.stop()
        return probe

    aio.run(main, "asyncio")
    assert reg.get("lag.loop.n") > 10
    assert reg.get("lag.loop.max") >= 45_000_000
    assert len(recs) == reg.get("lag.loop.n") and max(recs.column("late_ns")) >= 45_000_000
    with pytest.raises(ConfigError):
        aio.LoopLagProbe(None, reg, "x", 0)


def test_port_ready_stop_files(tmp_path, ident):
    pf = str(tmp_path / "port.json")
    lifecycle.write_port_file(pf, ident, "127.0.0.1", 12345)
    host, port, _doc = lifecycle.wait_port_file(pf, 1.0)
    assert (host, port) == ("127.0.0.1", 12345)
    with pytest.raises(ConfigError):
        lifecycle.write_port_file(pf, ident, "127.0.0.1", 0)
    with pytest.raises(lifecycle.WaitTimeout):
        lifecycle.wait_port_file(str(tmp_path / "never.json"), 0.1)
    rf = str(tmp_path / "ready.json")
    lifecycle.write_ready_file(rf, ident, versions.version_record(),
                               children=[{"pid": 1, "worker": 0, "role": "capload"}])
    _i, doc = lifecycle.wait_ready_file(rf, 1.0)
    assert doc["children"][0]["pid"] == 1 and doc["affinity"]
    with pytest.raises(ConfigError):
        lifecycle.write_ready_file(rf, ident, {}, children=[{"pid": 1}])
    assert lifecycle.read_stop_file(str(tmp_path / "none.json")) is None


def test_stop_controller_signal():
    sc = lifecycle.StopController(install=True)
    seen = []
    sc.on_stop(seen.append)
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        assert sc.wait(2.0) and sc.requested and not sc.hard
        os.kill(os.getpid(), signal.SIGTERM)
        deadline = time.monotonic() + 2
        while not sc.hard and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sc.hard and seen == [False, True]
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.default_int_handler)


def test_pin_self_readback():
    cur = sorted(os.sched_getaffinity(0))
    try:
        assert lifecycle.pin_self(cur[:1]) == cur[:1]
    finally:
        os.sched_setaffinity(0, set(cur))


def test_version_record_fields():
    rec = versions.version_record(extra={"tool": "t"})
    for k in ("python", "grpc", "protobuf_impl", "loop_impl", "gc_enabled", "switchinterval_s",
              "capkit_code_sha", "probe_groups", "env"):
        assert k in rec
    assert rec["tool"] == "t" and rec["loop_impl"] is None
    asy = {"asyncio": True, "uses_grpc": False, "probe_tier": None, "loop": "asyncio"}
    assert "loop_impl" in versions.missing_fields(rec, versions.required_fields(asy))
    # a non-asyncio, non-grpc process is not required to report a loop or grpc (no tautology)
    plain = {"asyncio": False, "uses_grpc": False, "probe_tier": None, "loop": None}
    assert "loop_impl" not in versions.required_fields(plain)
    assert "grpc" not in versions.required_fields(plain)
    with pytest.raises(ValueError):
        versions.version_record(extra={"pid": 1})

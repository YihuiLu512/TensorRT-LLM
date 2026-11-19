# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The limits on every capkit gRPC server's queue of calls the application has not picked up yet.

gRPC core ends a call in that queue when it has waited longer than
grpc.server_max_unrequested_time_in_server (grpcio default 30 s) or when the queue is longer than
grpc.server.max_pending_requests (soft) / _hard_limit. A server whose event loop falls behind
under overload then cancels streams that the load generator would have counted against its
registered timeout. The local reproductions block the server's event loop while clients keep
opening streams.
"""

import asyncio
import json
import os
import sys
import time
import types

import pytest

from capgrpc.common import (GRPC_INT_MAX, PENDING_HARD_OPTION, PENDING_OPTION,
                            UNREQUESTED_OPTION, AnyMethodHandler, bind_server, make_server,
                            resolve_max_unrequested)
from capcommon.exitcodes import ConfigError

from conftest import table_doc

METHOD = "/capkit.Test/Stream"
S = 1_000_000_000
# only the first pass through the site sleeps (every_n larger than any count here)
FIRST_ONLY = "every_n=100000,phase=0"


def _tt(timeout_ns, end_ns):
    return types.SimpleNamespace(request_timeout_ns=timeout_ns, end_ns=end_ns)


# ---- resolve_max_unrequested -------------------------------------------------------------------

def test_default_is_the_longer_of_timeout_and_table_rounded_up():
    got = resolve_max_unrequested(None, _tt(60 * S, 900 * S + 1))
    assert got[UNREQUESTED_OPTION] == 901 and got["source"].startswith("default")
    assert resolve_max_unrequested(None, _tt(60 * S + 1, 10 * S))[UNREQUESTED_OPTION] == 61
    assert resolve_max_unrequested(None, _tt(60 * S, 60 * S))[UNREQUESTED_OPTION] == 60


def test_default_without_a_registered_timeout_is_the_table_length():
    assert resolve_max_unrequested(None, _tt(None, 35 * S))[UNREQUESTED_OPTION] == 35
    # a table shorter than a second still gives a valid (>= 1 s) value
    assert resolve_max_unrequested(None, _tt(None, 1))[UNREQUESTED_OPTION] == 1


def test_explicit_value_at_or_above_the_timeout_is_used():
    got = resolve_max_unrequested(60, _tt(60 * S, 900 * S))
    assert got == {UNREQUESTED_OPTION: 60, PENDING_OPTION: GRPC_INT_MAX,
                   PENDING_HARD_OPTION: GRPC_INT_MAX, "source": "--max-unrequested-s"}
    assert resolve_max_unrequested(1, _tt(None, 900 * S))[UNREQUESTED_OPTION] == 1
    assert resolve_max_unrequested(GRPC_INT_MAX, _tt(None, S))[UNREQUESTED_OPTION] == GRPC_INT_MAX


@pytest.mark.parametrize("value,timeout_ns", [
    (59, 60 * S),            # shorter than the registered timeout
    (60, 60 * S + 1),        # shorter by 1 ns
    (0, None),               # 0 cancels every queued call in gRPC core
    (-1, None),
    (GRPC_INT_MAX + 1, None),
])
def test_explicit_value_refused(value, timeout_ns):
    with pytest.raises(ConfigError):
        resolve_max_unrequested(value, _tt(timeout_ns, 900 * S))


def test_make_server_has_no_default_for_the_option():
    with pytest.raises(TypeError):
        make_server(reuseport=False)  # pylint: disable=no-value-for-parameter


# ---- the tools refuse a bad value at start (exit 3, before binding) ----------------------------

@pytest.mark.parametrize("module,extra", [
    ("capgrpc.nullgrpc", ["--workers", "1", "--in-process", "--msg-bytes", "8",
                          "--pacing", "asap"]),
    ("capselfcheck.relay", ["--upstream", "127.0.0.1:1"]),
])
@pytest.mark.parametrize("value", ["0", "9"])
def test_tools_refuse_a_value_below_the_timeout(launch, module, extra, value):
    launch.write_table(table_doc([("warmup", 500, 4.0, {}), ("drain", 500, 0.0, {})],
                                 timeout_ns=10 * S))
    launch.set_origin(4.0)
    launch.server("srv", module, extra + ["--max-unrequested-s", value])
    assert launch.wait_exit("srv", 30) == 3, launch.stderr("srv")
    assert "max_unrequested" in launch.stderr("srv") or "max-unrequested" in launch.stderr("srv")


# ---- local reproduction ------------------------------------------------------------------------

def _load(launch, target_role, cpus):
    launch.start("load", "capgrpc.load",
                 launch.common("load", cpus=cpus)
                 + ["--target-port-file", launch.tree.port_file(target_role), "--method", METHOD,
                    "--workers", "1", "--in-process", "--channels-per-worker", "1",
                    "--schedule", launch.sched_path, "--request-format", "capkit",
                    "--prompts", launch.prompts_path, "--msg-sample-mod", "1",
                    "--max-records-per-worker", "100"])


def _failures(launch, role):
    out = {}
    for docs in launch.summaries(role).values():
        for d in docs:
            for f in d["failures"]:
                out[f["reason"]] = out.get(f["reason"], 0) + f["count"]
    return out


def _counts(launch, role, key):
    return sum(d["counts"][key] for docs in launch.summaries(role).values() for d in docs)


def _recorded_option(launch, role):
    """The time limit written to the role's ready file and to every tier summary (one value);
    every record also carries the queue length limits make_server sets."""
    with open(launch.tree.ready_file(role)) as f:
        ready = json.load(f)
    recs = [w["info"]["grpc_server_options"] for w in ready["info"]["workers"]]
    recs += [d["grpc_server_options"] for docs in launch.summaries(role).values() for d in docs]
    assert {(r[PENDING_OPTION], r[PENDING_HARD_OPTION]) for r in recs} == {(GRPC_INT_MAX,) * 2}
    vals = {r[UNREQUESTED_OPTION] for r in recs}
    assert len(vals) == 1, vals
    return vals.pop()


def _sink_stall(launch, value_s):
    """nullgrpc whose loop sleeps 3 s on its first message while 3 more streams arrive
    (fixed arrivals at 0, 0.25, 0.5, 0.75 s; no registered timeout)."""
    launch.write_table(table_doc([("warmup", 1000, 4.0, {"n": 1}), ("drain", 5000, 0.0, {})],
                                 timeout_ns=None))
    launch.set_origin(4.0)
    launch.server("sink", "capgrpc.nullgrpc",
                  ["--workers", "1", "--in-process", "--msg-bytes", "8", "--pacing", "asap",
                   "--inject", "site=send.msg,mode=sleep,delta_ns=%d,%s" % (3 * S, FIRST_ONLY),
                   "--max-unrequested-s", str(value_s)], cpus="0")
    launch.wait_file(launch.tree.port_file("sink"), 15, "sink")
    _load(launch, "sink", "1")
    assert launch.wait_exit("load", 60) == 0, launch.stderr("load")
    assert launch.stop("sink") == 0, launch.stderr("sink")
    assert _recorded_option(launch, "sink") == value_s
    return _failures(launch, "load")


def test_server_cancel_entry_records_zero_msgs_and_longest_duration(launch):
    """A server-side CANCELLED (the unrequested-time limit here) is grpc_CANCELLED; its failure
    entry counts the failures that got no message and the longest send-to-end time, the per-
    request evidence capjudge.admission reads (here every one waited >= the 1 s limit)."""
    _sink_stall(launch, 1)
    ents = [f for docs in launch.summaries("load").values() for d in docs
            for f in d["failures"]]
    canc = [f for f in ents if f["reason"] == "grpc_CANCELLED"]
    assert canc and all(f["zero_msgs"] == f["count"] for f in canc), ents
    assert max(f["max_send_to_end_ns"] for f in canc) >= 900_000_000, canc
    assert not [f for f in ents if f["reason"] == "client_cancelled"], ents


def test_short_value_cancels_streams_queued_behind_a_stalled_loop(launch):
    """The option reaches gRPC core: with 1 s, at least one stream queued behind the 3 s stall
    is cancelled, and every stream is either cancelled or completed."""
    fails = _sink_stall(launch, 1)
    assert fails.get("grpc_CANCELLED", 0) >= 1, fails
    assert _counts(launch, "load", "ok") + fails["grpc_CANCELLED"] == 4
    assert _counts(launch, "sink", "streams_accepted") == _counts(launch, "load", "ok")


def test_long_value_keeps_streams_queued_behind_a_stalled_loop(launch):
    fails = _sink_stall(launch, 10)
    assert fails == {}, fails
    assert _counts(launch, "load", "ok") == _counts(launch, "sink", "streams_accepted") == 4


def test_relay_default_outlasts_the_grpc_30s_default(launch):
    """SEAMS SM-18 reproduction: the relay's loop is blocked 32 s while 3 more streams arrive;
    with gRPC's own 30 s limit, streams queued longer than 30 s are cancelled (grpc_CANCELLED at
    the load generator, missing from the relay's streams_accepted). The relay's default (no
    --max-unrequested-s) is the longer of the registered timeout (45 s) and the table length, so
    all of them complete. ~45 s."""
    launch.write_table(table_doc([("warmup", 1000, 4.0, {"n": 1}), ("drain", 36000, 0.0, {})],
                                 timeout_ns=45 * S))
    launch.set_origin(4.0)
    launch.server("source", "capgrpc.nullgrpc",
                  ["--workers", "1", "--in-process", "--msg-bytes", "8", "--pacing", "asap"],
                  cpus="0")
    launch.wait_file(launch.tree.port_file("source"), 15, "source")
    launch.server("relay1", "capselfcheck.relay",
                  ["--upstream-port-file", launch.tree.port_file("source"), "--inject",
                   "site=relay.msg,mode=sleep,delta_ns=%d,%s" % (32 * S, FIRST_ONLY)], cpus="1")
    launch.wait_file(launch.tree.port_file("relay1"), 15, "relay1")
    _load(launch, "relay1", "2")
    assert launch.wait_exit("load", 90) == 0, launch.stderr("load")
    assert launch.stop("relay1") == 0, launch.stderr("relay1")
    assert launch.stop("source") == 0, launch.stderr("source")
    assert _recorded_option(launch, "relay1") == 45
    fails = _failures(launch, "load")
    assert fails == {}, fails
    assert _counts(launch, "load", "ok") == 4
    assert _counts(launch, "relay1", "streams_accepted") == 4
    assert _counts(launch, "relay1", "streams_completed") == 4


# ---- queue length limit ------------------------------------------------------------------------

PENDING_CLIENT = r"""
import asyncio, collections, json, os, sys, time
import grpc

port, n, stalled, go = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]


def wait_file(path, limit_s):
    end = time.monotonic() + limit_s
    while not os.path.exists(path):
        if time.monotonic() > end:
            raise SystemExit("no %s within %s s" % (path, limit_s))
        time.sleep(0.02)


async def main():
    ch = grpc.aio.insecure_channel("127.0.0.1:%d" % port,
                                   options=[("grpc.use_local_subchannel_pool", 1)])
    call = ch.unary_stream("/capkit.Test/Pending", request_serializer=None,
                           response_deserializer=None)

    async def one(body):
        try:
            async for _ in call(body, timeout=90):
                pass
            return "OK"
        except grpc.aio.AioRpcError as e:
            return e.code().name

    await asyncio.wait_for(ch.channel_ready(), 30)
    first = asyncio.ensure_future(one(b"stall"))
    await asyncio.get_running_loop().run_in_executor(None, wait_file, stalled, 30)
    rest = [asyncio.ensure_future(one(b"go")) for _ in range(n)]
    await asyncio.sleep(1.5)  # time for the n calls to reach the blocked server's queue
    open(go, "w").close()
    got = await asyncio.gather(*rest)
    print(json.dumps({"first": await first, "rest": dict(collections.Counter(got))}))
    await ch.close()


asyncio.run(main())
"""


def test_queue_length_limit_drops_no_call_behind_a_stalled_loop(tmp_path):
    """make_server's queue length limits: 2500 calls queue while the server's loop is blocked
    (without these limits gRPC core cancels some of them); all complete. The client is another
    process because the block stops this process's event loop."""
    n = 2500
    stalled, go = str(tmp_path / "stalled"), str(tmp_path / "go")

    def factory(_method):
        async def behavior(request, _context):
            if request == b"stall":
                open(stalled, "w").close()
                end = time.monotonic() + 60
                while not os.path.exists(go) and time.monotonic() < end:
                    time.sleep(0.02)  # blocks the event loop on purpose
            yield b"x"

        return behavior

    async def main():
        server = make_server(reuseport=False, max_unrequested_s=60)
        server.add_generic_rpc_handlers([AnyMethodHandler(factory)])
        port = bind_server(server, "127.0.0.1", 0)
        await server.start()
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-B", "-c", PENDING_CLIENT, str(port), str(n), stalled, go,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                out, err = await asyncio.wait_for(proc.communicate(), 150)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise
        finally:
            await server.stop(0)
        return proc.returncode, out.decode(), err.decode()

    rc, out, err = asyncio.run(main())
    assert rc == 0, err
    assert os.path.exists(go), "the client never released the stall"
    assert json.loads(out) == {"first": "OK", "rest": {"OK": n}}, out
